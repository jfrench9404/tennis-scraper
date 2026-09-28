"""Break/changeover PROPOSALS from synthetic rows: merged runs, partial long runs, output safety."""
import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

from tennis_vision import longrun
from tennis_vision.segments import absent_runs, load_run, main as segments_main, propose


def main(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        segments_main(argv)

FPS = 10.0  # Small synthetic fps keeps rows few; thresholds are in seconds.


def tracks(near=True, far=True, ball=False, predicted_near=False):
    out = []
    if near:
        out.append({"label": "player", "identity_id": "near", "predicted": False})
    if predicted_near:
        out.append({"label": "player", "identity_id": "near", "predicted": True})
    if far:
        out.append({"label": "player", "identity_id": "far", "predicted": False})
    if ball:
        out.append({"label": "ball", "predicted": False, "source": "gridtracknet"})
    return out


def write_run(folder, start, frame_tracks, fps=FPS):
    folder.mkdir(parents=True)
    (folder / "source-offset.json").write_text(json.dumps({"source_start_frame": start}))
    (folder / "summary.json").write_text(json.dumps({"fps": fps, "frames": len(frame_tracks)}))
    with (folder / "events.jsonl").open("w") as stream:
        for i, t in enumerate(frame_tracks):
            stream.write(json.dumps({"frame": i, "time_s": i / fps, "tracks": t, "events": []}) + "\n")


def write_partial_longrun(folder, start, frame_tracks, chunk_frames, complete_chunks, fps=FPS):
    """Same on-disk layout longrun.run_chunks writes, with only some chunks complete."""
    folder.mkdir(parents=True)
    manifest = {"version": longrun.LONGRUN_VERSION,
                "input": {"path": "synthetic.mp4", "sha256": "0", "fps": fps, "frames": start + len(frame_tracks),
                          "width": 1280, "height": 720},
                "selection": {"source_start_frame": start, "source_end_frame_exclusive": start + len(frame_tracks),
                              "frames": len(frame_tracks)},
                "settings": {"chunk_frames": chunk_frames}, "models": {}, "code": {},
                "chunks": longrun.plan_chunks(start, start + len(frame_tracks), chunk_frames)}
    manifest["fingerprint"] = longrun.fingerprint(manifest)
    longrun.save_json(folder / "longrun-manifest.json", manifest)
    for chunk in manifest["chunks"]:
        cdir = longrun.chunk_dir(folder, chunk)
        cdir.mkdir(parents=True)
        lines = [json.dumps({"frame": f, "time_s": f / fps, "source_frame": start + f,
                             "tracks": frame_tracks[f], "events": []}) + "\n"
                 for f in range(chunk["first_frame"], chunk["first_frame"] + chunk["frames"])]
        if chunk["index"] >= complete_chunks:
            # The chunk being processed: only a .partial file and no chunk.json.
            (cdir / "events.jsonl.partial").write_text("".join(lines[:3]))
            continue
        (cdir / "events.jsonl").write_text("".join(lines))
        with (cdir / "state.pkl").open("wb") as stream:
            pickle.dump({"frames": chunk["first_frame"] + chunk["frames"]}, stream)
        longrun.save_json(cdir / "chunk.json", {
            "status": "complete", "index": chunk["index"], "fingerprint": manifest["fingerprint"],
            "first_frame": chunk["first_frame"], "frames": chunk["frames"],
            "source_first_frame": chunk["source_first_frame"],
            "events_sha256": longrun.file_sha256(cdir / "events.jsonl"),
            "state_sha256": longrun.file_sha256(cdir / "state.pkl")})
    return manifest


def changeover_rows():
    """60 s: play 0-20 s, near side empty and no ball 20-45 s (one 0.5 s false near blip), play 45-60 s."""
    rows = []
    for f in range(600):
        s = f / FPS
        if 20 <= s < 45:
            rows.append(tracks(near=(300 <= f < 305), far=True, ball=False))
        else:
            rows.append(tracks(ball=(f % 3 == 0)))
    return rows


class AbsentRunsTest(unittest.TestCase):
    def test_bridges_short_blips_but_not_unknown(self):
        values = [True] + [False] * 5 + [True] * 2 + [False] * 5 + [None] + [False] * 3
        runs = absent_runs(values, min_frames=3, bridge_frames=3)
        self.assertEqual([(r["start"], r["end"]) for r in runs], [(1, 12), (14, 16)])
        self.assertTrue(runs[0]["open_end"])      # Stops at unknown, may be longer.
        self.assertFalse(runs[0]["open_start"])
        self.assertTrue(runs[1]["open_start"] and runs[1]["open_end"])

    def test_long_observation_splits_and_short_runs_drop(self):
        values = [False] * 5 + [True] * 3 + [False] * 5 + [True] + [False] * 2
        runs = absent_runs(values, min_frames=4, bridge_frames=3)  # 3-frame spell is not < 3.
        self.assertEqual([(r["start"], r["end"]) for r in runs], [(0, 4), (8, 12)])  # (14, 15) too short to isolate

    def test_frequent_short_observations_are_not_stray(self):
        values = [i % 3 == 0 for i in range(60)]  # Seen one frame in three: present, not empty.
        self.assertEqual(absent_runs(values, min_frames=4, bridge_frames=10), [])


class ProposeTest(unittest.TestCase):
    def test_changeover_proposal_with_evidence_and_source_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            write_run(run, 1000, changeover_rows())
            report = propose(load_run(run))
        self.assertEqual(report["source"], "merged")
        self.assertFalse(report["partial"])
        self.assertEqual(report["source_frames"], [1000, 1599])
        near = report["signals"]["near_side_empty"]
        self.assertEqual(len(near), 1)  # The 0.5 s blip is bridged (default bridge 1 s).
        self.assertEqual(near[0]["source_frames"], [1200, 1449])
        self.assertEqual(near[0]["source_seconds"], [120.0, 145.0])
        self.assertEqual(near[0]["near_observed_frames"], 5)
        self.assertEqual(report["signals"]["far_side_empty"], [])
        ball = report["signals"]["no_ball_observed"]
        self.assertEqual(len(ball), 1)
        self.assertLessEqual(ball[0]["source_frames"][0], 1200)
        self.assertGreaterEqual(ball[0]["source_frames"][1], 1449)
        [proposal] = report["proposals"]
        self.assertEqual(proposal["label"], "possible changeover or break")
        self.assertEqual(proposal["status"], "proposal_unreviewed")
        self.assertEqual(proposal["reasons"], ["near_side_empty", "no_ball_observed"])
        self.assertEqual(proposal["evidence"]["far_observed_frames"], proposal["evidence"]["frames"])
        self.assertEqual(proposal["evidence"]["ball_observed_frames"], 0)
        self.assertFalse(proposal["open_start"] or proposal["open_end"])
        self.assertEqual(report["thresholds"], {"min_side_empty_seconds": 20.0, "min_no_ball_seconds": 15.0,
                                                "bridge_seconds": 1.0})
        second_125 = next(b for b in report["per_second"] if b["second"] == 125)
        self.assertEqual((second_125["frames"], second_125["near"], second_125["far"], second_125["ball"]),
                         (10, 0, 10, 0))

    def test_short_absences_and_predicted_tracks(self):
        rows = [tracks(near=False, predicted_near=True, ball=True) for _ in range(150)]  # 15 s predicted only
        rows += [tracks(ball=True) for _ in range(100)]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            write_run(run, 0, rows)
            loaded = load_run(run)
            self.assertEqual(propose(loaded)["proposals"], [])  # 15 s < 20 s default.
            report = propose(loaded, min_side_empty_seconds=10)
        [proposal] = report["proposals"]
        self.assertEqual(proposal["source_frames"], [0, 149])  # Predicted near tracks do not count.
        self.assertTrue(proposal["open_start"])
        self.assertFalse(proposal["open_end"])

    def test_no_game_or_point_labels_in_output(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            write_run(run, 0, changeover_rows())
            report = propose(load_run(run))

        def keys(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    yield k
                    yield from keys(v)
            elif isinstance(value, list):
                for v in value:
                    yield from keys(v)
        for key in keys(report):
            for word in ("game", "point", "score", "ace", "fault", "serve", "winner"):
                self.assertNotIn(word, key.lower().split("_"))
        self.assertEqual({p["label"] for p in report["proposals"]}, {"possible changeover or break"})
        self.assertEqual({p["status"] for p in report["proposals"]}, {"proposal_unreviewed"})

    def test_bad_threshold_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            write_run(run, 0, changeover_rows())
            with self.assertRaises(ValueError):
                propose(load_run(run), bridge_seconds=float("nan"))


class PartialLongRunTest(unittest.TestCase):
    def test_reads_only_complete_leading_chunks(self):
        rows = changeover_rows()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "claude-game-partial"
            write_partial_longrun(run, 50, rows, chunk_frames=100, complete_chunks=4)  # 400/600 frames
            loaded = load_run(run)
            self.assertEqual(loaded["source"], "chunks")
            self.assertEqual(len(loaded["rows"]), 400)
            report = propose(loaded)
        self.assertTrue(report["partial"])
        self.assertEqual(report["chunks"], {"complete": 4, "planned": 6})
        self.assertEqual(report["source_frames"], [50, 449])
        [proposal] = report["proposals"]  # Near empty 20-40 s so far (still 20 s), reaching the data edge.
        self.assertEqual(proposal["source_frames"], [249, 449])  # Ball also unseen from 0.1 s earlier.
        self.assertTrue(proposal["open_end"])
        self.assertEqual(report["selection_frames"], 600)

    def test_tampered_chunk_stops_reading(self):
        rows = changeover_rows()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            manifest = write_partial_longrun(run, 0, rows, chunk_frames=100, complete_chunks=6)
            with (longrun.chunk_dir(run, manifest["chunks"][2]) / "events.jsonl").open("a") as stream:
                stream.write("\n")
            loaded = load_run(run)
        self.assertEqual(len(loaded["rows"]), 200)  # Same leading-complete rule as longrun resume.

    def test_no_complete_chunks_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            write_partial_longrun(run, 0, changeover_rows(), chunk_frames=100, complete_chunks=0)
            with self.assertRaisesRegex(SystemExit, "0/6 chunks complete"):
                main(["--run", str(run), "--output", str(Path(directory) / "out.json")])
            self.assertFalse((Path(directory) / "out.json").exists())

    def test_merged_events_preferred_over_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            write_partial_longrun(run, 0, changeover_rows(), chunk_frames=100, complete_chunks=2)
            with (run / "events.jsonl").open("w") as stream:
                stream.write(json.dumps({"frame": 0, "time_s": 0, "source_frame": 0, "tracks": [], "events": []}) + "\n")
            self.assertEqual(load_run(run)["source"], "merged")
            self.assertEqual(len(load_run(run, "chunks")["rows"]), 200)


class CliTest(unittest.TestCase):
    def test_writes_json_and_html_outside_run_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            write_run(run, 0, changeover_rows())
            before = sorted(p.name for p in run.iterdir())
            with self.assertRaises(SystemExit):
                main(["--run", str(run), "--output", str(run / "segments.json")])
            out = root / "out" / "segments.json"
            main(["--run", str(run), "--output", str(out), "--min-no-ball-seconds", "30"])
            self.assertEqual(sorted(p.name for p in run.iterdir()), before)  # Nothing applied to the run.
            report = json.loads(out.read_text())
            self.assertEqual(report["thresholds"]["min_no_ball_seconds"], 30.0)
            self.assertEqual(report["proposals"][0]["reasons"], ["near_side_empty"])
            html = (root / "out" / "segments.html").read_text()
            self.assertNotIn("__SEGMENTS_DATA__", html)
            embedded = html.split('type="application/json">', 1)[1].split("</script>", 1)[0]
            self.assertEqual(json.loads(embedded), report)

    def test_html_escapes_markup_in_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "<b>run"
            write_run(run, 0, changeover_rows())
            out = root / "out.json"
            main(["--run", str(run), "--output", str(out)])
            html = (root / "out.html").read_text()
            self.assertNotIn("<b>run", html)
            self.assertIn("\\u003cb>run", html)


if __name__ == "__main__":
    unittest.main()
