"""Run status: progress, rate, ETA, stalls and gaps from synthetic run folders.

Synthetic runs use longrun's own chunk plan and folder naming, and chunk.json in
the exact shape longrun writes (with and without the newer timing fields).
"""
import contextlib
import hashlib
import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from tennis_vision import longrun, run_status

PROJECT = Path(__file__).resolve().parent.parent
TZ = timezone(timedelta(hours=1))
T0 = datetime(2026, 9, 26, 0, 0, 0, tzinfo=TZ)


def stamp(path, dt):
    os.utime(path, (dt.timestamp(), dt.timestamp()))


def make_run(root, frames=100, chunk_frames=10, fps=10.0, start=0, name="game"):
    output = root / name
    output.mkdir(parents=True)
    manifest = {"version": 1, "input": {"path": "C:\\video.mp4", "sha256": "0" * 64, "fps": fps, "frames": 5000,
                                        "width": 1280, "height": 720},
                "selection": {"source_start_frame": start, "source_end_frame_exclusive": start + frames,
                              "frames": frames},
                "settings": {}, "models": {}, "code": {}, "device": "directml",
                "chunks": longrun.plan_chunks(start, start + frames, chunk_frames)}
    manifest["fingerprint"] = "f" * 64
    (output / "longrun-manifest.json").write_text(json.dumps(manifest))
    stamp(output / "longrun-manifest.json", T0)
    return output, manifest


def complete_chunk(output, manifest, index, finished, seconds, extra=None):
    chunk = manifest["chunks"][index]
    folder = longrun.chunk_dir(output, chunk)
    folder.mkdir(parents=True)
    (folder / "events.jsonl").write_text("".join(json.dumps({"frame": chunk["first_frame"] + i}) + "\n"
                                                 for i in range(chunk["frames"])))
    (folder / "state.pkl").write_bytes(b"state")
    record = {"status": "complete", "index": index, "fingerprint": manifest["fingerprint"],
              "first_frame": chunk["first_frame"], "frames": chunk["frames"],
              "source_first_frame": chunk["source_first_frame"],
              "source_last_frame": chunk["source_first_frame"] + chunk["frames"] - 1, "last_frame_sha256": "a" * 64,
              "events_sha256": hashlib.sha256((folder / "events.jsonl").read_bytes()).hexdigest(),
              "state_sha256": hashlib.sha256(b"state").hexdigest(), "seconds": seconds}
    record.update(extra or {})
    (folder / "chunk.json").write_text(json.dumps(record))
    stamp(folder / "chunk.json", finished)
    return folder


def partial_chunk(output, manifest, index, rows, modified):
    folder = longrun.chunk_dir(output, manifest["chunks"][index])
    folder.mkdir(parents=True)
    (folder / "events.jsonl.partial").write_text("".join("{}\n" for _ in range(rows)))
    stamp(folder / "events.jsonl.partial", modified)


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


def old_format_partial_run(root):
    """Six 10-frame chunks at 5 s/frame, then chunk 7 half written, then silence."""
    output, manifest = make_run(root)
    finished = T0
    for index in range(6):
        finished += timedelta(seconds=50)
        complete_chunk(output, manifest, index, finished, 50.0)
    partial_chunk(output, manifest, 6, 4, finished + timedelta(seconds=20))
    return output, manifest, finished + timedelta(seconds=20)


class ChunkLayoutTest(unittest.TestCase):
    def test_chunk_dir_matches_longrun(self):
        manifest = {"chunks": longrun.plan_chunks(4740, 5654, 300)}
        for chunk in manifest["chunks"]:
            self.assertEqual(run_status.chunk_dir(Path("r"), chunk), longrun.chunk_dir(Path("r"), chunk))


class PartialRunTest(unittest.TestCase):
    def test_old_format_partial_run_is_flagged_stalled(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _, last = old_format_partial_run(Path(directory))
            result = run_status.status(output, logs=[], now=last + timedelta(hours=3))
            self.assertEqual(result["state"], "stalled")
            self.assertTrue(result["stalled"])
            self.assertEqual((result["frames_done"], result["frames_total"]), (60, 100))
            self.assertEqual((result["chunks_complete"], result["chunks_total"]), (6, 10))
            self.assertEqual(result["current_chunk"], {"index": 6, "frames_written": 4, "frames": 10})
            self.assertEqual(result["source_covered_s"], [0.0, 6.0])
            self.assertEqual(result["last_progress"], last)
            self.assertIn("partial rows modified time", result["last_progress_basis"])
            self.assertAlmostEqual(result["rate_fps"], 0.2)
            self.assertAlmostEqual(result["eta_s"], 40 / 0.2)
            text = run_status.render(result)
            for expected in ("STALLED: no progress for 3 h 00 min", "60 / 100", "6 / 10 complete",
                             "5.00 s/frame", "once resumed", "chunk 7 has 4/10 frames written"):
                self.assertIn(expected, text)

    def test_recent_progress_is_running_and_threshold_is_configurable(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _, last = old_format_partial_run(Path(directory))
            now = last + timedelta(minutes=5)
            self.assertEqual(run_status.status(output, logs=[], now=now)["state"], "running")
            self.assertEqual(run_status.status(output, stall_minutes=4, logs=[], now=now)["state"], "stalled")
            self.assertEqual(run_status.status(output, stall_minutes=30, logs=[], now=now + timedelta(minutes=20))
                             ["state"], "running")
            self.assertIn("about", run_status.render(run_status.status(output, logs=[], now=now)))

    def test_idle_time_between_old_chunks_is_a_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            output, manifest = make_run(Path(directory))
            complete_chunk(output, manifest, 0, T0 + timedelta(seconds=50), 50.0)
            complete_chunk(output, manifest, 1, T0 + timedelta(hours=3), 50.0)  # resumed 3 h later
            result = run_status.status(output, logs=[], now=T0 + timedelta(hours=3, minutes=1))
            [gap] = result["gaps"]
            self.assertEqual(gap["chunks"], [1, 2])
            self.assertIn("approx.", gap["source"])
            self.assertAlmostEqual(gap["minutes"], (3 * 3600 - 100) / 60, places=1)

    def test_invalid_chunk_stops_the_count_and_is_explained(self):
        with tempfile.TemporaryDirectory() as directory:
            output, manifest = make_run(Path(directory))
            complete_chunk(output, manifest, 0, T0 + timedelta(seconds=50), 50.0)
            complete_chunk(output, manifest, 1, T0 + timedelta(seconds=100), 50.0, {"fingerprint": "x"})
            complete_chunk(output, manifest, 2, T0 + timedelta(seconds=150), 50.0)
            result = run_status.status(output, logs=[], now=T0 + timedelta(seconds=160))
            self.assertEqual(result["chunks_complete"], 1)
            notes = " ".join(result["notes"])
            self.assertIn("chunk 2: fingerprint differs", notes)
            self.assertIn("chunks 3 are complete but follow an incomplete chunk", notes)

    def test_verify_hashes_catches_edited_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output, manifest = make_run(Path(directory))
            folder = complete_chunk(output, manifest, 0, T0 + timedelta(seconds=50), 50.0)
            (folder / "events.jsonl").write_text("edited\n")
            now = T0 + timedelta(minutes=1)
            self.assertEqual(run_status.status(output, logs=[], now=now)["chunks_complete"], 1)
            verified = run_status.status(output, logs=[], now=now, verify=True)
            self.assertEqual(verified["chunks_complete"], 0)
            self.assertIn("hash differs", " ".join(verified["notes"]))


class NewTimingFieldsTest(unittest.TestCase):
    """Fields added by the long-run timing task are used when present."""

    def test_timestamps_and_pauses_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            output, manifest = make_run(Path(directory))
            iso = lambda dt: dt.isoformat(timespec="seconds")
            pause = {"start": iso(T0 + timedelta(seconds=70)), "end": iso(T0 + timedelta(seconds=3670)),
                     "seconds": 3600.0}
            complete_chunk(output, manifest, 0, T0, 50.0, {"started_at": iso(T0), "finished_at": iso(T0 + timedelta(seconds=50))})
            complete_chunk(output, manifest, 1, T0, 3650.0, {"started_at": iso(T0 + timedelta(seconds=60)),
                                                               "finished_at": iso(T0 + timedelta(seconds=3710)),
                                                               "pauses": [pause]})
            complete_chunk(output, manifest, 2, T0, 50.0, {"started_at": iso(T0 + timedelta(hours=2)),
                                                             "finished_at": iso(T0 + timedelta(hours=2, seconds=50))})
            (output / "progress.json").write_text(json.dumps({"frames_done": 30, "frames_total": 100,
                                                              "updated_at": iso(T0 + timedelta(hours=2, seconds=51))}))
            result = run_status.status(output, logs=[], now=T0 + timedelta(hours=2, minutes=5))
            self.assertEqual(result["state"], "running")
            self.assertEqual(result["last_progress_basis"], "progress.json updated_at")
            self.assertAlmostEqual(result["rate_fps"], 30 / 150)  # the hour-long pause is excluded
            self.assertIn("60.0 min of recorded pauses excluded", result["rate_basis"])
            [pause_out] = result["pauses"]
            self.assertEqual(pause_out["minutes"], 60.0)
            [gap] = result["gaps"]
            self.assertEqual(gap["source"], "between chunks (chunk.json timestamps)")
            self.assertEqual(gap["chunks"], [2, 3])

    def test_report_pauses_are_listed(self):
        with tempfile.TemporaryDirectory() as directory:
            output, manifest = make_run(Path(directory), frames=10)
            complete_chunk(output, manifest, 0, T0, 50.0)
            (output / "longrun-report.json").write_text(json.dumps({
                "fingerprint": manifest["fingerprint"], "chunk_seconds": [50.0],
                "pauses": [{"start": "2026-09-26T02:31:00+01:00", "end": "2026-09-26T05:30:00+01:00"}]}))
            result = run_status.status(output, logs=[], now=T0)
            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["pauses"][0]["minutes"], 179.0)


class LogTest(unittest.TestCase):
    def test_iso_log_gap_like_the_overnight_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            output, _, last = old_format_partial_run(root)
            log = root / "claude-overnight-chain.log"
            log.write_text("2026-09-26 02:25:00 [game] 1/4 detection + tracking -> runs\\game\n"
                           "2026-09-26 02:30:50 [longrun] 1780/5654 frames (31.5%), 150.0 min elapsed, 0.20 fps, ETA 321.0 min\n"
                           "2026-09-26 02:31:00 [longrun] chunk 6/19 complete: frames 1500-1799\n"
                           "a line without a stamp\n"
                           "2026-09-26 05:30:12 Stopping: session ended\n", encoding="utf-8")
            stamp(log, datetime(2026, 9, 26, 5, 30, 12).astimezone())
            now = datetime(2026, 9, 26, 7, 0, tzinfo=TZ)
            result = run_status.status(output, now=now)  # log found by name
            self.assertEqual([entry["path"] for entry in result["logs"]], [str(log)])
            gaps = [g for g in result["gaps"] if g["source"].startswith("log")]
            [gap] = gaps
            self.assertEqual(gap["lines"], [3, 5])
            self.assertEqual(gap["minutes"], 179.2)
            self.assertIn("chunk 6/19 complete", gap["before"])
            text = run_status.render(result)
            self.assertIn("02:31:00", text)
            self.assertIn("05:30:12", text)
            # The last [longrun] line (log stamps are local time) is a progress signal too.
            self.assertEqual(result["last_progress"], max(last, datetime(2026, 9, 26, 2, 31).astimezone()))

    def test_time_only_log_rolls_over_midnight_and_utf16(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            text = "[23:50:00] [longrun] a\n[23:59:00] [longrun] b\n[00:30:00] [longrun] c\n[03:00:05] d\n"
            log.write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))
            stamp(log, datetime(2026, 9, 26, 3, 0, 5).astimezone())  # local time, like the log
            parsed = run_status.parse_log(log)
            self.assertTrue(parsed["date_inferred_from_file_time"])
            self.assertEqual(parsed["timestamped_lines"], 4)
            self.assertEqual(parsed["entries"][0]["at"].day, 25)
            self.assertEqual(parsed["entries"][-1]["at"].day, 26)
            gaps = run_status.log_gaps(parsed, 600)
            self.assertEqual([g["minutes"] for g in gaps], [31.0, 150.1])
            self.assertEqual(parsed["last_longrun"]["text"], "[00:30:00] [longrun] c")

    def test_stamp_formats(self):
        seconds = lambda line: run_status.line_stamp(line)[1]
        self.assertEqual(run_status.line_stamp("2026-09-26T02:31:05.123+01:00 x")[1],
                         datetime(2026, 9, 26, 2, 31, 5, tzinfo=TZ))
        self.assertEqual(seconds("9/26/2026 2:31:05 AM [longrun] x"), 2 * 3600 + 31 * 60 + 5)
        self.assertEqual(seconds("26/09/2026 14:31 x"), 14 * 3600 + 31 * 60)
        for plain in ("[longrun] 1800/5654 frames (31.8%), 150.0 min elapsed, 0.20 fps, ETA 321.0 min",
                      "[longrun] chunk 6/19 complete: frames 1500-1799 (source 1500-1799, 50.0s-60.0s)",
                      "[game] 1/4 detection + tracking -> runs\\x"):
            self.assertIsNone(run_status.line_stamp(plain))

    def test_log_without_timestamps_is_noted(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _, last = old_format_partial_run(Path(directory))
            (output / "run.log").write_text("[longrun] 10/100 frames\n[longrun] chunk 1/10 complete\n")
            result = run_status.status(output, now=last)
            self.assertIn("has no wall-clock timestamps", " ".join(result["notes"]))

    def test_find_logs_matches_the_run_name_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _ = make_run(root, name="claude-game-v1-full")
            (root / "claude-game-v1-full.log").write_text("x\n")
            (root / "chain.log").write_text("-RunDir runs\\claude-game-v1-full\n")
            (root / "other.log").write_text("runs\\claude-game-v1-full-2 and runs\\claude-game-v1\n")
            (output / "inside.log").write_text("x\n")
            found = sorted(p.name for p in run_status.find_logs(output))
            self.assertEqual(found, ["chain.log", "claude-game-v1-full.log", "inside.log"])


class RunKindsTest(unittest.TestCase):
    def test_never_started_and_non_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _ = make_run(root)
            result = run_status.status(output, logs=[], now=T0 + timedelta(days=1))
            self.assertEqual(result["state"], "not started")
            self.assertFalse(result.get("stalled"))
            self.assertIn("unknown until a chunk completes", run_status.render(result))
            (root / "empty").mkdir()
            self.assertEqual(run_status.status(root / "empty", logs=[])["state"], "not started")
            (root / "single").mkdir()
            (root / "single" / "summary.json").write_text("{}")
            self.assertEqual(run_status.status(root / "single", logs=[])["state"], "not a long run")
            self.assertEqual(run_status.status(root / "absent", logs=[])["state"], "missing")

    def test_pipeline_layout_and_merge_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            output, manifest = make_run(Path(directory) / "claude-game-v1", frames=20, name="run")
            complete_chunk(output, manifest, 0, T0 + timedelta(seconds=50), 50.0)
            complete_chunk(output, manifest, 1, T0 + timedelta(seconds=100), 50.0)
            result = run_status.status(output.parent, logs=[], now=T0 + timedelta(minutes=2))
            self.assertEqual(result["run_folder"], str(output))
            self.assertEqual(result["state"], "chunks complete, merge pending")
            self.assertIn("none: detection finished", run_status.render(result))

    def test_committed_complete_run(self):
        run = PROJECT / "runs" / "claude-longrun-parity-30"
        if not (run / "longrun-report.json").is_file():
            self.skipTest("committed parity run not present")
        result = run_status.status(run, logs=[])
        self.assertEqual(result["state"], "complete")
        self.assertEqual((result["frames_done"], result["frames_total"]), (30, 30))
        self.assertEqual((result["chunks_complete"], result["chunks_total"]), (3, 3))
        self.assertEqual(result["source_covered_s"], [158.0, 159.0])
        self.assertAlmostEqual(result["rate_fps"], 30 / (152.4 + 222.68 + 195.11))

    def test_committed_manifest_only_run(self):
        run = PROJECT / "runs" / "claude-game-v1-full"
        manifest = run / "longrun-manifest.json"
        if not manifest.is_file():
            self.skipTest("committed manifest not present")
        result = run_status.status(run, logs=[])
        self.assertEqual((result["frames_total"], result["chunks_total"]), (5654, 19))
        self.assertEqual(result["source_selection_s"], [0.0, 188.47])
        if not (run / "chunks").exists():  # the clean checkout: no chunks, no progress
            self.assertEqual(result["state"], "not started")


class ReadOnlyAndCliTest(unittest.TestCase):
    def test_cli_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _, _ = old_format_partial_run(root)
            (root / "chain.log").write_text("2026-09-26 02:00:00 runs\\game [longrun] x\n")
            (output / "progress.json").write_text("{}")
            before = snapshot(root)
            for argv in ([str(output)], [str(output), "--json", "--verify-hashes"], [str(root / "absent")]):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_status.main(argv)
            self.assertEqual(snapshot(root), before)

    def test_cli_json_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _, _ = old_format_partial_run(Path(directory))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = run_status.main([str(output), "--json", "--stall-minutes", "1"])
            data = json.loads(buffer.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(data["frames_done"], 60)
            self.assertEqual(data["stall_minutes"], 1)
            self.assertTrue(data["stalled"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_status.main([str(Path(directory) / "absent")]), 1)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_status.main([str(output), "--stall-minutes", "0"])

    def test_committed_runs_are_untouched(self):
        runs = [PROJECT / "runs" / name for name in ("claude-longrun-parity-30", "claude-game-v1-full")]
        runs = [run for run in runs if run.is_dir()]
        before = {run: snapshot(run) for run in runs}
        for run in runs:
            with contextlib.redirect_stdout(io.StringIO()):
                run_status.main([str(run)])
        self.assertEqual({run: snapshot(run) for run in runs}, before)


if __name__ == "__main__":
    unittest.main()
