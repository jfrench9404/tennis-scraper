"""Offline court-side re-association and person tracks, on synthetic rows that mirror
the real run format (``detail.candidates`` with COCO keypoints, run-folder files)."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import cv2
import numpy as np

from tennis_vision import longrun
from tennis_vision.compare_runs import compare
from tennis_vision.court import CourtMapper
from tennis_vision.event_review import load_run as review_load_run
from tennis_vision.reassociate import (DEFAULTS, build_tracklets, load_source, main as reassociate_main,
                                       observations, reassociate, write_run)
from tennis_vision.segments import load_run as segments_load_run

FPS = 30.0
START = 4740
# Same four corners as project/court.yaml (near-left, near-right, far-right, far-left).
CORNERS = [[3, 535], [1188, 570], [782, 133], [427, 121]]
COURT_YAML = "image_corners:\n" + "".join(f"- - {x}\n  - {y}\n" for x, y in CORNERS)
COURT = CourtMapper(np.asarray(CORNERS, np.float32))
INVERSE = np.linalg.inv(COURT.matrix)
JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
          "right_knee", "left_ankle", "right_ankle")
# Joint positions as (fraction of width from left, fraction of height from top).
LAYOUT = (.5, .06), (.47, .05), (.53, .05), (.44, .06), (.56, .06), (.3, .2), (.7, .2), (.22, .35), (.78, .35), \
    (.2, .48), (.8, .48), (.38, .52), (.62, .52), (.38, .75), (.62, .75), (.4, 1.), (.6, 1.)


def pixel(x, y):
    return cv2.perspectiveTransform(np.asarray([[[x, y]]], np.float32), INVERSE)[0, 0].tolist()


def candidate(x, y, confidence=.85, pose=True):
    """A saved pre-filter player candidate whose ankles stand at court (x, y) metres."""
    u, v = pixel(x, y)
    ppm = np.hypot(*np.subtract(pixel(x + 1, y), (u, v)))
    h = 2.6 * ppm
    w = .4 * h
    box = [u - w / 2, v - h, u + w / 2, v + .02 * h]
    keypoints = {name: [round(box[0] + fx * w, 2), round(box[1] + fy * h, 2), .9]
                 for name, (fx, fy) in zip(JOINTS, LAYOUT)} if pose else None
    if pose:
        keypoints["left_ankle"][:2] = [round(u - .1 * w, 2), round(v, 2)]
        keypoints["right_ankle"][:2] = [round(u + .1 * w, 2), round(v, 2)]
    return {"label": "player", "bbox": box, "confidence": confidence, "keypoints": keypoints, "source": "model"}


def track_for(c, identity, track_id):
    """An ORIGINAL run track built from a candidate (what cli.FrameAnalyzer writes)."""
    from tennis_vision.tracking import Track
    return Track("player", tuple(c["bbox"]), c["confidence"], track_id, 0, c["keypoints"],
                 identity_id=identity).as_dict(COURT)


def lerp(a, b, t):
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def make_rows(frames, people, original=None, extra_tracks=None):
    """people(f) -> {name: (x, y) or (x, y, kwargs)}; original(f, cands) -> {identity: name}."""
    rows = []
    for f in range(frames):
        cands, named = [], {}
        for name, spec in people(f).items():
            c = candidate(*spec[:2], **(spec[2] if len(spec) > 2 else {}))
            named[name] = c
            cands.append(c)
        tracks = [track_for(named[name], identity, 1 + i)
                  for i, (identity, name) in enumerate((original(f) if original else {}).items()) if name in named]
        tracks += (extra_tracks(f, tracks) if extra_tracks else [])
        rows.append({"frame": f, "time_s": round(f / FPS, 4), "tracks": tracks, "events": [],
                     "filters": {"off_court_players": 0}, "source_frame": START + f,
                     "detail": {"crop_passes": 2, "candidates": cands, "scene": {"players_unselected": 0}}})
    return rows


def write_source(folder, rows):
    folder.mkdir(parents=True)
    (folder / "source-offset.json").write_text(json.dumps({"source_start_frame": START,
                                                           "source_start_seconds": START / FPS}))
    (folder / "summary.json").write_text(json.dumps({"input": "synthetic.mp4", "frames": len(rows), "fps": FPS,
                                                     "width": 1280, "height": 720, "events": 0}))
    (folder / "shots.json").write_text(json.dumps({"contacts": [], "bounces": [], "shots": []}))
    with (folder / "events.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    court = folder.parent / "court.yaml"
    court.write_text(COURT_YAML)
    return court


def run(rows, **settings):
    loaded = {"run": Path("synthetic"), "rows": rows, "fps": FPS, "source_start_frame": START, "partial": False,
              "source": "merged", "selection_frames": len(rows), "manifest": None, "court": COURT,
              "court_basis": "test", "court_corners": CORNERS, "rows_sha256": "0"}
    return reassociate(loaded, settings)


def players(row):
    return {t["identity_id"]: t for t in row["tracks"] if t["label"] == "player"}


def walk_changeover():
    """A near and B far; both walk across the net (5 s, never occluded) and swap ends."""
    def people(f):
        t = min(1., max(0., (f - 30) / 150))
        return {"A": lerp((3, -1), (3, 25), t), "B": lerp((8, 25), (8, -1), t)}
    return make_rows(240, people)


class SideRoles(unittest.TestCase):
    def test_changeover_walk_keeps_person_and_flips_side(self):
        result = run(walk_changeover())
        rows, report = result["rows"], result["report"]
        first, last = players(rows[0]), players(rows[-1])
        self.assertEqual((first["near"]["person_id"], first["far"]["person_id"]), ("player_a", "player_b"))
        self.assertEqual((last["near"]["person_id"], last["far"]["person_id"]), ("player_b", "player_a"))
        for row in rows:
            for side, track in players(row).items():
                self.assertEqual(side == "near", track["court_m"][1] < 11.885)
                self.assertNotEqual(track["person_id"], "unknown")
        table = report["stretches"]
        a = [s for s in table if s["person"] == "player_a"]
        b = [s for s in table if s["person"] == "player_b"]
        self.assertEqual([s["side"] for s in a], ["near", "far"])
        self.assertEqual([s["side"] for s in b], ["far", "near"])
        self.assertIn("walk across the net observed", a[1]["basis"])
        self.assertIn("walk across the net observed", b[1]["basis"])
        for s in table:
            self.assertTrue({"person", "side", "source_start_frame", "source_end_frame", "basis"} <= set(s))
            self.assertGreaterEqual(s["source_start_frame"], START)
        self.assertEqual(report["low_confidence_proposals"], [])

    def test_ambiguous_crossing_marks_persons_unknown_but_sides_correct(self):
        # Warm-up near the net; both are hidden for 0.8 s and reappear on swapped
        # sides close enough that either could be either: never guess.
        def people(f):
            if f < 60:
                return {"A": (4.5, 10.0), "B": (6.0, 13.5)}
            if f < 84:
                return {}
            return {"B2": (5.5, 10.2), "A2": (5.0, 13.8)}
        rows, report = (lambda r: (r["rows"], r["report"]))(run(make_rows(150, people)))
        self.assertEqual({t["person_id"] for t in players(rows[30]).values()}, {"player_a", "player_b"})
        late = players(rows[120])
        self.assertEqual(set(late), {"near", "far"})
        self.assertLess(late["near"]["court_m"][1], 11.885)
        self.assertGreater(late["far"]["court_m"][1], 11.885)
        self.assertEqual({t["person_id"] for t in late.values()}, {"unknown"})
        unknown = [s for s in report["stretches"] if s["source_start_frame"] >= START + 84]
        self.assertEqual(len(unknown), 2)
        for s in unknown:
            self.assertEqual((s["person"], s["link_confidence"]), ("unknown", "none"))
            self.assertNotIn("proposed_person", s)

    def test_long_absence_changeover_is_unknown_not_guessed(self):
        def people(f):
            if f < 60:
                return {"A": (3, -1), "B": (8, 25)}
            if f < 210:  # 5 s break: nobody detected
                return {}
            return {"A": (3, 25), "B": (8, -1)}
        rows, report = (lambda r: (r["rows"], r["report"]))(run(make_rows(270, people)))
        late = players(rows[250])
        self.assertEqual(set(late), {"near", "far"})
        self.assertEqual({t["person_id"] for t in late.values()}, {"unknown"})
        self.assertEqual(report["low_confidence_proposals"], [])

    def test_elimination_is_only_a_low_confidence_proposal(self):
        # A walks across the net in view; B vanishes for 3 s and reappears near.
        def people(f):
            t = min(1., max(0., (f - 30) / 150))
            out = {"A": lerp((3, -1), (3, 25), t)}
            if f < 60:
                out["B"] = (8, 25)
            elif f >= 150:
                out["B"] = (8, -1)
            return out
        rows, report = (lambda r: (r["rows"], r["report"]))(run(make_rows(240, people)))
        last = players(rows[-1])
        self.assertEqual(last["far"]["person_id"], "player_a")
        self.assertEqual(last["near"]["person_id"], "unknown")  # never applied
        proposals = report["low_confidence_proposals"]
        self.assertEqual([(p["side"], p["proposed_person"]) for p in proposals], [("near", "player_b")])
        self.assertIn("needs John", proposals[0]["basis"])

    def test_bystander_never_holds_a_role(self):
        # Referee off court (x = -5 m) the whole time, briefly stepping inside the
        # side margin while the far player is undetected; the original run gave
        # "far" to the referee (as in video 1).
        def people(f):
            ref = (-5.0, 13.0)
            if 150 <= f < 165:
                ref = lerp((-5.0, 13.0), (-1.0, 13.0), (f - 150) / 15)
            elif 165 <= f < 180:
                ref = (-1.0, 13.0)
            elif 180 <= f < 195:
                ref = lerp((-1.0, 13.0), (-5.0, 13.0), (f - 180) / 15)
            out = {"A": (5, -1), "R": (*ref, {"confidence": .95})}
            if not 100 <= f < 220:
                out["B"] = (5, 25)
            return out
        original = lambda f: {"near": "A", "far": "R" if 100 <= f < 220 else "B"}
        rows, report = (lambda r: (r["rows"], r["report"]))(run(make_rows(300, people, original)))
        for f in range(100, 220):
            self.assertNotIn("far", players(rows[f]), f)
            self.assertEqual(rows[f]["reassociation"]["far"]["status"], "removed")
        self.assertIn("outside court margins", rows[110]["reassociation"]["far"]["reason"])
        for row in rows:
            for t in players(row).values():
                self.assertGreater(t["court_m"][0], -1.5)
        self.assertGreaterEqual(report["tracklets"]["banned_off_court"], 1)
        self.assertEqual(players(rows[50])["far"]["person_id"], "player_b")

    def test_warm_up_at_net_seeds_by_side(self):
        def people(f):
            t = min(1., max(0., (f - 60) / 90))
            return {"A": lerp((5, 11.0), (5, -1), t), "B": lerp((5.5, 15.0), (5.5, 25), t)}
        rows, report = (lambda r: (r["rows"], r["report"]))(run(make_rows(180, people)))
        first = players(rows[0])
        self.assertAlmostEqual(first["near"]["court_m"][1], 11.0, delta=.2)
        self.assertAlmostEqual(first["far"]["court_m"][1], 15.0, delta=.2)
        self.assertEqual(report["seed_source_frame"], START)
        self.assertEqual([(s["person"], s["side"]) for s in report["stretches"]],
                         [("player_b", "far"), ("player_a", "near")])
        self.assertTrue(all(s["basis"].startswith("seed") for s in report["stretches"]))

    def test_video_one_failure_is_repaired(self):
        # Source 5040: original "near" on the far-baseline player, "far" on an off-court
        # person, the near-baseline server unidentified.
        people = lambda f: {"S": (8.5, 0.1), "F": (9.5, 28.4), "X": (-6.0, 20.0)}
        rows, report = (lambda r: (r["rows"], r["report"]))(
            run(make_rows(120, people, lambda f: {"near": "F", "far": "X"})))
        row = rows[60]
        self.assertAlmostEqual(players(row)["near"]["court_m"][1], .1, delta=.2)
        self.assertAlmostEqual(players(row)["far"]["court_m"][1], 28.4, delta=.2)
        self.assertEqual(row["reassociation"]["near"],
                         {"status": "replaced", "reason": "original holder's feet are on the other side of the net"})
        self.assertEqual(row["reassociation"]["far"]["status"], "replaced")
        self.assertIn("outside court margins", row["reassociation"]["far"]["reason"])
        self.assertEqual(report["coverage"]["original"].get("near_frames_6plus_joints"), 120)
        self.assertEqual(report["coverage"]["reassociated"]["far_frames_6plus_joints"], 120)
        changed = report["changed_intervals"]
        self.assertEqual(sorted((c["side"], c["status"], c["source_frames"][0], c["source_frames"][1]) for c in changed),
                         [("far", "replaced", START, START + 119), ("near", "replaced", START, START + 119)])

    def test_brief_wide_excursion_keeps_the_player(self):
        # Near player chases a wide ball to x = 12.8 m (outside the 1.5 m margin) for 1 s.
        def people(f):
            x = 12.8 if 60 <= f < 90 else 9.0
            if 50 <= f < 60:
                x = 9 + (f - 50) * .38
            elif 90 <= f < 100:
                x = 12.8 - (f - 90) * .38
            return {"A": (x, 2.0), "B": (5, 25)}
        rows = run(make_rows(150, people))["rows"]
        self.assertTrue(all("near" in players(r) for r in rows))
        self.assertEqual({players(r)["near"]["person_id"] for r in rows}, {"player_a"})

    def test_never_invents_and_reassigns_racquets(self):
        def extra(f, tracks):
            return [{"track_id": 77, "identity_id": None, "label": "racket", "source": "model",
                     "bbox": [0, 0, 5, 5], "confidence": .5, "predicted": False, "keypoints": None,
                     "player_track_id": 1}]
        people = lambda f: {"S": (8.5, 0.1), "F": (9.5, 28.4)}
        rows = make_rows(30, people, lambda f: {"near": "F"}, extra)
        # Racquet at the near player's right wrist: belongs to the new near track.
        wrist = rows[10]["detail"]["candidates"][0]["keypoints"]["right_wrist"]
        for row in rows:
            row["tracks"][-1]["bbox"] = [wrist[0] - 4, wrist[1] - 4, wrist[0] + 4, wrist[1] + 4]
        result = run(rows)
        for source, new in zip(rows, result["rows"]):
            for t in players(new).values():
                c = source["detail"]["candidates"][t["candidate_index"]]
                self.assertEqual(t["keypoints"], c["keypoints"])
                self.assertEqual(t["bbox"], [round(v, 2) for v in c["bbox"]])
                self.assertFalse(t["predicted"])
        near_id = result["report"]["player_track_ids"]["near"]
        self.assertEqual(result["rows"][10]["tracks"][0]["player_track_id"], near_id)
        self.assertEqual(result["report"]["racket_assignments_changed"], 30)
        self.assertNotIn(near_id, {t["track_id"] for t in rows[0]["tracks"]})

    def test_crossing_people_end_tracklets_instead_of_swapping(self):
        obs = observations(make_rows(3, lambda f: {"A": (5.0, 2.0)} if f < 2 else
                                     {"A": (4.9, 2.0), "B": (5.1, 2.0)}), COURT, DEFAULTS)
        tracklets = build_tracklets(obs, 5)
        self.assertEqual(tracklets[0]["end_reason"], "ambiguous")
        self.assertEqual(len(tracklets[0]["frames"]), 2)


class RunFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def digest(self, folder):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}

    def test_writes_new_standard_run_and_leaves_source_untouched(self):
        source = self.root / "runs" / "source"
        court = write_source(source, walk_changeover())
        before = self.digest(source)
        output = self.root / "runs" / "reassociated"
        with contextlib.redirect_stdout(io.StringIO()):
            reassociate_main(["--run", str(source), "--output", str(output), "--court", str(court)])
        self.assertEqual(self.digest(source), before)
        for name in ("events.jsonl", "summary.json", "source-offset.json", "reassociation.json", "shots.json"):
            self.assertTrue((output / name).is_file(), name)
        report = json.loads((output / "reassociation.json").read_text())
        self.assertEqual(report["kind"], "court_side_reassociation")
        self.assertEqual(report["copied_unchanged"], ["shots.json"])
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["frames"], 240)
        self.assertEqual(summary["identity_pass"]["report"], "reassociation.json")
        _, start, rows = review_load_run(output)  # event review / replay entry point
        self.assertEqual((start, len(rows)), (START, 240))
        self.assertEqual(segments_load_run(output)["source"], "merged")
        overlap = compare(source, output)
        self.assertEqual(overlap["shared_frames"], 240)
        self.assertGreaterEqual(overlap["b"]["near_player_frames"], 235)  # near briefly empty mid-crossing
        # Same source, different thresholds -> different fingerprint; never overwrite.
        with self.assertRaises(SystemExit):
            reassociate_main(["--run", str(source), "--output", str(output), "--court", str(court)])
        with self.assertRaises(SystemExit):
            reassociate_main(["--run", str(source), "--output", str(source / "inside"), "--court", str(court)])
        other = self.root / "runs" / "other"
        with contextlib.redirect_stdout(io.StringIO()):
            reassociate_main(["--run", str(source), "--output", str(other), "--court", str(court),
                              "--absence-seconds", "4"])
        self.assertNotEqual(json.loads((other / "reassociation.json").read_text())["fingerprint"],
                            report["fingerprint"])
        self.assertEqual(json.loads((other / "reassociation.json").read_text())["settings"]["absence_seconds"], 4)

    def test_reads_complete_chunks_of_a_partial_long_run(self):
        rows = walk_changeover()
        folder = self.root / "runs" / "partial"
        folder.mkdir(parents=True)
        manifest = {"version": longrun.LONGRUN_VERSION,
                    "input": {"path": "synthetic.mp4", "sha256": "abc", "fps": FPS, "frames": START + 240,
                              "width": 1280, "height": 720},
                    "selection": {"source_start_frame": START, "source_end_frame_exclusive": START + 240,
                                  "frames": 240},
                    "settings": {"chunk_frames": 120, "court": COURT_YAML}, "models": {}, "code": {},
                    "chunks": longrun.plan_chunks(START, START + 240, 120)}
        manifest["fingerprint"] = longrun.fingerprint(manifest)
        longrun.save_json(folder / "longrun-manifest.json", manifest)
        chunk = manifest["chunks"][0]
        cdir = longrun.chunk_dir(folder, chunk)
        cdir.mkdir(parents=True)
        (cdir / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:120]))
        with (cdir / "state.pkl").open("wb") as stream:
            pickle.dump({}, stream)
        longrun.save_json(cdir / "chunk.json", {
            "status": "complete", "index": 0, "fingerprint": manifest["fingerprint"],
            "events_sha256": longrun.file_sha256(cdir / "events.jsonl"),
            "state_sha256": longrun.file_sha256(cdir / "state.pkl")})
        loaded = load_source(folder)  # court from the manifest
        self.assertTrue(loaded["partial"])
        output = self.root / "runs" / "partial-reassociated"
        report = write_run(loaded, reassociate(loaded), output)
        self.assertEqual(report["source_run"]["longrun_fingerprint"], manifest["fingerprint"])
        self.assertEqual(report["source_run"]["input_sha256"], "abc")
        self.assertEqual(json.loads((output / "summary.json").read_text())["frames"], 120)
        self.assertEqual(review_load_run(output)[1], START)
        self.assertFalse((output / "longrun-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
