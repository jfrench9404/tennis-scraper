"""score_labels: candidates vs human labels in SOURCE-VIDEO frames, inside labelled spans only."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tennis_vision.labels import SHOT_TYPES, LabelsError
from tennis_vision.score_labels import (_shot_type_agreement, load_point_proposals, load_run, main, match_frames,
                                        score)

RUN_ID = "a" * 64          # review/replay fingerprint
LONGRUN_ID = "c" * 64      # long-run manifest fingerprint
VIDEO = "b" * 64           # source video SHA-256
START = 1000               # run frame 0 == source frame 1000


def dump(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def hit(eid, frame, player="near", **extra):
    return {"id": eid, "type": "hit", "status": "unreviewed", "frame": frame, "player_id": player, **extra}


def shot(eid, frame, classification, status="candidate"):
    return {"event_id": eid, "frame": frame, "classification": classification, "classification_status": status,
            "action_state": "shot_candidate"}


def write_longrun(folder, contacts=(), bounces=()):
    folder.mkdir()
    dump(folder / "longrun-manifest.json", {
        "fingerprint": LONGRUN_ID, "input": {"path": "C:\\media\\match.mp4", "sha256": VIDEO, "fps": 30.0},
        "selection": {"source_start_frame": START, "source_end_frame_exclusive": START + 300, "frames": 300}})
    dump(folder / "source-offset.json", {"source_start_frame": START})
    dump(folder / "shots.json", {"contacts": [{"type": "contact_candidate", "frame": f} for f in contacts],
                                 "bounces": [{"frame": f} for f in bounces], "shots": []})


def write_replay(folder, input_run="C:\\Users\\John\\project\\runs\\longrun"):
    """Synthetic replay folder. Candidate frames are RUN-relative (source = frame + 1000)."""
    folder.mkdir()
    dump(folder / "build-status.json", {"status": "complete"})
    dump(folder / "replay-report.json", {"review_run_id": RUN_ID, "source_start_frame": START, "frames": 300,
                                         "fps": 30.0, "input_run": input_run})
    events = [
        hit("hit-050", 50),                          # exact match for shot L1 (source 1050)
        hit("hit-083", 83, "far"),                   # +3 frames: inside default tolerance (L2)
        hit("hit-114", 114),                         # +4 frames from L3: outside tolerance -> missed + extra
        hit("hit-139", 139, "far"),                  # -1 frame; label says "unsure" (L4)
        hit("hit-170", 170, None),                   # exact; pipeline type unknown, no hitter (L5)
        hit("hit-120", 120),                         # extra inside labelled span
        hit("hit-250", 250),                         # extra outside labelled span -> ignored
        # Reviewer moved it onto L6 (1190); the pipeline said 186, which is what gets scored.
        hit("hit-186", 190, original_frame=186, status="confirmed"),
        hit("contact_window-160", 160, "far", support="review_window_only", contact_support="review_window_only"),
        hit("manual-1", 60, status="confirmed"),     # human-added: never a pipeline candidate
        {"id": "bounce-060", "type": "bounce", "status": "unreviewed", "frame": 60, "player_id": None},
        {"id": "bounce-102", "type": "bounce", "status": "unreviewed", "frame": 102, "player_id": None},
        {"id": "bounce-175", "type": "bounce", "status": "unreviewed", "frame": 175, "player_id": None},
    ]
    dump(folder / "reviewed-events.json", {"run_id": RUN_ID, "events": events})
    dump(folder / "shot-candidates.json", {"shots": [
        shot("hit-050", 50, "forehand"), shot("hit-083", 83, "forehand"), shot("hit-139", 139, "backhand"),
        shot("hit-170", 170, "unknown"), shot("hit-114", 114, "serve")]})


def labels(**overrides):
    data = {
        "schema_version": 1, "kind": "tennis_ground_truth_labels",
        "binding": {"run_id": RUN_ID, "source_video_sha256": VIDEO, "fps": 30.0},
        "labeller": {"name": "Synthetic test", "date": "2026-09-28"},
        "shot_types": list(SHOT_TYPES),
        "coverage": [{"source_start_frame": 900, "source_end_frame": 1199, "kinds": ["shots", "bounces", "points"]}],
        "shots": [
            {"id": "L1", "source_frame": 1050, "hitter": "near", "shot_type": "forehand", "decision": "human"},
            {"id": "L2", "source_frame": 1080, "hitter": "far", "shot_type": "backhand", "decision": "human"},
            {"id": "L3", "source_frame": 1110, "hitter": "near", "shot_type": "serve", "decision": "human"},
            {"id": "L4", "source_frame": 1140, "hitter": "far", "shot_type": "unsure", "decision": "human"},
            {"id": "L5", "source_frame": 1170, "hitter": "near", "shot_type": "volley", "decision": "human"},
            {"id": "L6", "source_frame": 1190, "hitter": "near", "shot_type": "forehand", "decision": "human"},
            {"id": "spot", "source_frame": 1250, "hitter": "far", "shot_type": "forehand", "decision": "human"},
            {"id": "late", "source_frame": 5000, "hitter": "far", "shot_type": "forehand", "decision": "human"},
        ],
        "bounces": [
            {"id": "B1", "source_frame": 1060, "call": "in", "decision": "human"},
            {"id": "B2", "source_frame": 1100, "call": "out", "decision": "human"},
            {"id": "B3", "source_frame": 1130, "call": "unsure", "decision": "human"},
        ],
        "points": [
            {"id": "P1", "source_start_frame": 1005, "source_end_frame": 1100, "server": "near", "winner": "far",
             "decision": "human"},
            {"id": "P2", "source_start_frame": 1110, "source_end_frame": 1195, "server": "far", "winner": "unknown",
             "decision": "human"},
            {"id": "P3", "source_start_frame": 1250, "source_end_frame": 1290, "server": "far", "winner": "far",
             "decision": "human"},
        ],
    }
    data.update(overrides)
    return data


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_longrun(self.root / "longrun", contacts=[50, 90], bounces=[60])
        write_replay(self.root / "replay")
        self.labels_path = self.root / "labels.json"
        dump(self.labels_path, labels())

    def tearDown(self):
        self._tmp.cleanup()


class MatchTest(unittest.TestCase):
    def test_one_to_one_closest_first(self):
        self.assertEqual(match_frames([10, 12], [11], 3), [(0, 0)])  # tie -> earlier label
        self.assertEqual(match_frames([10, 13], [12], 3), [(1, 0)])  # closest wins
        self.assertEqual(match_frames([10, 20], [13, 24], 3), [(0, 0)])  # 24 is 4 away
        self.assertEqual(match_frames([10, 11], [10, 11], 3), [(0, 0), (1, 1)])
        self.assertEqual(match_frames([], [5], 3), [])


class RunLoadingTest(Fixture):
    def test_replay_candidates_are_converted_to_source_frames(self):
        run = load_run(self.root / "replay")
        self.assertEqual((run["kind"], run["run_id"], run["source_start_frame"]), ("replay", RUN_ID, START))
        self.assertEqual(run["source_video_sha256"], VIDEO)
        self.assertIn("source run longrun", run["video_hash_basis"])
        first = run["hits"][0]
        self.assertEqual((first["run_frame"], first["source_frame"]), (50, 1050))
        moved = next(h for h in run["hits"] if h["id"] == "hit-186")
        self.assertEqual(moved["source_frame"], 1186)  # pipeline frame, not the reviewer's 190
        self.assertEqual(run["excluded_candidates"]["review_windows"], 1)
        self.assertEqual(run["excluded_candidates"]["human_added_manual_events"], 1)
        self.assertEqual(len(run["hits"]), 8)
        self.assertEqual(len(load_run(self.root / "replay", include_review_windows=True)["hits"]), 9)

    def test_parent_folder_prefers_replay(self):
        parent = self.root / "pipeline"
        parent.mkdir()
        write_replay(parent / "replay", input_run=str(self.root / "longrun"))
        self.assertEqual(load_run(parent)["kind"], "replay")

    def test_longrun_folder(self):
        run = load_run(self.root / "longrun")
        self.assertEqual((run["kind"], run["run_id"]), ("longrun", LONGRUN_ID))
        self.assertEqual([h["source_frame"] for h in run["hits"]], [1050, 1090])
        self.assertEqual([b["source_frame"] for b in run["bounces"]], [1060])

    def test_source_run_must_cover_the_same_frames(self):
        manifest_path = self.root / "longrun" / "longrun-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["selection"]["frames"] = 299
        dump(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "pass --video"):
            load_run(self.root / "replay")

    def test_unrecorded_hash_needs_video_and_video_must_match(self):
        write_replay(self.root / "legacy", input_run="C:\\gone\\rally-neural-ball")
        with self.assertRaisesRegex(ValueError, "does not record its source video"):
            load_run(self.root / "legacy")
        video = self.root / "fake.mp4"
        video.write_bytes(b"synthetic bytes standing in for a video")
        run = load_run(self.root / "legacy", video=video)
        self.assertEqual(run["source_video_sha256"], hashlib.sha256(video.read_bytes()).hexdigest())
        self.assertIn("--video", run["video_hash_basis"])
        with self.assertRaisesRegex(ValueError, "--video"):
            load_run(self.root / "replay", video=video)  # the run records VIDEO, not this file

    def test_review_folder_alignment_is_checked(self):
        review = self.root / "review"
        review.mkdir()
        dump(review / "review-report.json", {"run_id": RUN_ID, "source_start_frame": START, "frames": 300,
                                             "fps": 30.0, "source_run": str(self.root / "longrun")})
        event = {"id": "hit_candidate-000050", "type": "hit_candidate", "frame": 50, "source_frame": 1050,
                 "player_id": "near"}
        dump(review / "event-candidates.json", {"run_id": RUN_ID, "events": [event]})
        self.assertEqual(load_run(review)["hits"][0]["source_frame"], 1050)
        dump(review / "event-candidates.json", {"run_id": RUN_ID, "events": [dict(event, source_frame=1049)]})
        with self.assertRaisesRegex(ValueError, "alignment"):
            load_run(review)

    def test_incomplete_build_is_refused(self):
        dump(self.root / "replay" / "build-status.json", {"status": "running"})
        with self.assertRaisesRegex(ValueError, "not complete"):
            load_run(self.root / "replay")


class ScoreTest(Fixture):
    def test_hits_exact_tolerance_missed_extra_and_outside(self):
        hits = score(self.labels_path, self.root / "replay")["hits"]
        self.assertEqual(hits["scored_spans"], [[1000, 1199]])  # coverage clipped to the run's frames
        self.assertEqual(hits["labels_in_scored_spans"], 6)
        self.assertEqual(hits["labels_outside_coverage"], 1)
        self.assertEqual(hits["labels_outside_run_frames"], 1)
        self.assertEqual((hits["matched"], hits["missed"], hits["false_positives"]), (4, 2, 3))
        self.assertEqual(hits["candidates_outside_scored_spans"], 1)
        self.assertEqual(hits["precision"], round(4 / 7, 4))
        self.assertEqual(hits["recall"], round(4 / 6, 4))
        pairs = {m["label"]["id"]: (m["candidate"]["id"], m["offset_frames"]) for m in hits["details"]["matches"]}
        self.assertEqual(pairs, {"L1": ("hit-050", 0), "L2": ("hit-083", 3), "L4": ("hit-139", -1),
                                 "L5": ("hit-170", 0)})
        self.assertEqual({e["id"] for e in hits["details"]["missed"]}, {"L3", "L6"})
        self.assertEqual({c["id"] for c in hits["details"]["false_positives"]}, {"hit-114", "hit-120", "hit-186"})
        self.assertEqual(hits["timing"]["max_abs_offset_frames"], 3)
        self.assertIn("accuracy on 6 labelled shots from labels.json", hits["statement"])
        self.assertIn("precision 0.571 (4/7", hits["statement"])
        self.assertIn("recall 0.667 (4/6", hits["statement"])

    def test_hitter_and_shot_type_agreement_count_abstentions_separately(self):
        hits = score(self.labels_path, self.root / "replay")["hits"]
        self.assertEqual({k: hits["hitter_agreement"][k] for k in ("compared", "agree", "pipeline_gave_no_hitter")},
                         {"compared": 3, "agree": 3, "pipeline_gave_no_hitter": 1})
        types = hits["shot_type_agreement"]
        self.assertEqual((types["compared"], types["agree"]), (2, 1))
        self.assertEqual(types["not_compared"]["label_unsure"], 1)
        self.assertEqual(types["not_compared"]["pipeline_unknown"], 1)
        self.assertEqual(types["confusion_label_to_pipeline"], {"forehand": {"forehand": 1}, "backhand": {"forehand": 1}})
        self.assertIn("where both the label and the pipeline name a type: 1/2", types["statement"])

    def test_shot_type_exclusions(self):
        def match(label_type, cand_type, status="candidate"):
            return {"label": {"shot_type": label_type},
                    "candidate": {"shot_type": cand_type, "shot_type_status": status}}
        result = _shot_type_agreement([match("forehand", "forehand", "human_reviewed"), match("serve", None),
                                       match("unsure", "unknown"), match("volley", "not_a_shot"),
                                       match("other", "backhand")], "labels.json")
        self.assertEqual(result["not_compared"], {"pipeline_type_set_by_human_review": 1, "pipeline_has_no_type": 1,
                                                  "label_unsure": 1, "pipeline_unknown": 0,
                                                  "pipeline_not_a_shot": 1})
        self.assertEqual((result["compared"], result["agree"]), (1, 0))  # "other" never agrees

    def test_tolerance_is_configurable(self):
        wide = score(self.labels_path, self.root / "replay", tolerance_frames=4)["hits"]
        self.assertEqual((wide["matched"], wide["false_positives"]), (6, 1))
        self.assertEqual(wide["recall"], 1.0)
        exact = score(self.labels_path, self.root / "replay", tolerance_frames=0)["hits"]
        self.assertEqual({m["label"]["id"] for m in exact["details"]["matches"]}, {"L1", "L5"})
        with self.assertRaises(ValueError):
            score(self.labels_path, self.root / "replay", tolerance_frames=-1)

    def test_review_windows_only_when_asked(self):
        hits = score(self.labels_path, self.root / "replay", include_review_windows=True)["hits"]
        self.assertIn("contact_window-160", {c["id"] for c in hits["details"]["false_positives"]})

    def test_bounces(self):
        bounces = score(self.labels_path, self.root / "replay")["bounces"]
        self.assertEqual((bounces["matched"], bounces["missed"], bounces["false_positives"]), (2, 1, 1))
        self.assertEqual(bounces["label_calls"]["matched"], {"in": 1, "out": 1, "unsure": 0})
        self.assertEqual(bounces["label_calls"]["missed"], {"in": 0, "out": 0, "unsure": 1})
        self.assertIn("accuracy on 3 labelled bounces from labels.json", bounces["statement"])

    def test_nothing_scored_without_coverage(self):
        dump(self.labels_path, labels(coverage=[]))
        result = score(self.labels_path, self.root / "replay")
        self.assertEqual(result["hits"]["status"], "not_scored")
        self.assertIsNone(result["hits"]["precision"])
        self.assertIn("no span fully labelled for shots", result["hits"]["statement"])
        self.assertEqual(result["bounces"]["status"], "not_scored")

    def test_zero_candidates_leaves_precision_uncomputed(self):
        write_longrun(self.root / "empty")
        dump(self.labels_path, labels(binding={"run_id": LONGRUN_ID, "source_video_sha256": VIDEO, "fps": 30.0}))
        hits = score(self.labels_path, self.root / "empty")["hits"]
        self.assertIsNone(hits["precision"])
        self.assertEqual(hits["recall"], 0.0)
        self.assertIn("precision not computed (0 candidates", hits["statement"])

    def test_binding_mismatch_is_rejected(self):
        dump(self.labels_path, labels(binding={"run_id": "d" * 64, "source_video_sha256": VIDEO, "fps": 30.0}))
        with self.assertRaisesRegex(LabelsError, "bound to run"):
            score(self.labels_path, self.root / "replay")
        dump(self.labels_path, labels(binding={"run_id": RUN_ID, "source_video_sha256": "d" * 64, "fps": 30.0}))
        with self.assertRaisesRegex(LabelsError, "bound to source video"):
            score(self.labels_path, self.root / "replay")
        # Labels made for the review fingerprint do not score the long run's own id.
        dump(self.labels_path, labels())
        with self.assertRaisesRegex(LabelsError, "bound to run"):
            score(self.labels_path, self.root / "longrun")


class PointTest(Fixture):
    def proposals(self, data):
        path = self.root / "points.json"
        dump(path, data)
        return path

    def test_point_proposals_with_seconds_tolerance(self):
        path = self.proposals({"run_id": RUN_ID, "points": [
            {"start_frame": 10, "end_frame": 95, "server": "near", "winner": "far"},     # P1 within 5 frames
            {"source_start_frame": 1112, "source_end_frame": 1240},                     # end 45 frames late
            {"start_frame": 260, "end_frame": 280}]})                                   # outside coverage
        points = score(self.labels_path, self.root / "replay", points_path=path)["points"]
        self.assertEqual(points["tolerance_frames"], 30)
        self.assertEqual(points["labels_in_scored_spans"], 2)
        self.assertEqual((points["matched"], points["missed"], points["false_positives"]), (1, 1, 1))
        self.assertEqual(points["proposals_outside_scored_spans"], 1)
        self.assertEqual((points["precision"], points["recall"]), (0.5, 0.5))
        match = points["details"]["matches"][0]
        self.assertEqual((match["label"]["id"], match["start_offset_frames"], match["end_offset_frames"]),
                         ("P1", 5, -5))
        self.assertEqual(points["agreement"]["server"], {"compared": 1, "agree": 1,
                                                         "not_compared_label_or_proposal_unknown": 0})
        self.assertIn("accuracy on 2 labelled points from labels.json", points["statement"])
        tight = score(self.labels_path, self.root / "replay", points_path=path, point_tolerance_s=0.1)["points"]
        self.assertEqual(tight["matched"], 0)

    def test_points_not_scored_without_proposals(self):
        points = score(self.labels_path, self.root / "replay")["points"]
        self.assertEqual(points["status"], "not_scored")
        self.assertIn("no --points", points["statement"])

    def test_proposal_file_checks(self):
        run = load_run(self.root / "replay")
        self.assertEqual(load_point_proposals(self.proposals([{"start_frame": 0, "end_frame": 9}]), run)[0]
                         ["source_start_frame"], START)
        for bad in ({"run_id": "e" * 64, "points": []}, [{"start": 0, "end": 5}], [{"start_frame": 9, "end_frame": 1}]):
            with self.assertRaises(ValueError):
                load_point_proposals(self.proposals(bad), run)


class CliTest(Fixture):
    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def test_writes_output_and_never_touches_labels(self):
        before = self.labels_path.read_bytes()
        mtime = self.labels_path.stat().st_mtime_ns
        output = self.root / "scores" / "score.json"
        code, out, err = self.run_cli("--labels", self.labels_path, "--run", self.root / "replay", "--output", output)
        self.assertEqual(code, 0, err)
        self.assertIn("accuracy on 6 labelled shots from labels.json", out)
        result = json.loads(output.read_text())
        self.assertEqual(result["labels"]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(result["frame_basis"]["conversion"], "source_frame = run_frame + 1000")
        self.assertEqual(result["frame_basis"]["run_source_frames"], [1000, 1299])
        self.assertTrue(any("Nothing is claimed outside" in s for s in result["summary"]))
        self.assertEqual(self.labels_path.read_bytes(), before)
        self.assertEqual(self.labels_path.stat().st_mtime_ns, mtime)
        # Existing output is kept, and the labels file can never be the output.
        code, _, err = self.run_cli("--labels", self.labels_path, "--run", self.root / "replay", "--output", output)
        self.assertEqual(code, 1)
        self.assertIn("already exists", err)
        code, _, err = self.run_cli("--labels", self.labels_path, "--run", self.root / "replay",
                                    "--output", self.labels_path)
        self.assertEqual(code, 1)
        self.assertIn("never modified", err)
        self.assertEqual(self.labels_path.read_bytes(), before)

    def test_mismatched_labels_fail_cleanly(self):
        dump(self.labels_path, labels(binding={"run_id": "d" * 64, "source_video_sha256": VIDEO, "fps": 30.0}))
        output = self.root / "score.json"
        code, _, err = self.run_cli("--labels", self.labels_path, "--run", self.root / "replay", "--output", output)
        self.assertEqual(code, 1)
        self.assertIn("bound to run", err)
        self.assertFalse(output.exists())


class CommittedRunTest(unittest.TestCase):
    """The committed pipeline test run resolves its video hash through its source long run."""

    def test_claude_pipeline_test_30(self):
        folder = Path(__file__).resolve().parent.parent / "runs" / "claude-pipeline-test-30"
        if not (folder / "replay" / "reviewed-events.json").is_file() or \
                not (folder.parent / "claude-longrun-dml-30" / "longrun-manifest.json").is_file():
            self.skipTest("committed pipeline test run not present")
        run = load_run(folder)
        self.assertEqual(run["kind"], "replay")
        self.assertEqual(run["source_start_frame"], 4740)
        self.assertEqual(run["source_video_sha256"],
                         "0eb4675b7e21a672d35b1ac8c017dde81b3cb175877c48ce60de0e34151b104b")


if __name__ == "__main__":
    unittest.main()
