"""One labels.json through every tool: validator -> shot_table --labels -> points -> score_labels --points.

Synthetic data only (no footage). The replay folder is built in the shape the three
readers expect: ``replay-data.json`` + ``play-context.json`` + ``shot-candidates.json``
(points, shot_table), ``reviewed-events.json`` + ``replay-report.json`` (score_labels),
and the source run's ``longrun-manifest.json`` that records the source video hash.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tennis_vision import labels as labels_module
from tennis_vision import points, score_labels, shot_table
from tennis_vision.labels import LabelsError, SHOT_TYPES

from test_points import FPS, OFFSET, Clip

RUN_ID = "12" * 32
VIDEO_SHA = "3f" * 32
FRAMES = 300


def quiet(function, argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(argv)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_replay(root):
    """Serve + three rally shots (run frames 20-65), ball seen 15-75, then a dead ball."""
    source_run = root / "source-run"
    source_run.mkdir(parents=True)
    (source_run / "longrun-manifest.json").write_text(json.dumps({
        "fingerprint": "ee" * 32, "input": {"sha256": VIDEO_SHA, "fps": FPS},
        "selection": {"source_start_frame": OFFSET, "frames": FRAMES}}))
    clip = (Clip(FRAMES).balls(15, 75).serve("s1", 20, "near").rally("r1", 35, "far").rally("r2", 50, "near")
            .rally("r3", 65, "far"))
    replay = clip.write(root / "replay")
    data = json.loads((replay / "replay-data.json").read_text())
    data["report"].update(review_run_id=RUN_ID, input_run=str(source_run))
    (replay / "replay-data.json").write_text(json.dumps(data))
    (replay / "replay-report.json").write_text(json.dumps(data["report"]))
    (replay / "reviewed-events.json").write_text(json.dumps({"run_id": RUN_ID, "events": data["events"]}))
    (replay / "build-status.json").write_text(json.dumps({"status": "complete"}))
    return replay


def write_labels(path, video_sha=VIDEO_SHA):
    """John's labels in SOURCE frames: four shots, one point won by far, one point with no recorded winner."""
    shot = lambda lid, f, hitter, kind: {"id": lid, "source_frame": OFFSET + f, "hitter": hitter, "shot_type": kind,
                                         "decision": "human"}
    labels = {
        "schema_version": 1, "kind": "tennis_ground_truth_labels",
        "binding": {"run_id": RUN_ID, "source_video_sha256": video_sha, "fps": FPS},
        "labeller": {"name": "test", "date": "2026-10-08"}, "shot_types": list(SHOT_TYPES),
        "coverage": [{"source_start_frame": OFFSET, "source_end_frame": OFFSET + FRAMES - 1,
                      "kinds": ["shots", "bounces", "points"]}],
        "shots": [shot("ls1", 20, "near", "serve"), shot("ls2", 35, "far", "backhand"),
                  shot("ls3", 51, "near", "forehand"), shot("ls4", 65, "far", "unsure")],
        "bounces": [],
        "points": [{"id": "p1", "source_start_frame": OFFSET + 18, "source_end_frame": OFFSET + 76, "server": "near",
                    "winner": "far", "score_text": "15-0", "decision": "human"},
                   {"id": "p2", "source_start_frame": OFFSET + 200, "source_end_frame": OFFSET + 250,
                    "server": "far", "winner": "unknown", "decision": "human"}],
    }
    Path(path).write_text(json.dumps(labels))
    return Path(path)


class LabelsIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.replay = make_replay(self.root)
        self.labels = write_labels(self.root / "labels.json")
        self.labels_sha = sha(self.labels)

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_labels_file_flows_through_every_tool(self):
        # 1. Validator, including the binding against the replay (hash from the source run's manifest).
        self.assertEqual(quiet(labels_module.main, [str(self.labels), "--run", str(self.replay)]), 0)

        # 2. shot_table --labels: points, servers, winners and rally indices come only from the labels.
        table = shot_table.run(self.replay, self.root / "table", self.labels)
        meta = table["labels"]
        self.assertEqual(meta["sha256"], self.labels_sha)
        self.assertTrue(meta["reader"].startswith("tennis_vision.labels"))
        self.assertTrue(meta["video_check"].startswith("matched (longrun-manifest.json of source run source-run"))
        self.assertEqual(meta["counts_in_replay"], {"shots": 4, "bounces": 0, "points": 2})
        rows = {r["shot_id"]: r for r in table["shots"]}
        self.assertEqual(sorted(rows), ["r1", "r2", "r3", "s1"])  # every label matched a candidate
        for rank, eid in enumerate(["s1", "r1", "r2", "r3"], 1):
            row = rows[eid]
            self.assertEqual((row["shot_status"], row["shot_status_source"]), ("labelled_shot", "human_confirmed"))
            self.assertEqual((row["point_id"], row["point_source"]), ("p1", "human_confirmed"))
            self.assertEqual((row["rally_shot_index"], row["rally_shot_index_source"]), (rank, "human_confirmed"))
        self.assertEqual(rows["r2"]["contact_frame"], 51)  # the labelled frame, not the candidate's 50
        self.assertEqual((rows["r1"]["shot_type"], rows["r1"]["shot_type_source"]), ("backhand", "human_confirmed"))
        self.assertEqual((rows["r3"]["shot_type"], rows["r3"]["shot_type_source"]), (None, "unknown"))  # 'unsure'
        p1, p2 = table["points"]
        self.assertEqual((p1["point_id"], p1["start_source_frame"], p1["end_source_frame"]), ("p1", OFFSET + 18, OFFSET + 76))
        self.assertEqual((p1["server"], p1["server_source"], p1["winner"], p1["winner_source"]),
                         ("near", "human_confirmed", "far", "human_confirmed"))
        self.assertEqual((p1["shot_count"], p1["shot_count_source"], p1["score_text"]), (4, "human_confirmed", "15-0"))
        self.assertEqual((p2["server"], p2["winner"], p2["winner_source"]), ("far", None, "unknown"))

        # Without labels the same replay has no points, servers, winners or rally indices.
        bare = shot_table.run(self.replay, self.root / "bare", None)
        self.assertEqual(bare["points"], [])
        for row in bare["shots"]:
            self.assertEqual((row["point_source"], row["rally_shot_index_source"]), ("unknown", "unknown"))

        # 3. Point proposals, written by the CLI and handed to the scorer unchanged.
        proposals = self.root / "proposals.json"
        quiet(points.main, ["--run", str(self.replay), "--output", str(proposals)])
        report = json.loads(proposals.read_text())
        self.assertEqual(report["run_id"], RUN_ID)
        self.assertEqual([(p["source_start_frame"], p["source_end_frame"], p["server"]) for p in report["points"]],
                         [(OFFSET + 20, OFFSET + 75, "near")])
        self.assertNotIn("TODO", report["label_scoring"])

        # 4. score_labels --points against the same labels file.
        out = self.root / "score.json"
        self.assertEqual(quiet(score_labels.main, ["--labels", str(self.labels), "--run", str(self.replay),
                                                   "--points", str(proposals), "--output", str(out)]), 0)
        score = json.loads(out.read_text())
        self.assertEqual(score["labels"]["sha256"], self.labels_sha)
        self.assertEqual(score["run"]["run_id"], RUN_ID)
        hits = score["hits"]
        self.assertEqual((hits["matched"], hits["missed"], hits["false_positives"]), (4, 0, 0))
        self.assertEqual((hits["hitter_agreement"]["agree"], hits["hitter_agreement"]["compared"]), (4, 4))
        scored_points = score["points"]
        self.assertEqual((scored_points["status"], scored_points["matched"], scored_points["missed"],
                          scored_points["false_positives"]), ("scored", 1, 1, 0))
        self.assertEqual(scored_points["details"]["matches"][0]["label"]["id"], "p1")
        self.assertEqual(scored_points["details"]["matches"][0]["proposal"]["id"], "point-proposal-1")
        self.assertEqual(scored_points["agreement"]["server"], {"compared": 1, "agree": 1,
                                                                "not_compared_label_or_proposal_unknown": 0})
        # Proposals never name a winner, so winner agreement is never compared.
        self.assertEqual(scored_points["agreement"]["winner"]["compared"], 0)

        self.assertEqual(sha(self.labels), self.labels_sha, "no tool modifies the labels file")

    def test_labels_for_another_source_video_are_refused_by_every_tool(self):
        other = write_labels(self.root / "other-video.json", video_sha="9a" * 32)
        self.assertEqual(quiet(labels_module.main, [str(other), "--run", str(self.replay)]), 1)
        with self.assertRaisesRegex(LabelsError, "bound to source video"):
            shot_table.run(self.replay, self.root / "table", other)
        self.assertFalse((self.root / "table").exists())
        with self.assertRaisesRegex(LabelsError, "bound to source video"):
            score_labels.score(other, self.replay)

    def test_shot_table_and_scorer_reject_the_same_invalid_file(self):
        bad = json.loads(self.labels.read_text())
        bad["shots"][0]["decision"] = "model"
        path = self.root / "bad.json"
        path.write_text(json.dumps(bad))
        with self.assertRaises(LabelsError) as from_table:
            shot_table.run(self.replay, self.root / "table", path)
        with self.assertRaises(LabelsError) as from_scorer:
            score_labels.score(path, self.replay)
        self.assertEqual(str(from_table.exception), str(from_scorer.exception))

    def test_video_flag_and_manifest_are_both_checked(self):
        video = self.root / "original.bin"
        video.write_bytes(b"stand-in for the original video")
        manifest_path = self.root / "source-run" / "longrun-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["input"]["sha256"] = sha(video)
        manifest_path.write_text(json.dumps(manifest))
        good = write_labels(self.root / "good.json", video_sha=sha(video))
        meta = shot_table.run(self.replay, self.root / "t1", good, video=video)["labels"]
        self.assertEqual(meta["video_check"], "matched; matched (longrun-manifest.json of source run source-run, "
                                              "same source frames)")
        wrong = self.root / "wrong.bin"
        wrong.write_bytes(b"a different file than the run was made from")
        with self.assertRaisesRegex(LabelsError, "bound to source video"):
            shot_table.run(self.replay, self.root / "t2", good, video=wrong)


if __name__ == "__main__":
    unittest.main()
