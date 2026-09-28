"""Ground-truth labels.json schema v1: validation, run binding, read-only loading."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from tennis_vision.labels import (SHOT_TYPES, LabelsError, check_binding, load_labels, main, validate_labels)

RUN_ID = "a" * 64
VIDEO = "b" * 64


def make_labels(**overrides):
    labels = {
        "schema_version": 1, "kind": "tennis_ground_truth_labels",
        "binding": {"run_id": RUN_ID, "source_video_sha256": VIDEO, "fps": 30.0, "source_video_name": "match.mp4"},
        "labeller": {"name": "John", "date": "2026-09-28"},
        "shot_types": list(SHOT_TYPES),
        "coverage": [{"source_start_frame": 100, "source_end_frame": 400, "kinds": ["shots", "bounces", "points"]}],
        "shots": [{"id": "s1", "source_frame": 120, "hitter": "near", "shot_type": "serve", "decision": "human",
                   "notes": "first serve"},
                  {"id": "s2", "source_frame": 160, "hitter": "far", "shot_type": "unsure", "decision": "human"}],
        "bounces": [{"id": "b1", "source_frame": 140, "call": "in", "decision": "human"}],
        "points": [{"id": "p1", "source_start_frame": 110, "source_end_frame": 300, "server": "near",
                    "winner": "unknown", "score_text": "15-0", "decision": "human"}],
    }
    labels.update(overrides)
    return labels


class ValidateTest(unittest.TestCase):
    def assertRejected(self, labels, fragment):
        with self.assertRaises(LabelsError) as caught:
            validate_labels(labels)
        self.assertIn(fragment, str(caught.exception))

    def test_valid_file_passes_and_is_not_modified(self):
        labels = make_labels()
        before = copy.deepcopy(labels)
        self.assertIs(validate_labels(labels), labels)
        self.assertEqual(labels, before)

    def test_empty_lists_are_valid(self):
        validate_labels(make_labels(coverage=[], shots=[], bounces=[], points=[]))

    def test_rejects_wrong_header(self):
        self.assertRejected(make_labels(schema_version=2), "schema_version")
        self.assertRejected(make_labels(kind="labels"), "kind")
        labels = make_labels()
        del labels["coverage"]
        self.assertRejected(labels, "missing field(s) coverage")

    def test_rejects_unknown_fields_everywhere(self):
        self.assertRejected(make_labels(spin={}), "unknown field(s) spin")
        labels = make_labels()
        labels["shots"][0]["spin_rpm"] = 2000
        self.assertRejected(labels, "unknown field(s) spin_rpm")
        labels = make_labels()
        labels["binding"]["video"] = "x"
        self.assertRejected(labels, "binding: unknown")

    def test_binding_hashes_and_fps(self):
        labels = make_labels()
        labels["binding"]["run_id"] = "A" * 64
        self.assertRejected(labels, "binding.run_id")
        labels = make_labels()
        labels["binding"]["source_video_sha256"] = "b" * 63
        self.assertRejected(labels, "binding.source_video_sha256")
        labels = make_labels()
        labels["binding"]["fps"] = 0
        self.assertRejected(labels, "binding.fps")

    def test_labeller_name_and_date(self):
        self.assertRejected(make_labels(labeller={"name": " ", "date": "2026-09-28"}), "labeller.name")
        self.assertRejected(make_labels(labeller={"name": "John", "date": "28/09/2026"}), "labeller.date")
        self.assertRejected(make_labels(labeller={"name": "John", "date": "2026-02-30"}), "labeller.date")

    def test_shot_type_list_must_be_current(self):
        self.assertRejected(make_labels(shot_types=["serve", "forehand"]), "shot_types")
        labels = make_labels()
        labels["shots"][0]["shot_type"] = "drop shot"
        self.assertRejected(labels, "shot_type")

    def test_every_entry_is_a_human_decision(self):
        for kind in ("shots", "bounces", "points"):
            labels = make_labels()
            labels[kind][0]["decision"] = "pipeline"
            self.assertRejected(labels, "must be 'human'")

    def test_entry_values(self):
        labels = make_labels()
        labels["shots"][0]["hitter"] = "unknown"
        self.assertRejected(labels, "hitter")
        labels = make_labels()
        labels["bounces"][0]["call"] = "let"
        self.assertRejected(labels, "call")
        labels = make_labels()
        labels["points"][0]["winner"] = "server"
        self.assertRejected(labels, "winner")
        for bad in (-1, 1.5, True, "120"):
            labels = make_labels()
            labels["shots"][0]["source_frame"] = bad
            self.assertRejected(labels, "source_frame")

    def test_duplicates_and_overlaps(self):
        labels = make_labels()
        labels["bounces"][0]["id"] = "s1"
        self.assertRejected(labels, "duplicate id")
        labels = make_labels()
        labels["shots"][1]["source_frame"] = 120
        self.assertRejected(labels, "already labelled at source frame 120")
        labels = make_labels()
        labels["points"].append(dict(labels["points"][0], id="p2", source_start_frame=300, source_end_frame=350))
        self.assertRejected(labels, "overlaps point")
        labels = make_labels()
        labels["points"][0]["source_end_frame"] = 100
        self.assertRejected(labels, "before source_start_frame")

    def test_coverage_spans(self):
        span = {"source_start_frame": 350, "source_end_frame": 500, "kinds": ["shots"]}
        labels = make_labels()
        labels["coverage"].append(span)
        self.assertRejected(labels, "overlaps another shots coverage span")
        labels = make_labels()
        labels["coverage"].append(dict(span, kinds=[]))
        self.assertRejected(labels, "kinds")
        labels = make_labels()
        labels["coverage"].append(dict(span, kinds=["serves"]))
        self.assertRejected(labels, "kinds")
        labels = make_labels()
        labels["coverage"].append(dict(span, source_start_frame=600))
        self.assertRejected(labels, "before source_start_frame")
        labels = make_labels()  # Non-overlapping spans and different kinds may share frames.
        labels["coverage"].append({"source_start_frame": 401, "source_end_frame": 500, "kinds": ["shots"]})
        labels["coverage"].append({"source_start_frame": 401, "source_end_frame": 450, "kinds": ["bounces"]})
        validate_labels(labels)


class BindingTest(unittest.TestCase):
    def test_matching_binding_passes(self):
        check_binding(make_labels(), RUN_ID, VIDEO, 30.0)

    def test_rejects_other_run_video_or_fps(self):
        labels = make_labels()
        with self.assertRaisesRegex(LabelsError, "bound to run"):
            check_binding(labels, "c" * 64, VIDEO, 30.0)
        with self.assertRaisesRegex(LabelsError, "bound to source video"):
            check_binding(labels, RUN_ID, "c" * 64, 30.0)
        with self.assertRaisesRegex(LabelsError, "fps"):
            check_binding(labels, RUN_ID, VIDEO, 25.0)


class LoadTest(unittest.TestCase):
    def test_load_is_read_only_and_hashes_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps(make_labels(), indent=1), encoding="utf-8")
            before = path.read_bytes()
            mtime = path.stat().st_mtime_ns
            labels, digest = load_labels(path)
            self.assertEqual(labels["labeller"]["name"], "John")
            self.assertEqual(len(digest), 64)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mtime_ns, mtime)

    def test_load_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(LabelsError, "not valid"):
                load_labels(path)

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            good, bad = Path(directory) / "good.json", Path(directory) / "bad.json"
            good.write_text(json.dumps(make_labels()), encoding="utf-8")
            bad.write_text(json.dumps(make_labels(shot_types=[])), encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(main([str(good)]), 0)
                self.assertEqual(main([str(bad)]), 1)
            self.assertIn("valid schema v1 labels by John", out.getvalue())
            self.assertIn("INVALID: shot_types", err.getvalue())


if __name__ == "__main__":
    unittest.main()
