"""Calibration carry-over refuses anything that would reuse old run bindings."""
import json
from pathlib import Path
import tempfile
import unittest

from tennis_vision.carry_calibration import carry

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = ROOT / "runs/court-bounce-20260925-110335-545/calibration-corrections.json"


def folders(root, run_id, start=0, frames=30, size=(1280, 720)):
    run, review = root / "run", root / "review"
    run.mkdir(); review.mkdir()
    (run / "summary.json").write_text(json.dumps({"input": "missing.mp4", "frames": frames, "fps": 30.0,
                                                  "width": size[0], "height": size[1]}))
    (run / "source-offset.json").write_text(json.dumps({"source_start_frame": start}))
    (review / "review-report.json").write_text(json.dumps({"run_id": run_id, "source_start_frame": start,
                                                           "frames": frames}))
    return run, review


class CarryTest(unittest.TestCase):
    def test_refuses_draft_source_same_run_other_size_and_foreign_review(self):
        source = json.loads(REVIEWED.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            draft = root / "draft.json"
            draft.write_text(json.dumps(dict(source, calibration_status="draft")))
            run, review = folders(root, "new-run")
            with self.assertRaisesRegex(ValueError, "reviewed"):
                carry(draft, review, run, root / "out.json")
            (review / "review-report.json").write_text(json.dumps(
                {"run_id": source["run_id"], "source_start_frame": 0, "frames": 30}))
            with self.assertRaisesRegex(ValueError, "own run"):
                carry(REVIEWED, review, run, root / "out.json")
            (review / "review-report.json").write_text(json.dumps(
                {"run_id": "new-run", "source_start_frame": 99, "frames": 30}))
            with self.assertRaisesRegex(ValueError, "does not belong"):
                carry(REVIEWED, review, run, root / "out.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, review = folders(root, "new-run", size=(1920, 1080))
            with self.assertRaisesRegex(ValueError, "Image size"):
                carry(REVIEWED, review, run, root / "out.json")
            self.assertFalse((root / "out.json").exists())

    def test_carried_file_has_no_bounce_decisions(self):
        carried_path = ROOT / "runs/claude-pipeline-test-30/calibration-draft.json"
        if not carried_path.is_file():
            self.skipTest("pipeline test output not present")
        carried = json.loads(carried_path.read_text())
        source = json.loads(REVIEWED.read_text())
        self.assertEqual(carried["calibration_status"], "draft")
        self.assertEqual(carried["bounce_edits"], [])
        self.assertNotEqual(carried["run_id"], source["run_id"])
        self.assertEqual(carried["landmarks"], source["landmarks"])


if __name__ == "__main__":
    unittest.main()
