"""Calibration desk for a carried DRAFT: opens it, never marks it reviewed, never writes run folders."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tennis_vision import calibration_review
from tennis_vision.calibration_review import load_carried_draft, write_draft_desk

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "runs/claude-pipeline-test-30"
REVIEWED = ROOT / "runs/court-bounce-20260925-110335-545/calibration-corrections.json"
COURT = ROOT / "court.yaml"
RUN_ID = "new-run-" + "0" * 56


def fingerprint(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def synthetic(root, frames=30, start=4740, size=(1280, 720), **changes):
    """A replay folder + carried draft for run RUN_ID, built from the reviewed landmarks."""
    source = json.loads(REVIEWED.read_text(encoding="utf-8"))
    game = root / "game"
    replay = game / "replay"
    replay.mkdir(parents=True)
    (replay / "replay-data.json").write_text(json.dumps({"report": {
        "review_run_id": RUN_ID, "width": size[0], "height": size[1], "frames": frames, "fps": 30.0,
        "source_start_frame": start, "input_run": "C:\\runs\\new-run"}}), encoding="utf-8")
    draft = {"schema_version": 1, "kind": "court_and_bounce_review", "run_id": RUN_ID,
             "image_size": [1280, 720], "calibration_frame": 0, "calibration_status": "draft",
             "landmark_source": "carried_from_reviewed_run_x_after_camera_check",
             "landmarks": source["landmarks"], "bounce_edits": [],
             "carried_from": {"run_id": source["run_id"], "calibration_frame": 300, "status_there": "reviewed"},
             "camera_check": {"samples": [
                 {"frame": start, "time_s": start / 30, "score": 0.99, "best_shift_px": [0, 0]},
                 {"frame": start + frames - 1, "time_s": (start + frames - 1) / 30, "score": 0.97,
                  "best_shift_px": [0, 0]},
                 {"frame": start + 5000, "time_s": 0.0, "score": 0.9, "best_shift_px": [2, 0]}],
                 "summary": {"verdict": "consistent", "median_score": 0.97, "min_score": 0.9,
                             "threshold": 0.85, "failing_samples": []},
                 "meaning": "Consistency, not accuracy."}}
    draft.update(changes)
    path = game / "calibration-draft.json"
    path.write_text(json.dumps(draft), encoding="utf-8")
    return path, replay


def write_video(path, frames=30, size=(1280, 720)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    if not writer.isOpened():
        return False
    for i in range(frames):
        image = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.putText(image, str(i), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(image)
    writer.release()
    return path.is_file() and path.stat().st_size > 0


def packet_from_page(html):
    marker = '<script id="calibration-data" type="application/json">'
    return json.loads(html.split(marker, 1)[1].split("</script>", 1)[0])


class CarriedDraftDeskTest(unittest.TestCase):
    def test_committed_pipeline_draft_opens_and_nothing_is_written_to_the_run(self):
        draft_path = PIPELINE / "calibration-draft.json"
        if not (draft_path.is_file() and (PIPELINE / "replay/replay-data.json").is_file()):
            self.skipTest("pipeline test output not present")
        before = fingerprint(PIPELINE)
        reviewed_before = REVIEWED.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "desk"
            packet = write_draft_desk(draft_path, PIPELINE / "replay", out, COURT, allow_missing_video=True)
            original = json.loads(draft_path.read_text(encoding="utf-8"))
            self.assertEqual(packet["mode"], "carried_draft")
            self.assertEqual(packet["draft"], original, "the draft is shown exactly as carried")
            self.assertEqual(packet["draft"]["calibration_status"], "draft")
            self.assertEqual(packet["draft"]["bounce_edits"], [])
            self.assertEqual(packet["events"], [])
            self.assertEqual(packet["provenance"]["carried_from"]["run_id"], original["carried_from"]["run_id"])
            self.assertEqual(packet["provenance"]["review"], str((PIPELINE / "review").resolve()))
            samples = packet["camera_check"]["samples"]
            self.assertEqual([s["run_frame"] for s in samples], [0, 29])
            self.assertEqual(packet["camera_check"]["summary"]["verdict"], "consistent")
            self.assertFalse(packet["images_available"])
            self.assertTrue(all(s["image"] is None for s in samples))
            self.assertEqual(len(packet["curves"]), 9)
            self.assertEqual(len(packet["old_curves"]), 9)
            self.assertIn(original["run_id"][:16], packet["commands"]["apply"])
            self.assertIn("--corrections", packet["commands"]["apply"])
            self.assertEqual(packet_from_page((out / "calibration.html").read_text(encoding="utf-8")), packet)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["calibration-workspace.json", "calibration.html"])
        self.assertEqual(fingerprint(PIPELINE), before, "run folder must be untouched")
        self.assertEqual(REVIEWED.read_bytes(), reviewed_before, "reviewed source must be untouched")

    def test_refuses_reviewed_bounce_edits_other_run_and_other_size(self):
        cases = [({"calibration_status": "reviewed"}, "Only a draft"),
                 ({"calibration_status": "auto_confirmed"}, "Only a draft"),
                 ({"bounce_edits": [{"id": "bounce_candidate-000519", "frame": 1, "frame_range": [0, 2],
                                     "status": "confirmed", "landing_pixel": None}]}, "bounce edits"),
                 ({"run_id": "some-other-run"}, "different run"),
                 ({"image_size": [1920, 1080]}, "image size"),
                 ({"calibration_frame": 30}, "calibration frame"),
                 ({"kind": "something_else"}, "court_and_bounce_review")]
        for change, message in cases:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                draft, replay = synthetic(Path(directory), **change)
                with self.assertRaisesRegex(ValueError, message):
                    write_draft_desk(draft, replay, Path(directory) / "desk", allow_missing_video=True)
                self.assertFalse((Path(directory) / "desk").exists())

    def test_output_must_be_a_new_folder_outside_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            draft, replay = synthetic(root)
            before = fingerprint(root / "game")
            for out in (replay, replay / "desk", draft.parent, draft.parent / "desk"):
                with self.subTest(out=out), self.assertRaisesRegex(ValueError, "new folder"):
                    write_draft_desk(draft, replay, out, allow_missing_video=True)
            used = root / "used"
            used.mkdir()
            (used / "keep.txt").write_text("John's file")
            with self.assertRaisesRegex(ValueError, "already has files"):
                write_draft_desk(draft, replay, used, allow_missing_video=True)
            self.assertEqual((used / "keep.txt").read_text(), "John's file")
            with self.assertRaisesRegex(ValueError, "source.mp4 is missing"):
                write_draft_desk(draft, replay, root / "desk")
            self.assertFalse((root / "desk").exists())
            self.assertEqual(fingerprint(root / "game"), before)

    def test_frames_are_decoded_from_the_run_video_into_the_new_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            draft, replay = synthetic(root)
            video = root / "clip.mp4"
            if not write_video(video):
                self.skipTest("OpenCV cannot write a test video here")
            packet = write_draft_desk(draft, replay, root / "desk", COURT, video=video)
            self.assertTrue(packet["images_available"])
            self.assertTrue((root / "desk/calibration.png").is_file())
            samples = packet["camera_check"]["samples"]
            self.assertEqual([s["image"] for s in samples], ["frames/000000.jpg", "frames/000029.jpg", None])
            self.assertEqual(samples[2]["run_frame"], None, "a sample outside this run is not shown as one of its frames")
            for s in samples[:2]:
                self.assertTrue((root / "desk" / s["image"]).is_file())
            self.assertEqual(packet["draft"]["calibration_status"], "draft")
            self.assertEqual(packet["replay_href"], "../game/replay/replay.html")
            small = root / "small.mp4"
            if write_video(small, size=(640, 360)):
                with self.assertRaisesRegex(ValueError, "Video is 640x360"):
                    write_draft_desk(draft, replay, root / "desk2", video=small)

    def test_exported_reviewed_file_passes_the_existing_review_validator(self):
        """What the page exports (same run id, no bounce edits) is what shot_replay accepts."""
        from tennis_vision.court_refinement import validate_review
        with tempfile.TemporaryDirectory() as directory:
            draft_path, replay = synthetic(Path(directory))
            draft, report = load_carried_draft(draft_path, replay)
            exported = dict(draft, calibration_status="reviewed", landmark_source="user_adjusted_landmarks")
            self.assertIs(validate_review(exported, RUN_ID, 1280, 720, report["frames"], []), exported)
            with self.assertRaises(ValueError):
                validate_review(dict(exported, run_id=draft["carried_from"]["run_id"]), RUN_ID, 1280, 720, 30, [])

    def test_cli_needs_output_and_replay_for_a_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            draft, replay = synthetic(Path(directory))
            for argv in (["--draft", str(draft)], ["--draft", str(draft), "--output", str(Path(directory) / "d")],
                         ["--output", str(Path(directory) / "d"), "--run", "x"]):
                with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                        contextlib.redirect_stderr(io.StringIO()):
                    calibration_review.main(argv)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                calibration_review.main(["--draft", str(draft), "--replay", str(replay), "--no-frames",
                                         "--output", str(Path(directory) / "desk")])
            self.assertIn("nothing marked reviewed", printed.getvalue())
            self.assertTrue((Path(directory) / "desk/calibration.html").is_file())

    def test_template_has_draft_mode_and_keeps_the_standard_desk(self):
        template = (ROOT / "tennis_vision/calibration_review.html").read_text(encoding="utf-8")
        self.assertEqual(template.count("__CALIBRATION_DATA__"), 1)
        for element in ('id="provenance"', 'id="checkRows"', 'id="viewFrame"', 'id="bouncePanel"',
                        'id="applyDraft"', 'id="applyStandard"', 'id="checkedCourt"'):
            self.assertIn(element, template)


if __name__ == "__main__":
    unittest.main()
