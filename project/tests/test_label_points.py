"""label_points: builds the offline labelling page; its export validates with labels.py."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from tennis_vision import labels as labels_module
from tennis_vision.label_points import build, main, video_link
from tennis_vision.labels import SHOT_TYPES, load_labels

RUN_ID = "a" * 64
VIDEO = "b" * 64
START = 1000
FRAMES = 300
FPS = 30.0
UI_TEST = Path(__file__).with_name("test_label_points_ui.cjs")


def dump(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def write_longrun(folder):
    folder.mkdir()
    dump(folder / "longrun-manifest.json", {
        "fingerprint": "c" * 64, "input": {"path": "C:\\media\\match.mp4", "sha256": VIDEO, "fps": FPS},
        "selection": {"source_start_frame": START, "source_end_frame_exclusive": START + FRAMES, "frames": FRAMES}})
    dump(folder / "source-offset.json", {"source_start_frame": START})
    dump(folder / "shots.json", {"contacts": [], "bounces": [], "shots": []})


def write_video(path, frames=FRAMES, fps=FPS):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 36))
    if not writer.isOpened():
        return False
    for i in range(frames):
        writer.write(np.full((36, 64, 3), i % 256, np.uint8))
    writer.release()
    return path.is_file() and path.stat().st_size > 0


def write_replay(folder, input_run, video_frames=FRAMES):
    folder.mkdir()
    dump(folder / "build-status.json", {"status": "complete"})
    dump(folder / "replay-report.json", {"review_run_id": RUN_ID, "source_start_frame": START, "frames": FRAMES,
                                         "fps": FPS, "input_run": str(input_run)})
    dump(folder / "reviewed-events.json", {"run_id": RUN_ID, "events": [
        {"id": "hit-050", "type": "hit", "status": "unreviewed", "frame": 50, "player_id": "near"},
        {"id": "bounce-060", "type": "bounce", "status": "unreviewed", "frame": 60, "player_id": None},
        {"id": "manual-1", "type": "hit", "status": "confirmed", "frame": 70, "player_id": "far"},
        {"id": "win-80", "type": "hit", "status": "unreviewed", "frame": 80, "player_id": "far",
         "contact_support": "review_window_only"},
        {"id": "hit-</script>", "type": "hit", "status": "unreviewed", "frame": 90, "player_id": "far"},
    ]})
    dump(folder / "shot-candidates.json", {"shots": [
        {"event_id": "hit-050", "frame": 50, "classification": "forehand", "classification_status": "candidate",
         "action_state": "shot_candidate"}]})
    dump(folder / "replay-data.json", {"frames": [{"frame": i} for i in range(FRAMES)]})
    if video_frames is not None and not write_video(folder / "source.mp4", video_frames):
        raise unittest.SkipTest("OpenCV cannot write a test video here")


def snapshot(folder):
    return {str(p.relative_to(folder)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*"))}


class LabelPointsBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_longrun(self.root / "longrun")
        self.replay = self.root / "replay"

    def tearDown(self):
        self.tmp.cleanup()

    def test_builds_page_without_touching_the_replay(self):
        write_replay(self.replay, self.root / "longrun")
        before = snapshot(self.replay)
        out = self.root / "labelling"
        data = build(self.replay, out)
        self.assertEqual(snapshot(self.replay), before, "the replay folder must stay byte-identical")
        self.assertEqual((data["run_id"], data["source_video_sha256"], data["fps"]), (RUN_ID, VIDEO, FPS))
        self.assertEqual((data["source_start_frame"], data["frames"]), (START, FRAMES))
        self.assertEqual(data["shot_types"], list(SHOT_TYPES))
        self.assertEqual(data["video_check"]["status"], "aligned")
        self.assertEqual(data["video_src"], "../replay/source.mp4")
        # Candidates are hints in SOURCE frames; human-added events and review windows are not candidates.
        self.assertEqual([(c["id"], c["kind"], c["source_frame"]) for c in data["candidates"]],
                         [("hit-050", "hit", 1050), ("bounce-060", "bounce", 1060), ("hit-</script>", "hit", 1090)])
        self.assertEqual(data["candidates"][0]["shot_type"], "forehand")
        page = (out / "label.html").read_text(encoding="utf-8")
        self.assertNotIn("__LABEL_DATA__", page)
        self.assertEqual(page.count("</script>"), 2, "embedded data cannot close its script tag")
        self.assertEqual(json.loads((out / "label-data.json").read_text(encoding="utf-8")), data)
        copied = sorted(p.name for p in (out / "run-data").iterdir())
        self.assertEqual(copied, ["build-status.json", "replay-report.json", "reviewed-events.json",
                                  "shot-candidates.json"])
        self.assertFalse((out / "source.mp4").exists(), "the video is linked, not copied")
        self.assertFalse(any(out.rglob("labels*.json")), "building never writes a labels file")

    def test_refuses_existing_output_or_output_inside_the_run(self):
        write_replay(self.replay, self.root / "longrun")
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            build(self.replay, existing)
        with self.assertRaisesRegex(ValueError, "inside the run folder"):
            build(self.replay, self.replay / "labels-page")
        self.assertFalse((self.replay / "labels-page").exists())
        with self.assertRaisesRegex(ValueError, "long-run folder"):
            build(self.root / "longrun", self.root / "from-longrun")

    def test_refuses_a_video_that_is_not_frame_aligned(self):
        write_replay(self.replay, self.root / "longrun", video_frames=FRAMES - 10)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            build(self.replay, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_missing_video_still_builds_and_page_asks_for_it(self):
        write_replay(self.replay, self.root / "longrun", video_frames=None)
        data = build(self.replay, self.root / "out")
        self.assertEqual(data["video_check"]["status"], "missing")

    def test_run_without_recorded_hash_needs_the_source_video(self):
        write_replay(self.replay, "C:\\gone\\legacy-run", video_frames=None)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            build(self.replay, self.root / "out")
        original = self.root / "match.mp4"
        original.write_bytes(b"synthetic original video bytes")
        data = build(self.replay, self.root / "out", video=original)
        self.assertEqual(data["source_video_sha256"], hashlib.sha256(original.read_bytes()).hexdigest())

    def test_cli(self):
        write_replay(self.replay, self.root / "longrun", video_frames=None)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["--replay", str(self.replay), "--output", str(self.root / "cli")]), 0)
        self.assertIn("hints only", out.getvalue())
        self.assertTrue((self.root / "cli" / "label.html").is_file())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["--replay", str(self.replay), "--output", str(self.root / "cli")]), 1)
        self.assertIn("already exists", out.getvalue())

    def test_video_link_is_relative_and_url_safe(self):
        video = self.root / "a run" / "source.mp4"
        self.assertEqual(video_link(video, self.root / "pages" / "x"), "../../a%20run/source.mp4")

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed to drive the page's export")
    def test_page_export_validates_against_the_run(self):
        write_replay(self.replay, self.root / "longrun", video_frames=None)
        out = self.root / "labelling"
        build(self.replay, out)
        exported = self.root / "exported-labels.json"
        result = subprocess.run(["node", str(UI_TEST), str(out / "label.html"), str(exported)],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        labels, _ = load_labels(exported)
        self.assertEqual(labels["binding"]["run_id"], RUN_ID)
        self.assertEqual(labels["binding"]["source_video_sha256"], VIDEO)
        self.assertTrue(labels["shots"] and labels["bounces"] and labels["points"] and labels["coverage"])
        self.assertTrue(all(START <= s["source_frame"] < START + FRAMES for s in labels["shots"]))
        with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()) as err:
            code = labels_module.main([str(exported), "--run", str(self.replay)])
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("binding matches run", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
