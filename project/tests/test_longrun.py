"""Long-run chunking, resume and merge contracts, without running any model."""
import importlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tennis_vision import longrun
# Import analysis modules outside patch.dict(sys.modules): modules first imported
# inside it are evicted afterwards, and pickle would then see a different class.
from tennis_vision import ball_motion, court, detail, events, filters, scene, temporal_ball, tracking  # noqa: F401


class CountingAnalyzer:
    """Stands in for FrameAnalyzer: its rows depend on ALL prior frames.

    The running pixel sum makes any skipped, duplicated or reordered frame, or
    lost state across a resume, show up as a different row.
    """

    STATE_FIELDS = ("frames", "running")

    def __init__(self, fps=30.0):
        self.fps, self.frames, self.running = fps, 0, 0

    def state(self):
        return {"frames": self.frames, "running": self.running}

    def load_state(self, state):
        self.frames, self.running = state["frames"], state["running"]

    def process(self, frame, ball, stats):
        self.running += int(frame[0, 0, 0])
        row = {"frame": self.frames, "time_s": round(self.frames / self.fps, 4), "tracks": [],
               "events": [], "filters": {}, "detail": {"pixel": int(frame[0, 0, 0]), "running": self.running}}
        self.frames += 1
        return row, [], []


def make_video(path, frames=47, fps=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 24))
    if not writer.isOpened():
        raise unittest.SkipTest("Video encoder unavailable")
    for i in range(frames):
        writer.write(np.full((24, 32, 3), (i * 5) % 250, np.uint8))
    writer.release()


def manifest_for(video, start, end, chunk):
    info = longrun.probe(video)
    manifest = {"version": longrun.LONGRUN_VERSION,
                "input": {"path": str(video), "sha256": longrun.file_sha256(video), **info},
                "selection": {"source_start_frame": start, "source_end_frame_exclusive": end, "frames": end - start},
                "settings": {"chunk_frames": chunk, "ball_threshold": .5}, "models": {}, "code": {},
                "chunks": longrun.plan_chunks(start, end, chunk)}
    manifest["fingerprint"] = longrun.fingerprint(manifest)
    return manifest


def run(output, manifest, stop=None):
    return longrun.run_chunks(None, manifest, output, CountingAnalyzer, None,
                              progress_stream=open("nul" if sys.platform == "win32" else "/dev/null", "w"),
                              stop_after_chunks=stop)


def rows(output, manifest):
    result = []
    for chunk in manifest["chunks"]:
        path = longrun.chunk_dir(output, chunk) / "events.jsonl"
        result += [json.loads(line) for line in path.read_text().splitlines()]
    return result


class PlanTest(unittest.TestCase):
    def test_chunks_tile_selection_on_ball_block_boundaries(self):
        chunks = longrun.plan_chunks(100, 1337, 300)
        self.assertEqual(sum(c["frames"] for c in chunks), 1237)
        self.assertTrue(all(c["first_frame"] % longrun.BLOCK == 0 for c in chunks))
        self.assertEqual([c["source_first_frame"] for c in chunks], [100, 400, 700, 1000, 1300])
        with self.assertRaises(ValueError):
            longrun.plan_chunks(0, 100, 12)  # Not a multiple of the 5-frame model block.

    def test_selection_accepts_frames_or_seconds_and_rejects_bad_ranges(self):
        info = {"fps": 30.0, "frames": 5654}
        self.assertEqual(longrun.resolve_selection(info), (0, 5654))
        self.assertEqual(longrun.resolve_selection(info, start_seconds=158, end_seconds=178), (4740, 5340))
        for bad in ({"start_frame": 10, "start_seconds": 1.0}, {"start_frame": 5654}, {"end_frame": 6000},
                    {"start_frame": 50, "end_frame": 50}):
            with self.assertRaises(ValueError):
                longrun.resolve_selection(info, **bad)


class ResumeTest(unittest.TestCase):
    def test_interrupted_resumed_run_equals_one_pass_and_keeps_source_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            make_video(video)
            manifest = manifest_for(video, 7, 47, 10)  # 4 chunks, starting mid-file.
            one = root / "one"; one.mkdir()
            run(one, manifest)
            resumed = root / "resumed"; resumed.mkdir()
            session = run(resumed, manifest, stop=2)
            self.assertEqual(session["processed_chunks"], 2)
            self.assertEqual(len(longrun.completed_chunks(resumed, manifest)), 2)
            session = run(resumed, manifest)
            self.assertEqual(session["resumed_chunks"], 2)
            self.assertIn(session["positioning"][0], ("seek_verified", "sequential_verified"))
            expected = rows(one, manifest)
            self.assertEqual(rows(resumed, manifest), expected)
            self.assertEqual([r["source_frame"] for r in expected], list(range(7, 47)))
            self.assertEqual([r["frame"] for r in expected], list(range(40)))

    def test_incomplete_or_tampered_chunk_is_redone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            make_video(video, 30)
            manifest = manifest_for(video, 0, 30, 10)
            output = root / "run"; output.mkdir()
            run(output, manifest)
            reference = rows(output, manifest)
            second = longrun.chunk_dir(output, manifest["chunks"][1])
            (second / "events.jsonl").write_text("tampered\n")
            self.assertEqual(len(longrun.completed_chunks(output, manifest)), 1)
            session = run(output, manifest)
            self.assertEqual(session["processed_chunks"], 2)  # Chunk 2 and everything after it.
            self.assertEqual(rows(output, manifest), reference)

    def test_resume_refuses_when_boundary_frame_does_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            make_video(video, 30)
            manifest = manifest_for(video, 0, 30, 10)
            output = root / "run"; output.mkdir()
            run(output, manifest, stop=1)
            status_path = longrun.chunk_dir(output, manifest["chunks"][0]) / "chunk.json"
            status = json.loads(status_path.read_text())
            status["last_frame_sha256"] = "0" * 64
            status_path.write_text(json.dumps(status))
            with self.assertRaises(RuntimeError):
                run(output, manifest)


class MergeTest(unittest.TestCase):
    def test_merge_writes_standard_run_files_and_rejects_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            make_video(video, 25)
            manifest = manifest_for(video, 5, 25, 10)
            output = root / "run"; output.mkdir()
            run(output, manifest)
            summary = longrun.merge_run(output, manifest, None, None)
            self.assertEqual(summary["frames"], 20)
            offset = json.loads((output / "source-offset.json").read_text())
            self.assertEqual(offset["source_start_frame"], 5)
            merged = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual([r["frame"] for r in merged], list(range(20)))
            # Event review's loader accepts the merged folder as a normal run.
            from tennis_vision.event_review import load_run
            loaded_summary, start, loaded = load_run(output)
            self.assertEqual((start, len(loaded)), (5, 20))
            # A duplicated row must be refused rather than silently merged.
            path = longrun.chunk_dir(output, manifest["chunks"][1]) / "events.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join([lines[0]] + lines) + "\n")
            status_path = path.parent / "chunk.json"
            status = json.loads(status_path.read_text())
            status["events_sha256"] = longrun.file_sha256(path)
            status_path.write_text(json.dumps(status))
            with self.assertRaises(RuntimeError):
                longrun.merge_run(output, manifest, None, None)


class AnalyzerStateTest(unittest.TestCase):
    def test_frame_analyzer_state_round_trips_through_pickle(self):
        fake = types.ModuleType("ultralytics")
        fake.YOLO = lambda *a, **k: object()
        with patch.dict(sys.modules, {"ultralytics": fake}):
            cli = importlib.import_module("tennis_vision.cli")
        args = types.SimpleNamespace(no_auto_court=True, confidence=.15, imgsz=1280, no_detail_pass=True,
                                     court_side_margin=1.5, court_baseline_margin=6.0, all_players=True,
                                     motion_ball=False)
        from tennis_vision.tracking import Detection
        frames = [np.full((48, 64, 3), i, np.uint8) for i in range(6)]

        def detections(i):
            return [Detection("ball", (10 + 3 * i, 10, 14 + 3 * i, 14), .9)]

        def analyze(split):
            first = cli.FrameAnalyzer(args, None, 30.0, None, None)
            out = []
            with patch.object(cli, "detect", side_effect=lambda m, f, *a, **k: detections(int(f[0, 0, 0]))), \
                 patch.object(cli, "detect_pose", return_value=[]):
                for f in frames[:split]:
                    out.append(first.process(f, None, {})[0])
                second = cli.FrameAnalyzer(args, None, 30.0, None, None)
                second.load_state(pickle.loads(pickle.dumps(first.state())))
                for f in frames[split:]:
                    out.append(second.process(f, None, {})[0])
            return out

        self.assertEqual(json.dumps(analyze(3)), json.dumps(analyze(6)))


if __name__ == "__main__":
    unittest.main()
