"""Long-run chunking, resume and merge contracts, without running any model."""
from datetime import datetime
import importlib
import io
import json
from pathlib import Path, PureWindowsPath
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


VIDEO1_MANIFEST = Path(__file__).resolve().parent.parent / "runs" / "claude-game-v1-full" / "longrun-manifest.json"
# chunk.json keys written before wall-clock timing existed (resume must keep accepting these).
LEGACY_CHUNK_KEYS = {"status", "index", "fingerprint", "first_frame", "frames", "source_first_frame",
                     "source_last_frame", "last_frame_sha256", "events_sha256", "state_sha256", "seconds"}
TIMING_CHUNK_KEYS = {"started_at", "finished_at", "pause_threshold_s", "pauses", "paused_seconds"}


class WindowsPath(PureWindowsPath):
    """The video-1 manifest was built on Windows; resolve() must keep that path as-is."""

    def resolve(self):
        return self


class FakeClock:
    """Wall clock that only moves when the test says so."""

    def __init__(self, start=1_790_000_000.0):
        self.now = start

    def __call__(self):
        return self.now


class SleepyAnalyzer(CountingAnalyzer):
    """Each frame takes 1 s of wall time; selected frames also 'sleep' first."""

    clock, sleeps = None, {}

    def process(self, frame, ball, stats):
        self.clock.now += 1.0 + self.sleeps.get(self.frames, 0.0)
        return super().process(frame, ball, stats)


def parse_local_iso(text):
    value = datetime.fromisoformat(text)
    if value.utcoffset() is None:
        raise AssertionError(f"{text} has no UTC offset")
    return value


class ResumeCompatibilityTest(unittest.TestCase):
    """John's video-1 run (runs/claude-game-v1-full) must still resume after this change."""

    def load(self):
        return json.loads(VIDEO1_MANIFEST.read_text(encoding="utf-8"))

    def test_video1_manifest_fingerprint_rebuilds_from_the_same_arguments(self):
        stored = self.load()
        self.assertEqual(longrun.LONGRUN_VERSION, stored["version"])
        self.assertEqual(longrun.fingerprint(stored), stored["fingerprint"])
        # Orchestration code (this file) is deliberately not fingerprinted.
        self.assertEqual(longrun.CODE_FILES, tuple(stored["code"]))
        self.assertNotIn("longrun.py", longrun.CODE_FILES)
        # Rebuild through build_manifest with the arguments that run used. Input and
        # weights are not in the repo, so their hashes come from the manifest; the
        # analysis-code hashes are checked separately below.
        hashes = {stored["input"]["path"]: stored["input"]["sha256"],
                  **{m["path"]: m["sha256"] for m in stored["models"].values()}}
        with tempfile.TemporaryDirectory() as directory:
            court_path, scene_path = Path(directory) / "court.yaml", Path(directory) / "scene.json"
            court_path.write_bytes(stored["settings"]["court"].encode("utf-8"))
            scene_path.write_text(json.dumps(stored["settings"]["scene"]), encoding="utf-8")
            settings = stored["settings"]
            args = types.SimpleNamespace(
                input=WindowsPath(stored["input"]["path"]), court=court_path, scene=scene_path,
                weights=PureWindowsPath(stored["models"]["weights"]["path"]),
                pose_weights=PureWindowsPath(stored["models"]["pose_weights"]["path"]),
                ball_model=PureWindowsPath(stored["models"]["ball_model"]["path"]), device=None,
                **{k: settings[k] for k in ("confidence", "imgsz", "court_side_margin", "court_baseline_margin",
                                            "no_detail_pass", "all_players", "no_auto_court", "temporal_ball",
                                            "ball_threshold", "chunk_frames", "backend")})
            info = {k: stored["input"][k] for k in ("fps", "frames", "width", "height")}
            selection = stored["selection"]
            with patch.object(longrun, "file_sha256", side_effect=lambda p: hashes[str(p)]), \
                 patch.object(longrun, "code_hashes", return_value=dict(stored["code"])):
                built = longrun.build_manifest(args, info, selection["source_start_frame"],
                                               selection["source_end_frame_exclusive"])
        self.assertEqual(built["fingerprint"], stored["fingerprint"])
        self.assertEqual(built, stored)  # Same chunk plan and device too.

    def test_current_analysis_code_still_matches_video1_manifest(self):
        stored = self.load()
        current = longrun.code_hashes()
        changed = sorted(name for name in stored["code"] if current[name] != stored["code"][name])
        if changed:
            # Not this change's contract: analysis code may legitimately move on in dev,
            # which (by design) makes that run refuse to resume from this checkout.
            self.skipTest(f"analysis code changed since video-1 started: {changed}")
        self.assertEqual(current, stored["code"])

    def test_chunks_without_timing_fields_still_count_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            make_video(video, 40)
            manifest = manifest_for(video, 0, 40, 10)
            one = root / "one"; one.mkdir()
            run(one, manifest)
            output = root / "run"; output.mkdir()
            run(output, manifest, stop=2)
            for chunk in manifest["chunks"][:2]:
                path = longrun.chunk_dir(output, chunk) / "chunk.json"
                status = json.loads(path.read_text())
                self.assertEqual(set(status), LEGACY_CHUNK_KEYS | TIMING_CHUNK_KEYS)
                # Rewrite exactly as the pre-timing code did.
                longrun.save_json(path, {k: v for k, v in status.items() if k in LEGACY_CHUNK_KEYS})
            self.assertEqual(len(longrun.completed_chunks(output, manifest)), 2)
            session = run(output, manifest)
            self.assertEqual((session["resumed_chunks"], session["processed_chunks"]), (2, 2))
            self.assertEqual(rows(output, manifest), rows(one, manifest))
            timing = longrun.timing_report(output, manifest)
            self.assertEqual([c["timing_recorded"] for c in timing["chunks"]], [False, False, True, True])
            self.assertEqual(timing["chunks_without_timing"], 2)
            self.assertIsNone(timing["chunks"][0]["paused_seconds"])  # Unknown, not zero.
            self.assertIsNone(timing["chunks"][0]["started_at"])
            self.assertIsNotNone(timing["chunks"][0]["seconds"])


class TimingTest(unittest.TestCase):
    def run_sleepy(self, output, manifest, clock, sleeps, stop=None):
        SleepyAnalyzer.clock, SleepyAnalyzer.sleeps = clock, sleeps
        with patch.object(longrun, "wall_clock", clock):
            return longrun.run_chunks(None, manifest, output, SleepyAnalyzer, None,
                                      progress_stream=io.StringIO(), stop_after_chunks=stop)

    def test_chunks_record_wall_clock_and_pauses_without_changing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            make_video(video, 30)
            manifest = manifest_for(video, 0, 30, 10)
            plain = root / "plain"; plain.mkdir()
            run(plain, manifest)
            output = root / "run"; output.mkdir()
            clock = FakeClock()
            # A 3 h sleep before frame 13, and a 59 s stall (below threshold) before frame 25.
            self.run_sleepy(output, manifest, clock, {13: 3 * 3600.0, 25: 59.0 - 1.0})
            self.assertEqual(rows(output, manifest), rows(plain, manifest))
            statuses = [json.loads((longrun.chunk_dir(output, c) / "chunk.json").read_text())
                        for c in manifest["chunks"]]
            for status in statuses:
                started, finished = parse_local_iso(status["started_at"]), parse_local_iso(status["finished_at"])
                self.assertLessEqual(started, finished)
                self.assertEqual(status["pause_threshold_s"], longrun.PAUSE_THRESHOLD_S)
            self.assertEqual([s["pauses"] for s in statuses[::2]], [[], []])
            (pause,) = statuses[1]["pauses"]
            self.assertEqual((pause["frame"], pause["source_frame"]), (13, 13))
            self.assertEqual(pause["seconds"], 3 * 3600.0 + 1.0)
            self.assertEqual(statuses[1]["paused_seconds"], pause["seconds"])
            gap = parse_local_iso(pause["end"]) - parse_local_iso(pause["start"])
            self.assertAlmostEqual(gap.total_seconds(), pause["seconds"], delta=1.0)
            chunk_span = parse_local_iso(statuses[1]["finished_at"]) - parse_local_iso(statuses[1]["started_at"])
            self.assertGreaterEqual(chunk_span.total_seconds(), pause["seconds"])
            self.assertEqual(statuses[2]["paused_seconds"], 0.0)  # The 59 s stall is not a pause.

            progress = json.loads((output / "progress.json").read_text())
            parse_local_iso(progress["updated_at"])
            parse_local_iso(progress["session_started_at"])
            self.assertEqual(progress["session_paused_s"], pause["seconds"])
            self.assertEqual(len(progress["session_pauses"]), 1)
            # 30 frames of ~1 s (plus the 58 s stall); the 3 h pause is excluded from the rate.
            self.assertAlmostEqual(progress["session_active_s"], 30 + 58.0, delta=1.0)
            self.assertGreater(progress["measured_frames_per_s"], .3)

            timing = longrun.timing_report(output, manifest)
            self.assertEqual(timing["chunks_without_timing"], 0)
            self.assertEqual(timing["paused_seconds_total"], pause["seconds"])
            self.assertEqual(timing["pauses"], [dict(pause, chunk=1)])
            self.assertEqual([c["started_at"] for c in timing["chunks"]], [s["started_at"] for s in statuses])

    def test_resume_start_up_is_not_a_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            make_video(video, 20)
            manifest = manifest_for(video, 0, 20, 10)
            output = root / "run"; output.mkdir()
            clock = FakeClock()
            self.run_sleepy(output, manifest, clock, {}, stop=1)
            clock.now += 8 * 3600.0  # Overnight between the two sessions.
            real_open_at = longrun.open_at

            def slow_open_at(*args):
                clock.now += 600.0  # Sequential decoding to the resume point of a long file.
                return real_open_at(*args)

            with patch.object(longrun, "open_at", side_effect=slow_open_at):
                self.run_sleepy(output, manifest, clock, {})
            timing = longrun.timing_report(output, manifest)
            self.assertEqual(timing["pauses"], [])
            first, second = timing["chunks"]
            gap = parse_local_iso(second["started_at"]) - parse_local_iso(first["finished_at"])
            self.assertGreaterEqual(gap.total_seconds(), 8 * 3600.0)

    def test_eta_excludes_pauses(self):
        clock = FakeClock()
        with patch.object(longrun, "wall_clock", clock):
            progress = longrun.Progress(total=100, done=0, stream=io.StringIO())
            for frame in range(10):
                clock.now += 2.0
                self.assertIsNone(progress.update(frame=frame))
            clock.now += 2 * 3600.0
            pause = progress.update(frame=10)
            self.assertEqual(pause["seconds"], 2 * 3600.0)
            self.assertEqual(pause["frame"], 10)
            snapshot = progress.snapshot()
        self.assertEqual(snapshot["session_paused_s"], 7200.0)
        self.assertEqual(snapshot["session_elapsed_s"], 7220.0)
        self.assertEqual(snapshot["session_active_s"], 20.0)
        # 11 frames in 20 active seconds; 89 left.
        self.assertAlmostEqual(snapshot["eta_s"], 89 / (11 / 20.0), delta=.1)
        self.assertIn("pauses excluded", snapshot["eta_basis"])

    def test_iso_time_is_local_with_offset(self):
        text = longrun.iso_time(1_790_000_000.0)
        value = parse_local_iso(text)
        self.assertEqual(value.timestamp(), 1_790_000_000.0)
        self.assertRegex(text, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$")


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
