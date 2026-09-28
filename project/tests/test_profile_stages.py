"""Per-stage timing profile (issue #26) with fake models and a fake clock.

No weights, footage, ultralytics or onnxruntime are needed: the real FrameAnalyzer
runs with stand-in YOLO models that advance a fake clock by known amounts.
"""
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import yaml

from tennis_vision import longrun, profile_stages as ps
# Import analysis modules outside patch.dict(sys.modules) (see test_longrun.py): modules
# first imported inside it are evicted afterwards and would load twice.
from tennis_vision import (ball_motion, camera3d, detail, events, filters, flight3d, scene,  # noqa: F401
                           setup_temporal, temporal_ball)
from tennis_vision.court import CourtMapper
from tennis_vision.tracking import Detection

PROJECT = Path(__file__).resolve().parent.parent


def import_cli():
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda *a, **k: object()
    with patch.dict(sys.modules, {"ultralytics": fake}):
        return importlib.import_module("tennis_vision.cli")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeResult:
    names = {0: "person", 32: "sports ball", 38: "tennis racket"}
    boxes = keypoints = None

    def __init__(self, pose=False):
        if not pose:
            self.boxes = []


class FakeYolo:
    """Full-frame calls cost `full` seconds, crop calls `crop` seconds; counts both."""

    def __init__(self, clock, full, crop, shape, pose=False):
        self.clock, self.full, self.crop, self.shape, self.pose = clock, full, crop, shape, pose
        self.full_calls = self.crop_calls = 0

    def predict(self, image, **kwargs):
        if self.clock is not None:
            if image.shape == self.shape:
                self.full_calls += 1
                self.clock.advance(self.full)
            else:
                self.crop_calls += 1
                self.clock.advance(self.crop)
        return [FakeResult(self.pose)]


class FakeCapture:
    def __init__(self, frames, clock=None, cost=0.0):
        self.frames, self.clock, self.cost, self.index = frames, clock, cost, 0

    def read(self):
        if self.clock is not None:
            self.clock.advance(self.cost)
        if self.index >= len(self.frames):
            return False, None
        self.index += 1
        return True, self.frames[self.index - 1].copy()

    def release(self):
        pass


class FakeBall:
    """Every frame: one GridTrackNet ball on the near court, moving right."""

    def __init__(self, clock=None, cost=0.0):
        self.clock, self.cost, self.calls = clock, cost, 0

    def predict(self, frames):
        self.calls += 1
        if self.clock is not None:
            self.clock.advance(self.cost)
        out = []
        for i in range(5):
            x = 600 + 10 * (5 * (self.calls - 1) + i)
            out.append((Detection("ball", (x - 4, 500, x + 4, 508), .9, source="gridtracknet"), {}))
        return out


def manual_court():
    corners = yaml.safe_load((PROJECT / "court.yaml").read_text(encoding="utf-8"))["image_corners"]
    return CourtMapper(np.asarray(corners, np.float32), source="manual", confidence=1.0)


def analyzer_args():
    # The exact namespace longrun.py builds (its own parser), as the profiler uses.
    return longrun.parse_args(["--input", "v.mp4", "--output", "out", "--weights", "w.onnx", "--pose-weights",
                               "p.onnx", "--temporal-ball", "--backend", "onnx-directml", "--chunk-frames", "300"])


def frames(count, shape=(720, 1280, 3)):
    rng = np.random.default_rng(7)
    return [rng.integers(0, 255, shape, dtype=np.uint8) for _ in range(count)]


class StageTimerTest(unittest.TestCase):
    def test_nested_stages_record_exclusive_time_per_frame(self):
        clock = FakeClock()
        timer = ps.StageTimer(clock)
        timer.frame = 3
        with timer.stage("outer"):
            clock.advance(1)
            with timer.stage("inner"):
                clock.advance(2)
            clock.advance(.5)
        self.assertEqual(timer.calls, [{"stage": "inner", "seconds": 2.0, "frames": (3,)},
                                       {"stage": "outer", "seconds": 1.5, "frames": (3,)}])
        self.assertTrue(callable(timer.wrap(len, "x")))

    def test_block_calls_spread_evenly_and_warmup_is_excluded(self):
        timer = ps.StageTimer(FakeClock())
        timer.record("ball_model", 1.0, frames=range(0, 5))
        timer.record("ball_model", 2.0, frames=range(5, 10))
        for f in range(10):
            timer.record("detection", .1 * (f + 1), frames=(f,))
        summary = ps.summarize(timer, 10, 5)
        stages = {s["stage"]: s for s in summary["stages"]}
        ball, det = stages["ball_model"], stages["detection"]
        self.assertEqual(ball["calls"], 1)
        self.assertAlmostEqual(ball["calls_per_frame"], .2)
        self.assertEqual(ball["per_call_s"]["mean"], 2.0)
        self.assertAlmostEqual(ball["per_frame_s"]["mean"], .4)
        self.assertAlmostEqual(det["per_frame_s"]["median"], .8)  # frames 5..9: .6 .7 .8 .9 1.0
        self.assertAlmostEqual(det["per_frame_s"]["p95"], .98)
        self.assertAlmostEqual(det["total_s"], 4.0)
        self.assertAlmostEqual(det["share_of_group"], 4.0 / 6.0, places=4)
        self.assertAlmostEqual(summary["totals"]["detection_run"]["per_frame_s"]["mean"], 1.2)
        self.assertEqual(stages["pose"]["calls"], 0)
        self.assertIsNone(stages["pose"]["per_call_s"])
        # Warm-up cost is reported apart, never mixed into the statistics.
        self.assertAlmostEqual(summary["warmup"]["seconds_by_stage"]["ball_model"], 1.0)
        self.assertAlmostEqual(summary["warmup"]["seconds_by_stage"]["detection"], 1.5)
        self.assertEqual([f["frame"] for f in summary["frames"]], [5, 6, 7, 8, 9])


class ProfileDetectionTest(unittest.TestCase):
    def setUp(self):
        self.cli = import_cli()
        self.shape = (720, 1280, 3)
        self.images = frames(15, self.shape)
        self.court = manual_court()
        self.scene = json.loads((PROJECT / "sebbie-scene.json").read_text(encoding="utf-8"))

    def analyzer(self, clock=None):
        model = FakeYolo(clock, .5, .2, self.shape)
        pose = FakeYolo(clock, .4, .3, self.shape, pose=True)
        return self.cli.FrameAnalyzer(analyzer_args(), self.court, 30.0, model, pose, None, self.scene, True), model, pose

    def test_stages_add_up_and_results_are_unchanged(self):
        clock = FakeClock()
        analyzer, model, pose = self.analyzer(clock)
        ball = FakeBall(clock, 1.0)
        timer = ps.StageTimer(clock)
        original_filter = self.cli.filter_tracks
        rows, wall = ps.profile_detection(FakeCapture(self.images, clock, .01), ball, analyzer, self.cli, timer, 15, 5)
        # Instrumentation is removed afterwards.
        self.assertIs(self.cli.filter_tracks, original_filter)
        self.assertIs(analyzer.model, model)
        self.assertNotIn("update", vars(analyzer.detail))
        # The same frames through an un-instrumented analyzer give identical rows.
        plain, _, _ = self.analyzer()
        expected = [plain.process(frame, det, stats)[0] for frame, det, stats in
                    self.cli.frame_stream(FakeCapture(self.images), FakeBall(), 15)]
        self.assertEqual(json.dumps(rows), json.dumps(expected))

        summary = ps.summarize(timer, 15, 5)
        stages = {s["stage"]: s for s in summary["stages"]}
        self.assertEqual(model.full_calls, 15)
        self.assertEqual(pose.full_calls, 15)
        self.assertGreaterEqual(model.crop_calls, 30)  # two far halves per frame (+ ball crops)
        self.assertEqual(pose.crop_calls, 30)
        self.assertAlmostEqual(stages["detection"]["per_frame_s"]["mean"], .5)
        self.assertAlmostEqual(stages["pose"]["per_frame_s"]["mean"], .4)
        self.assertAlmostEqual(stages["detail_pose"]["per_frame_s"]["mean"], .6)
        self.assertAlmostEqual(stages["detail_pose"]["calls_per_frame"], 2.0)
        self.assertAlmostEqual(stages["ball_model"]["per_call_s"]["mean"], 1.0)
        self.assertAlmostEqual(stages["ball_model"]["per_frame_s"]["mean"], .2)
        self.assertAlmostEqual(stages["decode"]["per_frame_s"]["mean"], .01)
        self.assertEqual(stages["decode"]["calls_per_frame"], 1.0)
        # Detail crops: timer and fake model agree on the total.
        detail_total = sum(c["seconds"] for c in timer.calls if c["stage"] == "detail_detection")
        self.assertAlmostEqual(detail_total, .2 * model.crop_calls)
        # Only the fake models and decoder advance the clock; everything is attributed.
        run_total = summary["totals"]["detection_run"]["total_s"]
        self.assertAlmostEqual(wall["measured_seconds"], run_total)
        self.assertAlmostEqual(wall["seconds_per_frame"], run_total / 10)
        for name in ("scene_selection", "tracking_filters", "event_engine", "analyzer_other", "row_output"):
            self.assertAlmostEqual(stages[name]["total_s"], 0.0)
            self.assertGreater(stages[name]["calls"], 0, name)

    def test_short_video_is_an_error(self):
        analyzer, _, _ = self.analyzer()
        with self.assertRaisesRegex(RuntimeError, "ended early"):
            ps.profile_detection(FakeCapture(self.images[:7]), FakeBall(), analyzer, self.cli,
                                 ps.StageTimer(FakeClock()), 10, 5)


class AfterRunPassesTest(unittest.TestCase):
    def test_refinement_times_only_frames_with_one_far_anchor(self):
        clock = FakeClock()
        timer = ps.StageTimer(clock)
        far = {"label": "player", "identity_id": "far", "bbox": [500, 100, 540, 200], "predicted": False}
        rows = [{"tracks": [far]}, {"tracks": []}, {"tracks": [dict(far, predicted=True)]}, {"tracks": [far, far]}]
        seen = []

        def infer(model, image, region):
            seen.append(region)
            clock.advance(.25)
            return []

        counts = ps.profile_refinement(FakeCapture(frames(4, (720, 1280, 3)), clock, .01), rows, None, None, timer,
                                       infer=infer, choose=lambda *a: (None, "none"))
        self.assertEqual(counts, {"frames": 4, "inferred_frames": 1, "no_observed_identity_anchor": 3})
        self.assertEqual(len(seen), 1)
        pose = [c for c in timer.calls if c["stage"] == "far_refinement_pose"]
        self.assertEqual(pose, [{"stage": "far_refinement_pose", "seconds": .25, "frames": (0,)}])
        other = [c for c in timer.calls if c["stage"] == "far_refinement_other"]
        self.assertEqual(len(other), 4)
        self.assertAlmostEqual(sum(c["seconds"] for c in other), .04)

    def test_event_analysis_is_spread_over_all_frames(self):
        timer = ps.StageTimer(FakeClock())
        rows = [{"frame": f, "tracks": []} for f in range(10)]
        result = ps.profile_event_analysis(rows, 30.0, None, timer)
        self.assertEqual(result, {"event_candidates": 0, "contact_candidates": 0})
        calls = {c["stage"]: c for c in timer.calls}
        self.assertEqual(calls["event_candidates"]["frames"], tuple(range(10)))


class ReportTest(unittest.TestCase):
    def report(self):
        timer = ps.StageTimer(FakeClock())
        for f in range(10):
            timer.record("decode", .01, frames=(f,))
            timer.record("detection", 2.0, frames=(f,))
            timer.record("far_refinement_pose", .5, frames=(f,))
        backend = {"settings": {"backend": "onnx-directml", "imgsz": 1280, "confidence": .15,
                                "no_detail_pass": False, "temporal_ball": True},
                   "device": "directml", "execution_providers": {"detector": ["DmlExecutionProvider"], "pose": None}}
        return ps.build_report(timer, 10, 5, {"measured_frames": 5, "measured_seconds": 10.1, "seconds_per_frame": 2.02},
                               backend, {"path": "video.mp4"},
                               {"source_start_frame": 300, "source_end_frame_exclusive": 310,
                                "warmup_frames": 5, "measured_frames": 5})

    def test_report_is_json_and_table_names_every_stage(self):
        report = self.report()
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["schema"], "stage-profile-1")
        self.assertAlmostEqual(report["wall"]["unattributed_seconds_per_frame"], .01)
        self.assertEqual(report["frames"][0]["source_frame"], 305)
        stages = {s["stage"]: s for s in report["stages"]}
        self.assertAlmostEqual(stages["detection"]["share_of_group"], 2.0 / 2.01, places=4)
        self.assertAlmostEqual(stages["far_refinement_pose"]["share_of_group"], 1.0)
        self.assertAlmostEqual(stages["far_refinement_pose"]["share_of_all"], .5 / 2.51, places=4)
        table = ps.format_table(report)
        for name, _ in ps.RUN_STAGES + ps.AFTER_STAGES:
            self.assertIn(name, table)
        self.assertIn("Backend onnx-directml on directml, imgsz 1280", table)
        self.assertIn("2.020 s/frame", table)
        self.assertIn("detector ['DmlExecutionProvider']; pose unknown", table)

    def test_every_listed_stage_is_described_once(self):
        names = [n for n, _ in ps.RUN_STAGES + ps.AFTER_STAGES]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(ps.DESCRIPTIONS[n] for n in names))


class EndToEndTest(unittest.TestCase):
    """main() on a small synthetic video, with fake models in place of the weights."""

    def test_main_writes_profile_and_table_without_touching_other_files(self):
        import cv2
        cli = import_cli()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
            if not writer.isOpened():
                self.skipTest("MP4 encoder unavailable")
            for image in frames(20):
                writer.write(image)
            writer.release()
            models = root / "models"
            models.mkdir()
            paths = {name: models / name for name in ("det.onnx", "pose.onnx", "ball.onnx")}
            for path in paths.values():
                path.write_bytes(b"placeholder, not a model")
            shape = (720, 1280, 3)
            detector, pose = FakeYolo(None, 0, 0, shape), FakeYolo(None, 0, 0, shape, pose=True)
            refine = FakeYolo(None, 0, 0, shape, pose=True)

            class Ball(FakeBall):
                session = types.SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

            loaded = []

            def load_models(largs):
                loaded.append(largs)
                return detector, pose

            output = root / "profile-test" / "stage-profile.json"
            argv = ["--input", str(video), "--start-seconds", ".1", "--frames", "10", "--output", str(output),
                    "--weights", str(paths["det.onnx"]), "--pose-weights", str(paths["pose.onnx"]),
                    "--ball-model", str(paths["ball.onnx"]), "--court", str(PROJECT / "court.yaml"),
                    "--scene", str(PROJECT / "sebbie-scene.json")]
            before = sorted(p.name for p in root.rglob("*"))
            with patch.object(ps, "import_analysis", return_value=cli), \
                 patch.object(temporal_ball, "GridTrackNetDetector", return_value=Ball()), \
                 patch.object(longrun, "load_models", side_effect=load_models), \
                 patch.object(longrun, "enable_directml"), \
                 patch.object(cli, "YOLO", return_value=refine), \
                 patch.object(sys, "argv", ["profile_stages", *argv]), \
                 redirect_stdout(io.StringIO()) as printed, patch("sys.stderr", io.StringIO()):
                ps.main(argv)
            report = json.loads(output.read_text(encoding="utf-8"))
            # Only the profile file (and its new folder) was created.
            after = sorted(p.name for p in root.rglob("*"))
            self.assertEqual(sorted(set(after) - set(before)), ["profile-test", "stage-profile.json"])
        self.assertEqual(loaded[0].backend, "onnx-directml")
        self.assertTrue(loaded[0].temporal_ball)
        self.assertEqual(report["selection"], {"source_start_frame": 3, "source_end_frame_exclusive": 18,
                                               "warmup_frames": 5, "measured_frames": 10, "positioning": "seek"})
        self.assertEqual(report["backend"]["device"], "directml")
        self.assertEqual(report["backend"]["settings"]["imgsz"], 1280)
        self.assertEqual(report["backend"]["execution_providers"]["ball_model"], ["CPUExecutionProvider"])
        self.assertEqual(set(report["backend"]["code"]), set(longrun.CODE_FILES))
        self.assertEqual(detector.full_calls + pose.full_calls, 0)  # clock=None: fakes do not count
        self.assertEqual([f["source_frame"] for f in report["frames"]], list(range(8, 18)))
        self.assertIn("far_refinement", report["passes"])
        self.assertEqual(report["passes"]["far_refinement"]["frames"], 15)
        self.assertIn("event_candidates", report["passes"]["event_analysis"])
        self.assertIn("Detection run wall clock", printed.getvalue())
        self.assertIn("stage-profile.json", printed.getvalue())


class CommandLineTest(unittest.TestCase):
    def test_defaults_mirror_run_game_gpu(self):
        args = ps.parse_args(["--input", "v.mp4", "--output", "runs/p/x.json"])
        self.assertEqual(args.backend, "onnx-directml")
        self.assertEqual(args.weights, Path("models/onnx-export/yolo11x-1280-dynamic.onnx"))
        self.assertEqual(args.pose_weights, Path("models/onnx-export/yolo26l-pose-1280-dynamic.onnx"))
        self.assertEqual((args.frames, args.warmup_frames, args.chunk_frames), (60, 5, 300))
        largs = ps.longrun_args(args)
        self.assertTrue(largs.temporal_ball)
        self.assertEqual((largs.imgsz, largs.confidence, largs.backend, largs.no_detail_pass), (1280, .15, "onnx-directml", False))
        self.assertEqual(largs.court, Path("court.yaml"))
        self.assertEqual(largs.scene, Path("sebbie-scene.json"))
        cpu = ps.parse_args(["--input", "v.mp4", "--output", "x.json", "--backend", "pytorch", "--device", "cpu"])
        self.assertEqual((cpu.weights, cpu.pose_weights), (Path("yolo11x.pt"), Path("yolo26l-pose.pt")))
        self.assertEqual(ps.longrun_args(cpu).device, "cpu")

    def test_rejects_partial_blocks(self):
        for argv in (["--frames", "62"], ["--warmup-frames", "3"], ["--frames", "0"]):
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
                ps.parse_args(["--input", "v.mp4", "--output", "x.json", *argv])

    def test_missing_weights_fail_clearly_before_loading_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "v.mp4"
            video.write_bytes(b"not a video")
            args = ps.parse_args(["--input", str(video), "--output", str(root / "p" / "profile.json"),
                                  "--court", str(PROJECT / "court.yaml"), "--scene", str(PROJECT / "sebbie-scene.json")])
            with patch.object(ps, "import_analysis", side_effect=AssertionError("must not import models")):
                with self.assertRaisesRegex(FileNotFoundError, r"--weights .*local-only"):
                    ps.run(args)
            self.assertFalse((root / "p").exists())

    def test_never_writes_into_a_run_folder_or_over_a_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run").mkdir()
            (root / "run" / "longrun-manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "NEW folder"):
                ps.check_output(root / "run" / "stage-profile.json")
            (root / "p").mkdir()
            (root / "p" / "stage-profile.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                ps.check_output(root / "p" / "stage-profile.json")
            with self.assertRaisesRegex(ValueError, ".json"):
                ps.check_output(root / "p" / "profile.txt")
            ps.check_output(root / "q" / "stage-profile.json")  # A new folder is fine.

    def test_missing_packages_give_a_clear_message(self):
        with patch.dict(sys.modules, {"ultralytics": None}):
            sys.modules.pop("tennis_vision.cli", None)  # Restored by patch.dict.
            with self.assertRaisesRegex(RuntimeError, "'ultralytics' is missing"):
                ps.import_analysis()

    def test_compare_manifest_reports_matches(self):
        real = PROJECT / "runs" / "claude-game-v1-full" / "longrun-manifest.json"
        manifest = json.loads(real.read_text(encoding="utf-8"))
        result = ps.compare_manifest(real, manifest)
        self.assertTrue(all(result[k] for k in ("settings", "models", "code", "device")))
        changed = dict(manifest, settings=dict(manifest["settings"], imgsz=640))
        self.assertFalse(ps.compare_manifest(real, changed)["settings"])

    def test_providers_are_found_on_the_backend_session(self):
        session = types.SimpleNamespace(get_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
        model = types.SimpleNamespace(predictor=types.SimpleNamespace(model=types.SimpleNamespace(session=session)))
        self.assertEqual(ps.execution_providers(model), ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertIsNone(ps.execution_providers(object()))


if __name__ == "__main__":
    unittest.main()
