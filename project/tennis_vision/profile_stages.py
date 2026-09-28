r"""Where does the time per frame go? A per-stage timing profile of a long run.

Runs a short stretch of a video through exactly what ``longrun.py`` does per frame
(same argument parser, same model loading, same ``FrameAnalyzer``, same five-frame
ball blocks), with timers wrapped around each stage from the outside. Analysis
code is imported and wrapped, never edited, and the timers do not change any
result. Then, as separate passes over the same frames, it times the per-frame work
that happens after the run: far-player crop pose (``player_refinement``, stage 4 of
``game_pipeline``) and the event/contact candidate passes.

    python -m tennis_vision.profile_stages --input "media/sebbie-demo-shortest (1).mp4" \
        --start-seconds 10 --frames 60 --output runs/profile-v1-gpu/stage-profile.json

Writes one JSON file (refuses to overwrite it, and refuses any existing run folder)
and prints a table: mean / median / p95 per stage, calls per frame, share of the
total, plus the backend and settings used. The first ``--warmup-frames`` frames
(one ball block by default) are processed but left out of the statistics, because
they include one-off model setup; their cost is reported separately.

Measurement only: this is not a speed-up and it does not produce a run.
"""
import argparse
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import longrun  # Sets the offline Ultralytics environment before anything imports it.
from .temporal_ball import frame_stream

SCHEMA = "stage-profile-1"
BLOCK = longrun.BLOCK

# (name, description) in table order. "Detection run" is the per-frame work of
# longrun.py, i.e. what Get-RunStatus's s/frame measures.
RUN_STAGES = (
    ("decode", "cv2.VideoCapture.read, one call per frame"),
    ("ball_model", "GridTrackNet (ONNX Runtime, temporal_ball.py), one call per 5-frame block"),
    ("detection", "YOLO detector on the full frame (players, ball, racquet)"),
    ("pose", "YOLO pose on the full frame"),
    ("detail_detection", "YOLO detector on detail-pass crops (two far-court halves, plus a recent-ball crop)"),
    ("detail_pose", "YOLO pose on the two far-court detail-pass crops"),
    ("detail_other", "detail pass outside model calls (crop, translate, merge)"),
    ("scene_selection", "SceneSelector.update: two-player identities and ball area/colour filters"),
    ("tracking_filters", "NearestTracker.update + filter_tracks (court and racquet association)"),
    ("event_engine", "EventEngine.update: per-frame hit/bounce candidates"),
    ("analyzer_other", "rest of FrameAnalyzer.process (merge, candidates, court membership, row)"),
    ("row_output", "json.dumps(row) + frame hash, as longrun does per frame (nothing is written)"),
)
# Per-frame work after the detection run (game_pipeline stages 2 and 4). Not part
# of the s/frame that Get-RunStatus reports.
AFTER_STAGES = (
    ("far_refinement_pose", "far-player crop pose (player_refinement.infer_crop), frames with one far anchor"),
    ("far_refinement_other", "far-player refinement outside the model: decode, anchor, choose_pose"),
    ("event_candidates", "event_analysis.detect_events over the profiled rows (batch, spread per frame)"),
    ("contact_candidates", "contact_detection.detect_contacts over the profiled rows (batch, spread per frame)"),
)
GROUPS = (("detection_run", RUN_STAGES), ("after_run", AFTER_STAGES))
STAGE_GROUP = {name: group for group, stages in GROUPS for name, _ in stages}
DESCRIPTIONS = {name: text for _, stages in GROUPS for name, text in stages}
# A profile must never land inside a run or review folder.
RUN_MARKERS = ("longrun-manifest.json", "events.jsonl", "summary.json", "build-status.json",
               "review-report.json", "chunks", "progress.json")


class StageTimer:
    """Exclusive (self) time per stage call, attributed to run-relative frames.

    Stages nest: time spent in an inner stage is subtracted from the outer one, so
    the per-frame stage times add up to the time actually spent. A call that
    serves several frames (a ball block, a batch pass) is spread evenly over them.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        self.calls: list[dict[str, Any]] = []
        self.frame: int | None = None
        self._stack: list[list[Any]] = []

    def active(self, name: str) -> bool:
        return any(entry[0] == name for entry in self._stack)

    def record(self, name: str, seconds: float, frames=None) -> None:
        if frames is None:
            frames = () if self.frame is None else (self.frame,)
        self.calls.append({"stage": name, "seconds": float(seconds), "frames": tuple(frames)})

    @contextmanager
    def stage(self, name: str, frames=None):
        entry = [name, 0.0]
        self._stack.append(entry)
        started = self.clock()
        try:
            yield
        finally:
            elapsed = self.clock() - started
            self._stack.pop()
            if self._stack:
                self._stack[-1][1] += elapsed
            self.record(name, elapsed - entry[1], frames)

    def wrap(self, function: Callable, name: str) -> Callable:
        def timed(*args, **kwargs):
            with self.stage(name):
                return function(*args, **kwargs)
        timed.__wrapped__ = function
        return timed


class TimedCapture:
    """cv2.VideoCapture stand-in that times each read; read k is run frame k."""

    def __init__(self, cap, timer: StageTimer, stage: str = "decode"):
        self.cap, self.timer, self.stage_name, self.reads = cap, timer, stage, 0

    def read(self):
        with self.timer.stage(self.stage_name, frames=(self.reads,)):
            ok, frame = self.cap.read()
        if ok:
            self.reads += 1
        return ok, frame

    def __getattr__(self, name):
        return getattr(self.cap, name)


class TimedBallModel:
    """Times GridTrackNet per block and spreads the block over its real frames."""

    def __init__(self, detector, timer: StageTimer, capture: TimedCapture):
        self.detector, self.timer, self.capture, self.next_frame = detector, timer, capture, 0

    def predict(self, frames):
        block = tuple(range(self.next_frame, self.capture.reads))  # Padding frames are not real frames.
        self.next_frame = self.capture.reads
        with self.timer.stage("ball_model", frames=block):
            return self.detector.predict(frames)

    def __getattr__(self, name):
        return getattr(self.detector, name)


class TimedModel:
    """YOLO stand-in: full-frame calls and detail-pass crop calls are timed apart."""

    def __init__(self, model, timer: StageTimer, stage: str, detail_stage: str):
        self.model, self.timer, self.stage_name, self.detail_stage = model, timer, stage, detail_stage

    def predict(self, *args, **kwargs):
        name = self.detail_stage if self.timer.active("detail_other") else self.stage_name
        with self.timer.stage(name):
            return self.model.predict(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


@contextmanager
def instrument(analyzer, timer: StageTimer, cli_module):
    """Wrap the models and sub-steps a FrameAnalyzer holds; restore them afterwards.

    Instance attributes and one module global (cli.filter_tracks) are replaced
    for the duration only. Nothing in the analysis files is edited.
    """
    models = {"model": analyzer.model, "pose_model": analyzer.pose_model}
    analyzer.model = TimedModel(models["model"], timer, "detection", "detail_detection")
    analyzer.pose_model = TimedModel(models["pose_model"], timer, "pose", "detail_pose")
    wrapped = [(analyzer.detail, "detail_other"), (analyzer.scene, "scene_selection"),
               (analyzer.tracker, "tracking_filters"), (analyzer.engine, "event_engine")]
    for target, name in wrapped:
        target.update = timer.wrap(target.update, name)
    original_filter = cli_module.filter_tracks
    cli_module.filter_tracks = timer.wrap(original_filter, "tracking_filters")
    try:
        yield analyzer
    finally:
        cli_module.filter_tracks = original_filter
        for target, _ in wrapped:
            vars(target).pop("update", None)
        analyzer.model, analyzer.pose_model = models["model"], models["pose_model"]


def frame_digest(frame) -> str:
    return longrun.frame_hash(frame)


def profile_detection(cap, temporal, analyzer, cli_module, timer: StageTimer, frames: int,
                      warmup: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run ``frames`` frames the way longrun.run_chunks does, timing every stage.

    Returns (rows, wall) where wall has the measured wall-clock seconds for the
    frames after the warm-up.
    """
    capture = TimedCapture(cap, timer)
    ball = TimedBallModel(temporal, timer, capture) if temporal is not None else None
    rows: list[dict[str, Any]] = []
    stream = frame_stream(capture, ball, frames)
    measured_started = timer.clock() if warmup == 0 else None
    with instrument(analyzer, timer, cli_module):
        while True:
            index = len(rows)
            if index == warmup and measured_started is None:
                measured_started = timer.clock()
            timer.frame = None  # Decode and ball blocks name their own frames.
            try:
                frame, detection, stats = next(stream)
            except StopIteration:
                break
            timer.frame = index
            with timer.stage("analyzer_other"):
                row = analyzer.process(frame, detection, stats)[0]
            with timer.stage("row_output"):
                json.dumps(row)
                frame_digest(frame)
            rows.append(row)
    timer.frame = None
    ended = timer.clock()
    if len(rows) != frames:
        raise RuntimeError(f"Video ended early: got {len(rows)}/{frames} frames")
    measured = frames - warmup
    seconds = ended - measured_started if measured_started is not None else 0.0
    return rows, {"measured_frames": measured, "measured_seconds": round(seconds, 6),
                  "seconds_per_frame": round(seconds / measured, 6) if measured else None}


def far_anchor(row: dict[str, Any]) -> dict[str, Any] | None:
    """The single observed far-player track, as refine_far_player requires."""
    matches = [p for p in row["tracks"] if p.get("label") == "player" and p.get("identity_id") == "far"]
    return matches[0] if len(matches) == 1 and not matches[0].get("predicted") else None


def profile_refinement(cap, rows, model, court, timer: StageTimer, infer=None, choose=None) -> dict[str, int]:
    """Second pass: far-player crop pose per frame, as player_refinement does it."""
    from .player_refinement import choose_pose, crop_box, infer_crop
    infer, choose = infer or infer_crop, choose or choose_pose
    counts = {"frames": 0, "inferred_frames": 0, "no_observed_identity_anchor": 0}
    for index, row in enumerate(rows):
        timer.frame = index
        with timer.stage("far_refinement_other"):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Source decode failed at frame {index} during the refinement pass")
            counts["frames"] += 1
            anchor = far_anchor(row)
            if anchor is None:
                counts["no_observed_identity_anchor"] += 1
                continue
            with timer.stage("far_refinement_pose"):
                candidates = infer(model, image, crop_box(anchor["bbox"], image.shape))
            counts["inferred_frames"] += 1
            choose(candidates, anchor, image, court)
    timer.frame = None
    return counts


def profile_event_analysis(rows, fps: float, court, timer: StageTimer) -> dict[str, Any]:
    """Batch candidate passes over the profiled rows, spread evenly per frame."""
    from .contact_detection import detect_contacts
    from .event_analysis import detect_events, observed_balls
    frames = tuple(range(len(rows)))
    result: dict[str, Any] = {}
    with timer.stage("event_candidates", frames=frames):
        result["event_candidates"] = len(detect_events(rows, observed_balls(rows), fps, court))
    with timer.stage("contact_candidates", frames=frames):
        result["contact_candidates"] = len(detect_contacts(rows, fps)["contacts"])
    return result


def distribution(values) -> dict[str, float] | None:
    if not len(values):
        return None
    data = np.asarray(values, dtype=float)
    return {"mean": round(float(data.mean()), 6), "median": round(float(np.median(data)), 6),
            "p95": round(float(np.percentile(data, 95)), 6), "max": round(float(data.max()), 6)}


def per_frame_table(calls, frames: int) -> dict[str, np.ndarray]:
    """stage -> seconds per run frame (length ``frames``)."""
    table: dict[str, np.ndarray] = {}
    for call in calls:
        valid = [f for f in call["frames"] if 0 <= f < frames]
        if not call["frames"]:
            continue
        values = table.setdefault(call["stage"], np.zeros(frames))
        share = call["seconds"] / len(call["frames"])
        for f in valid:
            values[f] += share
    return table


def summarize(timer: StageTimer, frames: int, warmup: int) -> dict[str, Any]:
    """Statistics over the measured frames [warmup, frames); warm-up reported apart."""
    table = per_frame_table(timer.calls, frames)
    measured = range(warmup, frames)
    count = len(measured)
    stages, group_totals = [], {group: np.zeros(count) for group, _ in GROUPS}
    for group, entries in GROUPS:
        for name, description in entries:
            values = table.get(name, np.zeros(frames))[warmup:frames]
            calls = [c for c in timer.calls if c["stage"] == name and any(warmup <= f < frames for f in c["frames"])]
            weight = sum(sum(warmup <= f < frames for f in c["frames"]) / len(c["frames"]) for c in calls)
            group_totals[group] += values
            stages.append({"stage": name, "group": group, "description": description,
                           "calls": len(calls), "calls_per_frame": round(weight / count, 4) if count else None,
                           "total_s": round(float(values.sum()), 6),
                           "per_frame_s": distribution(values),
                           "per_call_s": distribution([c["seconds"] for c in calls])})
    everything = sum(group_totals.values())
    for item in stages:
        group_total = float(group_totals[item["group"]].sum())
        item["share_of_group"] = round(item["total_s"] / group_total, 4) if group_total > 0 else None
        item["share_of_all"] = round(item["total_s"] / float(everything.sum()), 4) if everything.sum() > 0 else None
    warm = {name: round(float(values[:warmup].sum()), 6) for name, values in table.items() if values[:warmup].any()}
    return {"stages": stages,
            "totals": {group: {"total_s": round(float(values.sum()), 6), "per_frame_s": distribution(values)}
                       for group, values in list(group_totals.items()) + [("all", everything)]},
            "warmup": {"frames": warmup, "seconds_by_stage": warm,
                       "total_s": round(sum(warm.values()), 6),
                       "note": "First-call setup (ONNX sessions, predictor setup, DirectML compile) lands here."},
            "frames": [{"frame": f, "stages": {name: round(float(values[f]), 6) for name, values in table.items()
                                                if values[f]}} for f in measured]}


def check_output(path: Path) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("--output must be a .json file path")
    if path.exists():
        raise FileExistsError(f"{path} exists; choose a new file (profiles are never overwritten)")
    folder = path.parent
    markers = [m for m in RUN_MARKERS if (folder / m).exists()]
    if markers:
        raise ValueError(f"{folder} looks like a run/review folder ({', '.join(markers)}); "
                         "write the profile into a NEW folder, e.g. runs/profile-<name>/stage-profile.json")


def execution_providers(model) -> list[str] | None:
    """Best effort: the ONNX Runtime providers a model's session really uses."""
    candidates = [model, getattr(model, "predictor", None)]
    predictor = getattr(model, "predictor", None)
    backend = getattr(predictor, "model", None)
    candidates += [backend, getattr(backend, "backend", None), getattr(backend, "model", None)]
    for item in candidates:
        session = getattr(item, "session", None)
        if session is not None and hasattr(session, "get_providers"):
            try:
                return list(session.get_providers())
            except Exception:  # noqa: BLE001 - reporting only
                return None
    return None


def versions() -> dict[str, Any]:
    found: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(),
                             "cpu_count": os.cpu_count()}
    for module in ("numpy", "cv2", "onnxruntime", "ultralytics", "torch"):
        try:
            found[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - optional packages
            found[module] = None
    try:
        import onnxruntime
        found["onnxruntime_available_providers"] = onnxruntime.get_available_providers()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        found["torch_threads"] = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        pass
    return found


def format_table(report: dict[str, Any]) -> str:
    backend, selection = report["backend"], report["selection"]
    settings = backend["settings"]
    lines = [f"Stage profile: {selection['measured_frames']} frames from source frame "
             f"{selection['source_start_frame'] + selection['warmup_frames']} after {selection['warmup_frames']} "
             f"warm-up frames ({report['input']['path']})",
             f"Backend {settings.get('backend')} on {backend.get('device')}, imgsz {settings.get('imgsz')}, "
             f"confidence {settings.get('confidence')}, detail pass {'off' if settings.get('no_detail_pass') else 'on'}, "
             f"ball model {'GridTrackNet' if settings.get('temporal_ball') else 'off'}"]
    providers = backend.get("execution_providers") or {}
    if providers:
        lines.append("Providers: " + "; ".join(f"{k} {v or 'unknown'}" for k, v in providers.items()))
    header = f"  {'stage':<22}{'calls/frame':>12}{'mean ms':>10}{'median ms':>11}{'p95 ms':>9}{'share':>8}"
    titles = {"detection_run": "Detection run (longrun.py per frame; this is what Get-RunStatus's s/frame measures)",
              "after_run": "After the run (far-player crop pose, event/contact candidates); share of these stages only"}
    ms = lambda value: f"{1000 * value:.1f}"
    for group, _ in GROUPS:
        lines += ["", titles[group], header]
        for item in (s for s in report["stages"] if s["group"] == group):
            dist = item["per_frame_s"] or {"mean": 0, "median": 0, "p95": 0}
            share = f"{100 * item['share_of_group']:.1f}%" if item["share_of_group"] is not None else "-"
            lines.append(f"  {item['stage']:<22}{item['calls_per_frame'] or 0:>12.2f}{ms(dist['mean']):>10}"
                         f"{ms(dist['median']):>11}{ms(dist['p95']):>9}{share:>8}")
        total = report["totals"][group]["per_frame_s"] or {"mean": 0, "median": 0, "p95": 0}
        lines.append(f"  {'total':<22}{'':>12}{ms(total['mean']):>10}{ms(total['median']):>11}{ms(total['p95']):>9}")
    wall = report["wall"]
    if wall.get("seconds_per_frame") is not None:
        lines += ["", f"Detection run wall clock: {wall['seconds_per_frame']:.3f} s/frame over "
                      f"{wall['measured_frames']} frames (not in any stage: {wall['unattributed_seconds_per_frame']:.3f} s/frame)"]
    lines.append(f"Warm-up ({report['warmup']['frames']} frames, excluded above): {report['warmup']['total_s']:.1f} s")
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Source video (local footage)")
    p.add_argument("--start-seconds", type=float, default=0.0, help="Where to start in the source")
    p.add_argument("--frames", type=int, default=60, help=f"Measured frames after the warm-up (multiple of {BLOCK})")
    p.add_argument("--warmup-frames", type=int, default=BLOCK,
                   help=f"Frames processed first and left out of the statistics (multiple of {BLOCK}; default {BLOCK})")
    p.add_argument("--output", required=True, type=Path, help="NEW .json file, e.g. runs/profile-v1-gpu/stage-profile.json")
    p.add_argument("--backend", choices=("pytorch", "onnx-directml"), default="onnx-directml",
                   help="Default onnx-directml, as Run-Game.ps1 (GPU); pytorch + --device cpu mirrors Run-Game.ps1 -Cpu")
    p.add_argument("--device", help="YOLO device for --backend pytorch (e.g. cpu)")
    p.add_argument("--weights", type=Path, help="Default models/onnx-export/yolo11x-1280-dynamic.onnx (onnx-directml) or yolo11x.pt")
    p.add_argument("--pose-weights", type=Path,
                   help="Default models/onnx-export/yolo26l-pose-1280-dynamic.onnx (onnx-directml) or yolo26l-pose.pt; "
                        "also used for far-player crops, as Run-Game.ps1 does")
    p.add_argument("--ball-model", type=Path, default=Path("models/gridtracknet/gridtracknet.onnx"))
    p.add_argument("--court", type=Path, default=Path("court.yaml"))
    p.add_argument("--scene", type=Path, default=Path("sebbie-scene.json"))
    p.add_argument("--chunk-frames", type=int, default=300, help="Recorded like game_pipeline's default; not used to chunk")
    p.add_argument("--no-refinement", action="store_true", help="Skip the far-player crop pose pass")
    p.add_argument("--no-event-analysis", action="store_true", help="Skip the event/contact candidate pass")
    p.add_argument("--compare-manifest", type=Path,
                   help="Read-only: a real run's longrun-manifest.json to check settings/models/code match")
    args = p.parse_args(argv)
    onnx = args.backend == "onnx-directml"
    if args.weights is None:
        args.weights = Path("models/onnx-export/yolo11x-1280-dynamic.onnx") if onnx else Path("yolo11x.pt")
    if args.pose_weights is None:
        args.pose_weights = Path("models/onnx-export/yolo26l-pose-1280-dynamic.onnx") if onnx else Path("yolo26l-pose.pt")
    if args.frames <= 0 or args.frames % BLOCK:
        p.error(f"--frames must be a positive multiple of {BLOCK} so every ball block is full, as in real runs")
    if args.warmup_frames < 0 or args.warmup_frames % BLOCK:
        p.error(f"--warmup-frames must be a nonnegative multiple of {BLOCK}")
    if not np.isfinite(args.start_seconds) or args.start_seconds < 0:
        p.error("--start-seconds must be finite and nonnegative")
    return args


def check_inputs(args) -> None:
    """Fail early and clearly: footage and weights exist only on John's laptop."""
    required = [("--input", args.input), ("--weights", args.weights), ("--pose-weights", args.pose_weights),
                ("--ball-model", args.ball_model), ("--court", args.court), ("--scene", args.scene)]
    missing = [f"{label} {path}" for label, path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing local file(s): " + "; ".join(missing) + ". Footage and weights are local-only and never "
            "downloaded: run this from project/ on the analysis PC (ONNX exports come from "
            "python -m tennis_vision.export_onnx yolo11x.pt yolo26l-pose.pt).")
    check_output(args.output)


def longrun_args(args) -> argparse.Namespace:
    """The exact namespace longrun.py would build for this selection (via its own parser)."""
    argv = ["--input", str(args.input), "--output", str(args.output.parent), "--weights", str(args.weights),
            "--pose-weights", str(args.pose_weights), "--court", str(args.court), "--scene", str(args.scene),
            "--temporal-ball", "--ball-model", str(args.ball_model), "--chunk-frames", str(args.chunk_frames),
            "--backend", args.backend]
    if args.device:
        argv += ["--device", args.device]
    return longrun.parse_args(argv)


def import_analysis():
    import importlib
    try:
        cli = importlib.import_module(f"{__package__}.cli")  # Imports ultralytics.
    except ModuleNotFoundError as error:
        raise RuntimeError(f"Python package {error.name!r} is missing. Run with project/.venv's python "
                           "(see requirements-lock-windows.txt); nothing is installed automatically.") from error
    return cli


def compare_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    real = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"manifest": str(path),
            **{key: real.get(key) == manifest.get(key) for key in ("settings", "models", "code", "device")},
            "note": "True means identical to that run's record. chunk_frames is part of settings, so pass "
                    "--chunk-frames with that run's value; code false means analysis code changed since."}


def run(args, clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    check_inputs(args)
    largs = longrun_args(args)
    info = longrun.probe(args.input)
    start, _ = longrun.resolve_selection(info, start_seconds=args.start_seconds)
    total = args.warmup_frames + args.frames
    end = start + total
    if end > info["frames"]:
        raise ValueError(f"Selection runs past the end of the video ({end} > {info['frames']} frames)")
    print("[profile] hashing input and models (same record as longrun-manifest.json)...", file=sys.stderr, flush=True)
    manifest = longrun.build_manifest(largs, info, start, end)
    cli = import_analysis()
    court = cli.load_court(largs.court)
    scene_config = json.loads(largs.scene.read_text(encoding="utf-8"))
    setup: dict[str, float] = {}
    started = clock()
    try:
        from .temporal_ball import GridTrackNetDetector
        temporal = GridTrackNetDetector(largs.ball_model, largs.ball_threshold)
    except ModuleNotFoundError as error:
        raise RuntimeError(f"Python package {error.name!r} is missing for the ball model; use project/.venv's python.") from error
    setup["ball_model_load_s"] = round(clock() - started, 3)
    started = clock()
    model, pose_model = longrun.load_models(largs)  # Same loading (and DirectML patch) as real runs.
    setup["yolo_load_s"] = round(clock() - started, 3)
    analyzer = cli.FrameAnalyzer(largs, court, info["fps"], model, pose_model, None, scene_config, True)
    timer = StageTimer(clock)
    cap, positioning = longrun.open_at(args.input, start, None)
    print(f"[profile] {total} frames from source frame {start} ({args.warmup_frames} warm-up)...", file=sys.stderr, flush=True)
    try:
        rows, wall = profile_detection(cap, temporal, analyzer, cli, timer, total, args.warmup_frames)
    finally:
        cap.release()
    providers = {"detector": execution_providers(model), "pose": execution_providers(pose_model),
                 "ball_model": list(temporal.session.get_providers())}
    extra: dict[str, Any] = {}
    if not args.no_refinement:
        # Stage 4 loads its own pose model (shot_replay.build); mirror its thread cap.
        started = clock()
        try:
            import torch
            torch.set_num_threads(min(4, torch.get_num_threads()))
        except ImportError:
            pass
        YOLO = cli.YOLO  # Same class shot_replay.build uses (ultralytics.YOLO).
        if str(largs.pose_weights).endswith(".onnx"):
            longrun.enable_directml([largs.pose_weights])
            refine_model = YOLO(str(largs.pose_weights), task="pose")
        else:
            refine_model = YOLO(str(largs.pose_weights))
        setup["refinement_model_load_s"] = round(clock() - started, 3)
        cap, _ = longrun.open_at(args.input, start, None)
        print("[profile] far-player crop pose pass...", file=sys.stderr, flush=True)
        try:
            extra["far_refinement"] = profile_refinement(cap, rows, refine_model, analyzer.court, timer)
        finally:
            cap.release()
        providers["far_refinement_pose"] = execution_providers(refine_model)
    if not args.no_event_analysis:
        try:
            extra["event_analysis"] = profile_event_analysis(rows, info["fps"], analyzer.court, timer)
        except Exception as error:  # noqa: BLE001 - keep the model timings even if a batch pass fails
            extra["event_analysis"] = {"error": f"{type(error).__name__}: {error}"}
    backend = {"settings": manifest["settings"], "device": manifest["device"], "models": manifest["models"],
               "code": manifest["code"], "execution_providers": providers, "setup": setup, "versions": versions()}
    if args.compare_manifest:
        backend["compare_manifest"] = compare_manifest(args.compare_manifest, manifest)
    report = build_report(timer, total, args.warmup_frames, wall, backend,
                          {"path": manifest["input"]["path"], "sha256": manifest["input"]["sha256"],
                           "fps": info["fps"], "width": info["width"], "height": info["height"]},
                          {"source_start_frame": start, "source_end_frame_exclusive": end,
                           "warmup_frames": args.warmup_frames, "measured_frames": args.frames,
                           "positioning": positioning}, extra)
    report["command"] = [sys.executable, "-m", "tennis_vision.profile_stages", *sys.argv[1:]]
    return report


def build_report(timer, frames, warmup, wall, backend, source, selection, extra=None) -> dict[str, Any]:
    summary = summarize(timer, frames, warmup)
    run_total = summary["totals"]["detection_run"]["total_s"]
    measured = frames - warmup
    wall = dict(wall)
    if wall.get("measured_seconds") is not None and measured:
        wall["unattributed_seconds_per_frame"] = round((wall["measured_seconds"] - run_total) / measured, 6)
    for item in summary["frames"]:
        item["source_frame"] = selection["source_start_frame"] + item["frame"]
    return {"schema": SCHEMA, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input": source, "selection": selection, "backend": backend, "wall": wall, **summary,
            "passes": extra or {},
            "notes": [
                "Timings only. Wrappers around the analysis code measure it; they do not change any detection.",
                "Detection-run stages are the per-frame work of longrun.py; the after-run stages run in "
                "game_pipeline stages 2 and 4 and are not part of Get-RunStatus's s/frame.",
                "Per-frame values: a ball block's time is spread evenly over its five frames; the batch "
                "event/contact passes are spread over all profiled frames.",
                "Not profiled: event_review video rendering (FFmpeg) and scene-cut decoding, "
                "recover_initial_players, shot classification and replay building.",
                "Numbers depend on this PC's load; close other GPU/CPU-heavy programs while profiling."]}


def main(argv=None) -> None:
    args = parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    check_output(args.output)  # Re-check just before writing.
    longrun.save_json(args.output, report)
    print(format_table(report))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
