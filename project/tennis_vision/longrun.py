"""Resumable, chunked analysis of long footage (a full game or a whole file).

Produces the same run-folder contract as ``python -m tennis_vision`` (events.jsonl,
summary.json, source-offset.json, shots.json, flight3d.json, ball-model.json), so
event review and the replay builder work unchanged. Differences:

* Work is split into chunks whose length is a multiple of the five-frame ball
  model block. Each completed chunk stores its rows plus a checkpoint of all
  cross-frame analyzer state, so an interrupted run resumes at the next chunk
  and produces the same rows as one uninterrupted pass.
* A fingerprint (input bytes, selection, settings, model hashes, code version)
  guards resume: a changed input or setting refuses to mix results.
* Resume verifies the decoder landed on the right source frame by comparing the
  hash of the last decoded frame before the boundary with the stored one.
* Memory is bounded: frames stream through; rows go straight to disk.
* Rows keep run-relative ``frame``/``time_s`` (the existing contract) and add
  ``source_frame`` so every observation maps back to the original file.
* Wall-clock timing is recorded next to the results, never inside rows or the
  fingerprint: ``chunk.json`` gets ``started_at``/``finished_at`` and any pause
  (wall-clock gap of more than PAUSE_THRESHOLD_S between consecutive frames,
  e.g. laptop sleep or a stalled decoder), ``progress.json`` gets ``updated_at``,
  and ``longrun-report.json`` lists every recorded pause. ETA excludes pauses.
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Read by Ultralytics at import time: never download weights or pip-install
# packages mid-run (it does not recognise onnxruntime-directml as onnxruntime).
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent.parent / ".inference-config"))

import cv2
import numpy as np

LONGRUN_VERSION = 1
BLOCK = 5  # GridTrackNet consumes non-overlapping five-frame blocks from the run start.
# Analysis code whose changes would alter rows. Orchestration changes in this file
# that alter chunk semantics must bump LONGRUN_VERSION instead.
CODE_FILES = ("cli.py", "tracking.py", "filters.py", "detail.py", "scene.py", "events.py",
              "court.py", "ball_motion.py", "temporal_ball.py")
# A wall-clock gap between two consecutive processed frames longer than this is
# recorded as a pause (sleep/hibernate, a stalled decoder or GPU, a paused console).
# Normal frames take about 5 s (GPU) to 15 s (CPU).
PAUSE_THRESHOLD_S = 60.0
# Wall clock (seconds since the epoch). Sleep always shows up here, whereas some
# monotonic clocks stop while the machine is suspended. Tests replace it.
wall_clock = time.time


def iso_time(timestamp: float) -> str:
    """ISO 8601 in local time with its UTC offset, e.g. 2026-09-27T01:02:03+01:00."""
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def save_json(path: Path, data: Any) -> None:
    """Write atomically so an interruption never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def probe(video: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {video}")
    try:
        return {"fps": float(cap.get(cv2.CAP_PROP_FPS) or 0), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def resolve_selection(info: dict[str, Any], start_frame=None, end_frame=None, start_seconds=None, end_seconds=None):
    """Return [start, end) in source frames. Seconds round to the nearest frame."""
    fps, total = info["fps"], info["frames"]
    if not np.isfinite(fps) or fps <= 0 or total <= 0:
        raise ValueError("Video reports no usable frame rate/frame count")
    if start_frame is not None and start_seconds is not None or end_frame is not None and end_seconds is not None:
        raise ValueError("Give each boundary as frames or seconds, not both")
    start = start_frame if start_frame is not None else int(round((start_seconds or 0.0) * fps))
    end = end_frame if end_frame is not None else (int(round(end_seconds * fps)) if end_seconds is not None else total)
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= total:
        raise ValueError(f"Selection [{start}, {end}) is outside the video's 0..{total} frames")
    return start, end


def plan_chunks(start: int, end: int, chunk_frames: int) -> list[dict[str, int]]:
    """Chunks tile [start, end) exactly; every boundary is block-aligned to the run start."""
    if chunk_frames <= 0 or chunk_frames % BLOCK:
        raise ValueError(f"--chunk-frames must be a positive multiple of {BLOCK} (ball-model block size)")
    chunks, relative, total = [], 0, end - start
    while relative < total:
        count = min(chunk_frames, total - relative)
        chunks.append({"index": len(chunks), "first_frame": relative, "frames": count,
                       "source_first_frame": start + relative})
        relative += count
    return chunks


def chunk_dir(output: Path, chunk: dict[str, int]) -> Path:
    return output / "chunks" / f"{chunk['first_frame']:07d}-{chunk['first_frame'] + chunk['frames']:07d}"


def fingerprint(manifest: dict[str, Any]) -> str:
    keys = ("version", "input", "selection", "settings", "models", "code")
    return hashlib.sha256(json.dumps({k: manifest[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: file_sha256(root / name) for name in CODE_FILES}


def build_manifest(args, info, start, end) -> dict[str, Any]:
    models = {"weights": {"path": str(args.weights), "sha256": file_sha256(args.weights)},
              "pose_weights": {"path": str(args.pose_weights), "sha256": file_sha256(args.pose_weights)}}
    if args.temporal_ball:
        models["ball_model"] = {"path": str(args.ball_model), "sha256": file_sha256(args.ball_model)}
    settings = {k: getattr(args, k) for k in ("confidence", "imgsz", "court_side_margin", "court_baseline_margin",
                                             "no_detail_pass", "all_players", "no_auto_court", "temporal_ball",
                                             "ball_threshold", "chunk_frames", "backend")}
    settings["court"] = args.court.read_text(encoding="utf-8") if args.court else None
    settings["scene"] = json.loads(args.scene.read_text(encoding="utf-8")) if args.scene else None
    manifest = {"version": LONGRUN_VERSION,
                "input": {"path": str(args.input.resolve()), "sha256": file_sha256(args.input), **info},
                "selection": {"source_start_frame": start, "source_end_frame_exclusive": end, "frames": end - start},
                "settings": settings, "models": models, "code": code_hashes(),
                # Device is recorded but not fingerprinted: GPU/CPU numerics can differ
                # slightly, so resume refuses a device switch separately (see below).
                "device": "directml" if args.backend == "onnx-directml" else (args.device or "auto"),
                "chunks": plan_chunks(start, end, args.chunk_frames)}
    manifest["fingerprint"] = fingerprint(manifest)
    return manifest


def completed_chunks(output: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Leading run of chunks whose files are complete and unmodified."""
    done = []
    for chunk in manifest["chunks"]:
        folder = chunk_dir(output, chunk)
        status_path = folder / "chunk.json"
        if not status_path.is_file():
            break
        status = json.loads(status_path.read_text(encoding="utf-8"))
        rows_path, state_path = folder / "events.jsonl", folder / "state.pkl"
        if (status.get("status") != "complete" or status.get("fingerprint") != manifest["fingerprint"]
                or not rows_path.is_file() or not state_path.is_file()
                or file_sha256(rows_path) != status.get("events_sha256")
                or file_sha256(state_path) != status.get("state_sha256")):
            break
        done.append(status)
    return done


def open_at(video: Path, source_frame: int, expected_previous_hash: str | None):
    """Open the video positioned so the next read() returns ``source_frame``.

    When resuming, seek one frame earlier and check that frame's hash against the
    one stored when that chunk finished. Falls back to sequential decoding if a
    container seek is inaccurate, so the chunk boundary never drifts.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {video}")
    if source_frame == 0:
        return cap, "start"
    if expected_previous_hash is None:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame) or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - source_frame) > .5:
            cap.release()
            raise RuntimeError("Could not seek to the requested start frame")
        return cap, "seek"
    if cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame - 1):
        ok, frame = cap.read()
        if ok and frame_hash(frame) == expected_previous_hash:
            return cap, "seek_verified"
    cap.release()
    cap = cv2.VideoCapture(str(video))
    for index in range(source_frame):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Video ended at frame {index} while positioning for resume")
    if frame_hash(frame) != expected_previous_hash:
        cap.release()
        raise RuntimeError("Decoded frame before the resume point does not match the stored hash; the input changed")
    return cap, "sequential_verified"


class KeepAwake:
    """While processing, ask Windows not to sleep (like a video player does).

    Uses SetThreadExecutionState for this process only; no power settings are
    changed and the request ends when the process exits. No-op elsewhere.
    """

    def __enter__(self):
        self.active = False
        if sys.platform == "win32":
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            self.active = bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
        return self

    def __exit__(self, *exc):
        if self.active:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        return False


class Progress:
    """Frame progress with elapsed time and ETA measured in this session only.

    All times come from ``wall_clock``. A gap longer than PAUSE_THRESHOLD_S
    between consecutive frames is recorded as a pause and left out of the
    measured rate, so one night of sleep does not inflate the ETA.
    """

    def __init__(self, total: int, done: int, stream=sys.stderr, every_s: float = 10.0):
        self.total, self.done, self.stream, self.every_s = total, done, stream, every_s
        self.session_start_done, self.started = done, wall_clock()
        self.last_print = 0.0
        self.last_frame_at = self.started
        self.pauses: list[dict[str, Any]] = []
        self.paused_s = 0.0

    def mark(self) -> None:
        """Restart the between-frame gap measurement (e.g. after positioning the decoder)."""
        self.last_frame_at = wall_clock()

    def active_elapsed(self, now: float | None = None) -> float:
        return max(0.0, (wall_clock() if now is None else now) - self.started - self.paused_s)

    def rate(self) -> float | None:
        elapsed = self.active_elapsed()
        processed = self.done - self.session_start_done
        return processed / elapsed if processed and elapsed > 0 else None

    def snapshot(self) -> dict[str, Any]:
        rate = self.rate()
        now = wall_clock()
        return {"frames_done": self.done, "frames_total": self.total,
                "percent": round(100 * self.done / self.total, 2),
                "session_elapsed_s": round(now - self.started, 1),
                "session_active_s": round(self.active_elapsed(now), 1),
                "session_paused_s": round(self.paused_s, 1),
                "measured_frames_per_s": round(rate, 3) if rate else None,
                "eta_s": round((self.total - self.done) / rate, 1) if rate else None,
                "eta_basis": "frames processed in this session only; resumed frames and pauses excluded"}

    def update(self, count: int = 1, force: bool = False, frame: int | None = None) -> dict[str, Any] | None:
        """Count processed frames. Returns the pause that ended with them, if any.

        ``frame`` is the run-relative index of the frame just processed; it is
        stored with a pause so the gap can be placed in the run.
        """
        self.done += count
        now = wall_clock()
        gap, pause = now - self.last_frame_at, None
        if gap > PAUSE_THRESHOLD_S:
            pause = {"start": iso_time(self.last_frame_at), "end": iso_time(now), "seconds": round(gap, 1),
                     "frame": frame}
            self.pauses.append(pause)
            self.paused_s += gap
            print(f"[longrun] pause: {gap / 60:.1f} min wall-clock gap before frame {frame} "
                  f"({pause['start']} to {pause['end']}); excluded from ETA", file=self.stream, flush=True)
        self.last_frame_at = now
        if force or now - self.last_print >= self.every_s:
            self.last_print = now
            s = self.snapshot()
            eta = f"{s['eta_s'] / 60:.1f} min" if s["eta_s"] is not None else "measuring"
            rate = f"{s['measured_frames_per_s']:.2f} fps" if s["measured_frames_per_s"] else "-"
            print(f"[longrun] {s['frames_done']}/{s['frames_total']} frames ({s['percent']:.1f}%), "
                  f"{s['session_elapsed_s'] / 60:.1f} min elapsed, {rate}, ETA {eta}", file=self.stream, flush=True)
        return pause


def enable_directml(model_paths) -> None:
    """Run the given exported YOLO ONNX files on the GPU through DirectML.

    Ultralytics only selects CUDA or CPU for ONNX, so sessions for exactly these
    files get the DirectML provider. Every other session (GridTrackNet) keeps the
    CPU provider the baseline used.
    """
    import onnxruntime as ort
    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("DirectML is unavailable; install onnxruntime-directml (same version as onnxruntime)")
    targets = {str(Path(p).resolve()).lower() for p in model_paths}
    original = ort.InferenceSession
    if getattr(original, "_tennis_directml", False):
        return

    class DirectMLSession(original):
        _tennis_directml = True

        def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
            if isinstance(path_or_bytes, (str, os.PathLike)) and str(Path(path_or_bytes).resolve()).lower() in targets:
                sess_options = sess_options or ort.SessionOptions()
                # DirectML requires these settings (ONNX Runtime documentation).
                sess_options.enable_mem_pattern = False
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                providers, provider_options = ["DmlExecutionProvider", "CPUExecutionProvider"], None
            super().__init__(path_or_bytes, sess_options, providers, provider_options, **kwargs)

    ort.InferenceSession = DirectMLSession


def load_models(args):
    # Keep Ultralytics settings inside the project and never download weights.
    config_dir = Path(__file__).resolve().parent.parent / ".inference-config"
    config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    if args.backend == "onnx-directml":
        if not all(str(p).endswith(".onnx") for p in (args.weights, args.pose_weights)):
            raise ValueError("--backend onnx-directml needs exported .onnx weights (python -m tennis_vision.export_onnx)")
        enable_directml((args.weights, args.pose_weights))
        args.device = None  # Device selection happens through the ONNX provider.
    from .cli import YOLO
    task = {"weights": "detect", "pose_weights": "pose"}
    if args.backend == "onnx-directml":
        return YOLO(str(args.weights), task=task["weights"]), YOLO(str(args.pose_weights), task=task["pose_weights"])
    return YOLO(str(args.weights)), YOLO(str(args.pose_weights))


def run_chunks(args, manifest, output, analyzer_factory, temporal=None, progress_stream=sys.stderr,
               stop_after_chunks: int | None = None) -> dict[str, Any]:
    """Process every incomplete chunk in order. Returns a session report."""
    from .temporal_ball import frame_stream
    done = completed_chunks(output, manifest)
    remaining = manifest["chunks"][len(done):]
    total = manifest["selection"]["frames"]
    progress = Progress(total, sum(c["frames"] for c in done), progress_stream)
    session = {"resumed_chunks": len(done), "processed_chunks": 0, "positioning": [],
               "started_at": iso_time(progress.started)}
    if not remaining:
        return session
    analyzer = analyzer_factory()
    previous_hash = None
    if done:
        last = done[-1]
        with (chunk_dir(output, manifest["chunks"][last["index"]]) / "state.pkl").open("rb") as stream:
            analyzer.load_state(pickle.load(stream))
        previous_hash = last["last_frame_sha256"]
        if analyzer.frames != remaining[0]["first_frame"]:
            raise RuntimeError("Checkpoint frame counter does not match the next chunk")
    cap, how = open_at(Path(manifest["input"]["path"]), remaining[0]["source_first_frame"], previous_hash)
    session["positioning"].append(how)
    progress.mark()  # Loading models and positioning the decoder are not between-frame pauses.
    fps = manifest["input"]["fps"]
    try:
        for chunk in remaining:
            folder = chunk_dir(output, chunk)
            if folder.exists():
                shutil.rmtree(folder)  # Incomplete leftovers from an interrupted chunk.
            folder.mkdir(parents=True)
            started = time.perf_counter()
            started_at = iso_time(wall_clock())
            pauses = []
            last_hash = None
            count = 0
            with (folder / "events.jsonl.partial").open("w", encoding="utf-8") as stream:
                for frame, ball, stats in frame_stream(cap, temporal, chunk["frames"]):
                    row, _, _ = analyzer.process(frame, ball, stats)
                    relative = chunk["first_frame"] + count
                    if row["frame"] != relative:
                        raise RuntimeError(f"Frame counter drift: expected {relative}, analyzer produced {row['frame']}")
                    row["source_frame"] = manifest["selection"]["source_start_frame"] + relative
                    stream.write(json.dumps(row) + "\n")
                    last_hash = frame_hash(frame)
                    count += 1
                    pause = progress.update(frame=relative)
                    if pause is not None:
                        pauses.append(dict(pause, source_frame=row["source_frame"]))
            if count != chunk["frames"]:
                raise RuntimeError(f"Video ended early: chunk {chunk['index']} got {count}/{chunk['frames']} frames")
            os.replace(folder / "events.jsonl.partial", folder / "events.jsonl")
            with (folder / "state.pkl.partial").open("wb") as stream:
                pickle.dump(analyzer.state(), stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(folder / "state.pkl.partial", folder / "state.pkl")
            # chunk.json is written last: its presence marks the chunk complete.
            save_json(folder / "chunk.json", {
                "status": "complete", "index": chunk["index"], "fingerprint": manifest["fingerprint"],
                "first_frame": chunk["first_frame"], "frames": count,
                "source_first_frame": chunk["source_first_frame"],
                "source_last_frame": chunk["source_first_frame"] + count - 1,
                "last_frame_sha256": last_hash, "events_sha256": file_sha256(folder / "events.jsonl"),
                "state_sha256": file_sha256(folder / "state.pkl"),
                "seconds": round(time.perf_counter() - started, 2),
                # Timing only; completed_chunks() never reads these, so older
                # chunk.json files without them still count as complete.
                "started_at": started_at, "finished_at": iso_time(wall_clock()),
                "pause_threshold_s": PAUSE_THRESHOLD_S, "pauses": pauses,
                "paused_seconds": round(sum(p["seconds"] for p in pauses), 1)})
            session["processed_chunks"] += 1
            snapshot = progress.snapshot()
            save_json(output / "progress.json", dict(snapshot, chunks_done=chunk["index"] + 1,
                                                    chunks_total=len(manifest["chunks"]),
                                                    updated_at=iso_time(wall_clock()),
                                                    session_started_at=session["started_at"],
                                                    pause_threshold_s=PAUSE_THRESHOLD_S,
                                                    session_pauses=progress.pauses))
            print(f"[longrun] chunk {chunk['index'] + 1}/{len(manifest['chunks'])} complete: frames "
                  f"{chunk['first_frame']}-{chunk['first_frame'] + count - 1} (source "
                  f"{chunk['source_first_frame']}-{chunk['source_first_frame'] + count - 1}, "
                  f"{chunk['source_first_frame'] / fps:.1f}s-{(chunk['source_first_frame'] + count) / fps:.1f}s)",
                  file=progress_stream, flush=True)
            if stop_after_chunks is not None and session["processed_chunks"] >= stop_after_chunks:
                break
    finally:
        cap.release()
    return session


def timing_report(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Per-chunk wall-clock timing and every recorded pause, read from chunk.json.

    Chunks written before timing was recorded have no ``started_at``/``pauses``;
    they are listed with ``timing_recorded: false`` (unknown, not "no pauses").
    """
    chunks, pauses = [], []
    for chunk in manifest["chunks"]:
        path = chunk_dir(output, chunk) / "chunk.json"
        status = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        recorded = "pauses" in status
        chunks.append({"index": chunk["index"], "seconds": status.get("seconds"),
                       "started_at": status.get("started_at"), "finished_at": status.get("finished_at"),
                       "paused_seconds": status.get("paused_seconds") if recorded else None,
                       "timing_recorded": recorded})
        pauses += [dict(p, chunk=chunk["index"]) for p in status.get("pauses", [])]
    return {"pause_threshold_s": PAUSE_THRESHOLD_S, "chunks": chunks, "pauses": pauses,
            "paused_seconds_total": round(sum(p["seconds"] for p in pauses), 1),
            "chunks_without_timing": sum(not c["timing_recorded"] for c in chunks)}


def merge_run(output: Path, manifest: dict[str, Any], ball_model: Path | None, court) -> dict[str, Any]:
    """Concatenate chunk rows into the standard run files, validating continuity."""
    from .events import build_shots
    from .flight3d import fit_shots
    from .camera3d import from_court
    done = completed_chunks(output, manifest)
    if len(done) != len(manifest["chunks"]):
        raise RuntimeError(f"Only {len(done)}/{len(manifest['chunks'])} chunks are complete; rerun to resume")
    start = manifest["selection"]["source_start_frame"]
    fps, info = manifest["input"]["fps"], manifest["input"]
    raw_events, expected, event_count = [], 0, 0
    tmp = output / "events.jsonl.partial"
    with tmp.open("w", encoding="utf-8") as merged:
        for chunk in manifest["chunks"]:
            with (chunk_dir(output, chunk) / "events.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["frame"] != expected or row["source_frame"] != start + expected \
                            or abs(row["time_s"] - expected / fps) > .51 / fps:
                        raise RuntimeError(f"Duplicate, missing or misaligned frame at run frame {expected}")
                    raw_events.extend(row["events"])
                    event_count += len(row["events"])
                    merged.write(line if line.endswith("\n") else line + "\n")
                    expected += 1
    if expected != manifest["selection"]["frames"]:
        raise RuntimeError(f"Merged {expected} frames; selection has {manifest['selection']['frames']}")
    os.replace(tmp, output / "events.jsonl")
    save_json(output / "source-offset.json", {"source_start_frame": start, "source_start_seconds": start / fps,
                                              "note": "Output frames and timestamps are relative to this starting point."})
    shots = build_shots(raw_events)
    save_json(output / "shots.json", shots)
    flight = fit_shots(output / "events.jsonl", shots, from_court(court, info["width"], info["height"]) if court else None)
    save_json(output / "flight3d.json", flight)
    summary = {"input": manifest["input"]["path"], "frames": expected, "fps": fps, "width": info["width"],
               "height": info["height"], "events": event_count, "shots": shots["consolidated_counts"]["shots"],
               "flight3d_fits": len(flight["fits"]), "court_calibrated": court is not None,
               "court_source": court.source if court else None, "court_confidence": court.confidence if court else 0,
               "note": "Events, shots, and single-camera 3D fits are candidates; validate before use."}
    save_json(output / "summary.json", summary)
    if ball_model is not None:
        save_json(output / "ball-model.json", {"name": "GridTrackNet", "model": str(Path(ball_model).resolve()),
                  "sha256": manifest["models"]["ball_model"]["sha256"], "threshold": manifest["settings"]["ball_threshold"],
                  "input_frames": 5, "maximum_lookahead_frames": 4, "interpolation": False, "source": "gridtracknet"})
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Source video")
    p.add_argument("--output", required=True, type=Path, help="Run folder; rerun the same command to resume")
    p.add_argument("--weights", required=True, type=Path, help="Existing local YOLO detector checkpoint (e.g. yolo11x.pt)")
    p.add_argument("--pose-weights", required=True, type=Path, help="Existing local YOLO pose checkpoint (e.g. yolo26l-pose.pt)")
    p.add_argument("--court", type=Path, help="court.yaml with four image_corners (manual calibration)")
    p.add_argument("--scene", type=Path, help="Scene selection JSON (e.g. sebbie-scene.json)")
    p.add_argument("--temporal-ball", action="store_true", help="Use the five-frame GridTrackNet ball model")
    p.add_argument("--ball-model", type=Path, default=Path("models/gridtracknet/gridtracknet.onnx"))
    p.add_argument("--ball-threshold", type=float, default=.5)
    p.add_argument("--start-frame", type=int); p.add_argument("--end-frame", type=int, help="Exclusive")
    p.add_argument("--start-seconds", type=float); p.add_argument("--end-seconds", type=float, help="Exclusive")
    p.add_argument("--chunk-frames", type=int, default=900, help=f"Frames per resumable chunk (multiple of {BLOCK}); default 900 = 30 s at 30 fps")
    p.add_argument("--device", help="YOLO device: cpu, 0 (first CUDA GPU), ... Default lets Ultralytics choose")
    p.add_argument("--backend", choices=("pytorch", "onnx-directml"), default="pytorch",
                   help="onnx-directml runs exported .onnx weights on any DirectX 12 GPU (see export_onnx.py); "
                        "results are close to, not identical with, PyTorch CPU")
    p.add_argument("--confidence", type=float, default=0.15)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--court-side-margin", type=float, default=1.5)
    p.add_argument("--court-baseline-margin", type=float, default=6.0)
    p.add_argument("--no-detail-pass", action="store_true")
    p.add_argument("--all-players", action="store_true")
    p.add_argument("--no-auto-court", action="store_true")
    p.add_argument("--keep-awake", action="store_true", help="Windows: keep the PC from sleeping while this run is processing")
    p.add_argument("--stop-after-chunks", type=int, help="Testing aid: stop after N newly processed chunks")
    args = p.parse_args(argv)
    args.motion_ball = False  # Experimental motion recovery is not part of long runs.
    args.racket_weights = None
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    for label, path in (("--input", args.input), ("--weights", args.weights), ("--pose-weights", args.pose_weights)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} {path} does not exist (weights are never downloaded implicitly)")
    if args.temporal_ball and not args.ball_model.is_file():
        raise FileNotFoundError(f"--ball-model {args.ball_model} does not exist")
    info = probe(args.input)
    start, end = resolve_selection(info, args.start_frame, args.end_frame, args.start_seconds, args.end_seconds)
    print(f"[longrun] hashing input and models...", file=sys.stderr, flush=True)
    manifest = build_manifest(args, info, start, end)
    output = args.output
    manifest_path = output / "longrun-manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != manifest["fingerprint"]:
            raise SystemExit(f"{output} holds a run with different inputs/settings/code; choose a new --output")
        if existing.get("device") != manifest["device"]:
            raise SystemExit(f"{output} was started on device {existing.get('device')!r}; resume with the same --device")
    elif output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} exists and is not a long run; choose a new --output")
    else:
        output.mkdir(parents=True, exist_ok=True)
        save_json(manifest_path, manifest)
    from .cli import FrameAnalyzer, load_court
    court = load_court(args.court)
    scene_config = json.loads(args.scene.read_text(encoding="utf-8")) if args.scene else None
    temporal = None
    if args.temporal_ball:
        from .temporal_ball import GridTrackNetDetector
        temporal = GridTrackNetDetector(args.ball_model, args.ball_threshold)
    model, pose_model = load_models(args)
    factory = lambda: FrameAnalyzer(args, court, info["fps"], model, pose_model, None, scene_config, temporal is not None)
    print(f"[longrun] {args.input.name}: source frames {start}-{end - 1} ({(end - start) / info['fps']:.1f} s) "
          f"in {len(manifest['chunks'])} chunks -> {output}", file=sys.stderr, flush=True)
    started = time.perf_counter()
    started_at = iso_time(wall_clock())
    if args.keep_awake:
        with KeepAwake():
            session = run_chunks(args, manifest, output, factory, temporal, stop_after_chunks=args.stop_after_chunks)
    else:
        session = run_chunks(args, manifest, output, factory, temporal, stop_after_chunks=args.stop_after_chunks)
    if len(completed_chunks(output, manifest)) < len(manifest["chunks"]):
        print("[longrun] stopped before the end; rerun the same command to resume.", file=sys.stderr)
        return
    # Court may have been auto-estimated during the run; use the final analyzer state.
    last = manifest["chunks"][-1]
    with (chunk_dir(output, last) / "state.pkl").open("rb") as stream:
        court = pickle.load(stream)["court"]
    summary = merge_run(output, manifest, args.ball_model if args.temporal_ball else None, court)
    timing = timing_report(output, manifest)
    pauses = timing.pop("pauses")
    report = {"fingerprint": manifest["fingerprint"], "session": session,
              "session_seconds": round(time.perf_counter() - started, 1),
              "started_at": started_at, "finished_at": iso_time(wall_clock()),
              "chunk_seconds": [c["seconds"] for c in timing["chunks"]],
              "timing": timing, "pauses": pauses,
              "device": manifest["device"], "summary": summary}
    try:
        import torch
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)
            report["cuda_peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except ImportError:
        pass
    save_json(output / "longrun-report.json", report)
    print(f"[longrun] complete: {summary['frames']} frames -> {output}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
