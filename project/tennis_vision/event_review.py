"""Build a local ball-continuity and event-review package from an existing run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from tqdm import tqdm
import yaml

from .court import CourtMapper
from .event_analysis import VERSION, observed_balls, detect_events, fill_short_gaps


def _save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def load_run(run):
    summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
    offset_path = run/"source-offset.json"
    if not offset_path.exists():
        raise ValueError("Missing source-offset.json. Specify the actual source_start_frame there; do not guess alignment.")
    offset = json.loads(offset_path.read_text(encoding="utf-8"))
    fps = float(summary["fps"])
    start = offset["source_start_frame"]
    if not np.isfinite(fps) or fps <= 0 or type(start) is not int or start < 0:
        raise ValueError("Invalid frame rate/source offset")
    if abs(offset.get("source_start_seconds", start/fps)-start/fps) > 1/fps:
        raise ValueError("Source offset seconds and frames disagree")
    rows = [json.loads(line) for line in (run/"events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != summary["frames"]:
        raise ValueError("Summary/log frame count mismatch")
    for f, row in enumerate(rows):
        timestamp = row.get("time_s", f/fps)
        if row["frame"] != f or not np.isfinite(timestamp) or abs(timestamp-f/fps) > .51/fps:
            raise ValueError(f"Frame/timestamp alignment mismatch at {f}")
    return summary, start, rows


def _capture(summary, start):
    cap = cv2.VideoCapture(str(summary["input"]))
    if not cap.isOpened():
        raise ValueError("Cannot open original video")
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if actual != (summary["width"], summary["height"]) or abs(cap.get(cv2.CAP_PROP_FPS)-summary["fps"]) > .01:
        cap.release()
        raise ValueError("Original video dimensions/fps differ from cached run")
    if start+summary["frames"] > cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.release()
        raise ValueError("Source interval exceeds video length")
    if start and not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
        cap.release()
        raise ValueError("Cannot seek source video")
    if abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-start) > .5:
        cap.release()
        raise ValueError("Source seek did not land at requested frame")
    return cap


def detect_cuts(summary, start):
    """Conservative guard for obvious abrupt image changes, not shot detection."""
    cap = _capture(summary, start)
    cuts, previous = [], None
    try:
        for f in tqdm(range(summary["frames"]), desc="Check source alignment/cuts", unit="frame"):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Source ended early at review frame {f}")
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(float)/255
            if previous is not None and np.mean(np.abs(small-previous)) > .22:
                cuts.append(f)
            previous = small
    finally:
        cap.release()
    return cuts


class H264Writer:
    def __init__(self, ffmpeg, path, summary):
        self.log = path.with_suffix(".ffmpeg.log").open("wb")
        try:
            self.process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f'{summary["width"]}x{summary["height"]}',
                "-r", str(summary["fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
                "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except Exception:
            self.log.close()
            raise

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def close(self):
        try:
            if not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass  # Still reap the failed encoder and report its exit.
            try:
                code = self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("Video encoder timed out")
            if code:
                raise RuntimeError("Video encoding failed; inspect the .ffmpeg.log in the output folder")
        finally:
            self.log.close()


def _overlay(image, row, samples, events, fps, start):
    f = row["frame"]
    sample = samples[f]
    colours = {"detected": (0, 235, 255), "estimated": (0, 145, 255), "missing": (160, 160, 160)}
    for player in row["tracks"]:
        if player.get("label") != "player" or not player.get("identity_id"):
            continue
        x1, y1, x2, y2 = map(round, player["bbox"])
        colour = (245, 185, 80) if not player.get("predicted") else (140, 140, 140)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 1)
        cv2.putText(image, player["identity_id"] + (" (predicted)" if player.get("predicted") else ""),
                    (x1, max(60, y1-5)), 0, .45, colour, 1, cv2.LINE_AA)
    # Short trails break at every missing sample. Orange segments are estimates.
    for k in range(max(1, f-9), f+1):
        a, b = samples[k-1], samples[k]
        if a["pixel"] is not None and b["pixel"] is not None:
            status = "estimated" if "estimated" in (a["status"], b["status"]) else "detected"
            cv2.line(image, tuple(map(round, a["pixel"])), tuple(map(round, b["pixel"])), colours[status], 1, cv2.LINE_AA)
    if sample["pixel"] is not None:
        point = tuple(map(round, sample["pixel"]))
        if sample["status"] == "estimated":
            cv2.drawMarker(image, point, colours["estimated"], cv2.MARKER_TILTED_CROSS, 14, 2)
        else:
            cv2.circle(image, point, 7, colours["detected"], 2, cv2.LINE_AA)
    cv2.rectangle(image, (0, image.shape[0]-65), (image.shape[1], image.shape[0]), (18, 24, 27), -1)
    cv2.putText(image, f'Frame {f} | clip {f/fps:.2f}s | source {(start+f)/fps:.2f}s | ball {sample["status"].upper()}',
                (14, image.shape[0]-40), 0, .58, colours[sample["status"]], 1, cv2.LINE_AA)
    nearby = [e for e in events if abs(e["frame"]-f) <= 5]
    text = (" / ".join(f'{e["type"]} at f{e["frame"]} {e["player_id"] or ""}' for e in nearby)
            or "Yellow = detected | orange X = estimated | candidate events need human review")
    cv2.putText(image, text, (14, image.shape[0]-15), 0, .50, (240, 240, 240), 1, cv2.LINE_AA)


def render_videos(summary, start, rows, samples, events, output, ffmpeg):
    cap = _capture(summary, start)
    writers = []
    try:
        writers.append(H264Writer(ffmpeg, output/"source.mp4", summary))
        writers.append(H264Writer(ffmpeg, output/"review.mp4", summary))
        for f, row in enumerate(tqdm(rows, desc="Render review + raw video", unit="frame")):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Source ended at frame {f}")
            writers[0].write(image)
            _overlay(image, row, samples, events, summary["fps"], start)
            writers[1].write(image)
    finally:
        cap.release()
        failures = []
        for writer in writers:
            try:
                writer.close()
            except Exception as error:
                failures.append(error)
        if failures:
            raise failures[0]
    for name in ("source.mp4", "review.mp4"):
        check = cv2.VideoCapture(str(output/name))
        count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        check.release()
        if count != len(rows):
            raise RuntimeError(f"Encoded frame mismatch in {name}: {count}")
    (output/"clips").mkdir()
    for event in tqdm(events, desc="Create candidate clips", unit="clip"):
        a = max(0, event["frame"]-round(summary["fps"]*.8))
        b = min(len(rows), event["frame"]+round(summary["fps"]*.8)+1)
        name = f'clips/{event["id"]}.mp4'
        result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(output/"review.mp4"),
            "-vf", f'trim=start_frame={a}:end_frame={b},setpts=N/({summary["fps"]}*TB)',
            "-r", str(summary["fps"]), "-fps_mode", "cfr", "-an", "-c:v", "libx264", "-threads", "2",
            "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output/name)],
            capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        check = cv2.VideoCapture(str(output/name))
        decoded = 0
        while check.read()[0]:
            decoded += 1
        check.release()
        if decoded != b-a:
            raise RuntimeError(f"Clip frame mismatch in {name}: {decoded} != {b-a}")
        event["clip"] = {"path": name, "start_frame": a, "end_frame_exclusive": b,
                         "source_start_frame": start+a, "event_offset_s": (event["frame"]-a)/summary["fps"]}


def write_review_page(output):
    """Regenerate only the HTML from completed exports after a UI-only update."""
    output = Path(output)
    report = json.loads((output/"review-report.json").read_text(encoding="utf-8"))
    events = json.loads((output/"event-candidates.json").read_text(encoding="utf-8"))["events"]
    gaps = json.loads((output/"gap-review.json").read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in (output/"ball-continuity.jsonl").read_text(encoding="utf-8").splitlines()]
    payload = {"report": report, "events": events, "samples": [
        {"frame": s["frame"], "status": s["status"]} for s in samples], "gaps": gaps}
    template = Path(__file__).with_name("event_review.html").read_text(encoding="utf-8")
    data = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    (output/"review.html").write_text(template.replace("__REVIEW_DATA__", data), encoding="utf-8")


def build_review(run, output, court_path=None, ffmpeg="ffmpeg", data_only=False):
    run, output = Path(run), Path(output)
    started = time.perf_counter()
    summary, start, rows = load_run(run)
    court = None
    if court_path:
        config = yaml.safe_load(Path(court_path).read_text(encoding="utf-8"))
        corners = np.asarray(config["image_corners"], dtype=np.float32)
        if corners.shape != (4, 2) or not np.isfinite(corners).all() or abs(cv2.contourArea(corners)) < 100:
            raise ValueError("Invalid manual court corners")
        court = CourtMapper(corners)
    if not data_only:
        ffmpeg = shutil.which(str(ffmpeg))
        if not ffmpeg:
            raise ValueError("FFmpeg is required for browser-compatible H.264 videos; supply --ffmpeg PATH or use --data-only")
        encoder_check = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, timeout=15)
        if encoder_check.returncode or b"libx264" not in encoder_check.stdout:
            raise ValueError("FFmpeg must include the libx264 encoder")
    cuts = detect_cuts(summary, start)
    balls = observed_balls(rows)
    events = detect_events(rows, balls, summary["fps"], court, cuts)
    samples, gaps = fill_short_gaps(rows, balls, summary["fps"], events, cuts)
    for event in events:
        event["source_frame"] = start+event["frame"]
        event["source_time_s"] = (start+event["frame"])/summary["fps"]
    fingerprint = hashlib.sha256((run/"events.jsonl").read_bytes()+json.dumps(
        {"summary": summary, "start": start, "court": court.image_corners.tolist() if court else None,
         "version": VERSION}, sort_keys=True).encode()).hexdigest()
    report = {"schema_version": 1, "algorithm": VERSION, "run_id": fingerprint, "source_run": str(run.resolve()),
              "source_video": summary["input"], "source_start_frame": start, "fps": summary["fps"],
              "frames": len(rows), "ball_frames": dict(Counter(s["status"] for s in samples)),
              "gap_decisions": dict(Counter(g["reason"] for g in gaps)), "scene_cut_frames": cuts,
              "candidate_counts": dict(Counter(e["type"] for e in events)), "court_calibrated": court is not None,
              "limitations": ["Coverage is not accuracy. All events start unreviewed; scores are NOT probabilities.",
                              "Estimated points are display-only and never feed events, legacy shots, or 3D.",
                              "No extrapolation; up to 3 missing frames filled only with consistent observed context away from impacts.",
                              "Missed events and false candidates remain possible, especially when the ball is occluded.",
                              "Bounce court coordinates are conditional on a real ground contact; no airborne XYZ is measured.",
                              "Court calibration assumes a fixed camera. Obvious-cut guard does not detect every camera movement.",
                              "Export review labels to retain them; browser-local autosave is best-effort. Review videos are silent."]}
    output.mkdir(parents=True, exist_ok=False)
    _save(output/"build-status.json", {"status": "building"})
    try:
        if not data_only:
            render_videos(summary, start, rows, samples, events, output, ffmpeg)
        with (output/"ball-continuity.jsonl").open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample, allow_nan=False)+"\n")
        _save(output/"gap-review.json", gaps)
        _save(output/"event-candidates.json", {"schema_version": 1, "run_id": fingerprint, "events": events})
        report["seconds_processing"] = round(time.perf_counter()-started, 2)
        _save(output/"review-report.json", report)
        if not data_only:
            write_review_page(output)
        _save(output/"build-status.json", {"status": "complete", "data_only": data_only})
    except Exception as error:
        _save(output/"build-status.json", {"status": "failed", "error": str(error)})
        raise
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Must be a new folder")
    parser.add_argument("--court", type=Path, help="Manual calibration; required for bounce candidates")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--data-only", action="store_true", help="Skip video/HTML export, but still validate source/cuts")
    args = parser.parse_args()
    build_review(args.run, args.output, args.court, args.ffmpeg, args.data_only)


if __name__ == "__main__":
    main()
