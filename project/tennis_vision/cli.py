import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from tqdm.auto import tqdm

from .court import CourtMapper, estimate_court
from .events import EventEngine, build_shots
from .camera3d import from_court
from .flight3d import fit_shots
from .tracking import Detection, NearestTracker
from .filters import filter_tracks, on_court
from .detail import DetailPass, merge
from .scene import SceneSelector
from .ball_motion import MotionBallRecovery
from .temporal_ball import GridTrackNetDetector, frame_stream, candidate_detections
from .setup_temporal import DEFAULT_MODEL, file_hash


LABELS = {"person": "player", "sports_ball": "ball", "tennis_ball": "ball", "racket": "racket", "tennis_racket": "racket"}
COCO_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
SKELETON = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze authorized tennis footage.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/tennis-analysis"))
    parser.add_argument("--weights", default="yolo11x.pt", help="Ball/racket detector checkpoint or model name")
    parser.add_argument("--pose-weights", default="yolo26l-pose.pt", help="Latest Ultralytics COCO-pose checkpoint; downloads on first use")
    parser.add_argument("--racket-weights", help="Optional custom tennis-racket detector checkpoint")
    parser.add_argument("--court", type=Path, help="YAML with near-left, near-right, far-right, far-left pixels")
    parser.add_argument("--no-auto-court", action="store_true", help="Disable automatic court estimate")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size; 1280 preserves more detail for distant players than 640, but is slower")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes all frames")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Begin at this time in the source; output timeline starts at zero")
    parser.add_argument("--no-progress", action="store_true", help="Hide terminal progress and ETA")
    parser.add_argument("--court-side-margin", type=float, default=1.5, help="Player allowance outside doubles sidelines in metres")
    parser.add_argument("--court-baseline-margin", type=float, default=6.0, help="Player allowance behind baselines in metres")
    parser.add_argument("--no-detail-pass", action="store_true", help="Disable extra far-court and recent-ball crop inference")
    parser.add_argument("--scene", type=Path, help="Optional normalized search/identity polygons and ball colour settings in JSON")
    parser.add_argument("--all-players", action="store_true", help="Disable two-player appearance selection and scene ball filtering")
    parser.add_argument("--motion-ball", action="store_true", help="Experimental fixed-camera motion recovery; orange tentative balls, excluded from shot/bounce events")
    parser.add_argument("--temporal-ball", action="store_true", help="Replace generic ball detections with the trained five-frame tennis model")
    parser.add_argument("--ball-model", type=Path, default=DEFAULT_MODEL, help="GridTrackNet ONNX file, used with --temporal-ball")
    parser.add_argument("--ball-threshold", type=float, default=.5, help="Temporal ball score threshold (not an accuracy percentage)")
    return parser.parse_args()


def load_court(path: Path | None) -> CourtMapper | None:
    if path is None:
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    corners = raw.get("image_corners", []) if isinstance(raw, dict) else []
    if len(corners) != 4:
        raise ValueError("court YAML must include exactly four image_corners")
    return CourtMapper(np.asarray(corners, dtype=np.float32), source="manual", confidence=1.0)


def normalize(label: str) -> str | None:
    return LABELS.get(label.lower().replace(" ", "_")) or LABELS.get(label.lower())


def detect(model: YOLO, frame: np.ndarray, confidence: float, include_players: bool = False, imgsz: int = 1280) -> list[Detection]:
    result = model.predict(frame, conf=confidence, imgsz=imgsz, verbose=False)[0]
    names: dict[int, str] = result.names
    rows: list[Detection] = []
    for box in result.boxes:
        label = normalize(str(names[int(box.cls[0])]))
        if label is None or (label == "player" and not include_players):
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        rows.append(Detection(label, (x1, y1, x2, y2), float(box.conf[0])))
    return rows


def detect_pose(model: YOLO, frame: np.ndarray, confidence: float, imgsz: int = 1280) -> list[Detection]:
    result = model.predict(frame, conf=confidence, imgsz=imgsz, verbose=False)[0]
    if result.boxes is None or result.keypoints is None:
        return []
    keypoint_data = result.keypoints.data.cpu().tolist()
    rows = []
    for box, joints in zip(result.boxes, keypoint_data):
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        parsed = {name: [round(float(point[0]), 2), round(float(point[1]), 2), round(float(point[2]) if len(point) > 2 else 1.0, 3)] for name, point in zip(COCO_JOINTS, joints)}
        rows.append(Detection("player", (x1, y1, x2, y2), float(box.conf[0]), parsed))
    return rows


def draw(frame: np.ndarray, records: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    colours = {"player": (255, 180, 0), "ball": (0, 255, 255), "racket": (255, 0, 255)}
    for row in records:
        x1, y1, x2, y2 = map(int, row["bbox"])
        colour = (0,165,255) if row.get('source')=='motion' else colours[row["label"]]
        thickness = 1 if row.get("predicted") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
        suffix = " (predicted)" if row.get("predicted") else ""
        if row.get('source')=='motion': suffix+=' (motion candidate)'
        if row.get('source')=='gridtracknet': suffix+=' (temporal)'
        cv2.putText(frame, f"{row.get('identity_id') or row['label']} #{row['track_id']}{suffix}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        joints = row.get("keypoints") or {}
        for a, b in SKELETON:
            pa, pb = joints.get(COCO_JOINTS[a]), joints.get(COCO_JOINTS[b])
            if pa and pb and pa[2] >= 0.25 and pb[2] >= 0.25:
                cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), colour, 2)
        for joint in joints.values():
            if joint[2] >= 0.25:
                cv2.circle(frame, (int(joint[0]), int(joint[1])), 3, colour, -1)
    for i, event in enumerate(events):
        cv2.putText(frame, event["type"].upper(), (20, 35 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


def main() -> None:
    args = parse_args()
    if args.temporal_ball and args.motion_ball:
        raise ValueError('Choose --temporal-ball or --motion-ball, not both; keep model evidence distinct from motion guesses.')
    if not np.isfinite(args.start_seconds) or args.start_seconds < 0:
        raise ValueError('--start-seconds must be finite and nonnegative')
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    temporal = GridTrackNetDetector(args.ball_model,args.ball_threshold) if args.temporal_ball else None
    args.output.mkdir(parents=True, exist_ok=True)
    court = load_court(args.court)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    start_frame = int(round(args.start_seconds * fps))
    if total_frames is not None:
        if start_frame >= total_frames:
            cap.release()
            raise ValueError('--start-seconds is past the end of the video')
        total_frames -= start_frame
    if start_frame and not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
        cap.release()
        raise RuntimeError('Could not seek to the requested video time')
    (args.output / 'source-offset.json').write_text(json.dumps({'source_start_frame': start_frame, 'source_start_seconds': start_frame / fps, 'note': 'Output frames and timestamps are relative to this starting point.'}, indent=2), encoding='utf-8')
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames) if total_frames else args.max_frames
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(args.output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model, pose_model, tracker, engine = YOLO(args.weights), YOLO(args.pose_weights), NearestTracker(), EventEngine(fps, court)
    racket_model = YOLO(args.racket_weights) if args.racket_weights else None
    detail = DetailPass()
    scene = SceneSelector(json.loads(args.scene.read_text()) if args.scene else None, fps)
    recovery = MotionBallRecovery()
    if args.motion_ball and (args.all_players or court is None or court.source != 'manual'):
        cap.release(); writer.release()
        raise ValueError('--motion-ball requires manual --court calibration and scene selection (no --all-players).')
    frames = event_count = 0
    raw_events: list[dict[str, Any]] = []
    progress = tqdm(total=total_frames, desc="Analyzing", unit="frame", dynamic_ncols=True, disable=args.no_progress)
    try:
        with (args.output / "events.jsonl").open("w", encoding="utf-8") as stream:
            for frame, temporal_ball, temporal_stats in frame_stream(cap,temporal,args.max_frames):
                if court is None and not args.no_auto_court and frames < int(fps * 8):
                    court = estimate_court(frame)
                engine.court = court
                detections = detect(model, frame, args.confidence, include_players=True, imgsz=args.imgsz) + detect_pose(pose_model, frame, args.confidence, imgsz=args.imgsz)
                detail_stats = {}
                if not args.no_detail_pass:
                    detections, detail_stats = detail.update(
                        frame, detections, court,
                        lambda crop: detect(model, crop, args.confidence, include_players=True, imgsz=args.imgsz),
                        lambda crop: detect_pose(pose_model, crop, args.confidence, imgsz=args.imgsz),
                        args.court_side_margin, args.court_baseline_margin)
                if racket_model:
                    detections = [d for d in detections if d.label != "racket"]
                    detections.extend(d for d in detect(racket_model, frame, args.confidence, imgsz=args.imgsz) if d.label == "racket")
                detections = merge(detections)
                if temporal is not None:
                    # No silent YOLO fallback: the comparison and exported source
                    # remain attributable to the dedicated tennis model.
                    detections = [d for d in detections if d.label != 'ball']
                    detections.extend(candidate_detections(temporal_ball,temporal_stats,frame.shape))
                    detail_stats['temporal_ball'] = temporal_stats
                # Preserve pre-filter observations so missing detections can be
                # distinguished from rejected ones without rerunning inference.
                detail_stats['candidates'] = [{'label': d.label, 'bbox': list(d.bbox), 'confidence': float(d.confidence), 'keypoints': d.keypoints, 'source': d.source} for d in detections]
                blockers = list(detections)
                if not args.all_players and court is not None and court.source == 'manual':
                    detections, scene_stats = scene.update(frame, detections, court)
                    detail_stats['scene'] = scene_stats
                if temporal is not None:
                    balls = [d for d in detections if d.label=='ball']
                    chosen = max(balls,key=lambda d:d.confidence) if balls else None
                    detections = [d for d in detections if d.label!='ball'] + ([chosen] if chosen else [])
                    detail_stats['temporal_ball']['selected_pixel'] = list(chosen.center) if chosen else None
                    detail_stats['temporal_ball']['selected_score'] = chosen.confidence if chosen else None
                if args.motion_ball:
                    players = [d for d in detections if d.label == 'player']
                    detections, detail_stats['motion_ball'] = recovery.update(
                        frame, detections, lambda p: scene.ball_allowed(p,frame.shape,court,players), blockers)
                detail.feedback(detections)
                rejected_players = sum(d.label == 'player' and not on_court(d, court, args.court_side_margin, args.court_baseline_margin) for d in detections)
                detections = [d for d in detections if d.label != 'player' or on_court(d, court, args.court_side_margin, args.court_baseline_margin)]
                tracks, filter_stats = filter_tracks(tracker.update(detections, frames), court, args.court_side_margin, args.court_baseline_margin)
                filter_stats['off_court_players_before_tracking'] = rejected_players
                events = engine.update([track for track in tracks if not track.predicted], frames)
                raw_events.extend(events)
                records = [t.as_dict(court) for t in tracks]
                stream.write(json.dumps({"frame": frames, "time_s": round(frames / fps, 4), "tracks": records, "events": events, "filters": filter_stats, "detail": detail_stats}) + "\n")
                draw(frame, records, events)
                writer.write(frame)
                frames += 1
                event_count += len(events)
                progress.update(1)
    except KeyboardInterrupt:
        print("\nAnalysis cancelled. Partial files were kept; rerun into a new output folder for a complete result.")
        return
    finally:
        cap.release(); writer.release(); progress.close()
    shots = build_shots(raw_events)
    (args.output / "shots.json").write_text(json.dumps(shots, indent=2), encoding="utf-8")
    flight = fit_shots(args.output / "events.jsonl", shots, from_court(court, width, height) if court else None)
    (args.output / "flight3d.json").write_text(json.dumps(flight, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({"input": str(args.input), "frames": frames, "fps": fps, "width": width, "height": height, "events": event_count, "shots": shots["consolidated_counts"]["shots"], "flight3d_fits": len(flight["fits"]), "court_calibrated": court is not None, "court_source": court.source if court else None, "court_confidence": court.confidence if court else 0, "note": "Events, shots, and single-camera 3D fits are candidates; validate before use."}, indent=2), encoding="utf-8")
    if temporal is not None:
        (args.output / 'ball-model.json').write_text(json.dumps({'name':'GridTrackNet','model':str(args.ball_model.resolve()),
            'sha256':file_hash(args.ball_model),'threshold':args.ball_threshold,'input_frames':5,'maximum_lookahead_frames':4,
            'interpolation':False,'source':'gridtracknet'},indent=2),encoding='utf-8')
    print(f"Wrote {frames} frames to {args.output}")
