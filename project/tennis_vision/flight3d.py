"""Gravity-constrained monocular ball-flight fitting with transparent confidence."""
import json
from pathlib import Path

import numpy as np

from .camera3d import Camera3D

GRAVITY = 9.81


def _world(court_m: list[float], z: float = 0.0) -> np.ndarray:
    return np.array([court_m[0] - 10.97 / 2, court_m[1] - 23.77 / 2, z], dtype=float)


def _positions(params: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    p0, velocity = params[:3], params[3:]
    return p0 + seconds[:, None] * velocity + np.column_stack((np.zeros(len(seconds)), np.zeros(len(seconds)), -0.5 * GRAVITY * seconds**2))


def _ball_pixels(events_path: Path) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            balls = [track for track in row.get("tracks", []) if track.get("label") == "ball"
                     and not track.get("predicted", False) and track.get('source','model') in ('model','gridtracknet')]
            if len(balls) != 1:
                continue
            ball = max(balls, key=lambda item: item.get("confidence", 0))
            x1, y1, x2, y2 = ball["bbox"]
            result[int(row["frame"])] = ((x1 + x2) / 2, (y1 + y2) / 2)
    return result


def _fit_one(camera: Camera3D, start: dict, landing: dict, pixels: dict[int, tuple[float, float]]) -> dict | None:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError("scipy is required for 3D flight fitting") from exc
    start_frame, bounce_frame = int(start["frame"]), int(landing["frame"])
    if bounce_frame - start_frame < 4 or not landing.get("court_m"):
        return None
    observed = [(frame, pixels[frame]) for frame in sorted(pixels) if start_frame <= frame <= bounce_frame]
    if len(observed) < 5:
        return None
    # Event times are resilient to variable frame rate because they originate in the decoder.
    duration = max(float(landing["time_s"] - start["time_s"]), 0.08)
    frames = np.array([frame for frame, _ in observed])
    times = (frames - start_frame) / max(bounce_frame - start_frame, 1) * duration
    target = _world(landing["court_m"])
    pixels_array = np.asarray([pixel for _, pixel in observed], dtype=float)
    first_pixel = pixels_array[0]
    best = None
    for contact_height in (0.7, 1.0, 1.35):
        origin = camera.pixel_to_plane(tuple(first_pixel), contact_height)
        if origin is None:
            continue
        velocity = (target - origin) / duration + np.array([0, 0, GRAVITY * duration / 2])
        initial = np.concatenate((origin, velocity))
        def residual(params: np.ndarray) -> np.ndarray:
            predicted = _positions(params, times)
            reprojection = (camera.project(predicted) - pixels_array).ravel()
            landing_error = (_positions(params, np.array([duration]))[0] - target) * 55
            return np.concatenate((reprojection, landing_error))
        solution = least_squares(residual, initial, loss="soft_l1", f_scale=4.0, max_nfev=180)
        if best is None or solution.cost < best.cost:
            best = solution
    if best is None:
        return None
    predicted = _positions(best.x, times)
    error_px = np.linalg.norm(camera.project(predicted) - pixels_array, axis=1)
    if float(np.median(error_px)) > 18:
        return None
    dense_frames = np.arange(start_frame, bounce_frame + 1)
    dense_times = (dense_frames - start_frame) / max(bounce_frame - start_frame, 1) * duration
    dense_positions = _positions(best.x, dense_times)
    # Net clearance where y crosses zero, if this shot crosses the net.
    crossings = []
    for a, b in zip(dense_positions[:-1], dense_positions[1:]):
        if a[1] * b[1] <= 0 and a[1] != b[1]:
            ratio = -a[1] / (b[1] - a[1])
            crossings.append(float(a[2] + ratio * (b[2] - a[2])))
    return {
        "start_frame": start_frame, "bounce_frame": bounce_frame,
        "median_reprojection_error_px": round(float(np.median(error_px)), 2),
        "fit_confidence": "medium" if float(np.median(error_px)) < 7 else "low",
        "net_crossing_height_m": round(crossings[0], 3) if crossings else None,
        "trajectory_m": [{"frame": int(frame), "point_m": [round(float(v), 3) for v in position]} for frame, position in zip(dense_frames, dense_positions)],
    }


def fit_shots(events_path: Path, shots: dict, camera: Camera3D | None) -> dict:
    if camera is None:
        return {"status": "skipped", "reason": "No plausible camera pose from the court estimate.", "fits": []}
    pixels = _ball_pixels(events_path)
    fits = []
    for shot in shots.get("shots", []):
        landing = shot.get("landing")
        if landing is None:
            continue
        fit = _fit_one(camera, shot["start_contact"], landing, pixels)
        if fit:
            fit["shot_id"] = shot["shot_id"]
            fits.append(fit)
    return {"status": "ok", "camera_position_m": [round(float(v), 3) for v in camera.position_m], "camera_confidence": camera.confidence, "fits": fits}
