"""Check whether a reviewed court calibration still fits other frames/videos.

Projects the reviewed court model (landmarks + fitted lens) into sampled frames
and measures how many projected court-line samples land on bright line pixels,
and whether a small image shift would fit better (a nudged/moved camera).

This is a consistency check for reusing a fixed-camera calibration, not an
accuracy measurement: a passing score says the painted lines are where the old
calibration expects them, within a few pixels.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .court_refinement import fit_court

# Court lines in centred metres (x across, y toward far baseline).
HALF_W, HALF_L, SINGLES, SERVICE = 5.485, 11.885, 4.115, 6.40
LINES = [
    ((-HALF_W, -HALF_L), (HALF_W, -HALF_L)), ((-HALF_W, HALF_L), (HALF_W, HALF_L)),        # baselines
    ((-HALF_W, -HALF_L), (-HALF_W, HALF_L)), ((HALF_W, -HALF_L), (HALF_W, HALF_L)),        # doubles sidelines
    ((-SINGLES, -HALF_L), (-SINGLES, HALF_L)), ((SINGLES, -HALF_L), (SINGLES, HALF_L)),    # singles sidelines
    ((-SINGLES, -SERVICE), (SINGLES, -SERVICE)), ((-SINGLES, SERVICE), (SINGLES, SERVICE)),  # service lines
    ((0, -SERVICE), (0, SERVICE)),                                                        # centre service line
]


def line_samples(camera, width, height, step_m=0.25):
    points = []
    for (x1, y1), (x2, y2) in LINES:
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step_m))
        for t in np.linspace(0, 1, n):
            points.append([x1 + t * (x2 - x1), y1 + t * (y2 - y1), 0.0])
    pixels = camera.project(np.asarray(points))
    keep = (pixels[:, 0] >= 2) & (pixels[:, 0] < width - 2) & (pixels[:, 1] >= 2) & (pixels[:, 1] < height - 2)
    return pixels[keep]


def line_mask(frame):
    """Bright, low-saturation thin structures: painted court lines."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    return ((tophat > 18) & (hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 120)).astype(np.uint8)


def score(mask, pixels, tolerance=2, shift=(0, 0)):
    near = cv2.dilate(mask, np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8))
    xy = np.round(pixels + np.asarray(shift)).astype(int)
    h, w = mask.shape
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    xy = xy[ok]
    return float(near[xy[:, 1], xy[:, 0]].mean()) if len(xy) else 0.0


def best_shift(mask, pixels, radius=8):
    best = (score(mask, pixels), (0, 0))
    for dx in range(-radius, radius + 1, 2):
        for dy in range(-radius, radius + 1, 2):
            s = score(mask, pixels, shift=(dx, dy))
            if s > best[0] + 1e-9:
                best = (s, (dx, dy))
    return best


def check(video, corrections, reference_frame, every_s=10.0, overlay_dir=None, min_ratio=0.85):
    data = json.loads(Path(corrections).read_text(encoding="utf-8"))
    width, height = data["image_size"]
    _, camera, fit = fit_court(data["landmarks"], width, height)
    pixels = line_samples(camera, width, height)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {video}")
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if list(size) != [width, height]:
        raise ValueError(f"Video is {size}, calibration is {width}x{height}: calibration cannot be reused")
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(frame_index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index}")
        return frame

    samples = []
    frames = list(range(0, total, max(1, int(round(every_s * fps)))))
    for index in frames:
        frame = read(index)
        mask = line_mask(frame)
        s0 = score(mask, pixels)
        s_best, shift = best_shift(mask, pixels)
        samples.append({"frame": index, "time_s": round(index / fps, 2), "score": round(s0, 4),
                        "best_shift_px": list(shift), "best_shift_score": round(s_best, 4)})
        if overlay_dir:
            Path(overlay_dir).mkdir(parents=True, exist_ok=True)
            for x, y in pixels.astype(int):
                cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)
            cv2.imwrite(str(Path(overlay_dir) / f"{index:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()
    return {"video": str(video), "corrections": str(corrections), "fps": fps, "frames": total,
            "line_samples": int(len(pixels)), "landmark_rmse_px": fit["landmark_rmse_px"],
            "samples": samples}


def summarise(result, reference_score, min_ratio=0.85, max_shift=2):
    scores = [s["score"] for s in result["samples"]]
    bad = [s for s in result["samples"] if s["score"] < min_ratio * reference_score
           or max(abs(v) for v in s["best_shift_px"]) > max_shift and s["best_shift_score"] > s["score"] + .05]
    return {"reference_score": reference_score, "min_score": min(scores), "median_score": float(np.median(scores)),
            "threshold": round(min_ratio * reference_score, 4), "failing_samples": [s["frame"] for s in bad],
            "verdict": "consistent" if not bad else "inconsistent",
            "meaning": "Projected court lines of the reviewed calibration land on painted lines as well as in the "
                       "calibration frame, with no better-fitting shift beyond the limit. Consistency, not accuracy."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True, action="append", help="Repeat for several videos")
    p.add_argument("--corrections", type=Path, required=True, help="Reviewed calibration-corrections.json")
    p.add_argument("--reference-video", type=Path, required=True, help="Video containing the calibration frame")
    p.add_argument("--reference-frame", type=int, required=True, help="Calibration frame in SOURCE frame numbers")
    p.add_argument("--every-seconds", type=float, default=10.0)
    p.add_argument("--output", type=Path, required=True, help="JSON report path")
    p.add_argument("--overlays", type=Path, help="Optional folder for overlay JPGs (footage stills; keep local)")
    args = p.parse_args()
    data = json.loads(args.corrections.read_text(encoding="utf-8"))
    _, camera, _ = fit_court(data["landmarks"], *data["image_size"])
    pixels = line_samples(camera, *data["image_size"])
    cap = cv2.VideoCapture(str(args.reference_video)); cap.set(cv2.CAP_PROP_POS_FRAMES, args.reference_frame)
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError("Cannot read reference frame")
    reference = score(line_mask(frame), pixels)
    report = {"reference": {"video": str(args.reference_video), "source_frame": args.reference_frame,
                            "score": round(reference, 4)}, "videos": []}
    for video in args.video:
        result = check(video, args.corrections, args.reference_frame, args.every_seconds,
                       args.overlays / video.stem if args.overlays else None)
        result["summary"] = summarise(result, reference)
        report["videos"].append(result)
        print(f"{video.name}: {result['summary']['verdict']} (median {result['summary']['median_score']:.3f}, "
              f"min {result['summary']['min_score']:.3f}, reference {reference:.3f}, "
              f"failing {result['summary']['failing_samples']})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
