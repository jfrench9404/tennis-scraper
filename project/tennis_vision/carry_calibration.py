"""Carry a reviewed fixed-camera calibration to a NEW run, as a draft.

Reviewed court corrections are bound to one run id and carry bounce decisions
for that run's event ids. For a different run (a full game from the same fixed
camera) this writes a new corrections file that:

* is bound to the new run's review id (never the old one);
* copies only the court landmarks, and only after camera_check finds the court
  lines where the old calibration expects them throughout the new selection;
* is marked calibration_status "draft" until John reviews it for this run;
* carries NO bounce edits: old event ids and decisions never transfer.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import camera_check


def carry(source_corrections, review_dir, run_dir, output, every_seconds=10.0, reference=None):
    source = json.loads(Path(source_corrections).read_text(encoding="utf-8"))
    if source.get("kind") != "court_and_bounce_review" or source.get("calibration_status") != "reviewed":
        raise ValueError("Source must be a reviewed court_and_bounce_review corrections file")
    report = json.loads((Path(review_dir) / "review-report.json").read_text(encoding="utf-8"))
    summary = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
    offset = json.loads((Path(run_dir) / "source-offset.json").read_text(encoding="utf-8"))["source_start_frame"]
    if report.get("source_start_frame") != offset or report.get("frames") != summary["frames"]:
        raise ValueError("Review folder does not belong to this run")
    if [summary["width"], summary["height"]] != source["image_size"]:
        raise ValueError("Image size differs from the reviewed calibration; it cannot be reused")
    if report["run_id"] == source["run_id"]:
        raise ValueError("This is the calibration's own run; use the reviewed file directly")
    start, end = offset, offset + summary["frames"]
    step = max(1, int(round(every_seconds * summary["fps"])))
    # Sample inside this run only, always including its first and last frame.
    frames = sorted(set(range(start, end, step)) | {end - 1})
    check = camera_check.check(summary["input"], source_corrections, None, every_seconds, frames=frames)
    if not check["samples"]:
        raise ValueError("No camera-check samples fall inside this run; lower --every-seconds")
    reference_score = reference if reference is not None else max(s["score"] for s in check["samples"])
    verdict = camera_check.summarise(check, reference_score)
    if verdict["verdict"] != "consistent":
        raise ValueError(f"Court lines do not match the reviewed calibration at source frames "
                         f"{verdict['failing_samples']}; re-annotate landmarks for this run instead")
    best = max(check["samples"], key=lambda s: s["score"])
    carried = {
        "schema_version": 1, "kind": "court_and_bounce_review", "run_id": report["run_id"],
        "image_size": source["image_size"], "calibration_frame": best["frame"] - offset,
        "calibration_status": "draft",
        "landmark_source": f"carried_from_reviewed_run_{source['run_id'][:16]}_after_camera_check",
        "landmarks": source["landmarks"], "bounce_edits": [],
        "carried_from": {"run_id": source["run_id"], "calibration_frame": source["calibration_frame"],
                         "landmark_source": source.get("landmark_source"), "status_there": "reviewed",
                         "file": str(Path(source_corrections).resolve())},
        "camera_check": {"samples": check["samples"], "summary": verdict,
                         "meaning": "Consistency of court-line positions with the reviewed calibration, not accuracy."},
        "notes": "Landmarks only. Bounce decisions and event ids from the source run are NOT carried. "
                 "Draft until reviewed for this run in the calibration desk."}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(carried, indent=2), encoding="utf-8")
    return carried


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrections", type=Path, required=True, help="Reviewed calibration-corrections.json")
    p.add_argument("--run", type=Path, required=True, help="New run folder (longrun output)")
    p.add_argument("--review", type=Path, required=True, help="Event-review folder built from --run")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--every-seconds", type=float, default=10.0)
    p.add_argument("--reference-score", type=float,
                   help="Line score at the reviewed calibration frame (camera_check reports it); default: best in-run sample")
    args = p.parse_args()
    carried = carry(args.corrections, args.review, args.run, args.output, args.every_seconds, args.reference_score)
    s = carried["camera_check"]["summary"]
    print(f"Wrote draft calibration for run {carried['run_id'][:16]} ({s['verdict']}, "
          f"median line score {s['median_score']:.3f}, {len(carried['camera_check']['samples'])} samples)")


if __name__ == "__main__":
    main()
