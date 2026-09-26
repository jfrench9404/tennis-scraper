"""One command from a video (or a frame/second range of it) to a 3D replay.

Stages, each skipped when its output is already complete, so rerunning the same
command resumes where it stopped:

  1. run/          longrun.py      chunked, resumable detection + tracking
  2. review/       event_review    ball continuity, candidates, H.264 videos
  3. calibration   carry_calibration  reviewed landmarks -> DRAFT for this run
                                  (only when --corrections is given and the
                                  camera check passes; no bounce decisions)
  4. replay/       shot_replay     far-player crop poses, contacts, bodies,
                                  racquets (always a new folder)

Nothing is uploaded; models must already exist locally.
"""
import argparse
import json
from pathlib import Path
import sys


def complete(folder: Path) -> bool:
    status = folder / "build-status.json"
    return status.is_file() and json.loads(status.read_text(encoding="utf-8")).get("status") == "complete"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True, help="Folder for run/, review/, replay/ ...")
    p.add_argument("--run-dir", type=Path, help="Existing/target long-run folder (default OUTPUT_ROOT/run)")
    p.add_argument("--court", type=Path, required=True, help="court.yaml (four image corners)")
    p.add_argument("--scene", type=Path, help="Scene selection JSON")
    p.add_argument("--corrections", type=Path, help="REVIEWED calibration-corrections.json to carry as a draft")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--pose-weights", type=Path, required=True)
    p.add_argument("--refine-pose-weights", type=Path, help="Pose model for far-player crops (default --pose-weights)")
    p.add_argument("--ball-model", type=Path, default=Path("models/gridtracknet/gridtracknet.onnx"))
    p.add_argument("--backend", choices=("pytorch", "onnx-directml"), default="pytorch")
    p.add_argument("--device")
    p.add_argument("--start-frame", type=int); p.add_argument("--end-frame", type=int)
    p.add_argument("--start-seconds", type=float); p.add_argument("--end-seconds", type=float)
    p.add_argument("--chunk-frames", type=int, default=300)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--keep-awake", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir or root / "run"
    review_dir, replay_dir = root / "review", root / "replay"
    log = lambda text: print(f"[game] {text}", file=sys.stderr, flush=True)

    from . import longrun
    longrun_args = ["--input", str(args.input), "--output", str(run_dir), "--weights", str(args.weights),
                    "--pose-weights", str(args.pose_weights), "--court", str(args.court), "--temporal-ball",
                    "--ball-model", str(args.ball_model), "--chunk-frames", str(args.chunk_frames),
                    "--backend", args.backend]
    for flag in ("scene", "device", "start_frame", "end_frame", "start_seconds", "end_seconds"):
        value = getattr(args, flag)
        if value is not None:
            longrun_args += ["--" + flag.replace("_", "-"), str(value)]
    if args.keep_awake:
        longrun_args.append("--keep-awake")
    if (run_dir / "longrun-report.json").is_file():
        log(f"1/4 detection already complete: {run_dir}")
    else:
        log(f"1/4 detection + tracking -> {run_dir}")
        longrun.main(longrun_args)
        if not (run_dir / "longrun-report.json").is_file():
            log("Detection stopped before the end; rerun this command to resume.")
            return

    from .event_review import build_review
    if complete(review_dir):
        log(f"2/4 event review already complete: {review_dir}")
    else:
        if review_dir.exists():
            raise SystemExit(f"{review_dir} is incomplete from an earlier attempt; move it aside and rerun")
        log(f"2/4 event review + review videos -> {review_dir}")
        build_review(run_dir, review_dir, args.court, args.ffmpeg)

    corrections = None
    if args.corrections:
        from .carry_calibration import carry
        corrections = root / "calibration-draft.json"
        report = json.loads((review_dir / "review-report.json").read_text(encoding="utf-8"))
        existing = json.loads(corrections.read_text(encoding="utf-8")) if corrections.is_file() else None
        if existing and existing.get("run_id") == report["run_id"]:
            log(f"3/4 draft calibration already bound to this run: {corrections}")
        else:
            log("3/4 checking camera + carrying reviewed landmarks as a DRAFT (no bounce decisions)")
            carry(args.corrections, review_dir, run_dir, corrections)
    else:
        log("3/4 no --corrections given: replay uses the four court.yaml corners only")

    from .shot_replay import build
    if complete(replay_dir):
        log(f"4/4 replay already complete: {replay_dir / 'replay.html'}")
    else:
        if replay_dir.exists():
            raise SystemExit(f"{replay_dir} is incomplete from an earlier attempt; move it aside and rerun")
        log(f"4/4 far-player crop poses, contacts, bodies, racquets -> {replay_dir}")
        build(run_dir, review_dir, replay_dir, args.court, corrections_path=corrections,
              pose_weights=args.refine_pose_weights or args.pose_weights, pose_cache=root / "pose-cache.json",
              contacts=True)
    log(f"Done. Open {replay_dir / 'replay.html'} (keep source.mp4 beside it).")


if __name__ == "__main__":
    main()
