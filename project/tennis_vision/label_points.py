"""Build an offline, keyboard-first page for labelling shots, bounces and points.

    python -m tennis_vision.label_points --replay <replay folder> --output <new folder> [--video <source video>]

The page (``label.html``) plays the run's review video (``source.mp4`` in the replay
or event-review folder, which starts at run frame 0) and exports a ground-truth
``labels.json`` in schema v1 (``docs/labels-schema.md``). Every exported frame is a
SOURCE-video frame (run frame + ``source_start_frame``).

Rules this builder and its page keep:

- The replay/run folder is only read. The output must be a NEW folder outside it;
  the page reads the video from the run folder by relative link and never writes it.
- Machine candidates are copied in as unconfirmed hints only. Nothing becomes a
  label without John's keypress, and labels leave the page only as an exported file.
- The binding (run id, source-video SHA-256, fps) comes from the run exactly as
  ``score_labels.load_run`` reads it, so the export validates with
  ``python -m tennis_vision.labels <file> --run <replay folder>``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .labels import BOUNCE_CALLS, COVERAGE_KINDS, HITTERS, POINT_SIDES, SCHEMA_VERSION, SHOT_TYPES
from .score_labels import load_run

PAGE_VERSION = "label-points-1"
# Small JSON files copied next to the page so the output folder documents which run
# data the hints came from. Large frame data (replay-data.json) is not duplicated.
COPIED_RUN_FILES = ("build-status.json", "replay-report.json", "reviewed-events.json", "shot-candidates.json",
                    "review-report.json", "event-candidates.json", "contact-candidates.json", "play-context.json")
HINT_LIMITATION = ("Machine candidates are unconfirmed pipeline hints shown to speed up navigation. They are not "
                   "labels, are never exported, and their presence or absence proves nothing.")


def _is_within(path: Path, folder: Path) -> bool:
    path, folder = path.resolve(), folder.resolve()
    return path == folder or folder in path.parents


def video_link(video: Path, output: Path) -> str:
    """URL the page uses for the video: relative to the page when possible."""
    try:
        relative = os.path.relpath(video.resolve(), output.resolve())
    except ValueError:  # different drive on Windows
        return video.resolve().as_uri()
    return quote(Path(relative).as_posix())


def check_video(video: Path, frames: int, fps: float) -> dict[str, Any]:
    """Confirm the review video is the run's frame-aligned clip (run frame 0 at t=0)."""
    if not video.is_file():
        return {"status": "missing", "path": str(video),
                "note": "source.mp4 was not found; the page asks for the file. Pick this run's review "
                        "source.mp4, which starts at run frame 0."}
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        opened = cap.isOpened()
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        video_fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    finally:
        cap.release()
    if not opened:
        raise ValueError(f"{video}: cannot be opened to check its frame alignment")
    if count != frames or abs(video_fps - fps) > 0.01:
        raise ValueError(f"{video}: has {count} frames at {video_fps:.3f} fps, but the run has {frames} frames at "
                         f"{fps:.3f} fps; labels made on it would be misaligned")
    return {"status": "aligned", "path": str(video), "frames": count, "fps": video_fps}


def page_data(run: dict[str, Any], video_src: str, video_status: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for kind, entries in (("hit", run["hits"]), ("bounce", run["bounces"])):
        for c in entries:
            candidates.append({"id": c["id"], "kind": kind, "source_frame": c["source_frame"],
                               "run_frame": c["run_frame"], "player_id": c.get("player_id"),
                               "shot_type": c.get("shot_type"), "review_status": c.get("review_status")})
    candidates.sort(key=lambda c: (c["source_frame"], c["kind"], c["id"]))
    return {
        "page_version": PAGE_VERSION,
        "labels_schema_version": SCHEMA_VERSION,
        "run_id": run["run_id"],
        "source_video_sha256": run["source_video_sha256"],
        "video_hash_basis": run["video_hash_basis"],
        "fps": run["fps"],
        "source_start_frame": run["source_start_frame"],
        "frames": run["frames"],
        "run_kind": run["kind"],
        "run_path": str(Path(run["path"]).resolve()),
        "video_src": video_src,
        "video_check": video_status,
        "shot_types": list(SHOT_TYPES),
        "hitters": list(HITTERS),
        "bounce_calls": list(BOUNCE_CALLS),
        "point_sides": list(POINT_SIDES),
        "coverage_kinds": list(COVERAGE_KINDS),
        "candidates": candidates,
        "candidate_limitation": HINT_LIMITATION,
    }


def render_page(data: dict[str, Any]) -> str:
    template = Path(__file__).with_name("label_points.html").read_text(encoding="utf-8")
    encoded = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return template.replace("__LABEL_DATA__", encoded, 1)


def build(replay: Path | str, output: Path | str, video: Path | str | None = None) -> dict[str, Any]:
    """Write ``label.html`` (+ copied run data) into a new ``output`` folder. Returns the page data."""
    replay, output = Path(replay), Path(output)
    run = load_run(replay, video=video)
    if run["kind"] == "longrun":
        raise ValueError(f"{replay}: a long-run folder has no frame-aligned review video; "
                         "pass a replay or event-review folder")
    folder = Path(run["path"])
    if output.exists():
        raise ValueError(f"{output}: already exists; choose a new folder so nothing is overwritten")
    for protected in {folder, replay}:
        if _is_within(output, protected):
            raise ValueError(f"{output}: is inside the run folder {protected}; the run folder is read-only here")
    source = folder / "source.mp4"
    video_status = check_video(source, run["frames"], run["fps"])
    data = page_data(run, video_link(source, output), video_status)

    output.mkdir(parents=True)
    run_copy = output / "run-data"
    run_copy.mkdir()
    for name in COPIED_RUN_FILES:
        if (folder / name).is_file():
            shutil.copy2(folder / name, run_copy / name)
    (output / "label-data.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    (output / "label.html").write_text(render_page(data), encoding="utf-8")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the offline point/shot labelling page for a replay.")
    parser.add_argument("--replay", type=Path, required=True,
                        help="Replay or event-review folder (with source.mp4); only read")
    parser.add_argument("--output", type=Path, required=True, help="New folder for label.html (must not exist)")
    parser.add_argument("--video", type=Path,
                        help="Original source video, hashed locally if the run does not record its SHA-256")
    args = parser.parse_args(argv)
    try:
        data = build(args.replay, args.output, args.video)
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}")
        return 1
    hits = sum(c["kind"] == "hit" for c in data["candidates"])
    print(f"Run {data['run_id'][:16]}... source frames {data['source_start_frame']}-"
          f"{data['source_start_frame'] + data['frames'] - 1} at {data['fps']} fps; "
          f"{hits} hit and {len(data['candidates']) - hits} bounce candidates copied as hints only.")
    if data["video_check"]["status"] != "aligned":
        print(data["video_check"]["note"])
    print(f"Open {(args.output / 'label.html').resolve()}")
    print(f"Validate an export with: python -m tennis_vision.labels <labels.json> --run {args.replay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
