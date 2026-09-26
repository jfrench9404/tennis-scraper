# tennis-scraper

Local tennis-video analysis: player and ball tracking, human-reviewed court
calibration, contact and shot candidates, and a portable HTML virtual replay.

## Layout

| Path | What it is |
| --- | --- |
| `START_HERE.md`, `CLAUDE_HANDOFF.md`, `CLAUDE_PROMPT.md` | 2026-09-25 handoff: current state, constraints and next priorities |
| `RETURN_TO_CODEX.md` | Report to fill in when work goes back to Codex |
| `MANIFEST.json` | SHA-256 inventory of the full handoff package, including assets not in git |
| `project/` | The Python package (`tennis_vision/`), tests, docs and reviewed run outputs |
| `project/runs/raw-play-ready/replay.html` | Current working replay (20-second baseline) |

Setup and regression checks are in `CLAUDE_HANDOFF.md` ("Setup and regression checks").

## What is not in git

These stay on the local machine only and are listed with SHA-256 in `MANIFEST.json`:

- **Footage and anything cut from it**: `project/media/*.mp4`, every `source.mp4`,
  `review.mp4`, review clips, timing-frame JPGs and calibration stills.
- **Model weights**: `yolo11x.pt` (114 MB, over GitHub's 100 MB file limit),
  `yolo26l-pose.pt`, `models/gridtracknet/*.onnx|*.h5`. They carry their own
  licenses (see `project/THIRD-PARTY-NOTICES.md`).

A fresh clone can run the unit tests. Replays need the local `source.mp4` files
beside them to show video, and rebuilding needs the weights.

`.gitattributes` disables line-ending conversion. `court.yaml`, run metadata and
review JSON are hashed into review fingerprints, so they must stay byte-identical.
