# Technical handoff: tennis vision

## Read this first

John's current priority is a longer game-length test, then visible 3D player
bodies and racquets. This package preserves a functioning 20-second baseline.
It does NOT implement those new features. Work only in this copied project and
new output folders. The source project is not a Git repository; consider making
a local initial commit in the copy if useful, without publishing it.

Current entry point: `project/runs/raw-play-ready/replay.html`.
Current behavior notes: `project/RAW-PLAY.md`, then `PLAYER-SHOTS.md` and
`COURT-BOUNCE-CALIBRATION.md`. Older docs describe earlier milestones; the raw-play
replay and this handoff identify the latest state.

## Baseline and reviewed state

- 600 frames, 30 fps, 1280x720, 20 seconds. Original source starts at frame 4740
  (158 seconds) in `sebbie-demo-shortest (1).mp4`.
- Original observations: `runs/rally-neural-ball/events.jsonl`; metadata:
  `summary.json`, `source-offset.json`, `ball-model.json` beside it.
- Source clip and event proposals: `runs/rally-event-review-verified/`.
- Review run ID:
  `39449302177328c739e679a953b51d001bf297e21541feb1cc38c76265aa9ab9`.
- Reviewed camera and bounce corrections:
  `runs/court-bounce-20260925-110335-545/calibration-corrections.json`.
  This folder also includes the calibration/timing desk and its frame images.
- Camera has 15 reviewed landmarks and fitted radial distortion. Approximate
  1.50-pixel fitting residual is NOT independent localization accuracy.
- Bounce `bounce_candidate-000122`: confirmed, corrected to frame 121, range
  [120,122]. Serve bounce `bounce_candidate-000519`: timing corrected to frame
  518, range [517,519], but still unreviewed. Do not auto-confirm it.
- Original `court.yaml` is part of the run fingerprint. Do not overwrite it with
  refined camera values or edit the historic `summary.json` input path: those
  changes invalidate review bindings. Preserve original metadata bytes even
  when their absolute source paths refer to John's original Windows machine.
- Pose cache: `runs/pose-cache-DD79B09788E82900.json`; exact checkpoint hash is
  `ad33da8a29ea5772318c4c980844e47b56792d2b63815ad4e8e09c078c7d1abf`.
- Current shot result: 1 serve candidate, 3 likely non-play activities, 7 uncertain
  contacts (including both far-player windows). No automatically confirmed hit.
- One confirmed bounce; no supported reviewed flight segment in this clip.
  Dashed cyan paths are separately labeled UNVALIDATED estimates, not measured
  airborne XYZ or validated bounce constraints.

The far-player crop pass replaced 70 poses; frames with >=6 supported body
joints improved from 472 to 497, supported body joints from 5476 to 5826. These
are coverage counts, not precision/accuracy. Near and far roots display in all
600 frames, including explicitly estimated identity recovery. Across both
players there are 1097 pose wireframes and 103 placeholders. CPU crop inference
took about 8.5 minutes for this 600-frame example; do not promise a game ETA by
linear extrapolation without measuring the chosen environment.

## Architecture map

| Area | Primary code | Important contract |
| --- | --- | --- |
| Raw inference | cli.py, detail.py, scene.py, filters.py, tracking.py | Court/identity filters must not reacquire referee, towel, adjacent-court players |
| Ball model | temporal_ball.py, setup_temporal.py | Learned five-frame GridTrackNet; preserve RGB/normalization/source labels |
| Human event review | event_analysis.py, event_review.py | Bind labels to exact run and frames; proposals are not truth |
| Court/bounce review | camera3d.py, court_refinement.py, calibration_review.py/.html | Distorted pixel-to-court mapping, immutable reviewed provenance |
| Far-player recovery | cached_identity.py, player_refinement.py | Identity-anchored crops; recovery/interpolation is not observed joint evidence |
| Contact/play | contact_detection.py, play_context.py | Continuous observed incoming/outgoing context; gaps/cuts stop evidence |
| Stroke execution | stroke_execution.py, shot_classification.py | Per-stroke left/right/both independent of forehand/backhand and player profile |
| Replay builder/UI | shot_replay.py, shot_replay.html | Portable HTML, video synchronization and review exports |
| Ball XYZ | validated_flight.py, preview_flight.py | Reviewed-constrained estimates and unvalidated previews remain separate |

All code in the table is under `project/tennis_vision/`.

Current renderer is a custom Canvas projection, NOT Three.js or a rigged mesh
renderer. Wireframes are a 2.5D lift into a camera-facing vertical plane, not
measured 3D human anatomy. Court-world coordinates are metres, origin at court
centre, Z up, X across the court, Y toward the far baseline. Older court-floor
coordinates use a near-left origin: subtract [5.485,11.885] when converting to
centred XY. Player floor roots use feet where supported, with labeled bbox-base
fallbacks; airborne feet are not automatically valid floor intersections.

Handedness fields are descriptive only. Per-stroke striking_hands supports
unknown/left/right/both; execution supports one_handed_candidate,
two_handed_candidate, unknown. Two nearby hands do not prove grip or backhand.
Human stroke review is separate from human contact confirmation. If contact
timing/identity changes, apply it first and export stroke feedback from the new
replay, because stale analysis bindings are rejected.

## Setup and regression checks

No virtual environments are bundled. For tests and cached replay work, create
an environment with Python 3.11+ (Windows baseline tested on 3.13.2) and run from
`project/`:

```text
python -m pip install -r requirements-replay.txt
python -m unittest discover -s tests -q
node tests/test_shot_replay_ui.cjs
node tests/test_player_shots_ui.cjs runs/raw-play-ready/replay.html
node tests/test_calibration_review_ui.cjs runs/court-bounce-ready/calibration.html
node tests/test_event_review_ui.cjs
```

Use your new environment's Python, not an arbitrary older system installation.
Before packaging: 125 Python tests and all four JavaScript suites passed.
JavaScript suites use fake DOMs: they are logic checks, NOT browser-rendering
verification. `PACKAGE_VERIFICATION.md` records checks performed on the copy.

`requirements-replay.txt` is an added minimal convenience file in this copy.
Original full `requirements.txt` and `requirements-temporal.txt` remain intact.
The lightweight tested environment used NumPy 2.5.3, OpenCV 4.13.0.92,
PyYAML 6.0.3, tqdm 4.67.3; temporal tools used ONNX 1.20.1,
ONNX Runtime 1.24.4, h5py 3.15.1. These are observed versions, not a portable
fully resolved lockfile. Node baseline was 24.19.0.

For fresh YOLO inference, install compatible Ultralytics and PyTorch for the
target OS/device. The existing local inference environment used Ultralytics
8.4.163 and CPU Torch 2.14.0/torchvision 0.29.0; it inherited system packages
and is deliberately not copied. FFmpeg must be available for review video
building. Avoid silently substituting weights or downloading new models.

### Rebuild the existing replay without rerunning detections

With the models archive extracted, execute as ONE command from `project/`:

```text
python -m tennis_vision.shot_replay --run runs/rally-neural-ball --review runs/rally-event-review-verified --court court.yaml --corrections runs/court-bounce-20260925-110335-545/calibration-corrections.json --pose-weights yolo26l-pose.pt --pose-cache runs/pose-cache-DD79B09788E82900.json --contacts --output runs/claude-baseline-rebuild
```

Choose a fresh output name for subsequent builds. Cached rebuild still requires
the exact local pose checkpoint to verify its hash. The review folder supplies
the cropped source video; old absolute raw-source metadata need not be edited.
If cache validation fails, investigate provenance before launching inference.

Existing Run-Player-Shots.ps1 and other wrappers are sample-specific and expect
local `.venv-*` paths. Generalizing them is part of the next task. They are NOT
already a portable arbitrary-game workflow. Inspect `python -m tennis_vision.cli
--help` and builder help for current flags before designing new wrappers.

## Longer footage supplied

| File in project/media | Frames | FPS | Duration | Resolution |
| --- | ---: | ---: | ---: | --- |
| sebbie-demo-shortest (1).mp4 | 5654 | 30 | 188.467 s | 1280x720 |
| sebbie-demo-shortestpar2.mp4 | 5549 | 30 | 184.967 s | 1280x720 |

Metadata was read from the files; neither is verified to contain a complete
game. Hashes and original source paths are recorded in media/media-index.json.
The original baseline excerpt comes from the first file. Do not concatenate
them or apply the same calibration to the second camera view without checking.

### Stage 1: safe longer-run pipeline

- Accept input video, start/end or full-video selection, court/scene config,
  output root and model paths explicitly. Leave original runs untouched.
- Preserve source-frame offsets across every stage and temporal-model overlap.
  Test first/last frames and chunk boundaries for duplicates, omissions or drift.
- Add resumable, fingerprinted chunk outputs and bounded-memory processing;
  preserve five-frame ball-model context. Show frame progress, elapsed time and
  measured ETA. Do not count padded/estimated frames as observations.
- Camera changes/cuts invalidate calibration and identity assumptions. Reuse
  landmarks only after validating camera/resolution; generate new run bindings.
  Never apply old 600-frame event IDs directly to a new full-video run.
- Handle idle walking, ball handling and missing evidence without hallucinated
  shots, returns, ace/fault/winner labels or automatic game scoring.
- Compare the baseline and longer sample. Report far-player/ball coverage and
  manually checked errors separately; claim accuracy only against labeled truth.

### Stage 2: 3D player bodies

- Start with articulated body meshes (e.g. segmented capsule/limb geometry),
  then a licensed rigged glTF model if useful. Do not call thicker projected
  2D lines a 3D mesh upgrade. Rendering realism does not establish 3D accuracy.
- Make an explicit renderer-adapter/migration decision. If using Three.js,
  vendor permitted dependencies for local/offline operation; avoid an unexpected
  CDN requirement or sending footage externally. Keep current review controls.
- Anchor meshes to existing player roots and court scale; maintain near/far
  identity, seeking, source sync, pause, speed, follow camera and wireframe toggle.
- Handle missing joints and occlusions without calling animation completion
  measured pose. Clearly label depth, orientation and inferred limb movement.
  Respect per-stroke hands; no permanent right-handed skeleton assumption.
- Verify mirrored sides, Z-up transforms, foot sliding, missing-data visibility,
  timeline rewinds and performance on a longer run.

### Stage 3: 3D racquets

- Associate each racquet with equipment/player evidence and supported wrist(s),
  not nearest bystander. Current bounding boxes do not measure racquet face
  orientation, handle endpoints, true grip, or string-bed contact.
- Use a racquet mesh with plausible scale. When orientation or grip is inferred,
  expose that fact and an optional visibility toggle; hide or label stale poses.
- Support left, right and both hands and hand switching. Test two-handed
  forehands and backhands on either side independently of handedness profiles.
- Rendered racquet paths must NOT become contact ground truth or invent contact
  events across missing observations. No spin RPM claims from these inputs.

## Assets, provenance and return requirements

Models: yolo11x.pt, yolo26l-pose.pt, models/gridtracknet/gridtracknet.onnx;
the numeric source model_weights.h5 and manifest are included in the models ZIP.
GridTrackNet source is https://github.com/VKorpelshoek/GridTrackNet at revision
0764162b73fb64d440fd9e6c363d592965400799. See THIRD-PARTY-NOTICES.md for its notice.
Ultralytics local package metadata indicated AGPL-3.0. Preserve provenance and
check applicable model/dependency/mesh licenses before redistribution or
deployment; this package does not grant new rights or relicense the assets.

The main ZIP includes one legacy shot replay HTML and one legacy calibration
HTML as regression fixtures; they are not the recommended current entry points.
The original detector annotated video and obsolete run history are not included.
No credentials, virtual environments or machine inference settings are copied.

Before returning: run all regressions, add tests for changed contracts, visually
inspect actual replay rendering if your environment permits, and record exactly
what was NOT verified. Provide new output names, runnable commands, observed
performance, dependency/asset provenance and completed RETURN_TO_CODEX.md.
Preserve all original human decisions unless John explicitly reviews changes.
