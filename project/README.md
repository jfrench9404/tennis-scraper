# Tennis Vision MVP

> **Latest (2026-09-26):** long-footage pipeline, 3D body meshes and racquets. Start with [CLAUDE-OVERNIGHT.md](CLAUDE-OVERNIGHT.md).

**Newest: raw-footage play context and per-stroke hand use.** Open
`runs/raw-play-ready/replay.html`. Shot candidates, uncertain contacts and likely
non-play are separated; left/right/two-handed stroke execution is independent of
player handedness. See [RAW-PLAY.md](RAW-PLAY.md) for review/apply commands and limits.

**Far-player crop pose + contact/shot candidates:** run `Run-Player-Shots.ps1`.
Already built: `runs/player-shots-final/replay.html` (open this to avoid rerunning).
This preserves your reviewed court/bounce run and produces a separate replay with
fresh focused pose observations, contact evidence, and an enlarged far-player view.
See [PLAYER-SHOTS.md](PLAYER-SHOTS.md) for controls, handedness settings and limits.

**Newest: lens-aware court calibration and frame-level bounce review.** Open
`runs/court-bounce-ready/calibration.html` for the prefilled 15-point court and
decoded bounce frames. It links to the corrected 3D replay. The serve's proposed
bounce is frame 518 rather than 519; it remains unreviewed. See
[COURT-BOUNCE-CALIBRATION.md](COURT-BOUNCE-CALIBRATION.md) for the ready result,
export/apply command and uncertainty limits. Original files are unchanged.

**Latest: shot-type candidates and feet-anchored 3D replay.** Open
`runs/shot-replay-trajectories/replay.html`. Source video, observed pose overlay, and the
orbitable virtual court share one timeline. Confirmed event-review labels can
constrain reviewed flight estimates. A separate **estimated preview mode** now
shows four partial 3D paths from candidate bounce hypotheses, with a moving cyan
ball, dashed paths and a Play trajectory button. These are unvalidated and never
become confirmed events or reviewed-flight data. Unsupported intervals stay empty.
The update also retains delayed-player recovery and the improved camera fit. See
[SHOT-REPLAY.md](SHOT-REPLAY.md) for the ready result, commands, and limitations.

**New: ball continuity + hit/bounce review.** Open
`runs/rally-event-review-verified/review.html` in Chrome or Edge. The completed
20-second package reuses `rally-neural-ball`; no model run is needed. It includes
raw/annotated playback, frame stepping, 17 candidate clips, gap inspection,
and editable/exportable human review labels. See [EVENT-REVIEW.md](EVENT-REVIEW.md).
Estimates remain separate from detections. Candidate counts are not accuracy.

The new **five-frame neural tennis ball detector** is installed and has a completed
review at `runs/gridtracknet-court-review/review.mp4`. See
[TEMPORAL-BALL.md](TEMPORAL-BALL.md) for results, limitations, the one-command review,
and enabling `--temporal-ball` in the main analyzer. Existing YOLO/pose operation
is unchanged unless that option is selected.

The latest scene-selection update is documented in [RECOGNITION-PLAN.md](RECOGNITION-PLAN.md).
For the supplied fixed-camera rally, add `--scene sebbie-scene.json` to enable the
tuned seed/search areas and known towel exclusion. Clothing-based near/far identity
labels persist outside the initial selection areas. Far-court inference now uses
smaller overlapping crops plus a person-box fallback if pose joints are unavailable.
The latest continuity update has been replayed against saved raw candidates in
`runs/temporal-review-2/review.mp4`; it does not require rerunning the neural models
to watch. Counts in its report measure coverage, not ground-truth accuracy.
Optional `--motion-ball` adds experimental, orange-labelled ball observations from
moving pixels. These are excluded from events and 3D fits. Full instructions and
the remaining limitations are in `RECOGNITION-PLAN.md`.

## Court and racquet filtering

New analyses automatically filter player feet against the manually calibrated court,
allowing 1.5 m outside doubles sidelines and 6 m behind baselines. Configure those
allowances with `--court-side-margin` and `--court-baseline-margin`. Players inside
this region are candidates, not verified match participants. Without a manual
calibration, court rejection is disabled because the automatic estimate is unreliable.

Overlapping racquet boxes are suppressed. Remaining boxes are matched to visible
players' confident wrists, with a stricter body-box fallback when wrists are missing.
At most one racquet is retained per observed player. Exported racquets include
`player_track_id`. Missing player detections can therefore also hide their racquets.
Airborne balls are not rejected using the court-plane projection.

For a fresh short test from this folder:

```powershell
python -m tennis_vision --input "C:\Users\John\Downloads\sebbie-demo-shortest (1).mp4" --output runs\court-filter-check --court court.yaml --max-frames 600
```

The separate `outputs/court-racket-filter-preview` deliverable re-filters the old
600-frame log without running models; its video and counts illustrate the filter
effect, not a measured increase in model accuracy. Original runs are preserved.

An offline, authorized-footage analysis pipeline for tennis broadcasts and training clips. It produces a time-stamped event log and an annotated MP4; it does not acquire or bypass livestreams.

## What this starter does

- reads a local video;
- runs a dedicated pose model and writes 17 COCO body joints per detected player; player court position uses detected ankles rather than a bounding-box edge;
- detects `person`, `sports ball`, and (with a custom checkpoint) `racket` objects;
- tracks each object through frames using class-aware nearest-neighbour tracking;
- automatically estimates the visible court and projects locations onto a 23.77 m × 10.97 m doubles court; manual coordinates remain an optional accuracy override;
- emits contact and bounce *candidates*, including court-plane locations when calibrated;
- writes JSONL data and a readable overlay video.

The bundled generic YOLO checkpoint normally supplies people and balls only. Accurate racket detection and shot classification require a fine-tuned checkpoint and labelled clips. The program deliberately marks uncertain inferences as candidates rather than presenting them as ground truth.

## Setup

Use an isolated environment, then install the pinned packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Before the first pose run, upgrade Ultralytics so it can resolve the current YOLO 26 pose weights:

```powershell
pip install --upgrade ultralytics
```

## Run

Run without court coordinates. It automatically tries to establish player/ball tracks and court geometry, then writes overlays:

```powershell
python -m tennis_vision `
  --input "C:\Users\John\Downloads\sebbie-demo-shortestpar2.mp4" `
  --output runs\sebbie-baseline
```

The terminal shows processed frames, rate, elapsed time, and estimated time remaining. Press `Ctrl+C` to stop safely; use a new output folder when restarting a cancelled run.

If the automatic court estimate reports low confidence, copy `court.example.yaml` to `court.yaml` and replace its four image coordinates with the visible doubles-court corners in this exact order: near-left, near-right, far-right, far-left. Then add `--court court.yaml`. This is an optional correction path, not a setup requirement.

For a racket-aware model (recommended), pass a custom Ultralytics checkpoint that contains the class names `person`, `sports ball`/`tennis_ball`, and `racket`/`tennis_racket`:

```powershell
python -m tennis_vision --input <video> --weights models\tennis.pt --court court.yaml
```

The default pose model downloads automatically on first run. It supplies wrists, elbows, shoulders, hips, knees, and ankles. The generic COCO detector is not a dependable tennis-racket or tennis-ball detector; for match-quality racket/ball tracks, pass purpose-trained checkpoints:

```powershell
python -m tennis_vision --input <video> --weights models\tennis-ball.pt --racket-weights models\tennis-racket.pt --court court.yaml
```

Short detector dropouts are filled with motion predictions: these remain visible in the annotated video and are written with `predicted: true`, but are excluded from contact/bounce event detection. This helps maintain continuous player/ball visualization without turning guesses into analytics.

Outputs:

- `events.jsonl` — frame-by-frame tracks plus event candidates;
- `shots.json` — deduplicated contacts/bounces and contact-to-contact shot intervals;
- `summary.json` — dimensions, counts, and assumptions;
- `annotated.mp4` — source frames with labels, tracks, and event markers.

To consolidate a previously created event log without rerunning detection:

```powershell
python -m tennis_vision.postprocess --events <events.jsonl> --output shots.json
```

## Browser interface / deployment

Start a local browser app with no coding interaction required:

```powershell
streamlit run app.py
```

To host it online, push this folder to a private GitHub repository and deploy `app.py` on Streamlit Community Cloud, Render, or another Python host with adequate CPU/RAM. Do not upload match footage to a third party unless you have the rights and their data policy is acceptable for that footage.

## 3D virtual-court replay

Open the virtual replay from any completed analysis:

```powershell
python -m tennis_vision.replay3d --events runs\sebbie-baseline\events.jsonl --flight runs\sebbie-baseline\flight3d.json
```

It opens an interactive Rerun viewer: orbit, pan, zoom, and scrub the frame timeline while seeing the regulation court, net, player-foot projections, ball ground projection/trail, and detected contact/bounce anchors. Save a portable replay instead with `--save replay.rrd`.

Player avatars and balls are shown only when their court projection is plausibly on or near the court. If the replay opens with only a court, inspect `summary.json`: a low `court_confidence` means the automatic court estimate was not usable for that camera. Supply `--court court.yaml` on analysis with four visible court corners for that clip, then rerun analysis and the replay. This calibration override is required for meaningful 3D placement when the automatic estimate fails.

Create `court.yaml` without editing coordinates by running:

```powershell
python -m tennis_vision.calibrate_court --input "C:\path\to\match.mp4" --output court.yaml --frame 300
```

Click the four labelled corners once, press Enter, and reuse that file for every clip from the same fixed camera.

The analysis now also produces `flight3d.json`: an automatic, gravity-constrained monocular fit between each consolidated contact and bounce. White trails in the replay are these fitted arcs; cyan trails remain the raw ground projection. The camera focal length is inferred from a broadcast-camera prior, so treat `fit_confidence` and reprojection error as mandatory quality checks—not as measurement-grade ball data.

## Assessment of the supplied clip

The local environment can see `sebbie-demo-shortestpar2.mp4` (about 58 MB), but lacks a media probe/decoder package, so I could not extract representative frames here without changing the environment. Run the baseline command after setup: `summary.json` will record the actual frame rate, resolution, and event counts. For dependable ball-contact detection, use the highest-resolution original available—60 fps or more is strongly preferred.

## Accuracy roadmap

1. Manually label 500–2,000 frames for ball, racket, player ID, contact, bounce, and shot type.
2. Fine-tune a detector for ball and racket, then replace `NearestTracker` with ByteTrack/BoT-SORT.
3. Add a pose model (RTMPose or YOLO pose) and map wrist/elbow/shoulder trajectories into the event classifier.
4. Train a temporal classifier on a 0.8–1.5 second window around contact to distinguish serve, forehand, backhand, volley, overhead, slice, and drop shot.
5. Use a second synchronized camera for 3D ball flight; a single broadcast camera can only provide court-plane estimates.

## Coordinate convention

Court coordinates are metres, origin at the near-left doubles corner. `x` moves left-to-right along the near baseline; `y` moves from near to far. Image-to-court mapping is a planar homography, appropriate for player feet, bounce locations, and contact projected to the court—not for the airborne ball’s true 3D position.
