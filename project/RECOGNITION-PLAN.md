# Recognition work in manageable stages

## Newer upgrade: trained five-frame ball inference

The motion-only work below is now followed by the optional GridTrackNet integration.
See [TEMPORAL-BALL.md](TEMPORAL-BALL.md) and watch
`runs/gridtracknet-court-review/review.mp4`. It runs a pretrained tennis model on
new image sequences, not cached ball detections, while retaining cached players
for the comparison. It is not fine-tuned on this camera and is not complete tracking.

## Latest verified review: continuity and motion-assisted ball search

Watch `runs/temporal-review-2/review.mp4` now; no new model run is needed. It reuses
all 600 frames of saved candidates from `rally-confirmed` and reads original pixels.
The older cache lacks rejected pose keypoints, so this is not identical to a fresh
inference run. Review overlays are boxes rather than full skeletons; the normal
analyzer still exports and draws pose keypoints.

Observed frame counts (not accuracy): near player 529 -> 598; far player 296 -> 530.
The far identity is continuous after first confirmation at frame 70. Model-backed
ball presence rises from 277 to 316 frames; another 49 have tentative motion evidence.
The ball is still absent in 235 frames. Some gaps occur around contact or while held;
others are genuinely missed visible balls. This is not complete ball recognition.

Changes: nearby overlapping player detections are matched by continuity even when
clothing appearance changes. An adaptive recent colour reference and median-height
penalty reduce pose-box drift; distant reacquisition still needs racquet confirmation.
Ball search extends a short distance around confirmed players making wide retrievals,
but never overrides explicit towel/background exclusions.

Optional `--motion-ball` searches for small moving bright/coloured components, using
recent ball observations or three-frame trajectory consistency. It excludes all
detected people/racquets, including unselected bystanders, and resets on large image
changes. No pixels means no ball output. Orange `source: motion` boxes are tentative;
their 0.25 score is a heuristic, not a calibrated probability. They are excluded from
bounce/contact events, physics fitting, and the 3D ball replay. Recovery can miss
balls near players/equipment and can still mistake background motion for balls. It
is experimental, off by default, and intended for this fixed-camera view, not pans.

To replay cached detections into a NEW directory (OpenCV + NumPy only):

```powershell
python -m tennis_vision.replay_cached --run runs\rally-confirmed --output runs\my-temporal-review --scene sebbie-scene.json --motion-ball
```

Review outputs deliberately contain no new shots or calibrated coordinates. For a
fresh full analysis in your existing model environment:

```powershell
python -m tennis_vision --input "C:\Users\John\Downloads\sebbie-demo-shortest (1).mp4" --output runs\rally-temporal --court court.yaml --scene sebbie-scene.json --start-seconds 158 --max-frames 600 --motion-ball
```

Remaining ball misses call for evaluating a tennis-specific temporal detector on
labelled frames from this camera. Neither continuity heuristics nor generic model
confidence can establish ball-location accuracy without that validation.

## 1. Court and background selection — implemented

Optional `--scene sebbie-scene.json` provides normalized image polygons for this
fixed camera. A ball search polygon excludes most neighbouring courts, while player
polygons are used only to initialize identities. Yellow/green pixel evidence rejects
white objects such as towels, and a short stationary-candidate test rejects persistent
background detections. These heuristics can also reject a motion-blurred, washed-out,
held, or stationary real ball. Set `yellow_ball_check` false to disable colour gating.
No model training or measured accuracy claim is implied by these filters. This
camera-specific configuration also excludes the small known towel area at the back
wall. A real ball crossing that patch will intentionally be unobserved there; remove
the exclusion if the camera or towel moves. Colour alone did not reject all towel
detections, so the explicit patch is necessary for this view.

## 2. Main player identity — first implementation

For singles, three nearby detections in an initial near/far region plus an associated
racquet observation establish a player reference. Torso/shorts colour histograms plus recent location match future observed
detections, including outside the seed region. `identity_id` stays near/far even if a
short-lived tracking number changes. These labels mean initial identity, not current
side. Up to five missed frames may show an explicitly predicted box, without stale
joints; longer gaps have no fabricated player position. Similar uniforms, camera cuts,
lighting changes, prolonged absences, and crossings remain failure cases. Equipment
colour is not used yet; racquets remain associated by wrist proximity. This is not a
trained re-identification model. A scale-aware movement gate limits identity jumps.
After more than five missed frames, reacquisition needs three consistent observations
and racquet evidence again. Continuous tracking does not require a visible racquet.
If the detector never sees a racquet, confirmation remains pending rather than
silently choosing a bystander. `require_racket_confirmation: false` in scene JSON
disables the racquet requirement, but leaves motion/appearance checks in place.
Wrong automatic initialization requires a new run
with adjusted seed polygons; this version has no click-to-select UI.

## 3. Distant-player and ball recognition — implemented, inference pending

Overlapping smaller far-court crops replace a single wide crop. Both general person
boxes and pose results are accepted; a missing skeleton no longer requires discarding
the detected person. Pose results are preferred when merging overlapping boxes.
Far crops now retain racquet detections as well as players and balls. The last-ball
crop is targeted from scene-accepted detections, so a rejected towel cannot keep
steering crop searches. Its search anchor survives two missed frames and expires
on the third; it never generates a synthetic ball observation. Colour gating now
allows weaker saturation and 3% yellow/green coverage (configurable with
`ball_yellow_fraction`) to accommodate compressed, pale balls. White objects still
fail the colour test, but coloured background false positives remain possible.
Pre-filter boxes/confidences/keypoints are saved in `events.jsonl` under `detail.candidates`
to distinguish detector misses from filter rejections. This still uses generic pretrained detectors;
tennis-specific temporal ball training is the next step if recall remains low.

## 4. Court coordinates and bounce validation

`feet_xyz_m` now exports ankle (or bounding-box fallback) positions in centred court
coordinates, with z=0 because feet are assumed on the court plane. This is not measured
joint height, and jumping players violate that assumption. Image keypoints remain 2D.
`position_basis` states the assumption. `court_m` for a ball remains a ground-plane ray
intersection for compatibility, not the position under an airborne ball.

Bounce candidates and physics fits are still experimental. Next work: verify bounce
frames against labelled rally examples, localize the ball at impact, and use those
anchors to constrain flight fits. Swing paths alone do not uniquely determine ball
depth or height from a single camera. Do not treat current bounce candidates as ground
truth or low reprojection error as proof of 3D accuracy.

## Run this same rally once the update is installed

```powershell
python -m tennis_vision --input "C:\Users\John\Downloads\sebbie-demo-shortest (1).mp4" --output runs\rally-temporal --court court.yaml --scene sebbie-scene.json --start-seconds 158 --max-frames 600 --motion-ball
```

Scene selection is enabled with manual court calibration; `--all-players` disables
identity selection and scene ball filtering for comparison. Crop passes cost extra
inference time. New `detail.scene` counts describe why detections were rejected.
Existing run outputs are preserved. The scene configuration is specific to this
camera view; do not reuse it after a camera cut or on another court.
