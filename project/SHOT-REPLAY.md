# Shot classification and improved 3D replay

## Open the completed build

Open `runs/shot-replay-trajectories/replay.html` in Chrome or Edge. No detector rerun or
new dependencies are needed. This is a new self-contained browser replay; the
legacy Rerun viewer and its old candidate-based flight files are unchanged.

From PowerShell, anywhere:

```powershell
Invoke-Item -LiteralPath "C:\Users\John\Documents\Codex\2026-09-17\can-x20\outputs\tennis-vision-mvp\runs\shot-replay-trajectories\replay.html"
```

- Play, slow down, step frames, or scrub the video and virtual court together.
- The page opens on its first available 3D trajectory. Click **Play trajectory**
  to animate that segment, or choose another segment above the videos. Segment
  playback stops at its last frame so the landing remains visible.
- **Estimated preview + reviewed** shows cyan dashed, unvalidated paths alongside
  any reviewed-constraint estimates (gold solid, given priority). **Reviewed
  constraints only** hides unvalidated paths. **Hidden** hides the 3D ball only.
- **Full segment path** shows the active segment including its future portion;
  the moving ball/trail remain synchronized to the video. No segment is extended
  into unsupported intervals and separate segments are not joined.
- Drag the court to orbit; scroll to zoom. Reset/top-view buttons are available.
- Use **Follow far player** for a closer view. This changes the virtual camera,
  not the detections or the estimated body shape. Reset restores the whole court.
- Toggle observed image poses, 3D body wireframes, and foot-position smoothing.
- Click **Bounces & candidates** to seek an event. Unreviewed bounces are restored
  as dashed orange rings (conditional locations IF a bounce), plus source-frame
  markers. Rejected events stay hidden. Candidate display does not require labels.
- Read the per-frame status to distinguish observed poses, height placeholders,
  back-associated identities, and missing 2D ball detections. Turning off the pose
  overlay no longer hides the observed ball.
- Select a shot to inspect movement, the proposed type, and its evidence.
- Save/export shot-type corrections separately as `tennis-shot-corrections.json`.
  These are feedback, not model retraining or hit/bounce confirmations. Browser
  autosave is best-effort; retain your exported JSON.

## What is implemented

### Temporal shot classification

The new rule-based first pass can propose **serve, forehand, backhand, volley,
or unknown** from the surrounding player pose and event context:

- Serve: baseline position, overhead ball/arm evidence, preceding raised-arm
  pattern, and movement. A similar overhead smash can still be misclassified.
- Forehand/backhand: handedness plus the dominant wrist's side in the anatomical
  shoulder coordinate frame before impact. Screen-left/right alone is not used.
  Collapsed/side-on shoulders or conflicting evidence produce unknown.
- Volley: near-net position, compact swing, racquet observations, a preceding
  confirmed opponent hit, and well-observed incoming ball motion without a
  reviewed bounce or unresolved intervening event. Simply missing a bounce is
  **not** enough to label a volley. These remain candidates, not proof.
- Racquet-to-wrist votes suggest handedness only if enough votes agree. User
  settings can specify left/right, or explicitly keep it unknown.

This is **not a trained shot-classification model** and no accuracy percentage is
claimed. Forehand/backhand labels describe swing side; groundstroke versus volley
can still be unresolved. Underlying unconfirmed hits may be catches/tosses or
false events, so shot type and event validation remain separate fields.

### Feet-anchored replay

Player positions are recomputed from confidently observed ankle pixels and manual
court calibration. Missing ankles use an explicitly labelled box-bottom fallback.
The supported region extends 2.5 m outside the doubles sidelines and 6.5 m behind
baselines, so a far player standing behind the baseline is no longer excluded by
the old viewer's 1.2 m cutoff. Only the selected near/far identities are shown.

Optional three-frame median smoothing applies only across consecutive observations,
not gaps or cuts. Missing players and balls clear immediately; the viewer never
keeps a last-seen object alive forever.

Wireframe joints lie on a camera-facing vertical plane through the player's feet.
This **2.5D construction is not recovered joint depth or full 3D body orientation**.
Unsupported joints are omitted. Insufficient pose data uses a labelled height
placeholder. Feet are grounded by convention, not a measurement of jump height.

The previous 1.2-times-image-width focal prior shrank the far-side wireframe.
The new camera uses a bounded focal search against the four clicked court corners,
assuming square pixels, a centred principal point and zero lens distortion. Bad
fits, boundary optima and weakly constrained views are rejected. On this clip the
coordinate residual drops from 25.41 to 1.47 pixels; this is a fit to the supplied
corners, **not** independent 3D accuracy or measurement-grade camera calibration.

Delayed identities can be linked backward for up to three seconds from a stable
five-frame confirmed track. Only cached learned person detections qualify, with
adjacent-frame overlap/motion/scale, clothing appearance, court-half and ambiguity
gates. Recovery stops at a missing candidate or camera cut; no boxes or joints are
invented. Recovered associations have explicit provenance and are used only for
replay. Shot classification and ball-flight fitting still use the original log.

### Reviewed-bounce flight fitting

Only **confirmed** review events can anchor a flight. Each arc ends at a confirmed
bounce with a usable reviewed-frame ball pixel and starts at a preceding confirmed
hit or bounce. Corrected frames are resampled from the original observed ball log;
stale candidate coordinates are never reused after an edit.

The fitter pins the landing endpoint and solves a gravity-constrained trajectory
against camera rays. Two confirmed bounces pin both endpoints. It rejects poor
reprojection fits, weak coverage, long gaps, intervening unresolved impacts,
camera cuts, and implausible heights/speeds. Estimates never become observations.

Airborne XYZ remains **estimated**, conditional on an approximate fitted camera,
gravity-only dynamics, and court calibration. Drag/spin and lens
distortion are not modeled. Confirming the bounce event does not make its projected
court coordinates ground truth. No acceptable fit means no aerial ball is drawn;
airborne pixels are never placed on the ground as if they were landings.

### Unvalidated trajectory preview (new)

Preview mode is separate from the reviewed-flight system. It fits a gravity arc
to a trailing window of observed ball pixels, **conditional on a bounce candidate
being a real landing**. It does not mark that candidate confirmed or infer a hit
at the start of the window. No arbitrary-height ground traces or invented rally
connections are used just to make a continuous path appear.

The longest passing window is selected from at most two seconds before the
candidate. A preview needs at least ten observations, at least 80% observed-frame
coverage and a span of at least ten frames (one third of a second or longer at
higher frame rates). The shared fitter also rejects long dropouts, poor pixel
agreement, negative/implausible heights, excessive speed, behind-camera solutions
and badly conditioned geometry. Unresolved events bound windows; rejected events
are ignored. After a detected camera cut, previews require a new calibration.
Short model-filled gaps are flagged and never added to the observed 2D ball log.

These gates do **not** validate metric depth or prove the candidate was a bounce.
Depth is sensitive to focal length, distortion and event timing. The numerical
fitter is shared, but `preview-flight3d.json` / `ball_3d_preview` are separate from
`validated-flight3d.json` / `ball_3d`. Preview paths cannot become reviewed output
by changing a display setting. Reviewed-constraint estimates are still approximate.

## Current real-video result

The completed 600-frame package uses the existing `rally-neural-ball` observations:

- Both player positions present in all 600 frames. This includes two near and
  70 far frames recovered from existing detections before identity initialization.
- 1,072 player-frame wireframes (600 near / 472 far); 128 far-pose placeholders.
- Nine unreviewed bounce candidates are visible and clickable again.
- One serve candidate; seven unknown shot candidates.
- Handedness remains unknown: racquet/wrist votes were not sufficiently consistent.
- No exported event-review labels were found. All 17 input events are still
  unreviewed, so **zero reviewed-constraint flights** are fitted.
- Four **unvalidated partial previews** cover 83 of 600 frames. Clip-time ranges:
  2.97–4.07 s (frames 89–122), 4.10–4.97 s (123–149), 16.97–17.30 s (509–519),
  and 19.00–19.33 s (570–580). These are neither a complete rally reconstruction
  nor four classified shots. They end at possible, not confirmed, bounces.
- The existing `tennis-shot-corrections.json` contains eight shot-type decisions,
  not hit/bounce confirmations. It is preserved, along with the old replay.
  Browser corrections are package-specific; this new package does not import
  old shot-type feedback automatically.
- 390 observed ball frames and 210 missing ball frames, unchanged from the input.
  Missing far-side ball contacts and absent poses still need detector improvement
  or explicit human review; the replay repair does not solve those observations.

These are counts, not accuracy measurements. This stage does not improve or rerun
the original player/ball detector. The first 70 frames lacked a *selected identity*,
not raw person detections. Those cached detections now provide replay positions;
poses absent from the cache remain unavailable, rather than being fabricated.

## Enable reviewed flight constraints

1. Open the existing event review at `runs/rally-event-review-verified/review.html`.
2. Correct/confirm the relevant hit and bounce, including their frames. Reject
   false intervening candidates. Add a missed hit if it interrupts the flight.
3. Export `tennis-review-labels.json` (not the separate shot-type corrections file).
4. Rebuild from this project folder:

```powershell
.\Run-Shot-Replay.ps1 -Labels "C:\Users\John\Downloads\tennis-review-labels.json"
```

It prints a new timestamped output folder and does not overwrite previous runs.
Open that folder's `replay.html`. The cached analysis normally builds in seconds;
no neural-model run is performed. A reviewed interval may still be rejected if
its observed pixels do not support the physics fit; reasons appear in the page.

If you know a player's handedness, optionally pass `-NearHand left` or
`-NearHand right`, and similarly `-FarHand`. Do not assume a hand just to force
forehand/backhand labels. Leave the default `auto` when unknown.

Equivalent Python command (choose a fresh output directory):

```powershell
.\.venv-ball\Scripts\python.exe -m tennis_vision.shot_replay --run runs\rally-neural-ball --review runs\rally-event-review-verified --court court.yaml --labels "C:\Users\John\Downloads\tennis-review-labels.json" --output runs\shot-replay-reviewed
```

The review fingerprint must match the source log and court calibration. A confirmed
bounce with no detected ball at the corrected frame cannot silently borrow an
estimated position. Advanced labels may supply an explicit reviewed `landing_pixel:
[x, y]`; otherwise that constraint is skipped with a reason.

## Outputs and verification

The new package contains `replay.html`, the unchanged H.264 `source.mp4`,
`replay-data.json`, `shot-candidates.json`, `reviewed-events.json`,
`validated-flight3d.json`, `preview-flight3d.json`, `replay-report.json`, and `build-status.json`.
Original detections, event candidates, review labels, and legacy flights are preserved.

97 Python tests pass, including mirrored handedness, all four shot types, abstention,
strict review/run matching, corrected-frame geometry, missing-object clearing,
known synthetic ballistic recovery, and an end-to-end export with/without confirmed
synthetic events. Regression coverage includes candidate-only bounce geometry,
known-focal camera recovery, boundary/degenerate camera rejection, and conservative
identity backfill (bystanders, ambiguity, missing detections, cuts and lookback limits),
and preview isolation, rejected/conflicting events, missing endpoints, short/long
gaps and camera cuts. JS tests exercise preview/reviewed/hidden modes, segment paths,
endpoint playback stopping, controls, orbit/far-follow math, candidate visibility,
synchronization and feedback export with a fake DOM. Offline Canvas draw-command
geometry and six
source-frame crops of recovered far-player detections were inspected. Original
shot results and per-frame observed ball coordinates were checked unchanged.
Actual browser layout/playback was not visually verified; no real-data accuracy
validation or confirmed real-video flight is claimed.

```powershell
.\.venv-ball\Scripts\python.exe -m unittest discover -s tests -q
node tests\test_shot_replay_ui.cjs
```
