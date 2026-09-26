# Ball continuity and event review

## Open the completed result

In this project folder, open **runs/rally-event-review-verified/review.html** in
Chrome or Edge. Keep the HTML, videos, and clips together. No server, model
download, cloud upload, or fresh inference is needed.

1. Select an event in the right-hand timeline. Use **Replay ±0.8s**, frame stepping,
   or the **Short clip** link. Toggle **Raw video** to see the original pixels.
2. Choose **Confirm**, **Reject**, or **Unsure**. Correct the type, frame, or player
   when needed. Changing a correction resets its decision to unreviewed.
3. Use **Add missed event here** if a real hit/bounce has no candidate. This starts
   as uncertain; human additions do not invent model geometry.
4. Click **Export review labels** and retain `tennis-review-labels.json`.
   Browser autosave is best-effort; **Import labels** restores an exported review.
   Imports must match this run's fingerprint. Imported labels merge by event ID.

The timeline is clip-relative (0–19.967s). The source starts at frame 4740 / 158s.
Both timestamps are displayed/exported. Playback is silent.

## What changed

- New offline module `tennis_vision.event_review`; the model pipeline is untouched.
- Uses only observed learned-model ball positions as event evidence. Experimental
  motion detections, predicted tracks, ambiguous multiple balls, and gap estimates
  cannot create hit/bounce candidates.
- Hit candidates combine a trajectory turn/speed change with player-scaled
  racquet/wrist proximity, tracked identity, and available body-relative swing
  movement. This replaces the old fixed 70-pixel contact radius in this review pass.
- Bounce candidates require an observed downward/upward turn, a sharper fit than
  a smooth arc, and a plausible conditional ground intersection using manual court
  calibration. Large detector jumps and obvious camera cuts are excluded.
- Brief gaps of at most three frames can be filled only when surrounding observed
  velocities agree. No extrapolation, smoothing of real observations, or filling
  around candidate impacts/player equipment. Estimates are orange X markers.
- Review exports are separate: original `events.jsonl`, `shots.json`, videos,
  and 3D fits are not overwritten or recomputed.

## Completed run: measured counts, not accuracy

600 frames at 30 fps; 390 detected ball positions are preserved exactly, **1**
missing frame is estimated, and 209 remain missing. Many gaps occur near equipment
or lack adequate surrounding observations, so they are deliberately not bridged.
The pass proposes 8 hit and 9 bounce candidates, all **unreviewed**. No far-player
hit was established by these conservative rules; absent candidates do not mean no
hit occurred. Review missed intervals before using any events for statistics.

These rules are heuristic, not a newly trained contact/bounce classifier. Scores
are ranking aids, not probabilities. A hit candidate can still be a hand catch,
ball toss, or ground bounce near a racquet. Single-view overlap does not prove
ball/string contact. Conditional bounce coordinates are in court metres from the
near-left doubles corner; they are valid only if a ground contact is confirmed.
Airborne ball XYZ and exact impact positions remain unknown.

User corrections are labels only. If the event type/frame is changed, the old
candidate geometry is **not** recalculated or asserted valid for the new label.
The next stage should use reviewed labels to evaluate/tune event detection before
adding shot classification or feeding confirmed bounces into a 3D estimator.

## Files in the completed folder

| File | Purpose |
| --- | --- |
| review.html | Local review interface |
| source.mp4 / review.mp4 | Raw and overlaid H.264 clips |
| clips/ | One short annotated video per candidate |
| event-candidates.json | Evidence, identity, source timestamps, conditional landing coordinates |
| ball-continuity.jsonl | Per-frame detected / estimated / missing status with provenance |
| gap-review.json | Why each gap was filled or left empty |
| review-report.json | Counts, limitations, source/run fingerprint |
| build-status.json | Successful completion versus partial/failed build |

`rally-event-review` is the initial development export; use the **verified** folder
above, which corrects short-clip end timestamps and verifies decoded frame counts.

## Generate another review (optional)

Run from this project folder. The output directory must not already exist.

```powershell
.\.venv-ball\Scripts\python.exe -m tennis_vision.event_review --run runs\rally-neural-ball --output runs\rally-event-review-next --court court.yaml --ffmpeg "C:\Users\John\miniconda3\envs\opencv-env\Library\bin\ffmpeg.exe"
```

This reuses cached player/racquet/ball observations. The local verification run took
about 35 seconds for this 20-second video; other hardware/videos will differ.
Progress bars include ETA for source checks, rendering, and candidate clips.

Dependencies: NumPy, OpenCV, PyYAML, tqdm, and FFmpeg with libx264. The supplied
`.venv-ball` and installed conda FFmpeg were used without changing global packages.
On another machine supply its FFmpeg path (or put FFmpeg on PATH). `--data-only`
skips video/HTML output but still checks the source and obvious cuts.

Inputs must include `summary.json`, contiguous frame-aligned `events.jsonl`, and an
accurate `source-offset.json`. Missing offsets are an error, not silently treated
as zero. A fixed camera and matching manual calibration are required for bounce
mapping. There is no automatic calibration transfer to a new camera.

## Verification

```powershell
.\.venv-ball\Scripts\python.exe -m unittest discover -s tests -q
node tests\test_event_review_ui.cjs
```

The JS test uses a fake DOM for label/export/import logic, not a real browser.
All 19 generated videos were fully decoded and their frame counts checked; original
observed ball coordinates were checked for exact preservation. Browser visual and
playback verification was blocked by the tool's local-file URL policy, so it still
needs a user check in a browser. No ground-truth accuracy assessment is claimed.
