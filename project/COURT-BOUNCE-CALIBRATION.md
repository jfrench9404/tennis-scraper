# Court calibration and frame-level bounce timing

## Open the ready draft

```powershell
Invoke-Item -LiteralPath "C:\Users\John\Documents\Codex\2026-09-17\can-x20\outputs\tennis-vision-mvp\runs\court-bounce-ready\calibration.html"
```

The desk links to its **calibrated 3D replay**. No model run is needed.
The original `court.yaml`, tracking log, event-review files and previous replays
are unchanged. This is a new, approximate calibration and timing proposal.

1. Check that the green lines follow the painted court lines. Toggle the original
   orange four-corner projection to compare. Fifteen points are already filled in;
   you do not need to click them all again. If a point is wrong, select its name
   and click the actual line intersection. Do not select the elevated net tape.
2. Select the alley bounce (frame **122**) and the serve (proposed frame **518**,
   previously 519). Inspect adjacent frames and the enlarged raw crop. These are
   individually decoded images, not approximate browser video seeks.
3. Confirm only when the evidence supports a bounce at that frame. Otherwise use
   **Mark unsure**, preserve a wider timing range, or reject a false candidate.
   A clicked pixel is optional, for a wrongly located/missing detection. It is tied
   to that exact frame and is cleared when the chosen frame changes.
4. Export `tennis-calibration-review.json`. From this project directory, apply it:

```powershell
.\Apply-Court-Review.ps1 -ReviewFile "$env:USERPROFILE\Downloads\tennis-calibration-review.json"
```

The command creates a new timestamped replay **and** calibration desk. Open the
printed `calibration.html` or `replay.html` path. Use the browser's actual exported
filename if it appends `(1)`. Browser autosave is best-effort; keep the exported file.
The existing Run-Shot-Replay script also accepts `-Corrections <file>`, but only
Apply-Court-Review builds the enlarged frame-review desk as well.

Changing a landmark hides the stale fitted overlay. Export and rebuild to refit
the camera; the browser does not claim a live numerical calibration after edits.
The new desk also reports before/after coordinates. Those numbers refer to its
built package, not unsaved or unapplied edits.

## What changed

- Camera fitting uses fifteen baseline, singles-line and service-line points,
  including the visibly curved near baseline, not only four outer corners.
- The constrained camera model fits focal length, pose and two radial lens
  coefficients. Square pixels, centred principal point and zero tangential
  distortion remain assumptions. The same lens model is used for court mapping,
  player joint rays, ball fitting and projection back into the source image.
- The draft points were annotated from clip frame 300 (source frame 5040), not
  adjusted to force the ball into an expected region. No ball locations were
  used to optimize the court fit.
- The draft serve timing moves from frame 519 to 518, based on the neighboring
  source frames. The alley candidate stays at frame 122. Both have a proposed
  +/- one-frame uncertainty range and remain **unreviewed** until a user confirms.
- The corrected serve projects inside the far service box. The alley landing
  moves closer to the middle of the alley. These are estimates to review, not
  certified line calls or independent validation of 3D accuracy.

The fit residual is about 1.5 pixels on the fifteen supplied landmarks, versus
about 19.4 pixels for the previous camera on those same points. These are fitted
residuals, not an accuracy score on unseen points. The pinhole-plus-radial method
uses [OpenCV camera calibration and point undistortion](https://docs.opencv.org/4.5.1/d9/d0c/group__calib3d.html).

## Timing and uncertainty

At this clip's 30 fps, adjacent decoded frames are 33.3 ms apart. Motion blur and
occlusion may mean the actual ground contact occurs between frames. The tool
supports frame-level review and explicit ranges; it does **not** recover an exact
sub-frame impact time. The range is a reviewer-provided bracket, not a statistical
confidence interval. A ball direction change alone is not proof of a bounce.

Reviewing calibration does not confirm any bounce. Confirming a bounce does not
prove its projected metric location or recover a missing hit. The reviewed-flight
channel still needs appropriate confirmed endpoints and adequate observations.
Preview trajectories remain separate and unvalidated; refitting can reduce their
coverage instead of forcing a plausible-looking continuous path.

## Safety and verification

Corrections bind to the original review fingerprint, frame count and image size.
They cannot be applied to another run accidentally. The new package ID includes
the calibration and timing changes; geometry is recomputed at the corrected frame.
Overlapping corrections and a separate event-labels file are rejected, not silently
merged over existing decisions. Inputs, original detections and old runs are preserved.

104 Python tests cover the earlier workflow plus radial-camera recovery, distorted
ray/flight consistency, bad landmark rejection, review binding, timing bounds,
separate-package export and conflict rejection. Fake-DOM JS tests cover point
editing, stale-overlay hiding, exact-frame selection, pixel clearing on frame
changes, confirm/unsure/reject, undo and export. Source frames and projected court
lines were visually inspected; actual browser layout/playback was not verified.

```powershell
.\.venv-ball\Scripts\python.exe -m unittest discover -s tests -q
node tests\test_calibration_review_ui.cjs runs\court-bounce-ready\calibration.html
node tests\test_shot_replay_ui.cjs runs\court-bounce-ready\replay.html
```
