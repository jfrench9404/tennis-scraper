# Distant player and ball detection update

New analyses now run additional inference on a far-court crop (from manual
calibration) and a crop around the latest unambiguous observed ball. Crop results
are remapped into source-image coordinates and deduplicated before tracking.
These are fresh detections, not extrapolated boxes. The recent-ball crop expires
after three missing or ambiguous frames. Full-frame inference remains active for
reacquisition. Ball detections from adjacent courts may still occur.

The baseline allowance is now 6 metres. With this clip's calibration, far-player
feet can project beyond the previous 4-metre limit. This wider allowance may also
admit bystanders behind the baseline; it is not player identity recognition.

The existing rally-check had a ball in 363/600 frames and never both players
together. Those are coverage counts, not accuracy scores. Four new regression
tests cover crop bounds, coordinate translation, duplicates, and missing-ball
handling; all 14 tests pass. Fresh model inference has not been verified in the
assistant's inspection environment, which lacks torch/ultralytics.

Run from the project directory:

```powershell
python -m tennis_vision --input "C:\Users\John\Downloads\sebbie-demo-shortest (1).mp4" --output runs\rally-detail --court court.yaml --start-seconds 158 --max-frames 600
```

Additional inference increases processing time. `--no-detail-pass` disables the
crop passes for comparison. Per-frame `detail` and `filters` fields record crop
counts, raw player/ball counts, and rejected-player counts to distinguish model
misses from filtering losses. Existing outputs are untouched.
