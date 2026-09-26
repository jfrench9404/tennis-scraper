# Review of sebbie-yolo26

Inspected sampled decoded frames from the supplied raw and annotated videos and the complete existing event log. These are different edits: raw contains 5,654 frames, annotation 5,549, both 30 fps. The run summary names `sebbie-demo-shortestpar2.mp4` as its source. Equal timestamps must not be used for frame-level accuracy comparisons.

The old log contains 4,368 predicted racket records versus 3,778 observed racket records; 1,293 predicted ball records versus 2,375 observed ball records. There are up to three ball tracks per frame. Samples show duplicate racket predictions and a missed far player.

Changes:

- No extrapolated ball/racket boxes after missed detections.
- Player predictions expire after five frames, with stale joint coordinates removed.
- Velocity recovery uses the last observed position and elapsed frame count.
- Bounce histories reset on missing/ambiguous observations or a changed ball track.
- Physics fitting excludes predicted and multiple-ball observations.
- Inference size defaults to 1280; use `--imgsz 640` for faster, lower-detail testing.
- A dedicated racket model replaces generic racket detections instead of duplicating them.

Three tracker/event regression tests pass. Higher-resolution inference has not been run here: the available video-decoding environment lacks PyTorch and Ultralytics. There is no measured accuracy improvement claim or newly trained tennis detector in this update. A short fresh analysis should be visually checked before running a full match. Existing output files remain unchanged.

The prior extrapolation update made the annotation noisier. Model version alone does not address this. Reliable far-player/ball recall still needs evaluation at higher resolution and likely tennis-specific training data. Court homography projects feet and bounce points; it does not measure airborne ball height.
