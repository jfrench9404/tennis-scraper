# Five-frame tennis ball detector

## Watch the completed test first

Open `runs/gridtracknet-court-review/review.mp4`. This contains **new neural ball
inference** over the same 600 original frames, beginning at source time 158 seconds.
Player and racquet detections are reused from `rally-confirmed` and selected using
the improved identity logic. This review uses boxes rather than rerunning pose;
the main analyzer still supports full skeleton overlays.

The tested model is [GridTrackNet](https://github.com/VKorpelshoek/GridTrackNet), a
pretrained tennis-specific network using five consecutive RGB frames. The network
predicts a confidence grid plus within-cell position offsets for each frame. This
is a model replacement, not optical-flow interpolation or another YOLO threshold.

## What changed in the test

| Observed ball coverage | Frames out of 600 |
| --- | ---: |
| Original `rally-confirmed` | 277 |
| Previous improved filters: model observations | 316 |
| Previous improved filters + tentative motion observations | 365 |
| New court-aware GridTrackNet: model observations only | 390 |

For output frames 60–149 (source 160–163 seconds), the previous motion-assisted
review had a ball in 66/90 frames; the new model has one in 81/90 frames. These are
**presence counts, not precision, recall, or ground-truth accuracy**. Visual checks
covered a fixed every-15-frame sample where a prediction exists, known dropouts,
and the alternative candidates selected by the court filter. We did not label
every frame or fine-tune/train the model on this footage.

There are still 210 frames with no accepted ball. Those include low-score misses,
occlusion/held balls, and rejected neighbouring-court predictions. The fixed towel
exclusion still creates a blind spot: the real ball crossing that patch in frames
72–74 is rejected. Do not treat the current output as reliable contact/bounce truth.

The second pass keeps alternative high-scoring grid cells and applies court
membership before selecting one ball. This recovered 9 observations that were
lost when a neighbouring-court ball was the strongest global prediction. A ball
can still be confused with equipment or another ball inside the same search area.

The 20-second review took about 171 seconds on this machine's CPU (~3.5 source
frames/second). That includes ball inference and rendering, but **not** fresh player
inference. It is not realtime livestream performance. GPU optimization is separate.

## Already installed in this project

- `.venv-ball`: isolated runtime; the existing model/Python environment is untouched.
- `models/gridtracknet/model_weights.h5`: pinned public numeric weights.
- `models/gridtracknet/gridtracknet.onnx`: converted inference graph.
- `models/gridtracknet/manifest.json`: source revision and SHA-256 checksums.
- `Run-Ball-Review.ps1`: reruns the review into a new timestamped directory.

From this folder, one command repeats the ball comparison without rerunning YOLO:

```powershell
.\Run-Ball-Review.ps1
```

If your PowerShell policy disallows scripts, run the already-prepared interpreter
directly and choose an unused output name:

```powershell
.\.venv-ball\Scripts\python.exe -m tennis_vision.temporal_review --run runs\rally-confirmed --output runs\my-ball-review --scene sebbie-scene.json
```

## Enable it in the complete analyzer

In the **existing environment you use for player/pose inference**, install the
optional inference runtime once:

```powershell
python -m pip install onnxruntime==1.24.4
```

Then run:

```powershell
python -m tennis_vision --input "C:\Users\John\Downloads\sebbie-demo-shortest (1).mp4" --output runs\rally-neural-ball --court court.yaml --scene sebbie-scene.json --start-seconds 158 --max-frames 600 --temporal-ball
```

Use a new output directory to preserve previous results. `--temporal-ball` replaces
generic YOLO ball detections; it does not silently fall back to them. Do not combine
it with `--motion-ball`. `--ball-threshold` defaults to 0.5; lowering it may add
false detections and is not a guaranteed improvement. `--ball-model` accepts a
compatible exported ONNX path; arbitrary TrackNet checkpoints are not compatible.

Frames remain aligned with the source. Five-frame blocks add at most four frames
of lookahead (~133 ms at 30 fps), in addition to processing time. A final short
block repeats its last image for context but emits only the real frames. No gap
filling is performed. The tested video is fixed-camera 1280x720 at 30 fps; camera
cuts, different frame rates, multiple balls, and changed framing need validation.

Exports label these observations `source: gridtracknet`; the frame log includes
raw grid candidates, selected pixels/scores, and model provenance. The small box
around a predicted point is a display marker, **not a measured ball diameter**.
Learned detections can enter the existing experimental event/3D routines; this
does not make those routines accurate. The standalone review deliberately generates
no new shots, bounce events, or calibrated 3D coordinates.

## Model and conversion provenance

Upstream revision: `0764162b73fb64d440fd9e6c363d592965400799`.
Weights SHA-256: `ac93a1f074b5292c6a06474db1fb8a2ff91553816eb337d6062b28535056ee42`.
Code/license attribution is in `THIRD-PARTY-NOTICES.md` (MIT).

The converter reads numeric HDF5 arrays; it does not execute downloaded Python or
unpickle a checkpoint. It preserves the upstream normalization along the width
axis, even though that differs from common channel-wise normalization. Input RGB
matches the training loader; the upstream video helper contains a double colour
swap that is not reproduced here. The graph passed ONNX validation and was tested
on the actual clip. Numerical parity with a running TensorFlow implementation has
not been independently measured.

TrackNetV4 was investigated, but its published results page currently links its
checkpoint downloads to `#` placeholders. This implementation therefore does not
claim to use V4 or the newest model. It uses an available tennis-trained checkpoint
that we could actually execute, with a replaceable detector interface for future
fine-tuning or newer released weights.

Tests cover RGB/order, heatmap/grid decoding, court-first candidate selection,
missing detections, frame alignment, partial blocks, model provenance propagation,
and CLI integration with mocked YOLO/temporal responses. The real ball model was
separately run over all 600 frames. Full fresh combined YOLO+pose+GridTrackNet
inference was not run in the isolated review environment.
