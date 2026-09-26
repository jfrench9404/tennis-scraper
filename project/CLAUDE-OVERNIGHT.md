# Overnight work, 2026-09-25 → 26 (Claude)

Branch `claude/overnight-2026-09-25` on github.com/jfrench9404/tennis-scraper.
Everything ran locally. No footage, stills or model weights were uploaded.

## What actually happened overnight (read first)

Video 1 detection ran normally until about 02:30, then the laptop slept (the log has
no progress from 02:31 to 05:30), and the process was stopped at 05:30 when the Claude
session ended. **1,800 of 5,654 frames (6 of 19 chunks) are complete and valid.**
Nothing needs redoing: the command below resumes at frame 1,800 (fingerprint verified).
The review, replay, comparison and video 2 steps never started.

Run it in **your own PowerShell window** (not inside Claude), with the laptop plugged
in and the lid open. `--keep-awake` cannot override lid-close or battery sleep.

```powershell
cd C:\Users\John\Documents\Codex\2026-09-17\can-x20\outputs\tennis-scraper\project
.\Run-Game.ps1 -Video "media\sebbie-demo-shortest (1).mp4" -Name claude-game-v1 -RunDir runs\claude-game-v1-full
```

It needs about 5.5 h more for detection (about 5 s/frame on the GPU), then about 30-45 min
for review videos, far-player crop poses and the replay. It prints progress and ETA.
If it stops, run the same line again. The result is
`runs\claude-game-v1\replay\replay.html`.

## Open these first

| What | File |
| --- | --- |
| Baseline replay with **3D body meshes and racquets** | `runs/claude-baseline-racquets/replay.html` |
| Full video 1 (188 s), when the overnight chain finishes | `runs/claude-game-v1/replay/replay.html` |
| Its event review (ball continuity, candidates, clips) | `runs/claude-game-v1/review/review.html` |
| Baseline vs full-run coverage on the same source frames | `runs/claude-game-v1/compare-vs-baseline.json` |
| Overnight chain log (what ran, when) | `runs/claude-overnight-chain.log` |

Open replays in Chrome or Edge by double-clicking (file://). Keep `source.mp4` beside
`replay.html`. In the 3D panel use **Bodies**, **Mesh wireframe**, **Complete missing
limbs (animation)** and **Racquets**.

## Environment

The original virtual environments were not copied, so a new one was made:
`project/.venv` (Python 3.13.1). Exact packages: `requirements-lock-windows.txt`.

- PyTorch **2.14.0 CPU** and Ultralytics **8.4.163**, the versions of the original
  inference environment. The CUDA build (about 4.5 GB installed) did not fit: C: had
  about 1.3 GB free, and I did not delete any of your files to make room.
- GPU inference instead uses **ONNX Runtime DirectML** (25 MB, works on the Quadro
  RTX 3000 through DirectX 12). `onnxruntime-directml 1.24.4` replaces `onnxruntime
  1.24.4`; GridTrackNet still runs on its CPU provider as in the baseline.
- `models/onnx-export/` holds ONNX exports of `yolo11x.pt` and `yolo26l-pose.pt`
  (same weights, format conversion; provenance JSON beside each).
- Node 24.19.0 (the Codex runtime copy) runs the JS suites; FFmpeg is
  `C:\Users\John\Downloads\ffmpeg\ffmpeg.exe` (has libx264).

Speed measured on this laptop: PyTorch CPU about 15 s/frame, DirectML about 5 s/frame
for the full detector + pose + far-crop pipeline. A 3-minute video takes about 7.5 h on
the GPU path. That is measured, not extrapolated from the 600-frame example.

## Stage 1: longer footage

**The two files.** Both come from the same fixed indoor camera and have no hard
cuts. Neither is verified as one complete game:

- **Video 1:** starts with players at the net/far end (warm-up or changeover?);
  near-side serving from about 80 s.
- **Video 2:** a navy/gold near player serves until about 64 s, the near side is
  empty around 64–96 s, and a different near player (white) plays from about 104 s,
  which looks like a changeover.

**Need from you:** game boundaries, or confirmation that a whole file is wanted.

**New tools** (all under `tennis_vision/`):

- `longrun.py`: chunked, resumable detection and tracking for a whole file or a range
  (`--start/--end-frame` or `--start/--end-seconds`).
  - **Chunks:** multiples of the 5-frame ball-model block. Each stores its rows and a
    checkpoint of all analyzer state; rerunning the same command resumes.
  - **Resume safety:** a fingerprint (input hash, selection, settings, model hashes,
    analysis-code hashes) refuses to mix runs. Resume re-decodes the frame before the
    boundary and checks its hash.
  - **Output:** every row gains `source_frame`, and the merged folder keeps the
    standard run contract, so review and replay tools work unchanged.
  - **Monitoring:** progress, elapsed time and measured ETA (this session's frames
    only), in the log and `progress.json`.
  - **Keep-awake:** `--keep-awake` asks Windows not to sleep while processing (no
    settings changed).
- `cli.py`: the per-frame loop moved into `FrameAnalyzer` (identical output).
- `camera_check.py`: projects the reviewed court model into sampled frames and scores
  line alignment. Both videos are consistent with the reviewed calibration (median
  0.998, minimum 0.967, versus 0.990 at the reviewed frame). The score drops to 0.57
  for a 5 px shift, so the check does discriminate.
- `carry_calibration.py`: copies **only the 15 reviewed landmarks** to a new run as
  `calibration_status: "draft"`, bound to the new run id, after the camera check
  passes. No bounce edits or old event ids are carried. Review it in the calibration
  desk before treating it as reviewed.
- `game_pipeline.py` and `Run-Game.ps1`: one command from video to replay (detection,
  review, draft calibration, replay with crop poses, contacts, bodies and racquets).
  Finished stages are skipped. This replaces the sample-specific `.venv-*` wrappers,
  which are left untouched.
- `compare_runs.py`: coverage on identical source frames plus per-30 s windows.
  Coverage counts, not accuracy.

**Verification**

- **CPU parity:** a 30-frame long run over the baseline window, in 3 chunks with an
  interruption and resume, is **byte-identical** to `runs/rally-neural-ball/events.jsonl`
  (apart from the added `source_frame`).
- **GPU agreement:** the same 30 frames on DirectML match the CPU run: player-box IoU
  1.000, maximum joint difference 0.01 px, same ball frames and racquet tracks. Tracks
  are byte-identical in 28 of 30 frames.
- **Tests:** unit tests cover chunk planning, frame/second selection, resume equality,
  tampered-chunk redo, boundary-hash refusal, merge gaps/duplicates and analyzer-state
  pickling.

## Stage 2: 3D player bodies

**Renderer decision.** The replay keeps its canvas renderer and adds a WebGL layer
(three.js 0.159.0, vendored and inlined, so no CDN and it works from file://).

- **Camera:** identical pinhole, so ball, bounce and label overlays stay aligned.
- **Fallback:** without WebGL the page falls back to the old pose lines.
- **Version:** 0.159.0 is the last three.js with a classic script. Newer releases are
  ES-module-only, which browsers block on file://.

**Bodies.** Articulated segments (capsule limbs, box torso, sphere head, hands, flat
feet), anchored to the existing grounded feet.

| Geometry | Status |
| --- | --- |
| Joint positions | Observed 2D pose, lifted onto the existing 2.5D camera-facing plane. Depth is not measured. |
| Limb thickness, torso depth, head size | Fixed visual template |
| Grey translucent limbs | Only with *Complete missing limbs*; animation, counted separately |
| Translucent capsule | Frame without usable pose (placeholder); grey if the position is predicted |

A per-frame note under the view states the counts.

**Browser-checked (Chrome, localhost):**
- **Speed:** renders at about 6.5 ms/frame across all 600 frames.
- **Placeholders:** 103 placeholder player-frames, matching the handoff.
- **Controls:** wireframe, completion and lines-mode toggles work.

**Not checked:** a full playback session with seeking. The test server could not seek
video, so frames were set directly. On file://, seeking uses the browser's normal
local-file path.

## Stage 3: racquets

`racquet_replay.py` builds display-only records. A racquet is drawn only when:

- a racquet box is assigned to **that identified player** (never a bystander), and
- a supported wrist of that player is in the same frame.

**Hand:** left, right or both, decided **per frame** with the stroke-execution
thresholds. Hand switching and two-handed strokes need no player handedness.

**Placement:** grip at the wrist(s); the axis points toward the box centre on the
player plane. Face orientation is a display choice.

**Visibility:** opacity follows detector confidence. Dropouts are held at most 0.2 s
and shown grey as *stale*; cuts clear the hold.

**Evidence and results**
- **Scope:** racquets are never contact evidence.
- **Unchanged:** events, shots, package id and review bindings are identical to
  `runs/raw-play-ready` (verified).
- **Baseline counts:**
  - 509 observed player-frames (320 left, 148 right, 41 both hands).
  - 199 stale player-frames.
  - 113 boxes hidden for lack of a supported wrist.

## Commands

From `project/` in PowerShell:

```powershell
# Whole video, GPU (DirectML); rerun the same line to resume
.\Run-Game.ps1 -Video "media\sebbie-demo-shortest (1).mp4" -Name game-v1
# A time range, e.g. once you give game boundaries
.\Run-Game.ps1 -Video media\sebbie-demo-shortestpar2.mp4 -Name game-v2a -StartSeconds 0 -EndSeconds 64
# Baseline rebuild with bodies + racquets (cached poses, no inference)
.venv\Scripts\python.exe -m tennis_vision.shot_replay --run runs/rally-neural-ball --review runs/rally-event-review-verified --court court.yaml --corrections runs/court-bounce-20260925-110335-545/calibration-corrections.json --pose-weights yolo26l-pose.pt --pose-cache runs/pose-cache-DD79B09788E82900.json --contacts --output runs/NEW-NAME
# Regression checks
.venv\Scripts\python.exe -m unittest discover -s tests -q
```

## Not done / limitations

- **Accuracy:** no accuracy measured. There is no labelled ground truth for these
  videos; coverage counts are not accuracy.
- **Game boundaries and scoring:** no game boundaries, point outcomes, ace/fault/winner
  labels or scoring. Missing observations prove nothing.
- **3D bodies:** depth and orientation are not measured (single camera, 2.5D plane).
  No rigged glTF model was added.
- **Racquet geometry:** handle endpoints, grip and face angle are not measured.
- **Spin:** no spin or RPM claims.
- **GPU numerics:** DirectML results can differ from PyTorch CPU in rare frames, at
  rounding level in the 30-frame check.
- **Calibration:** the full-run calibration is a **draft** until you review it.
