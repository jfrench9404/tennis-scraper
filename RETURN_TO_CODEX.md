# Claude -> Codex return report

Filled in from `project/CLAUDE-OVERNIGHT.md`, the git history on `dev` (up to
`3af8eac`, merge of PR #2, 2026-09-26) and the run folders committed in the repo.
Laptop-only state (footage, weights, `chunks/`, `*.log`, `progress.json` of the
full run) is not in git and could not be observed while writing this. Where this
report depends on it, it says so.

Paths below are relative to `project/` in John's main checkout:
`C:\Users\John\Documents\Codex\2026-09-17\can-x20\outputs\tennis-scraper\project`.

## Delivered result

- **What is implemented versus still planned:**
  - **Done (code, on `dev`):**
    - Stage 1: resumable long-run pipeline (`longrun.py`) and one-command wrapper
      (`game_pipeline.py`, `Run-Game.ps1`).
    - Stage 1 support tools: camera-consistency check (`camera_check.py`), draft
      calibration carry-over (`carry_calibration.py`), coverage comparison
      (`compare_runs.py`) and ONNX export for DirectML (`export_onnx.py`).
    - Stage 2: WebGL articulated body meshes in the replay (`body_meshes.js`,
      three.js 0.159.0 inlined).
    - Stage 3: display-only 3D racquets (`racquet_replay.py`).
  - **Done (runs):** 30-frame CPU parity and DirectML runs, the 30-frame
    end-to-end pipeline test, and the camera check on both long videos. Also the
    baseline replay rebuilt with bodies (`runs/claude-mesh-baseline`) and with
    bodies + racquets (`runs/claude-baseline-racquets`).
  - **Detection done, rest NOT done:** the full video 1 run.
    - **Detection is complete:** 5,654 of 5,654 frames, 19 of 19 chunks in
      `runs\claude-game-v1-full`, last progress 2026-09-26 13:09 -04:00. Source:
      John ran `Get-RunStatus.ps1` (PR #21) on the laptop on 2026-09-28.
    - The pipeline then stopped at stage 2 (event review) with
      `Unrecognized option 'fps_mode'`, because `Run-Game.ps1` fell back to
      FFmpeg 5.0 (issue #13, PR #18).
    - The same status run shows one wall-clock gap between chunks 6 and 7
      (about 02:15 to 07:03 on 2026-09-26, from file modified times; the log has
      no timestamps). That is the laptop-sleep stall.
    - Only `longrun-manifest.json` of that folder is in git; `chunks/`,
      `progress.json` and logs are laptop-only.
    - Review, draft calibration, replay and comparison for video 1 have not been
      produced. Nothing was started for video 2.
  - **Not done / planned:**
    - Game boundaries: not decided; they are John's decision.
    - Point outcomes and scoring: none.
    - Accuracy measurement: none.
    - Rigged glTF body model: not added.
    - Measured racquet orientation or grip: none.
    - Review of the full-run draft calibration in the calibration desk: not done.
- **Latest replay file to open:**
  - `runs\claude-baseline-racquets\replay.html`: the 20-second baseline (source
    frames 4740-5339 of video 1) with 3D body meshes and racquets.
    - Open it in Chrome or Edge from file://.
    - It needs `source.mp4` beside it. That file is gitignored, so it exists only
      on the laptop, if it was copied there.
  - `runs\claude-game-v1\replay\replay.html` (full video 1) **does not exist yet**.
    Step 1 of the next command creates it.
- **Exact commands John should run next.** Use John's own PowerShell window, not
  a Claude session. Keep the laptop plugged in with the lid open (`--keep-awake`
  cannot override lid-close or battery sleep).
  1. **Before starting, check that no run is active.** Look for a recent
     `[longrun]` line in `runs\*.log` or a `runs\<run>\progress.json` that is
     still growing.
  2. **Finish video 1.** Detection is complete, so the pipeline skips stage 1
     and goes to review, draft calibration and replay: about 30-45 min
     (estimate from CLAUDE-OVERNIGHT.md). If it stops, rerun the same line.
     - First move any half-built `runs\claude-game-v1\review` aside
       (`Rename-Item runs\claude-game-v1\review review-failed-ffmpeg5`); the
       pipeline refuses to reuse an incomplete review folder.
     - Stage 2 needs FFmpeg 5.1+ with libx264. Until PR #18 is merged, pass it
       explicitly as below; after #18, `-Ffmpeg` can be dropped (it picks a
       capable one and says which).
     ```powershell
     cd C:\Users\John\Documents\Codex\2026-09-17\can-x20\outputs\tennis-scraper\project
     .\Run-Game.ps1 -Video "media\sebbie-demo-shortest (1).mp4" -Name claude-game-v1 -RunDir runs\claude-game-v1-full -Ffmpeg "$env:USERPROFILE\miniconda3\envs\opencv-env\Library\bin\ffmpeg.exe"
     ```
     - **Resume conditions.** The fingerprint covers the absolute input path,
       input SHA-256, selection, settings (including the `court.yaml` text, the
       scene JSON and `chunk_frames` 300), model hashes and the hashes of the
       analysis files in `longrun.CODE_FILES`.
       - Run it from the main checkout above, not from a worktree.
       - Do not add `-Cpu`, because resume refuses a device switch.
       - Do not change `court.yaml`, `sebbie-scene.json` or any `CODE_FILES` file
         before it finishes.
     - As of `dev` at `3af8eac`, all nine code hashes in
       `runs/claude-game-v1-full/longrun-manifest.json` match the files on `dev`.
     - A later merge that touches `cli.py`, `tracking.py`, `filters.py`,
       `detail.py`, `scene.py`, `events.py`, `court.py`, `ball_motion.py` or
       `temporal_ball.py` would make this resume refuse. The run would then need
       a new `-RunDir`.
  3. **When step 2 prints `Done`, run the coverage comparison.** `Run-Game.ps1`
     does not produce it.
     ```powershell
     .venv\Scripts\python.exe -m tennis_vision.compare_runs --a runs\rally-neural-ball --b runs\claude-game-v1-full --output runs\claude-game-v1\compare-vs-baseline.json
     ```
  4. **Review the draft calibration.** John reviews
     `runs\claude-game-v1\calibration-draft.json` in a calibration desk before
     anyone treats it as reviewed. The command below builds the desk into a NEW
     folder.
     ```powershell
     .venv\Scripts\python.exe -m tennis_vision.calibration_review --run runs\claude-game-v1-full --review runs\claude-game-v1\review --court court.yaml --corrections runs\claude-game-v1\calibration-draft.json --output runs\claude-game-v1-calibration-desk
     ```
     - This command comes from the `calibration_review.py` CLI. It accepts
       `draft` status (`court_refinement.py`).
     - It has not been run on a carried draft or on a full-length run.
     - PR #25 (not merged when this was written) adds a dedicated draft desk:
       `--draft runs\claude-game-v1\calibration-draft.json --replay runs\claude-game-v1\replay`.
       Prefer it once merged; its description has the exact command.
     - It writes frame stills, which are local only and gitignored.

## Changes

- **Source files changed and why** (all under `project/tennis_vision/`):
  - `cli.py`:
    - The per-frame loop moved into `FrameAnalyzer`, which owns all cross-frame
      state so it can be checkpointed.
    - CLI output is unchanged.
    - Adds an optional `--device` flag.
  - `longrun.py` (new): chunked, fingerprinted, resumable detection and tracking
    for a whole file or a frame/second range.
    - Chunks are multiples of the 5-frame ball block.
    - Resume checks the hash of the frame before each boundary.
    - Reports progress with a measured ETA.
    - `--keep-awake` (Windows) keeps the PC from sleeping during a run.
    - Sets `YOLO_OFFLINE`/`YOLO_AUTOINSTALL=false`, so Ultralytics never downloads
      or pip-installs mid-run.
  - `export_onnx.py` (new) plus the opt-in `--backend onnx-directml`: GPU
    inference through DirectML, because the CUDA torch build did not fit on disk.
  - `camera_check.py` (new): projects the reviewed court model into sampled frames
    and scores how well it matches the painted lines (a consistency check, not an
    accuracy measure).
  - `carry_calibration.py` (new): copies only the 15 reviewed landmarks to a new
    run.
    - The copy is written as `calibration_status: "draft"`, after the camera check
      passes.
    - No bounce edits or old event ids are carried.
  - `game_pipeline.py` (new) and `project/Run-Game.ps1` (new): run the 4 stages in
    order (detection, event review, draft calibration, replay). Finished stages are
    skipped.
    - `-RunDir` continues an existing detection folder.
    - FFmpeg falls back to `%USERPROFILE%\Downloads\ffmpeg\ffmpeg.exe`.
  - `compare_runs.py` (new): observation coverage on identical source frames plus
    per-30-s windows. Coverage only, not accuracy.
  - `shot_replay.py`, `shot_replay.html`, `body_meshes.js` (new): a WebGL layer
    under the existing 2D canvas.
    - Uses the same pinhole camera, and falls back to pose lines without WebGL.
    - Adds articulated segment bodies, a racquet mesh, and toggles (Bodies, Mesh
      wireframe, Complete missing limbs (animation), Racquets).
    - Far-player crop refinement accepts the exported ONNX pose model.
  - `racquet_replay.py` (new): per-frame racquet display records.
    - A racquet appears only when a box is assigned to that identified player and
      a supported wrist of the same player is present.
    - The hand is decided per frame.
    - A dropped racquet is held at most 6 frames (0.2 s) and labelled stale.
  - `vendor/three-0.159.0.min.js`, `vendor/README.md`, `vendor/THREE-LICENSE.txt`
    (new).
  - Docs: `project/CLAUDE-OVERNIGHT.md`, `project/THIRD-PARTY-NOTICES.md`,
    `project/requirements-lock-windows.txt`, both READMEs.
  - Repo workflow: `CLAUDE.md`, `docs/WORKING-WITH-CLAUDE.md`, `.github/` (PR
    template, issue form, CI workflow).
- **Schema/API changes and migration instructions:**
  - Long-run rows keep the existing run-relative `frame`/`time_s` and gain
    `source_frame`.
    - Merged run folders keep the standard contract (`events.jsonl`,
      `summary.json`, `source-offset.json`, `shots.json`, `flight3d.json`,
      `ball-model.json`).
    - They add `longrun-manifest.json`, `longrun-report.json` and
      `progress.json`.
  - The replay report gains `racquet_display` (method
    `racquet_box_wrist_display_v1`, counts, limitations). Replay pages inline
    three.js and the mesh script.
  - The package id version string is unchanged (`shot-replay-6`).
    `runs/claude-baseline-racquets` has the same `package_id` and `review_run_id`
    as `runs/raw-play-ready`: `3c970d65…` and `39449302…`.
  - The pipeline writes `calibration-draft.json` (schema_version 1,
    `court_and_bounce_review`, `calibration_status: "draft"`, `landmark_source:
    carried_from_reviewed_run_…_after_camera_check`, no bounce edits), bound to
    the new run id.
  - No migration is needed: existing run and review folders are read unchanged.
- **New dependencies, versions, model weights, 3D assets, sources and licences:**

  | Item | Version | Source | Licence |
  | --- | --- | --- | --- |
  | three.js (vendored, inlined in replay pages) | 0.159.0 | `https://registry.npmjs.org/three/-/three-0.159.0.tgz`, `package/build/three.min.js`; tarball sha512 and file SHA-256 `7b1c5d75…a585` in `tennis_vision/vendor/README.md` (hash checked by `test_replay_page.py`) | MIT, text in `tennis_vision/vendor/THREE-LICENSE.txt` |
  | onnxruntime-directml (replaces onnxruntime 1.24.4) | 1.24.4, wheel sha256 `2f1031cb…d19e` in `requirements-lock-windows.txt` | PyPI | MIT (Microsoft), as stated in `THIRD-PARTY-NOTICES.md`; package metadata not in repo |
  | ONNX exports `models/onnx-export/yolo11x-1280-dynamic.onnx`, `yolo26l-pose-1280-dynamic.onnx` (local only, gitignored) | Ultralytics 8.4.163, torch 2.14.0+cpu, imgsz 1280, dynamic | Format conversions of the local `yolo11x.pt` (sha256 `7bc158aa…3a24`) and `yolo26l-pose.pt` (`ad33da8a…1abf`); provenance JSON with output SHA-256 beside each (`db4b0a3e…8b72`, `c6b7591c…9625`) | Same terms as the source checkpoints; `THIRD-PARTY-NOTICES.md` says Ultralytics package metadata indicated AGPL-3.0. Weight licence itself not confirmed in repo |
  | Python environment `project/.venv` (Python 3.13.1) | torch 2.14.0+cpu, torchvision 0.29.0+cpu, Ultralytics 8.4.163 (the original inference versions); full list in `requirements-lock-windows.txt` | PyPI; torch from `https://download.pytorch.org/whl/cpu` | Not confirmed in repo except as noted above |
  | 3D body/racquet meshes | none added | Procedural geometry in `body_meshes.js` (capsules, boxes, spheres); no external mesh or glTF asset | n/a (project code) |

  - The GridTrackNet model is unchanged: MIT, upstream revision `0764162b…`; see
    `models/gridtracknet/manifest.json` and `THIRD-PARTY-NOTICES.md`.
  - No new footage and no new weights beyond the ONNX format conversions.
- **Original human review and camera decisions preserved?** Yes, as far as git
  shows.
  - The overnight commits did not modify `court.yaml`, `runs/rally-neural-ball/`,
    `runs/rally-event-review-verified/`,
    `runs/court-bounce-20260925-110335-545/calibration-corrections.json`,
    `runs/raw-play-ready/` or `runs/pose-cache-DD79B09788E82900.json`. All new
    output went to new `runs/claude-*` folders.
  - Bounce `bounce_candidate-000519` remains unreviewed.
  - No changes were explicitly approved by John. The carried calibration is
    labelled draft.

## Data and execution

- **Input files** (from `project/media/media-index.json`; neither is verified as a
  complete game):

  | File | SHA-256 | FPS | Resolution | Frames | Duration |
  | --- | --- | ---: | --- | ---: | ---: |
  | `media\sebbie-demo-shortest (1).mp4` (video 1) | `0eb4675b7e21a672d35b1ac8c017dde81b3cb175877c48ce60de0e34151b104b` | 30 | 1280x720 | 5654 | 188.47 s |
  | `media\sebbie-demo-shortestpar2.mp4` (video 2) | `a57f058170a1e36dd28408eced1025c445d29c7270036b83e3d90294a0ffb11b` | 30 | 1280x720 | 5549 | 184.97 s |

- **Selected game/segment boundaries and source-frame offset convention:**
  - **Boundaries:** none selected. Game boundaries are John's decision and have
    not been given.
    - The video 1 run covers the whole file: source frames [0, 5654), in 19
      chunks of 300 frames (the last is 254).
    - The video 2 range in CLAUDE-OVERNIGHT.md (0-64 s) is an example command,
      not a decision.
    - Observations from CLAUDE-OVERNIGHT.md, unverified: video 1 has near-side
      serving from about 80 s; video 2 changes near player around 64-104 s.
  - **Offset convention:**
    - `source-offset.json` gives `source_start_frame`/`source_start_seconds` of
      the run.
    - Rows keep `frame`/`time_s` relative to that start. Long-run rows also carry
      absolute `source_frame`.
    - The baseline starts at source frame 4740 (158.0 s) of video 1. The
      reviewed calibration frame 300 is therefore source frame 5040.
- **Output folders** (committed in git unless marked):

  | Folder | What | How it was made |
  | --- | --- | --- |
  | `runs/claude-longrun-parity-30` | 30 frames at source 4740-4769, PyTorch CPU, 3 chunks of 10, interrupted and resumed | `longrun.py`; the manifest records selection [4740, 4770), `chunk_frames` 10, device cpu. The exact command line is not recorded in the repo |
  | `runs/claude-longrun-dml-30` | Same 30 frames on DirectML, plus `compare-vs-baseline.json` | `longrun.py`; the manifest records the same selection, `chunk_frames` 10 and backend `onnx-directml`. Then `python -m tennis_vision.compare_runs --a runs\rally-neural-ball --b runs\claude-longrun-dml-30 --output runs\claude-longrun-dml-30\compare-vs-baseline.json` |
  | `runs/claude-pipeline-test-30` | End-to-end pipeline test on the DML-30 run: `calibration-draft.json`, `pose-cache.json`, `review/`, `replay/` | `game_pipeline.py` with `--run-dir runs\claude-longrun-dml-30` (exact line not recorded in repo) |
  | `runs/claude-camera-check` | `camera-check.json` for both long videos | `camera_check.py`; reproduce into a new folder with the command below |
  | `runs/claude-mesh-baseline` | Baseline replay page rebuilt with body meshes (replay data unchanged) | Stage 2 commit `fd090f5` |
  | `runs/claude-baseline-racquets` | Baseline with bodies + racquets (cached poses, no inference) | Baseline rebuild command below |
  | `runs/claude-game-v1-full` | Full video 1 detection, **complete** (19/19 chunks, per `Get-RunStatus.ps1` on the laptop, 2026-09-28) | Only `longrun-manifest.json` is in git; `chunks/`, `progress.json` and logs are laptop-only |
  | `runs/claude-game-v1/` | Review, draft calibration, replay for video 1 | Not yet produced |

  All commands run from `project/` in PowerShell:
  ```powershell
  # Full-video pipeline (GPU/DirectML by default; rerun the same line to resume)
  .\Run-Game.ps1 -Video "media\sebbie-demo-shortest (1).mp4" -Name claude-game-v1 -RunDir runs\claude-game-v1-full
  # A range, only once John gives boundaries (NEW name per range)
  .\Run-Game.ps1 -Video media\sebbie-demo-shortestpar2.mp4 -Name game-v2a -StartSeconds 0 -EndSeconds 64
  # ONNX export (Run-Game.ps1 does this automatically if the .onnx files are missing)
  .venv\Scripts\python.exe -m tennis_vision.export_onnx yolo11x.pt yolo26l-pose.pt
  # Camera consistency check for both videos (choose a NEW output folder)
  .venv\Scripts\python.exe -m tennis_vision.camera_check --video "media\sebbie-demo-shortest (1).mp4" --video media\sebbie-demo-shortestpar2.mp4 --corrections runs\court-bounce-20260925-110335-545\calibration-corrections.json --reference-video "media\sebbie-demo-shortest (1).mp4" --reference-frame 5040 --output runs\NEW-NAME\camera-check.json
  # Baseline rebuild with bodies + racquets (cached poses, no inference; NEW output name)
  .venv\Scripts\python.exe -m tennis_vision.shot_replay --run runs/rally-neural-ball --review runs/rally-event-review-verified --court court.yaml --corrections runs/court-bounce-20260925-110335-545/calibration-corrections.json --pose-weights yolo26l-pose.pt --pose-cache runs/pose-cache-DD79B09788E82900.json --contacts --output runs/NEW-NAME
  ```
  Notes on these commands:
  - `.\Run-Game.ps1 ... -Cpu` uses PyTorch on the CPU (about 15 s/frame). A run
    started on one device cannot be resumed on the other.
  - The camera check, the baseline rebuild and the ONNX export need the local
    footage and/or weights.
  - These commands were not run while writing this report. The only command
    re-run here was `compare_runs` on the committed DML-30 run (see
    Verification).
- **Hardware/device, elapsed time, peak memory, resume behaviour:**
  - **Hardware:** Windows laptop, Quadro RTX 3000 through DirectML
    (onnxruntime-directml). GridTrackNet runs on the ONNX Runtime CPU provider.
  - **Measured speed** (from CLAUDE-OVERNIGHT.md), full detector + pose +
    far-crop pipeline: DirectML about 5 s/frame, PyTorch CPU about 15 s/frame.
    A 3-minute video takes about 7.5 h on the GPU path.
  - **Recorded in the committed 30-frame reports:**
    - DML-30: 163.7 s for 3 x 10-frame chunks (0.183 frames/s, about
      5.5 s/frame, including session start).
    - CPU parity: 418 s for the 2 chunks processed after resume (0.048 frames/s,
      about 21 s/frame).
  - **Overnight full run:** detection ran until about 02:30. The laptop then
    slept (no log progress from 02:31 to 05:30). The process was stopped at
    05:30 when the Claude session ended.
  - **Peak memory:** not measured. `longrun.py` records `cuda_peak_memory_mb`
    only on CUDA, which was not used. Rows stream to disk; frames are not held
    in memory.
  - **Resume:**
    - Completed chunks store their rows plus a pickled checkpoint of analyzer
      state.
    - Rerunning the same command skips valid chunks and redoes incomplete or
      tampered ones.
    - It refuses a fingerprint mismatch or a mismatched boundary-frame hash.
    - Finished pipeline stages (`build-status.json` complete) are skipped. An
      incomplete `review/` or `replay/` folder must be moved aside first; the
      pipeline says so and stops.
- **Observed versus estimated/render-only geometry:**

  | Geometry | Status |
  | --- | --- |
  | Player joint positions | Observed 2D pose, lifted onto the existing 2.5D camera-facing plane; depth and orientation not measured |
  | Limb thickness, torso depth, head size | Fixed visual template (render-only) |
  | Grey translucent limbs | Animation, only with "Complete missing limbs"; counted separately |
  | Translucent capsule | Placeholder for a frame without usable pose; grey if the position is predicted |
  | Racquet grip and axis | Grip at the supported wrist(s), axis toward the detected box centre on the player plane: estimated from box + wrist |
  | Racquet face orientation, handle endpoints | Display choice (render-only); not measured |
  | Racquet opacity / stale | Follows detector confidence; held at most 0.2 s and shown grey as stale |
  | Ball XYZ previews | Unchanged from baseline: reviewed-constrained estimates and dashed unvalidated previews stay separate |

  Rendered bodies and racquets are never contact evidence. Events, shots,
  `package_id` and review bindings are identical to `runs/raw-play-ready`,
  verified overnight per commit `7afcfd6`.

## Verification

- **Python/JavaScript tests and logs.** The branch point of this report is `dev` at
  `3af8eac`, run in a Linux cloud container with Python 3.11 and Node 22, from
  `project/`:
  - `python3 -m unittest discover -s tests -q`: 140 tests OK, none skipped
    (125 before the overnight work).
  - `node tests/test_shot_replay_ui.cjs`: passed (fake DOM).
  - `node tests/test_player_shots_ui.cjs runs/raw-play-ready/replay.html`:
    passed.
  - `node tests/test_calibration_review_ui.cjs runs/court-bounce-ready/calibration.html`:
    passed (fake DOM).
  - `node tests/test_event_review_ui.cjs`: passed (fake DOM).
  - CI (`.github/workflows/tests.yml`, Python 3.13, Node 24) runs the same suites
    on every push and PR.
  - On the laptop, run the same suites with `.venv\Scripts\python.exe` and
    `C:\Users\John\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe`.
  - Overnight logs (`runs\claude-overnight-chain.log`, `runs\*.log`) are
    gitignored and exist only on the laptop.
- **Added regression checks (15 tests):**
  - `test_longrun.py` (7):
    - Chunk tiling on 5-frame boundaries, and frame/second selection with
      bad-range rejection.
    - An interrupted and resumed run equals one pass and keeps source frames.
    - An incomplete or tampered chunk is redone.
    - Resume refuses a mismatched boundary frame.
    - Merge writes the standard files and rejects gaps.
    - FrameAnalyzer state survives pickling.
  - `test_carry_calibration.py` (2): refuses a draft source, the same run, a
    different size or a foreign review; the carried file has no bounce decisions.
  - `test_racquet_replay.py` (3): per-frame and both-hand decisions; hand
    switching, bystanders and the stale hold; a cut clears the hold; a missing
    grip wrist hides the racquet.
  - `test_replay_page.py` (2): the page is a single offline file with the legacy
    script first; the vendored three.js matches its recorded hash.
  - `test_compare_runs.py` (1): overlap uses source frames.
- **Actual browser/video/mesh checks (not just fake-DOM logic):**
  - **Browser-checked overnight** (Chrome, localhost, per CLAUDE-OVERNIGHT.md and
    commit `fd090f5`), on `runs/claude-mesh-baseline`:
    - Rendering takes about 6.5 ms/frame across all 600 frames.
    - There are 103 placeholder player-frames (matches the handoff).
    - The wireframe, completion and lines-mode toggles work.
  - **Not checked in a browser:**
    - A full playback session with seeking: the test server could not seek video,
      so frames were set directly.
    - file:// playback.
    - Racquet rendering in `runs/claude-baseline-racquets`: no browser check is
      recorded in the repo. Its racquet logic is covered by unit tests only.
    - The review and replay pages of `runs/claude-pipeline-test-30`.
  - The JS suites are fake-DOM logic checks, not rendering checks. No browser
    check was done while writing this report (documentation only).
- **Baseline comparison and manually reviewed error cases:**
  - **CPU parity:** `runs/claude-longrun-parity-30` (3 chunks, interrupted and
    resumed) is byte-identical to the first 30 rows of
    `runs/rally-neural-ball/events.jsonl` apart from `source_frame`, per the
    overnight run.
    - Re-checked while writing this report by parsing: 30/30 rows are equal once
      `source_frame` is removed.
  - **DirectML vs CPU:** on the same 30 frames, player-box IoU is 1.000, the
    maximum joint difference is 0.01 px, and ball frames and racquet tracks are
    the same (overnight measurement).
    - `compare-vs-baseline.json` shows identical coverage counts. Tracks are
      byte-identical in 28 of 30 frames.
    - Re-running `compare_runs` here reproduced the committed JSON (all fields
      except the run paths).
  - **Camera check** (`runs/claude-camera-check/camera-check.json`, 19 samples
    per video, every 10 s): the reference score at source frame 5040 is 0.9897.
    The threshold is 0.8412 and both verdicts are "consistent".

    | Video | Median | Minimum |
    | --- | ---: | ---: |
    | Video 1 | 0.9983 | 0.9673 |
    | Video 2 | 0.9983 | 0.9845 |

    CLAUDE-OVERNIGHT.md reports that a 5 px shift drops the score to 0.57. This
    is consistency, not accuracy.
  - **Full-run vs baseline coverage:** not produced yet (see step 3 above).
  - **Manually reviewed error cases:** none recorded in the repo for the new runs.
- **Any accuracy metric (ground-truth source and sample size):** none. There is no
  labelled ground truth for these videos. All counts above are coverage or
  agreement between runs, not accuracy.
- **Known failures, limitations and remaining priorities:**
  1. **Finish the full video 1 run** (detection complete; review, calibration
     draft and replay not built yet). Then build the
     comparison and inspect `runs\claude-game-v1\replay\replay.html` and
     `review\review.html` in a browser, including seeking and file:// playback.
  2. **The full-run calibration is a draft** until John reviews it in the
     calibration desk. The camera check shows consistency, not accuracy.
  3. **Game boundaries are unconfirmed.** Neither file is verified as one
     complete game. John must give boundaries, or confirm that a whole file is
     wanted, before any per-game runs (video 2 especially).
  4. **Laptop sleep stopped the first attempt.** Run with the laptop plugged in
     and the lid open, in John's own terminal.
  5. **Other limits:**
     - No accuracy measured.
     - No scoring, point outcomes or ace/fault/winner labels. Missing
       observations prove nothing.
     - No spin or RPM.
     - 3D depth and orientation are not measured (single camera, 2.5D plane).
     - No rigged glTF model.
     - Racquet handle, grip and face angle are not measured.
     - DirectML can differ from PyTorch CPU at rounding level in rare frames.
     - three.js 0.159.0 prints a harmless deprecation warning in the console.

Do not report planned work as completed or detection coverage as accuracy.
