# CLAUDE.md: working in tennis-scraper

Read this before changing anything. John reviews work as pull requests; see
`docs/WORKING-WITH-CLAUDE.md` for the day/night cadence.

## What this is

Local tennis-video analysis from one fixed camera: YOLO detection + pose, a
five-frame GridTrackNet ball model, human-reviewed court calibration, contact and
shot candidates, and a single-file HTML 3D replay (canvas + vendored three.js).
All code is in `project/tennis_vision/`, tests in `project/tests/`.
The latest state is described in `project/CLAUDE-OVERNIGHT.md`.

## Non-negotiable rules

1. **Human decisions are data. Never rewrite them.** Do not edit reviewed files or historic
   run folders to make something pass: `project/court.yaml`, `runs/rally-neural-ball/`,
   `runs/rally-event-review-verified/`, `runs/court-bounce-*/calibration-corrections.json`,
   `runs/raw-play-ready/`, `runs/pose-cache-*.json`. They are hashed into review
   fingerprints (`.gitattributes` keeps them byte-exact). New results go in NEW
   `runs/<name>/` folders.
2. **Footage and weights never leave this PC.** Never commit or upload `*.mp4`, frame
   stills, `*.pt`, `*.onnx`, `*.h5` (see `.gitignore`). No cloud inference or paid jobs
   without John saying so in chat.
3. **No invented tennis facts.** A swing is not a shot; missing observations prove no
   ace/return/winner. No spin or RPM. Coverage counts are not accuracy; only claim
   accuracy against labelled ground truth, and name the source and sample size.
4. **Label geometry honestly.** Observed, estimated, and render-only (animation, template)
   must stay distinguishable in data and UI. Rendered racquets or bodies are never
   contact evidence.
5. **Don't guess on decisions that belong to John**, such as game boundaries,
   reviewing calibration, confirming a bounce or shot, or uploading data. Ask in the PR
   under "Decisions needed" and leave the default conservative.

## Environment (Windows laptop)

- Python: `project/.venv/Scripts/python.exe` (3.13, CPU torch 2.14.0, Ultralytics
  8.4.163, onnxruntime-directml 1.24.4). Exact list: `project/requirements-lock-windows.txt`.
- Node for JS suites: `C:\Users\John\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe`
- FFmpeg: `C:\Users\John\Downloads\ffmpeg\ffmpeg.exe`
- Git over HTTPS needs `git -c http.sslBackend=schannel` (set in this repo's config).
- Disk is tight (a few GB free). Check `(Get-PSDrive C).Free` before large installs or
  outputs; never delete John's files to make room. Ask instead.
- GPU: Quadro RTX 3000 via DirectML only (the CUDA torch build does not fit on disk).

## Verify every change

From `project/`:

```
.venv\Scripts\python.exe -m unittest discover -s tests -q
node tests/test_shot_replay_ui.cjs
node tests/test_player_shots_ui.cjs runs/raw-play-ready/replay.html
node tests/test_calibration_review_ui.cjs runs/court-bounce-ready/calibration.html
node tests/test_event_review_ui.cjs
```

CI (`.github/workflows/tests.yml`) runs the same on every push and PR without footage
or weights, so tests must not require local-only assets (skip cleanly if they would).
Add tests for every changed contract. JS suites are fake-DOM logic checks. For UI
changes, also look at the page in a browser (serve `project/runs` on 127.0.0.1) and
say in the PR what you did and did not see.

## Long-running compute

- Inference runs are hours long (about 5 s/frame on GPU, about 15 s/frame on CPU). **Never
  start them inside a Claude session.** Session teardown kills child processes.
  Put the exact command in the PR/notes for John to run in his own terminal
  (`project/Run-Game.ps1`; rerunning resumes).
- Before using the GPU, check that no run is active: look for a recent `[longrun]` line
  in `project/runs/*.log`, or a growing `runs/<run>/progress.json`.
- Don't edit analysis code (files listed in `longrun.CODE_FILES`) in the working tree a
  run is using. Use a separate git worktree.

## Git and pull requests

- Never push to `main`, never force-push shared branches, never merge your own PR.
- One task = one branch = one PR. Branch name: `claude/<issue-number>-<short-slug>`
  (or `claude/<slug>` without an issue). Base on `main` unless the task says otherwise.
- Small, reviewable commits with messages that say what and why. End commit messages
  with the Co-Authored-By line the harness provides.
- Fill in `.github/pull_request_template.md` completely. Open as **draft** if anything
  is unverified or a decision is pending. Reference the issue (`Closes #N`).
- If a task turns out bigger or riskier than the issue suggests, stop at a safe point,
  push what works, and explain in the PR instead of pushing through.
- `gh` may not be installed. If it isn't, push the branch and give John the
  `https://github.com/jfrench9404/tennis-scraper/compare/main...<branch>` link.
