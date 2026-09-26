# Far-player pose and contact/shot candidates

This documents the previous contact pass. The current replay is
`runs/raw-play-ready/replay.html`; see [RAW-PLAY.md](RAW-PLAY.md). Global hand
settings no longer constrain stroke classification, and local contact proposals
must pass raw-footage play-context checks before being counted as shots.

Ready result: `runs/player-shots-final/replay.html`. No rerun needed to inspect it.
On the supplied 600-frame clip: 70 far-player skeletons replaced, usable body-pose
coverage (at least six supported body joints) increased from 472 to 497 frames.
Both player roots remain visible on all 600 frames. This is a coverage metric,
not a labeled accuracy score. Eight contact candidates pass the new gates (near
player); two far-player swing windows are separately queued for manual review.
There are no automatically supported far-player contacts on this clip yet.
One serve is classified; remaining types abstain, including both far windows.
Handedness is unresolved for both players; explicit hand settings can help but
cannot fix missing contact/pose evidence. Ball observations and bounce decisions
are unchanged. CPU crop inference took about 8.5 minutes; cached rebuilds take seconds.

Run `Run-Player-Shots.ps1` from this project. It creates a new timestamped replay,
keeps your reviewed court/bounce corrections, and opens the replay on success.
The first pass reruns only focused far-player pose crops (progress bar and ETA).
Subsequent runs with identical inputs reuse a fingerprint-bound pose cache.
The ball detector is not rerun. Raw events and previous runs are unchanged.

```powershell
.\Run-Player-Shots.ps1
```

If you know the players' dominant hands, supply `-NearHand left` or
`-FarHand right` (either hand is accepted). Defaults are automatic and abstain
when racquet/wrist evidence conflicts. Do not guess to increase label counts.
Use `-CalibrationRun 'runs\your-reviewed-run'` to select another calibration.

## What to inspect

Open **Enlarged far-player evidence** for source pixels and fresh model joints.
Select a shot on the left, then **Replay movement**. Purple circles indicate
candidate contact proximity, not exact ball/string contact. The evidence panel
shows the proposed contact frame/range, equipment distances, ball turn, and
body-relative swing. Shot corrections are separate from contact confirmation.

To review a contact, select its shot, pause and step to the contact, choose the
hitter, then Confirm/Unsure/Not a hit. Export contact review and apply it:

```powershell
.\Run-Player-Shots.ps1 -ContactReview "$env:USERPROFILE\Downloads\tennis-contact-review.json"
```

Only explicit contact decisions affect reviewed flight constraints. Exported
shot-type corrections are feedback labels and do not confirm contact events.
Contact files are bound to the exact underlying analysis; mismatched files fail
instead of silently applying to another clip. Timing corrections are limited to
0.3 seconds around the selected candidate.

Types supported: serve, forehand, backhand, volley, overhead, and unknown.
These are explainable temporal rules, not a trained tennis shot classifier.
Unknown handedness, missing poses, camera cuts, and insufficient incoming-flight
evidence cause abstention. Volleys require more than absence of a detected bounce.
Spin/RPM is not estimated by this update.

Far-player **swing review windows** are weaker than contact candidates: they
combine wrist movement and nearby observed ball pixels, but lack a clean impact
signal. They are excluded from automatic shot classification and all flight
constraints until explicitly confirmed by a human. Do not assume they are hits.

## Safety and limits

- Crops are anchored to an existing same-frame identity, with overlap, scale,
  appearance, court, and ambiguous-person gates. They do not search the full frame
  for a replacement player or invent joints when inference fails.
- A whole observed skeleton replaces the old one only if its joint coverage or
  score improves. More supported joints is not a measured accuracy increase.
- Initial back-associated identities remain estimated; only new crop observations
  may add those poses to contact evidence. Legacy backfill remains display-only.
- Contact candidates require observed ball motion on both sides, an abrupt change,
  wrist/racquet proximity, and body-relative swing. No gap-filled balls are evidence.
- Existing human decisions are preserved. Candidate contacts do not become
  reviewed flight constraints automatically. Timing remains frame-level (30 fps).
- Far and near shot counts are reported separately in `shot-candidates.json`.

## Environment

Inference uses a separate project-local `.venv-player`, leaving `.venv-ball`
and your system Python unchanged. The local `yolo26l-pose.pt` must exist.
There is no implicit model download during analysis. To recreate the runtime:

```powershell
.\.venv-ball\Scripts\python.exe -m venv --system-site-packages .venv-player
.\.venv-player\Scripts\python.exe -m pip install "ultralytics>=8.4,<9" tqdm scipy
```

API reference: [Ultralytics pose prediction](https://docs.ultralytics.com/tasks/pose/).
Neither footage nor annotations are uploaded for inference.

Verified runtime: Python 3.13, Ultralytics 8.4.163, Torch 2.14.0 (CPU).
Automated checks include Python regression tests and offline JavaScript UI logic.
Source-frame pose comparisons were inspected; actual browser rendering was not
automated or independently verified.
