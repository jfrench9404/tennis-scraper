# Ground-truth labels (`labels.json`), schema version 1

A `labels.json` file holds **John's own decisions** about what happened in a stretch of
video: where each shot was hit, who hit it and what kind of shot it was; where each bounce
was and whether it was in; and where each point started and ended, who served and who won.
It is the only thing an accuracy number may be measured against (CLAUDE.md rule 3).

- One file is bound to **one run** (its `run_id`) and **one source video** (the SHA-256 of
  the video file the run was made from). Scoring refuses a file that does not match.
- Every entry is a human decision (`"decision": "human"`). Pipeline candidates are never
  copied in as labels. A candidate John looked at and agreed with is fine to label, because
  the decision is then his.
- **No tool ever modifies a labels file.** The validator and the scorer only read it; the
  scorer records the file's SHA-256 in its output and refuses to write its output over it.
  To change a label, John edits the file (or the labelling page writes a new file).
- Labels are made on John's laptop from his own footage and are not uploaded. Commit a
  labels file only if John says so.

Code: `project/tennis_vision/labels.py` (validator, vocabularies),
`project/tennis_vision/score_labels.py` (scorer).

## Frame numbers: always source-video frames

Every frame number in a labels file is a **source-video frame**: the 0-based index of the
frame as decoded from the original video file whose SHA-256 is in the binding (frame 0 is
the first frame of that file). This is the same numbering as `source_frame` in
`event-candidates.json` and `source_start_frame` in `source-offset.json`,
`review-report.json`, `replay-report.json` and `longrun-manifest.json`.

Pipeline outputs mostly use **run-relative** frames instead (the `frame` field in
`event-candidates.json`, `reviewed-events.json`, `shot-candidates.json`,
`contact-candidates.json`, `play-context.json` and a long run's `shots.json`). A run can
start part-way into the video; the baseline rally run starts at `source_start_frame` 4740.
The conversion is:

```
source_frame = run_frame + source_start_frame
run_frame    = source_frame - source_start_frame
```

`score_labels` reads `source_start_frame` from the run folder, converts every candidate to
source frames, and does all matching in source frames. Its output shows both numbers. If a
candidate file carries its own `source_frame` and it disagrees with this conversion, the
scorer stops with an error rather than guess the alignment.

A labelling tool that shows the run's review video (`review/source.mp4`, which starts at
run frame 0) must add `source_start_frame` before writing a label.

## File layout

Example (frame numbers and ids are illustrative, not real labels):

```json
{
  "schema_version": 1,
  "kind": "tennis_ground_truth_labels",
  "binding": {
    "run_id": "39449302177328c739e679a953b51d001bf297e21541feb1cc38c76265aa9ab9",
    "source_video_sha256": "0eb4675b7e21a672d35b1ac8c017dde81b3cb175877c48ce60de0e34151b104b",
    "fps": 30.0,
    "source_video_name": "sebbie-demo-shortest (1).mp4"
  },
  "labeller": {"name": "John", "date": "2026-09-28"},
  "shot_types": ["serve", "forehand", "backhand", "volley", "overhead", "other", "unsure"],
  "coverage": [
    {"source_start_frame": 4740, "source_end_frame": 5339, "kinds": ["shots", "bounces", "points"]}
  ],
  "shots": [
    {"id": "shot-1", "source_frame": 4797, "hitter": "near", "shot_type": "serve",
     "decision": "human", "notes": "first serve"}
  ],
  "bounces": [
    {"id": "bounce-1", "source_frame": 4861, "call": "in", "decision": "human"}
  ],
  "points": [
    {"id": "point-1", "source_start_frame": 4770, "source_end_frame": 5100,
     "server": "near", "winner": "far", "score_text": "15-0", "decision": "human"}
  ],
  "notes": "optional free text about the whole file"
}
```

### Top level

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | yes | Always `1`. |
| `kind` | yes | Always `"tennis_ground_truth_labels"`. |
| `binding` | yes | Which run and video these labels belong to (below). |
| `labeller` | yes | `{"name": non-empty text, "date": "YYYY-MM-DD"}`: who made the decisions and when. |
| `shot_types` | yes | The shot-type list used for this file. Must be exactly the current list (see "Shot types"), so a file made with an older list is rejected instead of silently mis-scored. |
| `coverage` | yes | Spans that were **fully labelled** (below). May be empty. |
| `shots`, `bounces`, `points` | yes | Lists of entries (may be empty). |
| `notes` | no | Free text. |

Unknown fields are rejected (at every level) so a typo cannot silently drop a decision.

### `binding`

| Field | Required | Meaning |
|---|---|---|
| `run_id` | yes | 64 hex characters. For an event-review or replay folder this is the review fingerprint (`run_id` in `review-report.json` / `event-candidates.json` / `reviewed-events.json`, `review_run_id` in `replay-report.json`). For a long-run folder it is `fingerprint` in `longrun-manifest.json`. |
| `source_video_sha256` | yes | 64 hex characters, SHA-256 of the whole original video file (as `input.sha256` in `longrun-manifest.json`). Not the hash of a review clip. |
| `fps` | yes | Frames per second of the source video; must match the run. |
| `source_video_name` | no | File name, for people. Never used for matching. |

### `coverage`: what "fully labelled" means

Each coverage span is `{"source_start_frame": a, "source_end_frame": b, "kinds": [...]}`
with `a <= b`, both inclusive, and `kinds` a non-empty subset of `"shots"`, `"bounces"`,
`"points"`. It means: **between frames a and b, every shot / bounce / point of the listed
kinds has a label.** A frame inside the span with no label is a decision too ("nothing
happened here"). Spans of the same kind must not overlap.

Only covered spans are scored:

- A labelled entry is scored only if it lies inside a span covering its kind (a point must
  lie wholly inside one). Entries outside coverage are allowed (spot labels) but are only
  counted, never used for precision or recall.
- A pipeline candidate that matches a scored label counts as a true positive. An unmatched
  candidate is a false positive only if its frame lies inside a span covering that kind;
  otherwise it is counted as "outside the labelled span" and ignored.
- Coverage is clipped to the frames the run actually processed
  (`source_start_frame` .. `source_start_frame + frames - 1`), and labels outside the run's
  frames are counted separately.

### `shots`

| Field | Required | Values |
|---|---|---|
| `id` | yes | Unique text within the file (across all lists). |
| `source_frame` | yes | Integer >= 0. The frame of racquet-ball contact, as best John can see it. |
| `hitter` | yes | `"near"` or `"far"` (the player nearer / farther from the camera). |
| `shot_type` | yes | One of `shot_types`. `"unsure"` means John saw the hit but will not name the type. |
| `decision` | yes | Always `"human"`. |
| `notes` | no | Text. |

Two shots may not share a `source_frame`.

### `bounces`

| Field | Required | Values |
|---|---|---|
| `id` | yes | Unique text. |
| `source_frame` | yes | Integer >= 0. The frame the ball touches the ground. |
| `call` | yes | `"in"`, `"out"` or `"unsure"`. |
| `decision` | yes | Always `"human"`. |
| `notes` | no | Text. |

Two bounces may not share a `source_frame`.

### `points`

| Field | Required | Values |
|---|---|---|
| `id` | yes | Unique text. |
| `source_start_frame` | yes | Integer >= 0. First frame of the point (e.g. the serve toss). |
| `source_end_frame` | yes | Integer >= `source_start_frame`. Last frame of the point (the ball is dead). |
| `server` | yes | `"near"`, `"far"` or `"unknown"`. |
| `winner` | yes | `"near"`, `"far"` or `"unknown"`. |
| `score_text` | no | Free text such as `"30-15"`; never parsed or checked. |
| `decision` | yes | Always `"human"`. |
| `notes` | no | Text. |

Points may not overlap each other.

There are **no spin, speed or RPM fields** (CLAUDE.md rule 3).

## Shot types (default, pending John's confirmation)

`serve`, `forehand`, `backhand`, `volley`, `overhead`, `other`, `unsure`.

The pipeline's own stroke classes are `serve`, `forehand`, `backhand`, `volley`,
`overhead`, `unknown` and `not_a_shot`. For shot-type agreement the scorer compares only
where both sides name a type: a label of `unsure` and a pipeline `unknown` are abstentions
and counted separately; `not_a_shot`, and pipeline types that were set by human review
(`classification_status: "human_reviewed"`), are also counted separately and never compared.

## Scoring (`score_labels`)

```
python -m tennis_vision.score_labels --labels <labels.json> --run <run or replay folder> --output <new.json>
    [--tolerance-frames 3] [--points <proposals.json>] [--point-tolerance-s 1.0]
    [--video <source video>] [--include-review-windows]
```

- `--run` may be a replay folder (`reviewed-events.json`, `shot-candidates.json`,
  `replay-report.json`), an event-review folder (`event-candidates.json`,
  `review-report.json`), a folder holding `replay/` and/or `review/` (replay is used), or a
  long-run folder (`longrun-manifest.json` + `shots.json`).
- Hits and bounces: a candidate matches a label of the same kind when their source frames
  differ by at most `--tolerance-frames` (default 3, pending John's decision). Matching is
  one-to-one, closest pairs first. Pipeline candidates are scored at the frame the pipeline
  produced (`original_frame` if a reviewer moved it). Human-added manual events and review
  windows (`support: "review_window_only"`, explicitly "not a detected hit") are not
  candidates unless `--include-review-windows` is given.
- Precision = matched / candidates inside covered spans; recall = matched / labels inside
  covered spans. Both are reported with their counts, and `null` when the denominator is 0.
- Matched hits also report hitter agreement and shot-type agreement (abstentions apart).
- Points: `--points` takes a list of proposals, or `{"points": [...], "run_id": ...}`. Each
  proposal is `{"start_frame", "end_frame"}` in **run-relative** frames, or
  `{"source_start_frame", "source_end_frame"}` in source frames; optional `server` and
  `winner`. A proposal matches a labelled point when both its start and end are within
  `--point-tolerance-s` seconds (default 1.0, pending John's decision).
- The source-video hash of the run is taken from its `longrun-manifest.json`, or from the
  long-run folder named by `source_run` / `input_run` (checked to cover the same frames).
  A legacy run that recorded no hash needs `--video <file>`, which is hashed locally.
- Every statement names the sample and its file, e.g. *"Hit candidates: accuracy on 12
  labelled shots from labels.json (source frames 4740-5339): precision 0.800 (8/10),
  recall 0.667 (8/12)"*. Nothing is claimed outside the covered span.

Validate a file on its own (and optionally against a run) with:

```
python -m tennis_vision.labels <labels.json> [--run <folder>] [--video <source video>]
```
