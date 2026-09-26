# Raw footage: play context and per-stroke execution

Ready replay: `runs/raw-play-ready/replay.html`. Detection and pose do not need
to be rerun. Your reviewed court, bounces, original ball observations, and previous
replays remain unchanged. The focused far-player pose cache is reused.

## What changed

The activity list defaults to shot candidates/confirmed shots, not every swing.
Use the filter above the video to inspect uncertain contacts or likely non-play.

- **Shot candidate:** local contact cues plus an observed court-directed flight
  and serve pattern or incoming-flight context. Still not human-confirmed.
- **Uncertain contact:** insufficient evidence that a shot occurred. Not counted
  as a shot, and not used as a flight boundary or contact constraint.
- **No shot (candidate):** sustained local ball activity on both sides of the
  movement; often handling/dribbling. This is a hypothesis, not an automatic
  human rejection. Confirmed/rejected human decisions take precedence.
- **Unknown stroke:** a supported/confirmed contact whose stroke type is unresolved.

Play phases are provisional: preparation, serve, live ball, between-points, or
unresolved. Camera cuts, tracking jumps and long observation gaps stop evidence
windows. Ball rays are not projected onto the floor to decide airborne flight.
Missing observations cannot establish an ace, fault, winner or return.

## Hands are properties of strokes, not player restrictions

The classifier uses local racquet/wrist evidence around each event. It supports
left and right one-handed strokes by the same player in the same clip. Nearby
hands can suggest two-handed execution, but this does not prove a grip, a leading
hand, or a backhand. Two-handed forehand versus backhand remains unresolved unless
there is enough evidence or a human supplies the stroke type.

Select an event, choose Reviewed type and independently choose Reviewed hand(s)
(left/right/both/cannot determine), then Save correction and Export shot corrections.
Both hands can be paired with forehand OR backhand, on either anatomical side.
No permanent dominant-hand choice is required. Legacy `-NearHand`/`-FarHand`
parameters remain descriptive only; they do not override stroke evidence.

Apply the exported type/hand feedback:

```powershell
.\Run-Player-Shots.ps1 -StrokeReview "$env:USERPROFILE\Downloads\tennis-shot-corrections.json"
```

This creates a new replay using the cached poses. Type/hand feedback does not
confirm contact, retrain a model, or alter flight constraints. Use the separate
contact Confirm/Unsure/Not a hit controls and `-ContactReview` for that.
If changing contact timing or identity, apply contacts first, then export stroke
feedback from that rebuilt replay. Matching analysis bindings prevent stale
feedback from being silently applied to a changed event.

## This clip's current result

One automatic shot candidate (serve); three likely non-play activities; seven
uncertain contacts, including both far-side swing windows. The eight local
contact proposals from the previous version are no longer called eight shots.
The raw proposals remain available as evidence, not verified contacts.

You described the footage as an ace. That is useful review context, but no
hard-coded ace/return rule was added. The automated outcome remains undetermined.
No exact point-boundary annotation or frame-level ground-truth accuracy score
has been established for this raw clip.

## Verification

Tests cover walking/ball handling, missing observations, detector jumps, serve
flight without inventing a return or ace, per-stroke hand switching, two-handed
strokes on both sides, and independent bound review imports. UI logic is tested
offline; actual browser rendering was not automated. These are heuristic rules,
not a trained tennis-action model or a measured accuracy guarantee.
