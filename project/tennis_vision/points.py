"""Proposed point start/end boundaries, for John to confirm. Nothing here is a point yet.

Reads a replay folder (``replay-data.json`` + ``play-context.json`` +
``shot-candidates.json``, as written by ``shot_replay``), a game folder
(``<root>/replay/...``, as written by ``game_pipeline``) or, with no replay yet,
a run folder (``events.jsonl`` or complete long-run chunks; then there are no
serve candidates and so no proposals, only dead-ball evidence).

A proposal STARTS at a serve candidate: a hit that ``play_context`` marked
``serve_candidate`` (baseline + overhead contact + raised arm + outgoing court
flight) and/or that ``shot_classification`` labelled ``serve``. The server side
is that event's player (near/far) and the confidence basis lists which sources
agree; it is qualitative, not a probability.

A proposal ENDS at the earliest of these end signals after its (last) serve:

* ``no_ball_observed``: no observed ball for ``dead_ball_seconds`` or more; the
  end is the last observed ball before that gap;
* ``between_points_activity``: ``play_context`` ``no_shot_candidate`` (the ball
  stays with a player on both sides, e.g. bouncing it before a serve); the end
  is just before that ball handling starts;
* ``break_proposal``: a ``segments`` (#24) changeover/break proposal starts;
* ``next_serve_candidate``: the frame before the next serve candidate;
* otherwise the data ends (``open_end``).

A ball or break signal is ignored when a later non-serve shot candidate
follows it inside the same window: a missing ball mid-rally is a detector gap,
not the end of the point. Players walking towards or away from their own
baseline is recorded as corroborating evidence only, never as an end on its
own, because rally footwork can look like walking.

Two serve candidates by the same server less than ``min_serve_gap_seconds``
apart, with no rally shot or break between them, are a REPEAT serve candidate
(possible let, second serve, or a very short point). By default they stay in
one proposal (``--repeat-serve same_point``); ``new_point`` splits them. That
choice, and every threshold, is John's decision.

Frames: ``run_frame`` is the index inside the replay/run (0 = its first frame);
``source_frame`` = the replay's ``source_start_frame`` + ``run_frame`` (frame of
the original video). Seconds are frame timestamps (``frame / fps``); end frames
are inclusive. The top-level ``points`` list is a plain view for a label scorer
(#27): ``source_start_frame``, ``source_end_frame``, ``server``. Proposals never claim a winner, score, ace, fault, let, or a
winning shot, and nothing is written into the input folder.
"""
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from .segments import DEFAULTS as SEGMENT_DEFAULTS, _inside, absent_runs, load_run, observe, propose as propose_segments

SCHEMA_VERSION = 1
COURT_LENGTH_M = 23.77
DEFAULTS = {
    "dead_ball_seconds": 3.0,        # no observed ball this long ends a point (unless rally shots follow)
    "min_serve_gap_seconds": 10.0,   # same-server serve candidates closer than this are a repeat serve
    "ball_bridge_seconds": 0.2,      # a stray ball detection shorter than this does not split a dead-ball gap
    "walk_window_seconds": 2.0,      # window for the walking-to/from-baseline corroboration
    "walk_min_metres": 1.0,          # minimum change in distance to own baseline over that window
    "walk_max_speed_mps": 2.5,       # faster than this is running, not walking
}
REPEAT_POLICIES = ("same_point", "new_point")
LIVE_STATES = ("shot_candidate", "confirmed_shot")
LIVE_PHASES = ("serve_candidate", "live_ball_candidate")
END_SIGNALS = {
    "no_ball_observed": "no observed ball for at least dead_ball_seconds",
    "between_points_activity": "play_context no-shot candidate: ball stays with a player (e.g. bouncing it)",
    "break_proposal": "segments (#24) possible changeover or break starts",
    "next_serve_candidate": "next serve candidate",
    "data_end": "the data ends (open end: the point may continue)",
}
MEANING = ("Point PROPOSALS for John to confirm or reject; nothing is a point until he does. Starts are serve "
           "candidates (heuristic, not confirmed serves); ends are dead-ball evidence. No winner, score, ace, "
           "fault, let or winning shot is inferred: missing observations prove nothing. Coverage is not accuracy.")
SCORING_TODO = ("TODO(#27): score_labels is not merged into dev yet. The top-level 'points' list follows the "
                "--points input described in #27's draft docs/labels-schema.md ({'points': [{source_start_frame, "
                "source_end_frame, server}], 'run_id'}, source frames, inclusive); check it against the scorer "
                "once #27 lands. No 'winner' is ever given.")


# ----------------------------------------------------------------------------- loading

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _court_y(identity: str, xy: list[float]) -> float:
    """Distance from the player's OWN baseline along the court (negative = behind it)."""
    return xy[1] if identity == "near" else COURT_LENGTH_M - xy[1]


def _replay_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-frame observations from replay-data frames. Replay feet are centred on the net; convert to court metres."""
    out = []
    for f in frames:
        item: dict[str, Any] = {"ball": f.get("ball_observed_2d") is not None, "near": None, "far": None,
                                "near_seen": False, "far_seen": False,
                                "phase": (f.get("play_phase") or {}).get("phase", "unresolved")}
        for p in f.get("players", []):
            identity = p.get("identity_id")
            if identity not in ("near", "far") or p.get("predicted"):
                continue
            item[identity + "_seen"] = True
            feet = p.get("feet_xyz_m")
            if feet and all(isinstance(v, (int, float)) and math.isfinite(v) for v in feet[:2]):
                item[identity] = [feet[0] + 5.485, feet[1] + 11.885]
        out.append(item)
    return out


def _row_frames(rows: dict[int, dict[str, Any]], first: int, last: int) -> list[dict[str, Any] | None]:
    out: list[dict[str, Any] | None] = []
    for source in range(first, last + 1):
        row = rows.get(source)
        if row is None:
            out.append(None)
            continue
        seen = observe(row)
        item: dict[str, Any] = {"ball": seen["ball"], "near": None, "far": None, "near_seen": seen["near"],
                                "far_seen": seen["far"], "phase": "unresolved"}
        for t in row.get("tracks", []):
            if (t.get("label") == "player" and t.get("identity_id") in ("near", "far") and not t.get("predicted")
                    and t.get("court_m") and len(t["court_m"]) >= 2):
                item[t["identity_id"]] = [float(t["court_m"][0]), float(t["court_m"][1])]
        out.append(item)
    return out


def load_input(path: Path, source: str = "auto") -> dict[str, Any]:
    """Normalise a replay, game or run folder into frames + events. Frames are run-relative."""
    path = Path(path)
    replay = path if (path / "replay-data.json").is_file() else (
        path / "replay" if (path / "replay" / "replay-data.json").is_file() else None)
    if replay is not None:
        data = _read_json(replay / "replay-data.json")
        report = data["report"]
        play = _read_json(replay / "play-context.json") if (replay / "play-context.json").is_file() \
            else report.get("play_context")
        shots = _read_json(replay / "shot-candidates.json") if (replay / "shot-candidates.json").is_file() \
            else data.get("shots")
        frames = _replay_frames(data["frames"])
        return {"kind": "replay" if replay == path else "game", "path": str(path), "replay": str(replay),
                "run_id": report.get("review_run_id"),
                "fps": float(report["fps"]), "source_start_frame": int(report.get("source_start_frame") or 0),
                "frames": frames, "events": data.get("events", []),
                "assessments": (play or {}).get("assessments", []), "shots": (shots or {}).get("shots", []),
                "inputs": [n for n in ("replay-data.json", "play-context.json", "shot-candidates.json")
                           if (replay / n).is_file()], "partial": False,
                "notes": [] if play else ["No play-context in this replay: there are no serve candidates."]}
    run = path / "run" if (path / "run").is_dir() and not (path / "events.jsonl").is_file() else path
    loaded = load_run(run, source)
    rows = loaded["rows"]
    first, last = min(rows), max(rows)
    manifest = run / "longrun-manifest.json"
    run_id = _read_json(manifest).get("fingerprint") if manifest.is_file() else None
    return {"kind": "run", "path": str(path), "replay": None, "run_id": run_id, "fps": loaded["fps"],
            "source_start_frame": first,
            "frames": _row_frames(rows, first, last), "events": [], "assessments": [], "shots": [],
            "inputs": [loaded["source"]], "partial": loaded["partial"],
            "notes": ["Run rows only (no replay yet): no play_context or shot classification, so no serve "
                      "candidates and no point proposals. Build the replay first (game_pipeline stage 4)."]}


# ----------------------------------------------------------------------------- evidence

def serve_candidates(loaded: dict[str, Any]) -> list[dict[str, Any]]:
    """Serve candidates from play_context and shot_classification, merged by event id."""
    events = {e["id"]: e for e in loaded["events"] if "id" in e}
    shots = {s["event_id"]: s for s in loaded["shots"] if "event_id" in s}
    found: dict[str, dict[str, Any]] = {}
    for a in loaded["assessments"]:
        if a.get("phase") == "serve_candidate" or (a.get("serve_pattern") and a.get("action_state") in LIVE_STATES):
            found.setdefault(a["event_id"], {"frame": a["frame"], "sources": []})["sources"].append("play_context")
    for s in shots.values():
        if (s.get("classification") == "serve" and s.get("event_status") != "rejected"
                and s.get("action_state") not in ("no_shot", "no_shot_candidate", "uncertain_contact")):
            found.setdefault(s["event_id"], {"frame": s["frame"], "sources": []})["sources"].append(
                "shot_classification")
    out = []
    for event_id, item in found.items():
        event, shot = events.get(event_id, {}), shots.get(event_id, {})
        if event.get("status") == "rejected":
            continue  # a human rejected this contact
        assessment = next((a for a in loaded["assessments"] if a.get("event_id") == event_id), {})
        side = shot.get("player_id") or event.get("player_id")
        basis = {"sources": sorted(set(item["sources"])),
                 "agreement": "play_context_and_shot_classification" if len(set(item["sources"])) == 2
                 else item["sources"][0] + "_only",
                 "play_context_serve_pattern": bool(assessment.get("serve_pattern")),
                 "play_context_reason": assessment.get("reason"),
                 "outgoing_flight": (assessment.get("outgoing") or {}).get("relation"),
                 "classification": shot.get("classification"), "classification_support": shot.get("support"),
                 "classification_reasons": shot.get("reasons", []),
                 "contact_support": shot.get("contact_support") or event.get("contact_support"),
                 "event_status": event.get("status") or shot.get("event_status"),
                 "note": "Qualitative agreement of heuristic sources, not a calibrated probability. A confirmed "
                         "event status means a human confirmed the CONTACT, not that it was a serve."}
        feet = (shot.get("evidence") or {}).get("player_feet_court_m")
        out.append({"event_id": event_id, "run_frame": int(item["frame"]),
                    "server_side": side if side in ("near", "far") else "unknown",
                    "server_side_basis": "player_id of the serve-candidate event" if side in ("near", "far")
                    else "event has no established player identity",
                    "server_feet_court_m": feet, "confidence_basis": basis})
    return sorted(out, key=lambda s: (s["run_frame"], s["event_id"]))


def live_shots(loaded: dict[str, Any], serve_ids: set[str]) -> list[dict[str, Any]]:
    return sorted(({"event_id": a["event_id"], "run_frame": int(a["frame"])} for a in loaded["assessments"]
                   if a.get("action_state") in LIVE_STATES and a["event_id"] not in serve_ids),
                  key=lambda s: s["run_frame"])


def between_points_activity(loaded: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for a in loaded["assessments"]:
        if a.get("action_state") != "no_shot_candidate":
            continue
        start = min([int(a["frame"])] + [int(f) for f in (a.get("incoming") or {}).get("observed_frames", [])])
        out.append({"event_id": a["event_id"], "run_frame": int(a["frame"]), "activity_start_run_frame": start,
                    "reason": a.get("reason")})
    return sorted(out, key=lambda s: s["activity_start_run_frame"])


def dead_ball_gaps(frames: list[dict[str, Any] | None], fps: float, seconds: float,
                   bridge_seconds: float) -> list[dict[str, Any]]:
    values = [None if f is None else f["ball"] for f in frames]
    runs = absent_runs(values, max(1, int(math.ceil(seconds * fps))), int(round(bridge_seconds * fps)))
    return [{"start_run_frame": r["start"], "end_run_frame": r["end"], "duration_s": round((r["end"] - r["start"] + 1)
             / fps, 3), "open_start": r["open_start"], "open_end": r["open_end"]} for r in runs]


def walking_episodes(frames: list[dict[str, Any] | None], fps: float, window_s: float, min_m: float,
                     max_speed: float) -> list[dict[str, Any]]:
    """Sustained, fairly straight, walking-speed movement towards or away from a player's own baseline."""
    window = max(2, int(round(window_s * fps)))
    step = max(1, int(round(fps / 5)))
    episodes = []
    for identity in ("near", "far"):
        pos = [None if f is None else f[identity] for f in frames]
        hits: list[tuple[int, int, str]] = []
        for i in range(0, len(pos) - window, step):
            j = i + window
            if pos[i] is None or pos[j] is None:
                continue
            samples = [pos[k] for k in range(i, j + 1, step) if pos[k] is not None]
            if len(samples) < 0.6 * (window / step + 1):
                continue
            steps = [math.dist(a, b) for a, b in zip(samples, samples[1:])]
            path = sum(steps)
            toward = _court_y(identity, pos[i]) - _court_y(identity, pos[j])
            # Straight, walking pace on average, and no sprint burst (a sampled step above twice walking pace).
            if (abs(toward) >= min_m and path > 0 and math.dist(pos[i], pos[j]) / path >= 0.7
                    and path / (window / fps) <= max_speed and max(steps) <= 2 * max_speed * step / fps):
                hits.append((i, j, "towards_own_baseline" if toward > 0 else "away_from_own_baseline"))
        for i, j, direction in hits:
            last = episodes[-1] if episodes else None
            if last and last["player"] == identity and last["direction"] == direction \
                    and i <= last["end_run_frame"]:
                last["end_run_frame"] = j
            else:
                episodes.append({"player": identity, "direction": direction, "start_run_frame": i, "end_run_frame": j})
    for e in episodes:
        a, b = frames[e["start_run_frame"]][e["player"]], frames[e["end_run_frame"]][e["player"]]
        e["baseline_distance_change_m"] = round(_court_y(e["player"], a) - _court_y(e["player"], b), 2)
        e["duration_s"] = round((e["end_run_frame"] - e["start_run_frame"]) / fps, 3)
    return sorted(episodes, key=lambda e: (e["start_run_frame"], e["player"]))


def _segment_rows(loaded: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {}
    for i, f in enumerate(loaded["frames"]):
        if f is None:
            continue
        tracks = [{"label": "player", "identity_id": k, "predicted": False} for k in ("near", "far") if f[k + "_seen"]]
        if f["ball"]:
            tracks.append({"label": "ball", "predicted": False})
        rows[loaded["source_start_frame"] + i] = {"tracks": tracks}
    return rows


def break_proposals(loaded: dict[str, Any], segments_report: dict[str, Any] | None) -> tuple[list, dict]:
    """#24 break proposals: from a given segments JSON, else computed with segments' defaults on these frames."""
    if segments_report is not None:
        if segments_report.get("kind") != "segment_proposals":
            raise ValueError("--segments is not a segments.py proposal file")
        if abs(float(segments_report.get("fps", 0)) - loaded["fps"]) > 1e-6:
            raise ValueError("--segments fps does not match this run")
        info = {"source": "file", "thresholds": segments_report.get("thresholds")}
        proposals = segments_report.get("proposals", [])
    else:
        report = propose_segments({"fps": loaded["fps"], "rows": _segment_rows(loaded), "run": loaded["path"],
                                   "source": "replay_frames" if loaded["replay"] else "rows", "partial": False,
                                   "selection_frames": len(loaded["frames"])})
        info = {"source": "computed_with_segments_defaults", "thresholds": dict(SEGMENT_DEFAULTS)}
        proposals = report["proposals"]
    out = []
    for p in proposals:
        s, e = (int(v) - loaded["source_start_frame"] for v in p["source_frames"])
        out.append({"id": p["id"], "start_run_frame": s, "end_run_frame": e, "reasons": p.get("reasons", []),
                    "status": p.get("status")})
    return out, info


# ----------------------------------------------------------------------------- proposals

def propose(loaded: dict[str, Any], dead_ball_seconds: float = DEFAULTS["dead_ball_seconds"],
            min_serve_gap_seconds: float = DEFAULTS["min_serve_gap_seconds"],
            ball_bridge_seconds: float = DEFAULTS["ball_bridge_seconds"],
            walk_window_seconds: float = DEFAULTS["walk_window_seconds"],
            walk_min_metres: float = DEFAULTS["walk_min_metres"],
            walk_max_speed_mps: float = DEFAULTS["walk_max_speed_mps"],
            repeat_serve: str = "same_point", segments_report: dict[str, Any] | None = None) -> dict[str, Any]:
    thresholds = {"dead_ball_seconds": dead_ball_seconds, "min_serve_gap_seconds": min_serve_gap_seconds,
                  "ball_bridge_seconds": ball_bridge_seconds, "walk_window_seconds": walk_window_seconds,
                  "walk_min_metres": walk_min_metres, "walk_max_speed_mps": walk_max_speed_mps}
    for key, value in thresholds.items():
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
            raise ValueError(f"{key} must be a finite number >= 0")
    if repeat_serve not in REPEAT_POLICIES:
        raise ValueError(f"repeat_serve must be one of {REPEAT_POLICIES}")
    fps, offset, frames = loaded["fps"], loaded["source_start_frame"], loaded["frames"]
    last_frame = len(frames) - 1

    def at(f: int) -> dict[str, Any]:
        return {"run_frame": f, "source_frame": offset + f, "run_seconds": round(f / fps, 3),
                "source_seconds": round((offset + f) / fps, 3)}

    serves = serve_candidates(loaded)
    serve_ids = {s["event_id"] for s in serves}
    shots = live_shots(loaded, serve_ids)
    activity = between_points_activity(loaded)
    gaps = dead_ball_gaps(frames, fps, dead_ball_seconds, ball_bridge_seconds)
    walks = walking_episodes(frames, fps, walk_window_seconds, walk_min_metres, walk_max_speed_mps)
    breaks, break_info = break_proposals(loaded, segments_report)

    # Group serve candidates: a close repeat by the same server, with no rally shot or break between, is one group.
    groups: list[list[dict[str, Any]]] = []
    for s in serves:
        prev = groups[-1][-1] if groups else None
        s["repeat_of"] = None
        if prev is not None:
            gap_s = (s["run_frame"] - prev["run_frame"]) / fps
            rally_between = any(prev["run_frame"] < x["run_frame"] < s["run_frame"] for x in shots)
            break_between = any(prev["run_frame"] < b["start_run_frame"] < s["run_frame"] for b in breaks)
            if (gap_s < min_serve_gap_seconds and s["server_side"] == prev["server_side"] != "unknown"
                    and not rally_between and not break_between):
                s["repeat_of"] = prev["event_id"]
                if repeat_serve == "same_point":
                    groups[-1].append(s)
                    continue
        groups.append([s])

    proposals = []
    for n, group in enumerate(groups):
        start, last_serve = group[0]["run_frame"], group[-1]["run_frame"]
        next_start = groups[n + 1][0]["run_frame"] if n + 1 < len(groups) else None
        window_end = next_start - 1 if next_start is not None else last_frame

        def followed_by_rally(f: int) -> bool:
            return any(f < x["run_frame"] <= window_end for x in shots)

        signals, ignored = [], []

        def add(kind: str, end_by: int, trigger: int, detail: dict[str, Any]) -> None:
            item = {"signal": kind, "description": END_SIGNALS[kind], "end_run_frame": max(last_serve, end_by),
                    "trigger_run_frame": trigger, **detail}
            if kind not in ("next_serve_candidate", "data_end") and followed_by_rally(trigger):
                ignored.append({**item, "ignored_because": "a non-serve shot candidate follows inside this window"})
            else:
                signals.append(item)

        for g in gaps:
            if last_serve < g["start_run_frame"] <= window_end:
                add("no_ball_observed", g["start_run_frame"] - 1, g["start_run_frame"],
                    {"gap_run_frames": [g["start_run_frame"], g["end_run_frame"]], "gap_duration_s": g["duration_s"],
                     "gap_open_end": g["open_end"]})
        for a in activity:
            if last_serve < a["run_frame"] <= window_end:
                add("between_points_activity", a["activity_start_run_frame"] - 1, a["activity_start_run_frame"],
                    {"event_id": a["event_id"]})
        for b in breaks:
            if last_serve < b["start_run_frame"] <= window_end:
                add("break_proposal", b["start_run_frame"] - 1, b["start_run_frame"],
                    {"break_id": b["id"], "break_reasons": b["reasons"]})
        if next_start is not None:
            add("next_serve_candidate", next_start - 1, next_start, {"event_id": groups[n + 1][0]["event_id"]})
        else:
            add("data_end", last_frame, last_frame, {})
        # Earliest end wins; on a tie, the more direct dead-ball evidence (END_SIGNALS order) is the basis.
        signals.sort(key=lambda x: (x["end_run_frame"], list(END_SIGNALS).index(x["signal"])))
        chosen = signals[0]
        end = chosen["end_run_frame"]
        live = [x["run_frame"] for x in shots if start <= x["run_frame"] <= end] + [s["run_frame"] for s in group]
        live += [f for f in range(start, end + 1) if frames[f] is not None and frames[f]["phase"] in LIVE_PHASES]
        last_live = max(live)
        # Walking that begins around or after the last live evidence, up to one dead-ball span past the end.
        corroborating = [w for w in walks
                         if w["start_run_frame"] >= last_live - int(round(walk_window_seconds * fps))
                         and w["start_run_frame"] <= end + int(round(dead_ball_seconds * fps))]
        prev_end = proposals[-1]["end"]["run_frame"] if proposals else -1
        preceding_breaks = [b["id"] for b in breaks if prev_end < b["start_run_frame"] and b["end_run_frame"] < start]
        events_within = [{"event_id": a["event_id"], **at(int(a["frame"])), "action_state": a.get("action_state"),
                          "phase": a.get("phase")}
                         for a in loaded["assessments"] if start <= int(a["frame"]) <= end]
        prev_server = proposals[-1]["start"]["server_side"] if proposals else None
        proposals.append({
            "id": f"point-proposal-{n + 1}", "label": "possible point", "status": "proposal_unreviewed",
            "source_frames": [offset + start, offset + end], "run_frames": [start, end],
            "source_seconds": [round((offset + start) / fps, 3), round((offset + end) / fps, 3)],
            "duration_s": round((end - start) / fps, 3),
            "start": {**at(start), "basis": "serve_candidate", "event_id": group[0]["event_id"],
                      "server_side": group[0]["server_side"], "server_side_basis": group[0]["server_side_basis"],
                      "confidence_basis": group[0]["confidence_basis"]},
            "serve_candidates": [{"event_id": s["event_id"], **at(s["run_frame"]), "server_side": s["server_side"],
                                  "repeat_of": s["repeat_of"]} for s in group],
            "repeat_serve_candidate": len(group) > 1,
            "end": {**at(end), "basis": chosen["signal"], "open_end": chosen["signal"] == "data_end",
                    "last_live_evidence": at(last_live),
                    "uncertainty_run_frames": [last_live, end],
                    "signals": [{**x, **{"end": at(x["end_run_frame"])}} for x in signals],
                    "ignored_signals": ignored,
                    "corroborating_walking": corroborating},
            "preceded_by_break_proposals": preceding_breaks,
            "server_side_changed_from_previous": (prev_server is not None and group[0]["server_side"] != prev_server),
            "events_within": events_within,
        })

    covered = [(p["run_frames"][0], p["run_frames"][1]) for p in proposals]
    unattached = [{"event_id": a["event_id"], **at(int(a["frame"])), "action_state": a.get("action_state")}
                  for a in loaded["assessments"]
                  if a.get("action_state") in LIVE_STATES + ("uncertain_contact",)
                  and not any(s <= int(a["frame"]) <= e for s, e in covered)]
    for s in serves:
        s.update(at(s["run_frame"]))
    notes = list(loaded["notes"])
    if not serves and loaded["replay"]:
        notes.append("No serve candidate in this data, so no point proposals. A missed serve detection does not "
                     "mean there was no serve.")
    return {"schema_version": SCHEMA_VERSION, "kind": "point_proposals", "input": loaded["path"],
            "input_kind": loaded["kind"], "replay": loaded["replay"], "inputs_read": loaded["inputs"],
            "partial": loaded["partial"], "fps": fps, "source_start_frame": offset,
            "run_frames": [0, last_frame], "source_frames": [offset, offset + last_frame],
            "source_seconds": [round(offset / fps, 3), round((offset + last_frame) / fps, 3)],
            "frame_convention": "run_frame is the index inside this replay/run; source_frame = source_start_frame "
                                "+ run_frame (frame of the original video). Seconds are frame / fps; end frames "
                                "are inclusive.",
            "thresholds": thresholds, "repeat_serve_policy": repeat_serve,
            "break_proposals_source": break_info, "meaning": MEANING, "not_claimed": [
                "point winner", "score", "ace", "fault", "let", "winning shot", "game or set boundary"],
            "label_scoring": SCORING_TODO, "notes": notes, "run_id": loaded["run_id"],
            # Plain view for a label scorer (#27): source frames, inclusive; server side; never a winner.
            "points": [{"id": q["id"], "source_start_frame": q["source_frames"][0],
                        "source_end_frame": q["source_frames"][1], "server": q["start"]["server_side"],
                        "status": q["status"]} for q in proposals],
            "proposals": proposals, "serve_candidates": serves,
            "evidence": {"dead_ball_gaps": gaps, "between_points_activity": activity, "break_proposals": breaks,
                         "walking_episodes": walks, "non_serve_shot_candidates": shots},
            "unattached_events": unattached}


# ----------------------------------------------------------------------------- CLI

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="Replay folder, game folder (<root>/replay) or run folder")
    p.add_argument("--output", type=Path, required=True, help="JSON to write (outside the input folder)")
    p.add_argument("--source", choices=("auto", "merged", "chunks"), default="auto",
                   help="Run folders only: events.jsonl or complete chunks (as segments.py)")
    p.add_argument("--segments", type=Path, help="Existing segments.py proposal JSON (#24); default: compute")
    p.add_argument("--repeat-serve", choices=REPEAT_POLICIES, default="same_point",
                   help="Repeat serve candidate (possible let/second serve): keep in one proposal or split")
    for key, value in DEFAULTS.items():
        p.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    args = p.parse_args(argv)
    if _inside(args.output, args.run):
        raise SystemExit("--output must be outside the input folder: proposals are never written into a run")
    try:
        loaded = load_input(args.run, args.source)
        segments_report = _read_json(args.segments) if args.segments else None
        report = propose(loaded, **{k: getattr(args, k) for k in DEFAULTS}, repeat_serve=args.repeat_serve,
                         segments_report=segments_report)
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise SystemExit(f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"{len(report['proposals'])} point proposal(s), {len(report['serve_candidates'])} serve candidate(s) over "
          f"source {report['source_seconds'][0]:.1f}-{report['source_seconds'][1]:.1f} s -> {args.output}")
    for prop in report["proposals"]:
        print(f"  {prop['id']}: source frames {prop['source_frames'][0]}-{prop['source_frames'][1]} "
              f"({prop['source_seconds'][0]:.2f}-{prop['source_seconds'][1]:.2f} s), server {prop['start']['server_side']}, "
              f"end: {prop['end']['basis']}")


if __name__ == "__main__":
    main()
