"""Per-shot and per-point data table (CSV + JSON) with provenance on every field.

Assembles what an existing replay folder already knows (no inference, no video
decoding) into ``shots.csv``, ``points.csv`` and ``shot-table.json``.

Provenance vocabulary (every field has a ``<field>_source`` column):

* ``human_confirmed``: John decided it (a label, a confirmed review decision or
  an explicitly reviewed landing pixel projected through a reviewed calibration).
* ``observed``: taken from a detector observation (an observed foot/ball pixel)
  and, for metres, projected through a *reviewed* court calibration. It is a
  single-view ground-plane projection, not a 3D measurement, and its accuracy
  has not been measured against ground truth.
* ``estimated``: inferred (candidate timing/attribution, rule classifications,
  bbox fallbacks, estimated identity, draft/unreviewed calibration, derived
  zones, fitted flight speed).
* ``unknown``: no supported value; the value column is empty.

Rules this module follows (CLAUDE.md rules 3-5):

* Render-only geometry is never read: 2.5D joints (``joints_m``), body meshes,
  racquets, ``ball_3d_preview`` and unvalidated preview flights are ignored.
  Player positions come only from ``raw_feet_xyz_m`` (the per-frame ground
  projection of the foot pixel, before display smoothing).
* Landing positions come only from a human-confirmed bounce that is the first
  bounce after the contact and before the next contact candidate. An unreviewed
  bounce or an intervening unreviewed contact leaves the landing empty.
* Points (and therefore winners, servers and rally shot indices) come only from
  labels. Point proposals are not points. Winners are never inferred.
* No spin or RPM. Ball speed only from a validated flight, labelled estimated.

``--labels`` is read through ``tennis_vision.labels`` (schema v1 validator and
run/video binding, ``docs/labels-schema.md``), the same reader ``score_labels``
uses. See ``read_labels``.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .labels import COVERAGE_KINDS, LABEL_ABSTAIN_SHOT_TYPES, SCHEMA_VERSION as LABELS_SCHEMA_VERSION, LabelsError
from .labels import check_binding, load_labels
from .score_labels import _source_run_manifest

SCHEMA_VERSION = 1
SOURCES = ("observed", "estimated", "human_confirmed", "unknown")

COURT_WIDTH, COURT_LENGTH = 10.97, 23.77
CENTRE_X, NET_Y = COURT_WIDTH / 2, COURT_LENGTH / 2
SINGLES_HALF_WIDTH = 8.23 / 2
SERVICE_LINE_FROM_NET = 6.40
DEFAULT_FRAME_TOLERANCE = 3
MAX_LANDING_SECONDS = 2.5
OBSERVED_FOOT_BASES = ("two_ankles_ground", "one_ankle_ground")
SHOT_STATES = ("labelled_shot", "confirmed_shot", "shot_candidate")

# Proposed defaults; the zone definitions are John's decision (issue #30).
DEPTH_SCHEMES = {
    # Thirds of the half court, measured from the net along the court.
    "thirds": [(NET_Y / 3, "short"), (2 * NET_Y / 3, "mid"), (NET_Y, "deep")],
    # Service box vs back court.
    "service_line": [(SERVICE_LINE_FROM_NET, "service_box"), (NET_Y, "back_court")],
}
DIRECTION_MIDDLE_HALF_WIDTH = SINGLES_HALF_WIDTH / 3  # middle third of the singles width

# Column order. A field group ``f`` owns its value columns plus ``f_source`` and
# ``f_basis``. Key columns identify a row and carry no provenance.
SHOT_KEYS = ["shot_id", "candidate_event_id"]
SHOT_FIELDS = [
    ("contact", ["contact_frame", "contact_source_frame", "contact_time_s"]),
    ("hitter", ["hitter"]),
    ("shot_status", ["shot_status"]),
    ("shot_type", ["shot_type"]),
    ("hitter_position", ["hitter_position_x_m", "hitter_position_y_m"]),
    ("opponent_position", ["opponent_position_x_m", "opponent_position_y_m"]),
    ("landing", ["landing_x_m", "landing_y_m"]),
    ("landing_call", ["landing_call"]),
    ("direction", ["direction"]),
    ("depth", ["depth"]),
    ("ball_speed", ["ball_speed_kmh"]),
    ("point", ["point_id"]),
    ("rally_shot_index", ["rally_shot_index"]),
]
POINT_KEYS = ["point_id"]
POINT_FIELDS = [
    ("start", ["start_frame", "start_source_frame", "start_time_s"]),
    ("end", ["end_frame", "end_source_frame", "end_time_s"]),
    ("server", ["server"]),
    ("shot_count", ["shot_count"]),
    ("winner", ["winner"]),
    ("score_text", ["score_text"]),
]


def columns(keys, fields):
    out = list(keys)
    for name, values in fields:
        out += values + [name + "_source", name + "_basis"]
    return out


SHOT_COLUMNS = columns(SHOT_KEYS, SHOT_FIELDS)
POINT_COLUMNS = columns(POINT_KEYS, POINT_FIELDS)


def _set(row, field, values, source, basis):
    """Write one field group. Unknown always means empty values."""
    if source not in SOURCES:
        raise ValueError(f"Invalid source {source!r}")
    names = dict(SHOT_FIELDS + POINT_FIELDS)[field]
    if source == "unknown":
        values = [None] * len(names)
    if len(values) != len(names):
        raise ValueError(f"Field {field} needs {len(names)} values")
    row.update(zip(names, values))
    row[field + "_source"] = source
    row[field + "_basis"] = basis


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- inputs

def load_replay(folder):
    """Read a completed replay folder. Separate files win over embedded copies."""
    folder = Path(folder)
    data_path = folder / "replay-data.json"
    if not data_path.is_file():
        raise ValueError(f"{folder} has no replay-data.json; pass a completed replay folder")
    status = folder / "build-status.json"
    if status.is_file() and _read_json(status).get("status") != "complete":
        raise ValueError("Replay build is not complete")
    data = _read_json(data_path)
    report = data["report"]
    run_id = report.get("review_run_id")
    files = {"replay-data.json": data_path}
    events = data.get("events", [])
    if (folder / "reviewed-events.json").is_file():
        reviewed = _read_json(folder / "reviewed-events.json")
        if reviewed.get("run_id") != run_id:
            raise ValueError("reviewed-events.json belongs to a different run")
        events, files["reviewed-events.json"] = reviewed["events"], folder / "reviewed-events.json"
    shots = data.get("shots", {})
    if (folder / "shot-candidates.json").is_file():
        shots, files["shot-candidates.json"] = _read_json(folder / "shot-candidates.json"), folder / "shot-candidates.json"
    flights = data.get("flights", {})
    if (folder / "validated-flight3d.json").is_file():
        flights, files["validated-flight3d.json"] = _read_json(folder / "validated-flight3d.json"), folder / "validated-flight3d.json"
    corrections = report.get("calibration_review")
    if (folder / "calibration-corrections.json").is_file():
        corrections = _read_json(folder / "calibration-corrections.json")
        files["calibration-corrections.json"] = folder / "calibration-corrections.json"
        if corrections.get("run_id") != run_id:
            raise ValueError("calibration-corrections.json belongs to a different run")
    if (folder / "contact-candidates.json").is_file():
        files["contact-candidates.json"] = folder / "contact-candidates.json"
    frames = {f["frame"]: f for f in data.get("frames", [])}
    return {"folder": folder, "report": report, "events": events, "shots": shots.get("shots", []),
            "flights": flights, "corrections": corrections, "frames": frames, "files": files}


def calibration_state(report, corrections):
    """Reviewed calibration is the only one that can make metres 'observed'."""
    fit = report.get("camera_fit") or {}
    if corrections:
        status = corrections.get("calibration_status")
        if fit.get("review_status") not in (None, status):
            status = "inconsistent"
        return {"status": status, "reviewed": status == "reviewed", "basis": fit.get("basis", "unknown"),
                "landmark_source": corrections.get("landmark_source"),
                "landmark_rmse_px": fit.get("landmark_rmse_px"),
                "fit_status": fit.get("status")}
    return {"status": "four_corner_homography_not_reviewed_in_calibration_desk", "reviewed": False,
            "basis": fit.get("basis", "unknown"), "landmark_source": None,
            "landmark_rmse_px": fit.get("landmark_rmse_px"), "fit_status": fit.get("status")}


def read_labels(path, report, video=None, folder=None):
    """Read John's ``labels.json`` through ``tennis_vision.labels`` and bind it to this replay.

    ``labels.load_labels`` validates the whole file against schema v1
    (``docs/labels-schema.md``) and ``labels.check_binding`` rejects labels made for
    another run (``binding.run_id`` must be this replay's ``review_run_id``), another fps
    or another source video. The source-video SHA-256 is compared whenever it is known:
    from ``video`` (``--video``, hashed locally) and from the ``longrun-manifest.json`` of
    the run this replay was built from (found the way ``score_labels`` finds it). A replay
    folder's ``source.mp4`` is a review clip, not the original video, so it is never used.

    Label frames are SOURCE-video frames; they (and coverage spans) are converted to
    replay-local frames with ``source_start_frame``. Entries outside the replay's frames
    are counted and ignored. The file is read, never modified.
    """
    labels, digest = load_labels(path)
    run_id, fps = report.get("review_run_id"), float(report["fps"])
    offset, frames = int(report.get("source_start_frame") or 0), int(report["frames"])
    if not isinstance(run_id, str):
        raise LabelsError("This replay records no review_run_id, so labels cannot be bound to it")
    known = []
    if video is not None:
        known.append((_sha256(video), "matched"))
    found = _source_run_manifest(Path(folder), report.get("input_run"), offset, frames, fps) if folder else None
    if found:
        known.append((found[1]["input"]["sha256"],
                      f"matched (longrun-manifest.json of source run {found[0].name}, same source frames)"))
    for sha, _ in known:
        check_binding(labels, run_id, sha, fps)
    if known:
        video_check = "; ".join(check for _, check in known)
    else:
        # Nothing here records the original video's hash: run id and fps are still checked.
        check_binding(labels, run_id, labels["binding"]["source_video_sha256"], fps)
        video_check = "not_checked (pass --video with the original file to check it)"

    def in_replay(*values):
        return all(0 <= v < frames for v in values)

    parsed = {kind: [] for kind in COVERAGE_KINDS}
    outside = {kind: 0 for kind in COVERAGE_KINDS}
    for kind in COVERAGE_KINDS:
        for e in labels[kind]:
            if kind == "shots":
                item = {"id": e["id"], "frame": e["source_frame"] - offset, "hitter": e["hitter"],
                        "shot_type": e["shot_type"]}
                keep = in_replay(item["frame"])
            elif kind == "bounces":
                item = {"id": e["id"], "frame": e["source_frame"] - offset, "call": e["call"]}
                keep = in_replay(item["frame"])
            else:
                item = {"id": e["id"], "start_frame": e["source_start_frame"] - offset,
                        "end_frame": e["source_end_frame"] - offset, "server": e["server"], "winner": e["winner"],
                        "score_text": e.get("score_text")}
                keep = in_replay(item["start_frame"], item["end_frame"])
            if keep:
                parsed[kind].append(item)
            else:
                outside[kind] += 1
    coverage = {kind: [] for kind in COVERAGE_KINDS}
    for span in labels["coverage"]:
        for kind in span["kinds"]:
            coverage[kind].append((span["source_start_frame"] - offset, span["source_end_frame"] - offset))
    labeller = labels["labeller"]
    return {"shots": sorted(parsed["shots"], key=lambda s: s["frame"]),
            "bounces": sorted(parsed["bounces"], key=lambda b: b["frame"]),
            "points": sorted(parsed["points"], key=lambda p: p["start_frame"]), "coverage": coverage,
            "meta": {"file": Path(path).name, "sha256": digest, "run_binding": "review_run_id",
                     "video_check": video_check, "labeller": labeller["name"], "labelled_on": labeller["date"],
                     "reader": f"tennis_vision.labels (schema v{LABELS_SCHEMA_VERSION} validator and run binding; "
                               "docs/labels-schema.md)",
                     "frames": "label source frames converted to replay frames with source_start_frame",
                     "coverage_replay_frames": coverage,
                     "counts_in_replay": {k: len(v) for k, v in parsed.items()},
                     "counts_outside_replay_frames": outside}}


def covered(coverage, kind, frame):
    """The fully-labelled span (replay frames) of ``kind`` containing ``frame``, if any."""
    return next(((a, b) for a, b in (coverage or {}).get(kind, []) if a <= frame <= b), None)


# --------------------------------------------------------------------------- geometry

def player_position(frame_record, identity, calibration):
    """Ground position (court metres, near-left doubles corner origin) at one frame."""
    if frame_record is None or identity not in ("near", "far"):
        return None, "unknown", "no player identity or frame record at contact"
    matches = [p for p in frame_record.get("players", []) if p.get("identity_id") == identity]
    if len(matches) != 1:
        return None, "unknown", f"no single {identity} player record at contact frame"
    player = matches[0]
    if player.get("predicted"):
        return None, "unknown", "only a predicted track at contact frame; predictions are not used"
    raw = player.get("raw_feet_xyz_m")
    if not raw or not np.isfinite(raw[:2]).all():
        return None, "unknown", "no finite foot projection at contact frame"
    xy = [round(raw[0] + CENTRE_X, 3), round(raw[1] + NET_Y, 3)]
    weak = []
    if player.get("basis") not in OBSERVED_FOOT_BASES:
        weak.append(f"foot pixel is {player.get('basis')} (not observed ankles)")
    if player.get("identity_association_estimated"):
        weak.append(f"identity is estimated ({player.get('identity_basis')})")
    if not calibration["reviewed"]:
        weak.append(f"calibration is {calibration['status']}")
    basis = (f"{player.get('basis')} foot pixel projected to the ground plane through "
             f"{calibration['status']} calibration ({calibration['basis']}); unsmoothed")
    if weak:
        return xy, "estimated", basis + "; estimated because " + "; ".join(weak)
    return xy, "observed", basis


def depth_zone(hitter, landing_y, scheme):
    distance = (landing_y - NET_Y) if hitter == "near" else (NET_Y - landing_y)
    if distance < 0:
        return "hitter_side_of_net"
    for limit, name in DEPTH_SCHEMES[scheme]:
        if distance <= limit:
            return name
    return "beyond_baseline"


def direction_zone(hitter_x, landing_x):
    """Middle third of the singles width vs same side (down the line) vs opposite side."""
    landing_offset, hitter_offset = landing_x - CENTRE_X, hitter_x - CENTRE_X
    if abs(landing_offset) < DIRECTION_MIDDLE_HALF_WIDTH:
        return "middle", None
    if abs(hitter_offset) < DIRECTION_MIDDLE_HALF_WIDTH:
        return None, "hitter is in the middle third; cross-court vs down-the-line is undefined in the default zones"
    return ("down_the_line" if np.sign(landing_offset) == np.sign(hitter_offset) else "cross_court"), None


def flight_speed(flights, event_id, fps):
    for fit in flights.get("fits", []):
        if not event_id or fit.get("start_event_id") != event_id:
            continue
        path = fit.get("trajectory_m", [])
        if len(path) < 3 or fit.get("airborne_xyz_status") != "estimated_not_measured":
            continue
        p0, p1, p2 = (np.asarray(path[i]["point_m"], float) for i in range(3))
        velocity = (-3 * p0 + 4 * p1 - p2) * fps / 2  # exact for a gravity parabola
        return round(float(np.linalg.norm(velocity)) * 3.6, 1), fit.get("bounce_event_id")
    return None, None


# --------------------------------------------------------------------------- table

def match_labels(labelled, candidates, tolerance, same_player=True):
    """Greedy one-to-one match by frame distance. Returns {label index: candidate}."""
    pairs = []
    for i, label in enumerate(labelled):
        for c in candidates:
            if abs(c["frame"] - label["frame"]) > tolerance:
                continue
            if same_player and c.get("player_id") not in (None, label.get("hitter")):
                continue
            pairs.append((abs(c["frame"] - label["frame"]), i, c["frame"], c["id"], c))
    matched, used = {}, set()
    for _, i, _, cid, c in sorted(pairs, key=lambda p: p[:4]):
        if i not in matched and cid not in used:
            matched[i] = c
            used.add(cid)
    return matched


def build_table(replay, labels=None, frame_tolerance=DEFAULT_FRAME_TOLERANCE, depth_scheme="thirds"):
    if depth_scheme not in DEPTH_SCHEMES:
        raise ValueError(f"Unknown depth scheme {depth_scheme}")
    report, frames = replay["report"], replay["frames"]
    fps, offset = float(report["fps"]), int(report.get("source_start_frame") or 0)
    calibration = calibration_state(report, replay["corrections"])
    events = sorted(replay["events"], key=lambda e: (e["frame"], e["id"]))
    by_id = {e["id"]: e for e in events}
    candidates = [dict(s, id=s["event_id"]) for s in replay["shots"]]
    labelled_shots = labels["shots"] if labels else []
    labelled_bounces = labels["bounces"] if labels else []
    labelled_points = labels["points"] if labels else []
    coverage = labels["coverage"] if labels else {}

    # One row per shot candidate, plus labelled shots no candidate matched.
    matched = match_labels(labelled_shots, candidates, frame_tolerance)
    label_of = {c["id"]: labelled_shots[i] for i, c in matched.items()}
    entries = [{"candidate": c, "label": label_of.get(c["id"])} for c in candidates]
    entries += [{"candidate": None, "label": s} for i, s in enumerate(labelled_shots) if i not in matched]

    # Inside a fully-labelled bounce span, an unreviewed bounce candidate with no label is
    # John's "no bounce here" (labels schema coverage), so it cannot block or give a landing.
    bounce_events = [e for e in events if e["type"] == "bounce" and e["status"] != "rejected"]
    bounce_label_match = match_labels(labelled_bounces, bounce_events, frame_tolerance, same_player=False)
    bounce_label_of = {c["id"]: labelled_bounces[i] for i, c in bounce_label_match.items()}
    unmatched_bounce_labels = [b for i, b in enumerate(labelled_bounces) if i not in bounce_label_match]
    bounce_events = [e for e in bounce_events if e["status"] == "confirmed" or e["id"] in bounce_label_of
                     or not covered(coverage, "bounces", e["frame"])]

    rows, unlabelled_in_coverage = [], set()
    for entry in entries:
        c, label = entry["candidate"], entry["label"]
        event = by_id.get(c["id"]) if c else None
        row = {"shot_id": c["id"] if c else f"label:{label['id']}",
               "candidate_event_id": c["id"] if c else None}
        confirmed_event = bool(event and event["type"] == "hit" and event["status"] == "confirmed")
        # Contact timing.
        if label:
            frame, source = label["frame"], "human_confirmed"
            basis = f"labelled contact frame ({label['id']})" + (
                f"; matched candidate {c['id']} at frame {c['frame']}, tolerance +/-{frame_tolerance}" if c
                else "; no candidate matched")
        elif confirmed_event:
            frame, source, basis = event["frame"], "human_confirmed", "contact frame confirmed in event review"
        else:
            frame, source = c["frame"], "estimated"
            basis = f"contact candidate frame, frame-level precision (support: {c.get('contact_support')})"
        _set(row, "contact", [frame, frame + offset, round(frame / fps, 3)], source, basis)
        row["_frame"], row["_contact_source"] = frame, source
        # Hitter.
        if label:
            _set(row, "hitter", [label["hitter"]], "human_confirmed", "labelled hitter")
        elif confirmed_event and event.get("player_id") in ("near", "far"):
            _set(row, "hitter", [event["player_id"]], "human_confirmed", "hit confirmed in event review with player")
        elif c and c.get("player_id") in ("near", "far"):
            _set(row, "hitter", [c["player_id"]], "estimated", "contact candidate attributed to tracked player identity")
        else:
            _set(row, "hitter", [], "unknown", "no hitter attribution")
        hitter = row["hitter"]
        # Shot status (a swing is not a shot).
        if c and c.get("classification_status") == "human_reviewed" and c.get("classification") == "not_a_shot":
            _set(row, "shot_status", ["not_a_shot"], "human_confirmed", "stroke review marked this contact as not a shot")
        elif label:
            _set(row, "shot_status", ["labelled_shot"], "human_confirmed", "labelled shot")
        elif confirmed_event:
            _set(row, "shot_status", ["confirmed_shot"], "human_confirmed", "hit confirmed in event review")
        elif covered(coverage, "shots", frame):
            span = covered(coverage, "shots", frame)
            unlabelled_in_coverage.add(c["id"])
            _set(row, "shot_status", ["no_labelled_shot"], "human_confirmed",
                 f"inside fully-labelled shot span (replay frames {span[0]}-{span[1]}) with no labelled shot "
                 f"within +/-{frame_tolerance} frames; candidate was {c.get('action_state')}")
        else:
            state = c.get("action_state") or "uncertain_contact"
            basis = f"play-context assessment: {(c.get('play_assessment') or {}).get('reason') or state}"
            if labels:
                basis += "; outside labelled shot coverage"
            _set(row, "shot_status", [state], "estimated", basis)
        status = row["shot_status"]
        # Shot type and its basis.
        if label and label["shot_type"] in LABEL_ABSTAIN_SHOT_TYPES:
            _set(row, "shot_type", [], "unknown", "labelled 'unsure' (human abstention)")
        elif label:
            _set(row, "shot_type", [label["shot_type"]], "human_confirmed", "labelled")
        elif c and c.get("classification_status") == "human_reviewed" and c.get("classification") not in ("unknown", "not_a_shot"):
            _set(row, "shot_type", [c["classification"]], "human_confirmed", "stroke review (human_reviewed)")
        elif status in SHOT_STATES and c and c.get("classification") not in (None, "unknown", "not_a_shot"):
            _set(row, "shot_type", [c["classification"]], "estimated",
                 f"rule candidate ({c.get('support')} support): " + "; ".join(c.get("reasons", [])[:1]))
        elif status in SHOT_STATES:
            _set(row, "shot_type", [], "unknown", "abstained: " + "; ".join((c or {}).get("reasons", [])[:1]))
        else:
            _set(row, "shot_type", [], "unknown", f"abstained: contact is {status}, not a shot candidate")
        # Player positions at the contact frame.
        opponent = {"near": "far", "far": "near"}.get(hitter)
        for field, identity in (("hitter_position", hitter), ("opponent_position", opponent)):
            xy, source, basis = player_position(frames.get(frame), identity, calibration)
            _set(row, field, xy or [], source, basis)
        rows.append(row)
    rows.sort(key=lambda r: (r["_frame"], r["shot_id"]))

    # Contacts bound the landing window: any non-rejected hit event or labelled shot,
    # except candidates John's shot coverage says are not shots.
    contact_marks = sorted({(e["frame"], e["id"]) for e in events if e["type"] == "hit" and e["status"] != "rejected"
                            and e["id"] not in unlabelled_in_coverage}
                           | {(r["_frame"], r["shot_id"]) for r in rows if r["shot_status"] != "no_labelled_shot"})
    flights = replay["flights"]
    for row in rows:
        _landing(row, contact_marks, bounce_events, bounce_label_of, unmatched_bounce_labels, calibration,
                 fps, frame_tolerance)
        _zones(row, depth_scheme)
        speed, bounce_id = (flight_speed(flights, row["candidate_event_id"], fps)
                            if row["shot_status"] in SHOT_STATES else (None, None))
        if speed is None:
            _set(row, "ball_speed", [], "unknown", "no validated flight starts at this contact")
        else:
            _set(row, "ball_speed", [speed], "estimated",
                 f"validated_flight gravity fit ending at confirmed bounce {bounce_id}; speed at contact frame from the "
                 "fitted arc; approximate single-camera calibration; drag and spin not modelled")

    points = _points(rows, labelled_points, labelled_shots, fps, offset)
    for row in rows:
        row.pop("_frame")
        row.pop("_contact_source")
    return {"shots": rows, "points": points, "calibration": calibration}


def _landing(row, contact_marks, bounce_events, bounce_label_of, unmatched_bounce_labels, calibration, fps, tolerance):
    no_call = lambda basis: _set(row, "landing_call", [], "unknown", basis)
    if row["shot_status"] not in SHOT_STATES:
        _set(row, "landing", [], "unknown", f"no landing attributed: contact is {row['shot_status']}")
        return no_call("no labelled bounce attributed")
    frame = row["_frame"]
    later = [(f, i) for f, i in contact_marks if f > frame and i not in (row["shot_id"], row["candidate_event_id"])]
    limit = frame + round(MAX_LANDING_SECONDS * fps) + 1
    end = min(later[0][0], limit) if later else limit
    bound = (f"before next contact {later[0][1]} (frame {later[0][0]})" if later and later[0][0] == end
             else f"within {MAX_LANDING_SECONDS} s")
    window = [e for e in bounce_events if frame < e["frame"] < end]
    label_only = [b for b in unmatched_bounce_labels if frame < b["frame"] < end]
    first = window[0] if window else None
    if label_only and (first is None or label_only[0]["frame"] < first["frame"]):
        _set(row, "landing", [], "unknown",
             f"labelled bounce at frame {label_only[0]['frame']} has no matching candidate with a location")
        call = label_only[0].get("call", "unsure")
        return _set(row, "landing_call", [call], "human_confirmed", f"labelled bounce frame {label_only[0]['frame']}")
    if first is None:
        _set(row, "landing", [], "unknown", f"no bounce event {bound}")
        return no_call("no labelled bounce attributed")
    label = bounce_label_of.get(first["id"])
    if label:
        _set(row, "landing_call", [label.get("call", "unsure")], "human_confirmed",
             f"labelled bounce frame {label['frame']} matched {first['id']}")
    else:
        no_call("no labelled bounce attributed")
    if first["status"] == "confirmed" and first.get("court_m"):
        x, y = first["court_m"]
        explicit = first.get("pixel_basis") == "explicit_human_landing_pixel"
        pixel = "explicitly reviewed landing pixel" if explicit else "observed ball pixel at the confirmed bounce frame"
        if not calibration["reviewed"]:
            source = "estimated"
        else:
            source = "human_confirmed" if explicit else "observed"
        return _set(row, "landing", [x, y], source,
                    f"first bounce {bound} is confirmed ({first['id']}); {pixel} projected through "
                    f"{calibration['status']} calibration")
    if label and first.get("candidate_court_m"):
        x, y = first["candidate_court_m"]
        return _set(row, "landing", [x, y], "estimated",
                    f"labelled bounce matched {first['id']} (bounce timing is a human decision); location is the "
                    f"unreviewed ball projection at candidate frame {first['frame']} through {calibration['status']} calibration")
    later_confirmed = next((e["id"] for e in window if e["status"] == "confirmed"), None)
    basis = f"first bounce {bound} is {first['status']} ({first['id']})"
    if first["status"] == "confirmed":
        basis += f" without a located landing: {first.get('geometry_issue')}"
    if later_confirmed and later_confirmed != first["id"]:
        basis += f"; later confirmed bounce {later_confirmed} is not attributed across it"
    _set(row, "landing", [], "unknown", basis)


def _zones(row, scheme):
    have = row["landing_source"] != "unknown" and row["hitter_position_source"] != "unknown" and row["hitter"]
    if not have:
        _set(row, "direction", [], "unknown", "needs a known hitter, hitter position and landing")
        _set(row, "depth", [], "unknown", "needs a known hitter and landing")
        return
    inputs = (f"landing {row['landing_source']}, hitter position {row['hitter_position_source']}, "
              f"hitter {row['hitter_source']}")
    direction, reason = direction_zone(row["hitter_position_x_m"], row["landing_x_m"])
    if direction:
        _set(row, "direction", [direction], "estimated",
             f"derived, default zones (middle third of singles width; same side = down the line); {inputs}")
    else:
        _set(row, "direction", [], "unknown", reason)
    _set(row, "depth", [depth_zone(row["hitter"], row["landing_y_m"], scheme)], "estimated",
         f"derived, depth scheme '{scheme}' (distance from net on the opponent's half); {inputs}")


def _points(rows, labelled_points, labelled_shots, fps, offset):
    points = []
    for n, p in enumerate(labelled_points, 1):
        pid = str(p.get("id") or f"point-{n:03d}")
        a, b = p["start_frame"], p["end_frame"]
        inside = [r for r in rows if a <= r["_frame"] <= b]
        labelled_inside = [r for r in inside if r["shot_status"] == "labelled_shot"]
        use_labels = bool(labelled_inside)
        rally = labelled_inside if use_labels else [r for r in inside if r["shot_status"] in SHOT_STATES]
        for r in inside:
            src = "human_confirmed" if r["_contact_source"] == "human_confirmed" else "estimated"
            _set(r, "point", [pid], src, f"contact frame inside labelled point {pid} ({a}-{b})")
            if r in rally:
                idx = rally.index(r) + 1
                if use_labels:
                    _set(r, "rally_shot_index", [idx], "human_confirmed", "order among labelled shots in the point")
                else:
                    _set(r, "rally_shot_index", [idx], "estimated",
                         "order among unreviewed shot candidates in the labelled point (no labelled shots)")
            else:
                _set(r, "rally_shot_index", [], "unknown",
                     "not a labelled shot in this point" if use_labels else f"contact is {r['shot_status']}")
        row = {"point_id": pid}
        _set(row, "start", [a, a + offset, round(a / fps, 3)], "human_confirmed", "labelled point start")
        _set(row, "end", [b, b + offset, round(b / fps, 3)], "human_confirmed", "labelled point end")
        if p.get("server") in ("near", "far"):
            _set(row, "server", [p["server"]], "human_confirmed", "labelled server")
        else:
            _set(row, "server", [], "unknown", "server labelled unknown or missing; never inferred")
        if use_labels:
            _set(row, "shot_count", [len(rally)], "human_confirmed", "labelled shots inside the point")
        else:
            _set(row, "shot_count", [len(rally)], "estimated",
                 "unreviewed shot candidates inside the labelled point; coverage count, not accuracy")
        winner = p.get("winner", "unknown")
        if winner in ("near", "far"):
            _set(row, "winner", [winner], "human_confirmed", "labelled winner")
        else:
            _set(row, "winner", [], "unknown", "winner labelled unknown or missing; never inferred")
        if p.get("score_text"):
            _set(row, "score_text", [str(p["score_text"])], "human_confirmed", "labelled score text")
        else:
            _set(row, "score_text", [], "unknown", "score not labelled")
        points.append(row)
    for r in rows:
        if "point_source" not in r:
            _set(r, "point", [], "unknown", "no labelled point contains this contact" if labelled_points else
                 "points are not known (no labelled points); point proposals are not points")
            _set(r, "rally_shot_index", [], "unknown", "points are not known")
    return points


# --------------------------------------------------------------------------- output

def _csv_value(value):
    return "" if value is None else value


def write_outputs(output, table, replay, labels, frame_tolerance, depth_scheme):
    output = Path(output)
    folder = replay["folder"].resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new folder to preserve previous results")
    if output.resolve() == folder or folder in output.resolve().parents:
        raise ValueError("Write the table to a new folder outside the replay folder; replay folders are not modified")
    output.mkdir(parents=True)
    for name, cols, rows in (("shots.csv", SHOT_COLUMNS, table["shots"]), ("points.csv", POINT_COLUMNS, table["points"])):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _csv_value(row.get(k)) for k in cols})
    counts = {"shots": len(table["shots"]), "points": len(table["points"]),
              "shot_sources": {f: _source_counts(table["shots"], f) for f, _ in SHOT_FIELDS},
              "point_sources": {f: _source_counts(table["points"], f) for f, _ in POINT_FIELDS}}
    report = replay["report"]
    payload = {
        "schema_version": SCHEMA_VERSION, "kind": "shot_point_table",
        "replay": {"folder": replay["folder"].name, "package_id": report.get("package_id"),
                   "review_run_id": report.get("review_run_id"), "fps": report.get("fps"),
                   "frames": report.get("frames"), "source_start_frame": report.get("source_start_frame"),
                   "input_sha256": {k: _sha256(v) for k, v in sorted(replay["files"].items())}},
        "labels": labels["meta"] if labels else None,
        "calibration": table["calibration"],
        "conventions": {
            "coordinates": "court metres; origin at the near-left doubles corner; x across the court (0-10.97), "
                           "y from the near baseline to the far baseline (0-23.77); net at y=11.885",
            "frames": "contact_frame is the replay-local frame; contact_source_frame adds source_start_frame",
            "sources": {"human_confirmed": "John decided it (label, confirmed review, explicit landing pixel with reviewed calibration)",
                        "observed": "detector observation; metres are a ground-plane projection of an observed pixel through a reviewed calibration, not a measurement; accuracy unmeasured",
                        "estimated": "inferred: candidates, rule classifications, fallbacks, estimated identity, draft calibration, derived zones, fitted speed",
                        "unknown": "no supported value; value empty"},
            "frame_tolerance": frame_tolerance,
            "depth_scheme": {"name": depth_scheme, "limits_m_from_net": DEPTH_SCHEMES[depth_scheme],
                             "status": "proposed default; John's decision"},
            "direction": {"middle_half_width_m": round(DIRECTION_MIDDLE_HALF_WIDTH, 3),
                          "rule": "landing in the middle third of the singles width = middle; landing on the hitter's side "
                                  "of the centre line = down_the_line; opposite side = cross_court; hitter in the middle third = unknown",
                          "status": "proposed default; John's decision"},
            "never_used": ["joints_m (2.5D display)", "body meshes", "racquets", "ball_3d_preview",
                           "preview flights", "point proposals", "spin/RPM"],
        },
        "columns": {"shots": SHOT_COLUMNS, "points": POINT_COLUMNS},
        "counts": counts,
        "shots": table["shots"], "points": table["points"],
        "limitations": [
            "Coverage and source counts are not accuracy. No accuracy is claimed without labelled ground truth.",
            "Court positions are single-view ground-plane projections of foot or ball pixels; 'observed' names the pixel "
            "evidence and a reviewed calibration, not a measured 3D position.",
            "Shot types are heuristic candidates unless labelled or stroke-reviewed; a swing is not a shot.",
            "Landings are attributed only to the first human-confirmed bounce before the next contact candidate.",
            "Points, servers, winners and rally indices come only from labels; point proposals are not used.",
            "--labels is validated and bound to this replay's run by tennis_vision.labels (docs/labels-schema.md); "
            "the source-video hash is checked only when --video or the source run's manifest gives it (see labels.video_check).",
        ]}
    (output / "shot-table.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def _source_counts(rows, field):
    out = {s: 0 for s in SOURCES}
    for r in rows:
        out[r[field + "_source"]] += 1
    return out


def run(replay_folder, output, labels_path=None, frame_tolerance=DEFAULT_FRAME_TOLERANCE, depth_scheme="thirds",
        video=None):
    if type(frame_tolerance) is not int or frame_tolerance < 0:
        raise ValueError("Frame tolerance must be a non-negative integer")
    if video is not None and not labels_path:
        raise ValueError("--video only checks the labels binding; pass --labels too")
    replay = load_replay(replay_folder)
    labels = read_labels(labels_path, replay["report"], video, replay["folder"]) if labels_path else None
    table = build_table(replay, labels, frame_tolerance, depth_scheme)
    return write_outputs(output, table, replay, labels, frame_tolerance, depth_scheme)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--replay", type=Path, required=True, help="Completed replay folder (replay-data.json and siblings)")
    parser.add_argument("--labels", type=Path, help="John's labels.json (schema v1, docs/labels-schema.md; read, never modified)")
    parser.add_argument("--video", type=Path,
                        help="Original source video, hashed locally to check the labels' source_video_sha256")
    parser.add_argument("--output", type=Path, required=True, help="NEW folder for shots.csv, points.csv, shot-table.json")
    parser.add_argument("--frame-tolerance", type=int, default=DEFAULT_FRAME_TOLERANCE,
                        help="Frames within which a label matches a candidate (default 3)")
    parser.add_argument("--depth-scheme", choices=sorted(DEPTH_SCHEMES), default="thirds",
                        help="Proposed depth zones (John's decision): thirds of the half court, or service box vs back court")
    args = parser.parse_args(argv)
    payload = run(args.replay, args.output, args.labels, args.frame_tolerance, args.depth_scheme, args.video)
    print(json.dumps({"output": str(args.output), "shots": payload["counts"]["shots"],
                      "points": payload["counts"]["points"], "calibration": payload["calibration"]["status"],
                      "landing_sources": payload["counts"]["shot_sources"]["landing"]}, indent=2))


if __name__ == "__main__":
    main()
