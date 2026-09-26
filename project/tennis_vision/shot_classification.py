"""Explainable, abstaining shot-type candidates from temporal 2D pose evidence.

This is a first-pass rule classifier, not a trained or accuracy-validated model.
Shot type, event validation, and single-camera geometry are separate concepts.
"""
from collections import Counter
import copy

import numpy as np

from .event_analysis import _joint, _box_distance, observed_balls
from .stroke_execution import execution_evidence


def player_at(rows, frame, identity):
    matches = [p for p in rows[frame].get("tracks", []) if p.get("label") == "player"
               and p.get("identity_id") == identity and not p.get("predicted")]
    return matches[0] if len(matches) == 1 else None


def racket_at(row, player):
    matches = [r for r in row.get("tracks", []) if r.get("label") == "racket"
               and not r.get("predicted") and player.get("track_id") is not None
               and r.get("player_track_id") == player["track_id"]]
    return matches[0] if len(matches) == 1 else None


def resolve_review_events(candidates, labels, balls, fps, court, run_id, width, height):
    """Validate review identity; recompute geometry at CORRECTED event frames."""
    sources = {e["id"]: copy.deepcopy(e) for e in candidates}
    if len(sources) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    decisions = {}
    def valid_frame(f):
        return type(f) is int and 0 <= f < len(balls)
    if labels is not None:
        if labels.get("schema_version") != 1 or labels.get("run_id") != run_id:
            raise ValueError("Review labels do not belong to this exact run")
        if not isinstance(labels.get("manual_events"), list) or not isinstance(labels.get("labels"), list):
            raise ValueError("Invalid review labels structure")
        for e in labels["manual_events"]:
            if (not isinstance(e.get("id"), str) or not e["id"].startswith("manual-")
                    or e["id"] in sources or not valid_frame(e.get("frame"))):
                raise ValueError("Invalid/duplicate manual event")
            sources[e["id"]] = {"id": e["id"], "frame": e["frame"], "type": "manual_candidate"}
        for label in labels["labels"]:
            if (label.get("id") not in sources or label["id"] in decisions
                    or not valid_frame(label.get("frame"))
                    or label.get("status") not in ("confirmed", "rejected", "uncertain", "unreviewed")
                    or label.get("type") not in ("hit", "bounce", "uncertain")
                    or label.get("player_id") not in (None, "near", "far")
                    or (label.get("status") == "confirmed" and label.get("type") == "uncertain")):
                raise ValueError("Invalid/duplicate review decision")
            if label.get("landing_pixel") is not None:
                xy = label["landing_pixel"]
                if (not isinstance(xy, list) or len(xy) != 2 or not np.isfinite(xy).all()
                        or not 0 <= xy[0] < width or not 0 <= xy[1] < height):
                    raise ValueError("Invalid explicitly reviewed landing_pixel")
            decisions[label["id"]] = label
    events = []
    for event_id, original in sources.items():
        decision = decisions.get(event_id)
        frame = decision["frame"] if decision else original["frame"]
        if not valid_frame(frame):
            raise ValueError("Candidate frame outside run")
        kind = decision["type"] if decision else {"hit_candidate": "hit", "bounce_candidate": "bounce"}.get(original["type"], "uncertain")
        status = decision["status"] if decision else "unreviewed"
        ball = balls[frame]
        pixel = list(ball["pixel"]) if ball else None
        basis = "observed_ball_at_reviewed_frame" if ball else "missing_ball_at_reviewed_frame"
        if decision and decision.get("landing_pixel") is not None and kind == "bounce":
            pixel = decision["landing_pixel"]
            basis = "explicit_human_landing_pixel"
        # Candidate display geometry is explicitly conditional: "IF a bounce,
        # this would be its ground location." It cannot enter reviewed flights;
        # a separate, visibly unvalidated preview may use it as a hypothesis.
        projected = court.project(tuple(pixel)) if kind == "bounce" and status != "rejected" and pixel is not None else None
        ground = projected if status == "confirmed" else None
        candidate_ground = projected if status in ("unreviewed","uncertain") else None
        geometry_issue = None
        if projected is not None and (not np.isfinite(projected).all() or
                not (-3 <= projected[0] <= 13.97 and -7 <= projected[1] <= 30.77)):
            ground = None
            candidate_ground = None
            geometry_issue = "bounce projection outside supported court/runoff region"
        if kind == "bounce" and status == "confirmed" and ground is None and geometry_issue is None:
            geometry_issue = "No observed ball at corrected frame; supply an explicitly reviewed landing_pixel"
        events.append({"id": event_id, "type": kind, "status": status, "frame": frame, "time_s": frame/fps,
                       "player_id": decision.get("player_id") if decision else original.get("player_id"),
                       "pixel": pixel, "pixel_basis": basis, "court_m": ground,
                       "candidate_court_m": candidate_ground,
                       "candidate_geometry_basis": "conditional_on_unverified_bounce_not_a_flight_constraint" if candidate_ground is not None else None,
                       "geometry_issue": geometry_issue, "original_frame": original["frame"],
                       "notes": decision.get("notes", "") if decision else ""})
    return sorted(events, key=lambda e: (e["frame"], e["id"]))


def infer_handedness(rows, overrides=None):
    """Use racquet-to-wrist votes; never assume everyone is right-handed."""
    overrides = overrides or {}
    profiles = {}
    for identity in ("near", "far"):
        votes = Counter()
        for f, row in enumerate(rows):
            p = player_at(rows, f, identity)
            if not p:
                continue
            r = racket_at(row, p)
            left, right = _joint(p, "left_wrist"), _joint(p, "right_wrist")
            if not r or left is None or right is None:
                continue
            h = max(1., p["bbox"][3]-p["bbox"][1])
            dl, dr = _box_distance(left, r["bbox"])/h, _box_distance(right, r["bbox"])/h
            if min(dl, dr) < .18 and abs(dl-dr) > .05:
                votes["left" if dl < dr else "right"] += 1
        best, n = votes.most_common(1)[0] if votes else ("unknown", 0)
        ratio = n/max(1, sum(votes.values()))
        hand = best if n >= 8 and ratio >= .8 else "unknown"
        source = "racquet_wrist_votes" if hand != "unknown" else "insufficient_or_conflicting_evidence"
        if overrides.get(identity) in ("left", "right", "unknown"):
            hand, source = overrides[identity], "user_setting"
        profiles[identity] = {"hand": hand, "source": source, "votes": dict(votes), "majority_fraction": round(ratio, 3)}
        profiles[identity]['role']='descriptive_only_not_a_stroke_constraint'
    return profiles


def _max_missing(balls, a, b):
    longest = current = 0
    for observation in balls[a:b+1]:
        current = 0 if observation else current+1
        longest = max(longest, current)
    return longest


def _temporal_features(rows, balls, event, fps, hand):
    f, identity = event["frame"], event["player_id"]
    radius = max(3, round(.9*fps))
    contact = []
    arm_pre = {"left": [], "right": []}
    wrist_motion = {"left": [], "right": []}
    side_values, equipment_frames = [], 0
    for k in range(max(0, f-radius), min(len(rows), f+round(.25*fps)+1)):
        p = player_at(rows, k, identity)
        if p is None:
            continue
        h = max(1., p["bbox"][3]-p["bbox"][1])
        ls, rs = _joint(p, "left_shoulder"), _joint(p, "right_shoulder")
        lh, rh = _joint(p, "left_hip"), _joint(p, "right_hip")
        if ls is None or rs is None:
            continue
        shoulders = (ls+rs)/2
        torso = (lh+rh)/2 if lh is not None and rh is not None else shoulders
        wrists = {side: _joint(p, side+"_wrist") for side in ("left", "right")}
        r = racket_at(rows[k], p)
        near_contact = abs(k-f) <= max(2, round(.10*fps))
        equipment_frames += int(near_contact and r is not None)
        for side, wrist in wrists.items():
            if wrist is None:
                continue
            if k < f:
                arm_pre[side].append((k, float((shoulders[1]-wrist[1])/h)))
            if abs(k-f) <= round(.3*fps):
                wrist_motion[side].append((k, ((wrist-torso)/h).tolist()))
        if near_contact:
            pixel = balls[k]["pixel"] if balls[k] else None
            overhead = pixel is not None and (shoulders[1]-pixel[1])/h > .28
            elevated = max(((shoulders[1]-w[1])/h for w in wrists.values() if w is not None), default=-1) > .22
            contact.append({"frame": k, "overhead": overhead, "elevated_arm": elevated})
        # Use anatomical shoulder direction rather than screen-left/right.
        # Side-on collapsed shoulders cannot reliably tell forehand/backhand.
        if f-round(.16*fps) <= k <= f and hand in ("left", "right"):
            axis = rs-ls
            axis_len = float(np.linalg.norm(axis))
            dominant = wrists[hand]
            if dominant is not None and axis_len/h >= .10:
                side = float(np.dot(dominant-shoulders, axis/axis_len)/h)
                side_values.append(side if hand == "right" else -side)
    toss_sides = []
    for side, values in arm_pre.items():
        raised = [k for k, value in values if value > .25 and k <= f-round(.13*fps)]
        if len(raised) >= 2:
            toss_sides.append(side)
    movement = {}
    for side, samples in wrist_motion.items():
        positions = np.asarray([p for _, p in samples])
        if len(positions) >= 4:
            span = float(np.linalg.norm(np.ptp(positions, axis=0)))
            movement[side] = span
    p = player_at(rows, f, identity)
    if p is None:
        # A nearby observed pose can inform type, but never becomes a measured
        # contact position or a persistent 3D avatar at a missing frame.
        p = next((player_at(rows, k, identity) for k in range(max(0,f-2), min(len(rows),f+3))
                  if player_at(rows, k, identity)), None)
    feet = p.get("court_m") if p else None
    overhead_frames = sum(bool(v["overhead"] and v["elevated_arm"]) for v in contact)
    positive = sum(s > .08 for s in side_values)
    negative = sum(s < -.08 for s in side_values)
    side = "unknown"
    if len(side_values) >= 3 and max(positive, negative)/len(side_values) >= .7:
        side = "forehand" if positive > negative else "backhand"
    return {"overhead_contact_frames": overhead_frames, "pre_contact_raised_arms": toss_sides,
            "wrist_movement_body_heights": movement, "anatomical_side_samples": side_values,
            "swing_side": side, "racquet_observed_near_contact_frames": equipment_frames,
            "player_feet_court_m": feet, "pose_contact_frames": len(contact)}


def classify_shots(rows, events, fps, profiles, cuts=()):
    balls = observed_balls(rows)
    shots = []
    for event in events:
        if event["type"] != "hit" or event["status"] == "rejected":
            continue
        identity = event["player_id"]
        result = {"event_id": event["id"], "frame": event["frame"], "time_s": event["time_s"],
                  "event_status": event["status"], "player_id": identity, "classification": "unknown",
                  "classification_status": "candidate", "support": "insufficient", "reasons": [],
                  "evidence": {}, "handedness": profiles.get(identity, {"hand": "unknown"})}
        result['contact_support']=event.get('contact_support','legacy_candidate')
        result['contact_candidate']=event.get('contact_candidate')
        result['confidence_note']='Qualitative evidence support, not a calibrated probability'
        assessment=event.get('play_assessment')
        result['action_state']=assessment['action_state'] if assessment else ('confirmed_shot' if event['status']=='confirmed' else 'uncertain_contact')
        result['play_assessment']=assessment
        result['execution']={'striking_hands':'unknown','execution':'unknown','stroke_side':'unknown','swing_type':'unknown'}
        if assessment and assessment['action_state'] not in ('shot_candidate','confirmed_shot'):
            result['display_label']='no shot (candidate)' if assessment['action_state']=='no_shot_candidate' else 'uncertain contact'
            result['reasons'].append(assessment['reason'])
            shots.append(result)
            continue
        if event.get('contact_support') in ('not_supported_by_new_pass','review_window_only') and event['status']!='confirmed':
            result['reasons'].append('Existing hit candidate lacks independent contact support in the new pass')
            shots.append(result)
            continue
        if identity not in ("near", "far"):
            result["reasons"].append("Player identity has not been established")
            shots.append(result)
            continue
        execution=execution_evidence(rows,event,fps,cuts)
        result['execution']=execution
        hand=execution['striking_hands'] if execution['striking_hands'] in ('left','right') else 'unknown'
        features = _temporal_features(rows, balls, event, fps, hand)
        result["evidence"] = features
        f = event["frame"]
        if any(abs(c-f) <= round(.9*fps) for c in cuts):
            result["reasons"].append("Camera cut interrupts surrounding movement")
            shots.append(result)
            continue
        feet = features["player_feet_court_m"]
        at_baseline = feet is not None and (feet[1] <= 3 or feet[1] >= 20.77)
        near_net = feet is not None and abs(feet[1]-11.885) < 4.8
        motion = max(features["wrist_movement_body_heights"].values(), default=0)
        if features["pose_contact_frames"] < 3 or motion < .10:
            result["reasons"].append("Insufficient observed pose/movement around contact")
        elif at_baseline and features["overhead_contact_frames"] >= 1 and features["pre_contact_raised_arms"]:
            # Overhead + baseline + preceding raised-arm/toss pattern. A smash
            # can still imitate these cues, so this remains a serve candidate.
            result.update(classification="serve", support="moderate")
            result["reasons"].append("Baseline position, overhead contact and preceding raised-arm pattern")
        elif (features["overhead_contact_frames"] >= 2 and not features['pre_contact_raised_arms']
              and features['racquet_observed_near_contact_frames']>=2):
            result.update(classification='overhead',support='moderate')
            result['reasons'].append('Repeated overhead contact/arm evidence and visible racquet, without a preceding toss pattern')
        elif features['overhead_contact_frames']>=2:
            result['reasons'].append('Overhead movement remains ambiguous between serve and overhead')
        else:
            previous = [e for e in events if e["status"] == "confirmed" and e["type"] == "hit"
                        and e["player_id"] not in (None, identity) and 0 < f-e["frame"] <= round(2.5*fps)]
            previous_hit = max(previous, key=lambda e:e["frame"]) if previous else None
            between = [e for e in events if previous_hit and previous_hit["frame"] < e["frame"] < f
                       and e["status"] != "rejected"]
            confirmed_bounce = any(e["type"] == "bounce" and e["status"] == "confirmed" for e in between)
            ambiguous = any(e["status"] != "confirmed" for e in between)
            coverage = (sum(b is not None for b in balls[previous_hit["frame"]:f+1])/(f-previous_hit["frame"]+1)
                        if previous_hit else 0)
            complete_incoming = bool(previous_hit and coverage >= .9
                                     and _max_missing(balls, previous_hit["frame"], f) <= 2
                                     and not any(previous_hit["frame"] < c <= f for c in cuts))
            features.update(incoming_observation_coverage=coverage, confirmed_incoming_bounce=confirmed_bounce,
                            ambiguous_incoming_events=ambiguous)
            compact = .10 <= motion <= .55
            if (near_net and complete_incoming and not confirmed_bounce and not ambiguous and compact
                    and features["racquet_observed_near_contact_frames"] >= 2):
                result.update(classification="volley", support="moderate")
                result["reasons"].append("Near net, compact swing, racquet visible, and well-observed incoming flight without a reviewed bounce")
            elif near_net and not confirmed_bounce:
                result["reasons"].append("Possible volley, but no-bounce evidence is incomplete; absence of a detected bounce is not proof")
            elif execution['swing_type'] != 'unknown':
                result.update(classification=execution['swing_type'], support="moderate")
                result["reasons"].append("Per-stroke racquet-hand evidence and anatomical swing side; no fixed player handedness used")
            else:
                result["reasons"].append("Per-stroke hand/grip or anatomical-side evidence is ambiguous; two hands do not imply a backhand")
        if event["status"] != "confirmed":
            result["reasons"].append("Underlying hit is not human-confirmed; it may be a catch/toss or false event")
        shots.append(result)
    for s in shots:
        s.setdefault('display_label',s['classification'] if s['classification']!='unknown' else 'unknown stroke')
    eligible=[s for s in shots if s['action_state'] in ('shot_candidate','confirmed_shot')]
    return {"method": "per_stroke_execution_v3", "profiles": profiles, "shots": shots,
            "counts": dict(Counter(s["classification"] for s in eligible)),
            'action_counts':dict(Counter(s['action_state'] for s in shots)),
            'by_player':{identity:dict(Counter(s['classification'] for s in eligible if s['player_id']==identity)) for identity in ('near','far')},
            "limitation": "Heuristic stroke and hand-use candidates, not measured grip or trained tennis classification. Player hand tendencies never restrict a stroke. Two-handed forehand/backhand remains unknown without adequate grip/orientation evidence or human review."}
