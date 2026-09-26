"""Conservative offline 2D event candidates. Estimates are NEVER event evidence.

Scores rank review priority; they are not calibrated probabilities. A single
camera cannot establish exact contact, ground contact, or airborne XYZ.
"""
import math

import numpy as np


VERSION = "event-review-1"


def observed_balls(rows):
    """Return one unambiguous learned observation per frame; never motion/fill."""
    result = []
    for row in rows:
        valid = []
        for track in row.get("tracks", []):
            box = track.get("bbox", [])
            if (track.get("label") == "ball" and not track.get("predicted", False)
                    and track.get("source", "model") in ("model", "gridtracknet")
                    and len(box) == 4 and np.isfinite(box).all()
                    and box[2] >= box[0] and box[3] >= box[1]):
                valid.append({"pixel": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                              "source": track.get("source", "model"),
                              "detector_score": track.get("confidence"),
                              "track_id": track.get("track_id")})
        result.append(valid[0] if len(valid) == 1 else None)
    return result


def _box_distance(point, box):
    x, y = point
    return math.hypot(max(box[0] - x, 0, x - box[2]), max(box[1] - y, 0, y - box[3]))


def _joint(player, name):
    value = (player.get("keypoints") or {}).get(name)
    if value and len(value) >= 3 and np.isfinite(value[:3]).all() and value[2] >= .3:
        return np.asarray(value[:2], dtype=float)
    return None


def _players(row):
    return [p for p in row.get("tracks", []) if p.get("label") == "player"
            and p.get("identity_id") and not p.get("predicted")]


def proximity(row, pixel):
    """Associate using player-scaled distances, not a fixed far-court radius."""
    matches = []
    for player in _players(row):
        height = max(1., player["bbox"][3] - player["bbox"][1])
        rackets = [r for r in row["tracks"] if r.get("label") == "racket"
                   and not r.get("predicted") and player.get("track_id") is not None
                   and r.get("player_track_id") == player["track_id"]]
        racket_distance = min((_box_distance(pixel, r["bbox"]) / height for r in rackets), default=None)
        wrist_distance = min((float(np.linalg.norm(w - pixel)) / height
                              for name in ("left_wrist", "right_wrist")
                              if (w := _joint(player, name)) is not None), default=None)
        close_racket = racket_distance is not None and racket_distance <= .16
        close_wrist = wrist_distance is not None and wrist_distance <= .32
        if close_racket or close_wrist:
            matches.append({"player": player, "racket_distance_heights": racket_distance,
                            "wrist_distance_heights": wrist_distance,
                            "racket_close": close_racket, "wrist_close": close_wrist,
                            "distance_rank": min(racket_distance if racket_distance is not None else 99.,
                                                 wrist_distance if wrist_distance is not None else 99.)})
    return min(matches, key=lambda m: m["distance_rank"]) if matches else None


def _swing_speed(rows, frame, identity, radius):
    """Body-relative wrist/racquet movement, in player heights per frame."""
    points = {"left_wrist": [], "right_wrist": [], "racket": []}
    for f in range(max(0, frame-radius), min(len(rows), frame+radius+1)):
        player = next((p for p in _players(rows[f]) if p["identity_id"] == identity), None)
        if player is None:
            continue
        b = player["bbox"]
        height = max(1., b[3]-b[1])
        anchor = np.asarray([(b[0]+b[2])/2, (b[1]+b[3])/2])
        for name in ("left_wrist", "right_wrist"):
            point = _joint(player, name)
            if point is not None:
                points[name].append((f, (point-anchor)/height))
        racket = next((r for r in rows[f]["tracks"] if r.get("label") == "racket"
                       and not r.get("predicted") and player.get("track_id") is not None
                       and r.get("player_track_id") == player["track_id"]), None)
        if racket:
            a = racket["bbox"]
            points["racket"].append((f, (np.asarray([(a[0]+a[2])/2, (a[1]+a[3])/2])-anchor)/height))
    speeds = []
    for values in points.values():
        # Use endpoints separated in time, not a single noisy joint jump.
        if len(values) >= 3 and values[-1][0]-values[0][0] >= 3:
            speeds.append(float(np.linalg.norm(values[-1][1]-values[0][1]))/(values[-1][0]-values[0][0]))
    return max(speeds) if speeds else None


def _motion(balls, frame, radius):
    left = [f for f in range(max(0, frame-radius), frame+1) if balls[f]]
    right = [f for f in range(frame, min(len(balls), frame+radius+1)) if balls[f]]
    if len(left) < 3 or len(right) < 3:
        return None
    indices = sorted(set(left+right))
    if max(np.diff(indices)) > 3:
        return None
    def fit(ids):
        t = np.asarray(ids)-frame
        p = np.asarray([balls[f]["pixel"] for f in ids])
        coef = np.polyfit(t, p, 1)
        return coef[0], np.sum((p-(t[:, None]*coef[0]+coef[1]))**2)
    before, error1 = fit(left)
    after, error2 = fit(right)
    t = np.asarray(indices)-frame
    p = np.asarray([balls[f]["pixel"] for f in indices])
    quadratic = np.column_stack((t*t, t, np.ones(len(t))))
    coeff = np.linalg.lstsq(quadratic, p, rcond=None)[0]
    smooth_error = float(np.sqrt(np.mean((p-quadratic@coeff)**2)))
    corner_error = float(np.sqrt((error1+error2)/(2*(len(left)+len(right)))))
    speed1, speed2 = float(np.linalg.norm(before)), float(np.linalg.norm(after))
    if min(speed1, speed2) < .35:
        return None
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)/np.diff(indices)
    if (corner_error > max(2.5, .45*min(speed1, speed2))
            or max(steps) > max(25., 4*float(np.median(steps)))):
        return None  # Discontinuous detector jumps are not physical impacts.
    angle = math.degrees(math.acos(float(np.clip(np.dot(before, after)/(speed1*speed2), -1, 1))))
    return {"before": before, "after": after, "turn_degrees": angle,
            "speed_ratio": max(speed1, speed2)/min(speed1, speed2),
            "velocity_change_px_frame": float(np.linalg.norm(after-before)),
            "corner_rmse_px": corner_error, "smooth_rmse_px": smooth_error,
            "support_frames": indices,
            "frame_range": [left[-2], right[1]]}


def detect_events(rows, balls, fps, court=None, cuts=()):
    """Find review candidates, requiring observations on BOTH sides of a turn."""
    radius = max(3, round(fps*.167))
    candidates = []
    for f, observation in enumerate(balls):
        if not observation or any(abs(f-c) <= radius for c in cuts):
            continue
        m = _motion(balls, f, radius)
        if not m or m["velocity_change_px_frame"] < .8 * 30/fps:
            continue
        if m["turn_degrees"] < 30 and m["speed_ratio"] < 1.8:
            continue
        # A smooth arc/apex is not an impact. Noise also needs multiple supports.
        sharp = m["smooth_rmse_px"] > m["corner_rmse_px"]*1.10 + .08
        if not sharp:
            continue
        association = proximity(rows[f], observation["pixel"])
        swing = (_swing_speed(rows, f, association["player"]["identity_id"], radius)
                 if association else None)
        ground = court.project(tuple(observation["pixel"])) if court is not None else None
        ground_plausible = ground is not None and -2 <= ground[0] <= 12.97 and -6 <= ground[1] <= 29.77
        hit = bool(association and (association["racket_close"] or (swing is not None and swing > .008*30/fps)))
        bounce = (not association and ground_plausible and m["before"][1] > .35*30/fps
                  and m["after"][1] < -.35*30/fps and m["turn_degrees"] >= 35)
        if not hit and not bounce:
            continue
        kind = "hit_candidate" if hit else "bounce_candidate"
        score = min(.95, .35 + min(m["turn_degrees"]/180, .3)
                    + .1*int(m["corner_rmse_px"] < 2)
                    + .1*int(hit and association["racket_close"])
                    + .1*int(hit and swing is not None and swing > .008*30/fps))
        evidence = {k: v for k, v in m.items() if k not in ("before", "after", "frame_range")}
        evidence["incoming_px_per_frame"] = m["before"].tolist()
        evidence["outgoing_px_per_frame"] = m["after"].tolist()
        evidence["ball_evidence"] = "learned_observations_only"
        if hit:
            evidence.update({k: association[k] for k in ("racket_distance_heights", "wrist_distance_heights", "racket_close")})
            evidence["body_relative_swing_heights_per_frame"] = swing
        candidates.append({"type": kind, "frame": f, "time_s": f/fps,
                           "frame_range": m["frame_range"], "pixel": observation["pixel"],
                           "player_id": association["player"]["identity_id"] if hit else None,
                           "player_feet_court_m": association["player"].get("court_m") if hit else None,
                           "landing_if_bounce_m": ground if bounce else None,
                           "score": round(score, 3), "status": "unreviewed", "evidence": evidence,
                           "limitation": ("Ball observation near player equipment; NOT exact ball/string contact or 3D."
                                          if hit else "2D turn consistent with a bounce; court location valid ONLY if ground contact is confirmed.")})
    # Non-maximum suppression has a bounded radius; no transitive chain merging.
    selected = []
    for event in sorted(candidates, key=lambda e: (e["evidence"]["corner_rmse_px"]/e["evidence"]["smooth_rmse_px"], -e["score"], e["frame"])):
        if not any(abs(e["frame"]-event["frame"]) <= round((.20 if e["type"] == event["type"] else .10)*fps) for e in selected):
            selected.append(event)
    selected.sort(key=lambda e: e["frame"])
    for event in selected:
        event["id"] = f'{event["type"]}-{event["frame"]:06d}'
    return selected


def fill_short_gaps(rows, balls, fps, events=(), cuts=(), max_gap=3):
    """Display-only bounded interpolation. No smoothing or extrapolation."""
    if not 0 <= max_gap <= 3:
        raise ValueError("max_gap must be between 0 and 3 frames")
    samples = [{"frame": f, "time_s": f/fps, "status": "detected" if ball else "missing",
                "pixel": ball["pixel"] if ball else None,
                "observation": ball, "estimate": None} for f, ball in enumerate(balls)]
    gaps = []
    f = 0
    while f < len(balls):
        if balls[f]:
            f += 1
            continue
        start = f
        while f < len(balls) and balls[f] is None:
            f += 1
        end, a, b = f-1, start-1, f
        reason = None
        if end-start+1 > max_gap:
            reason = "long_gap"
        elif a <= 0 or b+1 >= len(balls) or not balls[a-1] or not balls[b+1]:
            reason = "insufficient_observed_context"
        elif any(a-1 < c <= b+1 for c in cuts):
            reason = "scene_cut"
        elif any(e["frame_range"][0]-2 <= b and e["frame_range"][1]+2 >= a for e in events):
            reason = "event_neighbourhood"
        else:
            pa, pb = np.asarray(balls[a]["pixel"]), np.asarray(balls[b]["pixel"])
            incoming = pa - balls[a-1]["pixel"]
            outgoing = np.asarray(balls[b+1]["pixel"]) - pb
            bridge = (pb-pa)/(b-a)
            speeds = [float(np.linalg.norm(v)) for v in (incoming, bridge, outgoing)]
            if min(speeds) < .35*30/fps or max(speeds)/min(speeds) > 2.5:
                reason = "inconsistent_speed"
            elif any(float(np.dot(v, bridge)/(np.linalg.norm(v)*np.linalg.norm(bridge))) < .85 for v in (incoming, outgoing)):
                reason = "possible_turn_or_impact"
            elif any(proximity(rows[k], balls[k]["pixel"]) for k in (a, b)):
                reason = "near_player_equipment"
            else:
                for k in range(start, end+1):
                    samples[k].update(status="estimated", pixel=(pa+(pb-pa)*(k-a)/(b-a)).tolist(),
                                      estimate={"method": "bounded_linear_display_only", "endpoints": [a, b],
                                                "not_event_evidence": True})
        gaps.append({"start_frame": start, "end_frame": end, "length": end-start+1,
                     "filled": reason is None, "reason": reason or "consistent_short_gap"})
    return samples, gaps
