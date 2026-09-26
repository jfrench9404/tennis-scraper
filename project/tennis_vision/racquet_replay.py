"""Per-frame racquet display records for the 3D replay.

Evidence is a detected racquet BOX associated with one identified player (near
or far) and at least one supported wrist of that player in the same frame.
Nothing here measures handle endpoints, grip, face orientation or string-bed
contact: the grip is placed at the supporting wrist(s) and the racquet axis
points toward the box centre lifted onto the player's 2.5D display plane.
Records are display geometry only and never feed contact/shot decisions.
"""
import numpy as np

from .event_analysis import _box_distance, _joint

LENGTH_M = 0.685          # 27 in, regulation maximum; a display scale, not a measurement.
HOLD_FRAMES_S = 0.2       # Brief detector dropouts are held and labelled stale.


def _plane_point(camera, record, pixel):
    """Ray/plane intersection matching shot_replay.pose_on_plane's display plane."""
    base = np.asarray(record["raw_feet_xyz_m"], float)
    shift = np.asarray(record["feet_xyz_m"], float) - base
    origin = camera.position_m
    normal = origin - base
    normal[2] = 0
    normal /= max(np.linalg.norm(normal), 1e-9)
    uv = camera.normalized_pixels([pixel])[0]
    ray = camera.R.T @ np.array([uv[0], uv[1], 1.])
    denominator = float(np.dot(normal, ray))
    if abs(denominator) < 1e-8:
        return None
    distance = float(np.dot(normal, base - origin) / denominator)
    if distance <= 0:
        return None
    return origin + distance * ray + shift


def hand_for(player, racquet_box):
    """Which hand(s) hold this racquet in THIS frame (same rule as stroke_execution)."""
    left, right = _joint(player, "left_wrist"), _joint(player, "right_wrist")
    h = max(1, player["bbox"][3] - player["bbox"][1])
    if left is not None and right is not None:
        dl, dr = _box_distance(left, racquet_box) / h, _box_distance(right, racquet_box) / h
        if max(dl, dr) < .10 and np.linalg.norm(left - right) / h < .16:
            return "both"
        if dl < .14 and dr - dl > .07:
            return "left"
        if dr < .14 and dl - dr > .07:
            return "right"
        return None
    for name, wrist in (("left", left), ("right", right)):
        if wrist is not None and _box_distance(wrist, racquet_box) / h < .14:
            return name  # Only one wrist supported; the other hand is unknown.
    return None


def _grip(record, hand):
    names = ["left_wrist", "right_wrist"] if hand == "both" else [hand + "_wrist"]
    points = [record["joints_m"].get(n) for n in names]
    if any(p is None for p in points):
        return None
    return np.mean(np.asarray(points, float), axis=0)


def build_racquets(rows, frames, camera, fps, cuts=()):
    """Attach frame['racquets'] (list) to replay frames built by build_players."""
    hold = max(1, round(HOLD_FRAMES_S * fps))
    last = {}
    counts = {"observed": 0, "stale": 0, "box_without_supported_wrist": 0, "hands": {}}
    for f, (row, frame) in enumerate(zip(rows, frames)):
        frame["racquets"] = []
        if any(f - 1 < c <= f for c in cuts):
            last.clear()  # Never carry equipment across a cut.
        for record in frame["players"]:
            identity = record["identity_id"]
            if record.get("predicted") or camera is None:
                continue
            tracks = [t for t in row["tracks"] if t.get("label") == "player" and t.get("identity_id") == identity]
            player = tracks[0] if len(tracks) == 1 else None
            boxes = [] if player is None else [
                t for t in row["tracks"] if t.get("label") == "racket" and not t.get("predicted")
                and (t.get("player_track_id") == player.get("track_id") and player.get("track_id") is not None)]
            association = "tracker_wrist_assignment"
            if player is not None and not boxes and player.get("pose_source") == "focused_crop_model":
                # Crop-refined far poses carry new track ids; re-associate by the
                # same wrist-proximity rule, still restricted to this identity.
                boxes = [t for t in row["tracks"] if t.get("label") == "racket" and not t.get("predicted")
                         and t.get("player_track_id") is None and hand_for(player, t["bbox"])]
                association = "wrist_proximity_same_identity"
            placed = None
            if player is not None and len(boxes) == 1:
                box = boxes[0]
                hand = hand_for(player, box["bbox"])
                grip = _grip(record, hand) if hand else None
                if grip is None:
                    counts["box_without_supported_wrist"] += 1
                else:
                    centre = ((box["bbox"][0] + box["bbox"][2]) / 2, (box["bbox"][1] + box["bbox"][3]) / 2)
                    head = _plane_point(camera, record, centre)
                    axis = None if head is None else head - grip
                    basis = "wrist_to_box_centre_on_player_plane"
                    if axis is None or not np.isfinite(axis).all() or not .12 <= np.linalg.norm(axis) <= 1.2:
                        axis, basis = None, "box_seen_axis_unresolved"
                    if axis is not None:
                        placed = {"identity_id": identity, "hand": hand, "status": "observed", "age_frames": 0,
                                  "grip_m": np.round(grip, 4).tolist(),
                                  "axis_unit": np.round(axis / np.linalg.norm(axis), 4).tolist(),
                                  "confidence": round(float(box.get("confidence", 0)), 3),
                                  "bbox": box["bbox"], "association": association, "axis_basis": basis}
                        last[identity] = (f, placed)
                        counts["observed"] += 1
                        counts["hands"][hand] = counts["hands"].get(hand, 0) + 1
            if placed is None and identity in last and f - last[identity][0] <= hold:
                previous = last[identity][1]
                grip = _grip(record, previous["hand"])
                if grip is not None:
                    placed = dict(previous, status="stale", age_frames=f - last[identity][0],
                                  grip_m=np.round(grip, 4).tolist())
                    counts["stale"] += 1
            if placed is not None:
                placed["length_m"] = LENGTH_M
                frame["racquets"].append(placed)
    return {"method": "racquet_box_wrist_display_v1", "hold_frames": hold, "counts": counts,
            "limitations": ["Racquet boxes do not measure handle endpoints, grip, face orientation or string-bed contact.",
                            "Axis points from the supporting wrist toward the box centre on the 2.5D player plane; face orientation is a display choice.",
                            f"Held up to {hold} frames after the last observed box and labelled stale; hidden afterwards.",
                            "Rendered racquets are never contact evidence and do not create or confirm events."]}
