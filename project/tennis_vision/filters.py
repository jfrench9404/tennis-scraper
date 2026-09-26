"""Court-footprint and player/racket association filters. No model dependency."""
import math


def feet_pixel(player):
    joints = player.keypoints or {}
    ankles = [joints[n] for n in ('left_ankle', 'right_ankle')
              if n in joints and len(joints[n]) >= 3 and joints[n][2] >= .25
              and all(math.isfinite(v) for v in joints[n][:3])]
    if ankles:
        return tuple(sum(p[i] for p in ankles) / len(ankles) for i in (0, 1))
    x1, _, x2, y2 = player.bbox
    return ((x1 + x2) / 2, y2)


def on_court(player, court, side_margin=1.5, baseline_margin=6.0):
    # Do not reject people using the unvalidated automatic Hough estimate.
    if getattr(player, 'identity_id', None) is not None:
        return True
    if court is None or court.source != 'manual':
        return True
    x, y = court.project(feet_pixel(player))
    return (math.isfinite(x) and math.isfinite(y)
            and -side_margin <= x <= 10.97 + side_margin
            and -baseline_margin <= y <= 23.77 + baseline_margin)


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return intersection / max(aa+ab-intersection, 1e-9), intersection / max(min(aa, ab), 1e-9)


def point_box_distance(point, box):
    x, y = point[:2]
    return math.hypot(max(box[0]-x, 0, x-box[2]), max(box[1]-y, 0, y-box[3]))


def filter_tracks(items, court, side_margin=1.5, baseline_margin=6.0):
    """Works on detections or tracks; returns retained objects and per-frame counts.

    Court membership only uses feet, never airborne ball/racket projections.
    Racket boxes use suppression plus at most one assignment per visible player.
    Missing wrists use a stricter body-box proximity fallback.
    """
    players = [p for p in items if p.label == 'player'
               and on_court(p, court, side_margin, baseline_margin)]
    rackets = [r for r in items if r.label == 'racket' and not getattr(r, 'predicted', False)]
    deduplicated = []
    for racket in sorted(rackets, key=lambda r: r.confidence, reverse=True):
        if any((lambda v: v[0] >= .35 or v[1] >= .7)(overlap(racket.bbox, kept.bbox)) for kept in deduplicated):
            continue
        deduplicated.append(racket)
    pairs = []
    for ri, racket in enumerate(deduplicated):
        for pi, player in enumerate(players):
            if getattr(player, 'predicted', False):
                continue
            height = max(1, player.bbox[3]-player.bbox[1])
            joints = player.keypoints or {}
            wrists = [joints[n] for n in ('left_wrist', 'right_wrist')
                      if n in joints and len(joints[n]) >= 3 and joints[n][2] >= .3
                      and all(math.isfinite(v) for v in joints[n][:3])]
            if wrists:
                distance = min(point_box_distance(w, racket.bbox) for w in wrists)
                allowed = max(6, .22*height)
            else:
                distance = point_box_distance(racket.center, player.bbox)
                allowed = max(4, .10*height)
            if distance <= allowed:
                pairs.append((distance/allowed + .25*(1-racket.confidence), ri, pi))
    used_rackets, used_players, kept_rackets = set(), set(), []
    for _, ri, pi in sorted(pairs):
        if ri in used_rackets or pi in used_players:
            continue
        racket = deduplicated[ri]
        if hasattr(racket, 'player_track_id'):
            racket.player_track_id = getattr(players[pi], 'track_id', None)
        kept_rackets.append(racket)
        used_rackets.add(ri)
        used_players.add(pi)
    stats = {'off_court_players': sum(p.label == 'player' for p in items)-len(players),
             'duplicate_rackets': len(rackets)-len(deduplicated),
             'unassigned_rackets': len(deduplicated)-len(kept_rackets)}
    return [p for p in items if p.label not in ('player', 'racket')] + players + kept_rackets, stats
