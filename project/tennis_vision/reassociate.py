"""Offline near/far re-association by COURT SIDE, plus anonymous person tracks.

Reads a finished run (``events.jsonl``) or the complete leading chunks of a long
run, and writes a NEW run folder with the same contract (``events.jsonl``,
``summary.json``, ``source-offset.json``; ``shots.json``/``flight3d.json``/
``ball-model.json`` copied unchanged when present) so event review and the
replay builder can read it. The source run is never modified.

Roles (decided by John, 2026-10-06):

* ``identity_id`` ``near``/``far`` is the COURT SIDE of the player's feet
  (near: court y < 11.885 m, far: beyond the net), chosen per frame among the
  saved pre-filter ``detail.candidates``. Nothing is invented: every player
  track written is a saved candidate (bbox, keypoints, confidence unchanged);
  original ``predicted`` player tracks (motion estimates) are dropped.
* People whose feet stay outside the court margins for a sustained time
  (referee, bystander, adjacent court) can never hold a role.
* Each player track also carries ``person_id``: ``player_a``/``player_b``
  (anonymous, no names) or ``unknown``. Persons are followed through short
  conservative tracklets of saved candidates; a person is linked across a gap
  (e.g. a changeover) only by saved evidence: one continuous tracklet (the walk
  across the net observed) or a short, unique gap in which only one person could
  have moved from where one tracklet ended to where the next began. Anything
  else, including "the other player by elimination", leaves the person
  ``unknown`` (the elimination is recorded as a low-confidence proposal for John).
* Optional ``--video`` (John, 2026-10-16): clothing colour of the SAVED candidate
  boxes (torso/shorts HSV histograms, CPU decode of only the sampled frames, no
  model) is extra evidence for linking a person across a gap. It only labels
  persons, never adds or moves a box, skips occluded candidates, and leaves the
  person ``unknown`` when colour is ambiguous. Without ``--video`` the output is
  exactly the geometry-only result.
* Break proposals from ``segments.py`` (#24) are attached to stretches as
  timing context only; timing alone never links a person.

Everything written here is an estimate from detector output, not ground truth.
Coverage counts are not accuracy. Thresholds: DEFAULTS were decided by John
(2026-10-16); COLOUR_DEFAULTS are proposals. Values used go in ``reassociation.json``.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from .court import CourtMapper
from .filters import feet_pixel, filter_tracks
from .tracking import Track

VERSION = "court_side_reassociation_v1"
NET_Y = 11.885
COURT_W, COURT_L = 10.97, 23.77
SIDES = ("near", "far")
PERSONS = ("player_a", "player_b")
# Thresholds decided by John on 2026-10-16 (approved as proposed in PR #41).
DEFAULTS = {
    "side_margin_m": 1.5,          # beyond the doubles sidelines (same as the live run)
    "baseline_margin_m": 6.0,      # behind the baselines (same as the live run)
    "bystander_seconds": 2.0,      # feet outside margins this long (contiguous) -> never a role
    "min_tracklet_frames": 3,      # shorter tracklets never hold a role (flicker)
    "max_missed_frames": 5,        # tracklet may skip this many frames (live tracker uses 5)
    "side_window_seconds": 0.5,    # median feet y over this window decides the side (noise at the net)
    "link_gap_seconds": 1.0,       # longest gap a person may be linked across (unique evidence only)
    "link_speed_mps": 8.0,         # plausible movement during a link gap (+1 m slack)
    "absence_seconds": 3.0,        # a side empty longer than this ends a stretch (re-seed)
    "break_context_seconds": 5.0,  # break proposals this close to a stretch start are attached
}
MEANING = ("near/far are court sides of observed (saved) detector candidates; person ids are anonymous and "
           "linked only on saved continuity evidence, otherwise 'unknown'. Coverage, not accuracy; no ground "
           "truth used. No shot, point, game or changeover is confirmed here.")


# ----------------------------------------------------------------------------- input

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def court_from_yaml_text(text: str) -> CourtMapper:
    raw = yaml.safe_load(text)
    corners = raw.get("image_corners", []) if isinstance(raw, dict) else []
    if len(corners) != 4:
        raise ValueError("court YAML must include exactly four image_corners")
    return CourtMapper(np.asarray(corners, dtype=np.float32), source="manual", confidence=1.0)


def load_source(run: Path, source: str = "auto", court_path: Path | None = None) -> dict[str, Any]:
    """Rows (ordered by run frame), fps, offset, court and source identity of a run."""
    from .segments import load_run
    run = Path(run)
    loaded = load_run(run, source)
    manifest_path = run / "longrun-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    frames = sorted(loaded["rows"])
    rows = [loaded["rows"][f] for f in frames]
    start = frames[0] - rows[0]["frame"]
    for i, (f, row) in enumerate(zip(frames, rows)):
        if row["frame"] != i or f != start + i:
            raise ValueError(f"{run}: rows are not consecutive from run frame 0 (at source frame {f})")
    if court_path is not None:
        court = court_from_yaml_text(Path(court_path).read_text(encoding="utf-8"))
        court_basis = str(court_path)
    elif manifest and manifest.get("settings", {}).get("court"):
        court = court_from_yaml_text(manifest["settings"]["court"])
        court_basis = "longrun-manifest.json settings.court"
    else:
        raise ValueError(f"{run}: no court in longrun-manifest.json; pass --court court.yaml")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True).encode())
    return {"run": run, "rows": rows, "fps": loaded["fps"], "source_start_frame": start, "partial": loaded["partial"],
            "source": loaded["source"], "selection_frames": loaded["selection_frames"], "manifest": manifest,
            "court": court, "court_basis": court_basis,
            "court_corners": court.image_corners.tolist(), "rows_sha256": digest.hexdigest(),
            "chunks": loaded.get("chunks")}


def valid_candidate(c: dict[str, Any]) -> bool:
    box = c.get("bbox") or []
    return (c.get("label") == "player" and len(box) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v)
            for v in box) and box[2] > box[0] and box[3] > box[1] and float(c.get("confidence", 0)) >= .1)


def pose_joints(c: dict[str, Any], threshold: float = .3) -> int:
    return sum(1 for k, v in (c.get("keypoints") or {}).items()
               if k not in ("nose", "left_eye", "right_eye", "left_ear", "right_ear") and v and v[2] >= threshold)


def observations(rows: list[dict[str, Any]], court: CourtMapper, settings: dict[str, float]) -> list[list[dict]]:
    """Per-frame player candidates with court feet; indices refer to detail.candidates."""
    out = []
    for row in rows:
        items = []
        for i, c in enumerate((row.get("detail") or {}).get("candidates") or []):
            if not valid_candidate(c):
                continue
            box = c["bbox"]
            feet = court.project(feet_pixel(SimpleNamespace(bbox=box, keypoints=c.get("keypoints"))))
            if not all(math.isfinite(v) for v in feet):
                continue
            x, y = feet
            inside = (-settings["side_margin_m"] <= x <= COURT_W + settings["side_margin_m"]
                      and -settings["baseline_margin_m"] <= y <= COURT_L + settings["baseline_margin_m"])
            outside_m = math.hypot(max(0., -x, x - COURT_W), max(0., -y, y - COURT_L))
            items.append({"index": i, "bbox": [float(v) for v in box], "confidence": float(c["confidence"]),
                          "height": float(box[3] - box[1]), "centre": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                          "feet": [float(x), float(y)], "inside": inside, "outside_m": outside_m,
                          "pose": pose_joints(c) >= 6})
        out.append(items)
    return out


# ----------------------------------------------------------------------------- tracklets

AMBIGUITY_MARGIN = .15


def build_tracklets(obs: list[list[dict]], max_missed: int) -> list[dict[str, Any]]:
    """Short, conservative person tracklets from saved candidates.

    A candidate continues a tracklet only when the match is unique: if a tracklet
    has two similar candidates, or a candidate two similar tracklets, the
    tracklet ends ('ambiguous') and the candidate starts a new one. Ambiguous
    ends/starts are never bridged later, so two people crossing never swap.
    """
    tracklets: list[dict[str, Any]] = []
    active: list[int] = []
    for f, items in enumerate(obs):
        live = [t for t in active if f - tracklets[t]["frames"][-1] <= max_missed + 1]
        for t in active:
            if t not in live:
                tracklets[t]["end_reason"] = "lost"
        pairs = []
        for t in live:
            last = tracklets[t]["obs"][-1]
            gap = f - tracklets[t]["frames"][-1]
            for j, o in enumerate(items):
                scale = o["height"] / max(1., last["height"])
                dist = math.dist(o["centre"], last["centre"]) / max(1., last["height"])
                if .67 <= scale <= 1.5 and dist <= .6 + .18 * (gap - 1):
                    pairs.append((dist + .5 * abs(math.log(scale)), t, j))
        by_t: dict[int, list[float]] = {}
        by_j: dict[int, list[float]] = {}
        for cost, t, j in pairs:
            by_t.setdefault(t, []).append(cost)
            by_j.setdefault(j, []).append(cost)
        ambiguous_t = {t for t, c in by_t.items() if len(c) > 1 and sorted(c)[1] - sorted(c)[0] < AMBIGUITY_MARGIN}
        ambiguous_j = {j for j, c in by_j.items() if len(c) > 1 and sorted(c)[1] - sorted(c)[0] < AMBIGUITY_MARGIN}
        used_t, used_j = set(), set()
        for cost, t, j in sorted(pairs):
            if t in used_t or j in used_j or t in ambiguous_t or j in ambiguous_j:
                continue
            tracklets[t]["frames"].append(f)
            tracklets[t]["obs"].append(items[j])
            used_t.add(t)
            used_j.add(j)
        for t in ambiguous_t:
            tracklets[t]["end_reason"] = "ambiguous"
        active = [t for t in live if t not in ambiguous_t]
        for j, o in enumerate(items):
            if j in used_j:
                continue
            tracklets.append({"id": len(tracklets), "frames": [f], "obs": [o], "end_reason": None,
                              "start_reason": "ambiguous" if j in ambiguous_j else "new"})
            active.append(len(tracklets) - 1)
    for t in active:
        if tracklets[t]["end_reason"] is None:
            tracklets[t]["end_reason"] = "end_of_rows"
    return tracklets


def classify_tracklets(tracklets: list[dict], fps: float, settings: dict[str, float]) -> None:
    """Smoothed side per observation; ban sustained off-court people; eligibility."""
    half = max(0, int(round(settings["side_window_seconds"] * fps / 2)))
    ban_frames = max(1, int(round(settings["bystander_seconds"] * fps)))
    for t in tracklets:
        frames, ys = t["frames"], [o["feet"][1] for o in t["obs"]]
        t["side"] = []
        lo = 0
        for k, f in enumerate(frames):
            while frames[lo] < f - half:
                lo += 1
            hi = k
            while hi + 1 < len(frames) and frames[hi + 1] <= f + half:
                hi += 1
            t["side"].append("near" if median(ys[lo:hi + 1]) < NET_Y else "far")
        # Sustained off-court person: a contiguous stretch outside the margins of at
        # least bystander_seconds, and outside for most of the tracklet. A player
        # who briefly chases a wide ball (mostly inside) is not banned.
        sustained, run_start = False, None
        for k, o in enumerate(t["obs"] + [None]):
            if o is not None and not o["inside"]:
                run_start = k if run_start is None else run_start
                continue
            if run_start is not None and frames[k - 1] - frames[run_start] + 1 >= ban_frames:
                sustained = True
            run_start = None
        outside = sum(not o["inside"] for o in t["obs"])
        t["banned"] = sustained and outside >= .5 * len(t["obs"])
        t["eligible"] = not t["banned"] and len(frames) >= settings["min_tracklet_frames"]


# ----------------------------------------------------------------------------- side roles

CONTINUITY_BONUS = 1.0


def assign_sides(tracklets: list[dict], frames: int, fps: float, settings: dict[str, float]) -> list[dict]:
    """Per frame, the tracklet observation holding each court side (or None)."""
    by_frame: list[list[tuple[int, int]]] = [[] for _ in range(frames)]
    for t in tracklets:
        if t["eligible"]:
            for k, f in enumerate(t["frames"]):
                by_frame[f].append((t["id"], k))
    absence = int(round(settings["absence_seconds"] * fps))
    holder: dict[str, tuple[int, int]] = {}
    out = []
    for f in range(frames):
        chosen: dict[str, tuple[int, int] | None] = {}
        for side in SIDES:
            best = None
            for tid, k in by_frame[f]:
                t = tracklets[tid]
                if t["side"][k] != side:
                    continue
                o = t["obs"][k]
                cost = (o["outside_m"] + .5 * (not o["pose"]) + .3 * (1 - o["confidence"])
                        - .5 * min(1., len(t["frames"]) / (2 * fps)))
                previous = holder.get(side)
                continuing = bool(previous and previous[0] == tid and f - previous[1] <= absence)
                if not o["inside"] and not continuing:
                    continue  # outside the margins: only the current holder may keep the side
                if continuing:
                    cost -= CONTINUITY_BONUS
                if best is None or cost < best[0]:
                    best = (cost, tid, k)
            chosen[side] = (best[1], best[2]) if best else None
            if best:
                holder[side] = (best[1], f)
        out.append(chosen)
    return out


# ----------------------------------------------------------------------------- persons

class _Union:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def gap_links(tracklets: list[dict], fps: float, settings: dict[str, float]) -> list[dict]:
    """Unique short-gap continuations between tracklets (saved-candidate evidence only)."""
    max_gap = int(round(settings["link_gap_seconds"] * fps))
    usable = [t for t in tracklets if not t["banned"]]
    options: dict[int, list] = {}
    reverse: dict[int, list] = {}
    for e in usable:
        if e["end_reason"] in ("ambiguous", "end_of_rows"):
            continue
        end = e["frames"][-1]
        for s in usable:
            gap = s["frames"][0] - end
            if s is e or not 0 < gap <= max_gap or s["start_reason"] == "ambiguous":
                continue
            distance = math.dist(e["obs"][-1]["feet"], s["obs"][0]["feet"])
            if distance <= 1. + settings["link_speed_mps"] * gap / fps:
                options.setdefault(e["id"], []).append((s["id"], gap, distance))
                reverse.setdefault(s["id"], []).append(e["id"])
    links = []
    for e, found in options.items():
        if len(found) == 1 and len(reverse[found[0][0]]) == 1:
            s, gap, distance = found[0]
            links.append({"from_tracklet": e, "to_tracklet": s, "gap_frames": gap,
                          "gap_s": round(gap / fps, 3), "distance_m": round(distance, 2)})
    return links


def persons(tracklets: list[dict], roles: list[dict], links: list[dict]) -> dict[str, Any]:
    """Union tracklets by links; label the first simultaneous near/far pair player_a/player_b."""
    union = _Union(len(tracklets))
    for link in links:
        union.union(link["from_tracklet"], link["to_tracklet"])
    groups: dict[int, list[int]] = {}
    for t in tracklets:
        if not t["banned"]:
            groups.setdefault(union.find(t["id"]), []).append(t["id"])
    conflict = set()
    for root, members in groups.items():
        spans = sorted((tracklets[m]["frames"][0], tracklets[m]["frames"][-1]) for m in members)
        if any(b[0] <= a[1] for a, b in zip(spans, spans[1:])):
            conflict.add(root)
    label: dict[int, str] = {}
    seed = None
    for f, chosen in enumerate(roles):
        if chosen["near"] and chosen["far"]:
            a, b = union.find(chosen["near"][0]), union.find(chosen["far"][0])
            if a != b and a not in conflict and b not in conflict:
                label[a], label[b], seed = "player_a", "player_b", f
                break
    group_of = {m: root for root, members in groups.items() for m in members}
    return {"group_of": group_of, "label": label, "conflict": conflict, "seed_frame": seed,
            "group_names": {root: f"group-{n}" for n, root in enumerate(sorted(groups, key=lambda r: min(
                tracklets[m]["frames"][0] for m in groups[r])), 1)}, "groups": groups}


def elimination_proposal(root, side_frames, roles, tracklets, people) -> str | None:
    """If, while this unknown group held a side, only one known person was seen (on the
    other side), the group could be the other known person. Low confidence; never applied."""
    label, group_of = people["label"], people["group_of"]
    if len(label) != 2:
        return None
    other_side, observed = set(), set()
    for f, side in side_frames:
        held = roles[f]["far" if side == "near" else "near"]
        if held:
            other_side.add(label.get(group_of[held[0]], "unknown"))
    span = (side_frames[0][0], side_frames[-1][0])
    for root, members in people["groups"].items():
        if root in label and any(span[0] <= f <= span[1] for m in members for f in tracklets[m]["frames"]):
            observed.add(label[root])
    seen_labels = other_side - {"unknown"}
    if len(seen_labels) != 1 or "unknown" in other_side:
        return None
    candidate = next(p for p in PERSONS if p not in seen_labels)
    return None if candidate in observed else candidate


def stretches(roles, tracklets, people, links, fps, settings, start, proposals) -> list[dict]:
    absence = int(round(settings["absence_seconds"] * fps))
    group_of, label, names = people["group_of"], people["label"], people["group_names"]
    raw = []
    for side in SIDES:
        current = None
        for f, chosen in enumerate(roles):
            held = chosen[side]
            if not held:
                continue
            root = group_of[held[0]]
            if current and current["root"] == root and f - current["frames"][-1][0] <= absence + 1:
                current["frames"].append((f, side))
                current["tracklets"].append(held[0])
                continue
            current = {"root": root, "side": side, "frames": [(f, side)], "tracklets": [held[0]]}
            raw.append(current)
    raw.sort(key=lambda s: (s["frames"][0][0], s["side"]))
    link_by_to = {link["to_tracklet"]: link for link in links}
    out, last_of, seen_roots = [], {}, set()
    colour_links, colour_unresolved = people.get("colour_links", {}), people.get("colour_unresolved", {})
    context = settings["break_context_seconds"] * fps
    for s in raw:
        root, first, last = s["root"], s["frames"][0][0], s["frames"][-1][0]
        person = label.get(root, "unknown")
        item = {"person": person, "side": s["side"], "source_start_frame": start + first,
                "source_end_frame": start + last, "basis": "", "group": names[root],
                "observed_frames": len(s["frames"]), "link_confidence": "observed"}
        if root in people["conflict"]:
            item.update(person="unknown", link_confidence="none",
                        basis="unknown: linked tracklets overlap in time (contradictory evidence); person not assigned")
        elif person == "unknown":
            proposal = elimination_proposal(root, s["frames"], roles, tracklets, people)
            item["link_confidence"] = "none"
            item["basis"] = ("unknown: no saved evidence (continuous tracklet or unique short gap) links this "
                             "person to player_a/player_b")
            if proposal:
                item["proposed_person"] = proposal
                item["proposal_confidence"] = "low"
                item["basis"] += (f"; by elimination it could be {proposal} (only the other player was seen "
                                  "meanwhile) - low confidence, needs John's confirmation")
            if root in colour_unresolved:
                evidence = colour_unresolved[root]
                item["colour_evidence"] = evidence
                item["basis"] += f"; clothing colour inconclusive ({evidence['reason']})"
        else:
            previous = last_of.get(person)
            if previous is None:
                item["basis"] = (f"seed: {person} first held a side; first frame with both sides held by different "
                                 f"tracked people is source frame {start + people['seed_frame']}")
            elif root in colour_links and root not in seen_roots:
                evidence = colour_links[root]
                item["colour_evidence"] = evidence
                item["basis"] = (f"linked by clothing colour of saved candidates (torso/shorts HSV, Bhattacharyya): "
                                 f"distance {evidence['distance']} to {person} vs {evidence['other_distance']} to "
                                 f"{evidence['other_person']} ({evidence['samples']} samples)")
                if previous["side"] != s["side"]:
                    item["basis"] += "; side changed"
            elif previous["tracklets"][-1] == s["tracklets"][0]:
                item["basis"] = ("walk across the net observed: one continuous tracklet"
                                 if previous["side"] != s["side"] else "same continuous tracklet after a role gap")
            else:
                chain, t = [], s["tracklets"][0]
                while t in link_by_to and t != previous["tracklets"][-1]:
                    chain.append(link_by_to[t])
                    t = link_by_to[t]["from_tracklet"]
                text = ", ".join(f"gap {c['gap_s']} s / {c['distance_m']} m" for c in reversed(chain))
                item["basis"] = ("linked through unique short-gap continuity of saved candidates"
                                 + (f" ({text})" if text else ""))
                if previous["side"] != s["side"]:
                    item["basis"] += "; side changed"
        near_breaks = [p["id"] for p in proposals
                       if p["source_frames"][1] >= start + first - context and p["source_frames"][0] <= start + first]
        if near_breaks:
            item["break_proposals"] = near_breaks
        out.append(item)
        seen_roots.add(root)
        if person != "unknown" and root not in people["conflict"]:
            last_of[person] = s
    return out


# ----------------------------------------------------------------------------- clothing colour (optional)

# Proposed defaults for the optional --video colour evidence (John approved the
# approach on 2026-10-16; the values themselves are proposals).
COLOUR_DEFAULTS = {
    "colour_samples": 8,            # observations sampled per tracklet (evenly spread, pose preferred)
    "colour_max_distance": .6,      # Bhattacharyya distance to the matched player's seed colour (cf. cached_identity)
    "colour_margin": .2,            # matched player must be this much closer than the other one,
                                    # and the two seed players must differ by at least this much
}
COLOUR_MEANING = ("Clothing colour (HSV histograms of the torso and shorts regions of SAVED candidate boxes/keypoints, "
                  "decoded from the source video on the CPU; no model) used only to link anonymous persons across gaps. "
                  "It never adds or moves a box. Ambiguous colour leaves the person 'unknown'.")
HIST_BINS = [8, 4, 4]


def colour_regions(c: dict[str, Any]) -> dict[str, tuple[int, int, int, int]]:
    """Torso and shorts pixel boxes of a saved candidate (keypoints when confident, else box fractions)."""
    x1, y1, x2, y2 = c["bbox"]
    w, h = x2 - x1, y2 - y1
    joints = c.get("keypoints") or {}
    get = lambda name: joints.get(name) if joints.get(name) and joints[name][2] >= .3 else None
    shoulders = [p for p in (get("left_shoulder"), get("right_shoulder")) if p]
    hips = [p for p in (get("left_hip"), get("right_hip")) if p]
    knees = [p for p in (get("left_knee"), get("right_knee")) if p]
    if len(shoulders) == 2 and len(hips) == 2:
        xs = [p[0] for p in shoulders + hips]
        left, right = min(xs), max(xs)
        inset = .15 * (right - left)
        top, bottom = min(p[1] for p in shoulders), max(p[1] for p in hips)
        torso = (left + inset, top + .1 * (bottom - top), right - inset, bottom)
        hip_y = sum(p[1] for p in hips) / 2
        knee_y = sum(p[1] for p in knees) / 2 if len(knees) == 2 else hip_y + .2 * h
        shorts = (left + inset, hip_y, right - inset, hip_y + .5 * max(1., knee_y - hip_y))
    else:  # same central torso crop as scene.appearance, plus a shorts band
        torso = (x1 + .25 * w, y1 + .22 * h, x2 - .25 * w, y1 + .5 * h)
        shorts = (x1 + .25 * w, y1 + .5 * h, x2 - .25 * w, y1 + .65 * h)
    out = {}
    for name, (a, b, c_, d) in (("torso", torso), ("shorts", shorts)):
        a, b, c_, d = int(math.floor(a)), int(math.floor(b)), int(math.ceil(c_)), int(math.ceil(d))
        if c_ - a < 2:
            a, c_ = a - 1, a + 2
        if d - b < 2:
            b, d = b - 1, b + 2
        out[name] = (a, b, c_, d)
    return out


def colour_descriptor(image: np.ndarray, c: dict[str, Any]) -> dict[str, np.ndarray | None]:
    import cv2
    out = {}
    for name, (a, b, c_, d) in colour_regions(c).items():
        crop = image[max(0, b):min(image.shape[0], d), max(0, a):min(image.shape[1], c_)]
        if crop.size == 0:
            out[name] = None
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, HIST_BINS, [0, 180, 0, 256, 0, 256]).flatten()
        out[name] = (hist / max(float(hist.sum()), 1.)).astype(np.float32)
    return out


def colour_distance(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    """Mean Bhattacharyya distance over the parts both descriptors have (0 same .. 1 disjoint)."""
    import cv2
    if not a or not b:
        return None
    parts = [cv2.compareHist(a[k], b[k], cv2.HISTCMP_BHATTACHARYYA) for k in ("torso", "shorts")
             if a.get(k) is not None and b.get(k) is not None]
    return float(sum(parts) / len(parts)) if parts else None


def _occluded(row: dict[str, Any], index: int, fraction: float = .1) -> bool:
    """True if another saved person box covers more than ``fraction`` of this candidate's box."""
    candidates = (row.get("detail") or {}).get("candidates") or []
    a = candidates[index]["bbox"]
    area = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    for j, other in enumerate(candidates):
        if j == index or not valid_candidate(other):
            continue
        b = other["bbox"]
        inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
        if inter / area > fraction:
            return True
    return False


def tracklet_colours(tracklets, rows, start, reader, colour_settings) -> int:
    """Sample saved candidates of each usable tracklet; decode only the frames needed."""
    plan: dict[int, list[tuple[int, int]]] = {}
    n = int(colour_settings["colour_samples"])
    for t in tracklets:
        t["colour"] = None
        if not t["eligible"] or n <= 0:
            continue
        # Only unoccluded observations: another person's box over this one would mix their colours.
        clear = [k for k in range(len(t["frames"])) if not _occluded(rows[t["frames"][k]], t["obs"][k]["index"])]
        posed = [k for k in clear if t["obs"][k]["pose"]] or clear
        if not posed:
            continue
        picks = sorted({posed[round(i * (len(posed) - 1) / max(1, n - 1))] for i in range(min(n, len(posed)))})
        for k in picks:
            plan.setdefault(start + t["frames"][k], []).append((t["id"], k))
    samples: dict[int, list[dict]] = {}
    for source_frame, image in reader(sorted(plan)):
        for tid, k in plan.get(source_frame, []):
            t = tracklets[tid]
            c = rows[t["frames"][k]]["detail"]["candidates"][t["obs"][k]["index"]]
            samples.setdefault(tid, []).append(colour_descriptor(image, c))
    for tid, items in samples.items():
        tracklets[tid]["colour"] = _mean_descriptor(items)
    return len(plan)


def _mean_descriptor(items: list[dict], weights: list[int] | None = None) -> dict[str, Any] | None:
    weights = weights or [d.get("samples", 1) for d in items]
    out: dict[str, Any] = {"samples": 0}
    for part in ("torso", "shorts"):
        found = [(d[part], w) for d, w in zip(items, weights) if d.get(part) is not None]
        out[part] = (sum(h * w for h, w in found) / sum(w for _, w in found)).astype(np.float32) if found else None
    out["samples"] = int(sum(weights))
    return out if out["torso"] is not None or out["shorts"] is not None else None


def colour_links(tracklets, people, colour_settings) -> dict[str, Any]:
    """Label unknown person groups as player_a/player_b only when colour clearly says so.

    References are the SEED groups only (no drift). A group is linked when it is close
    to one player's seed colour, clearly closer than to the other, the two players are
    distinguishable, and the link contradicts no observation (same person seen twice at
    once). Otherwise it stays 'unknown' and the evidence is recorded.
    """
    label, groups, conflict = people["label"], people["groups"], people["conflict"]
    names = people["group_names"]
    colour_of = {root: _mean_descriptor([tracklets[m]["colour"] for m in members if tracklets[m]["colour"]])
                 for root, members in groups.items()
                 if any(tracklets[m]["colour"] for m in members)}
    seeds = {person: root for root, person in label.items()}
    report: dict[str, Any] = {"links": [], "unresolved": [], "reference_separation": None}
    people["colour_links"], people["colour_unresolved"] = {}, {}
    if len(seeds) != 2 or any(seeds[p] not in colour_of for p in PERSONS):
        report["note"] = "no colour reference for both seed players; colour not used"
        return report
    reference = {p: colour_of[seeds[p]] for p in PERSONS}
    separation = colour_distance(reference["player_a"], reference["player_b"])
    report["reference_separation"] = None if separation is None else round(separation, 4)
    spans = {root: [(tracklets[m]["frames"][0], tracklets[m]["frames"][-1]) for m in members]
             for root, members in groups.items()}
    overlaps = lambda r1, r2: any(a[0] <= b[1] and b[0] <= a[1] for a in spans[r1] for b in spans[r2])
    proposed = {}
    for root in sorted(groups, key=lambda r: names[r]):
        if root in label or root in conflict:
            continue
        desc = colour_of.get(root)
        if desc is None:
            continue
        d = {p: colour_distance(desc, reference[p]) for p in PERSONS}
        if None in d.values():
            continue
        best = min(PERSONS, key=lambda p: d[p])
        other = next(p for p in PERSONS if p != best)
        evidence = {"group": names[root], "samples": desc["samples"], "distance": round(d[best], 4),
                    "other_distance": round(d[other], 4), "closest_person": best, "other_person": other}
        if separation is None or separation < colour_settings["colour_margin"]:
            reason = ("the two players' seed colours are too similar"
                      + ("" if separation is None else f" (distance {separation:.3f})"))
        elif d[best] > colour_settings["colour_max_distance"]:
            reason = f"closest seed colour ({best}) is {d[best]:.3f} away, above {colour_settings['colour_max_distance']}"
        elif d[other] - d[best] < colour_settings["colour_margin"]:
            reason = f"{best} {d[best]:.3f} vs {other} {d[other]:.3f}: margin below {colour_settings['colour_margin']}"
        elif any(label.get(r) == best and overlaps(root, r) for r in groups):
            reason = f"{best} is observed elsewhere at the same time"
        else:
            proposed[root] = (best, evidence)
            continue
        evidence["reason"] = reason
        people["colour_unresolved"][root] = evidence
    # Two groups seen at the same time cannot both be the same person.
    for root, (person, evidence) in list(proposed.items()):
        if any(r != root and p == person and overlaps(root, r) for r, (p, _) in proposed.items()):
            evidence["reason"] = f"another group seen at the same time also matches {person}"
            people["colour_unresolved"][root] = evidence
    for root, (person, evidence) in proposed.items():
        if root in people["colour_unresolved"]:
            continue
        label[root] = person
        evidence["person"] = person
        people["colour_links"][root] = evidence
    report["links"] = list(people["colour_links"].values())
    report["unresolved"] = list(people["colour_unresolved"].values())
    return report


def video_reader(path: Path, width: int | None = None, height: int | None = None, fps: float | None = None):
    """Sequential CPU decode of only the requested source frames (grab skips the rest)."""
    import cv2
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {path}")
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    if (width and info["width"] != width) or (height and info["height"] != height) \
            or (fps and abs(info["fps"] - fps) > .01):
        raise ValueError(f"{path}: {info['width']}x{info['height']} @ {info['fps']} fps does not match the run "
                         f"({width}x{height} @ {fps} fps)")

    def read(needed):
        if needed and needed[-1] >= info["frames"]:
            raise ValueError(f"{path} has {info['frames']} frames; the run needs source frame {needed[-1]}")
        cap = cv2.VideoCapture(str(path))
        try:
            wanted, frame = iter(needed), 0
            target = next(wanted, None)
            while target is not None:
                if frame < target:
                    if not cap.grab():
                        raise ValueError(f"{path}: decode failed at frame {frame}")
                    frame += 1
                    continue
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"{path}: decode failed at frame {frame}")
                yield frame, image
                frame += 1
                target = next(wanted, None)
        finally:
            cap.release()
    return read, info


# ----------------------------------------------------------------------------- rows

def _iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-9)


def _original_reason(track, tracklets_obs, side, settings):
    """Why an original role track is no longer this side's holder (from saved data only)."""
    court_m = track.get("court_m")
    if court_m and all(isinstance(v, (int, float)) for v in court_m):
        x, y = court_m
        if not (-settings["side_margin_m"] <= x <= COURT_W + settings["side_margin_m"]
                and -settings["baseline_margin_m"] <= y <= COURT_L + settings["baseline_margin_m"]):
            return "original holder's feet outside court margins"
    match = max(tracklets_obs, key=lambda item: _iou(item[0]["bbox"], track["bbox"]), default=None)
    if match and _iou(match[0]["bbox"], track["bbox"]) >= .5:
        o, t, k = match
        if t["banned"]:
            return "original holder is a sustained off-court person (bystander rule)"
        if t["side"][k] != side:
            return "original holder's feet are on the other side of the net"
        if not o["inside"]:
            return "original holder's feet outside court margins"
        if not t["eligible"]:
            return "original holder is only a brief tracklet"
        return "another saved candidate on this side has stronger continuity/position support"
    return "original holder has no matching saved candidate"


def rebuild_rows(rows, obs, tracklets, roles, people, court, settings):
    """New rows: player tracks replaced by side holders (saved candidates only)."""
    ids = [t.get("track_id") for row in rows for t in row.get("tracks", []) if isinstance(t.get("track_id"), int)]
    base = max(ids, default=0)
    track_ids = {"near": base + 1, "far": base + 2}
    group_of, label = people["group_of"], people["label"]
    per_frame_obs = [[] for _ in rows]
    for t in tracklets:
        for k, f in enumerate(t["frames"]):
            per_frame_obs[f].append((t["obs"][k], t, k))
    out, changes = [], []
    racket_changes = predicted_dropped = 0
    for f, row in enumerate(rows):
        candidates = (row.get("detail") or {}).get("candidates") or []
        original = {side: next((t for t in row.get("tracks", []) if t.get("label") == "player"
                                and t.get("identity_id") == side and not t.get("predicted")), None) for side in SIDES}
        predicted_dropped += sum(t.get("label") == "player" and bool(t.get("predicted")) for t in row.get("tracks", []))
        players, record = [], {}
        for side in SIDES:
            held = roles[f][side]
            new = None
            if held:
                tracklet = tracklets[held[0]]
                o = tracklet["obs"][held[1]]
                c = candidates[o["index"]]
                root = group_of[tracklet["id"]]
                conflict = root in people["conflict"]
                track = Track("player", tuple(c["bbox"]), float(c["confidence"]), track_ids[side], f,
                              c.get("keypoints"), identity_id=side, source=c.get("source", "model"))
                new = track.as_dict(court)
                new.update(person_id="unknown" if conflict else label.get(root, "unknown"),
                           person_group=people["group_names"][root], identity_basis=VERSION,
                           candidate_index=o["index"])
                if not conflict and root in people.get("colour_links", {}):
                    new["person_evidence"] = "clothing_colour"
                players.append(new)
            old = original[side]
            if old and new:
                if _iou(old["bbox"], new["bbox"]) >= .5:
                    status, reason = "unchanged", "same saved candidate holds this side"
                else:
                    status = "replaced"
                    reason = _original_reason(old, per_frame_obs[f], side, settings)
            elif new:
                status, reason = "added", "original run had no observed identity on this side; saved candidate used"
            elif old:
                status = "removed"
                reason = _original_reason(old, per_frame_obs[f], side, settings)
                if reason.startswith("another saved"):
                    reason = "no eligible saved candidate holds this side"
            else:
                status, reason = "none", "no observed player on this side in either run"
            record[side] = {"status": status, "reason": reason}
            if status not in ("unchanged", "none"):
                changes.append((f, side, status, reason))
        others = [dict(t) for t in row.get("tracks", []) if t.get("label") != "player"]
        rackets = [t for t in others if t.get("label") == "racket" and not t.get("predicted")]
        if rackets:
            subjects = [SimpleNamespace(label="player", bbox=p["bbox"], confidence=p["confidence"],
                                        keypoints=p["keypoints"], track_id=p["track_id"], predicted=False)
                        for p in players]
            items = [SimpleNamespace(label="racket", bbox=r["bbox"], confidence=r["confidence"], predicted=False,
                                     center=((r["bbox"][0] + r["bbox"][2]) / 2, (r["bbox"][1] + r["bbox"][3]) / 2),
                                     player_track_id=None, source=r) for r in rackets]
            kept, _ = filter_tracks(subjects + items, None)
            assigned = {id(k.source): k.player_track_id for k in kept if k.label == "racket"}
            for r, item in zip(rackets, items):
                new_id = assigned.get(id(item.source))
                old_id = r.get("player_track_id")
                old_player = None if old_id is None else next(
                    (t for t in row.get("tracks", []) if t.get("label") == "player" and t.get("track_id") == old_id),
                    None)
                new_player = next((p for p in players if p["track_id"] == new_id), None)
                # Changed = now held by a different side or a different person box.
                if (old_player is None) != (new_player is None) or (old_player and new_player and (
                        old_player.get("identity_id") != new_player["identity_id"]
                        or _iou(old_player["bbox"], new_player["bbox"]) < .5)):
                    racket_changes += 1
                r["player_track_id"] = new_id
        new_row = dict(row)
        new_row["tracks"] = others + players
        new_row["reassociation"] = record
        out.append(new_row)
    return out, changes, {"near": track_ids["near"], "far": track_ids["far"]}, racket_changes, predicted_dropped


def change_intervals(changes, start):
    out = []
    for f, side, status, reason in sorted(changes, key=lambda c: (c[1], c[2], c[3], c[0])):
        last = out[-1] if out else None
        if last and (last["side"], last["status"], last["reason"]) == (side, status, reason) \
                and last["source_frames"][1] == start + f - 1:
            last["source_frames"][1] = start + f
            last["frames"] += 1
        else:
            out.append({"side": side, "status": status, "reason": reason, "source_frames": [start + f, start + f],
                        "frames": 1})
    return sorted(out, key=lambda c: (c["source_frames"][0], c["side"]))


# ----------------------------------------------------------------------------- run

def reassociate(loaded: dict[str, Any], settings: dict[str, float] | None = None,
                colour: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pure computation: new rows plus the report (nothing written).

    ``colour`` (optional): {"frames": reader, "settings": {...}, "video": {...}} where
    ``reader(sorted_source_frames)`` yields (source_frame, BGR image). Without it the
    result is exactly the geometry-only result.
    """
    from .compare_runs import frame_stats
    from .segments import propose
    settings = {**DEFAULTS, **(settings or {})}
    for key, value in settings.items():
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
            raise ValueError(f"{key} must be a finite number >= 0")
    rows, fps, court, start = loaded["rows"], loaded["fps"], loaded["court"], loaded["source_start_frame"]
    obs = observations(rows, court, settings)
    tracklets = build_tracklets(obs, int(settings["max_missed_frames"]))
    classify_tracklets(tracklets, fps, settings)
    roles = assign_sides(tracklets, len(rows), fps, settings)
    links = gap_links(tracklets, fps, settings)
    people = persons(tracklets, roles, links)
    colour_report = None
    if colour is not None:
        colour_settings = {**COLOUR_DEFAULTS, **(colour.get("settings") or {})}
        for key, value in colour_settings.items():
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
                raise ValueError(f"{key} must be a finite number >= 0")
        sampled = tracklet_colours(tracklets, rows, start, colour["frames"], colour_settings)
        colour_report = colour_links(tracklets, people, colour_settings)
        colour_report.update(settings=colour_settings, video=colour.get("video"), sampled_frames=sampled,
                             tracklets_with_colour=sum(t.get("colour") is not None for t in tracklets),
                             meaning=COLOUR_MEANING)
    new_rows, changes, track_ids, racket_changes, predicted_dropped = rebuild_rows(
        rows, obs, tracklets, roles, people, court, settings)
    breaks = propose({"run": "reassociated rows", "source": "reassociated", "partial": loaded["partial"],
                      "fps": fps, "rows": {start + i: r for i, r in enumerate(new_rows)},
                      "selection_frames": loaded["selection_frames"]})
    proposals = [{"id": p["id"], "label": p["label"], "status": p["status"], "source_frames": p["source_frames"],
                  "reasons": p["reasons"]} for p in breaks["proposals"]]
    table = stretches(roles, tracklets, people, links, fps, settings, start, proposals)
    for link in links:
        for key in ("from_tracklet", "to_tracklet"):
            t = tracklets[link[key]]
            link[key.replace("tracklet", "source_frames")] = [start + t["frames"][0], start + t["frames"][-1]]
    coverage = {}
    for name, items in (("original", rows), ("reassociated", new_rows)):
        totals: dict[str, int] = {}
        for row in items:
            for key, value in frame_stats(row).items():
                if key.startswith(("near_", "far_")):
                    totals[key] = totals.get(key, 0) + int(value)
        coverage[name] = totals
    status_counts: dict[str, dict[str, int]] = {side: {} for side in SIDES}
    reason_counts: dict[str, int] = {}
    for row in new_rows:
        for side in SIDES:
            status = row["reassociation"][side]["status"]
            status_counts[side][status] = status_counts[side].get(status, 0) + 1
    for _, side, status, reason in changes:
        key = f"{side} {status}: {reason}"
        reason_counts[key] = reason_counts.get(key, 0) + 1
    person_frames: dict[str, int] = {}
    for row in new_rows:
        for t in row["tracks"]:
            if t.get("label") == "player":
                person_frames[t["person_id"]] = person_frames.get(t["person_id"], 0) + 1
    low = [s for s in table if s.get("proposed_person")]
    report = {
        "settings": settings, "meaning": MEANING,
        "frames": len(new_rows), "source_frames": [start, start + len(new_rows) - 1],
        "player_track_ids": track_ids,
        "tracklets": {"total": len(tracklets), "eligible": sum(t["eligible"] for t in tracklets),
                      "banned_off_court": sum(t["banned"] for t in tracklets),
                      "ended_ambiguous": sum(t["end_reason"] == "ambiguous" for t in tracklets)},
        "seed_source_frame": None if people["seed_frame"] is None else start + people["seed_frame"],
        "stretches": table,
        "person_links": links,
        "low_confidence_proposals": [{k: s[k] for k in ("side", "source_start_frame", "source_end_frame", "group",
                                                        "proposed_person", "basis")} for s in low],
        "break_proposals": proposals,
        "break_proposal_note": "segments.py proposals on the reassociated rows; timing context only, never a link",
        "status_counts": status_counts, "reason_counts": reason_counts,
        "changed_intervals": change_intervals(changes, start),
        "racket_assignments_changed": racket_changes, "predicted_player_tracks_dropped": predicted_dropped,
        "person_frames": person_frames,
        "coverage": {**coverage, "meaning": "Observation coverage over all rows. Not accuracy; no ground truth used."},
    }
    if colour_report is not None:
        report["colour"] = colour_report
    return {"rows": new_rows, "report": report}


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def _save_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_run(loaded: dict[str, Any], result: dict[str, Any], output: Path) -> dict[str, Any]:
    """Write a NEW run folder (standard run contract) plus reassociation.json."""
    source, output = Path(loaded["run"]), Path(output)
    if _inside(output, source) or _inside(source, output):
        raise ValueError("--output must be a new folder outside the source run")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} already exists; choose a new folder (results are never overwritten)")
    output.mkdir(parents=True, exist_ok=True)
    fps, start, rows = loaded["fps"], loaded["source_start_frame"], result["rows"]
    tmp = output / "events.jsonl.partial"
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    os.replace(tmp, output / "events.jsonl")
    events_sha = hashlib.sha256((output / "events.jsonl").read_bytes()).hexdigest()
    _save_json(output / "source-offset.json", {"source_start_frame": start, "source_start_seconds": start / fps,
                                               "note": "Output frames and timestamps are relative to this starting point."})
    copied = []
    for name in ("shots.json", "flight3d.json", "ball-model.json"):
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
            copied.append(name)
    manifest = loaded["manifest"] or {}
    if (source / "summary.json").is_file() and not loaded["partial"]:
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    else:
        info = manifest.get("input", {})
        summary = {"input": info.get("path"), "fps": fps, "width": info.get("width"), "height": info.get("height"),
                   "court_calibrated": True, "court_source": "manual", "court_confidence": 1.0,
                   "note": "Events, shots, and single-camera 3D fits are candidates; validate before use."}
        for name in ("shots.json", "flight3d.json"):
            if name in copied:  # a partial run's merged files would describe other frames
                (output / name).unlink()
                copied.remove(name)
    summary.update(frames=len(rows), events=sum(len(r.get("events", [])) for r in rows),
                   identity_pass={"version": VERSION, "report": "reassociation.json"})
    _save_json(output / "summary.json", summary)
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_identity = {"run": str(source.resolve()), "read": loaded["source"], "partial": loaded["partial"],
                       "rows_sha256": loaded["rows_sha256"],
                       "longrun_fingerprint": manifest.get("fingerprint"),
                       "input_sha256": manifest.get("input", {}).get("sha256"),
                       "court_basis": loaded["court_basis"], "court_image_corners": loaded["court_corners"]}
    if loaded.get("chunks"):
        source_identity["chunks"] = loaded["chunks"]
    identity = {"version": VERSION, "source": source_identity, "settings": result["report"]["settings"],
                "code_sha256": code_sha}
    if "colour" in result["report"]:  # only with --video; without it the fingerprint is unchanged
        identity["colour"] = {"settings": result["report"]["colour"]["settings"],
                              "video": result["report"]["colour"]["video"]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    report = {"schema_version": 1, "kind": "court_side_reassociation", "version": VERSION,
              "fingerprint": fingerprint, "source_run": source_identity, "code_sha256": code_sha,
              "events_sha256": events_sha, "copied_unchanged": copied, **result["report"]}
    _save_json(output / "reassociation.json", report)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="Source run folder (never modified)")
    p.add_argument("--output", type=Path, required=True, help="NEW run folder to write")
    p.add_argument("--court", type=Path, help="court.yaml (default: the long run's manifest settings.court)")
    p.add_argument("--source", choices=("auto", "merged", "chunks"), default="auto",
                   help="auto: events.jsonl if present, else complete chunks/*/events.jsonl")
    for key, value in DEFAULTS.items():
        p.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    p.add_argument("--video", type=Path, help="Optional source video: adds clothing-colour evidence for person links "
                   "(CPU decode of the needed frames only; no model). Without it the output is geometry-only")
    for key, value in COLOUR_DEFAULTS.items():
        p.add_argument("--" + key.replace("_", "-"), type=float, default=value, help="Only with --video")
    return p.parse_args(argv)


def colour_input(video: Path, loaded: dict[str, Any], colour_settings: dict[str, float]) -> dict[str, Any]:
    """Reader plus identity of the source video; refuses a video that is not the run's input."""
    from .longrun import file_sha256
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"--video {video} does not exist")
    summary_path = Path(loaded["run"]) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    info = (loaded["manifest"] or {}).get("input", {})
    reader, probe = video_reader(video, info.get("width") or summary.get("width"),
                                 info.get("height") or summary.get("height"), loaded["fps"])
    sha = file_sha256(video)
    if info.get("sha256") and info["sha256"] != sha:
        raise ValueError(f"--video {video} is not this run's input (sha256 differs from longrun-manifest.json)")
    return {"frames": reader, "settings": colour_settings,
            "video": {"path": str(video), "sha256": sha, "frames": probe["frames"],
                      "matches_run_input": bool(info.get("sha256")) or None}}


def main(argv=None) -> None:
    args = parse_args(argv)
    settings = {key: getattr(args, key) for key in DEFAULTS}
    try:
        loaded = load_source(args.run, args.source, args.court)
        colour = None
        if args.video is not None:
            colour = colour_input(args.video, loaded, {key: getattr(args, key) for key in COLOUR_DEFAULTS})
        result = reassociate(loaded, settings, colour)
        report = write_run(loaded, result, args.output)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error))
    c = report["coverage"]
    print(f"{report['frames']} frames -> {args.output} (fingerprint {report['fingerprint'][:12]})")
    for side in SIDES:
        print(f"  {side}: frames {c['original'].get(side + '_player_frames', 0)} -> "
              f"{c['reassociated'].get(side + '_player_frames', 0)}, 6+ joints "
              f"{c['original'].get(side + '_frames_6plus_joints', 0)} -> "
              f"{c['reassociated'].get(side + '_frames_6plus_joints', 0)}  {report['status_counts'][side]}")
    print(f"  {len(report['stretches'])} stretch(es); {len(report['low_confidence_proposals'])} low-confidence "
          f"person proposal(s) for review. Coverage, not accuracy.")
    if "colour" in report:
        c = report["colour"]
        print(f"  clothing colour: {len(c['links'])} link(s), {len(c['unresolved'])} left unknown "
              f"({c['sampled_frames']} frames decoded; seed separation {c['reference_separation']})")


if __name__ == "__main__":
    main()
