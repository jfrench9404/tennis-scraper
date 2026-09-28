"""Score pipeline hit/bounce candidates (and point proposals) against human labels.

    python -m tennis_vision.score_labels --labels labels.json --run <folder> --output score.json

Only spans that ``labels.json`` marks as fully labelled are scored; every number
names its sample size and labels file (CLAUDE.md rule 3). All matching happens in
SOURCE-VIDEO frames: run-relative candidate frames are shifted by the run's
``source_start_frame``. The labels file is only read, never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

from .labels import LABEL_ABSTAIN_SHOT_TYPES, LabelsError, check_binding, load_labels

DEFAULT_TOLERANCE_FRAMES = 3      # pending John's decision (issue #27)
DEFAULT_POINT_TOLERANCE_S = 1.0   # pending John's decision (issue #27)
SIDES = ("near", "far")

LIMITATIONS = [
    "Accuracy is measured only against the labels file named here and only inside its fully labelled spans; "
    "it says nothing about other footage, other runs or unlabelled frames.",
    "A candidate within the frame tolerance of a label counts as a match; timing inside the tolerance is not "
    "exact contact or bounce timing.",
    "Matching is one-to-one, closest pairs first. Near a span edge a candidate may belong to an unlabelled "
    "event just outside the span.",
    "Bounce in/out calls and point winners are John's labels; the pipeline makes no in/out call, so none is scored.",
    "Scores and support levels of candidates are not probabilities and are not used here.",
]


# ---------------------------------------------------------------- run loading

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _check_complete(folder: Path) -> None:
    status = folder / "build-status.json"
    if status.is_file() and _read_json(status).get("status") != "complete":
        raise ValueError(f"{folder}: build is not complete; score a finished run")


def _source_run_manifest(folder: Path, source_run: str | None, start: int, frames: int, fps: float):
    """Find the long-run manifest of the run a review/replay was built from.

    ``source_run`` is usually an absolute Windows path from John's laptop; if it does
    not exist here, a folder with the same name next to this run (or a parent) is
    tried. It is only accepted when it covers exactly the same source frames.
    """
    if not source_run:
        return None
    candidates = [Path(source_run)]
    name = PureWindowsPath(source_run).name
    for ancestor in list(folder.resolve().parents)[:3]:
        candidates.append(ancestor / name)
    for candidate in candidates:
        manifest_path = candidate / "longrun-manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        selection, video = manifest.get("selection", {}), manifest.get("input", {})
        if (selection.get("source_start_frame") == start and selection.get("frames") == frames
                and abs(float(video.get("fps", 0)) - fps) < 0.01 and video.get("sha256")):
            return candidate, manifest
    return None


def _candidate(cid: str, run_frame: int, start: int, **extra: Any) -> dict[str, Any]:
    return {"id": cid, "run_frame": int(run_frame), "source_frame": int(run_frame) + start, **extra}


def load_run(path: Path | str, video: Path | str | None = None, include_review_windows: bool = False) -> dict[str, Any]:
    """Identity (run id, video hash, frame offset) and hit/bounce candidates of a run folder.

    Accepts a replay folder, an event-review folder, a folder holding ``replay/`` or
    ``review/`` (replay preferred), or a long-run folder with ``shots.json``.
    """
    folder = Path(path)
    if (folder / "longrun-manifest.json").is_file():
        kind = "longrun"
    elif (folder / "reviewed-events.json").is_file():
        kind = "replay"
    elif (folder / "event-candidates.json").is_file():
        kind = "review"
    elif (folder / "replay" / "reviewed-events.json").is_file():
        folder, kind = folder / "replay", "replay"
    elif (folder / "review" / "event-candidates.json").is_file():
        folder, kind = folder / "review", "review"
    else:
        raise ValueError(f"{path}: not a replay, event-review or long-run folder")
    _check_complete(folder)

    hits: list[dict[str, Any]] = []
    bounces: list[dict[str, Any]] = []
    excluded = {"review_windows": 0, "human_added_manual_events": 0, "other_event_types": 0}
    recorded_sha = None
    basis = None
    if kind == "longrun":
        manifest = _read_json(folder / "longrun-manifest.json")
        if not (folder / "shots.json").is_file():
            raise ValueError(f"{folder}: long run has no shots.json yet (not finished)")
        run_id, recorded_sha = manifest["fingerprint"], manifest["input"]["sha256"]
        basis = "longrun-manifest.json input.sha256"
        start, frames = manifest["selection"]["source_start_frame"], manifest["selection"]["frames"]
        fps = float(manifest["input"]["fps"])
        offset_path = folder / "source-offset.json"
        if offset_path.is_file() and _read_json(offset_path)["source_start_frame"] != start:
            raise ValueError(f"{folder}: source-offset.json disagrees with longrun-manifest.json")
        shots = _read_json(folder / "shots.json")
        for c in shots.get("contacts", []):
            hits.append(_candidate(c.get("id") or f"contact@{c['frame']}", c["frame"], start,
                                   player_id=c.get("player_id"), shot_type=None, shot_type_status=None))
        for b in shots.get("bounces", []):
            bounces.append(_candidate(b.get("id") or f"bounce@{b['frame']}", b["frame"], start))
    else:
        if kind == "replay":
            report = _read_json(folder / "replay-report.json")
            run_id = report["review_run_id"]
            events_file = _read_json(folder / "reviewed-events.json")
            source_run = report.get("input_run")
        else:
            report = _read_json(folder / "review-report.json")
            run_id = report["run_id"]
            events_file = _read_json(folder / "event-candidates.json")
            source_run = report.get("source_run")
        if events_file.get("run_id") != run_id:
            raise ValueError(f"{folder}: candidate file run_id does not match its report")
        start, frames, fps = int(report["source_start_frame"]), int(report["frames"]), float(report["fps"])
        found = _source_run_manifest(folder, source_run, start, frames, fps)
        if found:
            recorded_sha = found[1]["input"]["sha256"]
            basis = f"longrun-manifest.json of source run {found[0].name} (same source frames)"
        types = {}
        if kind == "replay" and (folder / "shot-candidates.json").is_file():
            types = {s["event_id"]: s for s in _read_json(folder / "shot-candidates.json").get("shots", [])}
        for e in events_file["events"]:
            if str(e.get("id", "")).startswith("manual-"):
                excluded["human_added_manual_events"] += 1
                continue
            event_type = {"hit_candidate": "hit", "bounce_candidate": "bounce"}.get(e.get("type"), e.get("type"))
            if event_type not in ("hit", "bounce"):
                excluded["other_event_types"] += 1
                continue
            if "review_window_only" in (e.get("support"), e.get("contact_support")) and not include_review_windows:
                excluded["review_windows"] += 1
                continue
            # Score the frame the pipeline produced, not a reviewer's correction.
            run_frame = e.get("original_frame", e["frame"])
            if kind == "review" and "source_frame" in e and e["source_frame"] != e["frame"] + start:
                raise ValueError(f"{e['id']}: source_frame {e['source_frame']} != frame {e['frame']} + "
                                 f"source_start_frame {start}; alignment is ambiguous")
            if event_type == "hit":
                shot = types.get(e["id"], {})
                hits.append(_candidate(e["id"], run_frame, start, player_id=e.get("player_id"),
                                       review_status=e.get("status"), support=e.get("contact_support"),
                                       action_state=shot.get("action_state"), shot_type=shot.get("classification"),
                                       shot_type_status=shot.get("classification_status")))
            else:
                bounces.append(_candidate(e["id"], run_frame, start, review_status=e.get("status")))

    if video is not None:
        hashed = file_sha256(Path(video))
        if recorded_sha and hashed != recorded_sha:
            raise ValueError(f"--video {video} has SHA-256 {hashed[:16]}..., but the run records {recorded_sha[:16]}...")
        if not recorded_sha:
            recorded_sha, basis = hashed, f"SHA-256 of --video {Path(video).name} (the run records no hash)"
    if not recorded_sha:
        raise ValueError(f"{folder}: this run does not record its source video's SHA-256; "
                         "pass --video <the original source video> so it can be hashed locally")
    return {"kind": kind, "path": str(folder), "run_id": run_id, "source_video_sha256": recorded_sha,
            "video_hash_basis": basis, "source_start_frame": start, "frames": frames, "fps": fps,
            "hits": hits, "bounces": bounces, "excluded_candidates": excluded,
            "include_review_windows": include_review_windows}


# ------------------------------------------------------------------ matching

def match_frames(labels: list[int], candidates: list[int], tolerance: int) -> list[tuple[int, int]]:
    """One-to-one pairs (label index, candidate index) within ``tolerance`` frames, closest first."""
    pairs = sorted((abs(c - l), l, c, i, j) for i, l in enumerate(labels) for j, c in enumerate(candidates)
                   if abs(c - l) <= tolerance)
    used_l, used_c, out = set(), set(), []
    for _, _, _, i, j in pairs:
        if i not in used_l and j not in used_c:
            used_l.add(i), used_c.add(j), out.append((i, j))
    return sorted(out)


def _spans(labels: dict[str, Any], kind: str, run: dict[str, Any]) -> list[list[int]]:
    first, last = run["source_start_frame"], run["source_start_frame"] + run["frames"] - 1
    spans = []
    for span in labels["coverage"]:
        if kind in span["kinds"]:
            a, b = max(span["source_start_frame"], first), min(span["source_end_frame"], last)
            if a <= b:
                spans.append([a, b])
    return sorted(spans)


def _inside(frame: int, spans: list[list[int]]) -> bool:
    return any(a <= frame <= b for a, b in spans)


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _fmt(value: float | None, num: int, den: int, what: str) -> str:
    return f"{value:.3f} ({num}/{den})" if value is not None else f"not computed (0 {what})"


def _span_text(spans: list[list[int]]) -> str:
    return ", ".join(f"{a}-{b}" for a, b in spans)


def _window(run: dict[str, Any], frame: int) -> bool:
    return run["source_start_frame"] <= frame < run["source_start_frame"] + run["frames"]


def _score_events(kind: str, entries: list[dict], candidates: list[dict], labels: dict, run: dict,
                  tolerance: int, source: str) -> dict[str, Any]:
    noun, what = ("shots", "Hit candidates") if kind == "shots" else ("bounces", "Bounce candidates")
    spans = _spans(labels, kind, run)
    scored = [e for e in entries if _inside(e["source_frame"], spans)]
    outside_run = sum(not _window(run, e["source_frame"]) for e in entries)
    result: dict[str, Any] = {
        "scored_spans": spans, "labels_in_scored_spans": len(scored),
        "labels_outside_coverage": len(entries) - len(scored) - outside_run,
        "labels_outside_run_frames": outside_run, "candidates_total": len(candidates)}
    if not spans:
        result.update(status="not_scored", precision=None, recall=None,
                      statement=f"{what}: not scored; {source} has no span fully labelled for {noun} "
                                "inside this run's frames.")
        return result
    pairs = match_frames([e["source_frame"] for e in scored], [c["source_frame"] for c in candidates], tolerance)
    matched_c = {j for _, j in pairs}
    matched_l = {i for i, _ in pairs}
    false_pos = [c for j, c in enumerate(candidates) if j not in matched_c and _inside(c["source_frame"], spans)]
    outside = len(candidates) - len(pairs) - len(false_pos)
    tp, n = len(pairs), len(scored)
    precision, recall = _ratio(tp, tp + len(false_pos)), _ratio(tp, n)
    offsets = [candidates[j]["source_frame"] - scored[i]["source_frame"] for i, j in pairs]
    matches = [{"label": scored[i], "candidate": candidates[j],
                "offset_frames": candidates[j]["source_frame"] - scored[i]["source_frame"]} for i, j in pairs]
    result.update(
        status="scored", matched=tp, missed=n - tp, false_positives=len(false_pos),
        candidates_in_scored_spans=tp + len(false_pos), candidates_outside_scored_spans=outside,
        precision=precision, recall=recall,
        timing={"mean_abs_offset_frames": round(sum(map(abs, offsets)) / tp, 3) if tp else None,
                "max_abs_offset_frames": max(map(abs, offsets)) if tp else None,
                "note": "candidate source frame minus labelled source frame, within the tolerance only"},
        statement=(f"{what}: accuracy on {n} labelled {noun} from {source} (source frames {_span_text(spans)}; "
                   f"tolerance ±{tolerance} frames): precision {_fmt(precision, tp, tp + len(false_pos), 'candidates in the labelled span')}, "
                   f"recall {_fmt(recall, tp, n, 'labelled ' + noun)}."),
        details={"matches": matches, "missed": [e for i, e in enumerate(scored) if i not in matched_l],
                 "false_positives": false_pos})
    return result


def _hitter_agreement(matches: list[dict], source: str) -> dict[str, Any]:
    compared = [m for m in matches if m["candidate"].get("player_id") in SIDES]
    agree = sum(m["candidate"]["player_id"] == m["label"]["hitter"] for m in compared)
    abstained = len(matches) - len(compared)
    return {"compared": len(compared), "agree": agree, "agreement": _ratio(agree, len(compared)),
            "pipeline_gave_no_hitter": abstained,
            "statement": (f"Hitter agreement on {len(compared)} matched labelled shots from {source} where the "
                          f"pipeline names a hitter: {agree}/{len(compared)}; {abstained} matched without a hitter.")}


def _shot_type_agreement(matches: list[dict], source: str) -> dict[str, Any]:
    # Exclusive reasons, checked in this order; compared + sum(not_compared) = matched.
    not_compared = {"pipeline_type_set_by_human_review": 0, "pipeline_has_no_type": 0, "label_unsure": 0,
                    "pipeline_unknown": 0, "pipeline_not_a_shot": 0}
    confusion: dict[str, dict[str, int]] = {}
    agree = compared = 0
    for m in matches:
        label_type, cand = m["label"]["shot_type"], m["candidate"]
        cand_type = cand.get("shot_type")
        if cand.get("shot_type_status") == "human_reviewed":
            not_compared["pipeline_type_set_by_human_review"] += 1
        elif cand_type is None:
            not_compared["pipeline_has_no_type"] += 1
        elif label_type in LABEL_ABSTAIN_SHOT_TYPES:
            not_compared["label_unsure"] += 1
        elif cand_type == "unknown":
            not_compared["pipeline_unknown"] += 1
        elif cand_type == "not_a_shot":
            not_compared["pipeline_not_a_shot"] += 1
        else:
            compared += 1
            agree += cand_type == label_type
            row = confusion.setdefault(label_type, {})
            row[cand_type] = row.get(cand_type, 0) + 1
    skipped = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in not_compared.items() if v) or "none"
    return {"compared": compared, "agree": agree, "agreement": _ratio(agree, compared),
            "not_compared": not_compared, "confusion_label_to_pipeline": confusion,
            "statement": (f"Shot-type agreement on {compared} labelled shots from {source} where both the label "
                          f"and the pipeline name a type: {agree}/{compared}. Not compared: {skipped}.")}


def _bounce_calls(result: dict[str, Any]) -> None:
    if result.get("status") != "scored":
        return
    count = lambda entries: {c: sum(e["call"] == c for e in entries) for c in ("in", "out", "unsure")}
    result["label_calls"] = {"matched": count([m["label"] for m in result["details"]["matches"]]),
                             "missed": count(result["details"]["missed"]),
                             "note": "John's in/out calls, for context; the pipeline makes no in/out call."}


# -------------------------------------------------------------------- points

def load_point_proposals(path: Path | str, run: dict[str, Any]) -> list[dict[str, Any]]:
    """Generic point proposals: ``{start_frame, end_frame}`` (run-relative) or
    ``{source_start_frame, source_end_frame}``; optional ``server``/``winner``/``id``."""
    data = _read_json(Path(path))
    if isinstance(data, dict):
        if data.get("run_id") not in (None, run["run_id"]):
            raise ValueError(f"{path}: point proposals belong to another run")
        data = data.get("points")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of points or {{'points': [...]}}")
    start, out = run["source_start_frame"], []
    for i, p in enumerate(data):
        if "source_start_frame" in p and "source_end_frame" in p:
            a, b = p["source_start_frame"], p["source_end_frame"]
        elif "start_frame" in p and "end_frame" in p:
            a, b = p["start_frame"] + start, p["end_frame"] + start
        else:
            raise ValueError(f"{path}: point {i} needs start_frame/end_frame (run) or "
                             "source_start_frame/source_end_frame (source video)")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (a, b)) or b < a:
            raise ValueError(f"{path}: point {i} has invalid frames")
        out.append({"id": p.get("id", f"proposal-{i}"), "source_start_frame": a, "source_end_frame": b,
                    "run_start_frame": a - start, "run_end_frame": b - start,
                    "server": p.get("server"), "winner": p.get("winner")})
    return out


def _score_points(entries: list[dict], proposals: list[dict] | None, labels: dict, run: dict,
                  tolerance_s: float, source: str) -> dict[str, Any]:
    spans = _spans(labels, "points", run)
    scored = [e for e in entries if any(a <= e["source_start_frame"] and e["source_end_frame"] <= b for a, b in spans)]
    tolerance = int(round(tolerance_s * run["fps"]))
    result: dict[str, Any] = {"scored_spans": spans, "labels_in_scored_spans": len(scored),
                              "labels_outside_scored_spans": len(entries) - len(scored),
                              "tolerance_s": tolerance_s, "tolerance_frames": tolerance}
    if proposals is None:
        result.update(status="not_scored", precision=None, recall=None,
                      statement="Points: not scored; no --points proposals were given.")
        return result
    result["proposals_total"] = len(proposals)
    if not spans:
        result.update(status="not_scored", precision=None, recall=None,
                      statement=f"Points: not scored; {source} has no span fully labelled for points "
                                "inside this run's frames.")
        return result
    pairs = sorted((abs(p["source_start_frame"] - e["source_start_frame"]) + abs(p["source_end_frame"] - e["source_end_frame"]),
                    i, j) for i, e in enumerate(scored) for j, p in enumerate(proposals)
                   if abs(p["source_start_frame"] - e["source_start_frame"]) <= tolerance
                   and abs(p["source_end_frame"] - e["source_end_frame"]) <= tolerance)
    used_l, used_p, matched = set(), set(), []
    for _, i, j in pairs:
        if i not in used_l and j not in used_p:
            used_l.add(i), used_p.add(j), matched.append((i, j))
    matched.sort()
    false_pos = [p for j, p in enumerate(proposals) if j not in used_p and _inside(p["source_start_frame"], spans)]
    tp, n = len(matched), len(scored)
    precision, recall = _ratio(tp, tp + len(false_pos)), _ratio(tp, n)
    agreement = {}
    for field in ("server", "winner"):
        both = [(scored[i][field], proposals[j][field]) for i, j in matched
                if scored[i][field] in SIDES and proposals[j][field] in SIDES]
        agreement[field] = {"compared": len(both), "agree": sum(a == b for a, b in both),
                            "not_compared_label_or_proposal_unknown": tp - len(both)}
    result.update(
        status="scored", matched=tp, missed=n - tp, false_positives=len(false_pos),
        proposals_outside_scored_spans=len(proposals) - tp - len(false_pos),
        precision=precision, recall=recall, agreement=agreement,
        statement=(f"Point proposals: accuracy on {n} labelled points from {source} (source frames "
                   f"{_span_text(spans)}; start and end within ±{tolerance_s:g} s = ±{tolerance} frames): "
                   f"precision {_fmt(precision, tp, tp + len(false_pos), 'proposals in the labelled span')}, "
                   f"recall {_fmt(recall, tp, n, 'labelled points')}."),
        details={"matches": [{"label": scored[i], "proposal": proposals[j],
                              "start_offset_frames": proposals[j]["source_start_frame"] - scored[i]["source_start_frame"],
                              "end_offset_frames": proposals[j]["source_end_frame"] - scored[i]["source_end_frame"]}
                             for i, j in matched],
                 "missed": [e for i, e in enumerate(scored) if i not in used_l], "false_positives": false_pos})
    return result


# ------------------------------------------------------------------- scoring

def score(labels_path: Path | str, run_path: Path | str, *, tolerance_frames: int = DEFAULT_TOLERANCE_FRAMES,
          points_path: Path | str | None = None, point_tolerance_s: float = DEFAULT_POINT_TOLERANCE_S,
          video: Path | str | None = None, include_review_windows: bool = False) -> dict[str, Any]:
    if isinstance(tolerance_frames, bool) or not isinstance(tolerance_frames, int) or tolerance_frames < 0:
        raise ValueError("tolerance_frames must be a whole number >= 0")
    if not point_tolerance_s >= 0:
        raise ValueError("point tolerance must be >= 0 seconds")
    labels, labels_sha = load_labels(labels_path)
    run = load_run(run_path, video=video, include_review_windows=include_review_windows)
    check_binding(labels, run["run_id"], run["source_video_sha256"], run["fps"])
    source = Path(labels_path).name
    hits = _score_events("shots", labels["shots"], run["hits"], labels, run, tolerance_frames, source)
    if hits["status"] == "scored":
        hits["hitter_agreement"] = _hitter_agreement(hits["details"]["matches"], source)
        hits["shot_type_agreement"] = _shot_type_agreement(hits["details"]["matches"], source)
    bounces = _score_events("bounces", labels["bounces"], run["bounces"], labels, run, tolerance_frames, source)
    _bounce_calls(bounces)
    proposals = load_point_proposals(points_path, run) if points_path else None
    points = _score_points(labels["points"], proposals, labels, run, point_tolerance_s, source)
    summary = [hits["statement"]]
    if hits["status"] == "scored":
        summary += [hits["hitter_agreement"]["statement"], hits["shot_type_agreement"]["statement"]]
    summary += [bounces["statement"], points["statement"],
                f"Nothing is claimed outside the labelled spans of {source}."]
    first = run["source_start_frame"]
    return {
        "schema_version": 1, "kind": "label_score",
        "labels": {"file": source, "path": str(labels_path), "sha256": labels_sha,
                   "labeller": labels["labeller"], "binding": labels["binding"]},
        "run": {k: run[k] for k in ("kind", "path", "run_id", "source_video_sha256", "video_hash_basis",
                                    "source_start_frame", "frames", "fps", "excluded_candidates",
                                    "include_review_windows")},
        "frame_basis": {"basis": "source_video_frames",
                        "run_source_frames": [first, first + run["frames"] - 1],
                        "conversion": f"source_frame = run_frame + {first}"},
        "settings": {"tolerance_frames": tolerance_frames, "point_tolerance_s": point_tolerance_s,
                     "points_file": str(points_path) if points_path else None},
        "hits": hits, "bounces": bounces, "points": points,
        "summary": summary, "limitations": LIMITATIONS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score hit/bounce candidates and point proposals against "
                                                 "human ground-truth labels (labels are only read).")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="Replay, event-review or long-run folder")
    parser.add_argument("--output", required=True, type=Path, help="New JSON file to write")
    parser.add_argument("--tolerance-frames", type=int, default=DEFAULT_TOLERANCE_FRAMES)
    parser.add_argument("--points", type=Path, help="Point proposals JSON (list of {start_frame, end_frame})")
    parser.add_argument("--point-tolerance-s", type=float, default=DEFAULT_POINT_TOLERANCE_S)
    parser.add_argument("--video", type=Path, help="Original source video, hashed locally when the run "
                                                   "records no SHA-256")
    parser.add_argument("--include-review-windows", action="store_true",
                        help="Also treat 'review_window_only' swing windows as hit candidates")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == args.labels.resolve():
            raise ValueError("--output must not be the labels file; labels are never modified")
        if args.output.exists():
            raise ValueError(f"{args.output} already exists; choose a new file so earlier results are kept")
        result = score(args.labels, args.run, tolerance_frames=args.tolerance_frames, points_path=args.points,
                       point_tolerance_s=args.point_tolerance_s, video=args.video,
                       include_review_windows=args.include_review_windows)
    except KeyError as error:
        print(f"ERROR: a run or proposal file is missing the field {error}", file=sys.stderr)
        return 1
    except (LabelsError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("\n".join(result["summary"]))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
