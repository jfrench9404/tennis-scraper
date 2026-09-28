"""Proposed break / changeover intervals from a run's rows, for John to confirm.

Reads a finished run (``events.jsonl``) or a PARTIAL long run
(``chunks/*/events.jsonl``, only the leading chunks ``longrun`` marks complete)
and proposes intervals where, for a long stretch:

* the near side has no observed ``near`` player  (default 20 s or more),
* the far side has no observed ``far`` player    (default 20 s or more),
* no ball is observed                            (default 15 s or more).

Only observed tracks count (``predicted`` tracks are estimates and are ignored).
Brief observations shorter than ``--bridge-seconds`` do not split an interval,
so a single spurious detection does not hide a break. Frames the run did not
analyse are unknown, never "empty": intervals stop at them and are flagged.

Everything here is a PROPOSAL ("possible changeover or break"). It does not
label games, points, scores, serves, aces or faults, and nothing is written back
into the run: the output must live outside the run folder. An empty side proves
only that the detector did not observe a player there, not why.
"""
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
DEFAULTS = {"min_side_empty_seconds": 20.0, "min_no_ball_seconds": 15.0, "bridge_seconds": 1.0}
SIGNALS = {
    "near_side_empty": ("near", "min_side_empty_seconds", "near side: no observed near player"),
    "far_side_empty": ("far", "min_side_empty_seconds", "far side: no observed far player"),
    "no_ball_observed": ("ball", "min_no_ball_seconds", "no ball observed"),
}
MEANING = ("Proposals only, for a human to confirm or reject. Derived from detector coverage "
           "(observed near/far player tracks and observed ball tracks); predicted tracks are ignored. "
           "An empty side or missing ball means the detector did not observe one, not why. "
           "No game, point, score, serve, ace or fault is inferred, and nothing is applied to the run.")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _fps(run: Path, manifest: dict[str, Any] | None) -> float:
    if manifest and manifest.get("input", {}).get("fps"):
        return float(manifest["input"]["fps"])
    summary = run / "summary.json"
    if summary.is_file():
        fps = json.loads(summary.read_text(encoding="utf-8")).get("fps")
        if fps:
            return float(fps)
    raise ValueError(f"{run}: no fps in longrun-manifest.json or summary.json")


def load_run(run: Path, source: str = "auto") -> dict[str, Any]:
    """Rows of a run keyed by SOURCE frame, plus where they came from.

    ``source``: ``merged`` reads ``events.jsonl``; ``chunks`` reads the leading
    complete chunks of a long run (checked with ``longrun.completed_chunks``:
    fingerprint and file hashes); ``auto`` prefers ``events.jsonl`` and falls back
    to chunks when a long run has not been merged yet.
    """
    run = Path(run)
    manifest_path = run / "longrun-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    merged = run / "events.jsonl"
    if source == "auto":
        source = "merged" if merged.is_file() else "chunks"
    info: dict[str, Any] = {"run": str(run), "source": source, "fps": _fps(run, manifest)}
    frames: dict[int, dict[str, Any]] = {}
    if source == "merged":
        if not merged.is_file():
            raise FileNotFoundError(f"{merged} does not exist")
        offset_path = run / "source-offset.json"
        offset = (json.loads(offset_path.read_text(encoding="utf-8")).get("source_start_frame", 0)
                  if offset_path.is_file() else 0)
        for row in _read_jsonl(merged):
            frames[int(row.get("source_frame", offset + row["frame"]))] = row
        info.update(partial=False, selection_frames=manifest["selection"]["frames"] if manifest else len(frames))
    elif source == "chunks":
        if manifest is None:
            raise FileNotFoundError(f"{manifest_path} does not exist; chunks can only be read from a long run")
        from .longrun import chunk_dir, completed_chunks
        done = completed_chunks(run, manifest)
        for status in done:
            folder = chunk_dir(run, manifest["chunks"][status["index"]])
            for row in _read_jsonl(folder / "events.jsonl"):
                frames[int(row["source_frame"])] = row
        info.update(partial=len(done) < len(manifest["chunks"]),
                    chunks={"complete": len(done), "planned": len(manifest["chunks"])},
                    selection_frames=manifest["selection"]["frames"])
    else:
        raise ValueError(f"Unknown source {source!r}; use auto, merged or chunks")
    if not frames:
        detail = f", 0/{len(manifest['chunks'])} chunks complete" if source == "chunks" else ""
        raise ValueError(f"{run}: no rows to read ({source}{detail})")
    info["rows"] = frames
    return info


def observe(row: dict[str, Any]) -> dict[str, bool]:
    """Per-frame evidence. Only observed (not predicted) tracks count."""
    seen = {"near": False, "far": False, "ball": False}
    for t in row.get("tracks", []):
        if t.get("predicted"):
            continue
        if t.get("label") == "player" and t.get("identity_id") in ("near", "far"):
            seen[t["identity_id"]] = True
        elif t.get("label") == "ball":
            seen["ball"] = True
    return seen


def absent_runs(values: list[bool | None], min_frames: int, bridge_frames: int) -> list[dict[str, Any]]:
    """Runs where ``values`` is False, as [start, end] inclusive indices.

    A stray observation (a True spell shorter than ``bridge_frames`` with at least
    ``bridge_frames`` of False on BOTH sides) does not split a run. Frequent short
    observations (e.g. a ball seen every few frames) are not stray and do split.
    ``None`` (not analysed) always ends a run. Runs shorter than ``min_frames``
    are dropped. Each run records whether it touches unknown or the edge of the
    data (so the real interval may be longer).
    """
    raw, start = [], None
    for i, v in enumerate(values + [None]):
        if v is False and start is None:
            start = i
        elif v is not False and start is not None:
            raw.append([start, i - 1])
            start = None
    merged: list[list[int]] = []
    previous: list[int] | None = None
    for s, e in raw:
        stray = (previous is not None and s - previous[1] - 1 < bridge_frames
                 and previous[1] - previous[0] + 1 >= bridge_frames and e - s + 1 >= bridge_frames
                 and all(values[k] is True for k in range(previous[1] + 1, s)))
        previous = [s, e]
        if stray:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if e - s + 1 >= min_frames:
            out.append({"start": s, "end": e,
                        "open_start": s == 0 or values[s - 1] is None,
                        "open_end": e == len(values) - 1 or values[e + 1] is None})
    return out


def _evidence(seen: list[dict[str, bool] | None], s: int, e: int) -> dict[str, int]:
    part = seen[s:e + 1]
    known = [x for x in part if x is not None]
    return {"frames": len(part), "analysed_frames": len(known),
            "near_observed_frames": sum(x["near"] for x in known),
            "far_observed_frames": sum(x["far"] for x in known),
            "ball_observed_frames": sum(x["ball"] for x in known)}


def propose(loaded: dict[str, Any], min_side_empty_seconds: float = DEFAULTS["min_side_empty_seconds"],
            min_no_ball_seconds: float = DEFAULTS["min_no_ball_seconds"],
            bridge_seconds: float = DEFAULTS["bridge_seconds"]) -> dict[str, Any]:
    fps, rows = loaded["fps"], loaded["rows"]
    thresholds = {"min_side_empty_seconds": min_side_empty_seconds, "min_no_ball_seconds": min_no_ball_seconds,
                  "bridge_seconds": bridge_seconds}
    for key, value in thresholds.items():
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
            raise ValueError(f"{key} must be a finite number >= 0")
    first, last = min(rows), max(rows)
    seen = [observe(rows[f]) if f in rows else None for f in range(first, last + 1)]
    sec = lambda i: round((first + i) / fps, 3)
    interval = lambda s, e: {"source_frames": [first + s, first + e], "source_seconds": [sec(s), sec(e + 1)],
                             "duration_s": round((e - s + 1) / fps, 3)}

    signals: dict[str, list[dict[str, Any]]] = {}
    bridge = int(round(bridge_seconds * fps))
    for name, (key, threshold, text) in SIGNALS.items():
        values = [None if x is None else x[key] for x in seen]
        min_frames = max(1, int(math.ceil(thresholds[threshold] * fps)))
        signals[name] = [{**interval(r["start"], r["end"]), "description": text,
                          "open_start": r["open_start"], "open_end": r["open_end"], "_span": (r["start"], r["end"])}
                         for r in absent_runs(values, min_frames, bridge)]

    # Overlapping or touching signal intervals become one proposal listing every reason.
    spans = sorted((sig["_span"][0], sig["_span"][1], name) for name, items in signals.items() for sig in items)
    groups: list[list[Any]] = []
    for s, e, name in spans:
        if groups and s <= groups[-1][1] + 1:
            groups[-1][1] = max(groups[-1][1], e)
            groups[-1][2].append(name)
        else:
            groups.append([s, e, [name]])
    proposals = []
    for n, (s, e, names) in enumerate(groups, 1):
        reasons = [n_ for n_ in SIGNALS if n_ in names]
        open_start = any(sig["open_start"] for r in reasons for sig in signals[r] if sig["_span"][0] == s)
        open_end = any(sig["open_end"] for r in reasons for sig in signals[r] if sig["_span"][1] == e)
        proposals.append({"id": f"proposal-{n}", "label": "possible changeover or break",
                          "status": "proposal_unreviewed", **interval(s, e), "reasons": reasons,
                          "reason_text": [SIGNALS[r][2] for r in reasons],
                          "open_start": open_start, "open_end": open_end,
                          "evidence": _evidence(seen, s, e)})
    for items in signals.values():
        for sig in items:
            sig.update(_evidence(seen, *sig.pop("_span")))

    per_second: dict[int, dict[str, int]] = {}
    for i, x in enumerate(seen):
        bucket = per_second.setdefault(int((first + i) // fps), {"frames": 0, "analysed": 0, "near": 0, "far": 0,
                                                                  "ball": 0})
        bucket["frames"] += 1
        if x is not None:
            bucket["analysed"] += 1
            for key in ("near", "far", "ball"):
                bucket[key] += x[key]
    timeline = [{"second": s, **v} for s, v in sorted(per_second.items())]

    report = {"schema_version": SCHEMA_VERSION, "kind": "segment_proposals", "run": loaded["run"],
              "source": loaded["source"], "partial": loaded["partial"], "fps": fps,
              "source_frames": [first, last], "source_seconds": [round(first / fps, 3), round((last + 1) / fps, 3)],
              "rows_read": len(rows), "missing_frames_in_span": sum(x is None for x in seen),
              "selection_frames": loaded["selection_frames"], "thresholds": thresholds,
              "meaning": MEANING, "proposals": proposals, "signals": signals, "per_second": timeline}
    if "chunks" in loaded:
        report["chunks"] = loaded["chunks"]
    return report


def write_html(report: dict[str, Any], path: Path) -> None:
    template = Path(__file__).with_name("segments.html").read_text(encoding="utf-8")
    data = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    path.write_text(template.replace("__SEGMENTS_DATA__", data), encoding="utf-8")


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="Run folder (finished, or a partial long run)")
    p.add_argument("--output", type=Path, required=True,
                   help="JSON to write (outside the run folder); a timeline .html is written beside it")
    p.add_argument("--source", choices=("auto", "merged", "chunks"), default="auto",
                   help="auto: events.jsonl if present, else complete chunks/*/events.jsonl")
    p.add_argument("--min-side-empty-seconds", type=float, default=DEFAULTS["min_side_empty_seconds"])
    p.add_argument("--min-no-ball-seconds", type=float, default=DEFAULTS["min_no_ball_seconds"])
    p.add_argument("--bridge-seconds", type=float, default=DEFAULTS["bridge_seconds"],
                   help="Observations shorter than this do not split an empty interval")
    p.add_argument("--no-html", action="store_true")
    args = p.parse_args(argv)
    if _inside(args.output, args.run):
        raise SystemExit("--output must be outside the run folder: proposals are never written into a run")
    try:
        loaded = load_run(args.run, args.source)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error))
    report = propose(loaded, args.min_side_empty_seconds, args.min_no_ball_seconds, args.bridge_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, args.output)
    html = args.output.with_suffix(".html")
    if not args.no_html:
        write_html(report, html)
    partial = f" (partial: {report['chunks']['complete']}/{report['chunks']['planned']} chunks)" \
        if report.get("chunks") and report["partial"] else ""
    print(f"{len(report['proposals'])} proposed interval(s) over source {report['source_seconds'][0]:.1f}-"
          f"{report['source_seconds'][1]:.1f} s{partial} -> {args.output}" + ("" if args.no_html else f", {html}"))


if __name__ == "__main__":
    main()
