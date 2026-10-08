"""Ground-truth labels (``labels.json``): schema v1 validator and run binding.

A labels file holds John's own decisions about shots, bounces and points. It is
bound to one run id and one source video (SHA-256). Frame numbers are always
SOURCE-VIDEO frames (0-based in the hashed file); run-relative frames are
``source_frame - source_start_frame``. See ``docs/labels-schema.md``.

Nothing in this module (or any tool) writes a labels file: it is only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KIND = "tennis_ground_truth_labels"

# Default list pending John's confirmation (issue #27). "unsure" is the label's
# abstention: John saw the hit but will not name the type. No spin/speed labels.
SHOT_TYPES = ("serve", "forehand", "backhand", "volley", "overhead", "other", "unsure")
LABEL_ABSTAIN_SHOT_TYPES = frozenset({"unsure"})
HITTERS = ("near", "far")
BOUNCE_CALLS = ("in", "out", "unsure")
POINT_SIDES = ("near", "far", "unknown")
COVERAGE_KINDS = ("shots", "bounces", "points")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_TOP_REQUIRED = {"schema_version", "kind", "binding", "labeller", "shot_types", "coverage", "shots", "bounces",
                 "points"}
_TOP_OPTIONAL = {"notes"}
_BINDING_REQUIRED = {"run_id", "source_video_sha256", "fps"}
_BINDING_OPTIONAL = {"source_video_name"}
_ENTRY_FIELDS = {
    "shots": ({"id", "source_frame", "hitter", "shot_type", "decision"}, {"notes"}),
    "bounces": ({"id", "source_frame", "call", "decision"}, {"notes"}),
    "points": ({"id", "source_start_frame", "source_end_frame", "server", "winner", "decision"},
               {"score_text", "notes"}),
}


class LabelsError(ValueError):
    """The labels file is malformed or belongs to a different run/video."""


def _fields(obj: Any, where: str, required: set[str], optional: set[str]) -> None:
    if not isinstance(obj, dict):
        raise LabelsError(f"{where}: expected an object")
    missing = sorted(required - obj.keys())
    if missing:
        raise LabelsError(f"{where}: missing field(s) {', '.join(missing)}")
    unknown = sorted(obj.keys() - required - optional)
    if unknown:
        raise LabelsError(f"{where}: unknown field(s) {', '.join(unknown)} (typo? see docs/labels-schema.md)")


def _frame(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LabelsError(f"{where}: must be a whole source-video frame number >= 0")
    return value


def _text(value: Any, where: str, *, empty_ok: bool = True) -> None:
    if not isinstance(value, str) or (not empty_ok and not value.strip()):
        raise LabelsError(f"{where}: must be {'text' if empty_ok else 'non-empty text'}")


def _choice(value: Any, allowed: tuple[str, ...], where: str) -> None:
    if value not in allowed:
        raise LabelsError(f"{where}: {value!r} is not one of {', '.join(allowed)}")


def validate_labels(data: Any) -> dict[str, Any]:
    """Check a parsed labels file against schema v1. Returns ``data`` unchanged.

    Raises LabelsError naming the first problem. Never modifies its input.
    """
    _fields(data, "labels", _TOP_REQUIRED, _TOP_OPTIONAL)
    if data["schema_version"] != SCHEMA_VERSION:
        raise LabelsError(f"labels: schema_version must be {SCHEMA_VERSION}")
    if data["kind"] != KIND:
        raise LabelsError(f"labels: kind must be {KIND!r}")
    if "notes" in data:
        _text(data["notes"], "labels.notes")

    binding = data["binding"]
    _fields(binding, "binding", _BINDING_REQUIRED, _BINDING_OPTIONAL)
    for key in ("run_id", "source_video_sha256"):
        if not isinstance(binding[key], str) or not _HEX64.match(binding[key]):
            raise LabelsError(f"binding.{key}: must be 64 lowercase hex characters")
    fps = binding["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0 < fps < 1000:
        raise LabelsError("binding.fps: must be a positive number")
    if "source_video_name" in binding:
        _text(binding["source_video_name"], "binding.source_video_name")

    labeller = data["labeller"]
    _fields(labeller, "labeller", {"name", "date"}, set())
    _text(labeller["name"], "labeller.name", empty_ok=False)
    if not isinstance(labeller["date"], str) or not _DATE.match(labeller["date"]):
        raise LabelsError("labeller.date: must be YYYY-MM-DD")
    try:
        date.fromisoformat(labeller["date"])
    except ValueError as error:
        raise LabelsError(f"labeller.date: {error}") from None

    if data["shot_types"] != list(SHOT_TYPES):
        raise LabelsError(f"shot_types: must be exactly {list(SHOT_TYPES)} (the current list); "
                          "a file made with another list needs John to re-check its shot types")

    if not isinstance(data["coverage"], list):
        raise LabelsError("coverage: expected a list")
    spans_by_kind: dict[str, list[tuple[int, int]]] = {k: [] for k in COVERAGE_KINDS}
    for i, span in enumerate(data["coverage"]):
        where = f"coverage[{i}]"
        _fields(span, where, {"source_start_frame", "source_end_frame", "kinds"}, set())
        a = _frame(span["source_start_frame"], f"{where}.source_start_frame")
        b = _frame(span["source_end_frame"], f"{where}.source_end_frame")
        if b < a:
            raise LabelsError(f"{where}: source_end_frame is before source_start_frame")
        kinds = span["kinds"]
        if not isinstance(kinds, list) or not kinds or len(set(kinds)) != len(kinds):
            raise LabelsError(f"{where}.kinds: expected a non-empty list without repeats")
        for kind in kinds:
            _choice(kind, COVERAGE_KINDS, f"{where}.kinds")
            for c, d in spans_by_kind[kind]:
                if a <= d and c <= b:
                    raise LabelsError(f"{where}: overlaps another {kind} coverage span")
            spans_by_kind[kind].append((a, b))

    ids: set[str] = set()
    for kind, (required, optional) in _ENTRY_FIELDS.items():
        entries = data[kind]
        if not isinstance(entries, list):
            raise LabelsError(f"{kind}: expected a list")
        frames: set[int] = set()
        intervals: list[tuple[int, int, str]] = []
        for i, entry in enumerate(entries):
            where = f"{kind}[{i}]"
            _fields(entry, where, required, optional)
            _text(entry["id"], f"{where}.id", empty_ok=False)
            if entry["id"] in ids:
                raise LabelsError(f"{where}: duplicate id {entry['id']!r}")
            ids.add(entry["id"])
            if entry["decision"] != "human":
                raise LabelsError(f"{where}.decision: must be 'human'; labels are John's decisions only")
            for optional_text in ("notes", "score_text"):
                if optional_text in entry:
                    _text(entry[optional_text], f"{where}.{optional_text}")
            if kind == "points":
                a = _frame(entry["source_start_frame"], f"{where}.source_start_frame")
                b = _frame(entry["source_end_frame"], f"{where}.source_end_frame")
                if b < a:
                    raise LabelsError(f"{where}: source_end_frame is before source_start_frame")
                _choice(entry["server"], POINT_SIDES, f"{where}.server")
                _choice(entry["winner"], POINT_SIDES, f"{where}.winner")
                for c, d, other in intervals:
                    if a <= d and c <= b:
                        raise LabelsError(f"{where}: overlaps point {other!r}")
                intervals.append((a, b, entry["id"]))
                continue
            f = _frame(entry["source_frame"], f"{where}.source_frame")
            if f in frames:
                raise LabelsError(f"{where}: another {kind[:-1]} is already labelled at source frame {f}")
            frames.add(f)
            if kind == "shots":
                _choice(entry["hitter"], HITTERS, f"{where}.hitter")
                _choice(entry["shot_type"], SHOT_TYPES, f"{where}.shot_type")
            else:
                _choice(entry["call"], BOUNCE_CALLS, f"{where}.call")
    return data


def load_labels(path: Path | str) -> tuple[dict[str, Any], str]:
    """Read and validate a labels file. Returns (labels, sha256 of the file bytes)."""
    raw = Path(path).read_bytes()
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LabelsError(f"{path}: not valid UTF-8 JSON ({error})") from None
    return validate_labels(data), hashlib.sha256(raw).hexdigest()


def check_binding(labels: dict[str, Any], run_id: str, source_video_sha256: str, fps: float) -> None:
    """Reject labels that belong to another run or another source video."""
    binding = labels["binding"]
    if binding["run_id"] != run_id:
        raise LabelsError(f"Labels are bound to run {binding['run_id'][:16]}..., "
                          f"but this run is {run_id[:16]}...; they cannot be scored against it")
    if binding["source_video_sha256"] != source_video_sha256:
        raise LabelsError(f"Labels are bound to source video {binding['source_video_sha256'][:16]}..., "
                          f"but this run was made from {source_video_sha256[:16]}...")
    if abs(float(binding["fps"]) - float(fps)) > 0.01:
        raise LabelsError(f"Labels say {binding['fps']} fps but the run is {fps} fps")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a ground-truth labels.json (read-only).")
    parser.add_argument("labels", type=Path)
    parser.add_argument("--run", type=Path, help="Also check the binding against this run/replay/review folder")
    parser.add_argument("--video", type=Path, help="Source video to hash if the run did not record its SHA-256")
    args = parser.parse_args(argv)
    try:
        labels, digest = load_labels(args.labels)
        message = (f"{args.labels.name}: valid schema v{SCHEMA_VERSION} labels by {labels['labeller']['name']} "
                   f"({labels['labeller']['date']}): {len(labels['shots'])} shots, {len(labels['bounces'])} bounces, "
                   f"{len(labels['points'])} points, {len(labels['coverage'])} coverage span(s); sha256 {digest[:16]}...")
        if args.run:
            from .score_labels import load_run
            run = load_run(args.run, video=args.video)
            check_binding(labels, run["run_id"], run["source_video_sha256"], run["fps"])
            message += f"; binding matches run {run['run_id'][:16]}... ({run['video_hash_basis']})"
    except (LabelsError, OSError, ValueError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
