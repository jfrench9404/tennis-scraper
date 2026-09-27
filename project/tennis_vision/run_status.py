"""How is a long run doing? Progress, rate, ETA, stalls and wall-clock gaps.

    python -m tennis_vision.run_status runs/claude-game-v1-full
    python -m tennis_vision.run_status runs/claude-game-v1-full --stall-minutes 20 --json

Strictly read-only: it only reads the run folder written by ``longrun.py``
(``longrun-manifest.json``, ``chunks/*/chunk.json``, ``progress.json``,
``longrun-report.json``) and any ``.log`` that belongs to the run. It never writes,
moves or locks a file, so it is safe to use while a run is going.

Works on never-started, partial and complete runs. Newer timing fields
(``started_at``/``finished_at``/``pauses`` in chunk.json, ``updated_at`` in
progress.json, ``pauses`` in longrun-report.json) are used when present; older
runs fall back to file modified times, and the output says which source it used.

"Stalled" means no progress was written for longer than the threshold. A
stopped process, a sleeping laptop and a hung process look the same from the
files, so the tool reports the silence, not its cause.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_STALL_MINUTES = 10.0
LOG_PREFIX_CHARS = 48  # Timestamps are looked for only near the start of a log line.


# --- small read-only helpers -------------------------------------------------

def local(dt: datetime) -> datetime:
    """Aware datetime in local time (naive values are taken as local time)."""
    return dt.astimezone()


def mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).astimezone()


def parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return local(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def read_text(path: Path) -> str:
    """Decode a log as UTF-8 or, for Windows PowerShell 5.1 redirects, UTF-16."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def chunk_dir(output: Path, chunk: dict[str, int]) -> Path:
    # Same naming as longrun.chunk_dir (kept separate so this tool never imports
    # cv2/numpy; tests check they agree).
    return output / "chunks" / f"{chunk['first_frame']:07d}-{chunk['first_frame'] + chunk['frames']:07d}"


def pause_seconds(record: dict[str, Any]) -> float:
    """Total recorded pause time in a chunk/report record, whatever shape it has."""
    total = 0.0
    for pause in record.get("pauses") or []:
        if isinstance(pause, dict):
            seconds = pause.get("seconds")
            if not isinstance(seconds, (int, float)):
                start, end = parse_iso(pause.get("start")), parse_iso(pause.get("end"))
                seconds = (end - start).total_seconds() if start and end else 0
            total += max(float(seconds), 0.0)
    return total


def pause_list(record: dict[str, Any], where: str) -> list[dict[str, Any]]:
    out = []
    for pause in record.get("pauses") or []:
        if isinstance(pause, dict):
            out.append({"source": where, "start": pause.get("start"), "end": pause.get("end"),
                        "minutes": round(pause_seconds({"pauses": [pause]}) / 60, 1)})
    return out


# --- log parsing ---------------------------------------------------------------

ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?\s*(Z|[+-]\d{2}:?\d{2})?")
TIME_RE = re.compile(r"(?<![\d:.])(\d{1,2}):(\d{2})(?::(\d{2})(?:[.,]\d+)?)?(?:\s*([AaPp][Mm])\b)?(?![\d:])")


def line_stamp(line: str) -> tuple[str, Any] | None:
    """('datetime', aware dt) for a dated stamp, ('time', seconds of day) for time only."""
    head = line[:LOG_PREFIX_CHARS]
    match = ISO_RE.search(head)
    if match:
        y, mo, d, h, mi, s, frac, tz = match.groups()
        text = f"{y}-{mo}-{d}T{int(h):02d}:{mi}:{s or '00'}"
        if tz:
            text += "+00:00" if tz == "Z" else (tz if ":" in tz else tz[:3] + ":" + tz[3:])
        try:
            return "datetime", local(datetime.fromisoformat(text))
        except ValueError:
            return None
    match = TIME_RE.search(head)
    if match:
        h, mi, s, meridiem = int(match[1]), int(match[2]), int(match[3] or 0), match[4]
        if meridiem:
            if not 1 <= h <= 12:
                return None
            h = h % 12 + (12 if meridiem.lower() == "pm" else 0)
        if h > 23 or mi > 59 or s > 59:
            return None
        return "time", h * 3600 + mi * 60 + s
    return None


def parse_log(path: Path) -> dict[str, Any]:
    """Timestamped lines of a log, in file order.

    Lines with a full date use it. Lines with only a time of day get a date by
    counting midnight rollovers back from the log's modified time, so a gap
    longer than 24 h between two time-only lines cannot be seen.
    """
    lines = read_text(path).splitlines()
    stamps = []  # (line number, kind, value, text)
    for number, text in enumerate(lines, 1):
        stamp = line_stamp(text)
        if stamp:
            stamps.append((number, stamp[0], stamp[1], text.strip()))
    time_only = [s for s in stamps if s[1] == "time"]
    dated = [s for s in stamps if s[1] == "datetime"]
    entries, inferred = [], False
    if time_only and not dated:
        inferred = True
        rollovers, previous = 0, None
        for _, _, seconds, _ in time_only:
            if previous is not None and seconds < previous - 300:
                rollovers += 1
            previous = seconds
        day = mtime(path).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=rollovers)
        previous = None
        for number, _, seconds, text in time_only:
            if previous is not None and seconds < previous - 300:
                day += timedelta(days=1)
            previous = seconds
            entries.append({"line": number, "at": day + timedelta(seconds=seconds), "text": text})
    else:
        # Mixed logs: use the dated lines only; time-only lines there are ambiguous.
        entries = [{"line": n, "at": v, "text": t} for n, _, v, t in dated]
    longrun_lines = [e for e in entries if "[longrun]" in e["text"]]
    return {"path": str(path), "lines": len(lines), "timestamped_lines": len(entries),
            "date_inferred_from_file_time": inferred, "entries": entries,
            "last_longrun": longrun_lines[-1] if longrun_lines else None,
            "last_line": lines[-1].strip() if lines else None}


def log_gaps(log: dict[str, Any], minimum_s: float) -> list[dict[str, Any]]:
    gaps, entries = [], log["entries"]
    for before, after in zip(entries, entries[1:]):
        seconds = (after["at"] - before["at"]).total_seconds()
        if seconds > minimum_s:
            gaps.append({"source": f"log {log['path']}", "start": before["at"].isoformat(timespec="seconds"),
                         "end": after["at"].isoformat(timespec="seconds"), "minutes": round(seconds / 60, 1),
                         "lines": [before["line"], after["line"]], "before": before["text"][:160],
                         "after": after["text"][:160]})
    return gaps


def find_logs(run: Path) -> list[Path]:
    """The run's own logs: *.log inside it, <run>.log beside it, and sibling logs
    (e.g. an overnight chain log) that mention the run folder by name."""
    names, search = {run.name}, [run.parent]
    if run.name == "run":  # game_pipeline layout: runs/<name>/run
        names.add(run.parent.name)
        search.append(run.parent.parent)
    found = sorted(run.glob("*.log")) if run.is_dir() else []
    for folder in search:
        if not folder.is_dir():
            continue
        for candidate in sorted(folder.glob("*.log")):
            if candidate in found or not candidate.is_file():
                continue
            if candidate.stem in names:
                found.append(candidate)
                continue
            try:
                text = read_text(candidate)
            except OSError:
                continue
            if any(re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", text) for name in names):
                found.append(candidate)
    return found


# --- status --------------------------------------------------------------------

def locate_manifest(run: Path) -> Path | None:
    for candidate in (run / "longrun-manifest.json", run / "run" / "longrun-manifest.json"):
        if candidate.is_file():
            return candidate
    return None


def chunk_record(output: Path, chunk: dict[str, Any], fingerprint: str | None, verify: bool) -> dict[str, Any]:
    folder = chunk_dir(output, chunk)
    status_path = folder / "chunk.json"
    record = {"index": chunk["index"], "frames": chunk["frames"], "complete": False, "problem": None}
    if not status_path.is_file():
        partial = folder / "events.jsonl.partial"
        if partial.is_file():
            with partial.open("rb") as stream:
                record["partial_frames"] = sum(1 for _ in stream)
            record["partial_updated"] = mtime(partial)
        elif folder.is_dir():
            record["partial_frames"] = 0
            record["partial_updated"] = mtime(folder)
        return record
    status = read_json(status_path) or {}
    rows, state = folder / "events.jsonl", folder / "state.pkl"
    if status.get("status") != "complete":
        record["problem"] = "chunk.json is not marked complete"
    elif fingerprint and status.get("fingerprint") != fingerprint:
        record["problem"] = "fingerprint differs from the manifest"
    elif not rows.is_file() or not state.is_file():
        record["problem"] = "events.jsonl or state.pkl is missing"
    elif verify and (sha256(rows) != status.get("events_sha256") or sha256(state) != status.get("state_sha256")):
        record["problem"] = "file hash differs from chunk.json"
    else:
        record["complete"] = True
    record["frames"] = status.get("frames", chunk["frames"])
    record["seconds"] = status.get("seconds") if isinstance(status.get("seconds"), (int, float)) else None
    record["paused_seconds"] = pause_seconds(status)
    record["pauses"] = pause_list(status, f"chunk {chunk['index'] + 1} chunk.json")
    finished, started = parse_iso(status.get("finished_at")), parse_iso(status.get("started_at"))
    record["finished_basis"] = "chunk.json finished_at" if finished else "chunk.json modified time"
    record["finished"] = finished or mtime(status_path)
    if started:
        record["started"], record["started_basis"] = started, "chunk.json started_at"
    elif record["seconds"] is not None:
        # Approximate: finish time minus measured chunk seconds.
        record["started"] = record["finished"] - timedelta(seconds=record["seconds"])
        record["started_basis"] = f"{record['finished_basis']} minus chunk seconds (approx.)"
    return record


def status(run: Path, stall_minutes: float = DEFAULT_STALL_MINUTES, gap_minutes: float | None = None,
           logs: list[Path] | None = None, now: datetime | None = None, verify: bool = False) -> dict[str, Any]:
    """Everything the report prints, as plain data. Reads files only."""
    run = Path(run)
    now = local(now) if now else datetime.now().astimezone()
    stall_s = stall_minutes * 60
    gap_s = (gap_minutes if gap_minutes is not None else stall_minutes) * 60
    result: dict[str, Any] = {"run": str(run), "checked_at": now, "stall_minutes": stall_minutes,
                              "gap_minutes": gap_s / 60, "notes": [], "gaps": [], "pauses": []}
    if not run.is_dir():
        result["state"] = "missing"
        result["notes"].append(f"{run} is not a folder")
        return result
    manifest_path = locate_manifest(run)
    log_paths = find_logs(run) if logs is None else [Path(p) for p in logs]
    parsed_logs = []
    for path in log_paths:
        try:
            parsed_logs.append(parse_log(path))
        except OSError as error:
            result["notes"].append(f"could not read log {path}: {error}")
    result["logs"] = [{k: v for k, v in log.items() if k != "entries"} for log in parsed_logs]
    for log in parsed_logs:
        if not log["timestamped_lines"]:
            result["notes"].append(f"log {log['path']} has no wall-clock timestamps; gaps cannot be measured from it")
        elif log["date_inferred_from_file_time"]:
            result["notes"].append(f"log {log['path']} has times without dates; dates inferred from its modified time")
        result["gaps"].extend(log_gaps(log, gap_s))
    if manifest_path is None:
        result["state"] = "not a long run" if (run / "summary.json").is_file() else "not started"
        result["notes"].append("no longrun-manifest.json" + (" (single-pass run folder)" if
                                                             (run / "summary.json").is_file() else ""))
        return result
    output = manifest_path.parent
    result["run_folder"] = str(output)
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or "chunks" not in manifest or "selection" not in manifest:
        result["state"] = "unreadable"
        result["notes"].append(f"{manifest_path} is not a readable long-run manifest")
        return result
    fps = float(manifest.get("input", {}).get("fps") or 0)
    selection, chunks = manifest["selection"], manifest["chunks"]
    total, source_start = int(selection["frames"]), int(selection["source_start_frame"])
    report = read_json(output / "longrun-report.json") if (output / "longrun-report.json").is_file() else None
    progress = read_json(output / "progress.json") if (output / "progress.json").is_file() else None
    result.update(frames_total=total, chunks_total=len(chunks), fps=fps, device=manifest.get("device"),
                  source_start_frame=source_start, started=mtime(manifest_path),
                  started_basis="longrun-manifest.json modified time")

    records = [chunk_record(output, c, manifest.get("fingerprint"), verify) for c in chunks]
    leading = []
    for record in records:
        if not record["complete"]:
            break
        leading.append(record)
    later = [r for r in records[len(leading):] if r["complete"]]
    broken = [r for r in records if r["problem"]]
    in_progress = next((r for r in records if "partial_frames" in r), None)
    report_ok = isinstance(report, dict) and report.get("fingerprint") in (None, manifest.get("fingerprint"))
    if report_ok and len(leading) < len(chunks):
        # Finished runs may have had their chunks/ working folder removed (it is not
        # kept in git); the report is the record that every chunk completed.
        chunk_seconds = report.get("chunk_seconds") or []
        leading = [{"index": c["index"], "frames": c["frames"], "complete": True,
                    "seconds": chunk_seconds[i] if i < len(chunk_seconds) else None, "paused_seconds": 0.0,
                    "pauses": [], "finished": None} for i, c in enumerate(chunks)]
        later, broken, in_progress = [], [], None
        result["notes"].append("chunk folders absent or incomplete; counts taken from longrun-report.json")
    frames_done = sum(r["frames"] for r in leading)
    result.update(chunks_complete=len(leading), frames_done=frames_done,
                  percent=round(100 * frames_done / total, 1) if total else 0.0)
    if later:
        result["notes"].append(f"chunks {', '.join(str(r['index'] + 1) for r in later)} are complete but follow an "
                               "incomplete chunk; a resume redoes them")
    for record in broken:
        result["notes"].append(f"chunk {record['index'] + 1}: {record['problem']} (a resume redoes it)")
    if in_progress and in_progress["index"] == len(leading):
        result["current_chunk"] = {"index": in_progress["index"], "frames_written": in_progress["partial_frames"],
                                   "frames": in_progress["frames"]}
    if fps > 0:
        result["source_covered_s"] = [round(source_start / fps, 2), round((source_start + frames_done) / fps, 2)]
        result["source_selection_s"] = [round(source_start / fps, 2), round((source_start + total) / fps, 2)]

    # Last progress: the newest of every trustworthy time signal, with its source.
    candidates = [(r["finished"], f"chunk {r['index'] + 1} {r['finished_basis']}") for r in leading if r.get("finished")]
    if in_progress and in_progress.get("partial_updated"):
        candidates.append((in_progress["partial_updated"],
                           f"chunk {in_progress['index'] + 1} partial rows modified time"))
    if isinstance(progress, dict):
        updated = parse_iso(progress.get("updated_at"))
        candidates.append((updated, "progress.json updated_at") if updated else
                          (mtime(output / "progress.json"), "progress.json modified time"))
    if report_ok:
        candidates.append((mtime(output / "longrun-report.json"), "longrun-report.json modified time"))
    for log in parsed_logs:
        if log["last_longrun"]:
            candidates.append((log["last_longrun"]["at"], f"last [longrun] line in {log['path']}"))
    if candidates:
        last, basis = max(candidates, key=lambda c: c[0])
        result["last_progress"], result["last_progress_basis"] = last, basis
        result["since_progress_s"] = max((now - last).total_seconds(), 0.0)

    # Rate: processing time of completed chunks, recorded pauses excluded.
    timed = [r for r in leading if r.get("seconds")]
    work_s = sum(r["seconds"] - r.get("paused_seconds", 0.0) for r in timed)
    if timed and work_s > 0:
        paused = sum(r.get("paused_seconds", 0.0) for r in timed)
        result["rate_fps"] = sum(r["frames"] for r in timed) / work_s
        result["rate_basis"] = (f"{len(timed)} completed chunk(s), chunk.json/report seconds"
                                + (f", {paused / 60:.1f} min of recorded pauses excluded" if paused else ""))
    elif isinstance(progress, dict) and progress.get("measured_frames_per_s"):
        result["rate_fps"] = float(progress["measured_frames_per_s"])
        result["rate_basis"] = "progress.json measured_frames_per_s (" + str(progress.get("eta_basis", "last session")) + ")"
    remaining = total - frames_done
    result["frames_remaining"] = remaining
    if remaining and result.get("rate_fps"):
        result["eta_s"] = remaining / result["rate_fps"]

    # Gaps and pauses beyond the log: idle time between chunks, recorded pauses.
    for before, after in zip(leading, leading[1:]):
        if before.get("finished") and after.get("started"):
            idle = (after["started"] - before["finished"]).total_seconds()
            if idle > gap_s:
                approx = "approx." in after.get("started_basis", "") or "modified" in before.get("finished_basis", "")
                result["gaps"].append({"source": "between chunks " + ("(file modified times, approx.)" if approx
                                                                      else "(chunk.json timestamps)"),
                                       "start": before["finished"].isoformat(timespec="seconds"),
                                       "end": after["started"].isoformat(timespec="seconds"),
                                       "minutes": round(idle / 60, 1),
                                       "chunks": [before["index"] + 1, after["index"] + 1]})
    for record in leading:
        result["pauses"].extend(record.get("pauses", []))
    if isinstance(report, dict):
        result["pauses"].extend(pause_list(report, "longrun-report.json"))

    # State.
    if len(leading) == len(chunks):
        result["state"] = "complete" if report_ok else "chunks complete, merge pending"
        if not report_ok:
            result["stalled"] = result.get("since_progress_s", 0) > stall_s
    elif not leading and not in_progress and not isinstance(progress, dict):
        result["state"] = "not started"
        result["notes"].append("manifest written, no frames processed yet")
    else:
        result["stalled"] = result.get("since_progress_s", 0) > stall_s
        result["state"] = "stalled" if result["stalled"] else "running"
    return result


# --- text report -----------------------------------------------------------------

def duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    if seconds < 60:
        return f"{seconds} s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes:02d} min"
    return f"{hours // 24} d {hours % 24} h"


def when(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S %z")


def render(result: dict[str, Any]) -> str:
    out = [f"Run:            {result['run']}"]
    state = result["state"]
    if state == "stalled":
        state = (f"STALLED: no progress for {duration(result['since_progress_s'])} "
                 f"(threshold {result['stall_minutes']:g} min). Stopped, asleep or hung; rerun the same "
                 "command to resume")
    elif result.get("stalled"):
        state += f" (STALLED: no progress for {duration(result['since_progress_s'])})"
    out.append(f"State:          {state}")
    if "frames_total" in result:
        line = f"Frames:         {result['frames_done']} / {result['frames_total']} ({result['percent']:.1f}%) in complete chunks"
        current = result.get("current_chunk")
        if current:
            line += (f"; chunk {current['index'] + 1} has {current['frames_written']}/{current['frames']} "
                     "frames written (redone if interrupted)")
        out.append(line)
        out.append(f"Chunks:         {result['chunks_complete']} / {result['chunks_total']} complete")
        if "source_covered_s" in result:
            a, b = result["source_covered_s"]
            s, e = result["source_selection_s"]
            first = result["source_start_frame"]
            frames = (f"source frames {first}-{first + result['frames_done'] - 1}" if result["frames_done"]
                      else "no frames yet")
            out.append(f"Source covered: {a:.1f}-{b:.1f} s ({b - a:.1f} s of the {e - s:.1f} s selection {s:.1f}-{e:.1f} s; {frames})")
    if result.get("last_progress"):
        out.append(f"Last progress:  {when(result['last_progress'])} ({duration(result['since_progress_s'])} ago; "
                   f"from {result['last_progress_basis']})")
    elif "frames_total" in result:
        out.append("Last progress:  none recorded")
    if result.get("rate_fps"):
        rate = result["rate_fps"]
        out.append(f"Rate:           {rate:.3f} frames/s ({1 / rate:.2f} s/frame) from {result['rate_basis']}")
    elif "frames_total" in result and result.get("frames_remaining"):
        out.append("Rate:           not measured yet (no completed chunk with timing)")
    if "frames_total" in result:
        if not result.get("frames_remaining"):
            out.append("ETA:            none: detection finished")
        elif result.get("eta_s") is not None:
            eta = f"ETA:            {duration(result['eta_s'])} of processing for {result['frames_remaining']} remaining frames"
            if result["state"] == "running":
                eta += f", about {(result['checked_at'] + timedelta(seconds=result['eta_s'])).strftime('%a %H:%M')}"
            else:
                eta += " once resumed"
            out.append(eta + " (detection only; review and replay stages come after)")
        else:
            out.append("ETA:            unknown until a chunk completes")
    gaps = result.get("gaps") or []
    if result["state"] != "missing":
        out.append(f"Gaps > {result['gap_minutes']:g} min:" + ("" if gaps else "  none found"))
    for gap in gaps:
        where = f"lines {gap['lines'][0]}-{gap['lines'][1]}" if "lines" in gap else f"chunks {gap['chunks'][0]}-{gap['chunks'][1]}"
        out.append(f"  {gap['start'].replace('T', ' ')} -> {gap['end'].replace('T', ' ')}  {duration(gap['minutes'] * 60)}  [{gap['source']}, {where}]")
        if gap.get("before"):
            out.append(f"      before: {gap['before']}")
            out.append(f"      after:  {gap['after']}")
    for pause in result.get("pauses") or []:
        out.append(f"Pause:          {pause['start']} -> {pause['end']}  {pause['minutes']:g} min [{pause['source']}]")
    logs = result.get("logs")
    if logs is not None:
        out.append("Logs read:      " + (", ".join(log["path"] for log in logs) if logs else "none found (pass --log)"))
    for note in result.get("notes") or []:
        out.append(f"Note:           {note}")
    return "\n".join(out)


def jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="Long-run folder, e.g. runs/claude-game-v1-full (or runs/<name> with run/ inside)")
    p.add_argument("--stall-minutes", type=float, default=DEFAULT_STALL_MINUTES,
                   help="Flag the run as stalled after this long without progress (default 10)")
    p.add_argument("--gap-minutes", type=float, help="List wall-clock gaps longer than this (default: --stall-minutes)")
    p.add_argument("--log", type=Path, action="append",
                   help="Log file to read (repeatable). Default: *.log in the run folder, <run>.log beside it, "
                        "and sibling logs that mention the run folder name")
    p.add_argument("--verify-hashes", action="store_true",
                   help="Also hash each chunk's files like a resume does (slower; still read-only)")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of the text report")
    args = p.parse_args(argv)
    if args.stall_minutes <= 0 or (args.gap_minutes is not None and args.gap_minutes <= 0):
        p.error("--stall-minutes and --gap-minutes must be positive")
    result = status(args.run, args.stall_minutes, args.gap_minutes, args.log, verify=args.verify_hashes)
    if args.json:
        print(json.dumps(jsonable(result), indent=2))
    else:
        print(render(result))
    return 1 if result["state"] in ("missing", "unreadable") else 0


if __name__ == "__main__":
    sys.exit(main())
