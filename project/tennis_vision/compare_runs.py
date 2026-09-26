"""Coverage comparison between runs over the SAME source frames, plus per-window
coverage across a long run.

Everything here counts observations (identified players, supported joints,
ball and racquet detections, candidate events). Coverage is NOT accuracy: no
ground-truth labels are used, and more detections can include more errors.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def load(run):
    run = Path(run)
    offset = json.loads((run / "source-offset.json").read_text(encoding="utf-8"))["source_start_frame"]
    with (run / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            yield offset + row["frame"], row


def frame_stats(row):
    stats = Counter()
    for t in row["tracks"]:
        if t["label"] == "player" and t.get("identity_id") in ("near", "far") and not t.get("predicted"):
            ident = t["identity_id"]
            stats[ident + "_player_frames"] += 1
            joints = sum(v[2] >= .3 for k, v in (t.get("keypoints") or {}).items() if k not in (
                "nose", "left_eye", "right_eye", "left_ear", "right_ear"))
            stats[ident + "_body_joints"] += joints
            stats[ident + "_frames_6plus_joints"] += joints >= 6
        elif t["label"] == "ball" and not t.get("predicted"):
            stats["ball_frames"] += 1
        elif t["label"] == "racket" and not t.get("predicted") and t.get("player_track_id") is not None:
            stats["racquet_assigned"] += 1
    for e in row["events"]:
        stats["raw_" + e["type"]] += 1
    return stats


def compare(run_a, run_b):
    a = {f: row for f, row in load(run_a)}
    shared, totals = [], {"a": Counter(), "b": Counter()}
    identical = 0
    for f, row in load(run_b):
        if f in a:
            shared.append(f)
            totals["a"].update(frame_stats(a[f]))
            totals["b"].update(frame_stats(row))
            other = dict(row); other.pop("source_frame", None); other["frame"] = a[f]["frame"]
            other["time_s"] = a[f]["time_s"]
            mine = dict(a[f]); mine.pop("source_frame", None)
            identical += json.dumps(mine["tracks"]) == json.dumps(other["tracks"])
    return {"source_frames": [min(shared), max(shared)] if shared else None, "shared_frames": len(shared),
            "frames_with_identical_tracks": identical,
            "a": {"run": str(run_a), **dict(totals["a"])}, "b": {"run": str(run_b), **dict(totals["b"])},
            "meaning": "Observation coverage on identical source frames. Not accuracy; no ground truth used."}


def windows(run, seconds=30.0, fps=30.0):
    size = int(round(seconds * fps))
    out = {}
    for f, row in load(run):
        key = f // size
        out.setdefault(key, Counter({"frames": 0}))
        out[key]["frames"] += 1
        out[key].update(frame_stats(row))
    return [{"source_seconds": [k * seconds, (k + 1) * seconds], **dict(v)} for k, v in sorted(out.items())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True, help="Reference run (e.g. runs/rally-neural-ball)")
    p.add_argument("--b", type=Path, required=True, help="Run to compare (e.g. a full-game run)")
    p.add_argument("--window-seconds", type=float, default=30.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.b / "summary.json").read_text(encoding="utf-8"))
    report = {"overlap": compare(args.a, args.b),
              "b_windows": windows(args.b, args.window_seconds, summary["fps"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    o = report["overlap"]
    print(f"Shared source frames {o['source_frames']} ({o['shared_frames']}), identical tracks in "
          f"{o['frames_with_identical_tracks']}")
    for key in sorted(set(o["a"]) | set(o["b"]) - {"run"}):
        if key != "run":
            print(f"  {key:28s} a={o['a'].get(key, 0):>7}  b={o['b'].get(key, 0):>7}")


if __name__ == "__main__":
    main()
