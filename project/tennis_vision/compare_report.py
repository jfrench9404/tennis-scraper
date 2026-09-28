"""Readable single-file HTML report for a ``compare_runs.py`` JSON file.

    python -m tennis_vision.compare_report --input runs/<run>/compare-vs-baseline.json --output report.html

The report shows the overlap table (run A vs run B on identical source frames) and
a per-window chart and table of run B's coverage over source time. It only
re-presents the counts already in the JSON: coverage counts are NOT accuracy and
no ground truth is used. The page is fully offline (inline CSS/SVG/JS, no CDN).
"""
import argparse
import html
import json
import math
from pathlib import Path
import re

TEMPLATE = Path(__file__).with_name("compare_report.html")
NOT_ACCURACY = ("Coverage counts are not accuracy. No ground truth was used: these numbers count what "
                "each run observed (identified players, body joints, ball and racquet detections, candidate "
                "events), and more detections can include more errors.")

# Known compare_runs.py metrics: key -> (group, label). Unknown keys are still shown,
# under "Other", with their raw key, so nothing in the JSON is silently dropped.
METRICS = {
    "near_player_frames": ("Players", "Near player identified (observed, not predicted)"),
    "far_player_frames": ("Players", "Far player identified (observed, not predicted)"),
    "near_frames_6plus_joints": ("Players", "Near player frames with 6+ body joints"),
    "far_frames_6plus_joints": ("Players", "Far player frames with 6+ body joints"),
    "near_body_joints": ("Players", "Near player body joints, summed over frames"),
    "far_body_joints": ("Players", "Far player body joints, summed over frames"),
    "ball_frames": ("Ball", "Ball detections (not predicted)"),
    "racquet_assigned": ("Racquet", "Racquet detections assigned to a player"),
}
# Short column headers for the per-window table (the full label is the header's tooltip).
SHORT = {"near_player_frames": "Near player", "far_player_frames": "Far player",
         "near_frames_6plus_joints": "Near 6+ joints", "far_frames_6plus_joints": "Far 6+ joints",
         "near_body_joints": "Near joints", "far_body_joints": "Far joints", "ball_frames": "Ball",
         "racquet_assigned": "Racquet (assigned)"}
GROUPS = ["Players", "Ball", "Racquet", "Candidate events (raw, unreviewed)", "Other"]
NON_METRIC = {"run", "source_seconds", "frames"}

# Chart series in fixed categorical order (colour follows the entity, never its rank).
SERIES = [
    ("near_player_frames", "Near player", "near"),
    ("far_player_frames", "Far player", "far"),
    ("ball_frames", "Ball", "ball"),
    ("racquet_assigned", "Racquet (assigned)", "racquet"),
]

# Chart geometry (SVG user units; the SVG scales to the page width).
WIDTH, HEIGHT = 960, 340
LEFT, RIGHT, TOP, BOTTOM = 56, 200, 16, 44


def _esc(value):
    return html.escape(str(value), quote=True)


def _count(value):
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _fmt_int(value):
    return f"{value:,}"


def _fmt_delta(value):
    return "0" if value == 0 else f"{value:+,}"


def _fmt_rate(value):
    return f"{value:.2f}"


def _fmt_seconds(value):
    return f"{value:g}"


def metric_info(key):
    """(group, label) for a metric key; raw candidate events and unknown keys included."""
    if key in METRICS:
        return METRICS[key]
    if key.startswith("raw_"):
        return GROUPS[3], key[4:].replace("_", " ").capitalize()
    return "Other", key.replace("_", " ").capitalize()


def ordered_metrics(keys):
    known = list(METRICS)
    return sorted((k for k in set(keys) - NON_METRIC),
                  key=lambda k: (GROUPS.index(metric_info(k)[0]), known.index(k) if k in known else len(known), k))


def validate(report):
    if not isinstance(report, dict) or not isinstance(report.get("overlap"), dict):
        raise ValueError("Not a compare_runs.py report: missing the 'overlap' object.")
    overlap = report["overlap"]
    for side in ("a", "b"):
        if not isinstance(overlap.get(side), dict):
            raise ValueError(f"Not a compare_runs.py report: overlap.{side} is missing.")
    windows = report.get("b_windows", [])
    if not isinstance(windows, list):
        raise ValueError("Not a compare_runs.py report: 'b_windows' must be a list.")
    for w in windows:
        span = w.get("source_seconds") if isinstance(w, dict) else None
        if (not isinstance(span, list) or len(span) != 2
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in span) or span[1] <= span[0]):
            raise ValueError("Every b_windows entry needs source_seconds [start, end] with end > start.")
    return report


def overlap_rows(overlap):
    """One row per metric present on either side. Missing keys are zero (compare_runs uses Counter)."""
    a, b = overlap["a"], overlap["b"]
    rows = []
    for key in ordered_metrics(set(a) | set(b)):
        group, label = metric_info(key)
        va, vb = _count(a.get(key)), _count(b.get(key))
        rows.append({"key": key, "group": group, "label": label, "a": va, "b": vb, "delta": vb - va})
    return rows


def window_points(windows):
    """Per-window counts and count / frames rates for the chart and table, in source-time order."""
    points = []
    for w in sorted(windows, key=lambda w: w["source_seconds"][0]):
        frames = _count(w.get("frames"))
        counts = {key: _count(w.get(key)) for key, _, _ in SERIES}
        rates = {key: (counts[key] / frames if frames else None) for key, _, _ in SERIES}
        points.append({"start": float(w["source_seconds"][0]), "end": float(w["source_seconds"][1]),
                       "frames": frames, "counts": counts, "rates": rates, "raw": w})
    return points


def _nice_max(value):
    """Y-axis top: 1.0 (all frames) unless a rate exceeds it (e.g. two racquets in one frame)."""
    return 1.0 if value <= 1.0 else math.ceil(value * 4) / 4


def _label_positions(targets, low, high, gap=15.0):
    """Spread end labels vertically so they never overlap; keeps input order."""
    order = sorted(range(len(targets)), key=lambda i: targets[i])
    placed = [0.0] * len(targets)
    last = -math.inf
    for i in order:
        placed[i] = max(targets[i], last + gap, low)
        last = placed[i]
    overflow = last - high
    if overflow > 0:  # shift the whole stack up, then re-apply the gap from the top
        nxt = math.inf
        for i in reversed(order):
            placed[i] = min(placed[i] - overflow, nxt - gap)
            nxt = placed[i]
    return placed


def render_chart(points):
    """Inline SVG step chart: each window's count / frames drawn across its source-time span."""
    if not points:
        return '<p class="empty">No per-window data in this report (b_windows is empty).</p>'
    t0, t1 = points[0]["start"], max(p["end"] for p in points)
    top_rate = max([r for p in points for r in p["rates"].values() if r is not None] or [0.0])
    y_max = _nice_max(top_rate)
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    x = lambda t: LEFT + (t - t0) / (t1 - t0) * plot_w
    y = lambda r: TOP + plot_h - r / y_max * plot_h
    parts = [f'<svg id="chart" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chartTitle chartDesc" '
             f'preserveAspectRatio="xMidYMid meet">',
             '<title id="chartTitle">Run B coverage per window over source time</title>',
             f'<desc id="chartDesc">Step chart of count divided by frames for {len(points)} window(s) from '
             f'{_fmt_seconds(t0)} s to {_fmt_seconds(t1)} s of source time. The table below lists every value.</desc>',
             '<g class="grid">']
    steps = 4 if y_max == 1.0 else int(round(y_max / 0.25))
    for i in range(steps + 1):
        r = y_max * i / steps
        parts.append(f'<line x1="{LEFT}" x2="{LEFT + plot_w}" y1="{y(r):.1f}" y2="{y(r):.1f}"/>')
        parts.append(f'<text class="tick" x="{LEFT - 8}" y="{y(r) + 4:.1f}" text-anchor="end">{_fmt_rate(r)}</text>')
    parts.append("</g>")
    # X ticks at window boundaries (thinned so labels never collide).
    bounds = sorted({p["start"] for p in points} | {p["end"] for p in points})
    every = max(1, math.ceil(len(bounds) / 12))
    parts.append('<g class="xaxis">')
    parts.append(f'<line x1="{LEFT}" x2="{LEFT + plot_w}" y1="{TOP + plot_h}" y2="{TOP + plot_h}"/>')
    for i, t in enumerate(bounds):
        if i % every == 0 or i == len(bounds) - 1:
            parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{TOP + plot_h}" y2="{TOP + plot_h + 5}"/>')
            parts.append(f'<text class="tick" x="{x(t):.1f}" y="{TOP + plot_h + 19}" text-anchor="middle">'
                         f'{_fmt_seconds(t)}</text>')
    parts.append(f'<text class="axis-title" x="{LEFT + plot_w / 2:.1f}" y="{HEIGHT - 6}" text-anchor="middle">'
                 f'Source time (s)</text></g>')
    # Series: a step line per contiguous run of windows (a gap in windows breaks the line).
    ends = []
    for key, name, css in SERIES:
        segments, current, prev_end = [], [], None
        for p in points:
            rate = p["rates"][key]
            if rate is None or (prev_end is not None and p["start"] > prev_end + 1e-9):
                if current:
                    segments.append(current)
                current = []
            if rate is not None:
                current += [(x(p["start"]), y(rate)), (x(p["end"]), y(rate))]
            prev_end = p["end"]
        if current:
            segments.append(current)
        parts.append(f'<g class="series s-{css}" data-series="{_esc(key)}">')
        for seg in segments:
            coords = " ".join(f"{px:.1f},{py:.1f}" for px, py in seg)
            parts.append(f'<polyline points="{coords}"/>')
        for p in points:
            rate = p["rates"][key]
            if rate is not None:
                parts.append(f'<circle cx="{x((p["start"] + p["end"]) / 2):.1f}" cy="{y(rate):.1f}" r="4"/>')
        parts.append("</g>")
        last = next((p for p in reversed(points) if p["rates"][key] is not None), None)
        if last is not None:
            ends.append((css, name, last["rates"][key], y(last["rates"][key])))
    # Direct end labels in the right margin; text uses text tokens, a line key carries colour.
    placed = _label_positions([e[3] for e in ends], TOP + 6, TOP + plot_h)
    parts.append('<g class="end-labels">')
    for (css, name, rate, _), ly in zip(ends, placed):
        lx = LEFT + plot_w + 12
        parts.append(f'<line class="key s-{css}" x1="{lx}" x2="{lx + 14}" y1="{ly:.1f}" y2="{ly:.1f}"/>')
        parts.append(f'<text x="{lx + 20}" y="{ly + 4:.1f}">{_esc(name)} <tspan class="val">{_fmt_rate(rate)}'
                     f'</tspan></text>')
    parts.append("</g>")
    # Hover/focus targets: one full-height band per window (bigger than any mark).
    parts.append('<line id="crosshair" class="crosshair" y1="%d" y2="%d" x1="0" x2="0" visibility="hidden"/>'
                 % (TOP, TOP + plot_h))
    parts.append('<g class="hits">')
    for i, p in enumerate(points):
        values = "; ".join(f"{name} {_fmt_rate(p['rates'][key])}" if p["rates"][key] is not None
                           else f"{name} n/a" for key, name, _ in SERIES)
        label = "%s to %s s, %d frames: %s" % (_fmt_seconds(p["start"]), _fmt_seconds(p["end"]), p["frames"], values)
        parts.append(f'<rect data-index="{i}" x="{x(p["start"]):.1f}" y="{TOP}" '
                     f'width="{max(1.0, x(p["end"]) - x(p["start"])):.1f}" height="{plot_h}" tabindex="0" '
                     f'aria-label="{_esc(label)}"/>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def render_legend():
    return "".join(f'<li><span class="key s-{css}" aria-hidden="true"></span>{_esc(name)}</li>'
                   for _, name, css in SERIES)


def render_overlap_table(overlap):
    rows = overlap_rows(overlap)
    if not rows:
        return '<p class="empty">No metrics in the overlap section.</p>'
    out = ['<table class="data" id="overlapTable"><thead><tr><th scope="col">Metric</th>'
           '<th scope="col" class="num">Run A</th><th scope="col" class="num">Run B</th>'
           '<th scope="col" class="num">B − A</th></tr></thead><tbody>']
    group = None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            out.append(f'<tr class="group"><th colspan="4" scope="colgroup">{_esc(group)}</th></tr>')
        cls = "same" if r["delta"] == 0 else "diff"
        out.append(f'<tr data-key="{_esc(r["key"])}"><th scope="row">{_esc(r["label"])}'
                   f'<code>{_esc(r["key"])}</code></th><td class="num">{_fmt_int(r["a"])}</td>'
                   f'<td class="num">{_fmt_int(r["b"])}</td><td class="num {cls}">{_fmt_delta(r["delta"])}</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def render_window_table(points):
    if not points:
        return '<p class="empty">No per-window rows.</p>'
    keys = ordered_metrics({k for p in points for k in p["raw"]})
    head = "".join(f'<th scope="col" class="num" title="{_esc(metric_info(k)[1] + " (" + k + ")")}">'
                   f'{_esc(SHORT.get(k) or metric_info(k)[1])}</th>' for k in keys)
    out = ['<table class="data" id="windowTable"><thead><tr><th scope="col">Source time (s)</th>'
           f'<th scope="col" class="num">Frames</th>{head}</tr></thead><tbody>']
    for p in points:
        cells = "".join(f'<td class="num">{_fmt_int(_count(p["raw"].get(k)))}</td>' for k in keys)
        out.append(f'<tr><th scope="row" class="nowrap">{_fmt_seconds(p["start"])}–{_fmt_seconds(p["end"])}</th>'
                   f'<td class="num">{_fmt_int(p["frames"])}</td>{cells}</tr>')
    out.append("</tbody></table>")
    return "".join(out)


def render_summary(overlap, points):
    shared = _count(overlap.get("shared_frames"))
    identical = _count(overlap.get("frames_with_identical_tracks"))
    span = overlap.get("source_frames")
    span_text = f"{span[0]:,}–{span[1]:,}" if isinstance(span, list) and len(span) == 2 else "none"
    items = [("Shared source frames", _fmt_int(shared)), ("Source frame range", span_text),
             ("Frames with identical tracks", f"{_fmt_int(identical)} of {_fmt_int(shared)}"),
             ("Run B windows", _fmt_int(len(points)))]
    return "".join(f'<div class="stat"><span class="stat-label">{_esc(k)}</span><span class="stat-value">{_esc(v)}'
                   f'</span></div>' for k, v in items)


def build_html(report, source_name="compare JSON"):
    """Return the complete, self-contained HTML for a compare_runs.py report dict."""
    validate(report)
    overlap = report["overlap"]
    points = window_points(report.get("b_windows", []))
    meaning = overlap.get("meaning")
    slots = {
        "SOURCE": _esc(source_name),
        "RUN_A": _esc(overlap["a"].get("run", "run A")),
        "RUN_B": _esc(overlap["b"].get("run", "run B")),
        "NOTE": _esc(NOT_ACCURACY),
        "MEANING": _esc(meaning) if meaning else "(no 'meaning' field in the JSON)",
        "SUMMARY": render_summary(overlap, points),
        "OVERLAP_TABLE": render_overlap_table(overlap),
        "LEGEND": render_legend(),
        "CHART": render_chart(points),
        "WINDOW_TABLE": render_window_table(points),
        "DATA": json.dumps([{"start": p["start"], "end": p["end"], "frames": p["frames"], "counts": p["counts"],
                             "rates": p["rates"]} for p in points], allow_nan=False).replace("<", "\\u003c"),
        "SERIES": json.dumps([{"key": k, "name": n, "css": c} for k, n, c in SERIES]).replace("<", "\\u003c"),
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    # One pass, so substituted content can never be re-read as another slot.
    return re.sub(r"@@([A-Z_]+)@@", lambda m: slots[m.group(1)], template)


def write_report(input_path, output_path):
    input_path, output_path = Path(input_path), Path(output_path)
    report = json.loads(input_path.read_text(encoding="utf-8"))
    page = build_html(report, input_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Write a readable HTML page for a compare_runs.py JSON report.")
    p.add_argument("--input", type=Path, required=True, help="compare_runs.py output, e.g. compare-vs-baseline.json")
    p.add_argument("--output", type=Path, required=True, help="HTML file to write (single file, works offline)")
    args = p.parse_args(argv)
    out = write_report(args.input, args.output)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
