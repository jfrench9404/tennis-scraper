"""The compare report re-presents compare_runs.py counts as one offline HTML page."""
import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from tennis_vision import compare_report
from tennis_vision.compare_report import build_html, main, overlap_rows, window_points
from tennis_vision.compare_runs import compare, windows

PROJECT = Path(__file__).resolve().parents[1]
REAL = PROJECT / "runs" / "claude-longrun-dml-30" / "compare-vs-baseline.json"


def report(a=None, b=None, b_windows=None, **overlap):
    base = {"source_frames": [100, 102], "shared_frames": 3, "frames_with_identical_tracks": 2,
            "a": {"run": "runs/a", **(a or {})}, "b": {"run": "runs/b", **(b or {})},
            "meaning": "Observation coverage on identical source frames. Not accuracy; no ground truth used."}
    base.update(overlap)
    return {"overlap": base, "b_windows": b_windows if b_windows is not None else []}


def window(start, end, frames, **counts):
    return {"source_seconds": [start, end], "frames": frames, **counts}


def slot_free(page):
    return not re.search(r"@@[A-Z_]+@@", page)


class OverlapTableTest(unittest.TestCase):
    def test_rows_cover_both_sides_and_missing_keys_are_zero(self):
        rows = overlap_rows(report(a={"ball_frames": 5, "raw_contact_candidate": 1},
                                   b={"ball_frames": 7, "far_player_frames": 3})["overlap"])
        by_key = {r["key"]: r for r in rows}
        self.assertEqual(set(by_key), {"ball_frames", "raw_contact_candidate", "far_player_frames"})
        self.assertEqual((by_key["ball_frames"]["a"], by_key["ball_frames"]["b"], by_key["ball_frames"]["delta"]),
                         (5, 7, 2))
        self.assertEqual((by_key["far_player_frames"]["a"], by_key["far_player_frames"]["delta"]), (0, 3))
        self.assertEqual(by_key["raw_contact_candidate"]["b"], 0)
        # Players, then ball, then candidate events (grouped in a fixed order).
        self.assertEqual([r["key"] for r in rows], ["far_player_frames", "ball_frames", "raw_contact_candidate"])

    def test_unknown_metrics_are_shown_not_dropped(self):
        page = build_html(report(a={"mystery_count": 4}, b={"mystery_count": 4}))
        self.assertIn('data-key="mystery_count"', page)
        self.assertIn("Other", page)


class ChartTest(unittest.TestCase):
    def test_rates_are_count_over_window_frames(self):
        points = window_points([window(30, 60, 10, ball_frames=5), window(0, 30, 20, near_player_frames=20)])
        self.assertEqual([p["start"] for p in points], [0.0, 30.0])  # sorted by source time
        self.assertEqual(points[0]["rates"]["near_player_frames"], 1.0)
        self.assertEqual(points[1]["rates"]["ball_frames"], 0.5)
        self.assertEqual(points[1]["rates"]["far_player_frames"], 0.0)
        self.assertIsNone(window_points([window(0, 30, 0)])[0]["rates"]["ball_frames"])

    def test_one_polyline_per_series_for_contiguous_windows(self):
        page = build_html(report(b_windows=[window(0, 30, 30, ball_frames=3), window(30, 60, 30, ball_frames=6)]))
        self.assertEqual(page.count("<polyline"), len(compare_report.SERIES))
        for _, _, css in compare_report.SERIES:
            self.assertIn(f'class="series s-{css}"', page)
        self.assertEqual(page.count('<rect data-index='), 2)  # one hover/focus band per window

    def test_missing_window_breaks_the_line(self):
        page = build_html(report(b_windows=[window(0, 30, 30), window(30, 60, 30), window(90, 120, 30)]))
        self.assertEqual(page.count("<polyline"), 2 * len(compare_report.SERIES))

    def test_y_axis_grows_past_one_only_when_needed(self):
        normal = build_html(report(b_windows=[window(0, 30, 30, racquet_assigned=30)]))
        self.assertIn(">1.00</text>", normal)
        self.assertNotIn(">1.25</text>", normal)
        two_racquets = build_html(report(b_windows=[window(0, 30, 30, racquet_assigned=36)]))
        self.assertIn(">1.25</text>", two_racquets)

    def test_end_labels_do_not_overlap(self):
        placed = compare_report._label_positions([100.0, 100.0, 101.0, 300.0], 20, 310)
        ordered = sorted(placed)
        self.assertTrue(all(b - a >= 15 - 1e-9 for a, b in zip(ordered, ordered[1:])))
        self.assertTrue(all(20 <= v <= 310 for v in placed))
        crowded_bottom = compare_report._label_positions([300.0, 300.0, 300.0], 20, 310)
        self.assertLessEqual(max(crowded_bottom), 310)

    def test_empty_windows_render_a_message(self):
        page = build_html(report(b_windows=[]))
        self.assertNotIn("<svg", page)
        self.assertIn("b_windows is empty", page)
        self.assertTrue(slot_free(page))


class PageTest(unittest.TestCase):
    def test_note_and_tables_are_present(self):
        page = build_html(report(a={"ball_frames": 5}, b={"ball_frames": 5},
                                 b_windows=[window(150, 180, 30, ball_frames=5)]))
        self.assertIn('id="accuracyNote"', page)
        self.assertIn("Coverage is not accuracy. No ground truth was used.", page)
        self.assertIn('id="overlapTable"', page)
        self.assertIn('id="windowTable"', page)
        self.assertIn("2 of 3", page)  # identical-track frames of shared frames
        self.assertIn("100–102", page)
        self.assertTrue(slot_free(page))

    def test_works_offline(self):
        page = build_html(report(b_windows=[window(0, 30, 30, ball_frames=1)]))
        self.assertNotRegex(page, r"<script[^>]+\bsrc=")
        self.assertNotRegex(page, r"<link\b")
        self.assertNotRegex(page, r"@import|url\(")
        self.assertNotRegex(page, r"(?:src|href)=[\"']?(?:https?:)?//")

    def test_untrusted_strings_are_escaped(self):
        evil = '</script><img src=x onerror=alert(1)>@@CHART@@'
        page = build_html(report(a={"run": evil}, b={"run": "ok", f"raw_{evil}": 1},
                                 b_windows=[window(0, 30, 30)], meaning=evil), source_name=evil)
        self.assertNotIn("<img src=x", page)
        self.assertEqual(page.count("</script>"), 3)  # only the page's own three script blocks
        self.assertIn("@@CHART@@", page)  # user text stays literal, never substituted into a slot
        self.assertEqual(page.count("<svg"), 1)

    def test_no_overlap_is_stated(self):
        page = build_html(report(source_frames=None, shared_frames=0, frames_with_identical_tracks=0))
        self.assertIn(">none<", page)
        self.assertIn("0 of 0", page)

    def test_rejects_non_compare_json(self):
        for bad in ({}, {"overlap": {"a": {}}}, {"overlap": {"a": {}, "b": {}}, "b_windows": {}},
                    report(b_windows=[{"source_seconds": [30, 30], "frames": 1}]),
                    report(b_windows=[{"frames": 1}])):
            with self.assertRaises(ValueError):
                build_html(bad)

    def test_matches_compare_runs_output(self):
        """End to end: compare_runs.py output on synthetic runs feeds the report unchanged."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, start, rows in (("a", 100, 3), ("b", 0, 104)):
                folder = root / name
                folder.mkdir()
                (folder / "source-offset.json").write_text(json.dumps({"source_start_frame": start}))
                with (folder / "events.jsonl").open("w") as stream:
                    for i in range(rows):
                        tracks = [{"label": "ball", "predicted": False}] if (start + i) % 2 else []
                        stream.write(json.dumps({"frame": i, "time_s": i / 30, "tracks": tracks,
                                                 "events": []}) + "\n")
            data = {"overlap": compare(root / "a", root / "b"), "b_windows": windows(root / "b", 1.0)}
            data = json.loads(json.dumps(data))  # exactly what compare_runs.py writes to disk
            page = build_html(data)
            self.assertEqual(page.count('<rect data-index='), 4)  # 104 frames in 1 s windows at 30 fps
            self.assertIn('data-key="ball_frames"', page)
            self.assertTrue(slot_free(page))


class CliTest(unittest.TestCase):
    @unittest.skipUnless(REAL.exists(), "committed compare-vs-baseline.json not present")
    def test_cli_on_committed_compare_json(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "nested" / "report.html"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--input", str(REAL), "--output", str(out)])
            page = out.read_text(encoding="utf-8")
            data = json.loads(REAL.read_text(encoding="utf-8"))
            a, b = data["overlap"]["a"], data["overlap"]["b"]
            for key in (set(a) | set(b)) - {"run"}:
                self.assertRegex(page, rf'data-key="{key}">.*?</th><td class="num">{a.get(key, 0):,}</td>'
                                       rf'<td class="num">{b.get(key, 0):,}</td>')
            self.assertIn("compare-vs-baseline.json", page)
            self.assertEqual(page.count('<rect data-index='), len(data["b_windows"]))
            self.assertTrue(slot_free(page))


if __name__ == "__main__":
    unittest.main()
