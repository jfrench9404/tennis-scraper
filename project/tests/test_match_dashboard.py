"""Offline match dashboard builder: inlined table, no network, honest replay links."""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tennis_vision import match_dashboard as md
from tennis_vision import shot_table as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # helpers shared with the shot-table tests
from test_shot_table import FPS, bounce, gravity_fit, hit, label_bounce, label_point, label_shot, player, shot, write_labels

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RAW_PLAY = PROJECT / "runs" / "raw-play-ready"
UI_TEST = HERE / "test_match_dashboard_ui.cjs"
NODE = shutil.which("node")


def make_match_replay(folder):
    """A synthetic replay with two labelled points and every landing source.

    Point p1 (labelled shots): observed, human_confirmed (explicit pixel), estimated (labelled bounce on an
    unreviewed candidate) and unknown landings. Point p2 (no labelled shots): one shot candidate with an observed
    landing and one uncertain contact. One shot candidate outside any point.
    """
    frames = [{"frame": f, "players": [player("near", 3.0, 0.5), player("far", 8.0, 23.5)], "ball_3d": None}
              for f in range(300)]
    events = [hit("h10", 10, "near"), bounce("b25", 25, "confirmed", court=[7.0, 16.0]),
              hit("h40", 40, "far"),
              bounce("b55", 55, "confirmed", court=[2.0, 3.0], basis="explicit_human_landing_pixel"),
              hit("h70", 70, "near"), bounce("b85", 85, "unreviewed", candidate=[5.0, 20.0]),
              hit("h100", 100, "far"),
              hit("h160", 160, "far"), bounce("b175", 175, "confirmed", court=[4.0, 8.0]),
              hit("h200", 200, "near"), hit("h280", 280, "near")]
    shots = {"shots": [shot("h10", 10, "near", classification="serve"), shot("h40", 40, "far"),
                       shot("h70", 70, "near"), shot("h100", 100, "far"),
                       shot("h160", 160, "far", classification="serve"),
                       shot("h200", 200, "near", state="uncertain_contact"),
                       shot("h280", 280, "near", classification="forehand")]}
    flights = {"status": "ok", "fits": [gravity_fit("h10", "b25", 10, 25, [3.0, 20.0, 2.0])], "skipped": []}
    corrections = {"schema_version": 1, "kind": "court_and_bounce_review", "run_id": "run-x",
                   "calibration_status": "reviewed", "landmark_source": "synthetic", "landmarks": {},
                   "bounce_edits": []}
    report = {"schema_version": 1, "package_id": "pkg-x", "review_run_id": "run-x", "frames": 300, "fps": FPS,
              "source_start_frame": 1000,
              "camera_fit": {"basis": "multi_point_radial_camera", "review_status": "reviewed",
                             "status": "approximate_not_measurement_grade", "landmark_rmse_px": 1.5},
              "calibration_review": corrections}
    folder.mkdir(parents=True)
    (folder / "replay-data.json").write_text(json.dumps({"report": report, "frames": frames, "events": events,
                                                         "shots": shots, "flights": flights}))
    (folder / "build-status.json").write_text(json.dumps({"status": "complete"}))
    (folder / "reviewed-events.json").write_text(json.dumps({"run_id": "run-x", "events": events}))
    (folder / "shot-candidates.json").write_text(json.dumps(shots))
    (folder / "validated-flight3d.json").write_text(json.dumps(flights))
    (folder / "calibration-corrections.json").write_text(json.dumps(corrections))
    (folder / "replay.html").write_text("<!doctype html><title>synthetic replay stand-in</title>")
    return folder


def make_match_labels(path):
    return write_labels(
        path,
        shots=[label_shot("s1", 10, "near", "serve"), label_shot("s2", 40, "far", "forehand"),
               label_shot("s3", 70, "near", "backhand"), label_shot("s4", 100, "far", "forehand")],
        bounces=[label_bounce("lb85", 85, "out")],
        points=[label_point("p1", 5, 120, server="near", winner="near", score_text="15-0"),
                label_point("p2", 150, 260, server="far", winner="unknown")])


def build_synthetic(root):
    """replay/ + table/ (via shot_table) under ``root``; returns (replay, table)."""
    replay = make_match_replay(root / "replay")
    labels = make_match_labels(root / "labels.json")
    st.run(replay, root / "table", labels)
    return replay, root / "table"


def inlined(html):
    match = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>', html, re.S)
    return json.loads(match.group(1))


class MatchDashboardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.replay, self.table = build_synthetic(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_synthetic_table_has_every_landing_source(self):
        table = json.loads((self.table / "shot-table.json").read_text())
        rows = {r["shot_id"]: r for r in table["shots"]}
        self.assertEqual({k: rows[k]["landing_source"] for k in ("h10", "h40", "h70", "h100", "h160")},
                         {"h10": "observed", "h40": "human_confirmed", "h70": "estimated", "h100": "unknown",
                          "h160": "observed"})
        self.assertEqual([p["point_id"] for p in table["points"]], ["p1", "p2"])
        self.assertEqual(rows["h160"]["rally_shot_index_source"], "estimated")

    def test_writes_one_self_contained_offline_page_with_the_table_inlined(self):
        payload = md.run(self.table, self.root / "dash", replay=self.replay)
        out = self.root / "dash"
        self.assertEqual([p.name for p in out.iterdir()], ["dashboard.html"])
        html = (out / "dashboard.html").read_text(encoding="utf-8")
        self.assertNotIn(md.PLACEHOLDER, html)
        # Offline: no external scripts, stylesheets, fonts or fetches.
        self.assertIsNone(re.search(r'(src|href)\s*=\s*["\']?(https?:)?//', html))
        self.assertNotIn("@import", html)
        self.assertNotIn("fonts.g", html)
        self.assertNotIn("fetch(", html)
        data = inlined(html)
        table = json.loads((self.table / "shot-table.json").read_text())
        self.assertEqual(data["shots"], table["shots"])
        self.assertEqual(data["points"], table["points"])
        self.assertEqual(data["calibration"], table["calibration"])
        self.assertEqual(data["table"]["sha256"], md._sha256(self.table / "shot-table.json"))
        self.assertEqual(payload["kind"], "match_dashboard")

    def test_no_spin_and_no_new_tennis_fields(self):
        md.run(self.table, self.root / "dash", replay=self.replay)
        data = inlined((self.root / "dash" / "dashboard.html").read_text(encoding="utf-8"))
        for row in data["shots"]:
            self.assertEqual(sorted(row), sorted(st.SHOT_COLUMNS))
        for row in data["points"]:
            self.assertEqual(sorted(row), sorted(st.POINT_COLUMNS))
        page = md.TEMPLATE.read_text(encoding="utf-8").lower()
        disclaimer = "spin and rpm are never shown"
        self.assertEqual(page.count(disclaimer), 1)
        page = page.replace(disclaimer, "")
        self.assertNotIn("rpm", page)
        self.assertNotRegex(page, r"\bspin\b")
        self.assertIn("(estimated)", page, "speed is labelled as an estimate")

    def test_inlined_json_cannot_close_the_script(self):
        table = json.loads((self.table / "shot-table.json").read_text())
        table["shots"][0]["shot_type_basis"] = "</script><script>alert(1)</script>"
        (self.table / "shot-table.json").write_text(json.dumps(table))
        md.run(self.table, self.root / "dash")
        html = (self.root / "dash" / "dashboard.html").read_text(encoding="utf-8")
        self.assertNotIn("</script><script>alert", html)
        self.assertEqual(inlined(html)["shots"][0]["shot_type_basis"], "</script><script>alert(1)</script>")

    # ---------------------------------------------------------------- replay links

    def test_replay_link_is_relative_and_only_when_the_replay_matches(self):
        payload = md.run(self.table, self.root / "out" / "dash", replay=self.replay)
        self.assertEqual(payload["replay_link"]["href"], "../../replay/replay.html")
        self.assertTrue((self.root / "out" / "dash" / payload["replay_link"]["href"]).is_file())
        self.assertIn("does not seek", payload["replay_link"]["seek"])

    def test_replay_found_next_to_the_table_by_folder_name(self):
        table = json.loads((self.table / "shot-table.json").read_text())
        self.assertEqual(table["replay"]["folder"], "replay")
        payload = md.run(self.table, self.root / "dash")
        self.assertEqual(payload["replay_link"]["href"], "../replay/replay.html")

    def test_no_link_without_replay_html_or_with_a_different_replay(self):
        (self.replay / "replay.html").unlink()
        payload = md.run(self.table, self.root / "dash1")
        self.assertIsNone(payload["replay_link"]["href"])
        self.assertIn("not found", payload["replay_link"]["status"])
        (self.replay / "replay.html").write_text("stand-in")
        data = json.loads((self.replay / "replay-data.json").read_text())
        data["report"]["frames"] = 299
        (self.replay / "replay-data.json").write_text(json.dumps(data))
        payload = md.run(self.table, self.root / "dash2", replay=self.replay)
        self.assertIsNone(payload["replay_link"]["href"])
        self.assertIn("not the one this table was built from", payload["replay_link"]["status"])
        payload = md.run(self.table, self.root / "dash3", replay=self.root / "missing")
        self.assertIsNone(payload["replay_link"]["href"])

    # ---------------------------------------------------------------- refusals

    def test_refuses_existing_output_and_output_inside_inputs(self):
        (self.root / "exists").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            md.run(self.table, self.root / "exists")
        with self.assertRaisesRegex(ValueError, "outside the table folder"):
            md.run(self.table, self.table / "dash")
        with self.assertRaisesRegex(ValueError, "outside the replay folder"):
            md.run(self.table, self.replay / "dash", replay=self.replay)

    def test_rejects_tables_it_cannot_trust(self):
        with self.assertRaisesRegex(ValueError, "no shot-table.json"):
            md.load_table(self.root)
        table = json.loads((self.table / "shot-table.json").read_text())
        bad = dict(table, schema_version=2)
        (self.table / "shot-table.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "schema_version"):
            md.load_table(self.table)
        bad = json.loads(json.dumps(table))
        bad["shots"][0]["landing_source"] = "measured"
        (self.table / "shot-table.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "Invalid source"):
            md.load_table(self.table)
        bad = json.loads(json.dumps(table))
        bad["shots"].pop()
        (self.table / "shot-table.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "shots.csv rows differ"):
            md.load_table(self.table)

    def test_cli(self):
        md.main(["--table", str(self.table), "--output", str(self.root / "cli")])
        self.assertTrue((self.root / "cli" / "dashboard.html").is_file())

    # ---------------------------------------------------------------- page logic (Node, fake DOM)

    @unittest.skipUnless(NODE, "node is not installed")
    def test_fake_dom_suite_on_the_built_synthetic_page(self):
        md.run(self.table, self.root / "dash", replay=self.replay)
        result = subprocess.run([NODE, str(UI_TEST), str(self.root / "dash" / "dashboard.html"), "--synthetic"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(NODE, "node is not installed")
    def test_js_fixture_matches_the_shot_table_columns(self):
        result = subprocess.run([NODE, str(UI_TEST), "--print-fixture"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        fixture = json.loads(result.stdout)
        for row in fixture["shots"]:
            self.assertEqual(sorted(row), sorted(st.SHOT_COLUMNS))
        for row in fixture["points"]:
            self.assertEqual(sorted(row), sorted(st.POINT_COLUMNS))

    @unittest.skipUnless((RAW_PLAY / "replay-data.json").is_file(), "committed raw-play-ready replay not present")
    def test_real_raw_play_ready_table_builds_with_mostly_unknown_landings(self):
        st.run(RAW_PLAY, self.root / "raw-table")
        payload = md.run(self.root / "raw-table", self.root / "raw-dash", replay=RAW_PLAY)
        self.assertEqual(payload["points"], [])
        self.assertTrue(all(r["landing_source"] == "unknown" for r in payload["shots"]))
        self.assertTrue(payload["replay_link"]["href"].endswith("raw-play-ready/replay.html"))
        if NODE:
            result = subprocess.run([NODE, str(UI_TEST), str(self.root / "raw-dash" / "dashboard.html")],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
