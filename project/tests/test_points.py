"""Point PROPOSALS from synthetic replay folders: serve starts, dead-ball ends, repeat serves, changeovers."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tennis_vision.points import DEFAULTS, load_input, main as points_main, propose

FPS = 10.0      # Small synthetic fps keeps frames few; thresholds are in seconds.
OFFSET = 1000   # Replays start mid-video: source_frame = OFFSET + run_frame.
FORBIDDEN_KEYS = {"winner", "point_winner", "score", "ace", "fault", "let", "outcome", "winning_shot"}
PROJECT = Path(__file__).resolve().parents[1]


def run_main(argv):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        points_main(argv)
    return out.getvalue()


def keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from keys(v)


class Clip:
    """A synthetic replay folder in the shape shot_replay writes (only the fields points.py reads)."""

    def __init__(self, frames):
        self.n = frames
        self.ball = [False] * frames
        # Court metres (x, y along the court; near baseline y=0, far baseline y=23.77).
        self.near = [(5.0, 0.5)] * frames
        self.far = [(5.0, 23.0)] * frames
        self.events, self.assessments, self.shots = [], [], []

    def balls(self, first, last, seen=True):
        for f in range(first, last + 1):
            self.ball[f] = seen
        return self

    def hide(self, identity, first, last):
        for f in range(first, last + 1):
            getattr(self, identity)[f] = None
        return self

    def walk(self, identity, first, last, start_xy, end_xy):
        for f in range(first, last + 1):
            t = (f - first) / (last - first)
            getattr(self, identity)[f] = (start_xy[0] + t * (end_xy[0] - start_xy[0]),
                                          start_xy[1] + t * (end_xy[1] - start_xy[1]))
        for f in range(last + 1, self.n):
            getattr(self, identity)[f] = end_xy
        return self

    def _event(self, event_id, frame, side, status="unreviewed"):
        self.events.append({"id": event_id, "type": "hit", "status": status, "frame": frame, "player_id": side,
                            "contact_support": "moderate"})

    def serve(self, event_id, frame, side, status="unreviewed"):
        self._event(event_id, frame, side, status)
        # As play_context: the serve pattern is computed regardless, but a human-rejected contact is "no_shot".
        rejected = status == "rejected"
        self.assessments.append({"event_id": event_id, "frame": frame,
                                 "action_state": "no_shot" if rejected else "shot_candidate",
                                 "phase": "unresolved" if rejected else "serve_candidate", "serve_pattern": True,
                                 "reason": "synthetic serve",
                                 "incoming": {"relation": "unresolved", "observed_frames": []},
                                 "outgoing": {"relation": "court_flight", "observed_frames": [frame + 1]}})
        self.shots.append({"event_id": event_id, "frame": frame, "player_id": side, "classification": "serve",
                           "support": "moderate", "action_state": "shot_candidate", "event_status": status,
                           "contact_support": "moderate", "reasons": ["synthetic"],
                           "evidence": {"player_feet_court_m": [5.0, 0.5]}})
        return self

    def rally(self, event_id, frame, side):
        self._event(event_id, frame, side)
        self.assessments.append({"event_id": event_id, "frame": frame, "action_state": "shot_candidate",
                                 "phase": "live_ball_candidate", "serve_pattern": False, "reason": "synthetic rally",
                                 "incoming": {"relation": "court_flight", "observed_frames": [frame - 1]},
                                 "outgoing": {"relation": "court_flight", "observed_frames": [frame + 1]}})
        self.shots.append({"event_id": event_id, "frame": frame, "player_id": side, "classification": "forehand",
                           "support": "moderate", "action_state": "shot_candidate", "event_status": "unreviewed"})
        return self

    def handling(self, event_id, frame, side, first_seen):
        self._event(event_id, frame, side)
        self.assessments.append({"event_id": event_id, "frame": frame, "action_state": "no_shot_candidate",
                                 "phase": "between_points_candidate", "serve_pattern": False, "reason": "local",
                                 "incoming": {"relation": "local_activity",
                                              "observed_frames": list(range(frame - 1, first_seen - 1, -1))},
                                 "outgoing": {"relation": "local_activity", "observed_frames": [frame + 1]}})
        return self

    def write(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True)
        live = {a["frame"]: a["phase"] for a in self.assessments if a["action_state"] == "shot_candidate"}
        frames = []
        for f in range(self.n):
            players = [{"identity_id": k, "predicted": False,
                        "feet_xyz_m": [xy[0] - 5.485, xy[1] - 11.885, 0.0]}
                       for k, xy in (("near", self.near[f]), ("far", self.far[f])) if xy is not None]
            frames.append({"frame": f, "players": players, "ball_observed_2d": [640.0, 300.0] if self.ball[f] else None,
                           "play_phase": {"phase": live.get(f, "unresolved"), "basis": "synthetic"}})
        report = {"fps": FPS, "frames": self.n, "source_start_frame": OFFSET, "review_run_id": "synthetic-run"}
        (folder / "replay-data.json").write_text(json.dumps({"report": report, "frames": frames,
                                                             "events": self.events}))
        (folder / "play-context.json").write_text(json.dumps({"method": "raw_play_context_v1",
                                                              "assessments": self.assessments}))
        (folder / "shot-candidates.json").write_text(json.dumps({"method": "synthetic", "shots": self.shots}))
        return folder


class PointProposalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def proposals(self, clip, **kwargs):
        return propose(load_input(clip.write(self.root / "replay")), **kwargs)

    def assert_no_outcome_claims(self, report):
        self.assertFalse(FORBIDDEN_KEYS & set(keys(report["points"])))
        for p in report["proposals"]:
            self.assertEqual(p["status"], "proposal_unreviewed")
            self.assertFalse(FORBIDDEN_KEYS & set(keys(p)), "proposal claims an outcome")

    def test_clean_serve_rally_dead_ball(self):
        clip = (Clip(300).balls(15, 75).serve("s1", 20, "near").rally("r1", 35, "far").rally("r2", 50, "near")
                .rally("r3", 65, "far").walk("near", 50, 55, (5.0, 0.5), (5.0, 4.0))
                .walk("near", 80, 120, (5.0, 4.0), (5.0, 0.0)))
        report = self.proposals(clip)
        self.assert_no_outcome_claims(report)
        self.assertEqual(len(report["proposals"]), 1)
        p = report["proposals"][0]
        self.assertEqual(p["run_frames"], [20, 75])
        self.assertEqual(p["source_frames"], [OFFSET + 20, OFFSET + 75])
        self.assertEqual(p["source_seconds"], [(OFFSET + 20) / FPS, (OFFSET + 75) / FPS])
        self.assertEqual(p["start"]["basis"], "serve_candidate")
        self.assertEqual(p["start"]["server_side"], "near")
        self.assertEqual(p["start"]["confidence_basis"]["sources"], ["play_context", "shot_classification"])
        self.assertEqual(p["end"]["basis"], "no_ball_observed")
        self.assertFalse(p["end"]["open_end"])
        self.assertEqual(p["end"]["last_live_evidence"]["run_frame"], 65)
        walking = p["end"]["corroborating_walking"]
        self.assertTrue(walking and all(w["player"] == "near" and w["direction"] == "towards_own_baseline"
                                        for w in walking))
        self.assertEqual([e["event_id"] for e in p["events_within"]], ["s1", "r1", "r2", "r3"])
        self.assertEqual(report["thresholds"]["dead_ball_seconds"], DEFAULTS["dead_ball_seconds"])
        self.assertIn("score_labels", report["label_scoring"])
        self.assertNotIn("TODO", report["label_scoring"])
        # Plain view read by score_labels --points: source frames, inclusive, server side, no winner.
        self.assertEqual(report["points"], [{"id": "point-proposal-1", "source_start_frame": OFFSET + 20,
                                             "source_end_frame": OFFSET + 75, "server": "near",
                                             "status": "proposal_unreviewed"}])
        self.assertEqual(report["run_id"], "synthetic-run")

    def test_let_or_second_serve_stays_in_one_proposal_by_default(self):
        # First serve, ball dead for 4 s (ball handling), second serve 6 s later by the same server, then a rally.
        clip = (Clip(300).balls(15, 30).balls(66, 72).balls(75, 130).serve("s1", 20, "near")
                .handling("h1", 70, "near", 66).serve("s2", 80, "near").rally("r1", 95, "far")
                .rally("r2", 110, "near"))
        report = self.proposals(clip)
        self.assert_no_outcome_claims(report)
        self.assertEqual(len(report["proposals"]), 1)
        p = report["proposals"][0]
        self.assertTrue(p["repeat_serve_candidate"])
        self.assertEqual([s["event_id"] for s in p["serve_candidates"]], ["s1", "s2"])
        self.assertEqual(p["serve_candidates"][1]["repeat_of"], "s1")
        self.assertEqual(p["run_frames"], [20, 130])
        self.assertEqual(report["repeat_serve_policy"], "same_point")

        split = propose(load_input(self.root / "replay"), repeat_serve="new_point")
        self.assertEqual([q["run_frames"] for q in split["proposals"]], [[20, 30], [80, 130]])
        self.assertEqual(split["proposals"][1]["serve_candidates"][0]["repeat_of"], "s1")

    def test_serves_far_apart_or_by_different_servers_are_new_points(self):
        clip = (Clip(500).balls(15, 40).balls(255, 280).balls(300, 320).serve("s1", 20, "near")
                .serve("s2", 260, "near").serve("s3", 305, "far"))
        report = self.proposals(clip)
        self.assertEqual([p["start"]["event_id"] for p in report["proposals"]], ["s1", "s2", "s3"])
        self.assertFalse(any(p["repeat_serve_candidate"] for p in report["proposals"]))
        self.assertTrue(report["proposals"][2]["server_side_changed_from_previous"])

    def test_missing_ball_mid_rally_does_not_end_the_point(self):
        # A short gap 38-47 (1 s < 3 s) and a long gap 55-90 (3.6 s) that rally shot candidates follow.
        clip = (Clip(300).balls(15, 125).balls(38, 47, seen=False).balls(55, 90, seen=False)
                .serve("s1", 20, "near").rally("r1", 35, "far").rally("r2", 50, "near")
                .rally("r3", 95, "far").rally("r4", 110, "near"))
        report = self.proposals(clip)
        self.assertEqual(len(report["proposals"]), 1)
        p = report["proposals"][0]
        self.assertEqual(p["run_frames"], [20, 125])
        self.assertEqual(p["end"]["basis"], "no_ball_observed")
        ignored = p["end"]["ignored_signals"]
        self.assertEqual([(s["signal"], s["gap_run_frames"]) for s in ignored], [("no_ball_observed", [55, 90])])
        self.assertIn("shot candidate follows", ignored[0]["ignored_because"])
        self.assertEqual([e["event_id"] for e in p["events_within"]], ["s1", "r1", "r2", "r3", "r4"])

    def test_changeover_between_points(self):
        # Point 1 (near serves), players leave for 30 s (#24 break), point 2 (far serves).
        clip = (Clip(600).balls(15, 60).balls(445, 500).serve("s1", 20, "near").rally("r1", 40, "far")
                .hide("near", 100, 399).hide("far", 100, 399).serve("s2", 450, "far").rally("r2", 470, "near"))
        report = self.proposals(clip)
        self.assert_no_outcome_claims(report)
        first, second = report["proposals"]
        self.assertEqual(first["run_frames"], [20, 60])
        self.assertEqual(first["end"]["basis"], "no_ball_observed")
        self.assertIn("break_proposal", [s["signal"] for s in first["end"]["signals"]])
        self.assertEqual(report["break_proposals_source"]["source"], "computed_with_segments_defaults")
        self.assertEqual(len(report["evidence"]["break_proposals"]), 1)
        self.assertEqual(second["preceded_by_break_proposals"], [report["evidence"]["break_proposals"][0]["id"]])
        self.assertEqual(second["start"]["server_side"], "far")
        self.assertTrue(second["server_side_changed_from_previous"])
        self.assertEqual(second["run_frames"], [450, 500])

    def test_break_proposal_file_ends_a_point(self):
        clip = Clip(300).balls(10, 299).serve("s1", 20, "near").rally("r1", 40, "far")
        loaded = load_input(clip.write(self.root / "replay"))
        segments = {"kind": "segment_proposals", "fps": FPS, "thresholds": {"x": 1},
                    "proposals": [{"id": "proposal-1", "source_frames": [OFFSET + 150, OFFSET + 250],
                                   "reasons": ["near_side_empty"], "status": "proposal_unreviewed"}]}
        report = propose(loaded, segments_report=segments)
        self.assertEqual(report["proposals"][0]["run_frames"], [20, 149])
        self.assertEqual(report["proposals"][0]["end"]["basis"], "break_proposal")
        with self.assertRaises(ValueError):
            propose(loaded, segments_report={**segments, "fps": 30.0})

    def test_ball_seen_to_data_end_is_an_open_end(self):
        report = self.proposals(Clip(100).balls(15, 99).serve("s1", 20, "near").rally("r1", 40, "far"))
        p = report["proposals"][0]
        self.assertEqual(p["run_frames"], [20, 99])
        self.assertTrue(p["end"]["open_end"])
        self.assertEqual(p["end"]["basis"], "data_end")

    def test_rejected_serve_and_no_serve_give_no_proposal(self):
        clip = Clip(100).balls(15, 60).serve("s1", 20, "near", status="rejected").rally("r1", 40, "far")
        report = self.proposals(clip)
        self.assertEqual(report["proposals"], [])
        self.assertEqual(report["serve_candidates"], [])
        self.assertTrue(any("No serve candidate" in n for n in report["notes"]))
        self.assertEqual([e["event_id"] for e in report["unattached_events"]], ["r1"])

    def test_cli_game_folder_and_output_safety(self):
        game = self.root / "game"
        game.mkdir()
        Clip(300).balls(15, 75).serve("s1", 20, "near").rally("r1", 35, "far").write(game / "replay")
        with self.assertRaises(SystemExit):
            run_main(["--run", str(game), "--output", str(game / "points.json")])
        out = self.root / "out" / "points.json"
        text = run_main(["--run", str(game), "--output", str(out), "--dead-ball-seconds", "2.5"])
        self.assertIn("1 point proposal", text)
        report = json.loads(out.read_text())
        self.assertEqual(report["input_kind"], "game")
        self.assertEqual(report["thresholds"]["dead_ball_seconds"], 2.5)
        self.assertEqual(report["proposals"][0]["source_frames"], [OFFSET + 20, OFFSET + 75])

    def test_run_rows_without_replay_propose_nothing(self):
        run = self.root / "run"
        run.mkdir()
        (run / "source-offset.json").write_text(json.dumps({"source_start_frame": OFFSET}))
        (run / "summary.json").write_text(json.dumps({"fps": FPS, "frames": 50}))
        with (run / "events.jsonl").open("w") as stream:
            for f in range(50):
                tracks = [{"label": "player", "identity_id": "near", "predicted": False, "court_m": [5, 1]}]
                if f < 10:
                    tracks.append({"label": "ball", "predicted": False})
                stream.write(json.dumps({"frame": f, "tracks": tracks}) + "\n")
        report = propose(load_input(run))
        self.assertEqual(report["input_kind"], "run")
        self.assertEqual(report["proposals"], [])
        self.assertEqual(report["source_frames"], [OFFSET, OFFSET + 49])
        self.assertEqual([g["start_run_frame"] for g in report["evidence"]["dead_ball_gaps"]], [10])
        self.assertIn("no serve", report["notes"][0])

    def test_invalid_settings_are_refused(self):
        loaded = load_input(Clip(50).balls(0, 49).write(self.root / "replay"))
        with self.assertRaises(ValueError):
            propose(loaded, dead_ball_seconds=-1)
        with self.assertRaises(ValueError):
            propose(loaded, repeat_serve="merge")

    def test_committed_raw_play_ready_rally(self):
        folder = PROJECT / "runs" / "raw-play-ready"
        if not (folder / "replay-data.json").is_file():
            self.skipTest("runs/raw-play-ready is not present")
        report = propose(load_input(folder))
        self.assert_no_outcome_claims(report)
        self.assertEqual(report["source_start_frame"], 4740)
        self.assertEqual([p["start"]["source_frame"] for p in report["proposals"]], [4797])
        self.assertEqual(report["proposals"][0]["start"]["server_side"], "near")


if __name__ == "__main__":
    unittest.main()
