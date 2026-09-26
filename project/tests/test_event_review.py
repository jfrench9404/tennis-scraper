import copy
import json
from pathlib import Path
import tempfile
import unittest

from tennis_vision.event_analysis import observed_balls, detect_events, fill_short_gaps, proximity
from tennis_vision.event_review import load_run


def ball(x, y, **kwargs):
    return dict(label="ball", bbox=[x-2, y-2, x+2, y+2], source="gridtracknet",
                confidence=.9, predicted=False, track_id=1) | kwargs


def rows_for(points):
    return [{"frame": f, "time_s": f/30, "tracks": [ball(*p)] if p else []}
            for f, p in enumerate(points)]


class Court:
    def project(self, point):
        return [5., 12.]


class EventReviewTests(unittest.TestCase):
    def test_observations_exclude_estimates_motion_predictions_and_ambiguity(self):
        rows = rows_for([(10, 10)]*6)
        rows[1]["tracks"][0]["source"] = "motion"
        rows[2]["tracks"][0]["predicted"] = True
        rows[3]["tracks"][0]["source"] = "interpolated"
        rows[4]["tracks"].append(ball(20, 20))
        rows[5]["tracks"][0]["bbox"][0] = float("nan")
        self.assertEqual([b is not None for b in observed_balls(rows)], [True, False, False, False, False, False])

    def test_bounce_uses_observations_and_conditional_coordinates(self):
        rows = rows_for([(200+f*2, 200-abs(f-8)*6) for f in range(17)])
        events = detect_events(rows, observed_balls(rows), 30, Court())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "bounce_candidate")
        self.assertEqual(events[0]["frame"], 8)
        self.assertEqual(events[0]["status"], "unreviewed")
        self.assertEqual(events[0]["landing_if_bounce_m"], [5., 12.])
        self.assertEqual(events[0]["evidence"]["ball_evidence"], "learned_observations_only")
        self.assertEqual(detect_events(rows, observed_balls(rows), 30), [])

    def test_smooth_apex_is_not_bounce(self):
        rows = rows_for([(200+f*3, 200-.8*(f-8)**2) for f in range(17)])
        self.assertEqual(detect_events(rows, observed_balls(rows), 30, Court()), [])

    def test_no_event_across_camera_cut(self):
        rows = rows_for([(200+f*2, 200-abs(f-8)*6) for f in range(17)])
        self.assertEqual(detect_events(rows, observed_balls(rows), 30, Court(), cuts=[8]), [])

    def test_hit_needs_motion_change_not_only_racket_proximity(self):
        for turning in (True, False):
            rows = rows_for([(200+2*f, 200-abs(f-8)*6 if turning else 152+6*f) for f in range(17)])
            for row in rows:
                row["tracks"] += [dict(label="player", track_id=7, identity_id="far", predicted=False,
                    bbox=[195, 175, 235, 255], keypoints={}, court_m=[5, 26]),
                    dict(label="racket", bbox=[206, 190, 226, 210], player_track_id=7, predicted=False)]
            events = detect_events(rows, observed_balls(rows), 30, Court())
            self.assertEqual(len(events), int(turning))
            if turning:
                self.assertEqual(events[0]["type"], "hit_candidate")
                self.assertEqual(events[0]["player_id"], "far")
                self.assertIsNone(events[0]["landing_if_bounce_m"])

    def test_fixed_70px_far_player_false_contact_rejected(self):
        row = {"tracks": [dict(label="player", track_id=1, identity_id="far", bbox=[500, 60, 535, 110]),
                          dict(label="racket", player_track_id=1, bbox=[540, 85, 555, 100])]}
        self.assertIsNone(proximity(row, [598, 60]))
        self.assertIsNotNone(proximity(row, [553, 94]))
        row["tracks"][0]["predicted"] = True
        self.assertIsNone(proximity(row, [553, 94]))

    def test_detector_jump_is_not_an_event(self):
        rows = rows_for([(200+f*3, 200+f*2) for f in range(17)])
        rows[8]["tracks"] = [ball(1000, 100)]
        self.assertEqual(detect_events(rows, observed_balls(rows), 30, Court()), [])

    def test_short_gap_is_explicit_estimate_and_original_immutable(self):
        rows = rows_for([(50+f*3, 50+f*2) if f not in (5, 6) else None for f in range(12)])
        original = copy.deepcopy(rows)
        balls = observed_balls(rows)
        samples, gaps = fill_short_gaps(rows, balls, 30)
        self.assertEqual(gaps[0]["reason"], "consistent_short_gap")
        self.assertEqual(samples[5]["pixel"], [65, 60])
        self.assertEqual(samples[5]["status"], "estimated")
        self.assertIsNone(samples[5]["observation"])
        self.assertEqual(samples[5]["estimate"]["endpoints"], [4, 7])
        self.assertEqual(rows, original)
        self.assertIsNone(balls[5])
        for f, sample in enumerate(samples):
            if balls[f]:
                self.assertEqual(sample["pixel"], balls[f]["pixel"])

    def test_dont_fill_long_boundary_impact_equipment_or_cut_gaps(self):
        rows = rows_for([(50+f*3, 50+f*2) if f != 5 else None for f in range(12)])
        balls = observed_balls(rows)
        _, gaps = fill_short_gaps(rows, balls, 30, events=[{"frame_range": [5, 5]}])
        self.assertEqual(gaps[0]["reason"], "event_neighbourhood")
        _, gaps = fill_short_gaps(rows, balls, 30, cuts=[5])
        self.assertEqual(gaps[0]["reason"], "scene_cut")
        rows[4]["tracks"] += [dict(label="player", track_id=7, identity_id="near", bbox=[30, 20, 80, 100]),
                              dict(label="racket", player_track_id=7, bbox=[55, 50, 70, 65])]
        self.assertEqual(fill_short_gaps(rows, balls, 30)[1][0]["reason"], "near_player_equipment")
        long = rows_for([None]*5+[(50+f*3, 50+f*2) for f in range(8)]+[None]*2)
        samples, gaps = fill_short_gaps(long, observed_balls(long), 30)
        self.assertFalse(any(s["status"] == "estimated" for s in samples))
        self.assertEqual(gaps[0]["reason"], "long_gap")
        self.assertEqual(gaps[1]["reason"], "insufficient_observed_context")

    def test_dont_bridge_direction_reversal(self):
        rows = rows_for([(50+f*3, 50+f*2) for f in range(12)])
        rows[5]["tracks"] = []
        rows[6]["tracks"] = [ball(65, 60)]
        rows[7]["tracks"] = [ball(62, 58)]
        samples, gaps = fill_short_gaps(rows, observed_balls(rows), 30)
        self.assertEqual(samples[5]["status"], "missing")
        self.assertEqual(gaps[0]["reason"], "possible_turn_or_impact")

    def test_strict_cache_and_source_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run/"summary.json").write_text(json.dumps({"fps": 30, "frames": 2}))
            (run/"events.jsonl").write_text('\n'.join(json.dumps(r) for r in rows_for([(1, 1), (2, 2)])))
            with self.assertRaisesRegex(ValueError, "offset"):
                load_run(run)
            (run/"source-offset.json").write_text(json.dumps({"source_start_frame": 4740, "source_start_seconds": 158}))
            self.assertEqual(load_run(run)[1], 4740)
            (run/"events.jsonl").write_text(json.dumps({"frame": 0, "time_s": 0})+'\n'+json.dumps({"frame": 2, "time_s": 1/30}))
            with self.assertRaisesRegex(ValueError, "alignment"):
                load_run(run)


if __name__ == "__main__":
    unittest.main()
