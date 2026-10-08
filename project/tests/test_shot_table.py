"""Per-shot / per-point table: provenance on every field, no invented tennis facts."""
import copy
import csv
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from tennis_vision import shot_table as st
from tennis_vision.labels import LabelsError

FPS = 30.0
RAW_PLAY = Path(__file__).resolve().parent.parent / "runs" / "raw-play-ready"
# Labels bind to the replay's review_run_id, which tennis_vision.labels requires to be 64 hex characters.
RUN_ID = "ab" * 32


def player(identity, x, y, basis="two_ankles_ground", estimated=False, predicted=False):
    """x, y in court metres (near-left origin); stored as centred raw_feet_xyz_m like the replay."""
    raw = [x - 5.485, y - 11.885, 0.0]
    return {"identity_id": identity, "raw_feet_xyz_m": raw, "feet_xyz_m": raw, "basis": basis,
            "predicted": predicted, "identity_association_estimated": estimated,
            "identity_basis": "backward_cached_association_from_confirmed_track" if estimated else "original_confirmed_track",
            # Render-only 2.5D geometry that must never be read.
            "joints_m": {"left_ankle": [99.0, 99.0, 0.0]}, "avatar": "pose_wireframe"}


def hit(eid, frame, who, status="unreviewed"):
    return {"id": eid, "type": "hit", "status": status, "frame": frame, "time_s": frame / FPS, "player_id": who,
            "pixel": [1, 1], "pixel_basis": "observed_ball_at_reviewed_frame", "court_m": None,
            "candidate_court_m": None, "geometry_issue": None}


def bounce(eid, frame, status, court=None, candidate=None, basis="observed_ball_at_reviewed_frame"):
    return {"id": eid, "type": "bounce", "status": status, "frame": frame, "time_s": frame / FPS, "player_id": None,
            "pixel": [1, 1], "pixel_basis": basis, "court_m": court, "candidate_court_m": candidate,
            "geometry_issue": None}


def shot(eid, frame, who, state="shot_candidate", classification="unknown", status="candidate"):
    return {"event_id": eid, "frame": frame, "time_s": frame / FPS, "event_status": "unreviewed", "player_id": who,
            "classification": classification, "classification_status": status, "support": "moderate",
            "reasons": ["synthetic reason"], "action_state": state, "contact_support": "limited",
            "play_assessment": {"action_state": state, "reason": "synthetic assessment"}}


def gravity_fit(start_id, bounce_id, a, b, velocity):
    """Trajectory with a known initial velocity (m/s), sampled at FPS."""
    points = []
    for f in range(a, b + 1):
        t = (f - a) / FPS
        points.append({"frame": f, "point_m": [velocity[0] * t, velocity[1] * t, 1.0 + velocity[2] * t - 4.905 * t * t]})
    return {"start_frame": a, "end_frame": b, "airborne_xyz_status": "estimated_not_measured",
            "start_event_id": start_id, "bounce_event_id": bounce_id, "trajectory_m": points,
            "source": "gravity_fit_reviewed_bounce"}


def make_replay(folder, calibration_status="reviewed", mutate_frame=None):
    frames = []
    for f in range(100):
        record = {"frame": f, "players": [player("near", 2.0, 0.9), player("far", 8.0, 23.0)],
                  "ball_3d": None, "ball_3d_preview": {"point_m": [50, 50, 50], "status": "unvalidated_preview"}}
        if mutate_frame:
            mutate_frame(f, record)
        frames.append(record)
    events = [hit("h10", 10, "near"), bounce("b30", 30, "confirmed", court=[7.0, 20.0]),
              hit("h50", 50, "far"), bounce("b60", 60, "unreviewed", candidate=[3.0, 4.0]),
              bounce("b70", 70, "confirmed", court=[4.0, 2.0]), hit("w80", 80, "far")]
    shots = {"shots": [shot("h10", 10, "near", classification="forehand"), shot("h50", 50, "far"),
                       shot("w80", 80, "far", state="uncertain_contact")]}
    flights = {"status": "ok", "fits": [gravity_fit("h10", "b30", 10, 30, [3.0, 20.0, 2.0])], "skipped": []}
    corrections = {"schema_version": 1, "kind": "court_and_bounce_review", "run_id": RUN_ID,
                   "calibration_status": calibration_status, "landmark_source": "synthetic", "landmarks": {},
                   "bounce_edits": []}
    report = {"schema_version": 1, "package_id": "pkg-x", "review_run_id": RUN_ID, "frames": 100, "fps": FPS,
              "source_start_frame": 1000,
              "camera_fit": {"basis": "multi_point_radial_camera", "review_status": calibration_status,
                             "status": "approximate_not_measurement_grade", "landmark_rmse_px": 1.5},
              "calibration_review": corrections}
    data = {"report": report, "frames": frames, "events": events, "shots": shots, "flights": flights,
            "preview_flights": {"fits": [gravity_fit("h50", "b60", 50, 60, [40.0, 40.0, 0.0])]}}
    folder.mkdir(parents=True)
    (folder / "replay-data.json").write_text(json.dumps(data))
    (folder / "build-status.json").write_text(json.dumps({"status": "complete"}))
    (folder / "reviewed-events.json").write_text(json.dumps({"run_id": RUN_ID, "events": events}))
    (folder / "shot-candidates.json").write_text(json.dumps(shots))
    (folder / "validated-flight3d.json").write_text(json.dumps(flights))
    (folder / "calibration-corrections.json").write_text(json.dumps(corrections))
    return folder


OFFSET = 1000  # synthetic source_start_frame; labels use SOURCE frames (docs/labels-schema.md, #27)


def label_shot(lid, frame, hitter, shot_type):
    return {"id": lid, "source_frame": frame + OFFSET, "hitter": hitter, "shot_type": shot_type, "decision": "human"}


def label_bounce(lid, frame, call):
    return {"id": lid, "source_frame": frame + OFFSET, "call": call, "decision": "human"}


def label_point(lid, start, end, server="near", winner="unknown", **extra):
    return dict({"id": lid, "source_start_frame": start + OFFSET, "source_end_frame": end + OFFSET,
                 "server": server, "winner": winner, "decision": "human"}, **extra)


def span(start, end, *kinds):
    return {"source_start_frame": start + OFFSET, "source_end_frame": end + OFFSET, "kinds": list(kinds)}


def write_labels(path, binding=None, **overrides):
    labels = {"schema_version": 1, "kind": "tennis_ground_truth_labels",
              "binding": binding or {"run_id": RUN_ID, "source_video_sha256": "0" * 64, "fps": FPS},
              "labeller": {"name": "test", "date": "2026-09-28"},
              "shot_types": ["serve", "forehand", "backhand", "volley", "overhead", "other", "unsure"],
              "coverage": [], "shots": [], "bounces": [], "points": []}
    labels.update(overrides)
    path.write_text(json.dumps(labels))
    return path


class ShotTableTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, labels=None, name="out", **kw):
        replay = kw.pop("replay", None) or make_replay(self.root / ("replay-" + name), **kw)
        return st.run(replay, self.root / name, labels), self.root / name

    def rows(self, payload):
        return {r["shot_id"]: r for r in payload["shots"]}

    # ---------------------------------------------------------------- contract

    def test_writes_three_files_and_every_field_has_a_valid_source(self):
        payload, out = self.build()
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["points.csv", "shot-table.json", "shots.csv"])
        with (out / "shots.csv").open(encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, st.SHOT_COLUMNS)
            rows = list(reader)
        self.assertEqual(len(rows), 3)
        with (out / "points.csv").open(encoding="utf-8") as stream:
            self.assertEqual(csv.DictReader(stream).fieldnames, st.POINT_COLUMNS)
        for fields, keys, cols in ((st.SHOT_FIELDS, st.SHOT_KEYS, st.SHOT_COLUMNS),
                                   (st.POINT_FIELDS, st.POINT_KEYS, st.POINT_COLUMNS)):
            owned = set(keys)
            for name, values in fields:
                owned |= set(values) | {name + "_source", name + "_basis"}
            self.assertEqual(owned, set(cols), "every value column belongs to a field with a _source column")
        for row in rows:
            for name, values in st.SHOT_FIELDS:
                self.assertIn(row[name + "_source"], st.SOURCES)
                if row[name + "_source"] == "unknown":
                    self.assertTrue(all(row[v] == "" for v in values), (name, row))
        saved = json.loads((out / "shot-table.json").read_text())
        self.assertEqual(saved["columns"]["shots"], st.SHOT_COLUMNS)
        self.assertEqual(set(saved["replay"]["input_sha256"]),
                         {"replay-data.json", "reviewed-events.json", "shot-candidates.json",
                          "validated-flight3d.json", "calibration-corrections.json"})

    def test_no_spin_or_rpm_columns(self):
        for column in st.SHOT_COLUMNS + st.POINT_COLUMNS:
            self.assertNotIn("spin", column)
            self.assertNotIn("rpm", column)

    # ---------------------------------------------------------------- shots

    def test_candidate_row_provenance_with_reviewed_calibration(self):
        payload, _ = self.build()
        row = self.rows(payload)["h10"]
        self.assertEqual((row["contact_frame"], row["contact_source_frame"], row["contact_source"]), (10, 1010, "estimated"))
        self.assertEqual((row["hitter"], row["hitter_source"]), ("near", "estimated"))
        self.assertEqual((row["shot_type"], row["shot_type_source"]), ("forehand", "estimated"))
        self.assertTrue(row["shot_type_basis"].startswith("rule candidate"))
        self.assertEqual((row["hitter_position_x_m"], row["hitter_position_y_m"], row["hitter_position_source"]),
                         (2.0, 0.9, "observed"))
        self.assertIn("reviewed calibration", row["hitter_position_basis"])
        self.assertEqual((row["opponent_position_x_m"], row["opponent_position_y_m"]), (8.0, 23.0))
        # First bounce after the contact and before the next contact is confirmed.
        self.assertEqual((row["landing_x_m"], row["landing_y_m"], row["landing_source"]), (7.0, 20.0, "observed"))
        self.assertIn("b30", row["landing_basis"])
        # Hitter left of centre, landing right of the middle third: cross court; 8.1 m past the net: deep.
        self.assertEqual((row["direction"], row["direction_source"]), ("cross_court", "estimated"))
        self.assertEqual((row["depth"], row["depth_source"]), ("deep", "estimated"))
        self.assertEqual(row["ball_speed_source"], "estimated")
        self.assertAlmostEqual(row["ball_speed_kmh"], round((3 ** 2 + 20 ** 2 + 2 ** 2) ** .5 * 3.6, 1), delta=.2)
        self.assertEqual((row["point_id"], row["point_source"]), (None, "unknown"))
        self.assertEqual(row["rally_shot_index_source"], "unknown")

    def test_unreviewed_first_bounce_blocks_later_confirmed_landing(self):
        payload, _ = self.build()
        row = self.rows(payload)["h50"]
        self.assertEqual(row["landing_source"], "unknown")
        self.assertIsNone(row["landing_x_m"])
        self.assertIn("b60", row["landing_basis"])
        self.assertIn("b70", row["landing_basis"])
        self.assertEqual((row["direction_source"], row["depth_source"]), ("unknown", "unknown"))
        self.assertEqual((row["shot_type"], row["shot_type_source"]), (None, "unknown"))
        self.assertTrue(row["shot_type_basis"].startswith("abstained"))
        # Speed only from validated flights, never from unvalidated previews.
        self.assertEqual(row["ball_speed_source"], "unknown")

    def test_uncertain_contact_is_not_a_shot(self):
        payload, _ = self.build()
        row = self.rows(payload)["w80"]
        self.assertEqual((row["shot_status"], row["shot_status_source"]), ("uncertain_contact", "estimated"))
        self.assertEqual(row["shot_type_source"], "unknown")
        self.assertEqual(row["landing_source"], "unknown")
        self.assertIn("uncertain_contact", row["landing_basis"])

    def test_landing_is_not_attributed_across_an_intervening_contact(self):
        replay = make_replay(self.root / "r")
        data = json.loads((replay / "replay-data.json").read_text())
        events = data["events"] + [hit("x20", 20, "far")]
        (replay / "reviewed-events.json").write_text(json.dumps({"run_id": RUN_ID, "events": events}))
        payload = st.run(replay, self.root / "o")
        row = self.rows(payload)["h10"]
        self.assertEqual(row["landing_source"], "unknown")
        self.assertIn("x20", row["landing_basis"])

    def test_draft_calibration_makes_positions_and_landings_estimated(self):
        payload, _ = self.build(calibration_status="draft")
        self.assertEqual(payload["calibration"]["status"], "draft")
        row = self.rows(payload)["h10"]
        self.assertEqual(row["hitter_position_source"], "estimated")
        self.assertIn("calibration is draft", row["hitter_position_basis"])
        self.assertEqual(row["landing_source"], "estimated")
        self.assertEqual(row["landing_x_m"], 7.0)

    def test_missing_calibration_review_is_not_observed(self):
        replay = make_replay(self.root / "r")
        (replay / "calibration-corrections.json").unlink()
        data = json.loads((replay / "replay-data.json").read_text())
        data["report"]["calibration_review"] = None
        (replay / "replay-data.json").write_text(json.dumps(data))
        payload = st.run(replay, self.root / "o")
        self.assertFalse(payload["calibration"]["reviewed"])
        self.assertEqual(self.rows(payload)["h10"]["hitter_position_source"], "estimated")

    def test_fallback_estimated_identity_and_predicted_positions(self):
        def mutate(f, record):
            if f == 10:
                record["players"] = [player("near", 2.0, 0.9, basis="bbox_ground_fallback"),
                                     player("far", 8.0, 23.0, estimated=True)]
            if f == 50:
                record["players"] = [player("far", 8.0, 23.0, predicted=True)]
        payload, _ = self.build(mutate_frame=mutate)
        rows = self.rows(payload)
        self.assertEqual(rows["h10"]["hitter_position_source"], "estimated")
        self.assertIn("bbox_ground_fallback", rows["h10"]["hitter_position_basis"])
        self.assertEqual(rows["h10"]["opponent_position_source"], "estimated")
        self.assertIn("identity is estimated", rows["h10"]["opponent_position_basis"])
        self.assertEqual(rows["h50"]["hitter_position_source"], "unknown")
        self.assertIn("predicted", rows["h50"]["hitter_position_basis"])
        self.assertEqual(rows["h50"]["opponent_position_source"], "unknown")  # no near record at frame 50

    def test_render_only_geometry_is_never_used(self):
        base, _ = self.build(name="a")

        def mutate(f, record):
            for p in record["players"]:
                p["joints_m"] = {"left_ankle": [-50, -50, 0]}
                p["feet_xyz_m"] = [40.0, 40.0, 0.0]  # display-smoothed root, not read
                p["avatar"] = "height_placeholder"
            record["ball_3d_preview"] = {"point_m": [1, 2, 3]}
            record["racquets"] = [{"tip_m": [0, 0, 0]}]
        other, _ = self.build(name="b", mutate_frame=mutate)
        self.assertEqual(base["shots"], other["shots"])

    # ---------------------------------------------------------------- labels and points

    def test_labels_confirm_shots_points_and_rally_index(self):
        labels = write_labels(self.root / "labels.json",
                              shots=[label_shot("s1", 12, "near", "backhand"), label_shot("s2", 90, "far", "volley")],
                              points=[label_point("p1", 5, 95, winner="far", score_text="15-0")])
        before = hashlib.sha256(labels.read_bytes()).hexdigest()
        payload, out = self.build(labels)
        self.assertEqual(hashlib.sha256(labels.read_bytes()).hexdigest(), before, "labels are never modified")
        rows = self.rows(payload)
        h10 = rows["h10"]
        # Label source frame 1012 becomes replay frame 12.
        self.assertEqual((h10["contact_frame"], h10["contact_source_frame"], h10["contact_source"]),
                         (12, 1012, "human_confirmed"))
        self.assertEqual((h10["hitter"], h10["hitter_source"]), ("near", "human_confirmed"))
        self.assertEqual((h10["shot_status"], h10["shot_status_source"]), ("labelled_shot", "human_confirmed"))
        self.assertEqual((h10["shot_type"], h10["shot_type_source"], h10["shot_type_basis"]),
                         ("backhand", "human_confirmed", "labelled"))
        label_only = rows["label:s2"]
        self.assertIsNone(label_only["candidate_event_id"])
        self.assertEqual((label_only["hitter"], label_only["shot_type"]), ("far", "volley"))
        self.assertEqual((h10["rally_shot_index"], h10["rally_shot_index_source"]), (1, "human_confirmed"))
        self.assertEqual((label_only["rally_shot_index"], label_only["rally_shot_index_source"]), (2, "human_confirmed"))
        self.assertEqual(rows["h50"]["point_id"], "p1")
        self.assertEqual(rows["h50"]["point_source"], "estimated")  # candidate contact frame inside the span
        self.assertEqual(rows["h50"]["rally_shot_index_source"], "unknown")
        self.assertIn("outside labelled shot coverage", rows["h50"]["shot_status_basis"])
        (point,) = payload["points"]
        self.assertEqual((point["point_id"], point["start_frame"], point["start_source_frame"], point["end_frame"]),
                         ("p1", 5, 1005, 95))
        self.assertEqual((point["server"], point["server_source"]), ("near", "human_confirmed"))
        self.assertEqual((point["shot_count"], point["shot_count_source"]), (2, "human_confirmed"))
        self.assertEqual((point["winner"], point["winner_source"]), ("far", "human_confirmed"))
        self.assertEqual(point["score_text"], "15-0")
        with (out / "points.csv").open(encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 1)
        self.assertEqual(payload["labels"]["run_binding"], "review_run_id")
        self.assertTrue(payload["labels"]["video_check"].startswith("not_checked"))

    def test_point_without_labelled_shots_counts_candidates_as_estimated_and_never_infers_winner(self):
        labels = write_labels(self.root / "labels.json", points=[label_point("p1", 5, 95, server="unknown")])
        payload, _ = self.build(labels)
        (point,) = payload["points"]
        self.assertEqual((point["shot_count"], point["shot_count_source"]), (2, "estimated"))
        self.assertEqual((point["winner"], point["winner_source"]), (None, "unknown"))
        self.assertEqual((point["server"], point["server_source"]), (None, "unknown"))
        rows = self.rows(payload)
        self.assertEqual((rows["h10"]["rally_shot_index"], rows["h10"]["rally_shot_index_source"]), (1, "estimated"))
        self.assertEqual((rows["h50"]["rally_shot_index"], rows["h50"]["rally_shot_index_source"]), (2, "estimated"))
        self.assertEqual(rows["w80"]["rally_shot_index_source"], "unknown")

    def test_shot_coverage_turns_unlabelled_candidates_into_human_no_shot(self):
        labels = write_labels(self.root / "labels.json", coverage=[span(0, 99, "shots")],
                              shots=[label_shot("s1", 10, "near", "unsure")])
        rows = self.rows(self.build(labels)[0])
        self.assertEqual((rows["h10"]["shot_type"], rows["h10"]["shot_type_source"]), (None, "unknown"))
        self.assertIn("unsure", rows["h10"]["shot_type_basis"])
        for eid in ("h50", "w80"):
            self.assertEqual((rows[eid]["shot_status"], rows[eid]["shot_status_source"]),
                             ("no_labelled_shot", "human_confirmed"))
            self.assertEqual(rows[eid]["landing_source"], "unknown")

    def test_bounce_coverage_drops_unlabelled_bounce_candidates(self):
        # John fully labelled bounces and labelled none: candidate b60 is "no bounce here",
        # so h50's first bounce is the confirmed b70.
        labels = write_labels(self.root / "labels.json", coverage=[span(0, 99, "bounces")])
        row = self.rows(self.build(labels)[0])["h50"]
        self.assertEqual((row["landing_x_m"], row["landing_y_m"], row["landing_source"]), (4.0, 2.0, "observed"))
        self.assertEqual((row["direction"], row["depth"]), ("cross_court", "deep"))

    def test_stroke_review_type_is_human_confirmed_and_not_a_shot_is_respected(self):
        replay = make_replay(self.root / "r")
        shots = json.loads((replay / "shot-candidates.json").read_text())
        shots["shots"][0].update(classification="backhand", classification_status="human_reviewed")
        shots["shots"][1].update(classification="not_a_shot", classification_status="human_reviewed")
        (replay / "shot-candidates.json").write_text(json.dumps(shots))
        rows = self.rows(st.run(replay, self.root / "o"))
        self.assertEqual((rows["h10"]["shot_type"], rows["h10"]["shot_type_source"]), ("backhand", "human_confirmed"))
        self.assertEqual((rows["h50"]["shot_status"], rows["h50"]["shot_status_source"]), ("not_a_shot", "human_confirmed"))
        self.assertEqual(rows["h50"]["shot_type_source"], "unknown")

    def test_frame_tolerance_is_respected(self):
        labels = write_labels(self.root / "labels.json", shots=[label_shot("s1", 14, "near", "serve")])
        payload, _ = self.build(labels)
        rows = self.rows(payload)
        self.assertEqual(rows["h10"]["shot_status"], "shot_candidate")
        self.assertIn("label:s1", rows)
        wider = st.run(make_replay(self.root / "r2"), self.root / "o2", labels, frame_tolerance=4)
        self.assertEqual(self.rows(wider)["h10"]["shot_status"], "labelled_shot")

    def test_labelled_bounce_gives_estimated_landing_and_human_call(self):
        labels = write_labels(self.root / "labels.json", bounces=[label_bounce("b1", 61, "out")])
        payload, _ = self.build(labels)
        row = self.rows(payload)["h50"]
        self.assertEqual((row["landing_x_m"], row["landing_y_m"], row["landing_source"]), (3.0, 4.0, "estimated"))
        self.assertIn("unreviewed ball projection", row["landing_basis"])
        self.assertEqual((row["landing_call"], row["landing_call_source"]), ("out", "human_confirmed"))

    def test_labels_outside_the_replay_are_counted_not_used(self):
        labels = write_labels(self.root / "labels.json", shots=[label_shot("early", -500, "near", "serve")],
                              points=[label_point("late", 90, 400)])
        payload, _ = self.build(labels)
        self.assertEqual(payload["labels"]["counts_outside_replay_frames"], {"shots": 1, "bounces": 0, "points": 1})
        self.assertEqual(payload["points"], [])
        self.assertEqual(len(payload["shots"]), 3)

    def test_video_hash_is_checked_when_the_original_is_given(self):
        video = self.root / "original.bin"
        video.write_bytes(b"not really a video")
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        good = write_labels(self.root / "good.json", binding={"run_id": RUN_ID, "source_video_sha256": digest, "fps": FPS})
        replay = make_replay(self.root / "r")
        self.assertEqual(st.run(replay, self.root / "o1", good, video=video)["labels"]["video_check"], "matched")
        bad = write_labels(self.root / "bad.json")
        with self.assertRaisesRegex(LabelsError, "bound to source video"):
            st.run(replay, self.root / "o2", bad, video=video)

    def test_invalid_or_foreign_labels_are_rejected(self):
        # Messages are tennis_vision.labels' own (validate_labels / check_binding).
        replay = make_replay(self.root / "r")
        cases = {
            "bound to run": write_labels(self.root / "a.json", binding={"run_id": "cd" * 32, "source_video_sha256": "0" * 64,
                                                                       "fps": FPS}),
            "25 fps but the run is 30": write_labels(self.root / "b.json", binding={"run_id": RUN_ID,
                                                                                     "source_video_sha256": "0" * 64, "fps": 25}),
            "overlaps point": write_labels(self.root / "c.json", points=[label_point("p1", 1, 20), label_point("p2", 20, 30)]),
            "winner: 'draw'": write_labels(self.root / "d.json", points=[label_point("p1", 1, 20, winner="draw")]),
            "decision: must be 'human'": write_labels(self.root / "e.json",
                                                      shots=[dict(label_shot("s1", 5, "near", "serve"), decision="model")]),
            "unknown field": write_labels(self.root / "f.json", spin_rpm=[]),
            "64 lowercase hex": write_labels(self.root / "g.json", binding={"run_id": "run-x", "source_video_sha256": "0" * 64,
                                                                           "fps": FPS}),
            "shot_types: must be exactly": write_labels(self.root / "h.json", shot_types=["serve", "forehand"]),
            "overlaps another shots coverage": write_labels(self.root / "i.json",
                                                            coverage=[span(0, 50, "shots"), span(40, 60, "shots")]),
        }
        for n, (message, path) in enumerate(cases.items()):
            with self.subTest(message):
                with self.assertRaisesRegex(LabelsError, message):
                    st.run(replay, self.root / f"o{n}", path)
                self.assertFalse((self.root / f"o{n}").exists())

    # ---------------------------------------------------------------- safety and zones

    def test_output_must_be_new_and_outside_the_replay_folder(self):
        replay = make_replay(self.root / "r")
        with self.assertRaisesRegex(ValueError, "outside the replay folder"):
            st.run(replay, replay / "table")
        st.run(replay, self.root / "o")
        with self.assertRaisesRegex(ValueError, "already exists"):
            st.run(replay, self.root / "o")

    def test_zone_defaults(self):
        self.assertEqual(st.depth_zone("near", 12.5, "thirds"), "short")
        self.assertEqual(st.depth_zone("far", 5.0, "thirds"), "mid")
        self.assertEqual(st.depth_zone("far", -0.5, "thirds"), "beyond_baseline")
        self.assertEqual(st.depth_zone("near", 10.0, "thirds"), "hitter_side_of_net")
        self.assertEqual(st.depth_zone("near", 17.0, "service_line"), "service_box")
        self.assertEqual(st.depth_zone("near", 20.0, "service_line"), "back_court")
        self.assertEqual(st.direction_zone(9.0, 9.5), ("down_the_line", None))
        self.assertEqual(st.direction_zone(9.0, 2.0), ("cross_court", None))
        self.assertEqual(st.direction_zone(2.0, 5.9), ("middle", None))
        self.assertIsNone(st.direction_zone(5.5, 9.0)[0])

    def test_cli(self):
        replay = make_replay(self.root / "r")
        with redirect_stdout(io.StringIO()) as printed:
            st.main(["--replay", str(replay), "--output", str(self.root / "cli"), "--depth-scheme", "service_line"])
        self.assertEqual(json.loads(printed.getvalue())["shots"], 3)
        saved = json.loads((self.root / "cli" / "shot-table.json").read_text())
        self.assertEqual(saved["conventions"]["depth_scheme"]["name"], "service_line")
        self.assertEqual({r["shot_id"]: r["depth"] for r in saved["shots"]}["h10"], "back_court")

    @unittest.skipUnless((RAW_PLAY / "replay-data.json").is_file(), "raw-play-ready replay not present")
    def test_raw_play_ready_demo_is_conservative(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW_PLAY.iterdir() if p.is_file()}
        payload = st.run(RAW_PLAY, self.root / "demo")
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW_PLAY.iterdir() if p.is_file()}
        self.assertEqual(before, after, "the reviewed replay folder is never modified")
        self.assertEqual(len(payload["shots"]), 11)
        self.assertEqual(payload["points"], [])
        self.assertEqual(payload["calibration"]["status"], "reviewed")
        for row in payload["shots"]:
            self.assertIn(row["landing_source"], ("unknown", "estimated"))
            self.assertEqual(row["point_source"], "unknown")
            self.assertNotEqual(row["contact_source"], "human_confirmed")
        serve = self.rows(payload)["hit_candidate-000057"]
        self.assertEqual((serve["shot_type"], serve["shot_type_source"]), ("serve", "estimated"))


if __name__ == "__main__":
    unittest.main()
