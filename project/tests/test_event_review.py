import contextlib
import copy
import io
import json
import re
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tennis_vision.event_analysis import observed_balls, detect_events, fill_short_gaps, proximity
from tennis_vision import event_review
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


# Fake FFmpeg outputs (no real binary is run). Formats follow real builds.
VERSION_71 = "ffmpeg version 7.1-full_build-www.gyan.dev Copyright (c) 2000-2024 the FFmpeg developers\n"
VERSION_50 = "ffmpeg version 5.0.1-essentials_build-www.gyan.dev Copyright (c) 2000-2022 the FFmpeg developers\n"
VERSION_GIT = "ffmpeg version N-112233-gabcdef0123 Copyright (c) 2000-2024 the FFmpeg developers\n"
ENCODERS_X264 = ("Encoders:\n V..... = Video\n ------\n V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC (codec h264)\n"
                 " V....D libx264rgb           libx264 H.264 RGB (codec h264)\n V....D mpeg4                MPEG-4 part 2\n")
ENCODERS_NO_X264 = " V....D libx264rgb           libx264 H.264 RGB (codec h264)\n V....D mpeg4                MPEG-4 part 2\n"
HELP_51 = ("Advanced Video options:\n-vsync                           set video sync method globally; deprecated, use -fps_mode\n"
           "-fps_mode[:<stream_spec>] <mode>  set framerate mode for matching video streams; overrides vsync\n")
HELP_50 = "Advanced Video options:\n-vsync              video sync method\n-vf filter_graph    set video filters\n"


def fake_ffmpeg(version, encoders=ENCODERS_X264, help_text=HELP_51, calls=None):
    def run(cmd, **kwargs):
        if calls is not None:
            calls.append(cmd)
        out = {"-version": version, "-encoders": encoders, "-h": help_text}[cmd[2]]
        return subprocess.CompletedProcess(cmd, 0, out.encode(), b"")
    return run


class FfmpegCapabilityTests(unittest.TestCase):
    def check(self, *outputs, which="/opt/ffmpeg/bin/ffmpeg", **kwargs):
        with mock.patch.object(event_review.shutil, "which", return_value=which), \
             mock.patch.object(event_review.subprocess, "run", side_effect=fake_ffmpeg(*outputs, **kwargs)):
            return event_review.check_ffmpeg("ffmpeg")

    def test_version_parsing(self):
        self.assertEqual(event_review.ffmpeg_version(VERSION_71), "7.1")
        self.assertEqual(event_review.ffmpeg_version(VERSION_50), "5.0.1")
        self.assertEqual(event_review.ffmpeg_version("ffmpeg version n5.1.2 Copyright"), "5.1.2")
        self.assertEqual(event_review.ffmpeg_version(VERSION_GIT), "N-112233-gabcdef0123")
        self.assertIsNone(event_review.ffmpeg_version("not ffmpeg"))

    def test_capable_ffmpeg_passes(self):
        calls = []
        result = self.check(VERSION_71, calls=calls)
        self.assertTrue(result["ok"])
        self.assertEqual((result["path"], result["version"], result["missing"]), ("/opt/ffmpeg/bin/ffmpeg", "7.1", []))
        self.assertIsNone(result["reason"])
        self.assertEqual([c[0] for c in calls], ["/opt/ffmpeg/bin/ffmpeg"]*3)

    def test_ffmpeg_50_names_found_and_needed_version(self):
        result = self.check(VERSION_50, help_text=HELP_50)
        self.assertFalse(result["ok"])
        self.assertEqual(result["missing"], ["-fps_mode option"])
        self.assertIn("version 5.0.1", result["reason"])
        self.assertIn("5.1+", result["reason"])
        self.assertIn("-fps_mode", result["reason"])

    def test_missing_libx264_fails_even_if_only_libx264rgb(self):
        result = self.check(VERSION_71, encoders=ENCODERS_NO_X264)
        self.assertFalse(result["ok"])
        self.assertEqual(result["missing"], ["libx264 encoder"])
        result = self.check(VERSION_50, encoders=ENCODERS_NO_X264, help_text=HELP_50)
        self.assertEqual(result["missing"], ["-fps_mode option", "libx264 encoder"])

    def test_git_build_is_judged_by_its_options_not_its_version_string(self):
        self.assertTrue(self.check(VERSION_GIT)["ok"])
        self.assertIn("N-112233", self.check(VERSION_GIT, help_text=HELP_50)["reason"])

    def test_missing_or_broken_binary(self):
        with mock.patch.object(event_review.shutil, "which", return_value=None):
            result = event_review.check_ffmpeg("C:/nope/ffmpeg.exe")
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["reason"])
        with mock.patch.object(event_review.shutil, "which", return_value="/x/ffmpeg"), \
             mock.patch.object(event_review.subprocess, "run", side_effect=OSError("bad exe")):
            self.assertIn("bad exe", event_review.check_ffmpeg("/x/ffmpeg")["reason"])
        self.assertFalse(self.check("garbage")["ok"])

    def test_require_raises_value_error_with_versions(self):
        with mock.patch.object(event_review.shutil, "which", return_value="/f"), \
             mock.patch.object(event_review.subprocess, "run", side_effect=fake_ffmpeg(VERSION_50, help_text=HELP_50)):
            with self.assertRaisesRegex(ValueError, r"version 5\.0\.1.*5\.1\+"):
                event_review.require_ffmpeg("/f")
        with mock.patch.object(event_review.shutil, "which", return_value="/f"), \
             mock.patch.object(event_review.subprocess, "run", side_effect=fake_ffmpeg(VERSION_71)):
            self.assertEqual(event_review.require_ffmpeg("ffmpeg"), "/f")

    def test_build_review_checks_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            run, output = Path(temp)/"run", Path(temp)/"review"
            run.mkdir()
            (run/"summary.json").write_text(json.dumps({"fps": 30, "frames": 2, "input": "x.mp4"}))
            (run/"source-offset.json").write_text(json.dumps({"source_start_frame": 0, "source_start_seconds": 0}))
            (run/"events.jsonl").write_text('\n'.join(json.dumps(r) for r in rows_for([(1, 1), (2, 2)])))
            with mock.patch.object(event_review.shutil, "which", return_value="/f"), \
                 mock.patch.object(event_review.subprocess, "run", side_effect=fake_ffmpeg(VERSION_50, help_text=HELP_50)), \
                 mock.patch.object(event_review, "detect_cuts", side_effect=AssertionError("ran past the check")):
                with self.assertRaisesRegex(ValueError, "5.0.1"):
                    event_review.build_review(run, output, ffmpeg="ffmpeg")
            self.assertFalse(output.exists())

    def test_check_cli_prints_one_json_line_and_exit_code(self):
        for version, help_text, code in ((VERSION_71, HELP_51, 0), (VERSION_50, HELP_50, 1)):
            stdout = io.StringIO()
            with mock.patch.object(event_review.shutil, "which", return_value="/f"), \
                 mock.patch.object(event_review.subprocess, "run", side_effect=fake_ffmpeg(version, help_text=help_text)), \
                 contextlib.redirect_stdout(stdout):
                self.assertEqual(event_review.main(["--check-ffmpeg", "/f"]), code)
            lines = stdout.getvalue().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["ok"], code == 0)

    def test_render_commands_use_only_checked_new_options(self):
        # A new post-4.x option or another encoder in the render commands must be added to the check.
        source = Path(event_review.__file__).read_text(encoding="utf-8")
        self.assertIn('"-fps_mode"', source)
        for option in event_review.FFMPEG_REQUIRED_OPTIONS:
            self.assertIn(f'"{option}"', source)
        self.assertEqual(set(re.findall(r'"-c:v", "([^"]+)"', source)), set(event_review.FFMPEG_REQUIRED_ENCODERS))


if __name__ == "__main__":
    unittest.main()
