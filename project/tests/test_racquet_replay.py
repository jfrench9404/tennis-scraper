"""Racquet display records: per-frame hands, identity-only association, bounded staleness."""
import unittest

import numpy as np

from tennis_vision.camera3d import from_court
from tennis_vision.court import CourtMapper
from tennis_vision.racquet_replay import _plane_point, build_racquets, hand_for

# Camera from the reviewed baseline court corners (court.yaml).
COURT = CourtMapper(np.array([[3, 535], [1188, 570], [782, 133], [427, 121]], np.float32))
CAMERA = from_court(COURT, 1280, 720)
FEET = COURT.project((640, 560))
LEFT_PX, RIGHT_PX = (610, 420), (670, 420)


def player(track_id, identity, left=None, right=None, box=(600, 300, 680, 560)):
    joints = {}
    if left is not None:
        joints["left_wrist"] = [*left, .9]
    if right is not None:
        joints["right_wrist"] = [*right, .9]
    return {"label": "player", "identity_id": identity, "track_id": track_id, "bbox": list(box), "keypoints": joints}


def racquet(owner_track, box, confidence=.8):
    return {"label": "racket", "bbox": list(box), "confidence": confidence, "player_track_id": owner_track}


def record(identity, left=True, right=True):
    base = [FEET[0] - 5.485, FEET[1] - 11.885, 0.]
    rec = {"identity_id": identity, "predicted": False, "raw_feet_xyz_m": base, "feet_xyz_m": base, "joints_m": {}}
    # 3D wrists lifted from the same pixels, as shot_replay.pose_on_plane does.
    if left:
        rec["joints_m"]["left_wrist"] = _plane_point(CAMERA, rec, LEFT_PX).tolist()
    if right:
        rec["joints_m"]["right_wrist"] = _plane_point(CAMERA, rec, RIGHT_PX).tolist()
    return rec


def frames_for(rows, **kwargs):
    frames = [{"frame": i, "players": [record("near", **kwargs)]} for i in range(len(rows))]
    return frames


class HandTest(unittest.TestCase):
    def test_hand_is_decided_per_frame_and_both_hands_supported(self):
        h = 260  # player box height
        p = player(1, "near", left=LEFT_PX, right=RIGHT_PX)
        self.assertEqual(hand_for(p, (560, 380, 612, 430)), "left")
        self.assertEqual(hand_for(p, (668, 380, 720, 430)), "right")
        close = player(1, "near", left=(640, 420), right=(650, 420))
        self.assertEqual(hand_for(close, (630, 400, 660, 430)), "both")
        self.assertIsNone(hand_for(p, (900, 100, 950, 150)))  # Far from both wrists.
        one = player(1, "near", right=(670, 420))
        self.assertEqual(hand_for(one, (668, 380, 720, 430)), "right")


class BuildTest(unittest.TestCase):
    def test_switching_hands_bystanders_and_stale_hold(self):
        near = lambda: player(1, "near", left=LEFT_PX, right=RIGHT_PX)
        rows = []
        # Frames 0-1 left hand, 2-3 right hand (same player: hand switching).
        for box in [(560, 380, 612, 430)] * 2 + [(668, 380, 720, 430)] * 2:
            rows.append({"tracks": [near(), racquet(1, box)]})
        # Frame 4: the only racquet belongs to a bystander (track 9, no identity).
        rows.append({"tracks": [near(), player(9, None, right=(900, 420)), racquet(9, (890, 380, 950, 430))]})
        # Frames 5-12: no racquet box at all.
        rows += [{"tracks": [near()]} for _ in range(8)]
        frames = frames_for(rows)
        summary = build_racquets(rows, frames, CAMERA, 30.0)
        hands = [[q["hand"] for q in f["racquets"]] for f in frames]
        self.assertEqual(hands[:4], [["left"], ["left"], ["right"], ["right"]])
        statuses = [[q["status"] for q in f["racquets"]] for f in frames]
        # Bystander racquet is never given to the near player; last box is held
        # (stale) for at most 0.2 s = 6 frames after frame 3, then hidden.
        self.assertEqual(statuses[4:10], [["stale"]] * 6)
        self.assertEqual(statuses[10:], [[]] * 3)
        self.assertTrue(all(q["age_frames"] == i + 1 for i, f in enumerate(frames[4:10]) for q in f["racquets"]))
        self.assertEqual(summary["counts"]["observed"], 4)
        for f in frames[:4]:
            q = f["racquets"][0]
            self.assertAlmostEqual(float(np.linalg.norm(q["axis_unit"])), 1.0, places=3)
            self.assertEqual(q["axis_basis"], "wrist_to_box_centre_on_player_plane")

    def test_cut_clears_hold_and_missing_grip_wrist_hides_racquet(self):
        near = lambda: player(1, "near", left=LEFT_PX, right=RIGHT_PX)
        rows = [{"tracks": [near(), racquet(1, (560, 380, 612, 430))]}, {"tracks": [near()]}]
        frames = frames_for(rows)
        build_racquets(rows, frames, CAMERA, 30.0, cuts=[1])
        self.assertEqual(frames[1]["racquets"], [])
        # Box tied to the left wrist, but the 3D left wrist is not available:
        # no grip point, so nothing is drawn (never placed at a guessed hand).
        frames = frames_for(rows[:1], left=False)
        summary = build_racquets(rows[:1], frames, CAMERA, 30.0)
        self.assertEqual(frames[0]["racquets"], [])
        self.assertEqual(summary["counts"]["box_without_supported_wrist"], 1)


if __name__ == "__main__":
    unittest.main()
