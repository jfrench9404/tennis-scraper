"""Run comparison aligns on SOURCE frames, not run-relative frame numbers."""
import json
from pathlib import Path
import tempfile
import unittest

from tennis_vision.compare_runs import compare, windows


def write_run(folder, start, rows):
    folder.mkdir()
    (folder / "source-offset.json").write_text(json.dumps({"source_start_frame": start}))
    (folder / "summary.json").write_text(json.dumps({"fps": 30.0, "frames": len(rows)}))
    with (folder / "events.jsonl").open("w") as stream:
        for i, tracks in enumerate(rows):
            stream.write(json.dumps({"frame": i, "time_s": i / 30, "tracks": tracks, "events": []}) + "\n")


def near(joints=17):
    names = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
             "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    return {"label": "player", "identity_id": "near", "predicted": False,
            "keypoints": {n: [0, 0, .9 if i < joints else .1] for i, n in enumerate(names)}}


class CompareTest(unittest.TestCase):
    def test_overlap_uses_source_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ball = {"label": "ball", "predicted": False}
            write_run(root / "a", 100, [[near()], [near(), ball], [near(4)]])       # source 100-102
            write_run(root / "b", 0, [[]] * 101 + [[near(), ball], [near()], []])  # source 0-103
            result = compare(root / "a", root / "b")
            self.assertEqual(result["source_frames"], [100, 102])
            self.assertEqual(result["a"]["ball_frames"], 1)
            self.assertEqual(result["b"]["ball_frames"], 1)  # b's ball is at source 101 too.
            self.assertEqual(result["a"]["near_frames_6plus_joints"], 2)
            self.assertEqual(result["frames_with_identical_tracks"], 1)
            self.assertEqual(sum(w["frames"] for w in windows(root / "b", 1.0)), 104)


if __name__ == "__main__":
    unittest.main()
