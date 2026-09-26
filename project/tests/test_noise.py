import unittest
from tennis_vision.tracking import Detection, NearestTracker
from tennis_vision.events import EventEngine


class NoiseRegression(unittest.TestCase):
    def test_fast_objects_do_not_leave_ghosts(self):
        tracker = NearestTracker()
        tracker.update([Detection('ball', (0, 0, 5, 5), .8), Detection('racket', (10, 10, 20, 20), .8)], 0)
        self.assertEqual(tracker.update([], 1), [])

    def test_predicted_player_does_not_reuse_old_ankles(self):
        tracker = NearestTracker()
        tracker.update([Detection('player', (0, 0, 20, 40), .9, {'left_ankle': [10, 39, .9]})], 0)
        track = tracker.update([], 1)[0]
        self.assertTrue(track.predicted)
        self.assertIsNone(track.keypoints)
        for frame in range(2, 7):
            tracks = tracker.update([], frame)
        self.assertEqual(tracks, [])

    def test_bounce_history_breaks_across_missing_frames(self):
        tracker, engine = NearestTracker(), EventEngine(30, None)
        engine.update(tracker.update([Detection('ball', (0, 0, 5, 5), .9)], 0), 0)
        self.assertEqual(len(engine.ball_history), 1)
        engine.update([], 1)
        self.assertEqual(len(engine.ball_history), 0)

if __name__ == '__main__':
    unittest.main()
