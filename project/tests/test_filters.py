import unittest
from tennis_vision.filters import filter_tracks
from tennis_vision.tracking import Track

class Court:
    source = 'manual'
    def project(self, xy):
        return xy

def player(x=5, y=10, joints=None):
    return Track('player', (x-1, y-10, x+1, y), .9, 1, 0, joints)

def racket(box, identity=2, confidence=.8):
    return Track('racket', box, confidence, identity, 0)

class FiltersTest(unittest.TestCase):
    def test_adjacent_court_removed_baseline_player_retained(self):
        result, _ = filter_tracks([player(20), player(5, -3)], Court())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox[3], -3)

    def test_ankles_override_box_bottom(self):
        result, _ = filter_tracks([player(20, joints={'left_ankle':[5,10,.9]})], Court())
        self.assertEqual(len(result), 1)

    def test_duplicate_rackets_and_owner(self):
        p = player(joints={'left_wrist':[5,5,.9]})
        result, stats = filter_tracks([p, racket((4,4,7,7)), racket((4,4,7,7),3,.5)], Court())
        self.assertEqual(stats['duplicate_rackets'], 1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1].player_track_id, 1)

    def test_disjoint_duplicates_one_per_player(self):
        p = player(joints={'left_wrist':[5,5,.9]})
        result, _ = filter_tracks([p, racket((4,4,6,6)), racket((7,4,9,6),3)], Court())
        self.assertEqual(sum(t.label=='racket' for t in result), 1)

    def test_distant_racket_rejected_ball_not_ground_filtered(self):
        ball = Track('ball', (100,100,102,102), .9, 9, 0)
        result, _ = filter_tracks([player(), racket((80,80,85,85)), ball], Court())
        self.assertIn(ball, result)
        self.assertFalse(any(t.label=='racket' for t in result))

    def test_missing_calibration_does_not_remove_players(self):
        result, _ = filter_tracks([player(100)], None)
        self.assertEqual(len(result), 1)

    def test_missing_wrists_fallback_and_predicted_player(self):
        p = player()
        result, _ = filter_tracks([p, racket((4,4,6,6))], Court())
        self.assertEqual(len(result), 2)
        p.predicted = True
        result, _ = filter_tracks([p, racket((4,4,6,6))], Court())
        self.assertEqual(len(result), 1)
