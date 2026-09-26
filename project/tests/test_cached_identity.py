import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tennis_vision.cached_identity import choose_previous, recover_initial_players


def person(x=4.):
    return {'label':'player','source':'model','bbox':[x,15.,x+2,20.],
            'confidence':.3,'keypoints':{'nose':[x+1,16.,.9]}}


class CachedIdentityTests(unittest.TestCase):
    def setUp(self):
        self.court=SimpleNamespace(project=lambda xy:xy)
        self.hist=np.zeros(128,np.float32);self.hist[0]=1

    def choose(self,candidates):
        return choose_previous(candidates,person(),self.hist,self.hist,self.court,'far')

    def test_adjacent_detection_supported_but_bystander_rejected(self):
        match,reason=self.choose([(person(25),self.hist),(person(4.1),self.hist)])
        self.assertIsNone(reason);self.assertEqual(match[0]['bbox'][0],4.1)
        self.assertIsNone(self.choose([(person(25),self.hist)])[0])

    def test_competing_people_abstain_duplicates_do_not(self):
        match,reason=self.choose([(person(3.55),self.hist),(person(4.45),self.hist)])
        self.assertIsNone(match);self.assertIn('ambiguous',reason)
        self.assertIsNotNone(self.choose([(person(),self.hist),(person(),self.hist)])[0])

    def test_wrong_colours_predictions_motion_and_wrong_half_rejected(self):
        wrong=np.zeros(128,np.float32);wrong[100]=1
        self.assertIsNone(self.choose([(person(),wrong)])[0])
        for change in ({'predicted':True},{'source':'motion'},{'confidence':.05},
                       {'bbox':[4.,5.,6.,10.]},{'bbox':[4.,10.,6.,20.]},
                       {'bbox':[4.,float('nan'),6.,20.]}):
            self.assertIsNone(self.choose([(dict(person(),**change),self.hist)])[0],change)

    def rows(self,seed=5):
        return [{'frame':f,'tracks':([dict(person(),identity_id='far',track_id=7)] if f>=seed else []),
                 'detail':{'candidates':[person()]}} for f in range(seed+5)]

    def recover(self,rows,cuts=()):
        class FakeVideo:
            def set(self,*args):pass
            def read(self):return True,np.full((50,50,3),180,np.uint8)
            def release(self):pass
        with patch('tennis_vision.cached_identity.cv2.VideoCapture',return_value=FakeVideo()):
            return recover_initial_players(rows,'fixture.mp4',self.court,30,cuts)

    def test_delayed_identity_restored_no_input_mutation_or_joint_invention(self):
        rows=self.rows();original=copy.deepcopy(rows)
        result,report=self.recover(rows)
        self.assertEqual(report['recovered_frames']['far'],5)
        self.assertEqual(rows,original)
        for f in range(5):
            track=result[f]['tracks'][0]
            self.assertFalse(track['predicted'])
            self.assertTrue(track['identity_association_estimated'])
            self.assertEqual(track['keypoints'],rows[f]['detail']['candidates'][0]['keypoints'])
            self.assertEqual(track['identity_seed_frame'],5)
        self.assertEqual(result[5:],rows[5:])

    def test_stops_at_missing_candidate_cut_or_ambiguous_person(self):
        rows=self.rows();rows[2]['detail']['candidates']=[]
        result,report=self.recover(rows)
        self.assertEqual(report['recovered_frames']['far'],2)
        self.assertEqual(result[2]['tracks'],[]);self.assertEqual(result[0]['tracks'],[])
        result,report=self.recover(self.rows(),cuts=[3])
        self.assertEqual(report['recovered_frames']['far'],2)
        self.assertEqual(report['identities']['far']['stop_reason'],'camera cut')
        rows=self.rows();rows[3]['detail']['candidates']=[person(3.55),person(4.45)]
        self.assertEqual(self.recover(rows)[1]['recovered_frames']['far'],1)

    def test_maximum_lookback_and_unstable_seed(self):
        result,report=self.recover(self.rows(seed=100))
        self.assertEqual(report['recovered_frames']['far'],90)
        self.assertEqual(result[9]['tracks'],[])
        rows=self.rows();rows[7]['tracks']=[]
        self.assertEqual(self.recover(rows)[1]['recovered_frames']['far'],0)
        self.assertEqual(self.recover(self.rows(),cuts=[7])[1]['recovered_frames']['far'],0)

    def test_failed_source_decode_preserves_original(self):
        rows=self.rows()
        cap=SimpleNamespace(set=lambda *a:None,read=lambda:(False,None),release=lambda:None)
        with patch('tennis_vision.cached_identity.cv2.VideoCapture',return_value=cap):
            result,report=recover_initial_players(rows,'missing.mp4',self.court,30)
        self.assertEqual(result,rows)
        self.assertEqual(report['recovered_frames']['far'],0)
        self.assertIn('decode failed',report['identities']['far']['stop_reason'])


if __name__=='__main__':unittest.main()
