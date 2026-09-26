import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

from tennis_vision.player_refinement import choose_pose,crop_box,pose_quality,BODY,refine_far_player
from tennis_vision.contact_detection import detect_contacts,contact_evidence,merge_contacts,apply_contact_review
from tennis_vision.shot_classification import classify_shots
from tennis_vision.shot_replay import feet_record


class Court:
    def project(self,p): return [5.,24.]


def person(x=100):
    return {'label':'player','identity_id':'far','track_id':2,'bbox':[x,40,x+40,140],
            'confidence':.9,'keypoints':{k:[x+20,90,.9] for k in BODY}}


def motion():
    return {'velocity_change_px_frame':5.,'turn_degrees':95.,'speed_ratio':1.2,
        'smooth_rmse_px':3.,'corner_rmse_px':.5,'frame_range':[3,5],
        'before':np.array([2.,1.]),'after':np.array([-2.,1.]),'support_frames':list(range(9))}


class PlayerTests(unittest.TestCase):
    def test_invalid_ankles_fall_back_to_observed_box_not_hidden_person(self):
        p=person();p['keypoints']={'left_ankle':[120,20,.9]}
        class Ground:
            def project(self,xy):return [5,35 if xy[1]<30 else 24]
        record=feet_record(p,Ground())
        self.assertEqual(record['basis'],'bbox_ground_fallback_invalid_ankles')
        self.assertEqual(record['pixel'],[120,140])

    def test_fresh_pose_replaces_only_same_identity_without_mutating_ball(self):
        with tempfile.TemporaryDirectory() as tmp:
            video=Path(tmp)/'source.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(400,200))
            rows=[]
            for f in range(3):
                writer.write(np.full((200,400,3),100,np.uint8))
                p=person();p['keypoints']={}
                if f==2:p['predicted']=True
                rows.append({'frame':f,'tracks':[p,{'label':'ball','bbox':[10,10,12,12]}]})
            writer.release();original=copy.deepcopy(rows)
            refined,report=refine_far_player(rows,video,Court(),None,infer=lambda *a:[person()])
            self.assertEqual(rows,original)
            self.assertEqual(report['counts']['replaced_pose_frames'],2)
            self.assertEqual(refined[0]['tracks'][1],original[0]['tracks'][1])
            self.assertEqual(refined[2],original[2])
            self.assertEqual(refined[0]['tracks'][0]['pose_source'],'focused_crop_model')

    def test_crop_clamped_and_includes_body(self):
        self.assertEqual(crop_box([0,0,20,40],(200,300,3))[:2],[0,0])
        b=crop_box([100,40,140,140],(200,300,3))
        self.assertLess(b[0],100);self.assertGreater(b[2],140)

    def test_same_player_selected_bystander_rejected(self):
        image=np.full((200,400,3),100,np.uint8);anchor=person()
        chosen,_=choose_pose([person(280),person(101)],anchor,image,Court())
        self.assertEqual(chosen['bbox'][0],101)
        self.assertIsNone(choose_pose([person(280)],anchor,image,Court())[0])

    def test_low_quality_and_off_court_rejected(self):
        image=np.full((200,400,3),100,np.uint8);p=person();p['keypoints']={}
        self.assertEqual(pose_quality(p)[0],0)
        self.assertIsNone(choose_pose([p],person(),image,Court())[0])
        with patch.object(Court,'project',return_value=[25,24]):
            self.assertIsNone(choose_pose([person()],person(),image,Court())[0])


class ContactTests(unittest.TestCase):
    def test_explicit_review_bound_and_preserves_bounces(self):
        hit={'id':'h','frame':4,'time_s':4/30,'type':'hit','status':'unreviewed','player_id':'far'}
        bounce={'id':'b','frame':7,'type':'bounce','status':'confirmed','player_id':None}
        payload={'schema_version':1,'kind':'contact_review','binding':'same',
                 'labels':[{'id':'h','frame':5,'status':'confirmed','player_id':'far'}]}
        out=apply_contact_review([hit,bounce],payload,'same',self.balls,30)
        self.assertEqual(out[0]['frame'],5);self.assertEqual(out[0]['status'],'confirmed')
        self.assertEqual(out[1],bounce);self.assertEqual(hit['frame'],4)
        with self.assertRaises(ValueError):apply_contact_review([hit,bounce],payload,'wrong',self.balls,30)
        payload['labels'][0]['id']='b'
        with self.assertRaises(ValueError):apply_contact_review([hit,bounce],payload,'same',self.balls,30)

    def setUp(self):
        self.rows=[{'frame':f,'tracks':[person(),{'label':'ball','source':'gridtracknet',
            'bbox':[118,88,122,92]}]} for f in range(9)]
        self.balls=[{'pixel':[120,90]} for _ in range(9)]

    def test_motion_and_swing_required_not_proximity_alone(self):
        with patch('tennis_vision.contact_detection._motion',return_value=motion()), patch('tennis_vision.contact_detection._swing_speed',return_value=.02):
            c=contact_evidence(self.rows,self.balls,4,30)
            self.assertEqual(c['player_id'],'far');self.assertEqual(c['status'],'unreviewed')
            self.assertFalse(c['score_is_probability'])
            self.assertIsNone(contact_evidence(self.rows,self.balls,4,30,cuts=[5]))
        with patch('tennis_vision.contact_detection._motion',return_value=motion()), patch('tennis_vision.contact_detection._swing_speed',return_value=0):
            self.assertIsNone(contact_evidence(self.rows,self.balls,4,30))

    def test_predictions_not_evidence(self):
        for row in self.rows:row['tracks'][1]['predicted']=True
        self.assertEqual(detect_contacts(self.rows,30)['contacts'],[])

    def test_ambiguous_identity_abstains(self):
        for row in self.rows:
            p=copy.deepcopy(row['tracks'][0]);p['identity_id']='near';row['tracks'].append(p)
        with patch('tennis_vision.contact_detection._motion',return_value=motion()), patch('tennis_vision.contact_detection._swing_speed',return_value=.02):
            self.assertIsNone(contact_evidence(self.rows,self.balls,4,30))

    def test_review_decisions_never_overwritten(self):
        c={'id':'contact_candidate-000004','frame':4,'time_s':4/30,'type':'hit','status':'unreviewed',
           'player_id':'far','support':'limited'}
        b={'id':'bounce','frame':4,'type':'bounce','status':'confirmed','player_id':None}
        self.assertEqual(merge_contacts([b],{'contacts':[c]},30),[b])
        h=dict(b,id='hit',type='hit',player_id='far')
        result=merge_contacts([h],{'contacts':[c]},30)[0]
        self.assertEqual(result['frame'],4);self.assertEqual(result['status'],'confirmed')
        self.assertNotIn('contact_support',h)

    def test_unsupported_old_hits_abstain(self):
        e={'id':'h','frame':4,'time_s':4/30,'type':'hit','player_id':'far','status':'unreviewed',
           'contact_support':'not_supported_by_new_pass'}
        result=classify_shots(self.rows,[e],30,{'far':{'hand':'right'}})['shots'][0]
        self.assertEqual(result['classification'],'unknown')
        self.assertIn('independent contact support',result['reasons'][0])
        e['contact_support']='review_window_only'
        self.assertEqual(classify_shots(self.rows,[e],30,{'far':{'hand':'right'}})['shots'][0]['classification'],'unknown')

    def test_weak_windows_do_not_become_detected_contacts(self):
        with patch('tennis_vision.contact_detection._motion',return_value=None),patch('tennis_vision.contact_detection._swing_speed',return_value=.05):
            packet=detect_contacts(self.rows,30)
        self.assertEqual(packet['contacts'],[])
        self.assertTrue(packet['review_windows'])
        self.assertTrue(all(w['support']=='review_window_only' and w['status']=='unreviewed' for w in packet['review_windows']))


if __name__=='__main__':unittest.main()
