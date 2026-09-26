import unittest
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from tennis_vision.ball_motion import MotionBallRecovery
from tennis_vision.events import EventEngine
from tennis_vision.flight3d import _ball_pixels
from tennis_vision.tracking import Detection, NearestTracker


class MotionTest(unittest.TestCase):
    def frame(self,x=None):
        image=np.zeros((200,300,3),np.uint8)
        if x is not None: cv2.circle(image,(x,100),2,(0,255,255),-1)
        return image

    def test_reacquire_requires_three_observations_and_marks_source(self):
        recovery=MotionBallRecovery()
        outputs=[]
        for x in [None,40,45,50]:
            detections,_=recovery.update(self.frame(x),[],lambda p:True)
            outputs.append(detections)
        self.assertTrue(all(not d for d in outputs[:-1]))
        self.assertEqual(outputs[-1][0].source,'motion')
        self.assertLess(abs(outputs[-1][0].center[0]-50),2)

    def test_no_pixels_no_invented_ball(self):
        recovery=MotionBallRecovery()
        ball=Detection('ball',(38,98,42,102),.9)
        recovery.update(self.frame(40),[ball],lambda p:True)
        for _ in range(12):
            recovered,_=recovery.update(self.frame(),[],lambda p:True)
            self.assertEqual(recovered,[])

    def test_stationary_towel_not_a_motion_track(self):
        recovery=MotionBallRecovery()
        for _ in range(12):
            detections,_=recovery.update(self.frame(60),[],lambda p:True)
            self.assertEqual(detections,[])

    def test_unselected_person_is_still_a_motion_blocker(self):
        recovery=MotionBallRecovery()
        referee=Detection('player',(20,80,80,120),.9)
        for x in [None,40,45,50,55,60]:
            detections,_=recovery.update(self.frame(x),[],lambda p:True,[referee])
            self.assertEqual(detections,[])

    def test_search_exclusion_applies_to_recovery(self):
        recovery=MotionBallRecovery()
        for x in [None,40,45,50,55,60]:
            detections,_=recovery.update(self.frame(x),[],lambda p:False)
            self.assertEqual(detections,[])

    def test_cut_resets_anchor(self):
        recovery=MotionBallRecovery()
        recovery.update(self.frame(40),[Detection('ball',(38,98,42,102),.9)],lambda p:True)
        recovered,stats=recovery.update(np.full((200,300,3),255,np.uint8),[],lambda p:True)
        self.assertEqual(recovered,[])
        self.assertTrue(stats['camera_change'])
        self.assertIsNone(recovery.anchor)

    def test_source_export_and_event_exclusion(self):
        tracker=NearestTracker()
        engine=EventEngine(30,None)
        for i in range(6):
            tracks=tracker.update([Detection('ball',(i,10,i+3,13),.25,source='motion')],i)
            self.assertEqual(tracks[0].as_dict(None)['source'],'motion')
            self.assertEqual(engine.update(tracks,i),[])
        self.assertEqual(len(engine.ball_history),0)
        tracks=tracker.update([Detection('ball',(7,10,10,13),.9)],6)
        self.assertEqual(tracks[0].source,'model')

    def test_motion_not_used_for_physics_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl'
            rows=[{'frame':i,'tracks':[{'label':'ball','bbox':[1,2,3,4],'source':source}]} for i,source in enumerate(['model','motion'])]
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            self.assertEqual(_ball_pixels(path),{0:(2.,3.)})
