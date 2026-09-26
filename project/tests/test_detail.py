import unittest
import numpy as np
from tennis_vision.detail import translate, merge, DetailPass, far_crop
from tennis_vision.tracking import Detection
from tennis_vision.court import CourtMapper

class DetailTest(unittest.TestCase):
    def test_translate_pose_without_mutating_original(self):
        d = Detection('player',(1,2,10,20),.8,{'left_ankle':[5,18,.9]})
        moved = translate(d,100,200)
        self.assertEqual(moved.bbox,(101,202,110,220))
        self.assertEqual(moved.keypoints['left_ankle'],[105,218,.9])
        self.assertEqual(d.keypoints['left_ankle'],[5,18,.9])

    def test_cross_scale_duplicates_removed(self):
        a = Detection('player',(10,10,30,60),.8)
        b = Detection('player',(11,11,31,61),.9)
        self.assertEqual(merge([a,b]),[b])

    def test_missing_ball_not_fabricated_and_crop_expires(self):
        detail = DetailPass()
        image = np.zeros((720,1280,3),dtype=np.uint8)
        detail.feedback([Detection('ball',(300,200,306,206),.9)])
        for _ in range(3):
            found,stats = detail.update(image,[],None,lambda c:[],lambda c:[])
            self.assertEqual(found,[])
            self.assertEqual(stats['crop_passes'],1)
            detail.feedback(found)
        self.assertIsNone(detail.last_ball)

    def test_unaccepted_ball_cannot_steer_crop(self):
        detail=DetailPass()
        image=np.zeros((720,1280,3),dtype=np.uint8)
        detail.update(image,[Detection('ball',(300,200,306,206),.9)],None,lambda c:[],lambda c:[])
        detail.feedback([])
        self.assertIsNone(detail.last_ball)

    def test_far_crop_keeps_racket(self):
        court=CourtMapper(np.array([[3,535],[1188,570],[782,133],[427,121]],dtype=np.float32))
        detail=DetailPass()
        found,_=detail.update(np.zeros((720,1280,3),np.uint8),[],court,
                             lambda c:[Detection('racket',(10,20,20,35),.8)],lambda c:[])
        self.assertTrue(any(d.label=='racket' for d in found))

    def test_calibrated_far_crop_inside_frame(self):
        court=CourtMapper(np.array([[3,535],[1188,570],[782,133],[427,121]],dtype=np.float32))
        x1,y1,x2,y2=far_crop((720,1280,3),court,1.5,6)
        self.assertTrue(0<=x1<x2<=1280 and 0<=y1<y2<=720)
        self.assertTrue(x1<590<x2 and y1<100<y2)
