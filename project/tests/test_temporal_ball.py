import unittest

import numpy as np

from tennis_vision.events import EventEngine
from tennis_vision.scene import SceneSelector
from tennis_vision.temporal_ball import pack_frames, decode_grid, frame_stream, select_temporal_ball
from tennis_vision.tracking import Detection, NearestTracker


class Capture:
    def __init__(self,count):
        self.frames=[np.full((12,20,3),i,np.uint8) for i in range(count)]
        self.reads=0

    def read(self):
        self.reads+=1
        return (True,self.frames.pop(0)) if self.frames else (False,None)


class FakeDetector:
    def __init__(self): self.blocks=[]
    def predict(self,frames):
        self.blocks.append([int(f[0,0,0]) for f in frames])
        return [(None,{'context_index':i}) for i in range(5)]


class TemporalTests(unittest.TestCase):
    def test_rgb_and_time_channel_order(self):
        frames=[np.full((8,8,3),(10+i,20+i,30+i),np.uint8) for i in range(5)]
        packed=pack_frames(frames)
        self.assertEqual(packed.shape,(1,15,432,768))
        np.testing.assert_allclose(packed[0,:,0,0]*255,[30,20,10,31,21,11,32,22,12,33,23,13,34,24,14],atol=1e-5)

    def test_requires_five_equal_sized_images(self):
        with self.assertRaises(ValueError): pack_frames([])
        frames=[np.zeros((8,8,3),np.uint8)]*4+[np.zeros((9,8,3),np.uint8)]
        with self.assertRaises(ValueError): pack_frames(frames)

    def test_decode_grid_offsets_and_frame_alignment(self):
        grid=np.zeros((1,15,27,48),np.float32)
        for i in range(5):
            grid[0,i*3,3,5+i]=.9
            grid[0,i*3+1,3,5+i]=.25
            grid[0,i*3+2,3,5+i]=.5
        decoded=decode_grid(grid,(720,1280,3))
        for i,(d,stats) in enumerate(decoded):
            self.assertEqual(d.source,'gridtracknet')
            self.assertAlmostEqual(d.center[0],(5+i+.25)*1280/48)
            self.assertAlmostEqual(d.center[1],3.5*720/27)

    def test_missing_heatmap_does_not_output_origin_ball(self):
        self.assertTrue(all(d is None for d,_ in decode_grid(np.zeros((1,15,27,48)),(720,1280,3))))

    def test_invalid_scores_and_shape(self):
        with self.assertRaises(ValueError): decode_grid(np.zeros((1,3,27,48)),(720,1280,3))
        with self.assertRaises(ValueError): decode_grid(np.full((1,15,27,48),np.nan),(720,1280,3))
        for value in (0,-1,2,float('nan')):
            with self.assertRaises(ValueError): decode_grid(np.zeros((1,15,27,48)),(720,1280,3),value)

    def test_tail_keeps_every_frame_once(self):
        detector=FakeDetector()
        items=list(frame_stream(Capture(7),detector))
        self.assertEqual([int(f[0,0,0]) for f,_,_ in items],list(range(7)))
        self.assertEqual(detector.blocks,[[0,1,2,3,4],[5,6,6,6,6]])

    def test_max_frames_does_not_read_past_limit(self):
        cap=Capture(10); detector=FakeDetector()
        self.assertEqual(len(list(frame_stream(cap,detector,3))),3)
        self.assertEqual(cap.reads,3)
        self.assertEqual(detector.blocks,[[0,1,2,2,2]])

    def test_no_model_passthrough(self):
        self.assertEqual(len(list(frame_stream(Capture(3)))),3)
        self.assertEqual(list(frame_stream(Capture(0),FakeDetector())),[])

    def test_temporal_ball_keeps_scene_exclusions_but_not_colour_gate(self):
        config={'ball_polygon':[[0,0],[1,0],[1,1],[0,1]],
                'ball_exclusion_polygons':[[[.4,.4],[.6,.4],[.6,.6],[.4,.6]]]}
        scene=SceneSelector(config)
        image=np.zeros((100,100,3),np.uint8)
        visible=Detection('ball',(18,18,22,22),.9,source='gridtracknet')
        towel=Detection('ball',(48,48,52,52),.9,source='gridtracknet')
        for _ in range(20):
            selected,_=scene.update(image,[visible,towel],None)
            self.assertEqual(selected,[visible])
        self.assertEqual(select_temporal_ball(towel,scene,image,None,[]),([], 'outside_area'))

    def test_learned_temporal_observation_enters_event_history(self):
        engine=EventEngine(30,None)
        tracks=NearestTracker().update([Detection('ball',(1,2,3,4),.9,source='gridtracknet')],0)
        engine.update(tracks,0)
        self.assertEqual(len(engine.ball_history),1)

    def test_select_inside_court_before_taking_highest_peak(self):
        grid=np.zeros((1,15,27,48),np.float32)
        grid[0,0,5,45]=.99  # adjacent-court ball, stronger
        grid[0,0,5,20]=.8   # intended court
        detection,stats=decode_grid(grid,(720,1280,3))[0]
        scene=SceneSelector({'ball_polygon':[[.2,0],[.8,0],[.8,1],[.2,1]]})
        frame=np.zeros((720,1280,3),np.uint8)
        balls,status=select_temporal_ball(detection,scene,frame,None,[],stats)
        self.assertEqual(status,'accepted')
        self.assertAlmostEqual(balls[0].center[0],20*1280/48)


if __name__=='__main__': unittest.main()
