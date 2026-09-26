import unittest
import cv2
import numpy as np
from tennis_vision.scene import SceneSelector
from tennis_vision.tracking import Detection, NearestTracker

class SceneTest(unittest.TestCase):
    def config(self):
        return {'near_seed_polygon':[[.1,.3],[.4,.3],[.4,.95],[.1,.95]],
                'far_seed_polygon':[[.5,0],[.8,0],[.8,.3],[.5,.3]],
                'ball_polygon':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]}

    def test_colour_reacquisition_outside_seed_and_missing_state(self):
        scene=SceneSelector(self.config())
        image=np.zeros((100,100,3),np.uint8)
        image[:]=(255,0,0)
        for _ in range(3):
            selected,_=scene.update(image,[Detection('player',(20,40,30,80),.9),Detection('racket',(28,55,34,65),.9)],None)
        self.assertEqual([d.identity_id for d in selected if d.label=='player'],['near'])
        selected,_=scene.update(image,[],None)
        self.assertEqual(selected,[])
        for x in [25,30,35,40,45]:
            selected,_=scene.update(image,[Detection('player',(x,40,x+10,80),.9)],None)
        self.assertEqual(selected[0].identity_id,'near')

    def test_white_towel_rejected_tiny_yellow_ball_retained(self):
        scene=SceneSelector(self.config())
        image=np.zeros((100,100,3),np.uint8)
        image[40:45,40:45]=(255,255,255)
        image[60:64,60:64]=(0,255,255)
        selected,stats=scene.update(image,[Detection('ball',(40,40,44,44),.9),Detection('ball',(60,60,63,63),.9)],None)
        self.assertEqual(len(selected),1)
        self.assertEqual(stats['ball_colour_rejected'],1)

    def test_adjacent_ball_rejected(self):
        scene=SceneSelector(self.config())
        image=np.full((100,100,3),(0,255,255),np.uint8)
        selected,stats=scene.update(image,[Detection('ball',(0,0,4,4),.9)],None)
        self.assertEqual(selected,[])
        self.assertEqual(stats['ball_outside_area'],1)

    def test_identity_export(self):
        tracker=NearestTracker()
        d=Detection('player',(0,0,20,40),.9,identity_id='near')
        self.assertEqual(tracker.update([d],0)[0].as_dict(None)['identity_id'],'near')

    def test_two_players_have_distinct_persistent_labels(self):
        scene=SceneSelector(self.config())
        image=np.zeros((100,100,3),np.uint8)
        image[40:80,20:30]=(255,0,0)
        image[5:25,60:70]=(255,255,255)
        for _ in range(3):
            near=Detection('player',(20,40,30,80),.9)
            far=Detection('player',(60,5,70,25),.9)
            selected,_=scene.update(image,[near,far,Detection('racket',(28,55,34,65),.9),Detection('racket',(68,10,72,18),.9)],None)
        self.assertEqual({d.identity_id for d in selected if d.label=='player'},{'near','far'})
        selected,_=scene.update(image,[Detection('player',(60,5,70,25),.9)],None)
        self.assertEqual([d.identity_id for d in selected],['far'])

    def test_explicit_background_patch(self):
        config=self.config()
        config['ball_exclusion_polygons']=[[[.4,.4],[.6,.4],[.6,.6],[.4,.6]]]
        scene=SceneSelector(config)
        image=np.full((100,100,3),(0,255,255),np.uint8)
        selected,stats=scene.update(image,[Detection('ball',(48,48,52,52),.9)],None)
        self.assertEqual(selected,[])
        self.assertEqual(stats['ball_outside_area'],1)

    def test_no_initial_lock_without_racket(self):
        scene=SceneSelector(self.config())
        image=np.full((100,100,3),(255,0,0),np.uint8)
        for _ in range(10):
            selected,_=scene.update(image,[Detection('player',(60,5,70,25),.9)],None)
            self.assertFalse(any(d.label=='player' for d in selected))

    def test_referee_cannot_steal_identity_after_gap(self):
        scene=SceneSelector(self.config())
        image=np.full((100,100,3),(255,0,0),np.uint8)
        for _ in range(3):
            scene.update(image,[Detection('player',(60,5,70,25),.9),Detection('racket',(68,10,72,18),.9)],None)
        for _ in range(20): scene.update(image,[],None)
        for _ in range(30):
            selected,_=scene.update(image,[Detection('player',(15,5,25,25),.9)],None)
            self.assertFalse(any(d.label=='player' for d in selected))
        # Actual player returns: three consistent observations and racket evidence.
        for _ in range(3):
            selected,_=scene.update(image,[Detection('player',(60,5,70,25),.9),Detection('racket',(68,10,72,18),.9)],None)
        self.assertEqual([d.identity_id for d in selected if d.label=='player'],['far'])

    def test_pale_blurred_ball_not_rejected(self):
        scene=SceneSelector(self.config())
        hsv=np.zeros((100,100,3),np.uint8)
        hsv[60:64,60:64]=(40,45,210)
        image=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        selected,_=scene.update(image,[Detection('ball',(58,58,65,65),.9)],None)
        self.assertEqual(len(selected),1)

    def test_local_player_survives_colour_change_but_not_distant_jump(self):
        scene=SceneSelector(self.config())
        image=np.full((100,100,3),(255,0,0),np.uint8)
        for _ in range(3):
            scene.update(image,[Detection('player',(60,5,70,25),.9),Detection('racket',(68,10,72,18),.9)],None)
        # Same place/scale, radically different apparent torso colour.
        image[:]=(255,255,255)
        selected,_=scene.update(image,[Detection('player',(61,5,71,25),.8),Detection('player',(15,5,25,25),.95)],None)
        players=[d for d in selected if d.label=='player']
        self.assertEqual(len(players),1)
        self.assertEqual(players[0].bbox[0],61)

    def test_ball_area_expands_only_next_to_confirmed_player(self):
        config=self.config()
        config['ball_polygon']=[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]
        scene=SceneSelector(config)
        player=Detection('player',(80,40,90,80),.9,identity_id='near')
        self.assertFalse(scene.ball_allowed((95,60),(100,100,3),None,[]))
        self.assertTrue(scene.ball_allowed((95,60),(100,100,3),None,[player]))
        config['ball_exclusion_polygons']=[[[.9,.5],[1,.5],[1,.7],[.9,.7]]]
        self.assertFalse(scene.ball_allowed((95,60),(100,100,3),None,[player]))
