import copy
import unittest
import numpy as np
from tennis_vision.play_context import assess_play,flight_context
from tennis_vision.stroke_execution import execution_evidence,apply_stroke_review
from tennis_vision.shot_classification import classify_shots
from tennis_vision.event_analysis import observed_balls
from test_shot_replay import stroke_fixture


class ExecutionTests(unittest.TestCase):
    def test_review_allows_both_hand_types_without_confirming_contact(self):
        rows,e,profiles=stroke_fixture();e['status']='unreviewed'
        shots=classify_shots(rows,[e],30,profiles)
        for kind in ('forehand','backhand'):
            p={'schema_version':1,'package_id':'same','corrections':[{'event_id':e['id'],'shot_type':kind,
                'status':'human_reviewed','striking_hands':'both'}]}
            out=apply_stroke_review(shots,p,'same')['shots'][0]
            self.assertEqual(out['classification'],kind)
            self.assertEqual(out['execution']['execution'],'two_handed')
            self.assertEqual(out['event_status'],'unreviewed')
            self.assertEqual(out['action_state'],'uncertain_contact')
            with self.assertRaises(ValueError):apply_stroke_review(shots,p,'wrong')

    def test_same_player_forehands_with_either_hand(self):
        for hand in ('left','right'):
            rows,e,profiles=stroke_fixture(hand=hand)
            profiles['near']['hand']='left' # fixed tendency must not constrain
            s=classify_shots(rows,[e],30,profiles)['shots'][0]
            self.assertEqual(s['classification'],'forehand')
            self.assertEqual(s['execution']['striking_hands'],hand)

    def test_two_hands_on_either_side_are_not_automatically_backhands(self):
        for side in ('left','right'):
            rows,e,_=stroke_fixture()
            for f,row in enumerate(rows):
                x=(110 if side=='left' else 190)+5*np.sin(f/4)
                p=row['tracks'][0];p['keypoints']['left_wrist']=[x,200,.99];p['keypoints']['right_wrist']=[x+8,200,.99]
                row['tracks'][1]['bbox']=[x-4,196,x+12,204]
            result=execution_evidence(rows,e,30)
            self.assertEqual(result['striking_hands'],'both')
            self.assertEqual(result['execution'],'two_handed_candidate')
            self.assertEqual(result['swing_type'],'unknown')
            self.assertEqual(result['stroke_side'],'anatomical_'+side)

    def test_missing_racquet_and_cuts_abstain(self):
        rows,e,_=stroke_fixture()
        self.assertEqual(execution_evidence(rows,e,30,[30])['striking_hands'],'unknown')
        for row in rows:row['tracks']=[p for p in row['tracks'] if p['label']!='racket']
        self.assertEqual(execution_evidence(rows,e,30)['striking_hands'],'unknown')


class PlayTests(unittest.TestCase):
    def test_ball_handling_while_walking_not_counted_as_shot(self):
        rows,e,profiles=stroke_fixture();e.update(status='unreviewed',contact_support='moderate')
        for f,row in enumerate(rows):
            row['tracks'][0]['bbox']=[100+3*f,100,200+3*f,300]
            x,y=150+3*f,260+10*np.sin(f)
            row['tracks'][-1]['bbox']=[x-2,y-2,x+2,y+2]
        events,play=assess_play(rows,[e],30)
        self.assertEqual(events[0]['play_assessment']['action_state'],'no_shot_candidate')
        result=classify_shots(rows,events,30,profiles)
        self.assertEqual(result['counts'],{})
        self.assertEqual(result['shots'][0]['display_label'],'no shot (candidate)')
        self.assertEqual(e['status'],'unreviewed')

    def test_serve_flight_does_not_imply_ace_or_return(self):
        rows,e,profiles=stroke_fixture(serve=True);e.update(status='unreviewed',contact_support='limited')
        for f,row in enumerate(rows):
            row['tracks'].append({'label':'player','identity_id':'far','track_id':2,'bbox':[440,0,480,60],
                                  'keypoints':{},'court_m':[5,24]})
            if f>=30:
                x,y=150+12*(f-30),50-(f-30)*.5
                row['tracks'][2]['bbox']=[x-2,y-2,x+2,y+2]
        receiver={'id':'receiver','frame':52,'time_s':52/30,'type':'hit','status':'unreviewed',
                  'player_id':'far','contact_support':'review_window_only'}
        events,play=assess_play(rows,[e,receiver],30)
        self.assertEqual(events[0]['play_assessment']['action_state'],'shot_candidate')
        self.assertEqual(events[1]['play_assessment']['action_state'],'uncertain_contact')
        self.assertEqual(play['outcome'],'undetermined')
        self.assertEqual(classify_shots(rows,events,30,profiles)['counts'],{'serve':1})

    def test_missing_ball_is_uncertain_not_no_shot_or_point_end(self):
        rows,e,profiles=stroke_fixture();e.update(status='unreviewed',contact_support='moderate')
        for row in rows:row['tracks']=[p for p in row['tracks'] if p['label']!='ball']
        events,play=assess_play(rows,[e],30)
        self.assertEqual(events[0]['play_assessment']['action_state'],'uncertain_contact')
        self.assertEqual(play['outcome'],'undetermined')
        self.assertEqual(classify_shots(rows,events,30,profiles)['counts'],{})

    def test_human_contact_and_bounce_decisions_preserved(self):
        rows,e,_=stroke_fixture();bounce={'id':'b','type':'bounce','frame':40,'status':'confirmed','court_m':[5,10]}
        events,play=assess_play(rows,[e,bounce],30)
        self.assertEqual(events[0]['play_assessment']['action_state'],'confirmed_shot')
        self.assertEqual(events[1],bounce)
        e['status']='rejected'
        self.assertEqual(assess_play(rows,[e],30)[0][0]['play_assessment']['action_state'],'no_shot')

    def test_detector_jump_does_not_establish_flight(self):
        rows,e,_=stroke_fixture()
        rows[31]['tracks'][-1]['bbox']=[1000,0,1004,4]
        result=flight_context(rows,observed_balls(rows),30,'near',30)
        self.assertEqual(result['relation'],'unresolved')
        self.assertEqual(result['stop_reason'],'detector_jump')

if __name__=='__main__':unittest.main()
