import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from tennis_vision.camera3d import Camera3D, refine_from_court
from tennis_vision.court import CourtMapper
from tennis_vision.shot_classification import classify_shots, infer_handedness, resolve_review_events
from tennis_vision.shot_replay import build, build_players, feet_record
from tennis_vision.event_analysis import VERSION
from tennis_vision.validated_flight import BALL_RADIUS, fit_interval, fit_reviewed_flights, fit_landing_arc, positions_at
from tennis_vision.preview_flight import fit_preview_flights


def make_camera():
    eye=np.array([0.,-25.,8.]);forward=-eye/np.linalg.norm(eye)
    right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
    down=np.cross(forward,right);rotation=np.stack((right,down,forward))
    return Camera3D(np.array([[1100.,0.,640.],[0.,1100.,360.],[0.,0.,1.]]),
                    cv2.Rodrigues(rotation)[0],-rotation@eye,(1280,720),.45)


def arc_fixture(start_height=1.3, start_kind='hit'):
    camera=make_camera();a=np.array([-1.,-7.,start_height]);b=np.array([1.,5.,BALL_RADIUS]);duration=.8
    velocity=(b-a)/duration-np.array([0,0,.5*9.81*duration])
    positions=positions_at(b,velocity,(np.arange(25)-24)/30)
    balls=[{'pixel':p.tolist(),'source':'gridtracknet'} for p in camera.project(positions)]
    start={'id':'start','type':start_kind,'frame':0,'status':'confirmed','court_m':[a[0]+5.485,a[1]+11.885]}
    bounce={'id':'bounce','type':'bounce','frame':24,'status':'confirmed','court_m':[b[0]+5.485,b[1]+11.885]}
    return camera,balls,start,bounce,positions


def stroke_fixture(side='forehand',hand='right',ground_y=2,serve=False):
    rows=[]
    for f in range(61):
        moving_x=(195 if (side=='forehand')==(hand=='right') else 105)+8*np.sin(f/4)
        moving_y=195+28*np.sin(f/4)
        lw=[110.,200.,.99];rw=[190.,200.,.99]
        if hand=='right':rw=[moving_x,moving_y,.99]
        else:lw=[moving_x,moving_y,.99]
        # Explicit ONE-handed fixture: keep the idle hand away from the grip.
        if side=='backhand':
            if hand=='right':lw=[120.,270.,.99]
            else:rw=[180.,270.,.99]
        if serve:
            lw=[115.,50. if 8<=f<=25 else 150.,.99]
            rw=[180.,65. if 27<=f<=32 else 205.,.99]
        p={'label':'player','identity_id':'near','track_id':1,'bbox':[100,100,200,300],
           'court_m':[5,ground_y],'predicted':False,'keypoints':{
               'left_shoulder':[120,140,.99],'right_shoulder':[180,140,.99],
               'left_hip':[130,220,.99],'right_hip':[170,220,.99],
               'left_wrist':lw,'right_wrist':rw,'left_ankle':[130,295,.99],'right_ankle':[170,295,.99]}}
        wrist=rw if hand=='right' else lw
        ball_y=50 if serve and 27<=f<=32 else 185
        rows.append({'frame':f,'time_s':f/30,'tracks':[p,
             {'label':'racket','bbox':[wrist[0]-5,wrist[1]-5,wrist[0]+5,wrist[1]+5],'player_track_id':1},
             {'label':'ball','bbox':[147,ball_y-3,153,ball_y+3],'source':'gridtracknet','predicted':False}]})
    e={'id':'hit30','type':'hit','status':'confirmed','frame':30,'time_s':1.,'player_id':'near'}
    profiles={'near':{'hand':hand},'far':{'hand':'right'}}
    return rows,e,profiles


class ShotTypeTests(unittest.TestCase):
    def test_forehand_backhand_both_handedness(self):
        for hand in ('left','right'):
            for side in ('forehand','backhand'):
                rows,e,profiles=stroke_fixture(side,hand)
                result=classify_shots(rows,[e],30,profiles)
                self.assertEqual(result['shots'][0]['classification'],side,(hand,side,result))
                json.dumps(result,allow_nan=False)

    def test_unknown_player_handedness_does_not_override_visible_stroke_hand(self):
        rows,e,profiles=stroke_fixture();profiles['near']['hand']='unknown'
        self.assertEqual(classify_shots(rows,[e],30,profiles)['shots'][0]['classification'],'forehand')
        for row in rows:row['tracks']=[t for t in row['tracks'] if t['label']!='racket']
        self.assertEqual(classify_shots(rows,[e],30,profiles)['shots'][0]['classification'],'unknown')

    def test_serve_surrounding_toss_overhead_baseline(self):
        rows,e,profiles=stroke_fixture(serve=True)
        self.assertEqual(classify_shots(rows,[e],30,profiles)['shots'][0]['classification'],'serve')
        for row in rows:row['tracks'][0]['court_m'][1]=11
        self.assertEqual(classify_shots(rows,[e],30,profiles)['shots'][0]['classification'],'unknown')

    def test_volley_requires_incoming_evidence_not_just_absent_bounce(self):
        rows,e,profiles=stroke_fixture(ground_y=10)
        previous={'id':'other','type':'hit','status':'confirmed','frame':0,'time_s':0,'player_id':'far'}
        result=classify_shots(rows,[previous,e],30,profiles)['shots'][-1]
        self.assertEqual(result['classification'],'volley',result)
        # Missing incoming pixels must not be interpreted as "no bounce".
        for i in range(5,15):rows[i]['tracks']=[p for p in rows[i]['tracks'] if p['label']!='ball']
        self.assertEqual(classify_shots(rows,[previous,e],30,profiles)['shots'][-1]['classification'],'unknown')

    def test_confirmed_incoming_bounce_prevents_volley(self):
        rows,e,profiles=stroke_fixture(ground_y=10)
        previous={'id':'other','type':'hit','status':'confirmed','frame':0,'time_s':0,'player_id':'far'}
        bounce={'id':'b','type':'bounce','status':'confirmed','frame':20,'time_s':20/30,'player_id':None}
        result=classify_shots(rows,[previous,bounce,e],30,profiles)['shots'][-1]
        self.assertEqual(result['classification'],'forehand')

    def test_rejected_hits_ignored_camera_cut_abstains(self):
        rows,e,profiles=stroke_fixture()
        self.assertEqual(classify_shots(rows,[e],30,profiles,cuts=[29])['shots'][0]['classification'],'unknown')
        e['status']='rejected';self.assertEqual(classify_shots(rows,[e],30,profiles)['shots'],[])

    def test_handedness_votes_and_explicit_override(self):
        rows,_,_=stroke_fixture()
        self.assertEqual(infer_handedness(rows)['near']['hand'],'right')
        self.assertEqual(infer_handedness(rows,{'near':'left'})['near']['hand'],'left')
        self.assertEqual(infer_handedness([])['near']['hand'],'unknown')


class ReviewConstraintTests(unittest.TestCase):
    def setUp(self):
        self.court=CourtMapper(np.array([[0,100],[100,100],[100,0],[0,0]],np.float32))
        self.candidates=[{'id':'b','type':'bounce_candidate','frame':2,'landing_if_bounce_m':[99,99]}]
        self.balls=[{'pixel':[f*10,50]} for f in range(6)]
        self.labels={'schema_version':1,'run_id':'test','manual_events':[],
                     'labels':[{'id':'b','status':'confirmed','type':'bounce','frame':4,'player_id':None}]}

    def resolve(self,labels):
        return resolve_review_events(self.candidates,labels,self.balls,30,self.court,'test',100,100)

    def test_corrected_frame_recomputes_location_does_not_reuse_candidate(self):
        e=self.resolve(self.labels)[0]
        self.assertEqual(e['pixel'],[40,50]);self.assertEqual(e['frame'],4)
        self.assertEqual(e['court_m'],self.court.project((40,50)))

    def test_unreviewed_rejected_and_uncertain_are_not_ground_constraints(self):
        self.assertIsNone(self.resolve(None)[0]['court_m'])
        for status in ('unreviewed','uncertain','rejected'):
            self.labels['labels'][0]['status']=status
            self.assertIsNone(self.resolve(self.labels)[0]['court_m'])

    def test_candidate_locations_visible_but_never_physics_constraints(self):
        e=self.resolve(None)[0]
        self.assertEqual(e['candidate_court_m'],self.court.project((20,50)))
        self.assertIn('unverified',e['candidate_geometry_basis'])
        self.assertEqual(fit_reviewed_flights(None,[e],self.balls,30)['fits'],[])
        for status in ('rejected','confirmed'):
            self.labels['labels'][0]['status']=status
            self.assertIsNone(self.resolve(self.labels)[0]['candidate_court_m'])

    def test_missing_and_out_of_region_candidates_have_no_display_ground(self):
        self.balls[2]=None
        self.assertIsNone(self.resolve(None)[0]['candidate_court_m'])
        self.balls[2]={'pixel':[1000,50]}
        event=self.resolve(None)[0]
        self.assertIsNone(event['candidate_court_m']);self.assertIsNone(event['court_m'])
        self.assertIn('outside',event['geometry_issue'])

    def test_missing_ball_at_reviewed_frame_not_interpolated(self):
        self.balls[4]=None;e=self.resolve(self.labels)[0]
        self.assertIsNone(e['court_m']);self.assertIn('No observed ball',e['geometry_issue'])
        self.labels['labels'][0]['landing_pixel']=[45,60]
        self.assertEqual(self.resolve(self.labels)[0]['pixel_basis'],'explicit_human_landing_pixel')

    def test_mismatched_run_invalid_frame_duplicate_rejected(self):
        bad=copy.deepcopy(self.labels);bad['run_id']='wrong'
        with self.assertRaises(ValueError):self.resolve(bad)
        bad=copy.deepcopy(self.labels);bad['labels'][0]['frame']=100
        with self.assertRaises(ValueError):self.resolve(bad)
        bad=copy.deepcopy(self.labels);bad['labels'].append(bad['labels'][0])
        with self.assertRaises(ValueError):self.resolve(bad)


class PhysicsTests(unittest.TestCase):
    def test_known_synthetic_arc_recovered_and_endpoint_exact(self):
        camera,balls,start,bounce,expected=arc_fixture()
        fit,reason=fit_interval(camera,start,bounce,balls,30)
        self.assertIsNone(reason);self.assertIsNotNone(fit)
        points=np.array([p['point_m'] for p in fit['trajectory_m']])
        np.testing.assert_allclose(points,expected,atol=1e-4)
        self.assertEqual(fit['airborne_xyz_status'],'estimated_not_measured')
        self.assertLess(fit['median_reprojection_error_px'],.01)

    def test_two_confirmed_bounces_constrain_both_ends(self):
        camera,balls,start,bounce,expected=arc_fixture(BALL_RADIUS,'bounce')
        fit,reason=fit_interval(camera,start,bounce,balls,30)
        self.assertIsNone(reason);np.testing.assert_allclose(fit['trajectory_m'][0]['point_m'],expected[0],atol=1e-4)

    def test_negative_height_and_dropout_rejected(self):
        camera,balls,start,bounce,_=arc_fixture(-1)
        self.assertIn('height',fit_interval(camera,start,bounce,balls,30)[1])
        camera,balls,start,bounce,_=arc_fixture()
        balls[3:20]=[None]*17
        self.assertIsNone(fit_interval(camera,start,bounce,balls,30)[0])

    def test_confirmation_is_mandatory_and_no_bridge_across_impact_or_cut(self):
        camera,balls,start,bounce,_=arc_fixture()
        unreviewed=dict(bounce,status='unreviewed')
        self.assertEqual(fit_reviewed_flights(camera,[start,unreviewed],balls,30)['fits'],[])
        result=fit_reviewed_flights(camera,[start,bounce],balls,30,cuts=[10])
        self.assertIn('Camera cut',result['skipped'][0]['reason'])
        possible={'id':'maybe','frame':10,'status':'uncertain','type':'hit'}
        result=fit_reviewed_flights(camera,[start,possible,bounce],balls,30)
        self.assertIn('intervening',result['skipped'][0]['reason'])
        possible['status']='rejected'
        self.assertEqual(len(fit_reviewed_flights(camera,[start,possible,bounce],balls,30)['fits']),1)


class PlayerReplayTests(unittest.TestCase):
    def setUp(self):
        self.court=CourtMapper(np.array([[0,100],[100,100],[100,0],[0,0]],np.float32))

    def test_feet_not_box_centre_and_far_baseline_runoff_retained(self):
        p={'identity_id':'far','bbox':[40,-30,60,5], 'keypoints':{'left_ankle':[48,-20,.9],'right_ankle':[52,-20,.9]}}
        record=feet_record(p,self.court)
        self.assertIsNotNone(record)
        self.assertEqual(record['pixel'],[50,-20]);self.assertEqual(record['basis'],'two_ankles_ground')
        self.assertGreater(record['feet_xyz_m'][1],11.885+1.2)

    def test_fallback_prediction_labels_and_no_stale_player(self):
        p={'label':'player','identity_id':'near','bbox':[40,40,60,90], 'keypoints':{},'predicted':True}
        frames,_=build_players([{'frame':0,'tracks':[p]},{'frame':1,'tracks':[]}],self.court,None)
        self.assertEqual(frames[0]['players'][0]['basis'],'predicted_bbox_ground')
        self.assertEqual(frames[0]['players'][0]['joints_m'],{})
        self.assertEqual(frames[1]['players'],[])

    def test_no_anonymous_players_and_no_smoothing_across_gaps(self):
        p={'label':'player','identity_id':'near','bbox':[40,40,60,90], 'keypoints':{}}
        rows=[{'frame':0,'tracks':[p]},{'frame':1,'tracks':[]},{'frame':2,'tracks':[p,dict(p,identity_id=None)]}]
        frames,_=build_players(rows,self.court,None)
        self.assertEqual(len(frames[2]['players']),1)
        self.assertFalse(frames[2]['players'][0]['position_smoothed'])


class PreviewFlightTests(unittest.TestCase):
    def setUp(self):
        self.camera,self.balls,start,bounce,self.truth=arc_fixture()
        self.events=[dict(start,status='unreviewed',court_m=None),
                     dict(bounce,status='unreviewed',court_m=None,candidate_court_m=bounce['court_m'])]

    def preview(self,events=None,balls=None,cuts=()):
        return fit_preview_flights(self.camera,self.events if events is None else events,
                                   self.balls if balls is None else balls,30,cuts)

    def test_conditional_arc_recovered_without_confirming_events(self):
        before=copy.deepcopy((self.events,self.balls))
        result=self.preview();self.assertEqual(len(result['fits']),1,result)
        fit=result['fits'][0];self.assertEqual(fit['review_status'],'unvalidated_preview')
        self.assertTrue(fit['excluded_from_validated_flights'])
        self.assertEqual(fit['candidate_bounce_event_id'],'bounce')
        self.assertNotIn('bounce_event_id',fit)
        self.assertEqual(fit['start_frame_basis'],'observed_window_not_confirmed_contact')
        np.testing.assert_allclose([p['point_m'] for p in fit['trajectory_m']],self.truth[1:],atol=1e-4)
        self.assertEqual((self.events,self.balls),before)
        self.assertEqual(fit_reviewed_flights(self.camera,self.events,self.balls,30)['fits'],[])
        json.dumps(result,allow_nan=False)

    def test_rejected_landing_and_conflicting_events_never_previewed(self):
        self.events[1]['status']='rejected'
        self.assertEqual(self.preview()['fits'],[])
        self.events[1]['status']='unreviewed'
        events=self.events+[{'id':'conflict','frame':24,'type':'hit','status':'unreviewed'}]
        self.assertEqual(self.preview(events)['fits'],[])

    def test_cannot_bridge_unresolved_impact_but_rejected_event_is_ignored(self):
        barrier={'id':'impact','frame':15,'type':'uncertain','status':'uncertain'}
        self.assertEqual(self.preview(self.events+[barrier])['fits'],[])
        barrier['status']='rejected'
        self.assertEqual(len(self.preview(self.events+[barrier])['fits']),1)

    def test_missing_endpoint_or_long_gap_not_invented(self):
        balls=copy.deepcopy(self.balls);balls[-1]=None
        self.assertEqual(self.preview(balls=balls)['fits'],[])
        balls=copy.deepcopy(self.balls);balls[6:20]=[None]*14
        self.assertEqual(self.preview(balls=balls)['fits'],[])

    def test_short_gap_is_flagged_without_creating_2d_observations(self):
        self.balls[12]=None
        fit=self.preview()['fits'][0]
        self.assertEqual(fit['unobserved_frames'],[12])
        self.assertIsNone(self.balls[12])
        np.testing.assert_allclose(fit['trajectory_m'][11]['point_m'],self.truth[12],atol=1e-4)

    def test_camera_missing_or_cut_abstains(self):
        self.assertEqual(fit_preview_flights(None,self.events,self.balls,30)['fits'],[])
        self.assertEqual(self.preview(cuts=[5])['fits'],[])

    def test_missing_geometry_and_insufficient_duration(self):
        self.events[1]['candidate_court_m']=None
        self.assertEqual(self.preview()['fits'],[])
        self.events[1].update(candidate_court_m=[6,17],frame=9)
        self.assertEqual(self.preview()['fits'],[])

    def test_numerical_core_does_not_accept_invalid_bounds(self):
        target=np.array([1.,5.,BALL_RADIUS])
        for a,b,fps in [(-1,24,30),(0,30,30),(0,24,0),(0,24,float('nan'))]:
            self.assertIsNone(fit_landing_arc(self.camera,a,b,target,self.balls,fps)[0])
        self.assertIsNone(fit_landing_arc(self.camera,0,24,[float('nan'),5,.03],self.balls,30)[0])


class RefinedCameraTests(unittest.TestCase):
    def court(self,camera):
        corners=camera.project(np.array([[-5.485,-11.885,0],[5.485,-11.885,0],
                                        [5.485,11.885,0],[-5.485,11.885,0]]))
        return CourtMapper(corners.astype(np.float32))

    def test_recovers_known_focal_under_pinhole_assumptions(self):
        truth=make_camera();truth.K[0,0]=truth.K[1,1]=1024
        camera,report=refine_from_court(self.court(truth),1280,720)
        self.assertIsNotNone(camera);self.assertEqual(report['status'],'approximate')
        self.assertAlmostEqual(report['focal_ratio'],.8,places=3)
        self.assertLess(report['coordinate_rmse_px'],.01)
        np.testing.assert_allclose(camera.position_m,truth.position_m,atol=.01)
        self.assertGreater(report['prior_coordinate_rmse_px'],5)

    def test_boundary_optimum_not_silently_accepted(self):
        truth=make_camera();truth.K[0,0]=truth.K[1,1]=1280*.25
        camera,report=refine_from_court(self.court(truth),1280,720)
        self.assertIsNone(camera);self.assertIn('boundary',report['reason'])

    def test_invalid_dimensions_and_weak_frontal_view_rejected(self):
        truth=make_camera()
        self.assertIsNone(refine_from_court(self.court(truth),0,720)[0])
        truth.rvec=cv2.Rodrigues(np.diag([1.,-1.,-1.]))[0]
        truth.tvec=np.array([0.,0.,40.])
        self.assertIsNone(refine_from_court(self.court(truth),1280,720)[0])


class PackageIntegrationTests(unittest.TestCase):
    def test_full_export_with_synthetic_confirmed_events_and_without_labels(self):
        # Synthetic truth is kept in a temporary fixture, never written into the
        # user's real review labels or presented as a verified real-video arc.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';review=root/'review';run.mkdir();review.mkdir()
            camera,_,start,bounce,positions=arc_fixture()
            camera.K[0,0]=camera.K[1,1]=1024.
            pixels=camera.project(positions)
            raw=review/'source.mp4'
            writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
            if not writer.isOpened():self.skipTest('OpenCV MP4 encoder unavailable')
            for f in range(25):writer.write(np.full((720,1280,3),f*3,np.uint8))
            writer.release()
            summary={'input':str(raw),'frames':25,'fps':30.,'width':1280,'height':720}
            (run/'summary.json').write_text(json.dumps(summary))
            (run/'source-offset.json').write_text(json.dumps({'source_start_frame':0,'source_start_seconds':0}))
            rows=[{'frame':f,'time_s':f/30,'tracks':[{'label':'ball','bbox':[p[0]-2,p[1]-2,p[0]+2,p[1]+2],
                   'predicted':False,'source':'gridtracknet','confidence':.9}]} for f,p in enumerate(pixels)]
            (run/'events.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            corners=camera.project(np.array([[-5.485,-11.885,0],[5.485,-11.885,0],[5.485,11.885,0],[-5.485,11.885,0]]))
            court_path=root/'court.yaml';court_path.write_text(yaml.safe_dump({'image_corners':corners.tolist()}))
            court=CourtMapper(corners.astype(np.float32))
            run_id=hashlib.sha256((run/'events.jsonl').read_bytes()+json.dumps({'summary':summary,'start':0,
                 'court':court.image_corners.tolist(),'version':VERSION},sort_keys=True).encode()).hexdigest()
            candidates=[{'id':'start','frame':0,'type':'hit_candidate','player_id':'near'},
                        {'id':'bounce','frame':24,'type':'bounce_candidate','player_id':None}]
            (review/'review-report.json').write_text(json.dumps({'run_id':run_id,'scene_cut_frames':[]}))
            (review/'event-candidates.json').write_text(json.dumps({'run_id':run_id,'events':candidates}))
            (review/'build-status.json').write_text(json.dumps({'status':'complete'}))
            labels={'schema_version':1,'run_id':run_id,'manual_events':[],'labels':[
                 {'id':'start','type':'hit','frame':0,'status':'confirmed','player_id':'near'},
                 {'id':'bounce','type':'bounce','frame':24,'status':'confirmed','player_id':None}]}
            labels_path=root/'synthetic-labels.json';labels_path.write_text(json.dumps(labels))
            data=build(run,review,root/'with-labels',court_path,labels_path)
            self.assertEqual(data['report']['supported_flight_fits'],1,data['flights'])
            self.assertTrue(all(f['ball_3d'] is not None for f in data['frames']))
            self.assertTrue((root/'with-labels/replay.html').is_file())
            self.assertEqual(json.loads((root/'with-labels/build-status.json').read_text())['status'],'complete')
            no_labels=build(run,review,root/'without-labels',court_path)
            self.assertEqual(no_labels['flights']['fits'],[])
            self.assertTrue(all(f['ball_3d'] is None for f in no_labels['frames']))
            self.assertEqual(len(no_labels['preview_flights']['fits']),1)
            self.assertTrue(any(f['ball_3d_preview'] is not None for f in no_labels['frames']))
            self.assertTrue((root/'without-labels/preview-flight3d.json').is_file())
            self.assertTrue(all(e['status']=='unreviewed' for e in no_labels['events']))
            self.assertTrue(all(f['ball_3d_preview'] is None or f['ball_3d_preview']['status']=='unvalidated_preview'
                                for f in no_labels['frames']))
            # Per-stroke type/hand review round-trip does not confirm a contact.
            stroke_path=root/'synthetic-stroke-review.json'
            stroke_path.write_text(json.dumps({'schema_version':1,'package_id':no_labels['report']['stroke_review_binding'],
                'corrections':[{'event_id':'start','shot_type':'forehand','striking_hands':'both','status':'human_reviewed'}]}))
            stroke_replay=build(run,review,root/'stroke-reviewed',court_path,stroke_review_path=stroke_path)
            self.assertEqual(stroke_replay['shots']['shots'][0]['execution']['striking_hands'],'both')
            self.assertEqual(stroke_replay['shots']['shots'][0]['classification'],'forehand')
            self.assertEqual(stroke_replay['events'],no_labels['events'])
            self.assertEqual(stroke_replay['flights'],no_labels['flights'])
            self.assertEqual(stroke_replay['report']['stroke_review_binding'],no_labels['report']['stroke_review_binding'])
            self.assertNotEqual(stroke_replay['report']['package_id'],no_labels['report']['package_id'])
            from tennis_vision.court_refinement import LANDMARKS
            from tennis_vision.calibration_review import write_workspace
            xyz=np.array([[p[2]-5.485,p[3]-11.885,0.] for p in LANDMARKS])
            corrections={'schema_version':1,'kind':'court_and_bounce_review','run_id':run_id,
                         'image_size':[1280,720],'calibration_frame':0,'calibration_status':'draft',
                         'landmarks':{p[0]:xy.tolist() for p,xy in zip(LANDMARKS,camera.project(xyz))},
                         'bounce_edits':[{'id':'bounce','frame':24,'frame_range':[23,24],'status':'unreviewed',
                                          'landing_pixel':None,'notes':'synthetic fixture only'}]}
            corrections_path=root/'corrections.json';corrections_path.write_text(json.dumps(corrections))
            corrected=build(run,review,root/'calibrated',court_path,corrections_path=corrections_path)
            self.assertEqual(corrected['report']['review_run_id'],run_id)
            self.assertNotEqual(corrected['report']['package_id'],no_labels['report']['package_id'])
            self.assertEqual(corrected['flights']['fits'],[])
            self.assertEqual(corrected['events'][-1]['timing_review']['frame_range'],[23,24])
            self.assertEqual(corrected['report']['calibration_review']['calibration_status'],'draft')
            write_workspace(root/'calibrated',court_path)
            self.assertTrue((root/'calibrated/calibration.html').is_file())
            self.assertTrue((root/'calibrated/timing-frames/000024.jpg').is_file())
            self.assertEqual(json.loads((review/'event-candidates.json').read_text())['events'],candidates)
            with self.assertRaisesRegex(ValueError,'Overlapping'):
                build(run,review,root/'overlap',court_path,labels_path,corrections_path=corrections_path)


if __name__=='__main__':unittest.main()
