import copy
import unittest
import numpy as np

from test_shot_replay import make_camera,arc_fixture
from tennis_vision.court_refinement import LANDMARKS,fit_court,validate_review
from tennis_vision.validated_flight import fit_interval


class CourtRefinementTests(unittest.TestCase):
    def setUp(self):
        self.camera=make_camera()
        self.camera.distortion=np.array([-.2,.04,0,0,0.])
        xyz=np.array([[p[2]-5.485,p[3]-11.885,0.] for p in LANDMARKS])
        self.points={p[0]:xy.tolist() for p,xy in zip(LANDMARKS,self.camera.project(xyz))}

    def test_known_radial_camera_and_ground_round_trip(self):
        court,camera,report=fit_court(self.points,1280,720)
        self.assertLess(report['landmark_rmse_px'],.01)
        np.testing.assert_allclose(camera.distortion,self.camera.distortion,atol=.002)
        for ground in ([10.285,9.98],[6.,17.7],[1.5,3.],[5.485,0]):
            pixel=self.camera.project([[ground[0]-5.485,ground[1]-11.885,0.]])[0]
            np.testing.assert_allclose(court.project(pixel),ground,atol=.003)
        self.assertIn('not independent',report['limitation'])

    def test_distorted_flight_uses_undistorted_rays(self):
        camera,balls,start,bounce,truth=arc_fixture()
        camera.distortion=np.array([-.2,.04,0,0,0.])
        balls=[{'pixel':p.tolist()} for p in camera.project(truth)]
        fit,reason=fit_interval(camera,start,bounce,balls,30)
        self.assertIsNone(reason)
        np.testing.assert_allclose([p['point_m'] for p in fit['trajectory_m']],truth,atol=.001)

    def test_missing_unknown_and_degenerate_landmarks_rejected(self):
        for points in (dict(list(self.points.items())[:8]),dict(self.points,unknown=[1,2]),
                       {k:[500,300] for k in self.points}):
            with self.assertRaises(ValueError):fit_court(points,1280,720)
        bad=copy.deepcopy(self.points);bad['far_service_centre']=[-1,20]
        with self.assertRaises(ValueError):fit_court(bad,1280,720)

    def test_large_misplaced_landmark_is_not_silently_ignored(self):
        bad=copy.deepcopy(self.points);bad['near_service_centre'][0]+=160
        with self.assertRaises(ValueError):fit_court(bad,1280,720)


class CorrectionBindingTests(unittest.TestCase):
    def setUp(self):
        self.payload={'schema_version':1,'kind':'court_and_bounce_review','run_id':'test',
                      'image_size':[1280,720],'calibration_frame':10,'calibration_status':'draft',
                      'landmarks':{},'bounce_edits':[{'id':'b','frame':18,'frame_range':[17,19],
                                                   'status':'unreviewed','landing_pixel':None,'notes':''}]}
        self.candidates=[{'id':'b','type':'bounce_candidate'}]

    def validate(self,p):return validate_review(p,'test',1280,720,25,self.candidates)

    def test_validation_preserves_draft_and_does_not_confirm(self):
        original=copy.deepcopy(self.payload)
        self.validate(self.payload)
        self.assertEqual(self.payload,original)
        self.assertEqual(self.payload['bounce_edits'][0]['status'],'unreviewed')

    def test_other_runs_dimensions_and_bad_frame_ranges_rejected(self):
        for updates in ({'run_id':'other'},{'image_size':[720,1280]},{'calibration_frame':25},
                        {'calibration_status':'auto_confirmed'}):
            with self.assertRaises(ValueError):self.validate(dict(self.payload,**updates))
        for updates in ({'frame':25},{'frame_range':[19,20]},{'frame_range':[17,99]},
                        {'landing_pixel':[float('nan'),50]},{'landing_pixel':[1300,30]},
                        {'id':'other'},{'status':'automatic_truth'}):
            p=copy.deepcopy(self.payload);p['bounce_edits'][0].update(updates)
            with self.assertRaises(ValueError):self.validate(p)

    def test_duplicate_bounce_decisions_rejected(self):
        self.payload['bounce_edits']*=2
        with self.assertRaises(ValueError):self.validate(self.payload)


if __name__=='__main__':unittest.main()
