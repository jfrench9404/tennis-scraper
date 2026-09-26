"""Single-view, multi-point court refinement with a constrained radial lens model.

This is approximate calibration, not a substitute for a multi-view calibration
rig. Image points must be line intersections, never adjusted ball landings.
"""
import cv2
import numpy as np

from .camera3d import Camera3D


LANDMARKS = [
    ("near_left_doubles","Near left doubles corner",0.,0.),
    ("near_right_doubles","Near right doubles corner",10.97,0.),
    ("far_right_doubles","Far right doubles corner",10.97,23.77),
    ("far_left_doubles","Far left doubles corner",0.,23.77),
    ("near_left_singles","Near baseline / left singles",1.37,0.),
    ("near_right_singles","Near baseline / right singles",9.60,0.),
    ("far_left_singles","Far baseline / left singles",1.37,23.77),
    ("far_right_singles","Far baseline / right singles",9.60,23.77),
    ("near_service_left","Near service / left singles",1.37,5.485),
    ("near_service_centre","Near service T",5.485,5.485),
    ("near_service_right","Near service / right singles",9.60,5.485),
    ("far_service_left","Far service / left singles",1.37,18.285),
    ("far_service_centre","Far service T",5.485,18.285),
    ("far_service_right","Far service / right singles",9.60,18.285),
    ("near_baseline_centre","Near baseline centre mark / line",5.485,0.),
]
LOOKUP = {p[0]:p for p in LANDMARKS}


def validate_review(payload,run_id,width,height,frames,candidates):
    if (not isinstance(payload,dict) or payload.get('schema_version')!=1
            or payload.get('kind')!='court_and_bounce_review' or payload.get('run_id')!=run_id
            or payload.get('image_size')!=[width,height]):
        raise ValueError('Court review does not match this exact run and image size')
    f=payload.get('calibration_frame')
    if type(f) is not int or not 0<=f<frames or payload.get('calibration_status') not in ('draft','reviewed'):
        raise ValueError('Invalid calibration frame/status')
    if not isinstance(payload.get('landmarks'),dict) or not isinstance(payload.get('bounce_edits'),list):
        raise ValueError('Missing landmarks or bounce edits')
    ids={e['id'] for e in candidates if e['type']=='bounce_candidate'}
    seen=set()
    for e in payload['bounce_edits']:
        if (not isinstance(e,dict) or e.get('id') not in ids or e['id'] in seen
                or e.get('status') not in ('unreviewed','uncertain','confirmed','rejected')
                or type(e.get('frame')) is not int or not 0<=e['frame']<frames
                or not isinstance(e.get('notes',''),str) or len(e.get('notes',''))>10000):
            raise ValueError('Invalid/duplicate bounce correction')
        seen.add(e['id'])
        span=e.get('frame_range')
        if (not isinstance(span,list) or len(span)!=2 or any(type(v) is not int for v in span)
                or not 0<=span[0]<=e['frame']<=span[1]<frames):
            raise ValueError('Bounce timing range must contain the selected frame')
        pixel=e.get('landing_pixel')
        if pixel is not None and (not isinstance(pixel,list) or len(pixel)!=2 or not np.isfinite(pixel).all()
                                  or not 0<=pixel[0]<width or not 0<=pixel[1]<height):
            raise ValueError('Invalid corrected landing pixel')
    return payload


class RefinedCourt:
    source = "multipoint_radial_draft"
    confidence = .45

    def __init__(self,camera,points,report):
        self.camera,self.calibration_report = camera,report
        self.image_corners = np.asarray([points[p[0]] for p in LANDMARKS[:4]],np.float32)
        # Keep one camera geometry for ground projection, joints and ball flight.
        r,t = camera.R,camera.tvec.reshape(3)
        ground_to_undistorted = camera.K@np.column_stack((r[:,0],r[:,1],t-5.485*r[:,0]-11.885*r[:,1]))
        self.matrix = np.linalg.inv(ground_to_undistorted)

    def project(self,pixel):
        world = self.camera.pixel_to_plane(tuple(pixel),0.)
        if world is None:
            return [float('nan'),float('nan')]
        return [round(float(world[0]+5.485),3),round(float(world[1]+11.885),3)]


def fit_court(points,width,height):
    if not isinstance(points,dict) or set(points)-set(LOOKUP):
        raise ValueError("Unknown court landmark IDs")
    if not all(p[0] in points for p in LANDMARKS[:4]) or len(points)<12:
        raise ValueError("Use all four doubles corners and at least 12 distributed landmarks")
    if not all(k in points for k in ('near_service_centre','far_service_centre','near_baseline_centre')):
        raise ValueError("Include both service T points and the near baseline centre")
    if type(width) is not int or type(height) is not int or min(width,height)<=0:
        raise ValueError("Invalid image dimensions")
    names=list(points)
    pixels=np.asarray([points[k] for k in names],np.float32)
    world=np.asarray([[LOOKUP[k][2]-5.485,LOOKUP[k][3]-11.885,0.] for k in names],np.float32)
    if (pixels.shape!=(len(names),2) or not np.isfinite(pixels).all()
            or np.any(pixels<0) or np.any(pixels[:,0]>=width) or np.any(pixels[:,1]>=height)):
        raise ValueError("Landmarks must be finite in-frame pixels")
    if abs(cv2.contourArea(np.asarray([points[k[0]] for k in LANDMARKS[:4]],np.float32)))<.05*width*height:
        raise ValueError("Court corners are degenerate or too small")
    solutions=[]
    flags=(cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_PRINCIPAL_POINT|cv2.CALIB_FIX_ASPECT_RATIO
           |cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_K3)
    for ratio in (.65,.9,1.2):
        K=np.array([[width*ratio,0,width/2],[0,width*ratio,height/2],[0,0,1.]],float)
        try:
            rms,K,dist,rs,ts=cv2.calibrateCamera([world],[pixels],(width,height),K,np.zeros(5),flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,150,1e-9))
            camera=Camera3D(K,rs[0],ts[0],(width,height),.45,dist.reshape(-1))
            if (not np.isfinite(rms) or not .35*width<K[0,0]<3*width or camera.position_m[2]<=0
                    or np.any((camera.R@world.T+camera.tvec.reshape(3,1))[2]<=0)):
                continue
            # Reject folded radial mappings over the visible court and image.
            grid=np.array([[x,y] for x in np.linspace(0,width,9) for y in np.linspace(0,height,7)],float)
            uv=camera.normalized_pixels(grid)
            radii=np.linspace(0,max(np.linalg.norm(uv,axis=1)),100)
            k1,k2=dist.ravel()[:2]
            if np.any(1+3*k1*radii**2+5*k2*radii**4<.15):
                continue
            errors=np.linalg.norm(camera.project(world)-pixels,axis=1)
            if max(errors)>12 or rms>5:
                continue
            solutions.append((float(rms),camera,errors))
        except (cv2.error,ValueError,np.linalg.LinAlgError):
            continue
    if not solutions:
        raise ValueError("No consistent lens/court fit; check misplaced or swapped landmarks")
    rms,camera,errors=min(solutions,key=lambda row:row[0])
    report={"basis":"multi_point_radial_camera","status":"approximate_not_measurement_grade",
            "landmarks":len(names),"radial_distortion":camera.distortion.tolist(),
            "focal_length_px":float(camera.K[0,0]),"landmark_rmse_px":rms,
            "residuals_px":{k:round(float(e),3) for k,e in zip(names,errors)},
            "assumptions":"one fixed view; square pixels; centred principal point; k1/k2 radial distortion; no tangential distortion",
            "limitation":"Residuals use fitted landmarks, not independent 3D validation. Review all projected court lines; a single view cannot fully validate camera intrinsics or airborne XYZ."}
    return RefinedCourt(camera,points,report),camera,report
