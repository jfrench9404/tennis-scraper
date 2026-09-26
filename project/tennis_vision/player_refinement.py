"""Fresh, tightly cropped far-player pose inference; no synthetic joints.

Identity comes from the existing tracked player, never an unrestricted person
search. Failed/ambiguous crops leave the original record intact. This improves
pose evidence, not guaranteed anatomical accuracy or long-gap re-identification.
"""
import copy
from collections import Counter
from types import SimpleNamespace

import cv2
import numpy as np
from tqdm import tqdm

from .cached_identity import box_iou
from .scene import appearance

JOINTS = ('nose','left_eye','right_eye','left_ear','right_ear','left_shoulder',
          'right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
          'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle')
BODY = JOINTS[5:]


def pose_quality(track):
    values = track.get('keypoints') or {}
    scores = [float(values.get(k, [0,0,0])[2]) for k in BODY]
    return sum(s >= .3 for s in scores), float(np.mean(scores))


def crop_box(box, shape):
    h,w = shape[:2]
    x1,y1,x2,y2 = box
    height = max(32,y2-y1)
    cx,cy = (x1+x2)/2,(y1+y2)/2
    # Room for extended arms, racquet and overhead swings; not the whole court.
    radius = max(48, .95*height)
    return [max(0,int(cx-radius)),max(0,int(cy-radius)),
            min(w,int(cx+radius)),min(h,int(cy+radius))]


def choose_pose(candidates, anchor, image, court):
    """Associate a fresh observed box to a same-frame identity anchor."""
    b = anchor['bbox']; height = max(1,b[3]-b[1])
    centre = (np.array(b[:2])+b[2:])/2
    ref = appearance(image,SimpleNamespace(bbox=b))
    choices = []
    for p in candidates:
        a = p['bbox']; scale = (a[3]-a[1])/height
        distance = float(np.linalg.norm((np.array(a[:2])+a[2:])/2-centre)/height)
        hist = appearance(image,SimpleNamespace(bbox=a))
        colour = float(cv2.compareHist(ref,hist,cv2.HISTCMP_BHATTACHARYYA)) if ref is not None and hist is not None else 1.
        ground = court.project(((a[0]+a[2])/2,a[3]))
        # Far half plus runoff. Wide retrieval is allowed, not adjacent courts.
        if (not np.isfinite(ground).all() or not -2.5 <= ground[0] <= 13.47
                or not 10.5 <= ground[1] <= 30.27):
            continue
        if not (.6 <= scale <= 1.65 and distance <= .45 and box_iou(a,b) >= .2
                and colour <= .6 and p['confidence'] >= .2):
            continue
        if pose_quality(p)[0] < 6:
            continue
        choices.append((distance+.3*colour+.2*abs(np.log(scale)),p))
    choices.sort(key=lambda p:p[0])
    if not choices:
        return None,'no_supported_crop_detection'
    unique=[]
    for c in choices:
        if not any(box_iou(c[1]['bbox'],q[1]['bbox']) > .65 for q in unique):
            unique.append(c)
    if len(unique)>1 and unique[1][0]-unique[0][0]<.15:
        return None,'ambiguous_crop_people'
    return unique[0][1],None


def infer_crop(model, image, region, imgsz=640):
    x1,y1,x2,y2=region
    result=model.predict(image[y1:y2,x1:x2],conf=.15,imgsz=imgsz,verbose=False,device='cpu')[0]
    if result.boxes is None or result.keypoints is None:
        return []
    people=[]
    for box,joints in zip(result.boxes,result.keypoints.data.cpu().tolist()):
        a,b,c,d=map(float,box.xyxy[0].tolist())
        kp={name:[round(p[0]+x1,3),round(p[1]+y1,3),round(p[2],4)] for name,p in zip(JOINTS,joints)}
        people.append({'label':'player','bbox':[a+x1,b+y1,c+x1,d+y1],
                       'confidence':float(box.conf[0]),'keypoints':kp,'source':'model','predicted':False})
    return people


def refine_far_player(rows, source, court, model, cuts=(), infer=infer_crop):
    output=copy.deepcopy(rows); counts=Counter(); before=Counter(); after=Counter()
    cap=cv2.VideoCapture(str(source))
    try:
        for f in tqdm(range(len(rows)),desc='Far-player crop pose',unit='frame'):
            ok,image=cap.read()
            if not ok:
                raise ValueError(f'Source decode failed at frame {f}')
            matches=[p for p in rows[f]['tracks'] if p.get('label')=='player' and p.get('identity_id')=='far']
            if len(matches)!=1 or matches[0].get('predicted') or f in cuts:
                counts['no_observed_identity_anchor']+=1
                continue
            anchor=matches[0]; oldq=pose_quality(anchor); before['body_joints']+=oldq[0]
            before['pose_frames']+=int(oldq[0]>=6)
            region=crop_box(anchor['bbox'],image.shape)
            candidates=infer(model,image,region)
            counts['inferred_frames']+=1
            picked,reason=choose_pose(candidates,anchor,image,court)
            if picked is not None:
                newq=pose_quality(picked)
                # Replace complete skeletons only, never mix incompatible poses.
                # Existing well-supported poses win ties; more pixels isn't truth.
                if newq[0] > oldq[0] or (newq[0] >= oldq[0] and newq[1] > oldq[1]+.06):
                    replacement=copy.deepcopy(anchor)
                    replacement.update(keypoints=picked['keypoints'],pose_source='focused_crop_model',
                        pose_crop=region,pose_detector_score=picked['confidence'],
                        pose_refinement_basis='fresh_observation_same_frame_identity_gate')
                    # Keep established identity/bbox; fresh joints inform foot anchors.
                    index=next(i for i,p in enumerate(rows[f]['tracks']) if p is anchor)
                    output[f]['tracks'][index]=replacement
                    counts['replaced_pose_frames']+=1
                    oldq=newq
                else: counts['original_pose_retained']+=1
            else: counts[reason]+=1
            after['body_joints']+=oldq[0];after['pose_frames']+=int(oldq[0]>=6)
    finally:
        cap.release()
    return output,{'method':'identity_anchored_crop_pose_v1','counts':dict(counts),
        'before':dict(before),'after':dict(after),
        'limitation':'Fresh crop detections, not interpolated joints. Joint coverage is not accuracy. Missing identity anchors remain missing; no unrestricted referee/bystander reacquisition.'}
