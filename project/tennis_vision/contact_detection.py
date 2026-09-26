"""Temporal contact candidates, with independent motion/equipment evidence."""
import copy
from collections import Counter
import numpy as np

from .event_analysis import observed_balls, _motion, _joint, _box_distance, _swing_speed
from .shot_classification import player_at, racket_at


def contact_evidence(rows, balls, f, fps, cuts=()):
    radius=max(3,round(fps*.167))
    if not balls[f] or any(abs(c-f)<=radius for c in cuts):
        return None
    motion=_motion(balls,f,radius)
    if (not motion or motion['velocity_change_px_frame'] < .8*30/fps
            or (motion['turn_degrees']<30 and motion['speed_ratio']<1.8)
            or motion['smooth_rmse_px']<=motion['corner_rmse_px']*1.10+.08):
        return None
    matches=[]
    pixel=np.array(balls[f]['pixel'])
    for identity in ('near','far'):
        p=player_at(rows,f,identity)
        if p is None: continue
        h=max(1,p['bbox'][3]-p['bbox'][1]);r=racket_at(rows[f],p)
        rd=_box_distance(pixel,r['bbox'])/h if r else None
        wd=min((float(np.linalg.norm(w-pixel))/h for n in ('left_wrist','right_wrist')
                if (w:=_joint(p,n)) is not None),default=None)
        swing=_swing_speed(rows,f,identity,radius)
        close_racket=rd is not None and rd<=.16
        close_wrist=wd is not None and wd<=.32
        if not (close_racket or close_wrist) or swing is None or swing<=.008*30/fps:
            continue
        rank=min(rd if rd is not None else 99,wd if wd is not None else 99)
        matches.append((rank,identity,{'racket_distance_heights':rd,'wrist_distance_heights':wd,
            'racquet_observed':bool(r),'body_relative_swing_heights_per_frame':swing,
            'pose_source':p.get('pose_source','original_model'),
            'identity_association_estimated':bool(p.get('identity_association_estimated'))}))
    matches.sort(key=lambda x:x[0])
    if not matches or (len(matches)>1 and matches[1][0]-matches[0][0]<.12):
        return None
    _,identity,evidence=matches[0]
    evidence.update({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in motion.items()})
    score=min(.9,.4+.15*evidence['racquet_observed']+min(.2,motion['turn_degrees']/600)
              +.1*(motion['corner_rmse_px']<2))
    return {'id':f'contact_candidate-{f:06d}','frame':f,'time_s':f/fps,'type':'hit',
        'status':'unreviewed','player_id':identity,'pixel':pixel.tolist(),
        'frame_range':motion['frame_range'],'support':'moderate' if evidence['racquet_observed'] else 'limited',
        'ranking_score':round(score,3),'score_is_probability':False,'evidence':evidence,
        'limitation':'2D proximity plus observed ball turn and swing, not measured ball/string contact. Timing is frame-level; confirmation requires review.'}


def detect_contacts(rows,fps,cuts=()):
    balls=observed_balls(rows)
    proposals=[p for f in range(len(rows)) if (p:=contact_evidence(rows,balls,f,fps,cuts))]
    selected=[]
    for p in sorted(proposals,key=lambda p:(p['evidence']['corner_rmse_px']/max(.01,p['evidence']['smooth_rmse_px']),-p['ranking_score'])):
        if not any(abs(p['frame']-q['frame'])<=round(.2*fps) for q in selected):
            selected.append(p)
    selected.sort(key=lambda p:p['frame'])
    # Review-only queue for distant swings with nearby observed ball samples but
    # no clean two-sided impact evidence. NOT detections or classification input.
    weak=[]
    for f,ball in enumerate(balls):
        if not ball or any(abs(c-f)<=round(.3*fps) for c in cuts):continue
        p=player_at(rows,f,'far')
        if p is None:continue
        h=max(1,p['bbox'][3]-p['bbox'][1])
        distance=min((float(np.linalg.norm(w-np.array(ball['pixel'])))/h for n in ('left_wrist','right_wrist')
                      if (w:=_joint(p,n)) is not None),default=99)
        swing=_swing_speed(rows,f,'far',max(3,round(.167*fps)))
        if distance<=.7 and swing is not None and swing>=.03*30/fps:
            weak.append((f,distance,swing))
    windows=[]
    for f,distance,swing in sorted(weak,key=lambda q:q[1]):
        if any(abs(f-q['frame'])<=round(.6*fps) for q in selected+windows):continue
        neighbours=[q for q in weak if abs(q[0]-f)<=round(.2*fps)]
        if len(neighbours)<2:continue
        windows.append({'id':f'contact_window-{f:06d}','frame':f,'time_s':f/fps,'type':'hit',
            'status':'unreviewed','player_id':'far','pixel':balls[f]['pixel'],
            'frame_range':[max(0,min(q[0] for q in neighbours)-3),min(len(rows)-1,max(q[0] for q in neighbours)+3)],
            'support':'review_window_only','evidence':{'nearby_observed_ball_frames':[q[0] for q in neighbours],
                'wrist_distance_heights':distance,'body_relative_swing_heights_per_frame':swing},
            'limitation':'Possible far-player swing window only. Ball-turn/contact evidence is insufficient; do not treat this as a detected hit.'})
    windows.sort(key=lambda p:p['frame'])
    return {'method':'observed_turn_equipment_swing_v2','contacts':selected,
            'review_windows':windows,
            'by_player':dict(Counter(p['player_id'] for p in selected)),
            'limitation':'Candidate scores rank evidence, not calibrated confidence probabilities. Missing ball observations and uncertain identities cause abstention.'}


def merge_contacts(events,packet,fps):
    """Keep all reviewed events immutable; enrich nearby hits, append new ones.

    A reviewed rejection/bounce suppresses new automatic hits in its window.
    Existing unreviewed hits without refreshed evidence remain visibly unsupported.
    """
    output=copy.deepcopy(events)
    for event in output:
        if event['type']=='hit': event['contact_support']='not_supported_by_new_pass'
    for contact in packet['contacts']:
        close=[e for e in output if abs(e['frame']-contact['frame'])<=round(.2*fps)]
        if any(e['status'] in ('confirmed','rejected','uncertain') and
               (e['type']!='hit' or e['status']!='confirmed') for e in close):
            continue
        hits=[e for e in close if e['type']=='hit' and e['player_id']==contact['player_id'] and e['status']!='rejected']
        if hits:
            event=min(hits,key=lambda e:abs(e['frame']-contact['frame']))
            event['contact_candidate']=copy.deepcopy(contact)
            event['contact_support']=contact['support']
        else:
            # Never compete with an explicitly reviewed hit of another identity.
            if any(e['status']=='confirmed' for e in close): continue
            event=copy.deepcopy(contact)
            event.update(original_frame=contact['frame'],pixel_basis='observed_ball',court_m=None,
                         candidate_court_m=None,geometry_issue=None,notes='',contact_support=contact['support'],
                         contact_candidate=copy.deepcopy(contact))
            output.append(event)
    for window in packet.get('review_windows',[]):
        if any(abs(e['frame']-window['frame'])<=round(.3*fps) and
               (e['type']=='hit' or e['status'] in ('confirmed','rejected','uncertain')) for e in output):continue
        event=copy.deepcopy(window)
        event.update(original_frame=window['frame'],pixel_basis='nearby_observed_ball_not_contact',
            court_m=None,candidate_court_m=None,geometry_issue=None,notes='',
            contact_support='review_window_only',contact_candidate=copy.deepcopy(window))
        output.append(event)
    return sorted(output,key=lambda e:(e['frame'],e['id']))


def apply_contact_review(events,payload,binding,balls,fps):
    """Explicit hit review only; cannot alter calibration or bounce decisions."""
    if (payload.get('schema_version')!=1 or payload.get('kind')!='contact_review'
            or payload.get('binding')!=binding or not isinstance(payload.get('labels'),list)):
        raise ValueError('Contact review does not match this exact contact analysis')
    output=copy.deepcopy(events);lookup={e['id']:e for e in output};seen=set()
    for label in payload['labels']:
        event=lookup.get(label.get('id'));f=label.get('frame')
        if (event is None or event['type']!='hit' or event['id'] in seen
                or type(f) is not int or not 0<=f<len(balls)
                or label.get('player_id') not in ('near','far')
                or label.get('status') not in ('confirmed','rejected','uncertain')):
            raise ValueError('Invalid or duplicate contact review label')
        # Small timing correction; unrelated/manual contacts belong in event review.
        centre=event.get('contact_candidate',event)['frame']
        if abs(f-centre)>round(.3*fps):
            raise ValueError('Contact correction is outside the local review window')
        seen.add(event['id'])
        event.update(frame=f,time_s=f/fps,status=label['status'],player_id=label['player_id'],
                     pixel=balls[f]['pixel'] if balls[f] else None,
                     pixel_basis='human_review_frame_observed_ball' if balls[f] else 'missing_ball_at_reviewed_frame',
                     contact_review_basis='explicit_human_review')
    return sorted(output,key=lambda e:(e['frame'],e['id']))
