"""Conservative raw-footage context, separate from human event decisions.

All phases are hypotheses. Missing observations yield uncertainty, never an ace,
point-ending, or a fabricated return. Ground projection is NOT used for air balls.
"""
import copy
from collections import Counter
import numpy as np

from .event_analysis import observed_balls, _box_distance
from .shot_classification import player_at, _temporal_features


def flight_context(rows, balls, frame, identity, fps, direction=1, cuts=()):
    p=player_at(rows,frame,identity)
    if p is None or balls[frame] is None:
        return {'relation':'unresolved','reason':'Missing player or contact-frame ball'}
    height=max(1,p['bbox'][3]-p['bbox'][1]);start=np.array(balls[frame]['pixel'])
    other='far' if identity=='near' else 'near'
    samples=[];last=frame;lastpixel=start;stop='window_end'
    for f in range(frame+direction,frame+direction*(round(.9*fps)+1),direction):
        if not 0<=f<len(rows):stop='clip_boundary';break
        if any(min(last,f)<c<=max(last,f) for c in cuts):stop='camera_cut';break
        if balls[f] is None:
            if abs(f-last)>3:stop='observation_gap';break
            continue
        point=np.array(balls[f]['pixel']);step=float(np.linalg.norm(point-lastpixel))/abs(f-last)
        if step>max(45.,height*.3):stop='detector_jump';break
        hitter=player_at(rows,f,identity);receiver=player_at(rows,f,other)
        samples.append({'frame':f,'pixel':point,'distance':float(np.linalg.norm(point-start))/height,
            'hitter_gap':_box_distance(point,hitter['bbox'])/height if hitter else None,
            'receiver_gap':_box_distance(point,receiver['bbox'])/max(1,receiver['bbox'][3]-receiver['bbox'][1]) if receiver else None})
        last,lastpixel=f,point
    result={'relation':'unresolved','observed_frames':[s['frame'] for s in samples],'stop_reason':stop,
            'span_frames':abs(last-frame),'sample_count':len(samples)}
    if len(samples)<6 or abs(last-frame)<round(.23*fps):
        result['reason']='Too little continuous ball evidence';return result
    gaps=[s['hitter_gap'] for s in samples if s['hitter_gap'] is not None]
    excursion=max(s['distance'] for s in samples)
    near_fraction=sum(d<.25 for d in gaps)/max(1,len(gaps))
    other0=player_at(rows,frame,other)
    reference_frame=frame
    if other0 is None:
        # A receiver may initialize slightly later. Use a nearby observed pose
        # as a coarse court-direction reference, never as contact-frame evidence.
        reference_frame=next((s['frame'] for s in samples if player_at(rows,s['frame'],other)),None)
        other0=player_at(rows,reference_frame,other) if reference_frame is not None else None
    approaching=False
    if other0:
        receiver_height=max(1,other0['bbox'][3]-other0['bbox'][1])
        initial=_box_distance(start,other0['bbox'])/receiver_height
        ending=[s['receiver_gap'] for s in samples[-6:] if s['receiver_gap'] is not None]
        # Multiple observations approaching the other player's region, not one
        # detector jump onto their shirt. Far balls can be tiny in source pixels.
        approaching=len(ending)>=3 and sum(d<1.8 for d in ending)>=2 and initial-min(ending)>.8
    detached=sum(d>.35 for d in gaps)>=3
    result.update(excursion_body_heights=round(excursion,3),near_hitter_fraction=round(near_fraction,3),
                  approaches_opponent=approaching,detached_from_hitter=detached,receiver_reference_frame=reference_frame)
    if excursion>.35 and approaching and detached:
        result.update(relation='court_flight',reason='Continuous ball travel separates from hitter and approaches opponent region')
    elif len(gaps)>=6 and near_fraction>=.8 and not detached:
        result.update(relation='local_activity',reason='Observed ball stays with the player; no court-directed flight established')
    else:
        result['reason']='Neither court flight nor sustained local activity is established'
    return result


def assess_play(rows,events,fps,cuts=()):
    balls=observed_balls(rows);output=copy.deepcopy(events)
    phases=[{'phase':'unresolved','basis':'insufficient_evidence'} for _ in rows]
    assessments=[]
    for e in output:
        if e['type']!='hit':continue
        f,identity=e['frame'],e.get('player_id')
        if identity not in ('near','far'):
            incoming=outgoing={'relation':'unresolved','reason':'Unknown player'};features={}
        else:
            outgoing=flight_context(rows,balls,f,identity,fps,1,cuts)
            incoming=flight_context(rows,balls,f,identity,fps,-1,cuts)
            features=_temporal_features(rows,balls,e,fps,'unknown')
        feet=features.get('player_feet_court_m')
        baseline=feet is not None and (feet[1]<=3 or feet[1]>=20.77)
        serve=bool(baseline and features.get('overhead_contact_frames',0)>=1
                   and features.get('pre_contact_raised_arms') and outgoing['relation']=='court_flight')
        strong=e.get('contact_support') in ('moderate','limited')
        state='uncertain_contact';reason='A swing/proximity or missing ball track is not enough to establish a shot'
        if e['status']=='rejected':state='no_shot';reason='Human rejected this contact'
        elif e['status']=='confirmed':state='confirmed_shot';reason='Human confirmed this contact'
        elif strong and outgoing['relation']=='court_flight' and (serve or incoming['relation']=='court_flight'):
            state='shot_candidate';reason='Contact cues plus serve context or incoming/outgoing court flight'
        elif outgoing['relation']=='local_activity' and incoming['relation']=='local_activity':
            state='no_shot_candidate';reason='Ball remains near the player on both sides; likely ball handling/non-play activity'
        phase=('serve_candidate' if serve else 'live_ball_candidate') if state in ('shot_candidate','confirmed_shot') else (
            'between_points_candidate' if state=='no_shot_candidate' else 'unresolved')
        assessment={'event_id':e['id'],'frame':f,'action_state':state,'phase':phase,'reason':reason,
            'incoming':incoming,'outgoing':outgoing,'serve_pattern':serve,'is_human_decision':e['status'] in ('confirmed','rejected')}
        e['play_assessment']=assessment;assessments.append(assessment)
        if state in ('shot_candidate','confirmed_shot'):
            # No claim of point outcome; stop at unsupported ball intervals.
            if serve:
                for k in range(max(0,f-round(.9*fps)),f):
                    if phases[k]['phase']=='unresolved':phases[k]={'phase':'preparation_candidate','basis':e['id']}
            phases[f]={'phase':phase,'basis':e['id']}
            for k in outgoing.get('observed_frames',[]):
                phases[k]={'phase':'live_ball_candidate','basis':e['id']}
        elif state=='no_shot_candidate':
            for k in incoming.get('observed_frames',[])+[f]+outgoing.get('observed_frames',[]):
                if phases[k]['phase']=='unresolved':phases[k]={'phase':phase,'basis':e['id']}
    return output,{'method':'raw_play_context_v1','assessments':assessments,'frames':phases,
        'action_counts':dict(Counter(a['action_state'] for a in assessments)),
        'phase_counts':dict(Counter(p['phase'] for p in phases)),
        'outcome':'undetermined','limitation':'Heuristic play phases, not verified rally boundaries. Local activity is a no-shot hypothesis; missing ball observations never establish an ace, fault, winner, or return.'}
