"""Per-stroke hand usage; no fixed handedness restriction or grip measurement."""
from collections import Counter
import copy
import numpy as np
from .event_analysis import _joint,_box_distance


def execution_evidence(rows,event,fps,cuts=()):
    f,identity=event['frame'],event.get('player_id');votes=Counter();side_samples=[]
    # Import locally to keep the classifier/execution module dependency acyclic.
    from .shot_classification import player_at,racket_at
    radius=max(2,round(.13*fps))
    result={'striking_hands':'unknown','execution':'unknown','stroke_side':'unknown',
            'swing_type':'unknown','basis':'insufficient_per_stroke_equipment_evidence',
            'votes':{},'support_frames':[],
            'limitation':'Wrist/racquet proximity is not finger/grip measurement. Two nearby hands do not prove a two-handed grip or a backhand.'}
    if identity not in ('near','far') or any(abs(c-f)<=radius for c in cuts):return result
    for k in range(max(0,f-radius),min(len(rows),f+radius+1)):
        p=player_at(rows,k,identity)
        if not p:continue
        r=racket_at(rows[k],p)
        if not r:continue
        left,right=_joint(p,'left_wrist'),_joint(p,'right_wrist')
        if left is None or right is None:continue
        h=max(1,p['bbox'][3]-p['bbox'][1])
        dl,dr=_box_distance(left,r['bbox'])/h,_box_distance(right,r['bbox'])/h
        hand=None
        if max(dl,dr)<.10 and np.linalg.norm(left-right)/h<.16:hand='both'
        elif dl<.14 and dr-dl>.07:hand='left'
        elif dr<.14 and dl-dr>.07:hand='right'
        if hand is None:continue
        votes[hand]+=1;result['support_frames'].append(k)
        ls,rs=_joint(p,'left_shoulder'),_joint(p,'right_shoulder')
        if ls is not None and rs is not None and k<=f:
            axis=rs-ls;n=float(np.linalg.norm(axis))
            if n/h>=.1:
                wrist=(left+right)/2 if hand=='both' else left if hand=='left' else right
                side_samples.append((hand,float(np.dot(wrist-(ls+rs)/2,axis/n)/h)))
    result['votes']=dict(votes)
    if not votes:return result
    hand,n=votes.most_common(1)[0]
    if n<3 or n/sum(votes.values())<.75:return result
    result.update(striking_hands=hand,execution='two_handed_candidate' if hand=='both' else 'one_handed_candidate',
                  basis='local_racquet_wrist_sequence_not_player_dominance')
    samples=[v for label,v in side_samples if label==hand and abs(v)>.08]
    if len(samples)>=3:
        right=sum(v>0 for v in samples)/len(samples)
        if max(right,1-right)>=.75:
            result['stroke_side']='anatomical_right' if right>=.75 else 'anatomical_left'
            # Two hands require grip/orientation evidence or human review to
            # distinguish two-handed forehand from backhand. Do not guess.
            if hand!='both':
                same=(right>=.75)==(hand=='right')
                result['swing_type']='forehand' if same else 'backhand'
    return result


def apply_stroke_review(shots,payload,package_id):
    """Independent stroke/hand annotations, bound to a replay. Never confirms contact."""
    if payload.get('schema_version')!=1 or payload.get('package_id')!=package_id or not isinstance(payload.get('corrections'),list):
        raise ValueError('Stroke review does not match this replay; export from the matching replay')
    output=copy.deepcopy(shots);lookup={s['event_id']:s for s in output['shots']};seen=set()
    for label in payload['corrections']:
        eid=label.get('event_id');kind=label.get('shot_type');hand=label.get('striking_hands')
        if (eid not in lookup or eid in seen or label.get('status')!='human_reviewed'
                or kind not in ('unknown','serve','forehand','backhand','volley','overhead','not_a_shot')
                or hand not in (None,'unknown','left','right','both')):
            raise ValueError('Invalid/duplicate stroke review')
        seen.add(eid);s=lookup[eid]
        s.update(classification=kind,classification_status='human_reviewed')
        if s['action_state'] in ('shot_candidate','confirmed_shot'):
            s['display_label']=kind if kind!='unknown' else 'unknown stroke'
        if hand is not None:
            s['execution'].update(striking_hands=hand,execution='two_handed' if hand=='both' else
                'one_handed' if hand in ('left','right') else 'unknown',basis='explicit_per_stroke_human_review')
        s['reasons'].append('Stroke/hand feedback is separate from contact confirmation')
    eligible=[s for s in output['shots'] if s['action_state'] in ('shot_candidate','confirmed_shot') and s['classification']!='not_a_shot']
    output['counts']=dict(Counter(s['classification'] for s in eligible))
    output['by_player']={identity:dict(Counter(s['classification'] for s in eligible if s['player_id']==identity)) for identity in ('near','far')}
    return output
