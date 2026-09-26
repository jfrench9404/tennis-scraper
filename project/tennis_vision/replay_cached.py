"""Re-evaluate saved candidates without loading or rerunning neural networks."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2

from .ball_motion import MotionBallRecovery
from .filters import filter_tracks
from .scene import SceneSelector
from .tracking import Detection, NearestTracker


def replay(run, output, scene_path, motion_ball=False):
    summary=json.loads((run/'summary.json').read_text())
    offset=json.loads((run/'source-offset.json').read_text())
    config=json.loads(scene_path.read_text())
    if not all(config.get(key) for key in ('near_seed_polygon','far_seed_polygon','ball_polygon')):
        raise ValueError('Cached review requires explicit scene polygons (no court calibration is reloaded).')
    output.mkdir(parents=True,exist_ok=False)
    cap=cv2.VideoCapture(summary['input'])
    if not cap.isOpened(): raise RuntimeError('Cannot open original video')
    start=offset['source_start_frame']
    if start and not cap.set(cv2.CAP_PROP_POS_FRAMES,start):
        cap.release(); raise RuntimeError('Cannot seek original video')
    writer=cv2.VideoWriter(str(output/'review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),
                           summary['fps'],(summary['width'],summary['height']))
    if not writer.isOpened():
        cap.release(); raise RuntimeError('Cannot create review video')
    selector=SceneSelector(config,summary['fps'])
    recovery=MotionBallRecovery()
    tracker=NearestTracker()
    totals=Counter(); original=Counter(); frames=0
    try:
        with (run/'events.jsonl').open() as stream, (output/'review-tracks.jsonl').open('w') as out:
            for line in stream:
                row=json.loads(line)
                if row['frame']!=frames:
                    raise ValueError('Cached frames must be contiguous and start at zero')
                ok,image=cap.read()
                if not ok: raise RuntimeError('Source ended before cached detections')
                raw=row['detail'].get('candidates')
                if raw is None: raise ValueError('Run lacks pre-filter candidates; rerun analyzer to record them')
                detections=[Detection(d['label'],tuple(d['bbox']),d['confidence'],d.get('keypoints'),source=d.get('source','model')) for d in raw]
                kept,stats=selector.update(image,detections,None)
                motion={}
                if motion_ball:
                    players=[d for d in kept if d.label=='player']
                    kept,motion=recovery.update(image,kept,lambda p:selector.ball_allowed(p,image.shape,None,players),detections)
                tracks,_=filter_tracks(tracker.update(kept,frames),None)
                records=[t.as_dict(None) for t in tracks]
                totals.update({(d.identity_id or d.label)+('_motion' if d.source=='motion' else '')
                               for d in kept if d.label!='racket'})
                original.update({d.get('identity_id') or d['label'] for d in row['tracks'] if not d['predicted'] and d['label']!='racket'})
                out.write(json.dumps({'frame':frames,'tracks':records,'scene':stats,'motion':motion})+'\n')
                for d in records:
                    x1,y1,x2,y2=map(int,d['bbox'])
                    colour=(0,165,255) if d['source']=='motion' else ((0,255,255) if d['label']=='ball' else (255,180,0))
                    if d['label']=='racket': colour=(255,0,255)
                    label=d['identity_id'] or d['label']
                    if d['source']=='motion': label+=' (motion candidate)'
                    if d['predicted']: label+=' (predicted)'
                    cv2.rectangle(image,(x1,y1),(x2,y2),colour,1 if d['predicted'] else 2)
                    cv2.putText(image,label,(x1,max(18,y1-5)),0,.5,colour,1)
                cv2.putText(image,f'Cached review | frame {frames} | orange = tentative motion ball',
                            (15,summary['height']-15),0,.5,(255,255,255),1)
                writer.write(image)
                frames+=1
                if frames%150==0: print(f'Reviewed {frames} frames',flush=True)
    finally:
        cap.release(); writer.release()
    report={'frames':frames,'original_observed_frames':dict(original),'review_observed_frames':dict(totals),
            'source_run':str(run.resolve()),'motion_ball':motion_ball,
            'limitation':'Counts are presence, NOT ground-truth accuracy. Reuses cached detections; no new neural inference. Older caches omit rejected poses, affecting racket association. Motion balls are tentative and not used for events. No calibrated 3D coordinates or new shot analysis in this review.'}
    (output/'review-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--motion-ball',action='store_true')
    args=parser.parse_args()
    replay(args.run,args.output,args.scene,args.motion_ball)


if __name__=='__main__': main()
