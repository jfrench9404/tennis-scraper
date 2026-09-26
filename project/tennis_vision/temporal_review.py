"""Run NEW temporal ball inference while reusing saved player/racquet candidates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
from tqdm import tqdm

from .filters import filter_tracks
from .scene import SceneSelector
from .setup_temporal import DEFAULT_MODEL, file_hash
from .temporal_ball import GridTrackNetDetector, frame_stream, select_temporal_ball
from .tracking import Detection, NearestTracker


def review(run,output,scene_path,model=DEFAULT_MODEL,threshold=.5):
    run,output,scene_path=map(Path,(run,output,scene_path))
    summary=json.loads((run/'summary.json').read_text())
    offset=json.loads((run/'source-offset.json').read_text())
    config=json.loads(scene_path.read_text())
    if not all(config.get(k) for k in ('near_seed_polygon','far_seed_polygon','ball_polygon')):
        raise ValueError('Review requires explicit scene polygons')
    detector=GridTrackNetDetector(model,threshold)
    output.mkdir(parents=True,exist_ok=False)
    cap=cv2.VideoCapture(summary['input'])
    if not cap.isOpened(): raise RuntimeError('Cannot open source video')
    if offset['source_start_frame'] and not cap.set(cv2.CAP_PROP_POS_FRAMES,offset['source_start_frame']):
        cap.release(); raise RuntimeError('Cannot seek source video')
    writer=cv2.VideoWriter(str(output/'review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),summary['fps'],(summary['width'],summary['height']))
    if not writer.isOpened():
        cap.release(); raise RuntimeError('Cannot create review video')
    selector=SceneSelector(config,summary['fps']); tracker=NearestTracker()
    counts=Counter(); original=Counter(); frames=0
    started=time.perf_counter()
    try:
        with (run/'events.jsonl').open() as stream, (output/'review-tracks.jsonl').open('w') as out:
            with tqdm(total=summary['frames'],desc='Temporal ball model',unit='frame') as progress:
                for image,ball,model_stats in frame_stream(cap,detector,summary['frames']):
                    line=next(stream,None)
                    if line is None: raise ValueError('Candidate log ended early')
                    row=json.loads(line)
                    if row['frame']!=frames: raise ValueError('Cache frame alignment mismatch')
                    raw=row['detail'].get('candidates')
                    if raw is None: raise ValueError('Run lacks raw candidates; use rally-confirmed or newer')
                    detections=[Detection(d['label'],tuple(d['bbox']),d['confidence'],d.get('keypoints')) for d in raw if d['label']!='ball']
                    kept,scene_stats=selector.update(image,detections,None)
                    balls,status=select_temporal_ball(ball,selector,image,None,[d for d in kept if d.label=='player'],model_stats)
                    kept+=balls
                    model_stats['status']=status
                    model_stats['selected_pixel']=list(balls[0].center) if balls else None
                    model_stats['selected_score']=balls[0].confidence if balls else None
                    counts.update({d.identity_id or d.label for d in kept})
                    original.update({d.get('identity_id') or d['label'] for d in row['tracks'] if not d['predicted']})
                    counts['raw_temporal_ball']+=int(ball is not None)
                    counts['area_rejected']+=int(status=='outside_area')
                    tracks,_=filter_tracks(tracker.update(kept,frames),None)
                    records=[track.as_dict(None) for track in tracks]
                    out.write(json.dumps({'frame':frames,'time_s':frames/summary['fps'],
                                          'tracks':records,'temporal_ball':model_stats,'scene':scene_stats})+'\n')
                    for track in tracks:
                        x1,y1,x2,y2=map(int,track.bbox)
                        colour=(0,255,255) if track.label=='ball' else ((255,0,255) if track.label=='racket' else (255,180,0))
                        if track.label=='ball':
                            x,y=map(int,track.center)
                            cv2.circle(image,(x,y),9,colour,2)
                            label=f'ball temporal {track.confidence:.2f}'
                        else:
                            cv2.rectangle(image,(x1,y1),(x2,y2),colour,1 if track.predicted else 2)
                            label=(track.identity_id or track.label)+(' (predicted)' if track.predicted else '')
                        cv2.putText(image,label,(x1,max(18,y1-8)),0,.45,colour,1)
                    cv2.putText(image,f'NEW five-frame tennis inference | frame {frames} | no interpolation',
                                (15,summary['height']-15),0,.5,(255,255,255),1)
                    writer.write(image)
                    frames+=1; progress.update(1)
        if frames!=summary['frames']: raise RuntimeError('Video ended before the requested review finished')
    finally:
        cap.release(); writer.release()
    report={'frames':frames,'fps':summary['fps'],'seconds_processing':round(time.perf_counter()-started,2),
            'original_observed_frames':dict(original),'new_observed_frames':dict(counts),
            'model':'GridTrackNet five-frame pretrained tennis','threshold':threshold,
            'model_sha256':file_hash(model),'model_path':str(Path(model).resolve()),'source_run':str(run.resolve()),
            'source_start_frame':offset['source_start_frame'],
            'limitations':'Presence counts, NOT accuracy. New ball inference; cached players/racquets (older cache lacks rejected poses). No interpolation, fine-tuning, new shot analysis, or calibrated 3D coordinates.'}
    (output/'review-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--threshold',type=float,default=.5)
    args=parser.parse_args()
    review(args.run,args.output,args.scene,args.model,args.threshold)


if __name__=='__main__': main()
