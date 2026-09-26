"""Build a non-destructive court calibration + exact decoded-frame bounce desk."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .court import CourtMapper
from .court_refinement import LANDMARKS,fit_court
from .replay3d import court_lines
from .shot_replay import build,save_json


def write_workspace(output,court_path):
    output=Path(output)
    data=json.loads((output/'replay-data.json').read_text(encoding='utf-8'))
    corrections=data['report']['calibration_review']
    if not corrections:
        raise ValueError('This replay has no court/bounce corrections')
    width,height=corrections['image_size']
    court,camera,report=fit_court(corrections['landmarks'],width,height)
    old=CourtMapper(np.asarray(yaml.safe_load(Path(court_path).read_text())['image_corners'],np.float32))
    curves=[];old_curves=[]
    for segment in court_lines():
        xyz=np.linspace(segment[0],segment[-1],60)
        curves.append(camera.project(xyz).tolist())
        ground=xyz[:,:2]+[5.485,11.885]
        old_curves.append(cv2.perspectiveTransform(ground.astype(np.float32).reshape(1,-1,2),np.linalg.inv(old.matrix))[0].tolist())
    events=[];required={corrections['calibration_frame']}
    edits={e['id']:e for e in corrections['bounce_edits']}
    for e in data['events']:
        if e['type']!='bounce':
            continue
        centre=edits.get(e['id'],e)['frame']
        lo,hi=max(0,centre-6),min(len(data['frames'])-1,centre+6)
        frames=list(range(lo,hi+1));required.update(frames)
        pixel=e.get('pixel')
        original_pixel=data['frames'][e['original_frame']]['ball_observed_2d']
        events.append({'id':e['id'],'frame':centre,'original_frame':e['original_frame'],
                       'frames':frames,'pixel':pixel,
                       'old_court_m':old.project(tuple(pixel)) if pixel is not None else None,
                       'old_original_court_m':old.project(tuple(original_pixel)) if original_pixel is not None else None,
                       'new_court_m':court.project(tuple(pixel)) if pixel is not None else None})
    directory=output/'timing-frames';directory.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(output/'source.mp4'))
    try:
        for f in range(max(required)+1):
            ok,image=cap.read()
            if not ok:
                raise ValueError(f'Source decode failed at frame {f}')
            if f==corrections['calibration_frame']:
                if not cv2.imwrite(str(output/'calibration.png'),image):
                    raise ValueError('Could not write calibration image')
                overlay=image.copy()
                for points in curves:
                    cv2.polylines(overlay,[np.round(points).astype(np.int32)],False,(90,240,110),1)
                cv2.imwrite(str(output/'calibration-overlay.png'),overlay)
            if f in required:
                if not cv2.imwrite(str(directory/f'{f:06d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,94]):
                    raise ValueError('Could not write exact-frame evidence image')
    finally:
        cap.release()
    packet={'draft':corrections,'report':report,'fps':data['report']['fps'],
            'source_start_frame':data['report']['source_start_frame'],'frames':len(data['frames']),
            'landmark_definitions':[{'id':p[0],'name':p[1],'court_m':list(p[2:])} for p in LANDMARKS],
            'curves':curves,'old_curves':old_curves,'events':events,
            'ball_pixels':[f['ball_observed_2d'] for f in data['frames']]}
    save_json(output/'calibration-workspace.json',packet)
    template=Path(__file__).with_name('calibration_review.html').read_text(encoding='utf-8')
    encoded=json.dumps(packet,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    (output/'calibration.html').write_text(template.replace('__CALIBRATION_DATA__',encoded),encoding='utf-8')
    return packet


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','review','court','corrections','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    build(args.run,args.review,args.output,args.court,corrections_path=args.corrections)
    save_json(args.output/'calibration-build-status.json',{'status':'building'})
    try:
        write_workspace(args.output,args.court)
        save_json(args.output/'calibration-build-status.json',{'status':'complete'})
    except Exception as error:
        save_json(args.output/'calibration-build-status.json',{'status':'failed','error':str(error)})
        raise
    print(f"Court + bounce desk: {args.output.resolve()/'calibration.html'}")


if __name__=='__main__':main()
