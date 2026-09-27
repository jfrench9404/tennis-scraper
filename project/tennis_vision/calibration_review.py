"""Build a non-destructive court calibration + exact decoded-frame bounce desk.

With --draft/--replay it instead builds a calibration-only desk for a carried
DRAFT calibration (game_pipeline's calibration-draft.json) in a new folder, so
John can check it for the new run and export a reviewed file bound to that run.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from .court import CourtMapper
from .court_refinement import LANDMARKS,fit_court,validate_review
from .replay3d import court_lines
from .shot_replay import build,save_json


def projected_curves(camera,old=None):
    """Fitted court lines in image pixels, plus the original four-corner projection when given."""
    curves=[];old_curves=[]
    for segment in court_lines():
        xyz=np.linspace(segment[0],segment[-1],60)
        curves.append(camera.project(xyz).tolist())
        if old is not None:
            ground=xyz[:,:2]+[5.485,11.885]
            old_curves.append(cv2.perspectiveTransform(ground.astype(np.float32).reshape(1,-1,2),np.linalg.inv(old.matrix))[0].tolist())
    return curves,old_curves


def write_workspace(output,court_path):
    output=Path(output)
    data=json.loads((output/'replay-data.json').read_text(encoding='utf-8'))
    corrections=data['report']['calibration_review']
    if not corrections:
        raise ValueError('This replay has no court/bounce corrections')
    width,height=corrections['image_size']
    court,camera,report=fit_court(corrections['landmarks'],width,height)
    old=CourtMapper(np.asarray(yaml.safe_load(Path(court_path).read_text())['image_corners'],np.float32))
    curves,old_curves=projected_curves(camera,old)
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
    render_page(output,packet)
    return packet


def render_page(output,packet):
    template=Path(__file__).with_name('calibration_review.html').read_text(encoding='utf-8')
    encoded=json.dumps(packet,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    (Path(output)/'calibration.html').write_text(template.replace('__CALIBRATION_DATA__',encoded),encoding='utf-8')


def _inside(path,folder):
    path,folder=Path(path).resolve(),Path(folder).resolve()
    return path==folder or folder in path.parents


def load_carried_draft(draft_path,replay_dir):
    """Validate a carried DRAFT calibration against the replay built for its run.

    Refuses a reviewed file (that decision is already made), a draft bound to
    another run or image size, and any draft that carries bounce edits.
    """
    draft=json.loads(Path(draft_path).read_text(encoding='utf-8'))
    report=json.loads((Path(replay_dir)/'replay-data.json').read_text(encoding='utf-8'))['report']
    if not isinstance(draft,dict) or draft.get('kind')!='court_and_bounce_review':
        raise ValueError('Not a court_and_bounce_review calibration file')
    if draft.get('calibration_status')!='draft':
        raise ValueError('Only a draft calibration opens in the draft desk; a reviewed file is never re-opened as a draft')
    if draft.get('bounce_edits')!=[]:
        raise ValueError('A carried draft must not contain bounce edits; bounce decisions never transfer between runs')
    if draft.get('run_id')!=report['review_run_id']:
        raise ValueError('Draft calibration is bound to a different run than this replay')
    validate_review(draft,report['review_run_id'],report['width'],report['height'],report['frames'],[])
    return draft,report


def write_draft_desk(draft_path,replay_dir,output,court_path=None,video=None,review_dir=None,allow_missing_video=False):
    """Build a calibration desk for reviewing a carried draft in a NEW folder.

    Nothing is marked reviewed here: the page starts as a draft and only John's
    explicit tick changes the exported status. The draft file, the replay and
    the run folders are only read.
    """
    draft_path,replay_dir,output=Path(draft_path),Path(replay_dir),Path(output)
    draft,report=load_carried_draft(draft_path,replay_dir)
    for folder in (replay_dir,draft_path.parent):
        if _inside(output,folder):
            raise ValueError(f'Output must be a new folder outside {folder}; run folders are never written')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} already has files; choose a new folder')
    width,height=draft['image_size']
    _,camera,fit=fit_court(draft['landmarks'],width,height)
    old=None
    if court_path is not None:
        old=CourtMapper(np.asarray(yaml.safe_load(Path(court_path).read_text())['image_corners'],np.float32))
    curves,old_curves=projected_curves(camera,old)
    start,frames=report['source_start_frame'],report['frames']
    check=draft.get('camera_check') or {}
    samples=[]
    for s in check.get('samples',[]):
        relative=s['frame']-start
        samples.append(dict(s,run_frame=relative if 0<=relative<frames else None,image=None))
    video=Path(video) if video is not None else replay_dir/'source.mp4'
    if not video.is_file() and not allow_missing_video:
        raise ValueError(f'{video} is missing; pass --video, or --no-frames to build a desk that cannot be marked reviewed')
    images=False
    output.mkdir(parents=True,exist_ok=True)
    if video.is_file():
        wanted={draft['calibration_frame']}|{s['run_frame'] for s in samples if s['run_frame'] is not None}
        (output/'frames').mkdir(exist_ok=True)
        cap=cv2.VideoCapture(str(video))
        try:
            for f in range(max(wanted)+1):
                ok,image=cap.read()
                if not ok:
                    raise ValueError(f'Source decode failed at frame {f}')
                if image.shape[1::-1]!=(width,height):
                    raise ValueError(f'Video is {image.shape[1]}x{image.shape[0]}, calibration is {width}x{height}')
                if f==draft['calibration_frame'] and not cv2.imwrite(str(output/'calibration.png'),image):
                    raise ValueError('Could not write calibration image')
                if f in wanted and not cv2.imwrite(str(output/'frames'/f'{f:06d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,94]):
                    raise ValueError('Could not write camera-check frame')
        finally:
            cap.release()
        for s in samples:
            if s['run_frame'] is not None:
                s['image']=f"frames/{s['run_frame']:06d}.jpg"
        images=True
    if review_dir is None and (draft_path.parent/'review'/'review-report.json').is_file():
        review_dir=draft_path.parent/'review'
    try:
        replay_href=Path(os.path.relpath(replay_dir.resolve()/'replay.html',output.resolve())).as_posix()
    except ValueError:
        replay_href=None
    python='.venv\\Scripts\\python.exe -m tennis_vision.calibration_review'
    new_folder=lambda stem:f'("runs\\{stem}-" + (Get-Date -Format \'yyyyMMdd-HHmmss\'))'
    downloads=lambda status:f'"$env:USERPROFILE\\Downloads\\tennis-calibration-{status}-{draft["run_id"][:16]}.json"'
    court_arg=f'"{Path(court_path).resolve()}"' if court_path else 'court.yaml'
    commands={'apply':f'{python} --run "{report.get("input_run") or "<run folder>"}" '
                      f'--review "{Path(review_dir).resolve() if review_dir else "<review folder>"}" --court {court_arg} '
                      f'--corrections {downloads("reviewed")} --output {new_folder("court-reviewed")}',
              'rebuild':f'{python} --draft {downloads("draft")} --replay "{replay_dir.resolve()}" --court {court_arg} '
                        f'--output {new_folder("court-draft-desk")}'}
    packet={'mode':'carried_draft','commands':commands,'draft':draft,'report':fit,'fps':report['fps'],
            'source_start_frame':start,'frames':frames,
            'landmark_definitions':[{'id':p[0],'name':p[1],'court_m':list(p[2:])} for p in LANDMARKS],
            'curves':curves,'old_curves':old_curves,'events':[],'ball_pixels':[],
            'images_available':images,'replay_href':replay_href,
            'provenance':{'draft_file':str(draft_path.resolve()),'replay':str(replay_dir.resolve()),
                          'run':report.get('input_run'),'review':str(Path(review_dir).resolve()) if review_dir else None,
                          'court':str(Path(court_path).resolve()) if court_path else None,
                          'carried_from':draft.get('carried_from'),'landmark_source':draft.get('landmark_source'),
                          'notes':draft.get('notes')},
            'camera_check':{'summary':check.get('summary'),'meaning':check.get('meaning'),'samples':samples}}
    save_json(output/'calibration-workspace.json',packet)
    render_page(output,packet)
    return packet


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','review','court','corrections','output','draft','replay','video'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--no-frames',action='store_true',help='Draft desk without video stills (cannot be marked reviewed)')
    args=p.parse_args(argv)
    if args.output is None:
        p.error('--output is required')
    if args.draft is not None:
        if args.replay is None:
            p.error('--draft needs --replay (the replay folder built for the same run)')
        write_draft_desk(args.draft,args.replay,args.output,args.court,args.video,args.review,args.no_frames)
        print(f"Draft calibration desk (nothing marked reviewed): {args.output.resolve()/'calibration.html'}")
        return
    missing=[n for n in ('run','review','court','corrections') if getattr(args,n) is None]
    if missing:
        p.error('missing '+', '.join('--'+n for n in missing))
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
