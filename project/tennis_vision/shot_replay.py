"""Offline shot classification + feet-anchored 3D replay from a completed review."""
import argparse
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import yaml

from .camera3d import refine_from_court
from .cached_identity import recover_initial_players
from .court import CourtMapper
from .event_analysis import VERSION, observed_balls
from .event_review import load_run
from .replay3d import court_lines, net_lines
from .shot_classification import classify_shots, infer_handedness, resolve_review_events
from .validated_flight import fit_reviewed_flights
from .preview_flight import fit_preview_flights


BONES = [("left_shoulder","right_shoulder"),("left_shoulder","left_elbow"),
         ("left_elbow","left_wrist"),("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
         ("left_shoulder","left_hip"),("right_shoulder","right_hip"),("left_hip","right_hip"),
         ("left_hip","left_knee"),("left_knee","left_ankle"),("right_hip","right_knee"),
         ("right_knee","right_ankle"),("nose","left_shoulder"),("nose","right_shoulder")]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_replay_page(output):
    """Refresh presentation only; preserve analysis and review source data."""
    output=Path(output)
    data=json.loads((output/"replay-data.json").read_text(encoding="utf-8"))
    here=Path(__file__).parent
    template=here.joinpath("shot_replay.html").read_text(encoding="utf-8")
    # Vendored three.js and the body-mesh script are inlined so the page stays a
    # single offline file (no CDN, works from file://). Substitute them before
    # the data so replay JSON can never be mistaken for a placeholder.
    template=(template.replace('__THREE_JS__',here.joinpath('vendor','three-0.159.0.min.js').read_text(encoding='utf-8'))
              .replace('__BODY_MESHES_JS__',here.joinpath('body_meshes.js').read_text(encoding='utf-8')))
    encoded=json.dumps(data,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    (output/"replay.html").write_text(template.replace('__REPLAY_DATA__',encoded,1),encoding="utf-8")


def feet_record(track, court):
    box = track.get("bbox", [])
    if len(box) != 4 or not np.isfinite(box).all() or box[3] <= box[1]:
        return None
    keypoints = track.get("keypoints") or {}
    ankles = [v[:2] for k,v in keypoints.items() if k in ("left_ankle","right_ankle")
              and len(v)>=3 and np.isfinite(v[:3]).all() and v[2]>=.3]
    pixel = np.mean(ankles,axis=0).tolist() if ankles else [(box[0]+box[2])/2,box[3]]
    ground = court.project(tuple(pixel))
    anchor_fallback=False
    if ankles and (not np.isfinite(ground).all() or not (-2.5<=ground[0]<=13.47 and -6.5<=ground[1]<=30.27)):
        # A raised/mislocalized ankle is not a grounded foot. Keep an observed
        # body visible using its box base only when that fallback is plausible.
        pixel=[(box[0]+box[2])/2,box[3]]
        ground=court.project(tuple(pixel));anchor_fallback=True
    # Include legitimate baseline/runoff positioning; never clip a player to a
    # court boundary, and never promote anonymous neighbouring-court people.
    if not np.isfinite(ground).all() or not (-2.5<=ground[0]<=13.47 and -6.5<=ground[1]<=30.27):
        return None
    return {"identity_id":track["identity_id"],"pixel":pixel,
            "raw_feet_xyz_m":[ground[0]-5.485,ground[1]-11.885,0.],
            "feet_xyz_m":[ground[0]-5.485,ground[1]-11.885,0.],
            "basis":("predicted_bbox_ground" if track.get("predicted") else
                     "bbox_ground_fallback_invalid_ankles" if anchor_fallback else
                     "two_ankles_ground" if len(ankles)==2 else "one_ankle_ground" if ankles else "bbox_ground_fallback"),
            "predicted":bool(track.get("predicted")),"position_smoothed":False,
            "bbox":list(box),"identity_basis":track.get("identity_basis","original_confirmed_track"),
            "identity_association_estimated":bool(track.get("identity_association_estimated")),
            "identity_seed_frame":track.get("identity_seed_frame"),
            "identity_evidence":track.get("identity_evidence"),
            'pose_source':track.get('pose_source','original_model'),
            "joints_m":{},"joints_2d":keypoints if not track.get("predicted") else {},
            "pose_basis":"camera_facing_2_5d_plane_not_measured_joint_depth"}


def pose_on_plane(record, camera, court):
    """Lift image joints onto a vertical camera-facing plane through the feet.

    This is a 2.5D display construction, not recovered anatomical joint depth.
    Foot anchors use court homography; both feet stay at z=0 by convention.
    """
    if camera is None or record["predicted"]:
        return {}
    base = np.asarray(record["raw_feet_xyz_m"])
    shift = np.asarray(record["feet_xyz_m"])-base
    origin = camera.position_m
    normal = origin-base
    normal[2] = 0
    normal /= max(np.linalg.norm(normal),1e-9)
    result = {}
    for name,value in record["joints_2d"].items():
        if len(value)<3 or not np.isfinite(value[:3]).all() or value[2]<.3:
            continue
        if name.endswith("ankle"):
            ground = court.project(tuple(value[:2]))
            point = np.array([ground[0]-5.485,ground[1]-11.885,0.])
        else:
            uv = camera.normalized_pixels([value[:2]])[0]
            ray = camera.R.T@np.array([uv[0],uv[1],1.])
            denominator = float(np.dot(normal,ray))
            if abs(denominator)<1e-8:
                continue
            distance = float(np.dot(normal,base-origin)/denominator)
            if distance<=0:
                continue
            point = origin+distance*ray
        if point[2]<-.15 or point[2]>3.6 or np.linalg.norm(point[:2]-base[:2])>1.8:
            continue
        point[2] = max(0.,point[2])
        result[name] = np.round(point+shift,4).tolist()
    return result


def build_players(rows, court, camera, cuts=()):
    frames = []
    omitted = Counter()
    for row in rows:
        players = []
        for identity in ("near","far"):
            matches = [t for t in row["tracks"] if t.get("label")=="player" and t.get("identity_id")==identity]
            if len(matches)!=1:
                omitted[identity+"_missing_or_ambiguous"] += 1
                continue
            record = feet_record(matches[0],court)
            if record is None:
                omitted[identity+"_invalid_foot_projection"] += 1
            else:
                players.append(record)
        frames.append({"frame":row["frame"],"players":players})
    # Symmetric median on three consecutive observations only; no prediction
    # through gaps or cuts and no persistent last-seen avatars.
    for f, frame in enumerate(frames):
        for p in frame["players"]:
            neighbours = [next((q for q in frames[k]["players"] if q["identity_id"]==p["identity_id"]
                                and not q["predicted"]),None) for k in range(max(0,f-1),min(len(frames),f+2))]
            if (len(neighbours)==3 and all(neighbours) and not any(f-1<c<=f+1 for c in cuts)):
                points = np.asarray([q["raw_feet_xyz_m"] for q in neighbours])
                if np.max(np.linalg.norm(points-points[1],axis=1))<1.:
                    p["feet_xyz_m"] = np.median(points,axis=0).tolist()
                    p["position_smoothed"] = True
            p["joints_m"] = pose_on_plane(p,camera,court)
            p["avatar"] = "pose_wireframe" if len(p["joints_m"])>=6 else "height_placeholder"
    return frames,dict(omitted)


def build(run, review, output, court_path, labels_path=None, near_hand="auto", far_hand="auto", corrections_path=None,
          pose_weights=None, pose_cache=None, contacts=False, contact_review_path=None,stroke_review_path=None):
    started = time.perf_counter()
    run,review,output,court_path = map(Path,(run,review,output,court_path))
    if output.exists():
        raise ValueError('Output already exists; choose a new folder to preserve previous results')
    summary,offset,rows = load_run(run)
    config = yaml.safe_load(court_path.read_text(encoding="utf-8"))
    corners = np.asarray(config["image_corners"],np.float32)
    if corners.shape!=(4,2) or not np.isfinite(corners).all() or abs(cv2.contourArea(corners))<100:
        raise ValueError("Invalid manual court calibration")
    court = CourtMapper(corners)
    camera,camera_fit = refine_from_court(court,summary["width"],summary["height"])
    review_report = json.loads((review/"review-report.json").read_text(encoding="utf-8"))
    candidate_file = json.loads((review/"event-candidates.json").read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256((run/"events.jsonl").read_bytes()+json.dumps(
        {"summary":summary,"start":offset,"court":court.image_corners.tolist(),"version":VERSION},sort_keys=True).encode()).hexdigest()
    if review_report.get("run_id")!=fingerprint or candidate_file.get("run_id")!=fingerprint:
        raise ValueError("Review does not match this run and court calibration; use matching inputs")
    if json.loads((review/"build-status.json").read_text())["status"]!="complete":
        raise ValueError("Input review build is incomplete")
    source_video = review/"source.mp4"
    if not source_video.is_file():
        raise ValueError("Review source.mp4 is missing; generate the full event-review package first")
    cap = cv2.VideoCapture(str(source_video))
    valid_source = (cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(rows)
                    and abs(cap.get(cv2.CAP_PROP_FPS)-summary["fps"])<.01
                    and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==summary["width"]
                    and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==summary["height"])
    cap.release()
    if not valid_source:
        raise ValueError("Review video is not aligned to cached frames")
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8")) if labels_path else None
    corrections = None
    if corrections_path:
        from .court_refinement import fit_court,validate_review,LOOKUP
        corrections=validate_review(json.loads(Path(corrections_path).read_text(encoding='utf-8')),fingerprint,
                                    summary['width'],summary['height'],len(rows),candidate_file['events'])
        previous_camera=camera
        court,camera,camera_fit=fit_court(corrections['landmarks'],summary['width'],summary['height'])
        camera_fit['review_status']=corrections['calibration_status']
        if previous_camera is not None:
            xyz=np.array([[LOOKUP[k][2]-5.485,LOOKUP[k][3]-11.885,0] for k in corrections['landmarks']])
            pixels=np.array(list(corrections['landmarks'].values()))
            camera_fit['previous_landmark_rmse_px']=float(np.sqrt(np.mean(np.sum((previous_camera.project(xyz)-pixels)**2,axis=1))))
        labels=copy.deepcopy(labels) if labels is not None else {'schema_version':1,'run_id':fingerprint,'labels':[],'manual_events':[]}
        if {e['id'] for e in labels['labels']} & {e['id'] for e in corrections['bounce_edits']}:
            raise ValueError('Overlapping event labels and bounce edits; consolidate decisions instead of overwriting them')
        for edit in corrections['bounce_edits']:
            labels['labels'].append(dict(edit,type='bounce',player_id=None))
    balls = observed_balls(rows)
    events = resolve_review_events(candidate_file["events"],labels,balls,summary["fps"],court,fingerprint,
                                   summary["width"],summary["height"])
    if corrections:
        edits={e['id']:e for e in corrections['bounce_edits']}
        for event in events:
            if event['id'] in edits:
                event['timing_review']={'frame_range':edits[event['id']]['frame_range'],
                    'frame_period_ms':1000/summary['fps'],'precision':'frame_level_not_exact_subframe_contact',
                    'status':edits[event['id']]['status']}
    cuts = review_report.get("scene_cut_frames",[])
    print("Checking cached detections before delayed player initialization...",flush=True)
    replay_rows,identity_recovery = recover_initial_players(rows,source_video,court,summary["fps"],cuts)
    refinement=None
    if pose_weights or pose_cache:
        from .player_refinement import refine_far_player
        model_path=Path(pose_weights) if pose_weights else None
        if model_path is None or not model_path.is_file():
            raise ValueError('Supply an existing local --pose-weights checkpoint; no implicit model download')
        binding={'run_id':fingerprint,'source_sha256':hashlib.sha256(source_video.read_bytes()).hexdigest(),
                 'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
                 'corrections':corrections,'method':'identity_anchored_crop_pose_v1'}
        cache_path=Path(pose_cache) if pose_cache else None
        if cache_path and cache_path.exists():
            cache=json.loads(cache_path.read_text(encoding='utf-8'))
            if cache.get('binding')!=binding or len(cache.get('rows',[]))!=len(rows):
                raise ValueError('Pose cache belongs to different inputs; choose a new cache path')
            replay_rows,refinement=cache['rows'],cache['report']
            print('Using matching focused-pose cache.',flush=True)
        else:
            # Isolate inference settings in this project, not the user's global
            # Ultralytics account/config. Local checkpoints require no network.
            config_dir=Path(__file__).resolve().parent.parent/'.inference-config'
            config_dir.mkdir(exist_ok=True)
            os.environ['YOLO_CONFIG_DIR']=str(config_dir)
            os.environ['YOLO_OFFLINE']='true'
            os.environ['YOLO_AUTOINSTALL']='false'
            from ultralytics import YOLO
            from ultralytics import settings
            settings.update({'sync':False})
            import torch
            torch.set_num_threads(min(4,torch.get_num_threads()))
            if model_path.suffix=='.onnx':
                # Exported copy of the same checkpoint (export_onnx.py), run on the
                # GPU through DirectML; the cache binding records the ONNX hash.
                from .longrun import enable_directml
                enable_directml([model_path])
                pose_model=YOLO(str(model_path),task='pose')
            else:
                pose_model=YOLO(str(model_path))
            replay_rows,refinement=refine_far_player(replay_rows,source_video,court,pose_model,cuts)
            if cache_path:
                cache_path.parent.mkdir(parents=True,exist_ok=True)
                save_json(cache_path,{'binding':binding,'rows':replay_rows,'report':refinement})
        refinement=dict(refinement,checkpoint_sha256=binding['model_sha256'],
            observation_sha256=hashlib.sha256(json.dumps([[p for p in r['tracks'] if p.get('identity_id')=='far']
                for r in replay_rows],sort_keys=True).encode()).hexdigest())
    # Back-associated tracks without fresh inference remain display-only. Fresh
    # crop observations may support candidates, with estimated identity explicit.
    classification_rows=copy.deepcopy(rows)
    if refinement:
        for f,row in enumerate(replay_rows):
            fresh=[p for p in row['tracks'] if p.get('pose_source')=='focused_crop_model']
            if fresh:
                classification_rows[f]['tracks']=[p for p in classification_rows[f]['tracks']
                    if not (p.get('label')=='player' and p.get('identity_id')=='far')]+copy.deepcopy(fresh)
    profiles = infer_handedness(classification_rows,{"near":near_hand,"far":far_hand})
    print("Classifying surrounding pose movement...",flush=True)
    if corrections:
        for row in classification_rows:
            for track in row['tracks']:
                if track.get('label')=='player' and track.get('identity_id') in ('near','far'):
                    foot=feet_record(track,court)
                    track['court_m']=court.project(tuple(foot['pixel'])) if foot else None
    contact_packet=None
    contact_binding=None
    if contacts:
        from .contact_detection import detect_contacts,merge_contacts,apply_contact_review
        contact_packet=detect_contacts(classification_rows,summary['fps'],cuts)
        events=merge_contacts(events,contact_packet,summary['fps'])
        contact_binding=hashlib.sha256(json.dumps({'run':fingerprint,'corrections':corrections,
            'contacts':contact_packet,'events':events},sort_keys=True).encode()).hexdigest()
        if contact_review_path:
            events=apply_contact_review(events,json.loads(Path(contact_review_path).read_text(encoding='utf-8')),
                                        contact_binding,balls,summary['fps'])
    elif contact_review_path:
        raise ValueError('--contact-review requires --contacts')
    play=None
    if contacts:
        from .play_context import assess_play
        events,play=assess_play(classification_rows,events,summary['fps'],cuts)
    shots = classify_shots(classification_rows,events,summary["fps"],profiles,cuts)
    print("Building feet anchors and 2.5D body wireframes...",flush=True)
    frames,omitted = build_players(replay_rows,court,camera,cuts)
    # Display-only equipment geometry; never feeds events, contacts or shots.
    from .racquet_replay import build_racquets
    racquets = build_racquets(replay_rows,frames,camera,summary["fps"],cuts)
    print("Fitting flights using confirmed bounce constraints only...",flush=True)
    flight_events=[e for e in events if e['type']!='hit' or e['status']=='confirmed' or
                   (e.get('contact_support')!='review_window_only' and
                    e.get('play_assessment',{}).get('action_state','shot_candidate')=='shot_candidate')]
    flights = fit_reviewed_flights(camera,flight_events,balls,summary["fps"],cuts)
    for fit_index,fit in enumerate(flights["fits"]):
        for point in fit["trajectory_m"]:
            frames[point["frame"]]["ball_3d"] = {"point_m":point["point_m"],"fit_index":fit_index,
                                                "status":"estimated_not_measured"}
    print("Building separate, unvalidated trajectory previews...",flush=True)
    previews = fit_preview_flights(camera,flight_events,balls,summary["fps"],cuts)
    for fit_index,fit in enumerate(previews["fits"]):
        for point in fit["trajectory_m"]:
            f = point["frame"]
            frames[f]["ball_3d_preview"] = {"point_m":point["point_m"],"fit_index":fit_index,
                                           "status":"unvalidated_preview",
                                           "image_observation_missing":balls[f] is None}
    for f,frame in enumerate(frames):
        frame['play_phase']=play['frames'][f] if play else {'phase':'unresolved','basis':'not_assessed'}
        frame["ball_observed_2d"] = balls[f]["pixel"] if balls[f] else None
        frame.setdefault("ball_3d",None)
        frame.setdefault("ball_3d_preview",None)
    package_id = hashlib.sha256(json.dumps({"run":fingerprint,"labels":labels,"profiles":profiles,
                                          "camera_fit":camera_fit,"corrections":corrections,'refinement':refinement,
                                          'contacts':contact_packet,'events':events,"version":"shot-replay-6"},sort_keys=True).encode()).hexdigest()
    stroke_review=None
    if stroke_review_path:
        from .stroke_execution import apply_stroke_review
        stroke_review=json.loads(Path(stroke_review_path).read_text(encoding='utf-8'))
        shots=apply_stroke_review(shots,stroke_review,package_id)
    # Stable analysis binding allows cumulative feedback exports/rebuilds without
    # relabeling model output as human reviewed. The artifact id includes edits.
    stroke_review_binding=package_id
    if stroke_review:
        package_id=hashlib.sha256((package_id+json.dumps(stroke_review,sort_keys=True)).encode()).hexdigest()
    report = {"schema_version":1,"package_id":package_id,"review_run_id":fingerprint,"frames":len(rows),
              "fps":summary["fps"],"width":summary["width"],"height":summary["height"],"source_start_frame":offset,
              "input_run":str(run.resolve()),"review_labels":str(Path(labels_path).resolve()) if labels_path else None,
              "classification_counts":shots["counts"],"handedness":profiles,
              "event_review_counts":dict(Counter(e["status"] for e in events)),
              "player_frames":dict(Counter(p["identity_id"] for frame in frames for p in frame["players"])),
              "player_projection_omissions":omitted,"confirmed_bounces":flights["confirmed_bounces"],
              "visible_bounce_candidates":sum(e["type"]=="bounce" and e["status"] in ("unreviewed","uncertain") for e in events),
              "player_identity_recovery":identity_recovery,
              'far_player_refinement':refinement,'contact_detection':contact_packet,
              'contact_review_binding':contact_binding,
              'stroke_review_binding':stroke_review_binding,'stroke_review':stroke_review,
              'play_context':{k:v for k,v in play.items() if k!='frames'} if play else None,
              "avatar_counts":dict(Counter(p["avatar"] for frame in frames for p in frame["players"])),
              "racquet_display":racquets,
              "observed_ball_frames":sum(b is not None for b in balls),
              "missing_ball_frames":sum(b is None for b in balls),
              "supported_flight_fits":len(flights["fits"]),"camera_available":camera is not None,
              "unvalidated_preview_segments":len(previews["fits"]),
              "unvalidated_preview_frames":sum(f["ball_3d_preview"] is not None for f in frames),
              "camera_focal_length_basis":camera_fit["basis"],"camera_fit":camera_fit,
              "calibration_review":corrections,
              "limitations":[shots["limitation"],flights["limitation"],previews["limitation"],
                  "Player roots are estimated ground-plane foot positions, not measured jumping height. Missing ankles fall back to box bottom and are labelled.",
                  "Wireframes are 2.5D camera-facing pose constructions. Individual joint depth and body orientation are not recovered.",
                  "Delayed player initialization may be back-associated to cached person detections using adjacent motion, clothing and court-half gates. These identity assignments are estimates; the detector and original log are unchanged.",
                  "Dashed bounce-candidate rings are conditional ground locations IF the event is a bounce, not confirmed landings or reviewed-flight constraints. Only the separate preview channel may use them as hypotheses.",
                  "Camera calibration is approximate; consult camera_fit for its landmark, lens and review assumptions. Fitted residuals are not independent 3D accuracy measurements.",
                  "Reviewed-only mode hides unvalidated previews. Neither mode extends a ball path through unsupported intervals; unreviewed hypotheses never become confirmed observations.",
                  "An absent bounce detection is not sufficient evidence of a volley; uncertain classifications stay unknown."]}
    if refinement:
        report['limitations'].append(refinement['limitation'])
    if contact_packet:
        report['limitations'].append(contact_packet['limitation'])
    if play:
        report['limitations'].append(play['limitation'])
    data = {"report":report,"frames":frames,"events":events,"shots":shots,"flights":flights,"preview_flights":previews,
            "court_lines":court_lines(),"net_lines":net_lines(),"bones":BONES}
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/"build-status.json",{"status":"building"})
    try:
        shutil.copy2(source_video,output/"source.mp4")
        save_json(output/"shot-candidates.json",shots)
        save_json(output/"reviewed-events.json",{"run_id":fingerprint,"events":events})
        save_json(output/"validated-flight3d.json",flights)
        save_json(output/"preview-flight3d.json",previews)
        if corrections:
            save_json(output/'calibration-corrections.json',corrections)
        if refinement:
            save_json(output/'far-player-report.json',refinement)
        if contact_packet:
            save_json(output/'contact-candidates.json',contact_packet)
        if play:
            save_json(output/'play-context.json',play)
        if stroke_review:
            save_json(output/'stroke-review.json',stroke_review)
        report["seconds_processing"] = round(time.perf_counter()-started,2)
        save_json(output/"replay-data.json",data)
        save_json(output/"replay-report.json",report)
        write_replay_page(output)
        save_json(output/"build-status.json",{"status":"complete"})
    except Exception as error:
        save_json(output/"build-status.json",{"status":"failed","error":str(error)})
        raise
    print(json.dumps(report,indent=2))
    print(f"Open {output.resolve()/'replay.html'}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--court',type=Path,required=True)
    parser.add_argument('--labels',type=Path,help='Exported event-review labels for reviewed flights; unvalidated previews remain separate')
    parser.add_argument('--near-hand',choices=('auto','left','right','unknown'),default='auto',help='Legacy descriptive tendency only; does not constrain any stroke')
    parser.add_argument('--far-hand',choices=('auto','left','right','unknown'),default='auto',help='Legacy descriptive tendency only; does not constrain any stroke')
    parser.add_argument('--corrections',type=Path,help='Exported court/bounce review; binds to the original review run without overwriting it')
    parser.add_argument('--pose-weights',type=Path,help='Existing local checkpoint for identity-anchored far-player crop inference')
    parser.add_argument('--pose-cache',type=Path,help='Reusable crop observations, strictly bound to inputs')
    parser.add_argument('--contacts',action='store_true',help='Re-evaluate temporal ball/equipment contact candidates')
    parser.add_argument('--contact-review',type=Path,help='Explicit contact decisions exported from the matching upgraded replay')
    parser.add_argument('--stroke-review',type=Path,help='Per-stroke type and optional striking-hand feedback; never confirms contact')
    args = parser.parse_args()
    build(args.run,args.review,args.output,args.court,args.labels,args.near_hand,args.far_hand,args.corrections,
          args.pose_weights,args.pose_cache,args.contacts,args.contact_review,args.stroke_review)


if __name__=='__main__':
    main()
