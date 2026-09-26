"""Conservative offline identity backfill, using existing learned detections only.

This repairs delayed initialization in the replay, not the detector. Never invent
boxes/joints, bridge a missing candidate, or alter the original event evidence.
"""
import copy
from types import SimpleNamespace

import cv2
import numpy as np

from .scene import appearance


def box_iou(a, b):
    intersection = max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/max(1.,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def valid_person(track):
    box = track.get("bbox",[])
    return (track.get("label")=="player" and not track.get("predicted")
            and track.get("source")=="model" and len(box)==4 and np.isfinite(box).all()
            and box[2]>box[0] and box[3]>box[1] and track.get("confidence",0)>=.1)


def choose_previous(candidates, previous, anchor_hist, recent_hist, court, identity):
    """Return a uniquely supported adjacent-frame match, or abstain."""
    box = previous["bbox"]
    centre = (np.asarray(box[:2])+box[2:])/2
    h = box[3]-box[1]
    choices = []
    for candidate,hist in candidates:
        if hist is None or not valid_person(candidate):
            continue
        b = candidate["bbox"]
        xy = [(b[0]+b[2])/2,b[3]]
        ground = court.project(tuple(xy))
        # Initialization only: remain on this player's half and nearby runoff.
        if not np.isfinite(ground).all() or not -1.5<=ground[0]<=12.47:
            continue
        if not ((11.885<=ground[1]<=30.27) if identity=="far" else (-6.5<=ground[1]<11.885)):
            continue
        scale = (b[3]-b[1])/h
        distance = float(np.linalg.norm((np.asarray(b[:2])+b[2:])/2-centre)/h)
        overlap = box_iou(box,b)
        colour_seed = float(cv2.compareHist(anchor_hist,hist,cv2.HISTCMP_BHATTACHARYYA))
        colour_recent = float(cv2.compareHist(recent_hist,hist,cv2.HISTCMP_BHATTACHARYYA))
        if not (.65<=scale<=1.5 and distance<=.5 and overlap>=.18
                and colour_seed<=.6 and colour_recent<=.4):
            continue
        score = distance+.3*abs(np.log(scale))+.3*colour_seed+.3*colour_recent
        evidence = {"adjacent_displacement_body_heights":round(distance,4),"box_iou":round(overlap,4),
                    "seed_colour_distance":round(colour_seed,4),"adjacent_colour_distance":round(colour_recent,4)}
        choices.append((score,candidate,hist,evidence))
    choices.sort(key=lambda item:item[0])
    # Duplicate crop detections of the same box aren't a competing identity.
    unique = []
    for choice in choices:
        if not any(box_iou(choice[1]["bbox"],other[1]["bbox"])>=.65 for other in unique):
            unique.append(choice)
    if not unique:
        return None,"no spatial/appearance-supported cached detection"
    if len(unique)>1 and unique[1][0]-unique[0][0]<.15:
        return None,"ambiguous competing person detections"
    return unique[0][1:],None


def recover_initial_players(rows, source_video, court, fps, cuts=()):
    """Back-associate at most three seconds before each first stable identity.

    Only the returned replay rows change. Shot inference and ball-flight fitting
    must continue to use the unmodified, authoritative tracking rows.
    """
    output = [dict(row,tracks=list(row.get("tracks",[]))) for row in rows]
    report = {"basis":"backward_cached_association_from_confirmed_track",
              "scope":"replay_only_not_shot_or_ball_evidence","recovered_frames":{},"identities":{}}
    seeds = {}
    for identity in ("near","far"):
        first = next((f for f,row in enumerate(rows) if any(
            p.get("identity_id")==identity and valid_person(p) for p in row.get("tracks",[]))),None)
        reason = None
        if first is None:
            reason = "no confirmed observed identity seed"
        elif first==0:
            reason = "identity already present at clip start"
        else:
            stable = []
            for row in rows[first:first+5]:
                matches = [p for p in row.get("tracks",[]) if p.get("identity_id")==identity and valid_person(p)]
                if len(matches)!=1:
                    break
                stable.append(matches[0])
            if len(stable)!=5 or len({p.get("track_id") for p in stable})!=1:
                reason = "first identity seed is not stable over five frames"
            elif any(first<c<=first+4 for c in cuts):
                reason = "camera cut within identity seed"
            else:
                seeds[identity] = (first,stable)
        report["identities"][identity] = {"seed_frame":first,"recovered":0,"stop_reason":reason}
        report["recovered_frames"][identity] = 0
    if not seeds:
        return output,report
    start = max(0,min(f for f,_ in seeds.values())-round(3*fps))
    end = max(f+4 for f,_ in seeds.values())
    # Sequential decode, retaining only small colour descriptors, not full images.
    cap = cv2.VideoCapture(str(source_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    descriptors,seed_hists = {},{identity:[] for identity in seeds}
    try:
        for f in range(start,end+1):
            ok,image = cap.read()
            if not ok:
                for identity in seeds:
                    report["identities"][identity]["stop_reason"] = "source decode failed; backfill skipped"
                return output,report
            items = rows[f].get("detail",{}).get("candidates",[])
            descriptors[f] = [(p,appearance(image,SimpleNamespace(bbox=p["bbox"]))) for p in items if valid_person(p)]
            for identity,(seed,stable) in seeds.items():
                if seed<=f<=seed+4:
                    seed_hists[identity].append(appearance(image,SimpleNamespace(bbox=stable[f-seed]["bbox"])))
    finally:
        cap.release()
    for identity,(seed,stable) in seeds.items():
        state = report["identities"][identity]
        histograms = seed_hists[identity]
        if any(h is None for h in histograms):
            state["stop_reason"] = "seed appearance unavailable"
            continue
        anchor = np.mean(histograms,axis=0).astype(np.float32)
        recent,previous = histograms[0],stable[0]
        earliest = max(0,seed-round(3*fps))
        for f in range(seed-1,earliest-1,-1):
            if f+1 in cuts:
                state["stop_reason"] = "camera cut"
                break
            if any(p.get("identity_id")==identity for p in rows[f].get("tracks",[])):
                state["stop_reason"] = "existing identity record left unchanged"
                break
            match,reason = choose_previous(descriptors[f],previous,anchor,recent,court,identity)
            if match is None:
                state["stop_reason"] = reason
                break
            candidate,recent,evidence = match
            restored = copy.deepcopy(candidate)
            restored.update(identity_id=identity,track_id=stable[0].get("track_id"),predicted=False,
                            identity_basis=report["basis"],identity_seed_frame=seed,
                            identity_association_estimated=True,identity_evidence=evidence)
            output[f]["tracks"].append(restored)
            previous = restored
            state["recovered"] += 1
        else:
            state["stop_reason"] = "clip start" if earliest==0 else "three-second lookback limit"
        report["recovered_frames"][identity] = state["recovered"]
    return output,report
