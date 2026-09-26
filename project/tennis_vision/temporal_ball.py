"""Five-frame tennis model inference. Lazy runtime imports keep it optional."""
import math
from pathlib import Path

import cv2
import numpy as np

from .tracking import Detection

SEQUENCE_LENGTH=5


def pack_frames(frames):
    if len(frames)!=SEQUENCE_LENGTH:
        raise ValueError('GridTrackNet requires exactly five consecutive frames')
    shape=frames[0].shape
    if any(frame.shape!=shape or frame.ndim!=3 or frame.shape[2]!=3 for frame in frames):
        raise ValueError('Frames must have the same HxWx3 shape')
    # Match upstream training (Keras load_img is RGB), not its video helper's
    # accidental double colour swap. Preserve chronological order.
    return np.concatenate([cv2.cvtColor(cv2.resize(frame,(768,432)),cv2.COLOR_BGR2RGB).transpose(2,0,1)
                           for frame in frames],axis=0)[None].astype(np.float32)/255.


def decode_grid(grid,shape,threshold=.5):
    if not math.isfinite(threshold) or not 0<threshold<=1:
        raise ValueError('Ball threshold must be in (0,1]')
    grid=np.asarray(grid)
    if grid.shape!=(1,15,27,48) or not np.isfinite(grid).all():
        raise ValueError('Invalid GridTrackNet output; expected finite [1,15,27,48]')
    height,width=shape[:2]
    results=[]
    for index in range(5):
        confidence,x_offset,y_offset=grid[0,index*3:index*3+3]
        candidates=[]
        # Preserve alternative cells: neighbouring courts may contain stronger
        # balls. Court membership must be applied BEFORE picking one peak.
        for flat_index in np.argsort(confidence,axis=None)[::-1]:
            row,col=np.unravel_index(flat_index,confidence.shape)
            score=float(confidence[row,col])
            if score<threshold: break
            x=float((col+x_offset[row,col])*width/48)
            y=float((row+y_offset[row,col])*height/27)
            if not (0<=x<width and 0<=y<height): continue
            if any(math.dist((x,y),item['pixel'])<width/160 for item in candidates): continue
            candidates.append({'pixel':[x,y],'score':score})
            if len(candidates)>=16: break
        detection=point_detection(candidates[0],shape) if candidates else None
        results.append((detection,{'peak_score':float(confidence.max()),'pixel':list(detection.center) if detection else None,
                                  'candidates':candidates}))
    return results


def point_detection(candidate,shape):
    height,width=shape[:2]
    x,y=candidate['pixel']
    # Point marker, not a measured ball diameter.
    radius=min(max(2.,width/320),x,width-x,y,height-y)
    return Detection('ball',(x-radius,y-radius,x+radius,y+radius),candidate['score'],source='gridtracknet')


def candidate_detections(detection,stats,shape):
    if 'candidates' in stats:
        return [point_detection(candidate,shape) for candidate in stats['candidates']]
    return [detection] if detection is not None else []


class GridTrackNetDetector:
    def __init__(self,model,threshold=.5,threads=4):
        import onnxruntime as ort
        model=Path(model)
        if not model.is_file():
            raise FileNotFoundError(f'{model} not found. Run python -m tennis_vision.setup_temporal first.')
        if not math.isfinite(threshold) or not 0<threshold<=1:
            raise ValueError('Ball threshold must be in (0,1]')
        options=ort.SessionOptions()
        options.intra_op_num_threads=threads
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        if self.session.get_inputs()[0].shape!=[1,15,432,768]:
            raise ValueError('Not a compatible five-frame GridTrackNet model')
        self.threshold=threshold

    def predict(self,frames):
        grid=self.session.run(['grid'],{'frames':pack_frames(frames)})[0]
        return decode_grid(grid,frames[0].shape,self.threshold)


def frame_stream(cap,detector=None,max_frames=0):
    """Buffer five frames (<=4 frame lookahead); emit every real frame exactly once.

    Partial final blocks repeat the last image for context, but never emit padded
    frames. This is offline / bounded-latency inference, not zero-latency tracking.
    """
    read=0
    while True:
        block=[]
        for _ in range(5 if detector else 1):
            if max_frames and read>=max_frames: break
            ok,frame=cap.read()
            if not ok: break
            block.append(frame); read+=1
        if not block: return
        if detector:
            padded=block+[block[-1]]*(5-len(block))
            results=detector.predict(padded)
        else:
            results=[(None,{})]
        for frame,(detection,stats) in zip(block,results):
            yield frame,detection,stats


def select_temporal_ball(detection,selector,frame,court,players,stats=None):
    # Do not reapply YOLO's yellow-pixel gate to a learned temporal localization.
    # The scene and known background exclusions still apply.
    candidates=candidate_detections(detection,stats or {},frame.shape)
    if not candidates: return [],'below_threshold'
    valid=[d for d in candidates if selector is None or selector.ball_allowed(d.center,frame.shape,court,players)]
    if not valid: return [],'outside_area'
    return [max(valid,key=lambda d:d.confidence)],'accepted'
