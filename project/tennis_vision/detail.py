"""Additional crop inference for distant players and small balls.

All returned objects come from fresh inference; crops never fabricate positions.
Coordinates are translated back to the original frame before normal filtering.
"""
import math
import cv2
import numpy as np
from .tracking import Detection
from .filters import overlap


def translate(detection, x, y):
    a, b, c, d = detection.bbox
    joints = None if detection.keypoints is None else {
        name: [p[0]+x, p[1]+y, p[2]] for name, p in detection.keypoints.items()
    }
    return Detection(detection.label, (a+x, b+y, c+x, d+y), detection.confidence, joints)


def merge(detections):
    kept = []
    for item in sorted(detections, key=lambda d: (bool(d.keypoints), d.confidence), reverse=True):
        duplicate = False
        for other in kept:
            if item.label != other.label:
                continue
            iou, contained = overlap(item.bbox, other.bbox)
            if iou >= .35 or contained >= .75:
                duplicate = True
                break
            if item.label == 'ball':
                size = max(2, min(item.bbox[2]-item.bbox[0], other.bbox[2]-other.bbox[0]))
                if math.dist(item.center, other.center) <= size:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(item)
    return kept


def far_crop(shape, court, side_margin, baseline_margin):
    if court is None or court.source != 'manual':
        return None
    h, w = shape[:2]
    try:
        inverse = np.linalg.inv(court.matrix)
        world = np.asarray([[[-side_margin,11.885], [10.97+side_margin,11.885],
                             [10.97+side_margin,23.77+baseline_margin],
                             [-side_margin,23.77+baseline_margin]]], dtype=np.float32)
        pixels = cv2.perspectiveTransform(world, inverse)[0]
        if not np.isfinite(pixels).all():
            return None
        x1 = max(0, int(pixels[:,0].min())-12)
        x2 = min(w, int(pixels[:,0].max())+12)
        # Include bodies above foot-plane bounds and overhead swings.
        y1 = max(0, int(pixels[:,1].min())-int(.15*h))
        y2 = min(h, int(pixels[:,1].max())+12)
        return (x1,y1,x2,y2) if x2-x1 >= 32 and y2-y1 >= 32 else None
    except np.linalg.LinAlgError:
        return None


class DetailPass:
    def __init__(self):
        self.last_ball = None
        self.missed = 0

    def feedback(self, detections):
        """Only accepted observations steer crops; hold search anchor for 3 misses.

        This keeps searching, but never emits an invented ball detection.
        """
        balls = [d for d in detections if d.label == 'ball']
        if len(balls) == 1:
            self.last_ball, self.missed = balls[0].center, 0
        else:
            self.missed += 1
            if self.missed >= 3:
                self.last_ball = None

    def update(self, frame, detections, court, detect_objects, detect_people,
               side_margin=1.5, baseline_margin=6.0):
        regions = []
        far = far_crop(frame.shape, court, side_margin, baseline_margin)
        if far:
            x1,y1,x2,y2=far
            # Overlapping halves enlarge distant bodies further than one wide crop.
            regions.extend([((x1,y1,int(x1+.6*(x2-x1)),y2),True),
                            ((int(x1+.4*(x2-x1)),y1,x2,y2),True)])
        if self.last_ball is not None:
            h, w = frame.shape[:2]
            x, y = self.last_ball
            radius = max(96, int(.14*w))
            box = (max(0,int(x)-radius), max(0,int(y)-radius),
                   min(w,int(x)+radius), min(h,int(y)+radius))
            if box[2]-box[0] >= 32 and box[3]-box[1] >= 32:
                regions.append((box, False))
        found = list(detections)
        for (x1,y1,x2,y2), is_far in regions:
            crop = frame[y1:y2,x1:x2]
            objects = detect_objects(crop)
            found.extend(translate(d,x1,y1) for d in objects if d.label == 'ball')
            if is_far:
                found.extend(translate(d,x1,y1) for d in objects if d.label in ('player','racket'))
                found.extend(translate(d,x1,y1) for d in detect_people(crop) if d.label == 'player')
        found = merge(found)
        balls = [d for d in found if d.label == 'ball']
        return found, {'crop_passes':len(regions), 'far_crop':far,
                       'players_before_filters':sum(d.label=='player' for d in found),
                       'balls_before_filters':len(balls)}
