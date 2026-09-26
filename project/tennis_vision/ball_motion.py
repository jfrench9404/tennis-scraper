"""Experimental fixed-camera ball recovery from actual moving image components.

This is not a trained tennis detector. Positions require fresh pixel evidence;
no predicted positions are emitted. Motion observations are separately labelled
and must not be used as bounce/contact evidence without validation.
"""
import math

import cv2
import numpy as np

from .filters import point_box_distance
from .tracking import Detection


class MotionBallRecovery:
    def __init__(self):
        self.previous = None
        self.anchor = None
        self.velocity = (0., 0.)
        self.frame = 0
        self.seen = -999
        self.model_seen = -999
        self.paths = []

    def components(self, frame, detections, allowed):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous, self.previous = self.previous, gray
        if previous is None or previous.shape != gray.shape:
            return [], False
        difference = cv2.absdiff(gray, previous)
        # Cuts/pans invalidate the fixed-camera assumption and temporal anchor.
        if np.mean(difference > 20) > .20:
            return [], True
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (20, 20, 90), (65, 255, 255))
        bright = cv2.subtract(gray, cv2.GaussianBlur(gray, (9, 9), 0)) > 12
        pale = (hsv[:,:,1] < 90) & (hsv[:,:,2] > 150) & bright
        # Positive change only: absdiff also contains the previous ball's wake.
        mask = (((yellow > 0) | pale) & (cv2.subtract(gray,previous) > 12)).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        scale = frame.shape[1]/1280
        candidates = []
        for x,y,w,h,area in stats[1:]:
            if not (2<=area<=max(16,200*scale*scale) and max(w,h)<=max(10,26*scale)):
                continue
            if max(w,h)>5*max(1,min(w,h)):
                continue
            center=(float(x+w/2),float(y+h/2))
            if not allowed(center):
                continue
            # Clothing and swinging racquets are moving, too. Do not manufacture
            # a ball on them when the model itself saw no ball.
            if any(d.label in ('player','racket') and point_box_distance(center,d.bbox)<=max(3*scale,.08*(d.bbox[3]-d.bbox[1])) for d in detections):
                continue
            candidates.append((center,(float(x),float(y),float(x+w),float(y+h))))
        return candidates, False

    def update(self, frame, detections, allowed, blockers=None):
        self.frame += 1
        candidates, cut = self.components(frame,blockers if blockers is not None else detections,allowed)
        stats={'components':len(candidates),'recovered':0,'camera_change':cut}
        if cut:
            self.anchor=None; self.paths=[]; self.velocity=(0.,0.)
            self.seen=self.model_seen=-999
            return detections, stats
        balls=[d for d in detections if d.label=='ball']
        if len(balls)==1:
            self._observe(balls[0].center)
            self.model_seen=self.frame
            self.paths=[]
            return detections, stats
        if len(balls)>1:
            self.paths=[]
            return detections, stats
        chosen=None
        gap=self.frame-self.seen
        if self.anchor is not None and gap<=4 and self.frame-self.model_seen<=12:
            predicted=tuple(self.anchor[i]+gap*self.velocity[i] for i in (0,1))
            radius=max(12., .7*math.hypot(*self.velocity)+8)*gap
            ranked=sorted((math.dist(point,predicted),index) for index,(point,_) in enumerate(candidates))
            if ranked and ranked[0][0]<radius and (len(ranked)==1 or ranked[1][0]>ranked[0][0]+8):
                chosen=candidates[ranked[0][1]]
        # Independent reacquisition needs three successive moving components
        # following a locally consistent trajectory, not one bright pixel.
        paths=[]
        for point,box in candidates:
            matches=[]
            for old in self.paths:
                delta=tuple(point[i]-old['point'][i] for i in (0,1))
                speed=math.hypot(*delta)
                if not (1.5<speed<60): continue
                error=math.dist(delta,old['velocity']) if old['count']>1 else 0
                if old['count']>1 and error>max(6,.5*speed): continue
                matches.append((error+speed*.05,old,delta))
            if matches:
                _,old,delta=min(matches,key=lambda item:item[0])
                paths.append({'point':point,'box':box,'velocity':delta,'count':old['count']+1})
            else:
                paths.append({'point':point,'box':box,'velocity':(0.,0.),'count':1})
        self.paths=paths
        confirmed=[p for p in paths if p['count']>=3]
        if chosen is None and len(confirmed)==1:
            chosen=(confirmed[0]['point'],confirmed[0]['box'])
        if chosen is not None:
            point,box=chosen
            self._observe(point)
            stats['recovered']=1
            return detections+[Detection('ball',box,.25,source='motion')],stats
        return detections,stats

    def _observe(self, point):
        gap=self.frame-self.seen
        if self.anchor is not None and gap<=4:
            self.velocity=tuple((point[i]-self.anchor[i])/gap for i in (0,1))
        else:
            self.velocity=(0.,0.)
        self.anchor=point
        self.seen=self.frame
