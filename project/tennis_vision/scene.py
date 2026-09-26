"""Fixed-camera singles scene selection; conservative appearance association."""
import math
import cv2
import numpy as np
from .filters import feet_pixel, filter_tracks, overlap, point_box_distance


def inside(point, polygon):
    return cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(map(float, point)), False) >= 0


def appearance(frame, player):
    x1,y1,x2,y2=player.bbox
    w,h=x2-x1,y2-y1
    # Central torso/shorts exclude most court background and swinging equipment.
    crop=frame[max(0,int(y1+.22*h)):min(frame.shape[0],int(y1+.7*h)),
               max(0,int(x1+.25*w)):min(frame.shape[1],int(x2-.25*w))]
    if crop.size == 0: return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1,2],None,[8,4,4],[0,180,0,256,0,256]).flatten()
    return hist/max(float(hist.sum()),1)


class SceneSelector:
    def __init__(self, config=None, fps=30):
        self.config=config or {}
        self.fps=fps
        self.identities={}
        self.pending={}
        self.reacquiring={}
        self.stationary={}
        self.frame=0

    def ball_polygon(self, court, shape):
        h,w=shape[:2]
        configured=self.config.get('ball_polygon')
        if configured: return np.asarray(configured,np.float32)*[w,h]
        if court is None or court.source!='manual': return None
        pts=np.array([[[-.8,-6],[11.77,-6],[11.77,29.77],[-.8,29.77]]],np.float32)
        try: poly=cv2.perspectiveTransform(pts,np.linalg.inv(court.matrix))[0]
        except np.linalg.LinAlgError: return None
        # Leave headroom for airborne balls/serves without including side courts.
        poly[2:,1]-=.16*h
        return poly if np.isfinite(poly).all() else None

    def ball_allowed(self, point, shape, court, players=()):
        h,w=shape[:2]
        # Fixed exclusions always win, including near a confirmed player.
        if any(inside(point,np.asarray(p)*[w,h]) for p in self.config.get('ball_exclusion_polygons',[])):
            return False
        poly=self.ball_polygon(court,shape)
        if poly is None or inside(point,poly):
            return True
        # Allow wide retrievals only beside an identified main player. A ground
        # footprint is not a valid hard boundary for an airborne ball.
        return any(p.identity_id is not None and
                   point_box_distance(point,p.bbox)<=max(12,.35*(p.bbox[3]-p.bbox[1])) for p in players)

    def update(self, frame, detections, court):
        self.frame+=1
        h,w=frame.shape[:2]
        stats={'players_unselected':0,'ball_outside_area':0,'ball_colour_rejected':0,'ball_stationary_rejected':0}
        people=[d for d in detections if d.label=='player']
        # Reuse the wrist/body proximity and one-racquet-per-player assignment.
        # Assign temporary track numbers on copies, never mutate input detections.
        from copy import copy
        support_items=[copy(d) for d in detections if d.label in ('player','racket')]
        person_index=0
        for item in support_items:
            if item.label=='player':
                item.track_id=person_index; person_index+=1
            else: item.player_track_id=None
        associated,_=filter_tracks(support_items,None)
        supported={d.player_track_id for d in associated if d.label=='racket'}
        require_racket=self.config.get('require_racket_confirmation',True)
        stats['player_confirmation_pending']=0
        descriptors=[appearance(frame,p) for p in people]
        chosen=set()
        selected=[]
        # Existing identities may leave the initialization area; don't switch to
        # a similarly dressed distant bystander when identity evidence is weak.
        for role, state in self.identities.items():
            choices=[]
            gap=self.frame-state['seen']
            for i,p in enumerate(people):
                if i in chosen or descriptors[i] is None: continue
                colour=min(cv2.compareHist(ref.astype(np.float32),descriptors[i].astype(np.float32),cv2.HISTCMP_BHATTACHARYYA)
                           for ref in (state['hist'],state['recent_hist']))
                dist=math.dist(p.center,state['center'])/w
                height=max(1,p.bbox[3]-p.bbox[1])
                scale=height/state['height']
                typical_height=float(np.median(state['heights']))
                # Scale-aware motion gate: a tiny far player cannot jump hundreds
                # of pixels to a similarly dressed referee after a few misses.
                reach=min(.35*w, state['height']*(.6+.18*gap))
                local=(gap<=5 and dist*w<.8*state['height'] and
                       overlap(p.bbox,state['bbox'])[0]>.12 and .4<scale<1.8)
                if (colour < .55 or local) and .4<scale<2 and dist*w < reach:
                    # Continuity wins over clothing colour when a small player
                    # turns or moves through shadows. Never relax a distant jump.
                    shape_cost=.7*abs(math.log(height/typical_height))
                    choices.append((dist*w/state['height']+.25*colour+shape_cost-.15*p.confidence,i,local))
            if not choices:
                self.reacquiring.pop(role,None)
                continue
            _,i,local=min(choices)
            if gap>5:
                old=self.reacquiring.get(role)
                close=old and self.frame-old['frame']==1 and math.dist(old['center'],people[i].center)<max(6,.6*state['height'])
                count=old['count']+1 if close else 1
                evidence=(old['racket'] if close else False) or i in supported
                self.reacquiring[role]={'center':people[i].center,'frame':self.frame,'count':count,'racket':evidence}
                if count<3 or (require_racket and not evidence):
                    stats['player_confirmation_pending']+=1
                    continue
            p=people[i]; p.identity_id=role
            chosen.add(i); selected.append(p)
            state['center'],state['seen']=p.center,self.frame
            state['height']=max(1,p.bbox[3]-p.bbox[1])
            state['heights']=(state['heights']+[state['height']])[-7:]
            state['bbox']=p.bbox
            if local:
                state['recent_hist']=.8*state['recent_hist']+.2*descriptors[i]
            self.reacquiring.pop(role,None)
            # Freeze reference colours: avoid accumulating drift into other people.
        for role in ('near','far'):
            if role in self.identities: continue
            candidates=[]
            for i,p in enumerate(people):
                if i in chosen or descriptors[i] is None or p.confidence < .3: continue
                feet=feet_pixel(p)
                region=self.config.get(role+'_seed_polygon')
                if region:
                    valid=inside(feet,np.asarray(region)*[w,h])
                elif court is not None and court.source=='manual':
                    x,y=court.project(feet)
                    valid=0<=x<=10.97 and ((-6<=y<11.885) if role=='near' else (11.885<=y<=29.77))
                else: valid=False
                if valid: candidates.append((p.confidence,i))
            if not candidates:
                self.pending.pop(role,None); continue
            _,i=max(candidates)
            p=people[i]; old=self.pending.get(role)
            close=old and math.dist(old[0],p.center)<max(6,.6*(p.bbox[3]-p.bbox[1]))
            count=old[1]+1 if close else 1
            racket_seen=(old[2] if close else False) or i in supported
            self.pending[role]=(p.center,count,racket_seen)
            if count>=3 and (racket_seen or not require_racket):
                self.identities[role]={'hist':descriptors[i], 'recent_hist':descriptors[i].copy(), 'bbox':p.bbox,
                                       'center':p.center,'seen':self.frame,'height':max(1,p.bbox[3]-p.bbox[1]),
                                       'heights':[max(1,p.bbox[3]-p.bbox[1])]}
                p.identity_id=role; selected.append(p); chosen.add(i)
            else: stats['player_confirmation_pending']+=1
        stats['players_unselected']=len(people)-len(selected)
        balls=[]
        seen_cells=set()
        for b in [d for d in detections if d.label=='ball']:
            if not self.ball_allowed(b.center,frame.shape,court,selected):
                stats['ball_outside_area']+=1; continue
            if b.source=='gridtracknet':
                # Learned multi-frame localization is not constrained by the
                # generic detector's saturated-colour/stationary heuristics.
                balls.append(b); continue
            x1,y1,x2,y2=map(int,b.bbox)
            crop=frame[max(0,y1):min(h,y2+1),max(0,x1):min(w,x2+1)]
            if crop.size == 0: continue
            if self.config.get('yellow_ball_check',True):
                hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
                # Compressed/blurred far balls have weak saturation. Require some
                # chromatic evidence, not a mostly saturated yellow rectangle.
                yellow=cv2.inRange(hsv,np.array([20,35,65]),np.array([65,255,255]))
                if np.count_nonzero(yellow)/yellow.size < self.config.get('ball_yellow_fraction',.03):
                    stats['ball_colour_rejected']+=1; continue
            cell=tuple(int(v/8) for v in b.center)
            last,count=self.stationary.get(cell,(-99,0))
            count=count+1 if self.frame-last<=2 else 1
            if cell not in seen_cells: self.stationary[cell]=(self.frame,count)
            seen_cells.add(cell)
            if count>max(5,int(.35*self.fps)):
                stats['ball_stationary_rejected']+=1; continue
            balls.append(b)
        self.stationary={k:v for k,v in self.stationary.items() if self.frame-v[0]<=2}
        temporal=[b for b in balls if b.source=='gridtracknet']
        if temporal:
            balls=[b for b in balls if b.source!='gridtracknet']+[max(temporal,key=lambda d:d.confidence)]
        return selected+[d for d in detections if d.label=='racket']+balls, stats
