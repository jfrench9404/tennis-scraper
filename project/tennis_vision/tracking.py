from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .court import CourtMapper


@dataclass
class Detection:
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: dict[str, list[float]] | None = None
    identity_id: str | None = None
    source: str = 'model'

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class Track:
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    track_id: int
    last_frame: int
    keypoints: dict[str, list[float]] | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    missed: int = 0
    predicted: bool = False
    observed_center: tuple[float, float] | None = None
    observed_frame: int = -1
    player_track_id: int | None = None
    identity_id: str | None = None
    source: str = 'model'

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def as_dict(self, court: "CourtMapper | None") -> dict:
        x1, y1, x2, y2 = self.bbox
        ankles = [self.keypoints[name] for name in ("left_ankle", "right_ankle") if self.keypoints and name in self.keypoints and self.keypoints[name][2] >= 0.25]
        # Ankle average is a materially better player court anchor than the bounding-box bottom.
        pixel = ((sum(p[0] for p in ankles) / len(ankles), sum(p[1] for p in ankles) / len(ankles)) if ankles else ((x1 + x2) / 2, y2)) if self.label == "player" else ((x1 + x2) / 2, (y1 + y2) / 2)
        ground = court.project(pixel) if court else None
        return {"track_id": self.track_id, "identity_id": self.identity_id, "label": self.label, "source": self.source, "bbox": [round(v, 2) for v in self.bbox], "confidence": round(self.confidence, 3), "predicted": self.predicted, "keypoints": self.keypoints, "player_track_id": self.player_track_id, "court_m": ground, "feet_xyz_m": [ground[0]-5.485, ground[1]-11.885, 0.0] if ground and self.label=='player' else None, "position_basis": 'ankles_ground_plane' if ankles else ('bbox_ground_plane' if self.label=='player' else 'image_ray_ground_intersection_not_3d_ball')}


class NearestTracker:
    def __init__(self, max_distance: float = 180.0, max_age: int = 20) -> None:
        self.max_distance, self.max_age, self.next_id = max_distance, max_age, 1
        self.tracks: dict[int, Track] = {}

    def _max_distance(self, label: str) -> float:
        return 280.0 if label == "ball" else self.max_distance

    def _max_missed(self, label: str) -> int:
        # Fast balls and swinging rackets cannot safely be extrapolated through
        # missed frames. Keep only brief player continuity, without stale joints.
        return min(5, self.max_age) if label == "player" else 0

    @staticmethod
    def _shift(box: tuple[float, float, float, float], velocity: tuple[float, float]) -> tuple[float, float, float, float]:
        return (box[0] + velocity[0], box[1] + velocity[1], box[2] + velocity[0], box[3] + velocity[1])

    def update(self, detections: list[Detection], frame: int) -> list[Track]:
        active: list[Track] = []
        available = set(self.tracks)
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            cx, cy = det.center
            candidates = [self.tracks[i] for i in available if self.tracks[i].label == det.label and self.tracks[i].identity_id == det.identity_id]
            best = min(candidates, key=lambda t: (t.center[0] - cx) ** 2 + (t.center[1] - cy) ** 2, default=None)
            if best and (det.identity_id is not None or ((best.center[0] - cx) ** 2 + (best.center[1] - cy) ** 2) ** 0.5 <= self._max_distance(det.label)):
                previous = best.observed_center or best.center
                elapsed = max(1, frame - best.observed_frame)
                dx, dy = (cx - previous[0]) / elapsed, (cy - previous[1]) / elapsed
                best.velocity = (0.65 * best.velocity[0] + 0.35 * dx, 0.65 * best.velocity[1] + 0.35 * dy)
                best.bbox, best.confidence, best.last_frame, best.keypoints, best.missed, best.predicted = det.bbox, det.confidence, frame, det.keypoints, 0, False
                best.observed_center, best.observed_frame = det.center, frame
                best.source = det.source
                available.remove(best.track_id); active.append(best)
            else:
                track = Track(det.label, det.bbox, det.confidence, self.next_id, frame, det.keypoints)
                track.identity_id = det.identity_id
                track.source = det.source
                track.observed_center, track.observed_frame = det.center, frame
                self.tracks[self.next_id] = track; self.next_id += 1; active.append(track)
        # Keep a short motion prediction during detector dropouts. It is visible in
        # exports as predicted=true and never treated as a real ball event.
        for track_id in available:
            track = self.tracks[track_id]
            track.missed += 1
            if track.missed <= self._max_missed(track.label):
                track.bbox = self._shift(track.bbox, track.velocity)
                track.keypoints = None
                track.last_frame, track.predicted = frame, True
                active.append(track)
        self.tracks = {i: t for i, t in self.tracks.items() if t.missed <= self._max_missed(t.label)}
        return active
