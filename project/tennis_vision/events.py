from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .court import CourtMapper
    from .tracking import Track


class EventEngine:
    """Conservative heuristics; replace with a labelled temporal classifier for production."""

    def __init__(self, fps: float, court: "CourtMapper | None") -> None:
        self.fps, self.court = fps, court
        self.ball_history: deque[tuple[int, float, float]] = deque(maxlen=5)
        self.cooldowns: dict[str, int] = {"contact": -999, "bounce": -999}
        self.ball_id = None

    def update(self, tracks: list["Track"], frame: int) -> list[dict]:
        balls = [t for t in tracks if t.label == "ball" and not t.predicted and t.source in ('model','gridtracknet')]
        if len(balls) != 1:
            self.ball_history.clear()
            self.ball_id = None
            return []
        ball = balls[0]
        if ball.track_id != self.ball_id or (self.ball_history and frame != self.ball_history[-1][0] + 1):
            self.ball_history.clear()
        self.ball_id = ball.track_id
        bx, by = ball.center
        self.ball_history.append((frame, bx, by))
        events: list[dict] = []
        # Contact candidate: ball within 70 pixels of a custom-detected racket centre.
        rackets = [t for t in tracks if t.label == "racket"]
        if rackets and frame - self.cooldowns["contact"] > max(3, int(self.fps * 0.12)):
            racket = min(rackets, key=lambda t: (t.center[0] - bx) ** 2 + (t.center[1] - by) ** 2)
            distance = ((racket.center[0] - bx) ** 2 + (racket.center[1] - by) ** 2) ** 0.5
            if distance < 70:
                events.append(self._event("contact_candidate", frame, (bx, by), {"racket_track_id": racket.track_id, "distance_px": round(distance, 1)}))
                self.cooldowns["contact"] = frame
        # Broadcast y-axis bounce candidate: descending then ascending ball trajectory.
        if len(self.ball_history) == 5 and frame - self.cooldowns["bounce"] > int(self.fps * 0.18):
            ys = [row[2] for row in self.ball_history]
            if ys[1] < ys[2] < ys[3] and ys[3] > ys[4] and abs(ys[3] - ys[2]) > 2:
                pivot = self.ball_history[3]
                events.append(self._event("bounce_candidate", pivot[0], (pivot[1], pivot[2]), {}))
                self.cooldowns["bounce"] = frame
        return events

    def _event(self, kind: str, frame: int, pixel: tuple[float, float], extra: dict) -> dict:
        return {"type": kind, "frame": frame, "time_s": round(frame / self.fps, 4), "pixel": [round(pixel[0], 1), round(pixel[1], 1)], "court_m": self.court.project(pixel) if self.court else None, **extra}


def _cluster(events: list[dict], gap_s: float, score) -> list[dict]:
    """Keep one best observation from each short burst of the same event."""
    if not events:
        return []
    groups: list[list[dict]] = [[events[0]]]
    for event in events[1:]:
        if event["time_s"] - groups[-1][-1]["time_s"] <= gap_s:
            groups[-1].append(event)
        else:
            groups.append([event])
    return [min(group, key=score) for group in groups]


def build_shots(raw_events: list[dict], min_shot_s: float = 0.42, max_shot_s: float = 6.0) -> dict:
    """Fuse raw frame-level candidates into contacts, bounces and shot intervals.

    A shot begins at one consolidated contact and ends at the next. A bounce between
    them is its landing candidate. This refuses implausibly short/long intervals so
    scoreboard cuts and detector chatter do not become tennis strokes.
    """
    contacts = _cluster(
        sorted((e for e in raw_events if e["type"] == "contact_candidate"), key=lambda e: e["time_s"]),
        gap_s=0.38,
        score=lambda e: e.get("distance_px", float("inf")),
    )
    bounces = _cluster(
        sorted((e for e in raw_events if e["type"] == "bounce_candidate"), key=lambda e: e["time_s"]),
        gap_s=0.30,
        score=lambda e: e["frame"],
    )
    shots: list[dict] = []
    for index, start in enumerate(contacts[:-1]):
        end = contacts[index + 1]
        flight_s = round(end["time_s"] - start["time_s"], 4)
        if not min_shot_s <= flight_s <= max_shot_s:
            continue
        landing = next((bounce for bounce in bounces if start["time_s"] < bounce["time_s"] < end["time_s"]), None)
        shots.append({
            "shot_id": len(shots) + 1,
            "start_contact": start,
            "end_contact": end,
            "flight_s": flight_s,
            "landing": landing,
            "is_volley_candidate": landing is None,
            "classification": "unclassified",
        })
    return {
        "method": "temporal_candidate_fusion_v1",
        "raw_candidate_counts": {"contacts": sum(e["type"] == "contact_candidate" for e in raw_events), "bounces": sum(e["type"] == "bounce_candidate" for e in raw_events)},
        "consolidated_counts": {"contacts": len(contacts), "bounces": len(bounces), "shots": len(shots)},
        "contacts": contacts,
        "bounces": bounces,
        "shots": shots,
    }
