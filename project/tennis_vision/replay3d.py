"""Interactive virtual-court replay for a Tennis Vision events export.

This is an honest 3D view of the court and player foot positions. With a single
uncalibrated video camera, ball locations are ground-plane projections; their
height is not invented. Add a calibrated-camera / physics fit later to render
true aerial ball arcs.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

COURT_WIDTH, COURT_LENGTH = 10.97, 23.77
NET_HEIGHT, POST_HEIGHT = 0.914, 1.07


def point(court_m: list[float], height: float = 0.0) -> list[float]:
    """Convert exported near-left court coordinates to centred 3D metres."""
    return [court_m[0] - COURT_WIDTH / 2, court_m[1] - COURT_LENGTH / 2, height]


def valid_court_position(court_m: list[float], margin_m: float = 1.2) -> bool:
    """Reject projections that would put a player/ball implausibly off court."""
    return -margin_m <= court_m[0] <= COURT_WIDTH + margin_m and -margin_m <= court_m[1] <= COURT_LENGTH + margin_m


def court_lines() -> list[list[list[float]]]:
    half_w, half_l, half_s = COURT_WIDTH / 2, COURT_LENGTH / 2, 8.23 / 2
    service = 6.40
    return [
        [[-half_w, -half_l, 0], [half_w, -half_l, 0]], [[-half_w, half_l, 0], [half_w, half_l, 0]],
        [[-half_w, -half_l, 0], [-half_w, half_l, 0]], [[half_w, -half_l, 0], [half_w, half_l, 0]],
        [[-half_s, -half_l, 0], [-half_s, half_l, 0]], [[half_s, -half_l, 0], [half_s, half_l, 0]],
        [[-half_s, -service, 0], [half_s, -service, 0]], [[-half_s, service, 0], [half_s, service, 0]],
        [[0, -service, 0], [0, service, 0]],
    ]


def net_lines() -> list[list[list[float]]]:
    post_x = COURT_WIDTH / 2 + 0.914
    return [
        [[-post_x, 0, 0], [post_x, 0, 0]], [[-post_x, 0, POST_HEIGHT], [0, 0, NET_HEIGHT]],
        [[0, 0, NET_HEIGHT], [post_x, 0, POST_HEIGHT]], [[-post_x, 0, 0], [-post_x, 0, POST_HEIGHT]],
        [[post_x, 0, 0], [post_x, 0, POST_HEIGHT]],
    ]


def load_events(path: Path) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    tracks, events = defaultdict(list), defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            frame = int(row["frame"])
            tracks[frame].extend(item for item in row.get("tracks", []) if item.get("court_m") and valid_court_position(item["court_m"]))
            events[frame].extend(item for item in row.get("events", []) if item.get("court_m") and valid_court_position(item["court_m"]))
    return tracks, events


def _load_flights(path: Path | None) -> dict[int, list[list[float]]]:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["frame"]: [item["point_m"]] for fit in data.get("fits", []) for item in fit.get("trajectory_m", [])}


def run(events_path: Path, save: Path | None, flight_path: Path | None = None) -> None:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise SystemExit("Install rerun-sdk first: pip install rerun-sdk") from exc
    tracks_by_frame, events_by_frame = load_events(events_path)
    valid_tracks = sum(len(items) for items in tracks_by_frame.values())
    print(f"Loaded {valid_tracks} on-court track positions. Off-court projections are hidden.")
    fitted_by_frame = _load_flights(flight_path)
    blueprint = rrb.Blueprint(rrb.Horizontal(rrb.Spatial3DView(origin="world", name="Virtual court"), rrb.TextLogView(origin="events", name="Detected events")))
    rr.init("tennis_vision_3d", spawn=save is None, default_blueprint=blueprint)
    if save:
        rr.save(str(save), default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/court/lines", rr.LineStrips3D(court_lines(), colors=[(245, 245, 245)]), static=True)
    rr.log("world/court/net", rr.LineStrips3D(net_lines(), colors=[(255, 0, 255)]), static=True)
    trail: list[list[float]] = []
    fitted_trail: list[list[float]] = []
    for frame in sorted(set(tracks_by_frame) | set(events_by_frame) | set(fitted_by_frame)):
        rr.set_time("frame", sequence=frame)
        player_segments, player_centres, balls = [], [], []
        for track in tracks_by_frame.get(frame, []):
            if track["label"] == "player":
                feet = point(track["court_m"])
                player_segments.append([feet, [feet[0], feet[1], 1.85]])
                player_centres.append([feet[0], feet[1], 0.925])
            elif track["label"] == "ball" and track.get('source','model') in ('model','gridtracknet'):
                # Single-view estimate: the ball is a court-plane projection only.
                balls.append(point(track["court_m"], 0.08))
        if player_segments:
            # A deliberately simple human-scale avatar: body block plus height stick.
            rr.log("world/players/body_models", rr.Boxes3D(centers=player_centres, half_sizes=[(0.28, 0.20, 0.925)], colors=[(80, 160, 255)], labels=["player"] * len(player_centres)))
            rr.log("world/players/height", rr.LineStrips3D(player_segments, radii=0.06, colors=[(80, 160, 255)]))
        if balls:
            trail = (trail + [balls[0]])[-24:]
            rr.log("world/ball/ground_projection", rr.Points3D(balls, radii=0.075, colors=[(0, 220, 255)]))
        if len(trail) > 1:
            rr.log("world/ball/trail_projection", rr.LineStrips3D([trail], radii=0.018, colors=[(0, 220, 255)]))
        if frame in fitted_by_frame:
            fitted_trail = (fitted_trail + fitted_by_frame[frame])[-24:]
            rr.log("world/ball/physics_fit", rr.Points3D(fitted_by_frame[frame], radii=0.085, colors=[(255, 255, 255)]))
        if len(fitted_trail) > 1:
            rr.log("world/ball/physics_fit_trail", rr.LineStrips3D([fitted_trail], radii=0.022, colors=[(255, 255, 255)]))
        for event in events_by_frame.get(frame, []):
            location = point(event["court_m"], 0.03)
            colour = (255, 165, 0) if event["type"] == "bounce_candidate" else (255, 210, 0)
            rr.log(f"world/events/{event['type']}", rr.Points3D([location], radii=0.12, colors=[colour]))
            rr.log("events", rr.TextLog(f"frame {frame}: {event['type']}"))
    if save:
        print(f"Saved interactive replay: {save}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True, help="Tennis Vision events.jsonl")
    parser.add_argument("--flight", type=Path, help="flight3d.json from the same analysis")
    parser.add_argument("--save", type=Path, help="Write a portable .rrd replay instead of opening the viewer")
    args = parser.parse_args()
    run(args.events, args.save, args.flight)


if __name__ == "__main__":
    main()
