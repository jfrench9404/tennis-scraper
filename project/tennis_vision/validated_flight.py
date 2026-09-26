"""Gravity arcs pinned to human-confirmed bounce events. No SciPy dependency.

One confirmed landing fixes the endpoint; a linear ray/trajectory system solves
the three velocity components. Robust pixel reweighting rejects noisy fits.
Results remain estimates conditional on an approximate camera and reviewed events.
"""
import numpy as np

GRAVITY = 9.81
BALL_RADIUS = .0335


def world_bounce(event):
    ground = event.get("court_m")
    if event.get("type") != "bounce" or event.get("status") != "confirmed" or ground is None:
        return None
    return np.array([ground[0]-5.485, ground[1]-11.885, BALL_RADIUS], dtype=float)


def positions_at(target, velocity_at_bounce, relative_seconds):
    tau = np.asarray(relative_seconds, dtype=float)
    return (target+tau[:, None]*velocity_at_bounce
            + np.column_stack((np.zeros(len(tau)), np.zeros(len(tau)), -.5*GRAVITY*tau*tau)))


def fit_interval(camera, start, bounce, balls, fps):
    target = world_bounce(bounce)
    if target is None:
        return None, "Landing is not a confirmed, located bounce"
    if start.get("status") != "confirmed" or start.get("type") not in ("hit", "bounce"):
        return None, "Start event is not confirmed"
    first_target = world_bounce(start)
    if start["type"] == "bounce" and first_target is None:
        return None, "Starting bounce lacks reviewed ground geometry"
    fit,reason = fit_landing_arc(camera,start["frame"],bounce["frame"],target,balls,fps,
                                start_kind=start["type"],first_target=first_target)
    if fit is not None:
        fit.update(start_event_id=start["id"],bounce_event_id=bounce["id"],
                   source="gravity_fit_reviewed_bounce",endpoint_constraint="exact_reviewed_ground_projection")
    return fit,reason


def fit_landing_arc(camera, a, b, target, balls, fps, start_kind="observation_window", first_target=None):
    """Numerical geometry only; this function never confers review status.

    Callers must label the endpoint's provenance. In particular, a fit to an
    unreviewed candidate is only a conditional preview, NOT a validated flight.
    """
    if camera is None:
        return None, "No plausible calibrated camera"
    if (not np.isfinite(fps) or fps<=0 or type(a) is not int or type(b) is not int
            or not 0<=a<b<len(balls)):
        return None, "Invalid flight bounds or frame rate"
    target = np.asarray(target,dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        return None, "Invalid landing target"
    if not 4 <= b-a <= round(2.5*fps):
        return None, "Flight interval must span 4 frames to 2.5 seconds"
    indices = [f for f in range(a, b+1) if balls[f] is not None]
    if len(indices) < 6 or len(indices)/(b-a+1) < .6 or indices[0]-a > 2 or b-indices[-1] > 1:
        return None, "Too few observed ball pixels or weak interval-end coverage"
    if max(np.diff(indices)) > max(3, round(.2*fps)):
        return None, "Long ball dropout inside flight"
    pixels = np.array([balls[f]["pixel"] for f in indices], dtype=float)
    tau = (np.array(indices)-b)/fps
    normalized = camera.normalized_pixels(pixels)
    # P(t) = B + tau*V_bounce - 1/2*g*tau^2. The two perspective
    # equations per observation are linear in V_bounce with B fixed exactly.
    r, translation = camera.R, camera.tvec.reshape(3)
    matrices = np.stack((r[0]-normalized[:,0,None]*r[2], r[1]-normalized[:,1,None]*r[2]), axis=1)
    fixed = target+np.column_stack((np.zeros(len(tau)), np.zeros(len(tau)), -.5*GRAVITY*tau*tau))
    design = matrices*tau[:,None,None]
    rhs = -(np.einsum('nij,nj->ni',matrices,fixed)
            + np.column_stack((translation[0]-normalized[:,0]*translation[2],
                               translation[1]-normalized[:,1]*translation[2])))
    if first_target is not None:
        duration = (b-a)/fps
        velocity = (target-first_target)/duration-np.array([0,0,.5*GRAVITY*duration])
        condition = None
    else:
        weight = np.ones(len(indices))
        velocity = None
        for _ in range(6):
            system = (design*weight[:,None,None]).reshape(-1,3)
            vector = (rhs*weight[:,None]).ravel()
            condition = float(np.linalg.cond(system))
            if np.linalg.matrix_rank(system) < 3 or not np.isfinite(condition) or condition > 1e5:
                return None, "Camera rays make depth too ill-conditioned"
            velocity = np.linalg.lstsq(system, vector, rcond=None)[0]
            predicted = positions_at(target, velocity, tau)
            depth = (r@predicted.T+translation[:,None])[2]
            if np.any(depth <= .1):
                return None, "Fit goes behind camera"
            error = np.linalg.norm(camera.project(predicted)-pixels,axis=1)
            # Convert ray-equation error to approximately pixel error, then
            # reduce the influence of outliers. Never alter original pixels.
            weight = np.sqrt(np.minimum(1.,4/np.maximum(error,1e-6)))/depth
    predicted = positions_at(target, velocity, tau)
    error = np.linalg.norm(camera.project(predicted)-pixels,axis=1)
    frames = np.arange(a,b+1)
    dense = positions_at(target,velocity,(frames-b)/fps)
    duration = (b-a)/fps
    initial_velocity = velocity+np.array([0,0,GRAVITY*duration])
    if not np.isfinite(dense).all() or np.any((camera.R@dense.T+camera.tvec.reshape(3,1))[2] <= 0):
        return None, "Invalid or behind-camera trajectory"
    if float(np.median(error)) > 6 or float(np.percentile(error,90)) > 14 or np.mean(error>20) > .15:
        return None, "Observed pixels do not support a sufficiently consistent gravity arc"
    if dense[:,2].min() < BALL_RADIUS-.08 or dense[:,2].max() > 12:
        return None, "Estimated height is physically implausible"
    if np.max(np.abs(dense[:,0])) > 15 or np.max(np.abs(dense[:,1])) > 25:
        return None, "Estimated flight leaves supported court/runoff volume"
    if start_kind == "hit" and not .15 <= dense[0,2] <= 4.2:
        return None, "Estimated contact height is implausible"
    if np.linalg.norm(initial_velocity) > 90 or velocity[2] >= -.1:
        return None, "Implausible speed or landing is not descending"
    return {"start_frame": a, "end_frame": b,
            "airborne_xyz_status": "estimated_not_measured",
            "bounce_center_xyz_m": target.tolist(), "observed_frames": indices,
            "median_reprojection_error_px": round(float(np.median(error)),3),
            "p90_reprojection_error_px": round(float(np.percentile(error,90)),3),
            "ray_system_condition": condition,
            "trajectory_m": [{"frame": int(f), "point_m": np.round(p,5).tolist()} for f,p in zip(frames,dense)]}, None


def fit_reviewed_flights(camera, events, balls, fps, cuts=()):
    fits, skipped = [], []
    confirmed = [e for e in events if e["status"] == "confirmed" and e["type"] in ("hit","bounce")]
    for bounce in confirmed:
        if bounce["type"] != "bounce":
            continue
        previous = [e for e in confirmed if e["frame"] < bounce["frame"]]
        start = max(previous, key=lambda e:e["frame"]) if previous else None
        reason = None
        if camera is None:
            reason = "No plausible calibrated camera"
        elif world_bounce(bounce) is None:
            reason = bounce.get("geometry_issue") or "Confirmed bounce lacks ground location"
        elif start is None:
            reason = "No preceding confirmed hit/bounce to bound the flight"
        elif any(start["frame"] < c <= bounce["frame"] for c in cuts):
            reason = "Camera cut inside flight"
        elif any(start["frame"] < e["frame"] < bounce["frame"] and e["status"] != "rejected"
                 and e["status"] != "confirmed" for e in events):
            reason = "Review intervening uncertain events before fitting across a possible impact"
        if reason is None:
            fit, reason = fit_interval(camera,start,bounce,balls,fps)
            if fit:
                fits.append(fit)
        if reason:
            skipped.append({"bounce_event_id":bounce["id"],"reason":reason})
    return {"status": "ok" if fits else "no_supported_flights", "fits":fits, "skipped":skipped,
            "confirmed_bounces":sum(e["type"]=="bounce" for e in confirmed),
            "limitation":"Airborne XYZ is estimated from an approximate single-camera calibration and gravity-only motion. Reviewed bounce timing does not make court calibration or XYZ ground truth. Drag/spin are not modeled; radial lens compensation is used only when supplied by the calibration, and camera/lens errors can remain."}
