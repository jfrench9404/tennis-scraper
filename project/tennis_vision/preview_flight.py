"""Display-only, unvalidated flight previews from possible bounce landings.

This module neither confirms events nor writes to the reviewed-flight channel.
Every curve is conditional on a candidate really being a bounce and the camera
model being adequate. Pixel agreement is NOT evidence of accurate metric depth.
"""
from collections import Counter
import numpy as np

from .validated_flight import BALL_RADIUS, fit_landing_arc


def fit_preview_flights(camera, events, balls, fps, cuts=()):
    fits, skipped = [], []
    active = sorted((e for e in events if e.get("status")!="rejected"),key=lambda e:e["frame"])
    minimum_span = max(10,round(fps/3))
    for event in active:
        if event.get("type")!="bounce":
            continue
        b = event["frame"]
        ground = event.get("court_m") if event.get("status")=="confirmed" else event.get("candidate_court_m")
        reason = None
        if camera is None:
            reason = "No plausible calibrated camera"
        elif ground is None:
            reason = "No observed/corrected landing pixel projects into the supported region"
        elif not 0<=b<len(balls) or balls[b] is None:
            reason = "Preview requires an observed ball at the candidate bounce frame"
        elif any(c<=b for c in cuts):
            reason = "Camera cut: a new calibration is needed before previewing later frames"
        elif sum(e["frame"]==b for e in active)>1:
            reason = "Conflicting events at the landing frame need review"
        if reason:
            skipped.append({"candidate_bounce_event_id":event["id"],"reason":reason})
            continue
        target = np.array([ground[0]-5.485,ground[1]-11.885,BALL_RADIUS],float)
        # Only a trailing observation window, never a claimed contact-to-contact
        # reconstruction. Do not bridge any unresolved intervening event.
        lower = max([0,b-round(2*fps)]+[e["frame"]+1 for e in active if e["frame"]<b])
        failures = Counter()
        accepted = None
        for a in range(lower,b-minimum_span+1):
            if balls[a] is None:
                continue
            indices = [f for f in range(a,b+1) if balls[f] is not None]
            if len(indices)<10 or len(indices)/(b-a+1)<.8:
                failures["Too few observations for a preview"] += 1
                continue
            fit,why = fit_landing_arc(camera,a,b,target,balls,fps)
            if fit is None:
                failures[why] += 1
                continue
            # Prefer the longest supported window, not a hand-picked lowest-error
            # snippet. The shared fitter retains dropout/geometry/physics gates.
            fit.update(source="gravity_preview_candidate_landing",
                       review_status="unvalidated_preview",
                       candidate_bounce_event_id=event["id"],endpoint_review_status=event["status"],
                       endpoint_constraint="conditional_ground_projection_if_candidate_is_bounce",
                       start_frame_basis="observed_window_not_confirmed_contact",
                       excluded_from_validated_flights=True,
                       unobserved_frames=[f for f in range(a,b+1) if balls[f] is None])
            accepted = fit
            fits.append(fit)
            break
        if accepted is None:
            reason = (failures.most_common(1)[0][0] if failures else
                      "No sufficiently long observed window before this candidate")
            skipped.append({"candidate_bounce_event_id":event["id"],"reason":reason,
                            "failed_window_checks":dict(failures)})
    return {"schema_version":1,"algorithm":"conditional_landing_preview_1",
            "status":"unvalidated_previews_available" if fits else "no_supported_previews",
            "fits":fits,"skipped":skipped,
            "limitation":"Dashed cyan paths are unvalidated single-camera estimates, conditional on possible bounce landings. They are partial observation windows, not complete shot reconstructions. Camera/depth and bounce errors remain possible despite pixel agreement; no event is confirmed, no missing detection is repaired, and unsupported intervals stay empty."}
