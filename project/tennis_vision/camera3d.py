"""Approximate camera pose derived automatically from the calibrated court plane."""
from dataclasses import dataclass

import cv2
import numpy as np

from .court import CourtMapper


@dataclass
class Camera3D:
    K: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    image_size: tuple[int, int]
    confidence: float
    distortion: np.ndarray | None = None

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def position_m(self) -> np.ndarray:
        return (-self.R.T @ self.tvec.reshape(3)).ravel()

    def project(self, points_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
        pixels, _ = cv2.projectPoints(points, self.rvec, self.tvec, self.K, self.distortion)
        return pixels.reshape(-1, 2)

    def pixel_to_plane(self, pixel: tuple[float, float], height_m: float) -> np.ndarray | None:
        normalized = self.normalized_pixels([pixel])[0]
        ray = self.R.T @ np.array([normalized[0], normalized[1], 1.0])
        ray /= np.linalg.norm(ray)
        origin = self.position_m
        if abs(ray[2]) < 1e-8:
            return None
        distance = (height_m - origin[2]) / ray[2]
        return origin + distance * ray if distance > 0 else None

    def normalized_pixels(self,pixels):
        return cv2.undistortPointsIter(np.asarray(pixels,dtype=np.float64).reshape(-1,1,2),
            self.K,self.distortion,None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,1e-10)).reshape(-1,2)


def from_court(court: CourtMapper, width: int, height: int, focal_ratio: float = 1.2) -> Camera3D | None:
    """Solve a plausible camera pose from four detected court corners.

    The court plane alone cannot uniquely determine focal length, so this uses a
    conservative broadcast-camera focal prior. Fit quality is exposed downstream;
    provide a multi-point/manual calibration only when measurement-grade height is
    required.
    """
    f = focal_ratio * max(width, height)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
    # Camera/world frame: origin at court centre; x across, y toward far baseline, z up.
    object_points = np.array([[-10.97 / 2, -23.77 / 2, 0], [10.97 / 2, -23.77 / 2, 0], [10.97 / 2, 23.77 / 2, 0], [-10.97 / 2, 23.77 / 2, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(object_points, court.image_corners.astype(np.float64), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    camera = Camera3D(K, rvec, tvec, (width, height), confidence=round(court.confidence * 0.45, 2))
    # Reject the planar-PnP mirror solution.
    if camera.position_m[2] <= 0 or not np.all((camera.R @ object_points.T + tvec.reshape(3, 1))[2] > 0):
        return None
    return camera


def refine_from_court(court: CourtMapper, width: int, height: int):
    """Fit focal length under explicit pinhole assumptions, with failure gates.

    Square pixels, centred principal point and zero distortion leave focal length
    as the sole intrinsic unknown. Four court corners can constrain it under
    those assumptions, but a low residual is NOT full camera calibration or an
    independent accuracy measurement. Keep the legacy fixed-prior API unchanged.
    """
    world = np.array([[-5.485,-11.885,0], [5.485,-11.885,0],
                      [5.485,11.885,0], [-5.485,11.885,0]], np.float64)
    diagnostics = {"basis":"court_corner_focal_search",
                   "assumptions":"square pixels; centred principal point; zero distortion",
                   "focal_ratio_bounds":[.4,2.8], "status":"unavailable"}
    if width <= 0 or height <= 0:
        diagnostics["reason"] = "Invalid image dimensions"
        return None, diagnostics

    def trial(ratio):
        try:
            camera = from_court(court,width,height,float(ratio))
            if camera is None:
                return None
            residual = float(np.sqrt(np.mean((camera.project(world)-court.image_corners)**2)))
            return (residual,float(ratio),camera) if np.isfinite(residual) else None
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            return None

    prior = trial(1.2)
    diagnostics["prior_coordinate_rmse_px"] = round(prior[0],4) if prior else None
    coarse = [item for r in np.linspace(.4,2.8,49) if (item:=trial(r)) is not None]
    if not coarse:
        diagnostics["reason"] = "No positive-depth court camera solution"
        return None, diagnostics
    best = min(coarse,key=lambda item:item[0])
    fine = [item for r in np.linspace(max(.4,best[1]-.05),min(2.8,best[1]+.05),101)
            if (item:=trial(r)) is not None]
    best = min([best]+fine,key=lambda item:item[0])
    residual,ratio,camera = best
    diagnostics.update(focal_ratio=round(ratio,6),focal_length_px=round(float(camera.K[0,0]),3),
                       coordinate_rmse_px=round(residual,4))
    neighbours = [item for r in (ratio*.8,ratio*1.2) if (item:=trial(r)) is not None]
    if ratio <= .402 or ratio >= 2.798:
        diagnostics["reason"] = "Focal optimum reaches search boundary"
    elif residual > max(3.,.005*np.hypot(width,height)):
        diagnostics["reason"] = "Court corners do not support this pinhole camera"
    elif not neighbours or max(item[0]-residual for item in neighbours) < .25:
        diagnostics["reason"] = "Focal length is weakly constrained by this view"
    else:
        diagnostics["status"] = "approximate"
        return camera, diagnostics
    return None, diagnostics
