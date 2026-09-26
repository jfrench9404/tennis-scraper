import cv2
import numpy as np


class CourtMapper:
    """Map image pixels to a regulation doubles court plane in metres."""

    def __init__(self, image_corners: np.ndarray, source: str = "manual", confidence: float = 1.0) -> None:
        court_corners = np.asarray([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]], dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(image_corners, court_corners)
        self.image_corners = np.asarray(image_corners, dtype=np.float32)
        self.source = source
        self.confidence = confidence

    def project(self, pixel: tuple[float, float]) -> list[float]:
        point = np.asarray([[[pixel[0], pixel[1]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.matrix)[0, 0]
        return [round(float(mapped[0]), 3), round(float(mapped[1]), 3)]


def estimate_court(frame: np.ndarray) -> CourtMapper | None:
    """Estimate the four outer court corners from long, paired line segments.

    This is intentionally conservative: it returns None when the camera angle or
    contrast is unsuitable, instead of emitting plausible-looking false court data.
    A trained court-keypoint model should replace this baseline for production.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    min_length = max(80, int(min(frame.shape[:2]) * 0.17))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=75, minLineLength=min_length, maxLineGap=28)
    if lines is None:
        return None
    candidates = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = (dx * dx + dy * dy) ** 0.5
        if length < min_length:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx))) % 180
        candidates.append((length, angle, (float(x1), float(y1), float(x2), float(y2))))
    if len(candidates) < 4:
        return None
    # A broadcast court has two long baselines with similar direction. Pick the
    # strongest pair separated vertically, then use their endpoint envelope.
    candidates.sort(reverse=True, key=lambda row: row[0])
    for _, angle, first in candidates[:25]:
        parallel = [row for row in candidates[:40] if abs(((row[1] - angle + 90) % 180) - 90) < 8]
        if len(parallel) < 2:
            continue
        a, b = parallel[0][2], parallel[1][2]
        ya, yb = (a[1] + a[3]) / 2, (b[1] + b[3]) / 2
        if abs(ya - yb) < frame.shape[0] * 0.12:
            continue
        near, far = (a, b) if ya > yb else (b, a)
        near_pts = sorted([(near[0], near[1]), (near[2], near[3])])
        far_pts = sorted([(far[0], far[1]), (far[2], far[3])])
        corners = np.asarray([near_pts[0], near_pts[1], far_pts[1], far_pts[0]], dtype=np.float32)
        area = abs(cv2.contourArea(corners))
        if area > frame.shape[0] * frame.shape[1] * 0.03:
            return CourtMapper(corners, source="automatic_hough", confidence=0.35)
    return None
