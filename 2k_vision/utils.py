"""
utils.py — Shared helpers for 2k Vision.

Provides: color-conversion, geometry, timing, and safe OpenCV wrappers.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]   # x, y, w, h
BgrColor = Tuple[int, int, int]
HsvColor = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def bgr_to_hsv(bgr: BgrColor) -> HsvColor:
    """Convert a single BGR color tuple to HSV."""
    px = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)
    return tuple(int(v) for v in hsv[0, 0])  # type: ignore[return-value]


def hsv_to_bgr(hsv: HsvColor) -> BgrColor:
    """Convert a single HSV color tuple to BGR."""
    px = np.uint8([[list(hsv)]])
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)
    return tuple(int(v) for v in bgr[0, 0])  # type: ignore[return-value]


def build_hsv_mask(
    frame_hsv: np.ndarray,
    hue_low: int,
    hue_high: int,
    sat_low: int,
    sat_high: int,
    val_low: int,
    val_high: int,
) -> np.ndarray:
    """
    Build a binary mask using HSV thresholds with wrap-around hue handling.

    When hue_low > hue_high (e.g. red range that wraps 0/180), the mask
    is created as the union of two ranges.
    """
    lo1 = np.array([hue_low, sat_low, val_low], dtype=np.uint8)
    hi1 = np.array([hue_high, sat_high, val_high], dtype=np.uint8)

    if hue_low <= hue_high:
        return cv2.inRange(frame_hsv, lo1, hi1)

    # Wrap-around: split into [hue_low, 179] ∪ [0, hue_high]
    lo2 = np.array([0, sat_low, val_low], dtype=np.uint8)
    hi2 = np.array([hue_high, sat_high, val_high], dtype=np.uint8)
    lo1_wrap = np.array([hue_low, sat_low, val_low], dtype=np.uint8)
    hi1_wrap = np.array([179, sat_high, val_high], dtype=np.uint8)
    mask_a = cv2.inRange(frame_hsv, lo1_wrap, hi1_wrap)
    mask_b = cv2.inRange(frame_hsv, lo2, hi2)
    return cv2.bitwise_or(mask_a, mask_b)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def rect_center(rect: Rect) -> Point:
    """Return the center point of a (x, y, w, h) rect."""
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def rect_area(rect: Rect) -> int:
    """Return the area of a (x, y, w, h) rect."""
    return rect[2] * rect[3]


def rect_iou(a: Rect, b: Rect) -> float:
    """Intersection-over-Union of two (x, y, w, h) rects."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def contour_circularity(contour: np.ndarray) -> float:
    """Compute circularity = 4πA / P²  (1.0 = perfect circle)."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4.0 * np.pi * area) / (perimeter ** 2)


def fit_parabola(
    xs: np.ndarray, ys: np.ndarray
) -> Optional[np.ndarray]:
    """
    Fit a degree-2 polynomial (parabola) to the given x/y arrays.

    Returns coefficient array [a, b, c] for  y = a*x² + b*x + c,
    or None if fitting fails.
    """
    if len(xs) < 3:
        return None
    try:
        coeffs = np.polyfit(xs, ys, 2)
        return coeffs
    except (np.linalg.LinAlgError, ValueError):
        return None


def eval_parabola(coeffs: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Evaluate a parabola at the given x positions."""
    return np.polyval(coeffs, xs)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def crop_roi(img: np.ndarray, roi: Rect) -> np.ndarray:
    """Crop *img* to the given (x, y, w, h) rectangle. Clamps to bounds."""
    h, w = img.shape[:2]
    x1 = max(0, roi[0])
    y1 = max(0, roi[1])
    x2 = min(w, roi[0] + roi[2])
    y2 = min(h, roi[1] + roi[3])
    return img[y1:y2, x1:x2]


def resize_half(img: np.ndarray) -> np.ndarray:
    """Return image at 50% resolution (fast downscale)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)


def morph_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply morphological close to fill small gaps."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def morph_open(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply morphological open to remove small noise blobs."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def morph_open_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Open then close — removes noise AND fills gaps."""
    return morph_close(morph_open(mask, kernel_size), kernel_size)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def now_ms() -> float:
    """Return current monotonic time in milliseconds."""
    return time.monotonic() * 1000.0


class RateLimit:
    """
    Simple rate limiter: returns True at most once per `interval_ms`.

    Usage::
        rl = RateLimit(33.3)   # ~30 fps
        if rl.ready():
            ...
    """

    def __init__(self, interval_ms: float) -> None:
        self.interval_ms = interval_ms
        self._last: float = 0.0

    def ready(self) -> bool:
        t = now_ms()
        if t - self._last >= self.interval_ms:
            self._last = t
            return True
        return False


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def running_stats(values: list[float]) -> Tuple[float, float]:
    """Return (mean, std) of a list of floats. Returns (0, 0) for empty."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def coefficient_of_variation(values: list[float]) -> float:
    """CV = std / mean (0 = perfectly consistent, high = noisy)."""
    mean, std = running_stats(values)
    if abs(mean) < 1e-9:
        return 0.0
    return std / abs(mean)


# ---------------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------------


def draw_dashed_line(
    img: np.ndarray,
    pt1: Point,
    pt2: Point,
    color: BgrColor,
    thickness: int = 1,
    dash_len: int = 8,
    gap_len: int = 4,
) -> None:
    """Draw a dashed line between pt1 and pt2 on *img* in-place."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    length = max(1, int(np.hypot(dx, dy)))
    ux, uy = dx / length, dy / length
    i = 0
    while i < length:
        end = min(i + dash_len, length)
        px1 = int(x1 + ux * i), int(y1 + uy * i)
        px2 = int(x1 + ux * end), int(y1 + uy * end)
        cv2.line(img, px1, px2, color, thickness)
        i += dash_len + gap_len
