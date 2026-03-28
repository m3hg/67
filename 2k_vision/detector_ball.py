"""
detector_ball.py — Basketball detection + Kalman-filter tracker.

Detects orange basketballs via HSV thresholding + circularity filtering,
smooths their positions with a simple 2-D Kalman filter, maintains a
30-frame history, and fits a parabolic arc to the trajectory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import BallConfig
from .utils import (
    build_hsv_mask,
    contour_circularity,
    fit_parabola,
    eval_parabola,
    morph_open_close,
)

log = logging.getLogger(__name__)

Point = Tuple[int, int]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BallResult:
    """Detection + tracking result for the basketball."""

    found: bool = False
    position: Point = (0, 0)           # Current (x, y) in frame coords
    radius: int = 0
    trajectory: List[Point] = field(default_factory=list)   # last N positions
    predicted_arc: List[Point] = field(default_factory=list)  # forward prediction
    speed: float = 0.0                 # px/frame
    airborne: bool = False
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Tiny 2-D constant-velocity Kalman filter
# ---------------------------------------------------------------------------


class BallKalman:
    """
    Minimal 2-D Kalman filter tracking (x, y) with velocity.

    State vector: [x, y, vx, vy]
    """

    def __init__(self, process_noise: float = 1e-3, meas_noise: float = 1e-1) -> None:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self._kf = kf
        self._initialised = False

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Feed a measurement; return filtered position (x, y)."""
        meas = np.array([[x], [y]], dtype=np.float32)
        if not self._initialised:
            self._kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self._initialised = True
        self._kf.predict()
        corrected = self._kf.correct(meas)
        return float(corrected[0, 0]), float(corrected[1, 0])

    def predict(self) -> Tuple[float, float]:
        """Predict next position without a measurement."""
        pred = self._kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current velocity estimate (vx, vy) in px/frame."""
        state = self._kf.statePost
        return float(state[2, 0]), float(state[3, 0])

    def reset(self) -> None:
        self._initialised = False
        self._kf.statePost = np.zeros((4, 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class BallDetector:
    """
    Detect and track the basketball across frames.

    After each detection the position is smoothed by a Kalman filter,
    stored in a 30-frame history, and a parabolic arc is fitted.
    """

    _PRED_STEPS = 20  # how many future frames to predict

    def __init__(self, cfg: BallConfig) -> None:
        self._cfg = cfg
        self._kalman = BallKalman(cfg.kalman_process_noise, cfg.kalman_meas_noise)
        self._history: list[Tuple[float, float]] = []   # (x, y) per frame
        self._radius_history: list[int] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_config(self, cfg: BallConfig) -> None:
        self._cfg = cfg

    def reset(self) -> None:
        self._kalman.reset()
        self._history.clear()
        self._radius_history.clear()

    def detect(self, frame_bgr: np.ndarray) -> BallResult:
        """
        Detect the basketball in *frame_bgr*.

        Returns
        -------
        BallResult
        """
        cfg = self._cfg

        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = build_hsv_mask(
            frame_hsv,
            cfg.hue_low, cfg.hue_high,
            cfg.sat_low, cfg.sat_high,
            cfg.val_low, cfg.val_high,
        )
        mask = morph_open_close(mask, 3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[Tuple[int, int, int, float]] = None  # (cx, cy, r, circularity)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            circ = contour_circularity(cnt)
            if circ < cfg.circularity_threshold:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            r = int(radius)
            if not (cfg.min_radius <= r <= cfg.max_radius):
                continue
            if best is None or circ > best[3]:
                best = (int(cx), int(cy), r, circ)

        if best is None:
            # Kalman predict-only step
            px, py = self._kalman.predict()
            return self._build_result(found=False, x=px, y=py, radius=0)

        x_raw, y_raw, radius, circ = best
        fx, fy = self._kalman.update(float(x_raw), float(y_raw))
        return self._build_result(found=True, x=fx, y=fy, radius=radius, confidence=circ)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        found: bool,
        x: float,
        y: float,
        radius: int,
        confidence: float = 0.0,
    ) -> BallResult:
        cfg = self._cfg

        # Update history
        self._history.append((x, y))
        if len(self._history) > cfg.history_len:
            self._history = self._history[-cfg.history_len:]

        trajectory: list[Point] = [(int(p[0]), int(p[1])) for p in self._history]

        # Speed
        speed = 0.0
        if len(self._history) >= 2:
            dx = self._history[-1][0] - self._history[-2][0]
            dy = self._history[-1][1] - self._history[-2][1]
            speed = float(np.hypot(dx, dy))

        # Airborne detection: needs upward then downward motion trend
        airborne = self._check_airborne()

        # Parabolic arc prediction
        predicted_arc = self._predict_arc()

        return BallResult(
            found=found,
            position=(int(x), int(y)),
            radius=radius,
            trajectory=trajectory,
            predicted_arc=predicted_arc,
            speed=speed,
            airborne=airborne,
            confidence=confidence,
        )

    def _check_airborne(self) -> bool:
        """Heuristic: ball is airborne if recent vy was negative (up) then positive (down)."""
        if len(self._history) < 6:
            return False
        ys = [p[1] for p in self._history[-6:]]
        deltas = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        # Up then down = negative deltas followed by positive
        went_up = any(d < -2 for d in deltas[:3])
        came_down = any(d > 2 for d in deltas[2:])
        return went_up or came_down

    def _predict_arc(self) -> list[Point]:
        """Fit a parabola to history and project forward N steps."""
        if len(self._history) < 3:
            return []
        xs = np.array([p[0] for p in self._history], dtype=float)
        ys = np.array([p[1] for p in self._history], dtype=float)
        coeffs = fit_parabola(xs, ys)
        if coeffs is None:
            return []

        # Extrapolate self._PRED_STEPS frames beyond last known x
        vx, vy = self._kalman.velocity
        last_x = xs[-1]
        step = vx if abs(vx) > 0.5 else 2.0  # pixels per frame
        future_xs = np.array(
            [last_x + step * (i + 1) for i in range(self._PRED_STEPS)], dtype=float
        )
        future_ys = eval_parabola(coeffs, future_xs)

        h = 1080  # default clip
        arc: list[Point] = []
        for px, py in zip(future_xs, future_ys):
            if 0 <= py <= h:
                arc.append((int(px), int(py)))
        return arc
