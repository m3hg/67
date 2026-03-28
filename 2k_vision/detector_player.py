"""
detector_player.py — Player detection using background subtraction.

Uses OpenCV's MOG2 background subtractor to isolate moving objects,
then filters blobs by size and aspect ratio (tall/narrow = person).
Runs every N frames to save CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import PlayerConfig
from .utils import Rect, morph_open_close, resize_half

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PlayerResult:
    """Detection result for a single player."""

    rect: Rect = field(default_factory=lambda: (0, 0, 0, 0))  # x, y, w, h
    is_shooter: bool = False
    confidence: float = 0.0
    center: Tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class PlayerDetector:
    """
    Detect players via MOG2 background subtraction + contour analysis.

    Processes every ``cfg.run_every_n_frames`` frames and caches the
    last result in between to reduce CPU load.
    """

    def __init__(self, cfg: PlayerConfig) -> None:
        self._cfg = cfg
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=cfg.bg_history,
            varThreshold=cfg.bg_var_threshold,
            detectShadows=cfg.bg_detect_shadows,
        )
        self._frame_count = 0
        self._last_results: List[PlayerResult] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_config(self, cfg: PlayerConfig) -> None:
        """Hot-swap configuration."""
        self._cfg = cfg
        # Recreate background subtractor with new params
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=cfg.bg_history,
            varThreshold=cfg.bg_var_threshold,
            detectShadows=cfg.bg_detect_shadows,
        )

    def reset(self) -> None:
        """Clear state between possessions."""
        self._frame_count = 0
        self._last_results = []

    def detect(
        self,
        frame_bgr: np.ndarray,
        meter_rect: Optional[Rect] = None,
    ) -> List[PlayerResult]:
        """
        Detect players in *frame_bgr*.

        Parameters
        ----------
        frame_bgr:
            Full-resolution BGR frame.
        meter_rect:
            If provided, mark the nearest player as is_shooter.

        Returns
        -------
        list[PlayerResult]
            One result per detected player.  Returns cached result when
            skipping this frame.
        """
        self._frame_count += 1
        cfg = self._cfg

        if not cfg.enabled:
            return []

        # Always feed BG subtractor (even on skip frames) to keep model current
        small = resize_half(frame_bgr)
        fg_mask = self._bg_sub.apply(small)

        # Only do full analysis every N frames
        if self._frame_count % cfg.run_every_n_frames != 0:
            return self._last_results

        # --- Morphological cleanup ---
        fg_clean = morph_open_close(fg_mask, 3)
        fg_clean = cv2.threshold(fg_clean, 200, 255, cv2.THRESH_BINARY)[1]

        # --- Find contours ---
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: List[PlayerResult] = []

        scale = 2  # because we processed at half res

        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if not (cfg.min_contour_area <= area * (scale ** 2) <= cfg.max_contour_area):
                continue
            if cw == 0:
                continue
            aspect = ch / cw  # height / width
            if aspect < cfg.min_aspect_ratio:
                continue

            # Scale back to full-frame coords
            gx, gy, gw, gh = cx * scale, cy * scale, cw * scale, ch * scale
            center = (gx + gw // 2, gy + gh // 2)

            # Optional skin-tone confirmation
            conf = 1.0
            if cfg.skin_detect:
                conf = self._skin_confidence(frame_bgr, gx, gy, gw, gh)
                if conf < 0.05:
                    continue

            results.append(
                PlayerResult(
                    rect=(gx, gy, gw, gh),
                    is_shooter=False,
                    confidence=conf,
                    center=center,
                )
            )

        # --- Mark shooter (nearest to shot meter) ---
        if meter_rect is not None and results:
            mx = meter_rect[0] + meter_rect[2] // 2
            my = meter_rect[1] + meter_rect[3] // 2

            def _dist_to_meter(r: PlayerResult) -> float:
                return float(
                    (r.center[0] - mx) ** 2 + (r.center[1] - my) ** 2
                )

            nearest = min(results, key=_dist_to_meter)
            nearest.is_shooter = True

        self._last_results = results
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _skin_confidence(
        self, frame: np.ndarray, x: int, y: int, w: int, h: int
    ) -> float:
        """
        Estimate the fraction of pixels in the crop that match skin tone.

        HSV skin range: H 0-25 or 160-180, S 30-170, V 80-255.
        """
        patch = frame[y : y + h, x : x + w]
        if patch.size == 0:
            return 0.0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        lo1 = np.array([0, 30, 80], dtype=np.uint8)
        hi1 = np.array([25, 170, 255], dtype=np.uint8)
        lo2 = np.array([160, 30, 80], dtype=np.uint8)
        hi2 = np.array([180, 170, 255], dtype=np.uint8)
        m1 = cv2.inRange(hsv, lo1, hi1)
        m2 = cv2.inRange(hsv, lo2, hi2)
        skin_px = int(cv2.countNonZero(cv2.bitwise_or(m1, m2)))
        total_px = max(1, patch.shape[0] * patch.shape[1])
        return skin_px / total_px
