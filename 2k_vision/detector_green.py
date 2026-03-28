"""
detector_green.py — Green-release flash detection for 2k Vision.

Detects the bright green "perfect release" flash that briefly appears
near the basket when the player releases the ball at the perfect moment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import GreenConfig
from .utils import build_hsv_mask, Rect

log = logging.getLogger(__name__)

Point = Tuple[int, int]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GreenFlashResult:
    """Result of the green-release flash detector."""

    found: bool = False
    position: Point = (0, 0)          # centroid of the flash
    intensity: float = 0.0            # normalised pixel count (0-1)
    frames_remaining: int = 0         # countdown until flash is dismissed
    pixel_count: int = 0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class GreenDetector:
    """
    Detect the bright green "swish" flash.

    After detection the result is held for ``cfg.flash_duration_frames``
    frames so the overlay can show a sustained burst effect.
    """

    def __init__(self, cfg: GreenConfig) -> None:
        self._cfg = cfg
        self._prev_mask: Optional[np.ndarray] = None
        self._frames_remaining: int = 0
        self._last_result: GreenFlashResult = GreenFlashResult()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_config(self, cfg: GreenConfig) -> None:
        self._cfg = cfg

    def reset(self) -> None:
        self._prev_mask = None
        self._frames_remaining = 0
        self._last_result = GreenFlashResult()

    def detect(self, frame_bgr: np.ndarray) -> GreenFlashResult:
        """
        Detect the green release flash in *frame_bgr*.

        Parameters
        ----------
        frame_bgr:
            Full BGR frame.

        Returns
        -------
        GreenFlashResult
        """
        cfg = self._cfg
        h, w = frame_bgr.shape[:2]

        # Only scan the top portion of the screen (near basket)
        top_h = int(h * cfg.screen_top_fraction)
        roi = frame_bgr[:top_h, :]

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = build_hsv_mask(
            roi_hsv,
            cfg.hue_low, cfg.hue_high,
            cfg.sat_low, cfg.sat_high,
            cfg.val_low, cfg.val_high,
        )

        pixel_count = int(cv2.countNonZero(mask))

        # Frame-to-frame delta: detect *sudden* appearance
        is_new_flash = False
        if self._prev_mask is not None:
            prev_count = int(cv2.countNonZero(self._prev_mask))
            delta = pixel_count - prev_count
            if delta >= cfg.min_pixel_area and pixel_count >= cfg.min_pixel_area:
                is_new_flash = True
        elif pixel_count >= cfg.min_pixel_area:
            is_new_flash = True

        self._prev_mask = mask.copy()

        if is_new_flash:
            self._frames_remaining = cfg.flash_duration_frames
            pos = self._compute_centroid(mask)
            self._last_result = GreenFlashResult(
                found=True,
                position=pos,
                intensity=min(1.0, pixel_count / (top_h * w * 0.05 + 1)),
                frames_remaining=self._frames_remaining,
                pixel_count=pixel_count,
            )
        elif self._frames_remaining > 0:
            self._frames_remaining -= 1
            self._last_result = GreenFlashResult(
                found=True,
                position=self._last_result.position,
                intensity=self._last_result.intensity,
                frames_remaining=self._frames_remaining,
                pixel_count=self._last_result.pixel_count,
            )
        else:
            self._last_result = GreenFlashResult()

        return self._last_result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_centroid(self, mask: np.ndarray) -> Point:
        """Return the (x, y) image-moments centroid of the mask."""
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return (mask.shape[1] // 2, mask.shape[0] // 2)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
