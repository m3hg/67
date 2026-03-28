"""
detector_meter.py — Shot-meter HSV detection for 2k Vision.

Detects the magenta (configurable) shot-meter bar each frame, returning
its position, fill height, and fill percentage.  Uses an adaptive ROI
that narrows to the last-known location for speed after first detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import MeterConfig
from .utils import build_hsv_mask, morph_close, Rect

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MeterResult:
    """Detection result for the shot meter."""

    found: bool = False
    rect: Rect = field(default_factory=lambda: (0, 0, 0, 0))   # x, y, w, h (full meter)
    fill_height: float = 0.0   # pixels of fill from bottom
    fill_pct: float = 0.0      # 0-1 fill percentage
    confidence: float = 0.0    # pixel count confidence  (0-1)
    center_x: int = 0
    center_y: int = 0

    # Sub-pixel precision fill height (for threshold comparison)
    fill_height_subpx: float = 0.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class MeterDetector:
    """
    Detect the shot-meter bar via HSV color thresholding.

    The meter is expected to be a tall, narrow, solid-color rectangle.
    After the first detection the scan is limited to an adaptive ROI for speed.
    """

    # Adaptive ROI padding (px)
    _ROI_PAD_X = 80
    _ROI_PAD_Y = 120

    def __init__(self, cfg: MeterConfig) -> None:
        self._cfg = cfg
        self._last_rect: Optional[Rect] = None  # last confirmed meter rect
        self._adaptive_roi: Optional[Rect] = None  # narrowed scan region
        self._max_height: int = 0  # maximum meter height seen (full bar reference)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_config(self, cfg: MeterConfig) -> None:
        """Hot-swap configuration."""
        self._cfg = cfg

    def reset(self) -> None:
        """Clear tracking state (e.g. between possessions)."""
        self._last_rect = None
        self._adaptive_roi = None
        self._max_height = 0

    def detect(self, frame_bgr: np.ndarray) -> MeterResult:
        """
        Run meter detection on *frame_bgr*.

        Parameters
        ----------
        frame_bgr:
            Full-resolution BGR frame.

        Returns
        -------
        MeterResult
        """
        h, w = frame_bgr.shape[:2]
        cfg = self._cfg

        # --- Choose scan region -------------------------------------------
        roi_rect = self._get_scan_roi(w, h)
        x_off, y_off = roi_rect[0], roi_rect[1]

        roi = frame_bgr[
            roi_rect[1] : roi_rect[1] + roi_rect[3],
            roi_rect[0] : roi_rect[0] + roi_rect[2],
        ]

        # --- HSV mask --------------------------------------------------------
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = self._build_mask(roi_hsv)

        # --- Morphological cleanup -------------------------------------------
        mask = morph_close(mask, cfg.morph_kernel)

        # --- Find contours ---------------------------------------------------
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return MeterResult()

        # --- Filter candidates -----------------------------------------------
        candidates = []
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < cfg.min_area:
                continue
            if not (cfg.min_height <= ch <= cfg.max_height):
                continue
            if not (cfg.min_width <= cw <= cfg.max_width):
                continue
            if ch == 0:
                continue
            aspect = ch / cw
            if aspect < cfg.aspect_ratio_min:
                continue
            candidates.append((cx, cy, cw, ch, cnt))

        if not candidates:
            return MeterResult()

        # --- Pick best candidate (nearest to last known position) -----------
        best = self._pick_best(candidates, x_off, y_off)
        if best is None:
            return MeterResult()

        cx_roi, cy_roi, cw_r, ch_r, cnt = best

        # Convert ROI-local coords to full-frame coords
        gx = cx_roi + x_off
        gy = cy_roi + y_off

        full_rect: Rect = (gx, gy, cw_r, ch_r)
        self._last_rect = full_rect
        self._update_adaptive_roi(gx, gy, cw_r, ch_r, w, h)

        # Track max height for auto-learning reference (reset per shot in engine)
        if ch_r > self._max_height:
            self._max_height = ch_r

        # Determine reference height for fill_pct
        if cfg.ref_height > 0:
            ref_h = cfg.ref_height
        elif self._max_height > 0:
            ref_h = self._max_height
        else:
            ref_h = ch_r

        # --- Measure fill height (mask pixels column profile) ---------------
        cnt_mask_roi = mask[cy_roi : cy_roi + ch_r, cx_roi : cx_roi + cw_r]
        fill_height_px = float(cv2.countNonZero(cnt_mask_roi))
        total_pixels = max(1, cw_r * ch_r)

        # fill_pct: ratio of current height to reference (max observed) height
        fill_pct = min(1.0, ch_r / ref_h)

        # Sub-pixel precision via intensity-weighted centroid in Y
        fill_height_subpx = self._subpixel_fill(roi_hsv, cx_roi, cy_roi, cw_r, ch_r)

        # Confidence: ratio of colored pixels vs bounding box area
        confidence = min(1.0, fill_height_px / (total_pixels + 1e-9))

        return MeterResult(
            found=True,
            rect=full_rect,
            fill_height=fill_height_px,
            fill_pct=fill_pct,
            confidence=confidence,
            center_x=gx + cw_r // 2,
            center_y=gy + ch_r // 2,
            fill_height_subpx=fill_height_subpx,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_mask(self, roi_hsv: np.ndarray) -> np.ndarray:
        """Build HSV mask using config, handling hue wrap-around."""
        cfg = self._cfg
        from .utils import build_hsv_mask as _b
        return _b(
            roi_hsv,
            cfg.hue_low, cfg.hue_high,
            cfg.sat_low, cfg.sat_high,
            cfg.val_low, cfg.val_high,
        )

    def _get_scan_roi(self, w: int, h: int) -> Rect:
        """Return the current scan ROI (adaptive if available)."""
        if self._adaptive_roi is not None:
            return self._adaptive_roi
        return (0, 0, w, h)

    def _update_adaptive_roi(
        self, gx: int, gy: int, cw: int, ch: int, fw: int, fh: int
    ) -> None:
        """Shrink the scan ROI around the last-seen meter position."""
        pad_x = self._ROI_PAD_X
        pad_y = self._ROI_PAD_Y
        rx = max(0, gx - pad_x)
        ry = max(0, gy - pad_y)
        rw = min(fw - rx, cw + 2 * pad_x)
        rh = min(fh - ry, ch + 2 * pad_y)
        self._adaptive_roi = (rx, ry, rw, rh)

    def _pick_best(
        self,
        candidates: list,
        x_off: int,
        y_off: int,
    ):
        """Pick the candidate nearest to the last-known meter position."""
        if self._last_rect is None:
            # No prior position: pick the tallest candidate
            return max(candidates, key=lambda c: c[3])

        lx, ly, lw, lh = self._last_rect
        lc_x = lx + lw // 2
        lc_y = ly + lh // 2

        def _dist(c):
            cx_g = c[0] + x_off + c[2] // 2
            cy_g = c[1] + y_off + c[3] // 2
            return (cx_g - lc_x) ** 2 + (cy_g - lc_y) ** 2

        return min(candidates, key=_dist)

    def _subpixel_fill(
        self,
        roi_hsv: np.ndarray,
        cx: int,
        cy: int,
        cw: int,
        ch: int,
    ) -> float:
        """
        Estimate fill height with sub-pixel precision using the V-channel
        column mean — returns a float pixel count.
        """
        patch = roi_hsv[cy : cy + ch, cx : cx + cw, 2].astype(float)
        if patch.size == 0:
            return 0.0
        col_mean = patch.mean(axis=1)  # shape (ch,)
        threshold = 60.0
        filled_rows = float(np.sum(col_mean > threshold))
        return filled_rows
