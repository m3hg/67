"""
feedback.py — Post-shot feedback color scanning for 2k Vision.

Scans the frame after a shot is fired and classifies the release quality
by detecting the colored feedback banner the game displays.

Colors detected:
  GREEN  → TIMING: EXCELLENT (perfect release)
  WHITE  → TIMING: GOOD
  YELLOW → TIMING: SLIGHTLY LATE / EARLY
  RED    → TIMING: LATE / EARLY (bad)
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Release quality enum
# ---------------------------------------------------------------------------


class ReleaseQuality(enum.Enum):
    UNKNOWN = "UNKNOWN"
    EXCELLENT = "EXCELLENT"    # green
    GOOD = "GOOD"              # white/light
    SLIGHTLY_OFF = "SLIGHTLY_OFF"  # yellow
    BAD = "BAD"                # red


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FeedbackResult:
    """Result of post-shot feedback scanning."""

    found: bool = False
    quality: ReleaseQuality = ReleaseQuality.UNKNOWN
    color_name: str = "UNKNOWN"
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Color ranges (HSV) for feedback banner
# ---------------------------------------------------------------------------

_COLOR_RANGES = {
    "green": {
        "lo": np.array([35, 80, 150], dtype=np.uint8),
        "hi": np.array([90, 255, 255], dtype=np.uint8),
        "quality": ReleaseQuality.EXCELLENT,
    },
    "yellow": {
        "lo": np.array([18, 80, 150], dtype=np.uint8),
        "hi": np.array([35, 255, 255], dtype=np.uint8),
        "quality": ReleaseQuality.SLIGHTLY_OFF,
    },
    "red_lo": {
        "lo": np.array([0, 80, 100], dtype=np.uint8),
        "hi": np.array([10, 255, 255], dtype=np.uint8),
        "quality": ReleaseQuality.BAD,
    },
    "red_hi": {
        "lo": np.array([160, 80, 100], dtype=np.uint8),
        "hi": np.array([180, 255, 255], dtype=np.uint8),
        "quality": ReleaseQuality.BAD,
    },
    "white": {
        "lo": np.array([0, 0, 200], dtype=np.uint8),
        "hi": np.array([180, 40, 255], dtype=np.uint8),
        "quality": ReleaseQuality.GOOD,
    },
}

# Minimum pixels to count as a detection
_MIN_PIXELS = 300


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class FeedbackScanner:
    """
    Scan for post-shot feedback color banner.

    The banner typically appears in the center-lower portion of the screen.
    We scan the bottom 40% to avoid picking up court colors.
    """

    # Region to scan (fraction of frame): y from 0.3 to 0.7 of height
    _SCAN_Y_START = 0.30
    _SCAN_Y_END = 0.70

    def scan(self, frame_bgr: np.ndarray) -> FeedbackResult:
        """
        Scan *frame_bgr* for a feedback color banner.

        Returns
        -------
        FeedbackResult
        """
        h, w = frame_bgr.shape[:2]
        y1 = int(h * self._SCAN_Y_START)
        y2 = int(h * self._SCAN_Y_END)
        roi = frame_bgr[y1:y2, :]

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        best_color = None
        best_count = 0
        best_quality = ReleaseQuality.UNKNOWN

        for name, spec in _COLOR_RANGES.items():
            mask = cv2.inRange(roi_hsv, spec["lo"], spec["hi"])
            count = int(cv2.countNonZero(mask))
            if count > best_count and count >= _MIN_PIXELS:
                best_count = count
                best_color = name.replace("_lo", "").replace("_hi", "")
                best_quality = spec["quality"]

        if best_color is None:
            return FeedbackResult()

        total_px = roi.shape[0] * roi.shape[1]
        confidence = min(1.0, best_count / (total_px * 0.1 + 1))

        return FeedbackResult(
            found=True,
            quality=best_quality,
            color_name=best_color.upper(),
            confidence=confidence,
        )
