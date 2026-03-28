"""
calibrator.py — Adaptive fire_pct calibration from post-shot feedback.

Reads the FeedbackResult after each shot and nudges the fire_pct threshold
up or down to converge on the optimal release timing.
"""

from __future__ import annotations

import logging
from typing import List

from .config import PredictorConfig, save_config, get_config
from .feedback import FeedbackResult, ReleaseQuality

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step sizes
# ---------------------------------------------------------------------------

_STEP_EXCELLENT = 0.002   # small positive reinforcement
_STEP_GOOD = 0.001
_STEP_SLIGHTLY_OFF = -0.005  # nudge toward optimal
_STEP_BAD = -0.010
_MIN_PCT = 0.50
_MAX_PCT = 0.99


class Calibrator:
    """
    Adaptive calibrator that adjusts ``fire_pct`` based on shot feedback.

    After each shot the calibrator logs the result and applies a small
    gradient step to the threshold.  The updated config is persisted to disk.
    """

    def __init__(self, predictor_cfg: PredictorConfig) -> None:
        self._cfg = predictor_cfg
        self._history: List[ReleaseQuality] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def on_shot_feedback(self, result: FeedbackResult) -> None:
        """
        Process a single shot feedback result.

        Parameters
        ----------
        result:
            The FeedbackResult returned by FeedbackScanner after a shot.
        """
        if not result.found:
            return

        self._history.append(result.quality)
        self._adjust(result.quality)
        log.info(
            "Calibrator: feedback=%s fire_pct now=%.3f",
            result.quality.value,
            self._cfg.fire_pct,
        )

    @property
    def fire_pct(self) -> float:
        """Current fire threshold (0-1)."""
        return self._cfg.fire_pct

    @property
    def history(self) -> List[ReleaseQuality]:
        """Read-only shot quality history."""
        return list(self._history)

    def reset(self) -> None:
        """Clear history (does not reset the threshold)."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adjust(self, quality: ReleaseQuality) -> None:
        step_map = {
            ReleaseQuality.EXCELLENT: _STEP_EXCELLENT,
            ReleaseQuality.GOOD: _STEP_GOOD,
            ReleaseQuality.SLIGHTLY_OFF: _STEP_SLIGHTLY_OFF,
            ReleaseQuality.BAD: _STEP_BAD,
            ReleaseQuality.UNKNOWN: 0.0,
        }
        step = step_map.get(quality, 0.0)
        new_val = self._cfg.fire_pct + step
        self._cfg.fire_pct = max(_MIN_PCT, min(_MAX_PCT, new_val))

        # Persist to disk
        try:
            cfg = get_config()
            cfg.predictor.fire_pct = self._cfg.fire_pct
            save_config(cfg)
        except Exception:  # pylint: disable=broad-except
            pass
