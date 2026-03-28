"""
predictor.py — v9 EMA velocity + 4-zone graduated fire system.

Smooths the shot-meter fill velocity with an Exponential Moving Average,
rejects outliers, scores confidence, and decides WHEN to fire using
four escalating trigger zones.
"""

from __future__ import annotations

import enum
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from .config import PredictorConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fire zone enum
# ---------------------------------------------------------------------------


class Zone(enum.Enum):
    NONE = "NONE"          # No trigger; still waiting
    Z1_DIRECT = "Z1 DIRECT"       # Fill crossed threshold — fire immediately
    Z1B_LAST = "Z1b LAST-CHANCE"  # Within buffer pixels of threshold
    Z2_HI_CONF = "Z2 HI-CONF"     # High confidence + 85% fill — predictive
    Z3_MED_CONF = "Z3 MED-CONF"   # Medium confidence + 92% fill — predictive
    Z4_SAFE = "Z4 SAFE"           # Low confidence — no prediction


# ---------------------------------------------------------------------------
# Fire decision dataclass
# ---------------------------------------------------------------------------


@dataclass
class FireDecision:
    """Result of the predictor on each frame."""

    should_fire: bool = False
    zone: Zone = Zone.NONE
    fill_pct: float = 0.0
    velocity: float = 0.0       # px/frame (EMA-smoothed)
    confidence: float = 0.0     # 0-1 — how consistent the velocity is
    predicted_fill: float = 0.0  # projected fill when shot would release


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Predictor:
    """
    EMA-smoothed velocity predictor with 4 fire zones.

    Call ``update(fill_pct, fps)`` each frame; ``update`` returns a
    :class:`FireDecision` indicating whether to fire now.
    """

    _HIST_LEN = 20  # frames of velocity history for confidence

    def __init__(self, cfg: PredictorConfig) -> None:
        self._cfg = cfg
        self._ema_vel: float = 0.0
        self._prev_fill: Optional[float] = None
        self._vel_history: Deque[float] = deque(maxlen=self._HIST_LEN)
        self._fired_this_shot: bool = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_config(self, cfg: PredictorConfig) -> None:
        self._cfg = cfg

    def reset(self) -> None:
        """Reset for next shot."""
        self._ema_vel = 0.0
        self._prev_fill = None
        self._vel_history.clear()
        self._fired_this_shot = False

    def update(self, fill_pct: float, fps: float = 60.0) -> FireDecision:
        """
        Process one frame's meter fill.

        Parameters
        ----------
        fill_pct:
            Current meter fill, 0-1.
        fps:
            Current rendering FPS (used for lead-time calculation).

        Returns
        -------
        FireDecision
        """
        cfg = self._cfg

        if self._fired_this_shot:
            return FireDecision(fill_pct=fill_pct)

        # --- Compute raw velocity (fill delta per frame) -----------------
        if self._prev_fill is not None:
            raw_vel = fill_pct - self._prev_fill
        else:
            raw_vel = 0.0
        self._prev_fill = fill_pct

        # --- Outlier rejection (>3σ from running history) ----------------
        if self._vel_history:
            arr = list(self._vel_history)
            import numpy as np
            mean_v = float(np.mean(arr))
            std_v = float(np.std(arr))
            if abs(raw_vel - mean_v) > cfg.outlier_sigma * std_v + 1e-9:
                raw_vel = mean_v  # clamp to mean

        # --- EMA smoothing -----------------------------------------------
        alpha = cfg.ema_alpha
        self._ema_vel = alpha * raw_vel + (1 - alpha) * self._ema_vel
        self._vel_history.append(self._ema_vel)

        vel = self._ema_vel

        # --- Confidence (1 − coefficient-of-variation) -------------------
        confidence = self._calc_confidence()

        # --- Lead time (frames until signal travels through system) ------
        lead_frames = self._calc_lead_frames(fps)

        # --- Zone logic --------------------------------------------------
        threshold = cfg.fire_pct
        buf_px = cfg.buffer_px / 100.0  # convert px approximation to fill fraction

        decision = FireDecision(
            fill_pct=fill_pct,
            velocity=vel,
            confidence=confidence,
        )

        # Z1: crossed threshold directly
        if fill_pct >= threshold:
            decision.should_fire = True
            decision.zone = Zone.Z1_DIRECT
            decision.predicted_fill = fill_pct
            self._fired_this_shot = True
            return decision

        # Z1b: last-chance buffer below threshold
        if fill_pct >= threshold - buf_px:
            decision.should_fire = True
            decision.zone = Zone.Z1B_LAST
            decision.predicted_fill = fill_pct
            self._fired_this_shot = True
            return decision

        # Predictive zones — need positive velocity
        if vel <= 0:
            decision.zone = Zone.Z4_SAFE
            return decision

        # Z2: high confidence predictive
        if confidence >= cfg.z2_conf and fill_pct >= cfg.z2_fill:
            predicted = fill_pct + vel * lead_frames * cfg.z2_lead_frac
            if predicted < threshold * cfg.overshoot_guard:
                if predicted >= threshold:
                    decision.should_fire = True
                    decision.zone = Zone.Z2_HI_CONF
                    decision.predicted_fill = predicted
                    self._fired_this_shot = True
                    return decision

        # Z3: medium confidence predictive
        if confidence >= cfg.z3_conf and fill_pct >= cfg.z3_fill:
            predicted = fill_pct + vel * lead_frames * cfg.z3_lead_frac
            if predicted < threshold * cfg.overshoot_guard:
                if predicted >= threshold:
                    decision.should_fire = True
                    decision.zone = Zone.Z3_MED_CONF
                    decision.predicted_fill = predicted
                    self._fired_this_shot = True
                    return decision

        # Z4: safe — do nothing
        decision.zone = Zone.Z4_SAFE
        return decision

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calc_confidence(self) -> float:
        """
        Confidence = 1 − CV of recent velocity history.
        High CV (noisy) = low confidence.
        """
        hist = list(self._vel_history)
        if len(hist) < 3:
            return 0.0
        import numpy as np
        arr = np.array(hist)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr))
        if abs(mean_v) < 1e-9:
            return 0.0
        cv = std_v / abs(mean_v)
        # Clamp to 0-1; lower CV = higher confidence
        return max(0.0, min(1.0, 1.0 - cv))

    def _calc_lead_frames(self, fps: float) -> float:
        """
        Estimate how many frames ahead to fire to compensate for
        input-pipeline delay (controller + capture + processing lag).
        Approximately 3-5 frames at 60fps.
        """
        if fps <= 0:
            fps = 60.0
        # Baseline: ~50ms total latency (controller + capture + processing)
        latency_ms = 50.0
        return latency_ms * fps / 1000.0
