"""
pipeline.py — Detection pipeline orchestrator for 2k Vision.

Runs all detectors on each captured frame and packages results into a
FrameResult dataclass that is consumed by the overlay, UI, and engine.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import AppConfig
from .detector_ball import BallDetector, BallResult
from .detector_green import GreenDetector, GreenFlashResult
from .detector_meter import MeterDetector, MeterResult
from .detector_player import PlayerDetector, PlayerResult
from .feedback import FeedbackScanner, FeedbackResult
from .tracker_fps import FPSTracker

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FrameResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class FrameResult:
    """All detection results for a single captured frame."""

    frame: Optional[np.ndarray] = None        # Raw BGR frame
    timestamp_ms: float = 0.0                  # Monotonic time in ms
    fps: float = 0.0

    meter: MeterResult = field(default_factory=MeterResult)
    ball: BallResult = field(default_factory=BallResult)
    players: List[PlayerResult] = field(default_factory=list)
    green: GreenFlashResult = field(default_factory=GreenFlashResult)
    feedback: FeedbackResult = field(default_factory=FeedbackResult)

    # Convenience flags
    shot_active: bool = False    # Set by engine: True while X is held


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """
    Detection pipeline — called per-frame from the engine or preview mode.

    Detectors are run according to their configured cadence:
      - Meter:   every frame
      - Ball:    every frame
      - Player:  every N frames (default 3)
      - Green:   every frame
      - Feedback: on-demand (called by engine after FIRED state)
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._meter = MeterDetector(cfg.meter)
        self._ball = BallDetector(cfg.ball)
        self._player = PlayerDetector(cfg.player)
        self._green = GreenDetector(cfg.green)
        self._feedback_scanner = FeedbackScanner()
        self._fps = FPSTracker(window=60)
        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_config(self, cfg: AppConfig) -> None:
        """Hot-swap configuration."""
        self._cfg = cfg
        self._meter.update_config(cfg.meter)
        self._ball.update_config(cfg.ball)
        self._player.update_config(cfg.player)
        self._green.update_config(cfg.green)

    def reset(self) -> None:
        """Reset all detectors (call between possessions)."""
        self._meter.reset()
        self._ball.reset()
        self._player.reset()
        self._green.reset()
        self._fps.reset()
        self._frame_idx = 0

    def process(self, frame_bgr: np.ndarray, shot_active: bool = False) -> FrameResult:
        """
        Run all detectors on *frame_bgr*.

        Parameters
        ----------
        frame_bgr:
            BGR uint8 frame from the capture backend.
        shot_active:
            True when the engine is in SHOOTING state (X button held).

        Returns
        -------
        FrameResult
        """
        self._fps.tick()
        self._frame_idx += 1
        ts = time.monotonic() * 1000.0

        meter = self._meter.detect(frame_bgr)
        ball = self._ball.detect(frame_bgr)
        players = self._player.detect(
            frame_bgr,
            meter_rect=meter.rect if meter.found else None,
        )
        green = self._green.detect(frame_bgr)

        return FrameResult(
            frame=frame_bgr,
            timestamp_ms=ts,
            fps=self._fps.fps,
            meter=meter,
            ball=ball,
            players=players,
            green=green,
            feedback=FeedbackResult(),
            shot_active=shot_active,
        )

    def scan_feedback(self, frame_bgr: np.ndarray) -> FeedbackResult:
        """
        Run the feedback scanner (called by engine after a shot fires).
        """
        return self._feedback_scanner.scan(frame_bgr)

    @property
    def fps_tracker(self) -> FPSTracker:
        return self._fps
