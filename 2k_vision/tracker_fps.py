"""
tracker_fps.py — FPS and frame-timing tracker for 2k Vision.

Provides a rolling-window FPS counter and per-frame delta-time,
used by the pipeline and predictor to calibrate lead timing.
"""

from __future__ import annotations

import collections
import time
from typing import Deque


class FPSTracker:
    """
    Rolling-window FPS counter.

    Call ``tick()`` once per frame; read ``fps`` for the current rate.

    Parameters
    ----------
    window:
        Number of frames to average over (default 60).
    """

    def __init__(self, window: int = 60) -> None:
        self._window = max(2, window)
        self._times: Deque[float] = collections.deque(maxlen=self._window)
        self._last_tick: float = 0.0
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def tick(self) -> float:
        """
        Record a new frame tick.

        Returns
        -------
        float
            Delta time in **seconds** since the last call (0 on first call).
        """
        now = time.monotonic()
        if self._last_tick == 0.0:
            dt = 0.0
        else:
            dt = now - self._last_tick
            self._times.append(dt)
        self._last_tick = now
        self._frame_count += 1
        return dt

    @property
    def fps(self) -> float:
        """Current frames-per-second based on the rolling window."""
        if len(self._times) < 2:
            return 0.0
        avg_dt = sum(self._times) / len(self._times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    @property
    def frame_interval_ms(self) -> float:
        """Average time between frames in milliseconds."""
        if len(self._times) == 0:
            return 16.67  # Assume 60 fps
        avg_dt = sum(self._times) / len(self._times)
        return avg_dt * 1000.0

    @property
    def frame_count(self) -> int:
        """Total number of frames ticked since creation."""
        return self._frame_count

    @property
    def last_delta_ms(self) -> float:
        """Delta time of the most recent frame in milliseconds."""
        if not self._times:
            return 0.0
        return self._times[-1] * 1000.0

    def reset(self) -> None:
        """Reset all counters."""
        self._times.clear()
        self._last_tick = 0.0
        self._frame_count = 0

    def __repr__(self) -> str:
        return (
            f"FPSTracker(fps={self.fps:.1f}, "
            f"frame_interval_ms={self.frame_interval_ms:.2f}, "
            f"frames={self.frame_count})"
        )
