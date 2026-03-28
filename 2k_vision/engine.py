"""
engine.py — Main engine thread with state machine for 2k Vision.

State machine:
    IDLE → SHOOTING → FIRED → SCANNING → COOLDOWN → IDLE

The engine runs in a daemon thread, capturing frames, running the pipeline,
and coordinating with the predictor, controller, and calibrator.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .calibrator import Calibrator
from .capture import Capture
from .config import AppConfig
from .controller import ControllerProxy
from .feedback import FeedbackResult
from .pipeline import FrameResult, Pipeline
from .predictor import FireDecision, Predictor
from .utils import now_ms

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine states
# ---------------------------------------------------------------------------


class EngineState(enum.Enum):
    IDLE = "IDLE"
    SHOOTING = "SHOOTING"
    FIRED = "FIRED"
    SCANNING = "SCANNING"
    COOLDOWN = "COOLDOWN"


# ---------------------------------------------------------------------------
# Stats snapshot
# ---------------------------------------------------------------------------


@dataclass
class EngineStats:
    state: EngineState = EngineState.IDLE
    fps: float = 0.0
    shots_total: int = 0
    shots_green: int = 0
    last_decision: FireDecision = field(default_factory=FireDecision)
    last_feedback: FeedbackResult = field(default_factory=FeedbackResult)
    armed: bool = False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Engine:
    """
    Orchestrates capture → pipeline → predictor → controller in a thread.

    Parameters
    ----------
    cfg:
        Application config.
    on_frame:
        Callback invoked each frame with the FrameResult and current
        FireDecision; used by the overlay to refresh the display.
    """

    def __init__(
        self,
        cfg: AppConfig,
        on_frame: Optional[Callable[[FrameResult, FireDecision], None]] = None,
    ) -> None:
        self._cfg = cfg
        self._on_frame = on_frame

        self._capture = Capture(
            backend=cfg.capture.backend,
            monitor_index=cfg.capture.monitor_index,
            crop=(
                cfg.capture.crop_x,
                cfg.capture.crop_y,
                cfg.capture.crop_w,
                cfg.capture.crop_h,
            ) if cfg.capture.crop_w > 0 else None,
        )
        self._pipeline = Pipeline(cfg)
        self._predictor = Predictor(cfg.predictor)
        self._calibrator = Calibrator(cfg.predictor)

        self._controller: Optional[ControllerProxy] = None
        if cfg.controller.enabled:
            self._controller = ControllerProxy(
                on_x_press=self._on_x_press,
                on_x_release=self._on_x_release,
                remap_lb_to_x=cfg.controller.remap_lb_to_x,
            )

        self._state = EngineState.IDLE
        self._armed = False
        self._stats = EngineStats()

        self._state_entered_ms: float = 0.0
        self._shot_history: List[dict] = []

        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def arm(self) -> None:
        """Enable auto-fire."""
        self._armed = True
        log.info("Engine ARMED.")

    def disarm(self) -> None:
        """Disable auto-fire (overlay still runs)."""
        self._armed = False
        log.info("Engine DISARMED.")

    def start(self) -> None:
        """Start the engine thread and controller polling."""
        self._running = True
        if self._controller is not None:
            self._controller.start()
        self._thread.start()
        log.info("Engine started.")

    def stop(self) -> None:
        """Stop all threads and release resources."""
        self._running = False
        if self._controller is not None:
            self._controller.stop()
        self._capture.close()
        log.info("Engine stopped.")

    def update_config(self, cfg: AppConfig) -> None:
        """Hot-swap configuration."""
        self._cfg = cfg
        self._pipeline.update_config(cfg)
        self._predictor.update_config(cfg.predictor)

    @property
    def stats(self) -> EngineStats:
        with self._lock:
            return self._stats

    @property
    def shot_history(self) -> list:
        with self._lock:
            return list(self._shot_history)

    # ------------------------------------------------------------------
    # Controller callbacks
    # ------------------------------------------------------------------

    def _on_x_press(self) -> None:
        """Called by ControllerProxy when X is pressed (shot charge begins)."""
        if self._armed and self._state == EngineState.IDLE:
            self._enter_state(EngineState.SHOOTING)

    def _on_x_release(self) -> None:
        """Called by ControllerProxy when X is released (manual release)."""
        if self._state == EngineState.SHOOTING:
            # Manual release — still log it
            self._enter_state(EngineState.FIRED)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main capture + detection loop."""
        while self._running:
            frame = self._capture.grab()
            if frame is None:
                time.sleep(0.005)
                continue
            self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> None:
        """Process a single captured frame through the state machine."""
        shot_active = self._state == EngineState.SHOOTING

        result = self._pipeline.process(frame, shot_active=shot_active)
        decision = FireDecision()

        with self._lock:
            self._stats.fps = result.fps
            self._stats.state = self._state
            self._stats.armed = self._armed

        if self._state == EngineState.SHOOTING:
            decision = self._predictor.update(
                fill_pct=result.meter.fill_pct,
                fps=result.fps,
            )
            if decision.should_fire and self._armed:
                self._fire(result)

        elif self._state == EngineState.SCANNING:
            fb = self._pipeline.scan_feedback(frame)
            if fb.found:
                self._on_feedback(fb)
                self._enter_state(EngineState.COOLDOWN)
            elif self._state_elapsed_ms() > self._cfg.engine.scan_duration_ms:
                self._enter_state(EngineState.COOLDOWN)

        elif self._state == EngineState.COOLDOWN:
            if self._state_elapsed_ms() > self._cfg.engine.cooldown_ms:
                self._enter_state(EngineState.IDLE)

        elif self._state == EngineState.FIRED:
            # Brief pause then start scanning
            if self._state_elapsed_ms() > 100:
                self._enter_state(EngineState.SCANNING)

        # Notify overlay / UI
        if self._on_frame is not None:
            self._on_frame(result, decision)

        with self._lock:
            self._stats.last_decision = decision

    def _fire(self, result: FrameResult) -> None:
        """Execute the shot and record it."""
        if self._controller is not None:
            self._controller.fire_shot()

        self._predictor.reset()
        self._enter_state(EngineState.FIRED)

        shot_entry = {
            "timestamp": now_ms(),
            "fill_pct": result.meter.fill_pct,
            "fps": result.fps,
        }
        with self._lock:
            self._stats.shots_total += 1
            self._shot_history.append(shot_entry)
            if len(self._shot_history) > 100:
                self._shot_history = self._shot_history[-100:]

        log.debug("FIRE — fill=%.2f fps=%.1f", result.meter.fill_pct, result.fps)

    def _on_feedback(self, fb: FeedbackResult) -> None:
        """Process post-shot feedback."""
        self._calibrator.on_shot_feedback(fb)
        from .feedback import ReleaseQuality
        with self._lock:
            self._stats.last_feedback = fb
            if fb.quality == ReleaseQuality.EXCELLENT:
                self._stats.shots_green += 1

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _enter_state(self, state: EngineState) -> None:
        log.debug("Engine state: %s → %s", self._state.value, state.value)
        self._state = state
        self._state_entered_ms = now_ms()

    def _state_elapsed_ms(self) -> float:
        return now_ms() - self._state_entered_ms
