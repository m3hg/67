"""
app.py — Main control panel window for 2k Vision.

Hosts all tabs and coordinates with the Engine.
Hotkeys: F6 = ARM, F7 = DISARM, F8 = toggle overlay.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig, save_config
from ..engine import Engine, EngineStats
from ..overlay import DrawData, OverlayWidget
from ..pipeline import FrameResult
from ..predictor import FireDecision
from .tab_run import RunTab
from .tab_setup import SetupTab
from .tab_trainer import TrainerTab
from .tab_tune import TuneTab
from .theme import STYLESHEET

log = logging.getLogger(__name__)


class ControlPanel(QMainWindow):
    """
    Main control panel window.

    Spawns the Engine in a background thread and updates the UI
    from a 100ms QTimer.
    """

    # Signal to push draw data to overlay from engine thread
    _overlay_signal = pyqtSignal(object)
    # Signal to log shot result from engine thread
    _shot_log_signal = pyqtSignal(str)

    def __init__(self, cfg: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg

        self.setWindowTitle("2k Vision V2.0 — Control Panel")
        self.resize(560, 680)
        self.setMinimumWidth(480)
        self.setStyleSheet(STYLESHEET)

        self._engine: Optional[Engine] = None
        self._overlay: Optional[OverlayWidget] = None

        self._build_ui()
        self._setup_overlay()
        self._setup_engine()
        self._setup_hotkeys()

        # Refresh timer (100ms)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh_ui)
        self._timer.start()

        self._overlay_signal.connect(self._push_to_overlay)
        self._shot_log_signal.connect(self._run_tab.append_log)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        self._setup_tab = SetupTab()
        self._tune_tab = TuneTab(self._cfg, on_config_changed=self._on_config_changed)
        self._run_tab = RunTab(on_arm=self._arm, on_disarm=self._disarm)
        self._trainer_tab = TrainerTab(self._cfg)

        self._tabs.addTab(self._setup_tab, "Setup")
        self._tabs.addTab(self._tune_tab, "Tune")
        self._tabs.addTab(self._run_tab, "Run")
        self._tabs.addTab(self._trainer_tab, "Trainer")

    # ------------------------------------------------------------------
    # Engine / Overlay setup
    # ------------------------------------------------------------------

    def _setup_overlay(self) -> None:
        try:
            self._overlay = OverlayWidget(self._cfg)
        except Exception as exc:
            log.warning("Could not create overlay: %s", exc)
            self._overlay = None

    def _setup_engine(self) -> None:
        try:
            self._engine = Engine(
                cfg=self._cfg,
                on_frame=self._on_engine_frame,
            )
            self._engine.start()
        except Exception as exc:
            log.error("Could not start engine: %s", exc)
            self._engine = None

    def _setup_hotkeys(self) -> None:
        try:
            import keyboard
            keyboard.add_hotkey("F6", self._arm)
            keyboard.add_hotkey("F7", self._disarm)
            keyboard.add_hotkey("F8", self._toggle_overlay)
        except Exception as exc:
            log.warning("Hotkeys not available: %s", exc)

    # ------------------------------------------------------------------
    # Engine callback (runs in engine thread)
    # ------------------------------------------------------------------

    def _on_engine_frame(self, result: FrameResult, decision: FireDecision) -> None:
        """Called from engine thread; push data to overlay via signal."""
        if self._overlay is None:
            return

        w, h = 1920, 1080
        if result.frame is not None:
            h, w = result.frame.shape[:2]

        stats = self._engine.stats if self._engine else None
        data = DrawData(
            meter=result.meter,
            ball=result.ball,
            players=result.players,
            green=result.green,
            decision=decision,
            fps=result.fps,
            fire_pct=self._cfg.predictor.fire_pct,
            show_fire=decision.should_fire,
            frame_w=w,
            frame_h=h,
            shots_total=stats.shots_total if stats else 0,
            shots_green=stats.shots_green if stats else 0,
        )
        self._overlay_signal.emit(data)

        if decision.should_fire:
            self._shot_log_signal.emit(
                f"FIRE — fill={decision.fill_pct * 100:.1f}% "
                f"zone={decision.zone.value} conf={decision.confidence:.2f}"
            )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _push_to_overlay(self, data: DrawData) -> None:
        if self._overlay:
            self._overlay.push_frame(data)

    def _refresh_ui(self) -> None:
        if self._engine:
            self._run_tab.update_stats(self._engine.stats)

    def _on_config_changed(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        if self._engine:
            self._engine.update_config(cfg)
        if self._overlay:
            self._overlay.update_config(cfg)
        save_config(cfg)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _arm(self) -> None:
        if self._engine:
            self._engine.arm()

    def _disarm(self) -> None:
        if self._engine:
            self._engine.disarm()

    def _toggle_overlay(self) -> None:
        if self._overlay:
            self._overlay.toggle_visibility()

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        if self._engine:
            self._engine.stop()
        save_config(self._cfg)
        super().closeEvent(event)
