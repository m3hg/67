"""
tab_run.py — ARM/DISARM, live stats, shot log tab for 2k Vision.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..engine import EngineStats
from .theme import ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_ORANGE
from .widgets import HLine, SectionHeader, StatCard


class RunTab(QWidget):
    """
    Run tab: ARM/DISARM button, live stats cards, shot log.
    """

    def __init__(
        self,
        on_arm: Optional[Callable[[], None]] = None,
        on_disarm: Optional[Callable[[], None]] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_arm = on_arm
        self._on_disarm = on_disarm
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # --- ARM / DISARM ---
        btn_row = QHBoxLayout()

        self._arm_btn = QPushButton("▶  ARM  (F6)")
        self._arm_btn.setObjectName("armBtn")
        self._arm_btn.clicked.connect(self._arm)
        btn_row.addWidget(self._arm_btn)

        self._disarm_btn = QPushButton("■  DISARM  (F7)")
        self._disarm_btn.setObjectName("disarmBtn")
        self._disarm_btn.clicked.connect(self._disarm)
        btn_row.addWidget(self._disarm_btn)

        layout.addLayout(btn_row)

        # Status label
        self._status_lbl = QLabel("Status: DISARMED")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet(f"color: {ACCENT_RED}; font-size: 13px; font-weight: bold;")
        layout.addWidget(self._status_lbl)

        layout.addWidget(HLine())

        # --- Stats cards ---
        layout.addWidget(SectionHeader("Live Statistics"))

        grid = QGridLayout()
        grid.setSpacing(8)

        self._fps_card = StatCard("FPS", "0.0", accent=ACCENT_CYAN)
        self._shots_card = StatCard("Shots", "0", accent=ACCENT_GREEN)
        self._greens_card = StatCard("Greens", "0", accent=ACCENT_GREEN)
        self._zone_card = StatCard("Zone", "—", accent=ACCENT_ORANGE)
        self._conf_card = StatCard("Confidence", "0.00", accent=ACCENT_CYAN)
        self._fill_card = StatCard("Fill %", "0.0%", accent="#ff69b4")

        grid.addWidget(self._fps_card, 0, 0)
        grid.addWidget(self._shots_card, 0, 1)
        grid.addWidget(self._greens_card, 0, 2)
        grid.addWidget(self._zone_card, 1, 0)
        grid.addWidget(self._conf_card, 1, 1)
        grid.addWidget(self._fill_card, 1, 2)

        layout.addLayout(grid)
        layout.addWidget(HLine())

        # --- Shot log ---
        layout.addWidget(SectionHeader("Shot Log"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(200)
        self._log.setStyleSheet(
            "QTextEdit { background: #0d0d1a; color: #88ddaa; font-family: Consolas; font-size: 10px; }"
        )
        layout.addWidget(self._log)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_stats(self, stats: EngineStats) -> None:
        """Called by the main app each tick to refresh stats."""
        self._fps_card.set_value(f"{stats.fps:.1f}")
        self._shots_card.set_value(str(stats.shots_total))
        self._greens_card.set_value(str(stats.shots_green))
        self._zone_card.set_value(stats.last_decision.zone.value)
        self._conf_card.set_value(f"{stats.last_decision.confidence:.2f}")
        self._fill_card.set_value(f"{stats.last_decision.fill_pct * 100:.1f}%")

        armed_color = ACCENT_GREEN if stats.armed else ACCENT_RED
        status_text = "ARMED" if stats.armed else "DISARMED"
        self._status_lbl.setStyleSheet(
            f"color: {armed_color}; font-size: 13px; font-weight: bold;"
        )
        self._status_lbl.setText(f"Status: {status_text}  |  State: {stats.state.value}")

    def append_log(self, line: str) -> None:
        """Append a line to the shot log."""
        self._log.append(line)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _arm(self) -> None:
        if self._on_arm:
            self._on_arm()

    def _disarm(self) -> None:
        if self._on_disarm:
            self._on_disarm()
