"""
tab_setup.py — Setup/info tab for 2k Vision control panel.

Shows system information, dependency status, and feature checklist.
"""

from __future__ import annotations

import importlib
import platform
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .theme import ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, TEXT_SECONDARY
from .widgets import HLine, LabeledRow, SectionHeader


def _check_import(name: str) -> str:
    """Return '✓' in green or '✗' in red based on import availability."""
    try:
        importlib.import_module(name)
        return f'<span style="color:{ACCENT_GREEN}">✓</span>'
    except ImportError:
        return f'<span style="color:{ACCENT_RED}">✗</span>'


class SetupTab(QWidget):
    """System info, dependency status, feature list."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # --- System Info ---
        layout.addWidget(SectionHeader("System Information"))
        layout.addWidget(LabeledRow("OS", platform.system() + " " + platform.release()))
        layout.addWidget(LabeledRow("Python", sys.version.split()[0]))
        layout.addWidget(LabeledRow("Architecture", platform.machine()))
        layout.addWidget(HLine())

        # --- Dependencies ---
        layout.addWidget(SectionHeader("Dependency Status"))
        deps = [
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("PyQt6", "PyQt6"),
            ("mss", "mss (capture)"),
            ("dxcam", "dxcam (capture, optional)"),
            ("vgamepad", "vgamepad (ViGEm)"),
            ("keyboard", "keyboard (hotkeys)"),
            ("ultralytics", "ultralytics (YOLO)"),
            ("onnxruntime", "onnxruntime"),
            ("filterpy", "filterpy"),
        ]
        for mod, label in deps:
            status = _check_import(mod)
            row = QLabel(f"  {status} &nbsp; {label}")
            row.setTextFormat(Qt.TextFormat.RichText)
            row.setStyleSheet(f"color: white; background: transparent;")
            layout.addWidget(row)

        layout.addWidget(HLine())

        # --- Features ---
        layout.addWidget(SectionHeader("Features"))
        features = [
            "Real-time transparent click-through overlay",
            "Shot meter HSV detection (magenta/configurable)",
            "Basketball detection + Kalman tracker",
            "Parabolic trajectory prediction",
            "Player bounding box detection (MOG2)",
            "Green release flash detection",
            "EMA v9 predictor with 4 fire zones",
            "Adaptive fire_pct calibration",
            "XInput controller proxy via ViGEm",
            "YOLO model training pipeline",
            "ONNX inference (GPU/CPU)",
            "Dark-theme control panel",
            "F6=ARM, F7=DISARM, F8=toggle overlay",
        ]
        for feat in features:
            lbl = QLabel(f"  <span style='color:{ACCENT_CYAN}'>▸</span> {feat}")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setStyleSheet("background: transparent;")
            layout.addWidget(lbl)

        layout.addStretch()
