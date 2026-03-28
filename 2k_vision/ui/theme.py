"""
theme.py — Dark theme colors and Qt stylesheet for 2k Vision UI.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

BG_DARK = "#0d0d1a"
BG_MID = "#12122a"
BG_CARD = "#1a1a35"
ACCENT_CYAN = "#00e5ff"
ACCENT_GREEN = "#00ff88"
ACCENT_ORANGE = "#ff8c00"
ACCENT_RED = "#ff3c3c"
TEXT_PRIMARY = "#e8e8f0"
TEXT_SECONDARY = "#8888aa"
BORDER_COLOR = "#2a2a50"

# ---------------------------------------------------------------------------
# QSS stylesheet
# ---------------------------------------------------------------------------

STYLESHEET = f"""
QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", Consolas, monospace;
    font-size: 11px;
}}

QTabWidget::pane {{
    border: 1px solid {BORDER_COLOR};
    background-color: {BG_MID};
}}

QTabBar::tab {{
    background-color: {BG_CARD};
    color: {TEXT_SECONDARY};
    padding: 6px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background-color: {BG_MID};
    color: {ACCENT_CYAN};
    border-bottom: 2px solid {ACCENT_CYAN};
}}

QPushButton {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 6px 14px;
    min-height: 28px;
}}

QPushButton:hover {{
    background-color: #25254a;
    border-color: {ACCENT_CYAN};
    color: {ACCENT_CYAN};
}}

QPushButton:pressed {{
    background-color: #1a1a40;
}}

QPushButton#armBtn {{
    background-color: #0a3a1a;
    color: {ACCENT_GREEN};
    border-color: {ACCENT_GREEN};
    font-size: 16px;
    font-weight: bold;
    min-height: 48px;
}}

QPushButton#armBtn:hover {{
    background-color: #0f5025;
}}

QPushButton#disarmBtn {{
    background-color: #3a0a0a;
    color: {ACCENT_RED};
    border-color: {ACCENT_RED};
    font-size: 16px;
    font-weight: bold;
    min-height: 48px;
}}

QPushButton#disarmBtn:hover {{
    background-color: #5a1010;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER_COLOR};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT_CYAN};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT_CYAN};
    border-radius: 2px;
}}

QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}

QLabel#sectionHeader {{
    color: {ACCENT_CYAN};
    font-size: 12px;
    font-weight: bold;
    border-bottom: 1px solid {BORDER_COLOR};
    padding-bottom: 4px;
}}

QGroupBox {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 10px;
    color: {TEXT_SECONDARY};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {ACCENT_CYAN};
    font-weight: bold;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_PRIMARY};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT_CYAN};
}}

QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 6px;
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    background: {BG_CARD};
}}

QCheckBox::indicator:checked {{
    background: {ACCENT_CYAN};
    border-color: {ACCENT_CYAN};
}}

QScrollArea {{
    border: none;
}}

QTextEdit {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    color: {TEXT_PRIMARY};
    font-family: Consolas, monospace;
}}

QProgressBar {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRIMARY};
    height: 18px;
}}

QProgressBar::chunk {{
    background-color: {ACCENT_CYAN};
    border-radius: 3px;
}}
"""
