"""
widgets.py — Reusable card/button/stat widgets for 2k Vision UI.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .theme import ACCENT_CYAN, BG_CARD, BORDER_COLOR, TEXT_SECONDARY


class StatCard(QFrame):
    """
    A small stat display card with a title and a value label.

    Example::

        card = StatCard("FPS", "0.0")
        card.set_value("58.3")
    """

    def __init__(
        self,
        title: str,
        initial_value: str = "—",
        unit: str = "",
        accent: str = ACCENT_CYAN,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._unit = unit
        self._accent = accent

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; "
            f"border-radius: 6px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self._title_lbl = QLabel(title)
        self._title_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; background: transparent;")
        layout.addWidget(self._title_lbl)

        self._value_lbl = QLabel(initial_value)
        self._value_lbl.setStyleSheet(
            f"color: {accent}; font-size: 20px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(self._value_lbl)

    def set_value(self, value: str) -> None:
        """Update the displayed value."""
        self._value_lbl.setText(value + self._unit)

    def set_accent(self, color: str) -> None:
        """Change the accent colour dynamically."""
        self._value_lbl.setStyleSheet(
            f"color: {color}; font-size: 20px; font-weight: bold; background: transparent;"
        )


class SectionHeader(QLabel):
    """Section header label with cyan underline."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("sectionHeader")


class HLine(QFrame):
    """Horizontal separator line."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setStyleSheet(f"QFrame {{ background: {BORDER_COLOR}; max-height: 1px; }}")


class LabeledRow(QWidget):
    """A label + value row for settings display."""

    def __init__(
        self,
        label: str,
        value: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        lbl = QLabel(label + ":")
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; min-width: 120px;")
        layout.addWidget(lbl)

        self._val = QLabel(value)
        self._val.setStyleSheet(f"color: white; background: transparent;")
        layout.addWidget(self._val)
        layout.addStretch()

    def set_value(self, value: str) -> None:
        self._val.setText(value)
