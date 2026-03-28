"""
overlay.py — Real transparent click-through always-on-top overlay for 2k Vision.

Uses PyQt6 to create a frameless, transparent window that sits directly on top
of the NBA 2K game.  All game elements (bounding boxes, trajectory arc, HUD)
are drawn via QPainter in the paintEvent.

On Windows the window is made click-through using Win32 SetWindowLong with
WS_EX_TRANSPARENT so mouse events pass through to the game below.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import time
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
    QBrush,
    QLinearGradient,
    QRadialGradient,
    QPainterPath,
)
from PyQt6.QtWidgets import QApplication, QWidget

from .config import AppConfig, OverlayConfig
from .detector_ball import BallResult
from .detector_green import GreenFlashResult
from .detector_meter import MeterResult
from .detector_player import PlayerResult
from .pipeline import FrameResult
from .predictor import FireDecision, Zone

log = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"

# Win32 window extended styles
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
GWL_EXSTYLE = -20


# ---------------------------------------------------------------------------
# Helper: find and match NBA 2K window position on Windows
# ---------------------------------------------------------------------------


def _find_game_window() -> Optional[Tuple[int, int, int, int]]:
    """
    Return (x, y, w, h) of the NBA 2K26 game window.
    Returns None if not found or not on Windows.
    """
    if not IS_WINDOWS:
        return None
    try:
        import win32gui  # type: ignore

        result: Optional[Tuple[int, int, int, int]] = None

        def _enum_cb(hwnd, lp):
            nonlocal result
            title = win32gui.GetWindowText(hwnd).lower()
            if "nba 2k" in title:
                rect = win32gui.GetWindowRect(hwnd)
                x, y, x2, y2 = rect
                result = (x, y, x2 - x, y2 - y)
                return False  # stop enumeration
            return True

        win32gui.EnumWindows(_enum_cb, None)
        return result
    except Exception:  # pylint: disable=broad-except
        return None


# ---------------------------------------------------------------------------
# Overlay colours
# ---------------------------------------------------------------------------

_CYAN = QColor(0, 255, 255, 220)
_MAGENTA = QColor(255, 0, 220, 200)
_ORANGE = QColor(255, 140, 0, 200)
_GREEN_FLASH = QColor(0, 255, 60, 180)
_TEXT_BG = QColor(0, 0, 0, 160)
_WHITE = QColor(255, 255, 255, 220)
_RED = QColor(255, 60, 60, 220)
_YELLOW = QColor(255, 220, 0, 220)

# Zone-specific HUD background colors
_ZONE_COLOR_Z1_DIRECT = QColor(0, 180, 0, 200)
_ZONE_COLOR_Z2_HI_CONF = QColor(0, 100, 200, 200)
_ZONE_COLOR_Z3_MED_CONF = QColor(0, 100, 200, 200)
_ZONE_COLOR_Z1B_LAST = QColor(255, 140, 0, 200)
_ZONE_COLOR_DEFAULT = QColor(20, 20, 40, 180)


# ---------------------------------------------------------------------------
# Draw-data snapshot (passed from engine thread → overlay via signal)
# ---------------------------------------------------------------------------


class DrawData:
    """Snapshot of all detection results needed for one paint call."""

    __slots__ = (
        "meter",
        "ball",
        "players",
        "green",
        "decision",
        "fps",
        "fire_pct",
        "show_fire",
        "frame_w",
        "frame_h",
        "shots_total",
        "shots_green",
    )

    def __init__(
        self,
        meter: MeterResult,
        ball: BallResult,
        players: List[PlayerResult],
        green: GreenFlashResult,
        decision: FireDecision,
        fps: float,
        fire_pct: float,
        show_fire: bool,
        frame_w: int,
        frame_h: int,
        shots_total: int = 0,
        shots_green: int = 0,
    ) -> None:
        self.meter = meter
        self.ball = ball
        self.players = players
        self.green = green
        self.decision = decision
        self.fps = fps
        self.fire_pct = fire_pct
        self.show_fire = show_fire
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.shots_total = shots_total
        self.shots_green = shots_green


# ---------------------------------------------------------------------------
# Overlay widget
# ---------------------------------------------------------------------------


class OverlayWidget(QWidget):
    """
    Transparent, frameless, always-on-top overlay window.

    Draws all CV detection results on top of the game in real time.
    Toggle visibility with F8.
    """

    # Signal emitted from engine thread to trigger repaint
    frame_ready = pyqtSignal(object)  # DrawData

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(parent=None)
        self._cfg = cfg
        self._draw_data: Optional[DrawData] = None
        self._visible = True
        self._fire_flash_frames = 0

        self._setup_window()
        self.frame_ready.connect(self._on_frame_ready)

        # Timer as fallback repaint at 60fps
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """Configure window flags for transparent, click-through overlay."""
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        # Match game window geometry, fall back to primary screen
        geom = _find_game_window()
        if geom:
            self.setGeometry(*geom)
        else:
            screen = QApplication.primaryScreen()
            if screen:
                sg = screen.geometry()
                self.setGeometry(sg)

        self.setWindowTitle("2k Vision Overlay")

        if IS_WINDOWS:
            self._apply_win32_click_through()

        self.show()

    def _apply_win32_click_through(self) -> None:
        """Use SetWindowLong to make the window fully click-through."""
        try:
            hwnd = int(self.winId())
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ex_style |= WS_EX_TRANSPARENT | WS_EX_LAYERED
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("Could not set click-through: %s", exc)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push_frame(self, data: DrawData) -> None:
        """Thread-safe: push new draw data from engine thread."""
        self.frame_ready.emit(data)

    def toggle_visibility(self) -> None:
        """Toggle overlay visibility (F8 hotkey)."""
        self._visible = not self._visible
        self.update()

    def update_config(self, cfg: AppConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_frame_ready(self, data: DrawData) -> None:
        self._draw_data = data
        if data.decision.should_fire:
            self._fire_flash_frames = 20
        elif self._fire_flash_frames > 0:
            self._fire_flash_frames -= 1
        self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self._visible:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        oc = self._cfg.overlay
        d = self._draw_data
        w, h = self.width(), self.height()

        if d is None:
            self._draw_watermark(painter, w, h)
            painter.end()
            return

        if oc.show_scan_lines:
            self._draw_scan_lines(painter, d, w, h)
        if oc.show_player_boxes:
            self._draw_players(painter, d.players)
        if oc.show_meter_box and d.meter.found:
            self._draw_meter(painter, d)
        if oc.show_ball_trail and d.ball.found:
            self._draw_ball_trail(painter, d.ball)
        if oc.show_predicted_arc:
            self._draw_predicted_arc(painter, d.ball)
        if oc.show_green_flash and d.green.found:
            self._draw_green_flash(painter, d.green)
        if oc.show_timing_hud:
            self._draw_timing_hud(painter, d, w)
        if oc.show_stats_text and d.meter.found:
            self._draw_stats(painter, d)
        if oc.show_fire_indicator and self._fire_flash_frames > 0:
            self._draw_fire_indicator(painter, w, h)
        if oc.show_watermark:
            self._draw_watermark(painter, w, h)

        painter.end()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_scan_lines(self, painter: QPainter, d: DrawData, w: int, h: int) -> None:
        """Draw horizontal cyan scan region lines."""
        pen = QPen(_CYAN, 1, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        y_top = int(h * 0.1)
        y_bot = int(h * 0.9)
        painter.drawLine(0, y_top, w, y_top)
        painter.drawLine(0, y_bot, w, y_bot)
        # Corner accents
        for x in (0, w - 40):
            painter.drawLine(x, y_top, x + 40, y_top)
            painter.drawLine(x, y_bot, x + 40, y_bot)

    def _draw_players(self, painter: QPainter, players: List[PlayerResult]) -> None:
        """Draw cyan bounding boxes around detected players."""
        for p in players:
            x, y, pw, ph = p.rect
            color = QColor(0, 255, 120, 220) if p.is_shooter else _CYAN
            pen = QPen(color, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(x, y, pw, ph)
            # Small label
            if p.is_shooter:
                self._draw_label(painter, x, y - 18, "SHOOTER", color)

    def _draw_meter(self, painter: QPainter, d: DrawData) -> None:
        """Draw shot meter bounding box and magenta fill bar."""
        mx, my, mw, mh = d.meter.rect
        oc = self._cfg.overlay

        # Outer box (cyan)
        if oc.show_meter_box:
            pen = QPen(_CYAN, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(mx, my, mw, mh)

        # Fill bar (magenta gradient from bottom)
        if oc.show_meter_fill:
            fill_h = int(mh * d.meter.fill_pct)
            fill_y = my + mh - fill_h
            grad = QLinearGradient(mx, fill_y + fill_h, mx, fill_y)
            grad.setColorAt(0.0, QColor(255, 0, 180, 200))
            grad.setColorAt(1.0, QColor(255, 120, 255, 200))
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(mx + 1, fill_y, mw - 2, fill_h)

        # Fire threshold line (orange, dashed)
        if oc.show_fire_line:
            thr_y = my + mh - int(mh * d.fire_pct)
            pen = QPen(_ORANGE, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(mx - 8, thr_y, mx + mw + 8, thr_y)

    def _draw_ball_trail(self, painter: QPainter, ball: BallResult) -> None:
        """Draw orange dots showing ball history (fading opacity)."""
        trail = ball.trajectory
        n = len(trail)
        for i, pt in enumerate(trail):
            alpha = int(60 + 180 * (i / max(1, n - 1)))
            color = QColor(200, 100, 20, alpha)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            r = 4 + int(4 * i / max(1, n - 1))
            painter.drawEllipse(pt[0] - r, pt[1] - r, r * 2, r * 2)

    def _draw_predicted_arc(self, painter: QPainter, ball: BallResult) -> None:
        """Draw lighter orange dots for the predicted parabolic arc."""
        arc = ball.predicted_arc
        n = len(arc)
        for i, pt in enumerate(arc):
            alpha = int(160 - 120 * (i / max(1, n - 1)))
            color = QColor(255, 165, 0, alpha)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(pt[0] - 4, pt[1] - 4, 8, 8)

    def _draw_green_flash(self, painter: QPainter, flash: GreenFlashResult) -> None:
        """Draw a radial green burst at the green-release position."""
        cx, cy = flash.position
        alpha = int(220 * (flash.frames_remaining / max(1, 15)))
        rad = 40 + int(20 * flash.intensity)

        gradient = QRadialGradient(QPointF(cx, cy), rad)
        gradient.setColorAt(0.0, QColor(0, 255, 60, alpha))
        gradient.setColorAt(1.0, QColor(0, 255, 60, 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(cx - rad, cy - rad, rad * 2, rad * 2)

    def _draw_timing_hud(self, painter: QPainter, d: DrawData, w: int) -> None:
        """Draw top-left timing HUD: 'Timing: XX.X' on colored background."""
        fill = d.meter.fill_pct * 100.0
        zone = d.decision.zone

        # Background color based on zone (use module-level constants)
        if zone == Zone.Z1_DIRECT:
            bg = _ZONE_COLOR_Z1_DIRECT
        elif zone in (Zone.Z2_HI_CONF, Zone.Z3_MED_CONF):
            bg = _ZONE_COLOR_Z2_HI_CONF
        elif zone == Zone.Z1B_LAST:
            bg = _ZONE_COLOR_Z1B_LAST
        else:
            bg = _ZONE_COLOR_DEFAULT

        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(8, 8, 180, 36, 6, 6)

        painter.setPen(QPen(_WHITE))
        font = QFont("Consolas", 14, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(14, 33, f"Timing: {fill:.1f}")

        # FPS sub-line
        font2 = QFont("Consolas", 9)
        painter.setFont(font2)
        painter.setPen(QPen(QColor(180, 220, 180, 200)))
        painter.drawText(14, 54, f"FPS: {d.fps:.1f}  Zone: {zone.value}")

    def _draw_stats(self, painter: QPainter, d: DrawData) -> None:
        """Draw stats text near the shot meter."""
        mx, my, mw, mh = d.meter.rect
        stats = [
            f"Fill:  {d.meter.fill_pct * 100:.1f}%",
            f"Conf:  {d.decision.confidence:.2f}",
            f"Vel:   {d.decision.velocity * 100:.2f}",
            f"Zone:  {d.decision.zone.value}",
            f"FPS:   {d.fps:.1f}",
        ]
        x = mx + mw + 8
        y = my
        font = QFont("Consolas", 9)
        painter.setFont(font)
        line_h = 16
        for line in stats:
            rect = QRectF(x, y, 160, line_h)
            painter.setBrush(QBrush(_TEXT_BG))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(rect)
            painter.setPen(QPen(_WHITE))
            painter.drawText(x + 3, y + 12, line)
            y += line_h + 2

    def _draw_fire_indicator(self, painter: QPainter, w: int, h: int) -> None:
        """Draw 'FIRE!' indicator when auto-fire triggers."""
        alpha = int(255 * (self._fire_flash_frames / 20))
        painter.setBrush(QBrush(QColor(255, 60, 0, alpha)))
        painter.setPen(Qt.PenStyle.NoPen)
        cx, cy = w // 2, h // 2
        painter.drawRoundedRect(cx - 80, cy - 30, 160, 60, 10, 10)
        painter.setPen(QPen(QColor(255, 255, 255, alpha)))
        font = QFont("Impact", 28, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(cx - 50, cy + 18, "FIRE!")

    def _draw_watermark(self, painter: QPainter, w: int, h: int) -> None:
        """Draw '2k Vision V2.0' watermark bottom-right."""
        font = QFont("Consolas", 9)
        painter.setFont(font)
        painter.setPen(QPen(QColor(100, 200, 200, 120)))
        painter.drawText(w - 130, h - 12, "2k Vision V2.0")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _draw_label(
        self,
        painter: QPainter,
        x: int,
        y: int,
        text: str,
        color: QColor,
    ) -> None:
        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.setBrush(QBrush(_TEXT_BG))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(x, y, len(text) * 7, 14)
        painter.setPen(QPen(color))
        painter.drawText(x + 2, y + 11, text)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_F8:
            self.toggle_visibility()
        super().keyPressEvent(event)
