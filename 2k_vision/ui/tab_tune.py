"""
tab_tune.py — Color tuning + detection parameters tab for 2k Vision.

Allows live adjustment of all detector parameters with instant hot-swap.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig, save_config
from .widgets import HLine, SectionHeader


class TuneTab(QWidget):
    """
    Color tuning and detection parameter editor.

    All changes are applied live to the running config and persisted to disk.
    """

    def __init__(
        self,
        cfg: AppConfig,
        on_config_changed: Optional[Callable[[AppConfig], None]] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._on_changed = on_config_changed
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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

        # --- Meter color ---
        layout.addWidget(SectionHeader("Shot Meter Color (HSV)"))
        layout.addWidget(self._build_meter_color_group())

        # --- Ball color ---
        layout.addWidget(SectionHeader("Basketball Color (HSV)"))
        layout.addWidget(self._build_ball_color_group())

        # --- Predictor ---
        layout.addWidget(SectionHeader("Predictor / Fire Zones"))
        layout.addWidget(self._build_predictor_group())

        # --- Overlay toggles ---
        layout.addWidget(SectionHeader("Overlay Elements"))
        layout.addWidget(self._build_overlay_toggles())

        # --- Save button ---
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_meter_color_group(self) -> QGroupBox:
        mc = self._cfg.meter
        grp = QGroupBox("Meter HSV")
        form = QFormLayout(grp)

        self._m_h_lo = self._make_spin(0, 179, mc.hue_low, lambda v: setattr(mc, "hue_low", v))
        self._m_h_hi = self._make_spin(0, 179, mc.hue_high, lambda v: setattr(mc, "hue_high", v))
        self._m_s_lo = self._make_spin(0, 255, mc.sat_low, lambda v: setattr(mc, "sat_low", v))
        self._m_s_hi = self._make_spin(0, 255, mc.sat_high, lambda v: setattr(mc, "sat_high", v))
        self._m_v_lo = self._make_spin(0, 255, mc.val_low, lambda v: setattr(mc, "val_low", v))
        self._m_v_hi = self._make_spin(0, 255, mc.val_high, lambda v: setattr(mc, "val_high", v))

        form.addRow("Hue Low", self._m_h_lo)
        form.addRow("Hue High", self._m_h_hi)
        form.addRow("Sat Low", self._m_s_lo)
        form.addRow("Sat High", self._m_s_hi)
        form.addRow("Val Low", self._m_v_lo)
        form.addRow("Val High", self._m_v_hi)
        return grp

    def _build_ball_color_group(self) -> QGroupBox:
        bc = self._cfg.ball
        grp = QGroupBox("Ball HSV")
        form = QFormLayout(grp)

        self._b_h_lo = self._make_spin(0, 179, bc.hue_low, lambda v: setattr(bc, "hue_low", v))
        self._b_h_hi = self._make_spin(0, 179, bc.hue_high, lambda v: setattr(bc, "hue_high", v))
        self._b_s_lo = self._make_spin(0, 255, bc.sat_low, lambda v: setattr(bc, "sat_low", v))
        self._b_s_hi = self._make_spin(0, 255, bc.sat_high, lambda v: setattr(bc, "sat_high", v))
        self._b_v_lo = self._make_spin(0, 255, bc.val_low, lambda v: setattr(bc, "val_low", v))
        self._b_v_hi = self._make_spin(0, 255, bc.val_high, lambda v: setattr(bc, "val_high", v))

        form.addRow("Hue Low", self._b_h_lo)
        form.addRow("Hue High", self._b_h_hi)
        form.addRow("Sat Low", self._b_s_lo)
        form.addRow("Sat High", self._b_s_hi)
        form.addRow("Val Low", self._b_v_lo)
        form.addRow("Val High", self._b_v_hi)
        return grp

    def _build_predictor_group(self) -> QGroupBox:
        pc = self._cfg.predictor
        grp = QGroupBox("Predictor")
        form = QFormLayout(grp)

        self._p_fire = self._make_dspin(0.50, 0.99, pc.fire_pct, 0.01,
                                        lambda v: setattr(pc, "fire_pct", v))
        self._p_alpha = self._make_dspin(0.01, 1.0, pc.ema_alpha, 0.01,
                                         lambda v: setattr(pc, "ema_alpha", v))
        self._p_z2c = self._make_dspin(0.0, 1.0, pc.z2_conf, 0.05,
                                        lambda v: setattr(pc, "z2_conf", v))
        self._p_z2f = self._make_dspin(0.0, 1.0, pc.z2_fill, 0.01,
                                        lambda v: setattr(pc, "z2_fill", v))

        form.addRow("Fire PCT", self._p_fire)
        form.addRow("EMA Alpha", self._p_alpha)
        form.addRow("Z2 Conf", self._p_z2c)
        form.addRow("Z2 Fill", self._p_z2f)
        return grp

    def _build_overlay_toggles(self) -> QGroupBox:
        oc = self._cfg.overlay
        grp = QGroupBox("Show Overlay Elements")
        layout = QVBoxLayout(grp)

        toggles = [
            ("Scan lines", "show_scan_lines"),
            ("Meter box", "show_meter_box"),
            ("Meter fill", "show_meter_fill"),
            ("Fire threshold line", "show_fire_line"),
            ("Player boxes", "show_player_boxes"),
            ("Ball trail", "show_ball_trail"),
            ("Predicted arc", "show_predicted_arc"),
            ("Stats text", "show_stats_text"),
            ("Timing HUD", "show_timing_hud"),
            ("Green flash", "show_green_flash"),
            ("FIRE indicator", "show_fire_indicator"),
            ("Watermark", "show_watermark"),
        ]
        for label, attr in toggles:
            cb = QCheckBox(label)
            cb.setChecked(getattr(oc, attr))
            cb.toggled.connect(lambda v, a=attr: setattr(oc, a, v))
            layout.addWidget(cb)
        return grp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_spin(
        self,
        lo: int,
        hi: int,
        value: int,
        callback: Callable,
    ) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(value)
        sb.valueChanged.connect(lambda v: (callback(v), self._notify()))
        return sb

    def _make_dspin(
        self,
        lo: float,
        hi: float,
        value: float,
        step: float,
        callback: Callable,
    ) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setDecimals(3)
        sb.setValue(value)
        sb.valueChanged.connect(lambda v: (callback(v), self._notify()))
        return sb

    def _notify(self) -> None:
        if self._on_changed:
            self._on_changed(self._cfg)

    def _save(self) -> None:
        save_config(self._cfg)
