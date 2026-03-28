"""
tab_trainer.py — Model training UI tab for 2k Vision.

Allows the user to start/stop dataset capture, launch YOLO training,
and monitor training progress.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from .theme import ACCENT_CYAN, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED
from .widgets import HLine, LabeledRow, SectionHeader

log = logging.getLogger(__name__)


class TrainerTab(QWidget):
    """
    Dataset capture + model training tab.
    """

    # Signals emitted from background threads
    _log_signal = pyqtSignal(str)
    _progress_signal = pyqtSignal(int)
    _status_signal = pyqtSignal(str)

    def __init__(
        self,
        cfg: AppConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._builder = None
        self._build_ui()
        self._log_signal.connect(self._append_log)
        self._progress_signal.connect(self._progress.setValue)
        self._status_signal.connect(self._status_lbl.setText)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # --- Dataset capture ---
        layout.addWidget(SectionHeader("Dataset Capture"))

        self._sample_lbl = LabeledRow("Samples captured", "0")
        layout.addWidget(self._sample_lbl)

        save_dir_lbl = LabeledRow("Save directory", self._cfg.trainer.save_dir)
        layout.addWidget(save_dir_lbl)

        cap_row = QHBoxLayout()
        self._start_cap_btn = QPushButton("▶  Start Capture")
        self._start_cap_btn.clicked.connect(self._start_capture)
        cap_row.addWidget(self._start_cap_btn)

        self._stop_cap_btn = QPushButton("■  Stop Capture")
        self._stop_cap_btn.setEnabled(False)
        self._stop_cap_btn.clicked.connect(self._stop_capture)
        cap_row.addWidget(self._stop_cap_btn)
        layout.addLayout(cap_row)

        layout.addWidget(HLine())

        # --- Training ---
        layout.addWidget(SectionHeader("Model Training"))

        train_row = QHBoxLayout()
        self._train_btn = QPushButton("🚀  Train YOLOv8")
        self._train_btn.clicked.connect(self._start_training)
        train_row.addWidget(self._train_btn)
        layout.addLayout(train_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(f"color: {ACCENT_CYAN};")
        layout.addWidget(self._status_lbl)

        layout.addWidget(HLine())

        # --- Log ---
        layout.addWidget(SectionHeader("Log"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            "QTextEdit { background: #0d0d1a; color: #88ddaa; font-family: Consolas; font-size: 10px; }"
        )
        layout.addWidget(self._log)

    # ------------------------------------------------------------------
    # Dataset capture handlers
    # ------------------------------------------------------------------

    def _start_capture(self) -> None:
        try:
            from ..model_trainer.dataset_builder import DatasetBuilder
            self._builder = DatasetBuilder(
                cfg=self._cfg,
                on_sample_saved=self._on_sample_saved,
            )
            self._builder.start()
            self._start_cap_btn.setEnabled(False)
            self._stop_cap_btn.setEnabled(True)
            self._log_signal.emit("Dataset capture started.")
        except Exception as exc:
            self._log_signal.emit(f"Error: {exc}")

    def _stop_capture(self) -> None:
        if self._builder:
            self._builder.stop()
            self._log_signal.emit(
                f"Capture stopped. {self._builder.sample_count} samples saved."
            )
            self._builder = None
        self._start_cap_btn.setEnabled(True)
        self._stop_cap_btn.setEnabled(False)

    def _on_sample_saved(self, count: int) -> None:
        self._sample_lbl.set_value(str(count))

    # ------------------------------------------------------------------
    # Training handlers
    # ------------------------------------------------------------------

    def _start_training(self) -> None:
        dataset_yaml = os.path.join(self._cfg.trainer.save_dir, "dataset.yaml")
        if not os.path.exists(dataset_yaml):
            self._log_signal.emit(f"dataset.yaml not found at {dataset_yaml}. Run capture first.")
            return

        self._train_btn.setEnabled(False)
        self._status_signal.emit("Training started…")
        self._progress_signal.emit(0)

        def _run():
            try:
                from ..model_trainer.train import train
                tcfg = self._cfg.trainer
                self._log_signal.emit(
                    f"Training {tcfg.model_variant} for {tcfg.epochs} epochs…"
                )
                best_pt = train(
                    dataset_yaml=dataset_yaml,
                    model_variant=tcfg.model_variant,
                    epochs=tcfg.epochs,
                    batch_size=tcfg.batch_size,
                    img_size=tcfg.img_size,
                )
                self._log_signal.emit(f"Training complete! Best weights: {best_pt}")
                self._status_signal.emit(f"Done — {best_pt}")
                self._progress_signal.emit(100)
            except Exception as exc:
                self._log_signal.emit(f"Training error: {exc}")
                self._status_signal.emit("Error — see log")
            finally:
                self._train_btn.setEnabled(True)

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append_log(self, line: str) -> None:
        self._log.append(line)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
