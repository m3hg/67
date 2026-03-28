"""
dataset_builder.py — Capture screenshots and auto-label using heuristic detectors.

Saves images and YOLO-format label files (.txt) ready for training.

Classes:
  0 = ball
  1 = player
  2 = shot_meter
  3 = basket
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from ..capture import Capture
from ..config import AppConfig, TrainerConfig
from ..detector_ball import BallDetector
from ..detector_meter import MeterDetector
from ..detector_player import PlayerDetector

log = logging.getLogger(__name__)

CLASS_BALL = 0
CLASS_PLAYER = 1
CLASS_METER = 2
CLASS_BASKET = 3


# ---------------------------------------------------------------------------
# YOLO label helper
# ---------------------------------------------------------------------------


def _to_yolo(rect: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> str:
    """Convert (x, y, w, h) pixel rect to YOLO normalised format."""
    x, y, w, h = rect
    cx = (x + w / 2) / frame_w
    cy = (y + h / 2) / frame_h
    nw = w / frame_w
    nh = h / frame_h
    return f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


class DatasetBuilder:
    """
    Capture screenshots from the game and auto-label them.

    Usage::

        builder = DatasetBuilder(cfg)
        builder.start()
        # ... game is running ...
        builder.stop()
        print(f"Saved {builder.sample_count} samples to {builder.save_dir}")
    """

    def __init__(
        self,
        cfg: AppConfig,
        on_sample_saved: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._cfg = cfg
        self._tcfg: TrainerConfig = cfg.trainer
        self._on_sample = on_sample_saved

        self._cap = Capture(
            backend=cfg.capture.backend,
            monitor_index=cfg.capture.monitor_index,
        )
        self._meter_det = MeterDetector(cfg.meter)
        self._ball_det = BallDetector(cfg.ball)
        self._player_det = PlayerDetector(cfg.player)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._count = 0

        # Ensure directory structure exists
        self._img_dir = os.path.join(self._tcfg.save_dir, "images", "train")
        self._lbl_dir = os.path.join(self._tcfg.save_dir, "labels", "train")
        os.makedirs(self._img_dir, exist_ok=True)
        os.makedirs(self._lbl_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def sample_count(self) -> int:
        return self._count

    @property
    def save_dir(self) -> str:
        return self._tcfg.save_dir

    def start(self) -> None:
        """Start background capture thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("DatasetBuilder started — saving to %s", self._tcfg.save_dir)

    def stop(self) -> None:
        """Stop capture and write YAML dataset descriptor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._write_yaml()
        self._cap.close()
        log.info("DatasetBuilder stopped. %d samples saved.", self._count)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        interval = self._tcfg.capture_interval_ms / 1000.0
        while self._running:
            t0 = time.monotonic()
            frame = self._cap.grab()
            if frame is not None:
                self._process_frame(frame)
            elapsed = time.monotonic() - t0
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _process_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        labels: List[str] = []

        meter = self._meter_det.detect(frame)
        ball = self._ball_det.detect(frame)
        players = self._player_det.detect(frame)

        if meter.found:
            labels.append(f"{CLASS_METER} {_to_yolo(meter.rect, w, h)}")

        if ball.found:
            r = ball.radius
            bx = ball.position[0] - r
            by = ball.position[1] - r
            bw = bh = r * 2
            labels.append(f"{CLASS_BALL} {_to_yolo((bx, by, bw, bh), w, h)}")

        for p in players:
            labels.append(f"{CLASS_PLAYER} {_to_yolo(p.rect, w, h)}")

        if not labels:
            return  # skip frames with no detections

        stem = f"frame_{self._count:06d}"
        cv2.imwrite(os.path.join(self._img_dir, stem + ".jpg"), frame)
        with open(os.path.join(self._lbl_dir, stem + ".txt"), "w") as fh:
            fh.write("\n".join(labels) + "\n")

        self._count += 1
        if self._on_sample:
            self._on_sample(self._count)

    def _write_yaml(self) -> None:
        """Write dataset.yaml for ultralytics."""
        yaml_path = os.path.join(self._tcfg.save_dir, "dataset.yaml")
        classes = self._tcfg.classes
        nc = len(classes)
        names_str = "[" + ", ".join(f'"{c}"' for c in classes) + "]"
        content = (
            f"path: {self._tcfg.save_dir}\n"
            f"train: images/train\n"
            f"val: images/train\n"
            f"nc: {nc}\n"
            f"names: {names_str}\n"
        )
        with open(yaml_path, "w") as fh:
            fh.write(content)
        log.info("Dataset YAML written to %s", yaml_path)
