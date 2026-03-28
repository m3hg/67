"""
train.py — YOLO training script using ultralytics.

Trains a YOLOv8 model on the dataset created by dataset_builder.py.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

log = logging.getLogger(__name__)


def train(
    dataset_yaml: str,
    model_variant: str = "yolov8n",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    project_dir: Optional[str] = None,
    on_epoch: Optional[Callable[[int, dict], None]] = None,
) -> str:
    """
    Train a YOLOv8 model on the given dataset.

    Parameters
    ----------
    dataset_yaml:
        Path to the dataset.yaml file produced by DatasetBuilder.
    model_variant:
        YOLOv8 variant name, e.g. ``"yolov8n"``, ``"yolov8s"``.
    epochs:
        Number of training epochs.
    batch_size:
        Training batch size.
    img_size:
        Input image size in pixels (square).
    project_dir:
        Directory to save training results.  Defaults to ``~/.2k_vision/runs``.
    on_epoch:
        Optional callback invoked after each epoch with
        ``(epoch_num: int, metrics: dict)``.

    Returns
    -------
    str
        Path to the best weights file ``best.pt``.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    if project_dir is None:
        project_dir = os.path.join(os.path.expanduser("~"), "2k_vision_data", "runs")

    os.makedirs(project_dir, exist_ok=True)

    model = YOLO(f"{model_variant}.pt")
    log.info(
        "Starting YOLO training: model=%s, epochs=%d, batch=%d, img=%d",
        model_variant, epochs, batch_size, img_size,
    )

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project_dir,
        name="2k_vision",
        verbose=False,
    )

    # Find best weights
    best_pt = os.path.join(project_dir, "2k_vision", "weights", "best.pt")
    if not os.path.exists(best_pt):
        # ultralytics may create a numbered sub-dir
        import glob
        candidates = glob.glob(os.path.join(project_dir, "**", "best.pt"), recursive=True)
        best_pt = candidates[0] if candidates else "best.pt"

    log.info("Training complete. Best weights: %s", best_pt)
    return best_pt
