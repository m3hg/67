"""
export.py — Export trained YOLOv8 weights to ONNX for fast inference.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def export_to_onnx(
    weights_path: str,
    opset: int = 11,
    dynamic: bool = False,
    simplify: bool = True,
) -> str:
    """
    Export a ``best.pt`` YOLO model to ONNX format.

    Parameters
    ----------
    weights_path:
        Path to the trained ``.pt`` weights file.
    opset:
        ONNX opset version (default 11).
    dynamic:
        Whether to use dynamic input axes.
    simplify:
        Apply ONNX simplification (requires ``onnxsim``).

    Returns
    -------
    str
        Path to the exported ``.onnx`` file.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = YOLO(weights_path)
    log.info("Exporting %s to ONNX (opset=%d)...", weights_path, opset)

    model.export(format="onnx", opset=opset, dynamic=dynamic, simplify=simplify)

    onnx_path = weights_path.replace(".pt", ".onnx")
    if not os.path.exists(onnx_path):
        # ultralytics may place it in the same directory under a different name
        base_dir = os.path.dirname(weights_path)
        import glob
        candidates = glob.glob(os.path.join(base_dir, "*.onnx"))
        onnx_path = candidates[0] if candidates else onnx_path

    log.info("ONNX model saved to: %s", onnx_path)
    return onnx_path
