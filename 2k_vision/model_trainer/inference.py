"""
inference.py — Load and run an ONNX YOLO model for player/ball detection.

Provides a drop-in alternative to the heuristic detectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

CLASS_NAMES = ["ball", "player", "shot_meter", "basket"]

# (x1, y1, x2, y2, confidence, class_id)
Detection = Tuple[float, float, float, float, float, int]


@dataclass
class OnnxDetection:
    """Single object detection from ONNX model inference."""

    class_id: int = 0
    class_name: str = ""
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # x1, y1, x2, y2
    confidence: float = 0.0

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) integer rect."""
        x1, y1, x2, y2 = self.bbox
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


class OnnxInference:
    """
    ONNX model runner for YOLOv8 detection.

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` model file.
    conf_threshold:
        Minimum confidence for detections.
    iou_threshold:
        NMS IoU threshold.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> None:
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._session = None
        self._input_name: str = ""
        self._input_shape: Tuple[int, int] = (640, 640)
        self._load(model_path)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def infer(self, frame_bgr: np.ndarray) -> List[OnnxDetection]:
        """
        Run inference on *frame_bgr*.

        Parameters
        ----------
        frame_bgr:
            BGR uint8 numpy array.

        Returns
        -------
        list[OnnxDetection]
        """
        if self._session is None:
            return []

        orig_h, orig_w = frame_bgr.shape[:2]
        blob, scale_x, scale_y = self._preprocess(frame_bgr)

        outputs = self._session.run(None, {self._input_name: blob})
        detections = self._postprocess(outputs, scale_x, scale_y, orig_w, orig_h)
        return detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: str) -> None:
        try:
            import onnxruntime as ort  # type: ignore

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(path, providers=providers)
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            _, _, h, w = inp.shape
            self._input_shape = (int(h), int(w))
            log.info("ONNX model loaded: %s  input=%s", path, inp.shape)
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Failed to load ONNX model: %s", exc)
            self._session = None

    def _preprocess(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Resize + normalise frame to model input size."""
        import cv2

        ih, iw = self._input_shape
        orig_h, orig_w = frame_bgr.shape[:2]
        scale_x = orig_w / iw
        scale_y = orig_h / ih

        resized = cv2.resize(frame_bgr, (iw, ih))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis]  # NCHW
        return blob, scale_x, scale_y

    def _postprocess(
        self,
        outputs: list,
        scale_x: float,
        scale_y: float,
        orig_w: int,
        orig_h: int,
    ) -> List[OnnxDetection]:
        """Parse YOLOv8 output tensor and apply NMS."""
        # YOLOv8 output shape: (1, 4+nc, num_anchors) or (1, num_anchors, 4+nc)
        preds = outputs[0]
        if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
            # (1, 4+nc, num_anchors) → (1, num_anchors, 4+nc)
            preds = np.transpose(preds, (0, 2, 1))

        preds = preds[0]  # (num_anchors, 4+nc)
        nc = preds.shape[1] - 4

        boxes_xywh = preds[:, :4]
        scores = preds[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences >= self._conf
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes_xywh) == 0:
            return []

        # Convert xywh → xyxy (model outputs are in input space)
        ih, iw = self._input_shape
        x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2) * scale_x
        y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2) * scale_y
        x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2) * scale_x
        y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2) * scale_y

        import cv2

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xyxy.tolist(),
            scores=confidences.tolist(),
            score_threshold=self._conf,
            nms_threshold=self._iou,
        )

        results = []
        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            for i in indices:
                cid = int(class_ids[i])
                results.append(
                    OnnxDetection(
                        class_id=cid,
                        class_name=CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid),
                        bbox=(
                            float(x1[i]), float(y1[i]),
                            float(x2[i]), float(y2[i]),
                        ),
                        confidence=float(confidences[i]),
                    )
                )
        return results
