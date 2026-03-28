"""
capture.py — Screen capture backends for 2k Vision.

Supports two backends:
  - ``mss``   : pure-Python, cross-platform (always available)
  - ``dxcam`` : Windows-only, lower latency (optional)

Usage::

    cap = Capture(backend="mss", monitor_index=1)
    frame = cap.grab()   # returns BGR numpy array or None
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# BGR numpy frame type alias
Frame = np.ndarray


class Capture:
    """
    Unified screen-capture interface.

    Parameters
    ----------
    backend:
        ``"mss"`` (default) or ``"dxcam"``.
    monitor_index:
        1-based monitor index.  ``1`` = primary display.
    crop:
        Optional ``(x, y, w, h)`` crop in screen coordinates.
        Pass ``None`` to capture the full monitor.
    """

    def __init__(
        self,
        backend: str = "mss",
        monitor_index: int = 1,
        crop: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self._backend = backend.lower()
        self._monitor_index = monitor_index
        self._crop = crop
        self._mss_sct = None
        self._dxcam_cam = None
        self._monitor_info: dict = {}
        self._init_backend()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        if self._backend == "dxcam":
            try:
                dxcam = importlib.import_module("dxcam")
                self._dxcam_cam = dxcam.create(
                    output_idx=self._monitor_index - 1,
                    output_color="BGR",
                )
                self._dxcam_cam.start(target_fps=60, video_mode=True)
                log.info("dxcam backend initialised.")
                return
            except Exception as exc:  # pylint: disable=broad-except
                log.warning("dxcam unavailable (%s), falling back to mss.", exc)
                self._backend = "mss"

        # mss fallback
        mss_mod = importlib.import_module("mss")
        self._mss_sct = mss_mod.mss()
        monitors = self._mss_sct.monitors  # index 0 = virtual combined
        idx = min(self._monitor_index, len(monitors) - 1)
        mon = monitors[idx]
        self._monitor_info = dict(mon)
        log.info("mss backend initialised — monitor %d: %s", idx, mon)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def grab(self) -> Optional[Frame]:
        """
        Capture a single frame.

        Returns
        -------
        numpy.ndarray or None
            BGR uint8 array, or ``None`` on failure.
        """
        if self._backend == "dxcam" and self._dxcam_cam is not None:
            return self._grab_dxcam()
        return self._grab_mss()

    @property
    def backend(self) -> str:
        """Active backend name."""
        return self._backend

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return the captured ``(width, height)``."""
        if self._crop and self._crop[2] > 0 and self._crop[3] > 0:
            return (self._crop[2], self._crop[3])
        mon = self._monitor_info
        return (mon.get("width", 1920), mon.get("height", 1080))

    def close(self) -> None:
        """Release backend resources."""
        if self._dxcam_cam is not None:
            try:
                self._dxcam_cam.stop()
            except Exception:  # pylint: disable=broad-except
                pass
        if self._mss_sct is not None:
            try:
                self._mss_sct.close()
            except Exception:  # pylint: disable=broad-except
                pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _grab_mss(self) -> Optional[Frame]:
        """Capture frame using mss."""
        try:
            import mss.tools  # noqa: F401 – ensure mss is importable
            import mss as mss_mod
            if self._mss_sct is None:
                self._init_backend()

            mon = self._monitor_info
            if self._crop and self._crop[2] > 0:
                x, y, w, h = self._crop
                region = {
                    "left": mon["left"] + x,
                    "top": mon["top"] + y,
                    "width": w,
                    "height": h,
                }
            else:
                region = mon

            screenshot = self._mss_sct.grab(region)  # type: ignore[union-attr]
            # mss returns BGRA; drop alpha channel
            frame = np.frombuffer(screenshot.bgra, dtype=np.uint8)
            frame = frame.reshape((screenshot.height, screenshot.width, 4))
            return frame[:, :, :3].copy()
        except Exception as exc:  # pylint: disable=broad-except
            log.error("mss grab failed: %s", exc)
            return None

    def _grab_dxcam(self) -> Optional[Frame]:
        """Capture frame using dxcam."""
        try:
            frame = self._dxcam_cam.get_latest_frame()  # type: ignore[union-attr]
            if frame is None:
                return None
            if self._crop and self._crop[2] > 0:
                x, y, w, h = self._crop
                frame = frame[y : y + h, x : x + w]
            return frame
        except Exception as exc:  # pylint: disable=broad-except
            log.error("dxcam grab failed: %s", exc)
            return None
