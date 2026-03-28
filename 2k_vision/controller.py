"""
controller.py — XInput reading + ViGEm virtual pad proxy for 2k Vision.

Polls the physical Xbox controller at 1000 Hz in a daemon thread,
mirrors all inputs to a ViGEm virtual X360 pad, and intercepts the X
button so the engine can release the shot at the optimal frame.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import platform
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XInput structs (ctypes)
# ---------------------------------------------------------------------------

XINPUT_GAMEPAD_X = 0x4000
XINPUT_GAMEPAD_LB = 0x0100

# Try loading XInput DLL
_XINPUT: Optional[ctypes.WinDLL] = None
for _dll_name in ("xinput1_4", "xinput9_1_0", "xinput1_3"):
    try:
        _XINPUT = ctypes.windll.LoadLibrary(_dll_name)  # type: ignore[attr-defined]
        break
    except (OSError, AttributeError):
        continue


class _XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_uint16),
        ("bLeftTrigger", ctypes.c_uint8),
        ("bRightTrigger", ctypes.c_uint8),
        ("sThumbLX", ctypes.c_int16),
        ("sThumbLY", ctypes.c_int16),
        ("sThumbRX", ctypes.c_int16),
        ("sThumbRY", ctypes.c_int16),
    ]


class _XInputState(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.c_uint32),
        ("Gamepad", _XInputGamepad),
    ]


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------


@dataclass
class ControllerState:
    """Current snapshot of the controller inputs."""

    buttons: int = 0
    lt: int = 0
    rt: int = 0
    lx: int = 0
    ly: int = 0
    rx: int = 0
    ry: int = 0
    x_pressed: bool = False
    x_held: bool = False
    connected: bool = False


# ---------------------------------------------------------------------------
# Controller proxy
# ---------------------------------------------------------------------------


class ControllerProxy:
    """
    XInput reader + ViGEm virtual pad proxy.

    The physical controller is mirrored to a ViGEm virtual pad.
    The X button is intercepted: the engine can inject a release at
    any moment by calling ``release_shot()``.

    If either XInput or ViGEm is unavailable (non-Windows, missing DLL)
    the class degrades gracefully — state reading still works but no
    virtual pad is created.
    """

    def __init__(
        self,
        controller_index: int = 0,
        on_x_press: Optional[Callable[[], None]] = None,
        on_x_release: Optional[Callable[[], None]] = None,
        remap_lb_to_x: bool = True,
    ) -> None:
        self._index = controller_index
        self._on_x_press = on_x_press
        self._on_x_release = on_x_release
        self._cfg_remap_lb = remap_lb_to_x

        self._state = ControllerState()
        self._x_was_held = False
        self._intercept_x = False  # engine sets this during SHOOTING state

        self._vpad = None
        self._init_vigem()

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._running = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the 1000 Hz polling thread."""
        self._running = True
        self._thread.start()
        log.info("ControllerProxy polling started (index=%d).", self._index)

    def stop(self) -> None:
        """Stop the polling thread and release ViGEm pad."""
        self._running = False
        if self._vpad is not None:
            try:
                self._vpad.target_remove()
            except Exception:  # pylint: disable=broad-except
                pass

    @property
    def state(self) -> ControllerState:
        """Current snapshot of the controller state."""
        return self._state

    def set_intercept(self, intercept: bool) -> None:
        """
        When ``intercept=True``, the X button press is suppressed and the
        engine controls the shot release.
        """
        self._intercept_x = intercept

    def fire_shot(self) -> None:
        """
        Inject an X-button RELEASE on the virtual pad to fire the shot.
        Called by the engine at the optimal frame.
        """
        if self._vpad is None:
            return
        try:
            import vgamepad as vg  # noqa: F401
            self._vpad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            self._vpad.update()
        except Exception as exc:  # pylint: disable=broad-except
            log.debug("fire_shot vgamepad error: %s", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_vigem(self) -> None:
        """Try to create a ViGEm virtual X360 pad."""
        try:
            import vgamepad as vg
            self._vpad = vg.VX360Gamepad()
            log.info("ViGEm virtual X360 pad created.")
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("ViGEm not available (%s) — no virtual pad.", exc)
            self._vpad = None

    def _poll_loop(self) -> None:
        """Poll XInput at 1000 Hz and mirror to virtual pad."""
        interval = 1.0 / 1000.0
        while self._running:
            start = time.perf_counter()
            self._poll()
            elapsed = time.perf_counter() - start
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _poll(self) -> None:
        """Read one XInput frame and update state + virtual pad."""
        if _XINPUT is None:
            return

        xi_state = _XInputState()
        ret = _XINPUT.XInputGetState(self._index, ctypes.byref(xi_state))
        connected = ret == 0
        gp = xi_state.Gamepad

        buttons = gp.wButtons
        x_held = bool(buttons & XINPUT_GAMEPAD_X)
        lb_held = bool(buttons & XINPUT_GAMEPAD_LB)

        # LB → X remap (so user can still charge shot with LB)
        if lb_held and self._cfg_remap_lb:
            buttons |= XINPUT_GAMEPAD_X

        # X press/release callbacks
        if x_held and not self._x_was_held:
            if self._on_x_press:
                self._on_x_press()
        if not x_held and self._x_was_held:
            if self._on_x_release:
                self._on_x_release()
        self._x_was_held = x_held

        self._state = ControllerState(
            buttons=buttons,
            lt=gp.bLeftTrigger,
            rt=gp.bRightTrigger,
            lx=gp.sThumbLX,
            ly=gp.sThumbLY,
            rx=gp.sThumbRX,
            ry=gp.sThumbRY,
            x_pressed=(x_held and not self._x_was_held),
            x_held=x_held,
            connected=connected,
        )

        # Mirror to virtual pad (suppress X if intercepting)
        self._update_vpad(buttons, gp, suppress_x=self._intercept_x)

    def _update_vpad(
        self,
        buttons: int,
        gp: _XInputGamepad,
        suppress_x: bool = False,
    ) -> None:
        """Update ViGEm virtual pad with current physical state."""
        if self._vpad is None:
            return
        try:
            import vgamepad as vg

            if suppress_x:
                # Remove X from button mask so the game sees it held
                buttons &= ~XINPUT_GAMEPAD_X

            self._vpad.left_joystick(x_value=gp.sThumbLX, y_value=gp.sThumbLY)
            self._vpad.right_joystick(x_value=gp.sThumbRX, y_value=gp.sThumbRY)
            self._vpad.left_trigger(value=gp.bLeftTrigger)
            self._vpad.right_trigger(value=gp.bRightTrigger)

            # Set all buttons
            for btn in vg.XUSB_BUTTON:
                if buttons & btn.value:
                    self._vpad.press_button(button=btn)
                else:
                    self._vpad.release_button(button=btn)

            self._vpad.update()
        except Exception as exc:  # pylint: disable=broad-except
            log.debug("vpad update error: %s", exc)
