"""
main.py — Entry point for 2k Vision.

Launch with either::

    python -m 2k_vision.main          # recommended (relative imports work)

or directly::

    python 2k_vision/main.py          # also supported
    python main.py  (from inside the 2k_vision/ folder)
"""

from __future__ import annotations

import logging
import os
import sys

# ---------------------------------------------------------------------------
# When run directly as a script (`python 2k_vision/main.py`) Python sets
# __package__ to None and __name__ to "__main__", which breaks all relative
# imports inside the package.
#
# Fix: register the 2k_vision package in sys.modules and set __package__ on
# this module so that every subsequent `from .x import y` resolves correctly —
# without re-executing the file (avoids confusing traceback wrapping).
# ---------------------------------------------------------------------------
if __name__ == "__main__" and __package__ in (None, ""):
    import importlib.util as _ilu

    # Absolute path of the package directory (the folder containing this file)
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    # Repo root = parent of the package directory
    _repo_root = os.path.dirname(_pkg_dir)

    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    # Load and register the package so sub-module relative imports work
    _pkg_spec = _ilu.spec_from_file_location(
        "2k_vision",
        os.path.join(_pkg_dir, "__init__.py"),
        submodule_search_locations=[_pkg_dir],
    )
    _pkg_mod = _ilu.module_from_spec(_pkg_spec)
    sys.modules.setdefault("2k_vision", _pkg_mod)
    _pkg_spec.loader.exec_module(_pkg_mod)  # type: ignore[union-attr]

    # Register this module as 2k_vision.main and fix its __package__
    sys.modules["2k_vision.main"] = sys.modules["__main__"]
    __package__ = "2k_vision"  # noqa: WPS125  (shadows built-in for relative-import fix)

    del _ilu, _pkg_dir, _repo_root, _pkg_spec, _pkg_mod

# Configure logging before importing anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("2k_vision.main")


def main() -> None:
    """Application entry point."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        log.critical("PyQt6 is not installed.  Run: pip install PyQt6")
        sys.exit(1)

    from .config import get_config
    from .ui.app import ControlPanel

    cfg = get_config()

    app = QApplication(sys.argv)
    app.setApplicationName("2k Vision")
    app.setApplicationVersion("2.0")

    window = ControlPanel(cfg)
    window.show()

    log.info("2k Vision V2.0 started.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
