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
# imports inside the package.  Fix this by ensuring the *repo root* (the
# directory that CONTAINS the 2k_vision/ folder) is on sys.path and then
# re-importing this module as part of the package.
# ---------------------------------------------------------------------------
if __name__ == "__main__" and __package__ in (None, ""):
    # Absolute path of the repo root (parent of this file's parent dir)
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    # Re-launch via runpy so relative imports resolve correctly
    import runpy
    runpy.run_module("2k_vision.main", run_name="__main__", alter_sys=True)
    sys.exit(0)

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
