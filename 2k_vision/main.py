"""
main.py — Entry point for 2k Vision.

Launch with::

    python -m 2k_vision.main

or::

    python 2k_vision/main.py
"""

from __future__ import annotations

import logging
import sys

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
