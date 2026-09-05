"""
NSE Cup & Handle Scanner - Logging Utilities
=============================================
Single shared logger configuration used across every module.
Writes to console (INFO+) and to a rotating debug.log file (DEBUG+)
so verbose per-symbol diagnostics don't flood the GitHub Actions
console but are still available for download as part of the run.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_CONFIGURED: set[str] = set()


def get_logger(name: str = "scanner") -> logging.Logger:
    """
    Return a configured logger. Safe to call repeatedly with the same
    name from different modules — handlers are only attached once.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above, this is what shows in the
    # GitHub Actions log viewer
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Rotating file handler — DEBUG and above, full detail
    try:
        file_handler = RotatingFileHandler(
            LOG_DIR / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception:
        # If the filesystem is read-only or restricted, console-only
        # logging still works fine — never let logging setup crash
        # the scanner.
        pass

    # Separate error-only file for quick triage of a failed run
    try:
        error_handler = RotatingFileHandler(
            LOG_DIR / "error.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=2,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(fmt)
        logger.addHandler(error_handler)
    except Exception:
        pass

    _CONFIGURED.add(name)
    return logger
