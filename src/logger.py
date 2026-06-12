"""
src/logger.py
=============
Centralized logging configuration for the RAG pipeline.

WHY USE LOGGING INSTEAD OF PRINT?
  1. Log levels — filter noise: DEBUG in dev, INFO in production
  2. Timestamps — know WHEN each step ran
  3. Module names — know WHERE the log came from
  4. File output — persist logs for post-mortem debugging
  5. Industry standard — every production Python project uses logging

USAGE:
    from logger import get_logger
    logger = get_logger(__name__)

    logger.info("Processing started")
    logger.debug("Record count: %d", len(records))
    logger.error("Download failed: %s", str(e))
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and returns a configured logger instance.

    Args:
        name:  module name (use __name__ for automatic naming)
        level: logging level (default: INFO)

    Returns:
        logging.Logger with console handler and consistent format.

    FORMAT EXPLANATION:
        %(asctime)s      → timestamp (2026-04-14 13:00:00)
        %(name)s         → module name (e.g., preprocessing_data.pre_processing)
        %(levelname)-8s  → level padded to 8 chars (INFO    , ERROR   )
        %(message)s      → the actual log message
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Console handler — writes to stdout ────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-40s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Prevent log messages from propagating to the root logger
    logger.propagate = False

    return logger
