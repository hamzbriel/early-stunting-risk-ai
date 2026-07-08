"""
utils/logger.py - Centralized logging configuration.

Provides a factory function that returns a consistently configured
Python standard-library logger for any module in the pipeline.

Usage:
    from synthetic_data.src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    Return a configured Logger instance.

    Parameters
    ----------
    name:
        Logger name - typically pass ``__name__`` from the calling module.
    level:
        Logging level (e.g. logging.DEBUG, logging.INFO).
        Defaults to INFO.
    fmt:
        Optional custom format string. Defaults to a structured format that
        includes timestamp, level, logger name, and message.

    Returns
    -------
    logging.Logger
        A logger with a StreamHandler writing to stdout.
    """
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def set_global_log_level(level_name: str) -> None:
    """
    Set the log level for all synthetic_data loggers.

    Parameters
    ----------
    level_name:
        One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.getLogger("synthetic_data").setLevel(level)
