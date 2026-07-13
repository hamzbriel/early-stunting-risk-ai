"""
Logging configuration for the backend application.

Provides consistent logging setup across all modules with
appropriate formatting and log levels.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup and configure a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Format: timestamp - logger name - level - message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get or create a logger with standard configuration.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Optional logging level override

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    if level is None:
        level = logging.INFO

    return setup_logger(name, level)
