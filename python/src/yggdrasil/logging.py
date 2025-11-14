"""Logging utilities for Yggdrasil."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: The name of the logger, typically the module name.
        level: Optional log level to set. If None, doesn't change the level.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    # Only add a handler if the logger doesn't have one already
    # and if it's not a child logger (with a dot in the name)
    if not logger.handlers and "." not in name:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a standard format and level.

    Args:
        level: The logging level to set (default: INFO)
    """
    root_logger = logging.getLogger("yggdrasil")
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # Configure the root handler
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Set propagate False for the yggdrasil logger to avoid duplicate logs
    root_logger.propagate = False


# Export symbols
__all__ = ["get_logger", "setup_logging"]