"""Yggdrasil - Python utilities for multi-language repository."""

from __future__ import annotations

# Initialize logging for the package
from .logging import setup_logging

# Set up the default logger at import time
setup_logging()

__version__ = "0.1.0"