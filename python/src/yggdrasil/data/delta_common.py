"""Common Delta Lake functionality for Yggdrasil."""

from __future__ import annotations

from typing import Any

# Conditionally import delta for interoperability
try:
    import deltalake
    HAS_DELTA = True
except ImportError:
    HAS_DELTA = False
    deltalake = Any  # type: ignore