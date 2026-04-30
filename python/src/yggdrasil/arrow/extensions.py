import os

from .tzdata import ensure_tzdata

if os.getenv("ENSURE_TZDATA"):
    ensure_tzdata()

__all__ = ["ensure_tzdata"]
