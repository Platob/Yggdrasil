from __future__ import annotations

from enum import IntEnum

__all__ = ["IOKind"]


class IOKind(IntEnum):
    """What a backend reports a path/holder entry is."""

    MISSING = 0
    FILE = 1
    DIRECTORY = 2
    MEMORY = 3
