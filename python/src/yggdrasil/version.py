from __future__ import annotations

from typing import NamedTuple

__all__ = [
    "VersionInfo",
    "__version_info__",
    "__version__"
]

class VersionInfo(NamedTuple):
    major: int
    minor: int
    patch: int
    pre: str = ""        # e.g. "a1", "b2", "rc1"
    build: str = ""      # e.g. "dev", "gitabcdef", "20260227"

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre:
            base += f"{self.pre}"
        if self.build:
            base += f"+{self.build}"
        return base


__version_info__ = VersionInfo(0, 4, 3)
__version__ = str(__version_info__)