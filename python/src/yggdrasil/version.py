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

    @classmethod
    def from_string(cls, s: str) -> VersionInfo:
        parts = s.split(".")

        if len(parts) < 2:
            raise ValueError(f"Invalid version string: {s}")

        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2]) if len(parts) == 3 else 0
        pre = parts[3] if len(parts) == 4 else ""
        build = parts[4] if len(parts) == 5 else ""

        return cls(major, minor, patch, pre, build)

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre:
            base += f"{self.pre}"
        if self.build:
            base += f"+{self.build}"
        return base


__version_info__ = VersionInfo.from_string("0.6.2")
__version__ = str(__version_info__)
