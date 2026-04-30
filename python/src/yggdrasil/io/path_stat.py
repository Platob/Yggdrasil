from dataclasses import dataclass
from enum import Enum
from typing import Any


__all__ = ["PathStats", "PathKind"]


class PathKind(str, Enum):
    MISSING = "missing"
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    SOCKET = "socket"
    FIFO = "fifo"
    CHAR_DEVICE = "char_device"
    BLOCK_DEVICE = "block_device"


@dataclass(frozen=True, slots=True)
class PathStats:
    """Minimal ``os.stat_result`` stand-in usable across backends.

    Backends with richer metadata (ETag, content-type, owner…) should
    subclass and extend rather than cram extras into ``mode``.
    """

    size: int = 0
    mtime: float = 0.0
    kind: PathKind = PathKind.MISSING
    mode: int = 0

    # os.stat_result is subscript-compatible — keep that affordance.
    def __getitem__(self, idx: int) -> Any:
        return (self.mode, 0, 0, 0, 0, 0, self.size, 0, self.mtime, 0)[idx]

    # Drop-in aliases for code that already reads os.stat_result.
    @property
    def st_size(self) -> int:
        return self.size

    @property
    def st_mtime(self) -> float:
        return self.mtime

    @property
    def st_mode(self) -> int:
        return self.mode

    def with_(
        self,
        *,
        size: int | None = None,
        mtime: float | None = None,
        kind: PathKind | None = None,
        mode: int | None = None,
        copy: bool = False,
    ):
        if copy:
            return PathStats(
                size=size or self.size,
                mtime=mtime or self.mtime,
                kind=kind or self.kind,
                mode=mode or self.mode,
            )

        if size is not None:
            object.__setattr__(self, "size", size)

        if mtime is not None:
            object.__setattr__(self, "mtime", mtime)

        if kind is not None:
            object.__setattr__(self, "kind", kind)

        if mode is not None:
            object.__setattr__(self, "mode", mode)

        return self