"""Backend-agnostic filesystem abstractions for yggdrasil.

Public surface
--------------
- :class:`Path`       — abstract ``pathlib.Path``-like path (no inheritance)
- :class:`PurePath`   — pure manipulation (name/parts/suffix/…)
- :class:`StatResult` — ``os.stat_result``-style dataclass
- :class:`PathKind`   — file/dir/symlink/missing enum
- :class:`FileSystem` — abstract backend (open/ls/stat/mkdir/rm/rename)

Backends plug in via :func:`register_filesystem` (auto-registered when the
subclass declares a ``scheme``). Looked up with :func:`get_filesystem`.
"""

from __future__ import annotations

from .filesystem import FileSystem, get_filesystem, register_filesystem
from .path import Path, PathKind, PurePath, StatResult

__all__ = [
    "FileSystem",
    "Path",
    "PathKind",
    "PurePath",
    "StatResult",
    "get_filesystem",
    "register_filesystem",
]
