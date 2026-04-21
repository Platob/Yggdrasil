"""Backend-agnostic filesystem abstractions for yggdrasil.

Public surface
--------------
- :class:`Path`             — abstract ``pathlib.Path``-like path (no inheritance)
- :class:`PathKind`         — file/directory/symlink/other/missing enum
- :class:`StatResult`       — ``os.stat_result``-style dataclass
- :class:`FileSystem`       — abstract backend (path/stat/open/ls/mkdir/rm/…)
- :class:`LocalPath`        — local :class:`Path` backed by :mod:`pathlib`
- :class:`LocalFileSystem`  — local :class:`FileSystem` — scheme ``"file"``

Backends plug in via :func:`register_filesystem` (auto-registered when a
concrete subclass declares a ``scheme``). Look them up with
:func:`get_filesystem`.
"""

from __future__ import annotations

from .filesystem import FileSystem, get_filesystem, register_filesystem
from .local import LocalFileSystem, LocalPath
from .path import Path, PathKind, StatResult

__all__ = [
    "FileSystem",
    "LocalFileSystem",
    "LocalPath",
    "Path",
    "PathKind",
    "StatResult",
    "get_filesystem",
    "register_filesystem",
]
