"""Backend-agnostic filesystem abstractions for yggdrasil.

Public surface
--------------
- :class:`Path`             — abstract ``pathlib.Path``-like path
- :class:`RemotePath`       — abstract base for network-backed paths (S3, Databricks, …)
- :class:`LocalPath`        — local :class:`Path` backed by :mod:`os` syscalls
- :class:`MemoryPath`       — in-process :class:`Path` over a :class:`BytesIO`

Stat-related types live on :mod:`yggdrasil.io.io_stats`
(:class:`IOStats` / :class:`IOKind`).

Concrete remote backends register themselves on import (see
``yggdrasil.aws.fs`` and ``yggdrasil.databricks.fs``).
"""

from __future__ import annotations

from .path import Path
from .remote_path import RemotePath
from .local_path import LocalPath
from .memory_io import MemoryPath

__all__ = [
    "Path",
    "RemotePath",
    "LocalPath",
    "MemoryPath",
]
