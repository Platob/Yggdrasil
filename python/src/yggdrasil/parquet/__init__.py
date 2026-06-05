"""Parquet I/O — the Tabular leaf for a single ``.parquet`` file and the
optimized :class:`ParquetFolder` over a directory of them.

Lives in its own package (split out of :mod:`yggdrasil.io`) so the
format-specific machinery has room to grow; :mod:`yggdrasil.io` keeps a
back-compat re-export shim (``yggdrasil.io.parquet_file``).

- :class:`ParquetFile` — footer-indexed single-file leaf (registered under
  :data:`~yggdrasil.enums.MimeTypes.PARQUET`).
- :class:`ParquetFolder` — :class:`~yggdrasil.path.folder.Folder` over a
  directory of parquet part files, with a Hive-partition-aware, data-skipping
  read path (per-file row-group min/max pruning + projection/predicate
  pushdown into each leaf).

:class:`ParquetFolder` is imported lazily (it pulls in
:mod:`yggdrasil.path.folder`): the bootstrap that registers the single-file
leaves imports ``yggdrasil.io.parquet_file`` *from within* ``folder``'s import,
so eagerly importing the folder here would be a circular import. ``ParquetFile``
itself only needs the low-level IO substrate and imports cleanly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.parquet.parquet_file import ParquetFile, ParquetOptions

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.parquet.parquet_folder import ParquetFolder, ParquetFolderOptions

__all__ = [
    "ParquetFile",
    "ParquetOptions",
    "ParquetFolder",
    "ParquetFolderOptions",
]


def __getattr__(name: str) -> Any:
    if name in ("ParquetFolder", "ParquetFolderOptions"):
        from yggdrasil.parquet import parquet_folder
        return getattr(parquet_folder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
