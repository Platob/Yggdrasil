"""Back-compat shim — parquet I/O moved to :mod:`yggdrasil.parquet`.

``ParquetFile`` / ``ParquetOptions`` now live in
:mod:`yggdrasil.parquet.parquet_file`; importing them here keeps the old
``yggdrasil.io.parquet_file`` path working (and re-triggers the
:data:`~yggdrasil.enums.MimeTypes.PARQUET` registration on import).
"""
from __future__ import annotations

from yggdrasil.parquet.parquet_file import ParquetFile, ParquetOptions

__all__ = ["ParquetFile", "ParquetOptions"]
