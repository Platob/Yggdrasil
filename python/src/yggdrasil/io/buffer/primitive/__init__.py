"""Single-buffer tabular-format IO handlers.

Each concrete subclass under this package implements one file format
(Parquet, CSV, JSON, …). They all share the :class:`PrimitiveIO`
abstract base, which wraps a :class:`BytesIO` and exposes
:meth:`PrimitiveIO._open_reader` / :meth:`PrimitiveIO._open_writer`
hooks returning Arrow-native file-likes.

New format handlers go under this package as ``primitive/<name>.py``
and register themselves via ``media_type`` at class-definition time.
"""

from .base import PrimitiveIO
from .arrow_ipc_io import ArrowIPCIO
from .parquet_io import ParquetIO
from .csv_io import CsvIO
from .json_io import JsonIO
from .ndjson_io import NDJsonIO
from .xlsx_io import XlsxIO

__all__ = [
    'PrimitiveIO',
    'ArrowIPCIO',
    'ParquetIO', 'CsvIO', 'JsonIO', 'NDJsonIO', 'XlsxIO',
]
