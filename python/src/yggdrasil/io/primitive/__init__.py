"""Single-buffer tabular-format IO handlers.

Each concrete subclass under this package implements one file format
(Parquet, CSV, JSON, …). They all single-inherit from
:class:`yggdrasil.io.buffer.bytes_io.BytesIO`; the marker
``PrimitiveIO`` layer is gone — every concrete leaf is just a
:class:`BytesIO` with format-specific ``_read_arrow_batches`` /
``_write_arrow_batches`` hooks and a registered
``default_media_type``.

Importing this package as a whole forces every leaf module to load,
which auto-registers them in :data:`Tabular._DATAIO_REGISTRY` via
``__init_subclass__``. :meth:`Tabular.class_for_media_type` triggers
this import lazily so registry hits stay correct even when callers
start at a leaf module directly.
"""

from .arrow_ipc_io import ArrowIPCIO
from .parquet_io import ParquetIO
from .csv_io import CsvIO
from .json_io import JsonIO
from .ndjson_io import NDJsonIO
from .xlsx_io import XlsxIO

__all__ = [
    'ArrowIPCIO',
    'ParquetIO', 'CsvIO', 'JsonIO', 'NDJsonIO', 'XlsxIO',
]
