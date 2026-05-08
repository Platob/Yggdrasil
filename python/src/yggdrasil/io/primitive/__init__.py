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

from .arrow_ipc_io import ArrowIPCFile
from .parquet_io import ParquetFile
from .csv_io import CsvFile
from .json_io import JsonFile
from .ndjson_io import NDJsonFile
from .xlsx_io import XlsxFile

__all__ = [
    'ArrowIPCFile',
    'ParquetFile', 'CsvFile', 'JsonFile', 'NDJsonFile', 'XlsxFile',
]
