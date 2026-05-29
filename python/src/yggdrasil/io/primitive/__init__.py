"""Single-buffer tabular-format IO handlers.

Each concrete subclass under this package implements one file format
(Parquet, CSV, JSON, …). They all single-inherit from
:class:`yggdrasil.io.holder.IO`; the marker
``PrimitiveIO`` layer is gone — every concrete leaf is just an
:class:`IO` with format-specific ``_read_arrow_batches`` /
``_write_arrow_batches`` hooks and a registered
``default_media_type``.

Importing this package as a whole forces every leaf module to load,
which auto-registers them in :data:`Tabular._DATAIO_REGISTRY` via
``__init_subclass__``. :meth:`Tabular.class_for_media_type` triggers
this import lazily so registry hits stay correct even when callers
start at a leaf module directly.
"""

from .arrow_ipc_file import ArrowIPCFile
from .parquet_file import ParquetFile
from .csv_file import CSVFile
from .json_file import JSONFile
from .ndjson_file import NDJSONFile
from .xlsx_file import XLSXFile
from .pickle_file import PickleFile

__all__ = [
    'ArrowIPCFile',
    'ParquetFile', 'CSVFile', 'JSONFile', 'NDJSONFile', 'XLSXFile',
    'PickleFile',
]
