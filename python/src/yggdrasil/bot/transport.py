from __future__ import annotations

import logging
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc

LOGGER = logging.getLogger(__name__)

CONTENT_TYPE_ARROW_STREAM = "application/vnd.apache.arrow.stream"
CONTENT_TYPE_PICKLE = "application/x-yggdrasil-pickle"

_ARROW_TABULAR_TYPES = (pa.Table, pa.RecordBatch)

try:
    import polars as pl
    _POLARS_TYPES = (pl.DataFrame, pl.LazyFrame, pl.Series)
except ImportError:
    pl = None
    _POLARS_TYPES = ()

try:
    import pandas as pd
    _PANDAS_TYPES = (pd.DataFrame, pd.Series)
except ImportError:
    pd = None
    _PANDAS_TYPES = ()

# Cache pickle imports at module level to avoid per-call import overhead.
_ygg_dumps = None
_ygg_loads = None


def _ensure_pickle():
    global _ygg_dumps, _ygg_loads
    if _ygg_dumps is None:
        from yggdrasil.pickle import dumps, loads
        _ygg_dumps = dumps
        _ygg_loads = loads


def is_tabular(obj: Any) -> bool:
    if isinstance(obj, _ARROW_TABULAR_TYPES):
        return True
    if _POLARS_TYPES and isinstance(obj, _POLARS_TYPES):
        return True
    if _PANDAS_TYPES and isinstance(obj, _PANDAS_TYPES):
        return True
    return False


def to_arrow_table(obj: Any) -> pa.Table:
    if isinstance(obj, pa.Table):
        return obj
    if isinstance(obj, pa.RecordBatch):
        return pa.Table.from_batches([obj])
    if pl is not None:
        if isinstance(obj, pl.LazyFrame):
            obj = obj.collect()
        if isinstance(obj, pl.DataFrame):
            return obj.to_arrow()
        if isinstance(obj, pl.Series):
            return pa.table({obj.name or "value": obj.to_arrow()})
    if pd is not None:
        if isinstance(obj, pd.Series):
            return pa.table({obj.name or "value": pa.array(obj, from_pandas=True)})
        if isinstance(obj, pd.DataFrame):
            return pa.Table.from_pandas(obj, preserve_index=False)
    raise TypeError(f"Cannot convert {type(obj).__name__} to Arrow table")


# -- Arrow IPC streaming ---------------------------------------------------

def write_arrow_stream(table: pa.Table) -> Iterator[bytes]:
    sink = pa.BufferOutputStream()
    with ipc.RecordBatchStreamWriter(sink, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=65536):
            writer.write_batch(batch)
    yield sink.getvalue().to_pybytes()


def write_arrow_stream_chunked(
    table: pa.Table,
    max_chunksize: int = 65536,
) -> Iterator[bytes]:
    yield write_arrow_stream_bytes(table)


def write_arrow_stream_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.RecordBatchStreamWriter(sink, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=65536):
            writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def read_arrow_stream(data: bytes) -> pa.Table:
    reader = ipc.open_stream(data)
    return reader.read_all()


# -- Pickle transport (compressed) -----------------------------------------

def serialize_pickle(obj: Any) -> bytes:
    _ensure_pickle()
    return _ygg_dumps(obj)


def deserialize_pickle(data: bytes) -> Any:
    _ensure_pickle()
    return _ygg_loads(data)


def serialize_result(obj: Any) -> tuple[bytes, str]:
    if is_tabular(obj):
        table = to_arrow_table(obj)
        return write_arrow_stream_bytes(table), CONTENT_TYPE_ARROW_STREAM
    return serialize_pickle(obj), CONTENT_TYPE_PICKLE


def deserialize_result(data: bytes, content_type: str) -> Any:
    if CONTENT_TYPE_ARROW_STREAM in content_type:
        return read_arrow_stream(data)
    return deserialize_pickle(data)
