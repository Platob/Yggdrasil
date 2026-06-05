from __future__ import annotations

import logging
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc

from yggdrasil.enums import MimeTypes

LOGGER = logging.getLogger(__name__)

# Canonical content types come from the project's MimeType registry so
# the wire labels stay in one place (Arrow stream framing, Parquet).
CONTENT_TYPE_ARROW_STREAM = MimeTypes.ARROW_STREAM.value
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


def iter_arrow_ipc_stream(
    batches: "Iterator[pa.RecordBatch]",
    schema: pa.Schema,
) -> Iterator[bytes]:
    """Encode a record-batch iterator into a streaming IPC byte iterator.

    Unlike :func:`write_arrow_stream` (which buffers the whole table and yields
    one blob), this drains the writer after each batch so a multi-GB result
    leaves the server in bounded chunks. Framing is preserved — concatenating
    every yielded chunk yields exactly one valid Arrow IPC stream.
    """
    import io

    buf = io.BytesIO()
    sink = pa.output_stream(buf)
    writer = ipc.RecordBatchStreamWriter(sink, schema)

    def _drain() -> bytes:
        sink.flush()
        data = buf.getvalue()
        if data:
            buf.seek(0)
            buf.truncate(0)
        return data

    head = _drain()  # schema message
    if head:
        yield head
    for batch in batches:
        writer.write_batch(batch)
        chunk = _drain()
        if chunk:
            yield chunk
    writer.close()
    tail = _drain()  # EOS marker
    if tail:
        yield tail


def iter_file_chunks(path: str, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    """Stream a file from disk in ``chunk_size`` byte chunks.

    Used to hand back a spilled Arrow IPC result without re-reading it into
    memory — the node spills heavy query results to ``spill_root`` and streams
    the file straight off disk.
    """
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            yield chunk


def write_arrow_ipc_file(path: str, batches: "Iterator[pa.RecordBatch]", schema: pa.Schema) -> int:
    """Write a record-batch iterator to an Arrow IPC stream file on disk.

    Returns the row count written. Memory stays bounded by one batch — the
    spill path for results too large to hold whole.
    """
    rows = 0
    with pa.OSFile(path, "wb") as sink:
        with ipc.RecordBatchStreamWriter(sink, schema) as writer:
            for batch in batches:
                writer.write_batch(batch)
                rows += batch.num_rows
    return rows


# -- Parquet transport (Power Query reads this natively) -------------------

CONTENT_TYPE_PARQUET = MimeTypes.PARQUET.value


def write_parquet_bytes(table: pa.Table, *, compression: str = "snappy") -> bytes:
    """Serialize a table to a Parquet file in memory.

    Parquet is the lingua franca for Excel / Power BI: M reads it
    natively via ``Parquet.Document`` with full type fidelity, so the
    Excel-facing endpoints hand back Parquet by default.
    """
    import pyarrow.parquet as pq

    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=compression)
    return sink.getvalue().to_pybytes()


def read_parquet_bytes(data: bytes) -> pa.Table:
    import pyarrow.parquet as pq

    return pq.read_table(pa.BufferReader(data))


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
