"""Node transport — tabular payloads ride Arrow IPC, everything else rides pickle.

The node speaks two wire formats. Tabular results (Arrow tables, polars frames)
serialize to the Arrow IPC stream so they stay zero-copy and columnar end to
end; scalars, dicts, and bytes ride the yggdrasil pickle (orjson fast-path for
simple shapes, pickle for the rest). :func:`serialize_result` picks the format
from the value and reports the content type so the peer can deserialize it.
"""
from __future__ import annotations

import io
from typing import Any, Iterator

import pyarrow as pa

from yggdrasil.pickle.ser.serde import dumps as _ygg_dumps, loads as _ygg_loads

CONTENT_TYPE_ARROW_STREAM = "application/vnd.apache.arrow.stream"
CONTENT_TYPE_PICKLE = "application/x-ygg-pickle"


def is_tabular(obj: object) -> bool:
    """True if *obj* is an Arrow table or a polars (Lazy)Frame."""
    if isinstance(obj, pa.Table):
        return True
    type_name = type(obj).__name__
    module = type(obj).__module__
    return module.startswith("polars") and type_name in ("DataFrame", "LazyFrame")


def to_arrow_table(obj: object) -> pa.Table:
    """Coerce *obj* (Arrow table or polars (Lazy)Frame) to a ``pa.Table``."""
    if isinstance(obj, pa.Table):
        return obj
    type_name = type(obj).__name__
    if type_name == "LazyFrame":
        obj = obj.collect()
    to_arrow = getattr(obj, "to_arrow", None)
    if to_arrow is not None:
        return to_arrow()
    raise TypeError(
        f"Cannot convert {type(obj).__name__} to a pyarrow.Table; expected a "
        f"pyarrow.Table or a polars DataFrame/LazyFrame."
    )


def write_arrow_stream(table: pa.Table) -> Iterator[bytes]:
    """Serialize *table* to an Arrow IPC stream, yielding the buffer as bytes."""
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    yield sink.getvalue()


def write_arrow_stream_chunked(table: pa.Table, max_chunksize: int = 8192) -> Iterator[bytes]:
    """Serialize *table* to an Arrow IPC stream in ``max_chunksize``-row batches."""
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=max_chunksize)
    yield sink.getvalue()


def read_arrow_stream(data: bytes) -> pa.Table:
    """Read an Arrow IPC stream back into a ``pa.Table``."""
    with pa.ipc.open_stream(pa.py_buffer(data)) as reader:
        return reader.read_all()


def serialize_pickle(obj: object) -> bytes:
    """Serialize *obj* with the yggdrasil pickle (orjson fast-path + pickle)."""
    return _ygg_dumps(obj)


def deserialize_pickle(data: bytes) -> Any:
    """Inverse of :func:`serialize_pickle`."""
    return _ygg_loads(data)


def serialize_result(obj: object) -> tuple[bytes, str]:
    """Serialize a handler result, choosing Arrow for tabular, pickle otherwise."""
    if is_tabular(obj):
        return b"".join(write_arrow_stream(to_arrow_table(obj))), CONTENT_TYPE_ARROW_STREAM
    return serialize_pickle(obj), CONTENT_TYPE_PICKLE


def deserialize_result(data: bytes, content_type: str) -> Any:
    """Inverse of :func:`serialize_result`, dispatched on *content_type*."""
    if content_type == CONTENT_TYPE_ARROW_STREAM:
        return read_arrow_stream(data)
    return deserialize_pickle(data)
