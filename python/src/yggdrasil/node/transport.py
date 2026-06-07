"""Serialization layer for the node server.

Two binary wire formats sit alongside JSON:

- **Arrow IPC stream** (``application/vnd.apache.arrow.stream``) — the
  fast path for tabular results. Zero-copy on the client, columnar, and
  understood by polars / pandas / DuckDB / the JS arrow reader out of the box.
- **yggdrasil pickle** (``application/x-python-pickle``) — the fallback
  for arbitrary Python objects that aren't tabular.

``serialize_result``/``deserialize_result`` pick the right codec from the
content type. The FastAPI helpers turn an Arrow table into a streaming
``Response`` with the correct header so a client sending
``Accept: application/vnd.apache.arrow.stream`` gets binary back.
"""
from __future__ import annotations

from typing import Any

import pyarrow as pa

from yggdrasil.pickle import dumps as pickle_dumps, loads as pickle_loads

__all__ = [
    "CONTENT_TYPE_ARROW_STREAM",
    "CONTENT_TYPE_PICKLE",
    "CONTENT_TYPE_JSON",
    "serialize_result",
    "deserialize_result",
    "write_arrow_stream",
    "read_arrow_stream",
    "to_arrow_table",
    "is_tabular",
    "arrow_response",
    "negotiate_response",
]

CONTENT_TYPE_ARROW_STREAM = "application/vnd.apache.arrow.stream"
CONTENT_TYPE_PICKLE = "application/x-python-pickle"
CONTENT_TYPE_JSON = "application/json"


def write_arrow_stream(table: pa.Table) -> bytes:
    """Serialize *table* to an Arrow IPC stream frame."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def read_arrow_stream(data: bytes) -> pa.Table:
    """Read an Arrow IPC stream frame back into a :class:`pyarrow.Table`."""
    with pa.ipc.open_stream(pa.py_buffer(data)) as reader:
        return reader.read_all()


def to_arrow_table(obj: Any) -> pa.Table | None:
    """Coerce *obj* to a :class:`pyarrow.Table`, or ``None`` if not tabular.

    Handles pyarrow ``Table``/``RecordBatch``, polars ``DataFrame``, and
    anything exposing ``to_arrow``/``arrow`` (the yggdrasil ``Tabular``
    protocol — ``Table`` and ``StatementResult`` both do).
    """
    if isinstance(obj, pa.Table):
        return obj
    if isinstance(obj, pa.RecordBatch):
        return pa.Table.from_batches([obj])

    cls = type(obj)
    qual = f"{cls.__module__}.{cls.__qualname__}"
    if qual == "polars.dataframe.frame.DataFrame" or cls.__name__ == "DataFrame":
        to_arrow = getattr(obj, "to_arrow", None)
        if callable(to_arrow):
            result = to_arrow()
            if isinstance(result, pa.Table):
                return result
            if isinstance(result, pa.RecordBatch):
                return pa.Table.from_batches([result])

    for attr in ("to_arrow", "arrow", "to_arrow_table"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            result = fn()
            if isinstance(result, pa.Table):
                return result
            if isinstance(result, pa.RecordBatch):
                return pa.Table.from_batches([result])

    return None


def is_tabular(obj: Any) -> bool:
    """Whether *obj* can be represented as an Arrow table."""
    return to_arrow_table(obj) is not None


def serialize_result(obj: Any, content_type: str) -> bytes:
    """Serialize *obj* to bytes for *content_type*.

    Arrow IPC for tabular data, yggdrasil pickle otherwise. Asking for
    Arrow on a non-tabular object falls back to pickle rather than failing.
    """
    if content_type == CONTENT_TYPE_ARROW_STREAM:
        table = to_arrow_table(obj)
        if table is not None:
            return write_arrow_stream(table)
        return pickle_dumps(obj)
    return pickle_dumps(obj)


def deserialize_result(data: bytes, content_type: str) -> Any:
    """Inverse of :func:`serialize_result`."""
    if content_type == CONTENT_TYPE_ARROW_STREAM:
        return read_arrow_stream(data)
    return pickle_loads(data)


def arrow_response(table: pa.Table) -> Any:
    """Wrap *table* in a FastAPI ``Response`` carrying the Arrow stream body."""
    from fastapi import Response

    return Response(
        content=write_arrow_stream(table),
        media_type=CONTENT_TYPE_ARROW_STREAM,
    )


def negotiate_response(obj: Any, accept: str | None) -> Any:
    """Return a binary Arrow ``Response`` when the client accepts it.

    When *accept* contains the Arrow stream content type and *obj* is
    tabular, serialize to Arrow IPC. Otherwise return *obj* unchanged so
    FastAPI's default JSON encoder handles it.
    """
    if accept and CONTENT_TYPE_ARROW_STREAM in accept:
        table = to_arrow_table(obj)
        if table is not None:
            return arrow_response(table)
    return obj
