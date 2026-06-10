"""Wire transport for the node — Arrow IPC for tabular, ygg-pickle for the rest.

The node speaks two content types over ``/api/call`` and its result
endpoints: tabular payloads (Arrow ``Table`` / polars / pandas frames) go
out as a zero-copy Arrow IPC stream, everything else rides the yggdrasil
pickle wire format. :func:`serialize_result` picks the format from the
object; :func:`deserialize_result` reverses it from the content type.
"""
from __future__ import annotations

from typing import Any, Iterator

import pyarrow as pa

from yggdrasil.pickle.ser.serde import dumps as ygg_dumps, loads as ygg_loads

CONTENT_TYPE_ARROW_STREAM = "application/vnd.apache.arrow.stream"
CONTENT_TYPE_PICKLE = "application/x-python-pickle"

__all__ = [
    "CONTENT_TYPE_ARROW_STREAM",
    "CONTENT_TYPE_PICKLE",
    "serialize_pickle",
    "deserialize_pickle",
    "serialize_result",
    "deserialize_result",
    "is_tabular",
    "to_arrow_table",
    "write_arrow_stream",
    "write_arrow_stream_chunked",
    "read_arrow_stream",
]


def serialize_pickle(obj: Any) -> bytes:
    return ygg_dumps(obj)


def deserialize_pickle(data: bytes) -> Any:
    return ygg_loads(data)


def is_tabular(obj: Any) -> bool:
    if isinstance(obj, pa.Table):
        return True
    mod = type(obj).__module__.split(".", 1)[0]
    if mod == "polars" and type(obj).__name__ == "DataFrame":
        return True
    if mod == "pandas" and type(obj).__name__ == "DataFrame":
        return True
    return False


def to_arrow_table(obj: Any) -> pa.Table:
    if isinstance(obj, pa.Table):
        return obj
    # polars and pandas both expose to_arrow(); polars returns a Table
    # directly, pandas via pa.Table.from_pandas.
    mod = type(obj).__module__.split(".", 1)[0]
    if mod == "polars":
        return obj.to_arrow()
    if mod == "pandas":
        return pa.Table.from_pandas(obj, preserve_index=False)
    raise TypeError(f"Cannot convert {type(obj).__name__!r} to a pyarrow.Table")


def write_arrow_stream(table: pa.Table) -> Iterator[bytes]:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    yield sink.getvalue().to_pybytes()


def write_arrow_stream_chunked(table: pa.Table, max_chunksize: int) -> Iterator[bytes]:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=max_chunksize):
            writer.write_batch(batch)
    yield sink.getvalue().to_pybytes()


def read_arrow_stream(data: bytes) -> pa.Table:
    with pa.ipc.open_stream(pa.py_buffer(data)) as reader:
        return reader.read_all()


def serialize_result(obj: Any) -> tuple[bytes, str]:
    if is_tabular(obj):
        return b"".join(write_arrow_stream(to_arrow_table(obj))), CONTENT_TYPE_ARROW_STREAM
    return serialize_pickle(obj), CONTENT_TYPE_PICKLE


def deserialize_result(data: bytes, content_type: str) -> Any:
    if content_type.split(";", 1)[0].strip() == CONTENT_TYPE_ARROW_STREAM:
        return read_arrow_stream(data)
    return deserialize_pickle(data)
