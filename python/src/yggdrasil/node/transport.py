"""Node wire format: Arrow IPC for tabular payloads, yggdrasil pickle for the rest.

The node speaks two content types. Tabular results (Arrow tables, polars/pandas
frames) ride the Arrow IPC stream so the client materializes them zero-copy.
Everything else (scalars, dicts, bytes, exceptions) rides yggdrasil pickle.
"""
from __future__ import annotations

from typing import Any, Iterator

import pyarrow as pa

from yggdrasil.pickle.ser.serde import dumps as ygg_dumps, loads as ygg_loads

CONTENT_TYPE_ARROW_STREAM = "application/vnd.apache.arrow.stream"
CONTENT_TYPE_PICKLE = "application/octet-stream"


def serialize_pickle(obj: Any) -> bytes:
    return ygg_dumps(obj)


def deserialize_pickle(data: bytes) -> Any:
    return ygg_loads(data)


def write_arrow_stream(table: pa.Table) -> Iterator[bytes]:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    yield sink.getvalue().to_pybytes()


def write_arrow_stream_chunked(table: pa.Table, max_chunksize: int = 8192) -> Iterator[bytes]:
    # Emit one IPC message per batch so a large table streams to the client
    # without buffering the whole serialized payload at once.
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=max_chunksize):
            writer.write_batch(batch)
    yield sink.getvalue().to_pybytes()


def read_arrow_stream(data: bytes) -> pa.Table:
    with pa.ipc.open_stream(pa.py_buffer(data)) as reader:
        return reader.read_all()


def is_tabular(obj: Any) -> bool:
    if isinstance(obj, (pa.Table, pa.RecordBatch)):
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
    if isinstance(obj, pa.RecordBatch):
        return pa.Table.from_batches([obj])
    mod = type(obj).__module__.split(".", 1)[0]
    if mod == "polars":
        return obj.to_arrow()
    if mod == "pandas":
        return pa.Table.from_pandas(obj)
    raise TypeError(f"cannot convert {type(obj).__name__} to an Arrow table")


def serialize_result(obj: Any) -> tuple[bytes, str]:
    if is_tabular(obj):
        return b"".join(write_arrow_stream(to_arrow_table(obj))), CONTENT_TYPE_ARROW_STREAM
    return serialize_pickle(obj), CONTENT_TYPE_PICKLE


def deserialize_result(data: bytes, content_type: str) -> Any:
    if content_type and content_type.startswith(CONTENT_TYPE_ARROW_STREAM):
        return read_arrow_stream(data)
    return deserialize_pickle(data)
