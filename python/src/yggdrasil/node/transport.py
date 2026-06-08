"""Wire transport for ``yggdrasil.node``.

Two formats move over the wire:

- **Arrow IPC stream** (``application/vnd.apache.arrow.stream``) for tabular
  payloads (``pa.Table`` / ``pl.DataFrame`` / ``pd.DataFrame``). Columnar,
  zero-copy on read, the fast path for bulk data.
- **yggdrasil pickle** (``application/x-yggdrasil-pickle``) for everything
  else — scalars, dicts, bytes, arbitrary Python objects. Built on
  :func:`yggdrasil.pickle.ser.serde.dumps` / ``loads``.

``serialize_result`` auto-picks: tabular → Arrow stream, else pickle.
"""
from __future__ import annotations

from typing import Any, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc

from yggdrasil.pickle.ser.serde import dumps as _ygg_dumps, loads as _ygg_loads

__all__ = [
    "CONTENT_TYPE_ARROW_STREAM",
    "CONTENT_TYPE_PICKLE",
    "serialize_pickle",
    "deserialize_pickle",
    "write_arrow_stream",
    "write_arrow_stream_chunked",
    "read_arrow_stream",
    "is_tabular",
    "to_arrow_table",
    "deserialize_result",
    "serialize_result",
]

CONTENT_TYPE_ARROW_STREAM = "application/vnd.apache.arrow.stream"
CONTENT_TYPE_PICKLE = "application/x-yggdrasil-pickle"


# ---------------------------------------------------------------------------
# pickle
# ---------------------------------------------------------------------------

def serialize_pickle(obj: Any) -> bytes:
    """Serialize *obj* to yggdrasil-pickle bytes."""
    return _ygg_dumps(obj)


def deserialize_pickle(data: bytes) -> Any:
    """Inverse of :func:`serialize_pickle`."""
    return _ygg_loads(data)


# ---------------------------------------------------------------------------
# Arrow IPC stream
# ---------------------------------------------------------------------------

def write_arrow_stream(table: pa.Table) -> Iterator[bytes]:
    """Yield the Arrow IPC stream framing for *table* as one buffer.

    Single-shot: the whole table serializes into one ``pyarrow`` sink, then
    the bytes are yielded once. Use :func:`write_arrow_stream_chunked` when
    the consumer wants incremental record-batch framing.
    """
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    yield sink.getvalue().to_pybytes()


def write_arrow_stream_chunked(
    table: pa.Table, *, max_chunksize: int = 8192
) -> Iterator[bytes]:
    """Yield the Arrow IPC stream for *table*, batched at *max_chunksize* rows.

    Splits into record batches so a streaming HTTP response can flush partial
    results without materializing the full encoded buffer up front. The whole
    encoded stream still lands in one ``pyarrow`` sink (pyarrow owns the
    framing) and is yielded once it is complete — the chunking governs the
    record-batch boundaries inside the stream, not the Python-level yields.
    """
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=max_chunksize):
            writer.write_batch(batch)
    yield sink.getvalue().to_pybytes()


def read_arrow_stream(data: bytes) -> pa.Table:
    """Read an Arrow IPC stream back into a :class:`pa.Table`."""
    with ipc.open_stream(pa.py_buffer(data)) as reader:
        return reader.read_all()


# ---------------------------------------------------------------------------
# tabular detection + coercion
# ---------------------------------------------------------------------------

def is_tabular(obj: Any) -> bool:
    """True if *obj* is a ``pa.Table`` or a ``pl.DataFrame``.

    Polars is checked by module/class name so importing this module never
    drags in polars when it isn't installed.
    """
    if isinstance(obj, pa.Table):
        return True
    cls = type(obj)
    mod = getattr(cls, "__module__", "")
    return mod.startswith("polars") and cls.__name__ == "DataFrame"


def to_arrow_table(obj: Any) -> pa.Table:
    """Coerce *obj* (Arrow / polars / pandas) into a :class:`pa.Table`."""
    if isinstance(obj, pa.Table):
        return obj
    # polars.DataFrame → Arrow is zero-copy for primitive columns.
    if hasattr(obj, "to_arrow"):
        result = obj.to_arrow()
        if isinstance(result, pa.Table):
            return result
        return pa.table(result)
    # pandas.DataFrame
    cls = type(obj)
    if getattr(cls, "__module__", "").startswith("pandas"):
        return pa.Table.from_pandas(obj)
    raise TypeError(
        f"to_arrow_table cannot convert {cls.__module__}.{cls.__name__}; "
        f"expected a pyarrow.Table, polars.DataFrame, or pandas.DataFrame."
    )


# ---------------------------------------------------------------------------
# auto-dispatch
# ---------------------------------------------------------------------------

def serialize_result(obj: Any) -> tuple[bytes, str]:
    """Serialize *obj* picking the best format.

    Tabular objects → Arrow IPC stream; everything else → yggdrasil pickle.
    Returns ``(payload, content_type)``.
    """
    if is_tabular(obj):
        table = to_arrow_table(obj)
        return b"".join(write_arrow_stream(table)), CONTENT_TYPE_ARROW_STREAM
    return serialize_pickle(obj), CONTENT_TYPE_PICKLE


def deserialize_result(data: bytes, content_type: str) -> Any:
    """Inverse of :func:`serialize_result`, dispatched on *content_type*."""
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    if ct == CONTENT_TYPE_ARROW_STREAM:
        return read_arrow_stream(data)
    if ct == CONTENT_TYPE_PICKLE:
        return deserialize_pickle(data)
    # Unknown / generic binary: fall back to pickle (our own wire format
    # carries a magic header, so a mislabeled pickle still decodes).
    return deserialize_pickle(data)
