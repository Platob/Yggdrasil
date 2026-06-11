"""Arrow IPC + pickle transport for the node server.

Two wire formats cross the node boundary:

* **Arrow IPC stream** (``application/vnd.apache.arrow.stream``) for anything
  tabular — pa.Table, polars DataFrame/LazyFrame. Typed, columnar, zero-copy on
  decode, and streamable batch-by-batch so a big result never lands in RAM as
  one contiguous Python copy.
* **pickle** (``application/x-pickle``) for everything else — scalars, dicts,
  call payloads. cloudpickle when available (handles closures / lambdas for the
  remote-call path), stdlib pickle otherwise.

``serialize_result`` picks the format for you: tabular → Arrow, else pickle.
"""
from __future__ import annotations

import pickle
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc

CONTENT_TYPE_ARROW_STREAM = "application/vnd.apache.arrow.stream"
CONTENT_TYPE_PICKLE = "application/x-pickle"

# cloudpickle round-trips closures / lambdas / locally-defined functions that
# stdlib pickle rejects — the remote-call path leans on that. Fall back to
# stdlib when it isn't installed; scalar payloads pickle fine either way.
try:
    import cloudpickle as _pickle_impl
except ImportError:  # pragma: no cover - cloudpickle is a normal dep here
    _pickle_impl = pickle


def serialize_pickle(obj: Any) -> bytes:
    return _pickle_impl.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_pickle(data: bytes) -> Any:
    # stdlib pickle.loads reads cloudpickle output and vice-versa, so we don't
    # need to know which impl wrote the bytes.
    return pickle.loads(data)


def is_tabular(obj: Any) -> bool:
    """True for the shapes we send as Arrow: pa.Table, polars DataFrame/LazyFrame."""
    if isinstance(obj, pa.Table):
        return True
    # Polars is optional; check by module/name so importing it isn't required
    # just to answer "is this tabular".
    cls = type(obj)
    mod = getattr(cls, "__module__", "")
    if mod.startswith("polars"):
        return cls.__name__ in ("DataFrame", "LazyFrame")
    return False


def to_arrow_table(obj: Any) -> pa.Table:
    """Coerce a tabular object to a pa.Table. Polars goes through its native
    zero-copy ``to_arrow`` bridge; a LazyFrame is collected first."""
    if isinstance(obj, pa.Table):
        return obj
    cls = type(obj)
    name = cls.__name__
    mod = getattr(cls, "__module__", "")
    if mod.startswith("polars"):
        if name == "LazyFrame":
            obj = obj.collect()
        return obj.to_arrow()
    raise TypeError(
        f"to_arrow_table can't convert {name!r} (module {mod!r}). "
        f"Expected pa.Table or a polars DataFrame/LazyFrame. "
        f"Check is_tabular(obj) before calling."
    )


def write_arrow_stream(table: pa.Table) -> Iterator[bytes]:
    """Encode a table as an Arrow IPC stream, yielding one bytes chunk per
    record batch after the schema header."""
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    yield sink.getvalue().to_pybytes()


def write_arrow_stream_chunked(table: pa.Table, max_chunksize: int = 8192) -> Iterator[bytes]:
    """Same wire format as :func:`write_arrow_stream` but re-batches the table
    to ``max_chunksize`` rows so the encoder drains in bounded steps."""
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=max_chunksize):
            writer.write_batch(batch)
    yield sink.getvalue().to_pybytes()


def write_arrow_stream_bytes(table: pa.Table) -> bytes:
    """One contiguous Arrow IPC buffer. Convenient when the caller wants the
    whole encoded result in hand (small results, in-memory spill threshold)."""
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def read_arrow_stream(data: bytes) -> pa.Table:
    """Decode an Arrow IPC stream back into a pa.Table (zero-copy over the
    input buffer)."""
    return ipc.open_stream(pa.py_buffer(data)).read_all()


class _PyChunkSink:
    """Minimal write-only sink the pyarrow IPC writer accepts.

    ``BufferOutputStream.getvalue()`` closes the stream, so it can't be polled
    mid-encode. This sink instead buffers each ``write`` and lets the caller
    drain the accumulated chunks between batches — peak memory is one chunk.
    The writer only needs ``write``/``tell``/``flush``/``closed`` off it.
    """

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self._pos = 0
        self.closed = False

    def write(self, data) -> int:
        b = bytes(data)
        self.chunks.append(b)
        self._pos += len(b)
        return len(b)

    def tell(self) -> int:
        return self._pos

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def drain(self) -> bytes:
        if not self.chunks:
            return b""
        out = b"".join(self.chunks)
        self.chunks.clear()
        return out


def iter_arrow_ipc_stream(batches: Iterator[pa.RecordBatch], schema: pa.Schema) -> Iterator[bytes]:
    """Stream an Arrow IPC encoding of ``batches`` chunk by chunk.

    Peak memory stays near one batch: each ``write_batch`` buffers into the
    sink, we drain it, then move on. This is the heavy/remote result path where
    materialising the whole encoded result would blow up RAM.
    """
    sink = _PyChunkSink()
    writer = ipc.new_stream(sink, schema)
    # The schema header is written eagerly; drain it before the first batch so
    # the consumer gets a well-formed stream prefix immediately.
    header = sink.drain()
    if header:
        yield header
    for batch in batches:
        writer.write_batch(batch)
        chunk = sink.drain()
        if chunk:
            yield chunk
    writer.close()
    tail = sink.drain()
    if tail:
        yield tail


def write_arrow_ipc_file(path: str, batches: Iterator[pa.RecordBatch], schema: pa.Schema) -> int:
    """Spill ``batches`` to an Arrow IPC stream file batch-by-batch.

    Returns the row count written. Used for the disk-spill result path: the
    whole result never sits in RAM, and the file can be re-read with
    ``pa.memory_map`` for a zero-copy second pass.
    """
    rows = 0
    with pa.OSFile(path, "wb") as fh:
        with ipc.new_stream(fh, schema) as writer:
            for batch in batches:
                writer.write_batch(batch)
                rows += batch.num_rows
    return rows


def iter_file_chunks(path: str, chunk_size: int = 65536) -> Iterator[bytes]:
    """Stream a file off disk in fixed-size chunks (for serving a spilled
    Arrow IPC file without loading it whole)."""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            yield chunk


def serialize_result(obj: Any) -> tuple[bytes, str]:
    """Serialize a result, choosing the format by shape.

    Tabular → Arrow IPC stream; everything else → pickle. Returns
    ``(bytes, content_type)`` so the HTTP layer can stamp the right header.
    """
    if is_tabular(obj):
        return write_arrow_stream_bytes(to_arrow_table(obj)), CONTENT_TYPE_ARROW_STREAM
    return serialize_pickle(obj), CONTENT_TYPE_PICKLE


def deserialize_result(data: bytes, content_type: str) -> Any:
    """Inverse of :func:`serialize_result`, dispatching on the content type."""
    if content_type == CONTENT_TYPE_ARROW_STREAM:
        return read_arrow_stream(data)
    if content_type == CONTENT_TYPE_PICKLE:
        return deserialize_pickle(data)
    raise ValueError(
        f"Unknown result content-type {content_type!r}. "
        f"Expected {CONTENT_TYPE_ARROW_STREAM!r} (Arrow) or "
        f"{CONTENT_TYPE_PICKLE!r} (pickle)."
    )
