"""Backend-agnostic coverage of the streaming-upload commit seam.

When a holder declares ``SUPPORTS_STREAMING_UPLOAD``, an Arrow/Parquet write
spills the encode to a temp file (``_ArrowOutputStreamContext``) and commits it
via ``IO._commit_format_source`` → ``holder._upload_stream(spill)`` — the
payload never lives whole in memory. Append isn't a streaming op, so it falls
back to the materialising ``_commit_format_payload``. These tests pin that
routing without any S3/Databricks backend.
"""
from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.io.parquet_file import ParquetFile
from yggdrasil.path.memory import Memory


class _StreamingHolder(Memory):
    """An in-memory holder that opts into streaming uploads and captures the
    spilled source instead of writing it anywhere."""

    SUPPORTS_STREAMING_UPLOAD = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.streamed_bytes: bytes | None = None
        self.stream_source_type: type | None = None

    def _upload_stream(self, source) -> int:
        self.stream_source_type = type(source)
        self.streamed_bytes = source.read_bytes()
        return len(self.streamed_bytes)


def test_arrow_write_spills_then_streams_intact_bytes():
    from yggdrasil.path.local_path import LocalPath

    holder = _StreamingHolder()
    table = pa.table({"a": pa.array([1, 2, 3], type=pa.int64()), "b": ["x", "y", "z"]})
    ParquetFile(holder=holder, owns_holder=False).write_arrow_table(table)

    # The commit streamed a real on-disk spill file, not an in-memory buffer.
    assert holder.stream_source_type is LocalPath
    assert holder.streamed_bytes is not None
    # ...and the spilled bytes are the exact encoded Parquet.
    back = pq.read_table(io.BytesIO(holder.streamed_bytes))
    assert back.num_rows == 3
    assert back.column("a").to_pylist() == [1, 2, 3]
    assert back.column("b").to_pylist() == ["x", "y", "z"]


def test_commit_format_source_append_does_not_stream():
    # Append → materialise via _commit_format_payload, never _upload_stream
    # (the streaming backends are whole-object overwrite).
    holder = _StreamingHolder()
    source = Memory(binary=b"appended-bytes")
    holder._commit_format_source(source, append=True)

    assert holder.streamed_bytes is None          # streaming path skipped
    assert holder.read_bytes() == b"appended-bytes"  # committed the normal way


def test_non_streaming_holder_keeps_in_memory_commit():
    # A plain Memory holder (no opt-in) takes the unchanged in-memory path —
    # the write lands in the holder, nothing is "streamed".
    holder = Memory()
    table = pa.table({"a": pa.array([1, 2, 3], type=pa.int64())})
    ParquetFile(holder=holder, owns_holder=False).write_arrow_table(table)
    back = pq.read_table(io.BytesIO(holder.read_bytes()))
    assert back.num_rows == 3
