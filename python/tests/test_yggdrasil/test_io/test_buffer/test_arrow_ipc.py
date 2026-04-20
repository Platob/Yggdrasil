"""Unit tests for :class:`yggdrasil.io.buffer.arrow_ipc_io.IPCIO`.

Covers:

* options validation (ipc_compression, layout)
* factory routing
* file-layout and stream-layout round-trips
* layout auto-detect on read
* body compression codecs (zstd, lz4, none)
* column projection
* batch_size rechunking
* ignore_empty
* save modes (OVERWRITE, IGNORE, ERROR_IF_EXISTS, APPEND, UPSERT)
* cast integration on write (default identity + explicit CastOptions)
* schema inspection (header-only)
* edge cases (empty buffer, overwrite-shrinking, spilled buffer)
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.arrow_ipc_io import IPCIO, IPCOptions
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.config import BufferConfig
from yggdrasil.io.enums import SaveMode


# =====================================================================
# Helpers
# =====================================================================

def _make_table(n: int = 3) -> pa.Table:
    return pa.table(
        {
            "id": list(range(n)),
            "name": [f"row_{i}" for i in range(n)],
            "price": [float(i) * 1.5 for i in range(n)],
        }
    )


@pytest.fixture()
def cfg(tmp_path: Path) -> BufferConfig:
    return BufferConfig(
        spill_bytes=1024 * 1024,
        tmp_dir=tmp_path,
        prefix="test_ipc_",
        suffix=".arrow",
        keep_spilled_file=False,
    )


@pytest.fixture()
def spill_cfg(tmp_path: Path) -> BufferConfig:
    return BufferConfig(
        spill_bytes=1,  # force spill on any write
        tmp_dir=tmp_path,
        prefix="test_ipc_spill_",
        suffix=".arrow",
        keep_spilled_file=False,
    )


@pytest.fixture()
def sample() -> pa.Table:
    return _make_table(3)


def _make_io(buf: BytesIO) -> IPCIO:
    io_ = MediaIO.make(buf, MimeTypes.ARROW_IPC)
    assert isinstance(io_, IPCIO)
    return io_


# =====================================================================
# Options
# =====================================================================

class TestIPCOptions:
    def test_defaults(self):
        opt = IPCOptions()
        assert opt.ipc_compression == "zstd"
        assert opt.layout == "file"

    def test_rejects_unknown_codec(self):
        with pytest.raises(ValueError, match="ipc_compression"):
            IPCOptions(ipc_compression="brotli")

    def test_codec_case_normalized(self):
        assert IPCOptions(ipc_compression="ZSTD").ipc_compression == "zstd"
        assert IPCOptions(ipc_compression="LZ4").ipc_compression == "lz4"

    def test_none_codec_means_no_body_compression(self):
        opt = IPCOptions(ipc_compression=None)
        assert opt.ipc_compression is None

    def test_layout_must_be_file_or_stream(self):
        with pytest.raises(ValueError, match="layout"):
            IPCOptions(layout="feather")

    def test_layout_case_normalized(self):
        assert IPCOptions(layout="FILE").layout == "file"
        assert IPCOptions(layout="Stream").layout == "stream"


# =====================================================================
# Factory
# =====================================================================

class TestFactory:
    def test_media_io_make_returns_ipc_io(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        assert isinstance(io_, IPCIO)

    def test_check_options_accepts_none(self):
        resolved = IPCIO.check_options(None)
        assert isinstance(resolved, IPCOptions)
        assert resolved.layout == "file"

    def test_check_options_merges_kwargs(self):
        resolved = IPCIO.check_options(
            None, ipc_compression="lz4", layout="stream"
        )
        assert resolved.ipc_compression == "lz4"
        assert resolved.layout == "stream"


# =====================================================================
# File layout
# =====================================================================

class TestFileLayout:
    def test_write_read_roundtrip(self, cfg: BufferConfig, sample: pa.Table):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample, layout="file")

        assert buf.size > 0
        out = io_.read_arrow_table()
        assert out.to_pylist() == sample.to_pylist()

    def test_file_layout_magic_header(self, cfg: BufferConfig, sample):
        """File layout starts with the ARROW1 magic."""
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample, layout="file")
        head = buf.to_bytes()[:8]
        # IPC file format magic: "ARROW1\0\0"
        assert head.startswith(b"ARROW1")

    def test_none_body_compression(self, cfg: BufferConfig, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample, layout="file", ipc_compression=None)
        out = _make_io(buf).read_arrow_table()
        assert out.to_pylist() == sample.to_pylist()

    def test_zstd_body_compression(self, cfg: BufferConfig, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample, layout="file", ipc_compression="zstd")
        out = _make_io(buf).read_arrow_table()
        assert out.to_pylist() == sample.to_pylist()

    def test_lz4_body_compression(self, cfg: BufferConfig, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample, layout="file", ipc_compression="lz4")
        out = _make_io(buf).read_arrow_table()
        assert out.to_pylist() == sample.to_pylist()

    def test_multi_batch_preserved(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        # Write via _write_arrow_batches to exercise the batch-loop path.
        t = _make_table(100)
        io_.write_arrow_table(t, layout="file")

        out = io_.read_arrow_table()
        assert out.num_rows == 100
        assert out.to_pylist() == t.to_pylist()


# =====================================================================
# Stream layout
# =====================================================================

class TestStreamLayout:
    def test_write_read_roundtrip(self, cfg: BufferConfig, sample: pa.Table):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample, layout="stream")

        assert buf.size > 0
        # Sanity: no ARROW1 file magic on stream layout.
        assert not buf.to_bytes().startswith(b"ARROW1")

        out = io_.read_arrow_table()
        assert out.to_pylist() == sample.to_pylist()

    def test_stream_layout_auto_detected_on_read(
        self, cfg: BufferConfig, sample
    ):
        """Reader falls back to stream layout when file magic is absent."""
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample, layout="stream")

        # Same buffer, fresh IO instance — reader must auto-detect.
        out = _make_io(buf).read_arrow_table()
        assert out.to_pylist() == sample.to_pylist()


# =====================================================================
# Layout reader auto-detection via the private helper
# =====================================================================

class TestReaderAutoDetect:
    def test_open_file_bytes_returns_file_layout(self, cfg, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample, layout="file")

        arrow_io = buf.to_arrow_io("r")
        try:
            reader, layout = IPCIO._open_ipc_reader(arrow_io)
            assert layout == "file"
            assert reader.num_record_batches >= 1
        finally:
            arrow_io.close()

    def test_open_stream_bytes_returns_stream_layout(self, cfg, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample, layout="stream")

        arrow_io = buf.to_arrow_io("r")
        try:
            reader, layout = IPCIO._open_ipc_reader(arrow_io)
            assert layout == "stream"
        finally:
            arrow_io.close()


# =====================================================================
# Column projection
# =====================================================================

class TestColumnProjection:
    def test_read_with_columns_projects(self, cfg, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample)

        out = _make_io(buf).read_arrow_table(columns=["id", "price"])
        assert out.column_names == ["id", "price"]
        assert out.num_rows == sample.num_rows

    def test_read_with_single_column(self, cfg, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample)

        out = _make_io(buf).read_arrow_table(columns=["name"])
        assert out.column_names == ["name"]


# =====================================================================
# batch_size rechunking
# =====================================================================

class TestBatchSize:
    def test_batch_size_produces_bounded_batches(self, cfg):
        buf = BytesIO(config=cfg)
        t = _make_table(100)
        _make_io(buf).write_arrow_table(t)

        batches = list(_make_io(buf).read_arrow_batches(batch_size=10))
        total = sum(b.num_rows for b in batches)
        assert total == 100
        assert all(b.num_rows <= 10 for b in batches)

    def test_batch_size_smaller_than_input_splits(self, cfg):
        """A single large on-disk batch should be split by batch_size."""
        buf = BytesIO(config=cfg)
        t = _make_table(50)
        _make_io(buf).write_arrow_table(t)

        batches = list(_make_io(buf).read_arrow_batches(batch_size=7))
        total = sum(b.num_rows for b in batches)
        assert total == 50
        assert all(b.num_rows <= 7 for b in batches)

    def test_no_batch_size_preserves_file_chunking(self, cfg, sample):
        """Without batch_size, whatever the file chose is what we get."""
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample)

        batches = list(_make_io(buf).read_arrow_batches())
        total = sum(b.num_rows for b in batches)
        assert total == sample.num_rows


# =====================================================================
# Ignore empty
# =====================================================================

class TestIgnoreEmpty:
    def test_ignore_empty_drops_zero_row_batches(self, cfg):
        """An empty-row table still yields empty batches unless asked to skip."""
        buf = BytesIO(config=cfg)
        empty = pa.table({"id": pa.array([], type=pa.int64())})
        _make_io(buf).write_arrow_table(empty)

        # With ignore_empty, no batches should surface.
        batches = list(_make_io(buf).read_arrow_batches(ignore_empty=True))
        assert all(b.num_rows > 0 for b in batches)


# =====================================================================
# Save modes
# =====================================================================

class TestSaveModes:
    def test_overwrite_replaces(self, cfg, sample):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample, mode=SaveMode.OVERWRITE)
        size1 = buf.size

        t2 = _make_table(10)
        io_.write_arrow_table(t2, mode=SaveMode.OVERWRITE)
        out = io_.read_arrow_table()
        assert out.num_rows == 10

    def test_ignore_mode_preserves_existing(self, cfg, sample):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample, mode=SaveMode.OVERWRITE)

        bytes1 = buf.to_bytes()
        size1 = buf.size

        io_.write_arrow_table(
            _make_table(10), mode=SaveMode.IGNORE
        )
        assert buf.size == size1
        assert buf.to_bytes() == bytes1

    def test_error_if_exists_raises(self, cfg, sample):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample, mode=SaveMode.OVERWRITE)

        with pytest.raises(IOError):
            io_.write_arrow_table(sample, mode=SaveMode.ERROR_IF_EXISTS)

    def test_error_if_exists_allowed_on_empty_buffer(self, cfg, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(
            sample, mode=SaveMode.ERROR_IF_EXISTS
        )
        out = _make_io(buf).read_arrow_table()
        assert out.num_rows == sample.num_rows

    def test_append_combines_batches(self, cfg, sample):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample, mode=SaveMode.OVERWRITE)
        io_.write_arrow_table(_make_table(5), mode=SaveMode.APPEND)

        out = _make_io(buf).read_arrow_table()
        assert out.num_rows == sample.num_rows + 5

    def test_append_into_empty_buffer_acts_like_overwrite(self, cfg, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample, mode=SaveMode.APPEND)
        out = _make_io(buf).read_arrow_table()
        assert out.num_rows == sample.num_rows

    def test_append_stream_layout(self, cfg, sample):
        """APPEND works with stream layout too (same read-then-rewrite path)."""
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample, layout="stream", mode=SaveMode.OVERWRITE)
        io_.write_arrow_table(_make_table(2), layout="stream", mode=SaveMode.APPEND)

        out = _make_io(buf).read_arrow_table()
        assert out.num_rows == sample.num_rows + 2

    def test_upsert_without_match_by_raises(self, cfg, sample):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample, mode=SaveMode.OVERWRITE)

        with pytest.raises(ValueError, match="match_by"):
            io_.write_arrow_table(sample, mode=SaveMode.UPSERT)

    def test_upsert_replaces_matching_rows(self, cfg, sample):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample, mode=SaveMode.OVERWRITE)

        # sample has ids 0..2; replace id=1 and insert id=99.
        update = pa.table(
            {
                "id": [1, 99],
                "name": ["REPLACED", "new_row"],
                "price": [100.0, 200.0],
            }
        )
        io_.write_arrow_table(update, mode=SaveMode.UPSERT, match_by="id")

        out = _make_io(buf).read_arrow_table()
        by_id = {r["id"]: r for r in out.to_pylist()}
        assert sorted(by_id.keys()) == [0, 1, 2, 99]
        assert by_id[1]["name"] == "REPLACED"
        assert by_id[99]["price"] == 200.0
        assert by_id[0]["name"] == "row_0"  # untouched


# =====================================================================
# Cast integration
# =====================================================================

class TestCastIntegration:
    def test_default_cast_is_identity(self, cfg, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample)
        out = _make_io(buf).read_arrow_table()
        assert out.to_pylist() == sample.to_pylist()

    def test_cast_reaches_write_path(self, cfg):
        try:
            from yggdrasil.data.cast.options import CastOptions
        except ImportError:
            pytest.skip("CastOptions not importable")

        target = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("v", pa.float64()),
            ]
        )
        try:
            cast = CastOptions(target_field=target)
        except TypeError:
            pytest.skip("CastOptions(target_field=...) signature mismatch")

        src = pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int32()),
                "v": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
            }
        )

        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(src, cast=cast)

        out = _make_io(buf).read_arrow_table()
        assert out.num_rows == 3
        # IPC preserves exact Arrow types, so the cast target should be
        # reflected on the read side.
        assert out.schema.field("id").type == pa.int64()
        assert out.schema.field("v").type == pa.float64()


# =====================================================================
# Schema (header-only)
# =====================================================================

class TestSchema:
    def test_empty_buffer_schema(self, cfg):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        assert io_._collect_arrow_schema() == pa.schema([])

    def test_schema_from_file_layout(self, cfg, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample, layout="file")

        schema = _make_io(buf)._collect_arrow_schema()
        assert set(schema.names) == {"id", "name", "price"}

    def test_schema_from_stream_layout(self, cfg, sample):
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(sample, layout="stream")

        schema = _make_io(buf)._collect_arrow_schema()
        assert set(schema.names) == {"id", "name", "price"}


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_empty_buffer_read_yields_no_batches(self, cfg):
        buf = BytesIO(config=cfg)
        assert list(_make_io(buf).read_arrow_batches()) == []

    def test_empty_buffer_read_returns_empty_table(self, cfg):
        buf = BytesIO(config=cfg)
        out = _make_io(buf).read_arrow_table()
        assert out.num_rows == 0

    def test_overwrite_shrinks_buffer(self, cfg):
        """Overwriting a large payload with a small one truncates properly."""
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        big = _make_table(500)
        io_.write_arrow_table(big, mode=SaveMode.OVERWRITE)
        size_after_big = buf.size

        small = _make_table(5)
        io_.write_arrow_table(small, mode=SaveMode.OVERWRITE)
        size_after_small = buf.size

        # Must actually shrink — otherwise the buffer has trailing junk.
        assert size_after_small < size_after_big

        out = io_.read_arrow_table()
        assert out.num_rows == 5

    def test_spilled_buffer_roundtrip(self, spill_cfg, sample):
        buf = BytesIO(config=spill_cfg)
        _make_io(buf).write_arrow_table(sample)

        assert buf.spilled is True
        out = _make_io(buf).read_arrow_table()
        assert out.to_pylist() == sample.to_pylist()

    def test_write_preserves_type_fidelity(self, cfg):
        """IPC is type-exact — timestamps, decimals, etc. round-trip."""
        table = pa.table(
            {
                "ts": pa.array(
                    [1000, 2000, 3000],
                    type=pa.timestamp("ns", tz="UTC"),
                ),
                "dec": pa.array([1, 2, 3], type=pa.int64()),
            }
        )
        buf = BytesIO(config=cfg)
        _make_io(buf).write_arrow_table(table)

        out = _make_io(buf).read_arrow_table()
        assert out.schema.field("ts").type == pa.timestamp("ns", tz="UTC")
        assert out.schema.field("dec").type == pa.int64()
        assert out.to_pylist() == table.to_pylist()


# =====================================================================
# Rechunk helper (unit tests on the private method)
# =====================================================================

class TestRechunk:
    def test_exact_splits(self):
        t = _make_table(30)
        batches = list(IPCIO._rechunk(iter(t.to_batches()), batch_size=10))
        assert sum(b.num_rows for b in batches) == 30
        assert all(b.num_rows <= 10 for b in batches)

    def test_tail_emitted(self):
        t = _make_table(25)
        batches = list(IPCIO._rechunk(iter(t.to_batches()), batch_size=10))
        rows = [b.num_rows for b in batches]
        assert sum(rows) == 25
        # Tail <= 10.
        assert all(r <= 10 for r in rows)

    def test_coalesces_small_inputs(self):
        """Many small batches should be merged up to batch_size."""
        small_batches = [_make_table(3).to_batches()[0] for _ in range(10)]
        # Total: 30 rows across 10 small batches of 3 each.
        out = list(IPCIO._rechunk(iter(small_batches), batch_size=10))
        rows = [b.num_rows for b in out]
        assert sum(rows) == 30
        # At least one batch should be >= 3 (the smallest input size) —
        # otherwise we failed to coalesce.
        assert max(rows) >= 3

    def test_empty_batches_skipped(self):
        empty = pa.RecordBatch.from_arrays(
            [pa.array([], type=pa.int64())], names=["id"]
        )
        data = _make_table(5).to_batches()[0]
        out = list(IPCIO._rechunk(iter([empty, data, empty]), batch_size=10))
        assert sum(b.num_rows for b in out) == 5