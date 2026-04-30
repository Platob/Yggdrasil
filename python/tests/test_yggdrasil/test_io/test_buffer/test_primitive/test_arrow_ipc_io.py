"""Tests for :class:`ArrowIPCIO`.

Coverage:

- Round-trip preserves types exactly (IPC's selling point).
- Cached reader is invalidated on write, refreshed on next read.
- Write of empty iterator is a no-op.
- APPEND goes through ``_arrow_append_via_rewrite``.
- Compression knob plumbs through to IpcWriteOptions.
- Native scanner gating.
"""

from __future__ import annotations

import pytest
import pyarrow as pa

from yggdrasil.io.enums import Mode
from yggdrasil.io.buffer.primitive.arrow_ipc_io import ArrowIPCIO, ArrowIPCOptions


# ---------------------------------------------------------------------------
# Round-trip — IPC preserves types exactly
# ---------------------------------------------------------------------------


def test_round_trip_preserves_types(arrow_table):
    io = ArrowIPCIO()
    with io:
        io.write_arrow_table(arrow_table)
        io.seek(0)
        result = io.read_arrow_table()

    # IPC is the no-translation format; this should be a literal
    # round-trip of types and values.
    assert result.equals(arrow_table)


def test_round_trip_via_batches(arrow_batches):
    io = ArrowIPCIO()
    with io:
        io.write_arrow_batches(iter(arrow_batches))
        io.seek(0)
        result = list(io.read_arrow_batches())

    # Same number of batches written = same number read (IPC
    # preserves batch boundaries).
    assert len(result) == len(arrow_batches)
    reassembled = pa.Table.from_batches(result)
    expected = pa.Table.from_batches(arrow_batches)
    assert reassembled.equals(expected)


def test_empty_iterator_write_is_noop():
    """Writing an empty iterator should leave the buffer empty
    (no header-only IPC file)."""
    io = ArrowIPCIO()
    with io:
        io.write_arrow_batches(iter([]))
        assert io.is_empty()


def test_empty_buffer_read_yields_no_batches():
    io = ArrowIPCIO()
    with io:
        batches = list(io.read_arrow_batches())
    assert batches == []


def test_empty_buffer_collect_schema_returns_empty():
    io = ArrowIPCIO()
    with io:
        schema = io.collect_schema()
    assert schema.is_empty() or len(schema) == 0


# ---------------------------------------------------------------------------
# Cached reader lifecycle
# ---------------------------------------------------------------------------


class TestCachedReader:

    def test_reader_dropped_on_write(self, arrow_table):
        """The cached reader must be invalidated before a write —
        otherwise it'd hold stale memoryviews into truncated bytes."""
        io = ArrowIPCIO()
        with io:
            io.write_arrow_table(arrow_table)
            io.seek(0)
            _ = io.reader  # populate cache
            assert io._reader is not None

            # Write again — this should drop the reader.
            io.write_arrow_table(arrow_table)
            assert io._reader is None

    def test_reader_dropped_on_release(self, arrow_table, tmp_path):
        """``_before_release`` drops the reader before buffer cleanup."""
        io = ArrowIPCIO()
        with io:
            io.write_arrow_table(arrow_table)
            io.seek(0)
            _ = io.reader
            assert io._reader is not None

        # After context exit, _before_release should have run.
        assert io._reader is None

    def test_reader_property_raises_on_closed(self):
        io = ArrowIPCIO(auto_open=False)
        # Don't open it.
        with pytest.raises(ValueError, match="closed"):
            _ = io.reader


# ---------------------------------------------------------------------------
# Compression option plumbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression", [None, "lz4", "zstd"])
def test_compression_round_trip(arrow_table, compression):
    """Each supported compression value should round-trip."""
    io = ArrowIPCIO()
    options = ArrowIPCOptions(compression=compression)

    with io:
        io.write_arrow_table(arrow_table, options=options)
        io.seek(0)
        result = io.read_arrow_table(options=options)

    assert result.equals(arrow_table)


# ---------------------------------------------------------------------------
# APPEND — via rewrite
# ---------------------------------------------------------------------------


def test_append_via_rewrite_doubles_rows(arrow_table):
    io = ArrowIPCIO()
    with io:
        io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
        io.write_arrow_table(arrow_table, mode=Mode.APPEND)
        result = io.read_arrow_table()

    assert result.num_rows == 2 * arrow_table.num_rows


def test_arrow_dataset(arrow_table):
    io = ArrowIPCIO()
    with io:
        io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
        io.write_arrow_table(arrow_table, mode=Mode.APPEND)
        ds = io.read_arrow_dataset()

        result = ds.to_table()
        result2 = io.read_arrow_dataset().to_table()
    assert io.closed

    assert result.num_rows == 2 * arrow_table.num_rows
    assert result.equals(result2)

# ---------------------------------------------------------------------------
# Native scanner gating
# ---------------------------------------------------------------------------


class TestNativeScannerGating:

    def test_no_path_blocks_native(self, arrow_table):
        io = ArrowIPCIO()
        with io:
            io.write_arrow_table(arrow_table)
            options = io.check_options()
            assert not io._can_use_native_scanner(options)

    def test_target_field_blocks_native(self, arrow_table):
        io = ArrowIPCIO()
        with io:
            io.write_arrow_table(arrow_table)
            field = arrow_table.schema.field(0)
            options = io.check_options(target_field=field)
            assert not io._can_use_native_scanner(options)
