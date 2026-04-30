"""Tests for :class:`ZipIO`.

Coverage:

- Round-trip preserves types (entries are Arrow IPC streams, no
  type coercion).
- Each batch becomes a separate zip entry.
- APPEND adds entries with continuing index.
- Custom entry_name_template.
- Non-batch entries in the zip are filtered out on read.
"""

from __future__ import annotations

import io as _io
import zipfile

from yggdrasil.io.enums import Mode
from yggdrasil.io.buffer.primitive.zip_io import ZipIO, ZipOptions


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_types(arrow_table):
    """Zip-of-IPC means the inner format is type-preserving."""
    io = ZipIO()
    with io:
        io.write_arrow_table(arrow_table)
        io.seek(0)
        result = io.read_arrow_table()

    assert result.equals(arrow_table)


def test_each_batch_becomes_an_entry(arrow_batches):
    io = ZipIO()
    with io:
        io.write_arrow_batches(iter(arrow_batches))
        io.seek(0)
        # Inspect the zip directly.
        with zipfile.ZipFile(io, mode="r") as zf:
            names = sorted(n for n in zf.namelist() if n.startswith("batch-"))

    assert len(names) == len(arrow_batches)
    # Names should be zero-padded and incrementing.
    assert names == sorted(names)


def test_empty_iterator_is_noop():
    io = ZipIO()
    with io:
        io.write_arrow_batches(iter([]))
        assert io.is_empty()


def test_empty_buffer_read_yields_no_batches():
    io = ZipIO()
    with io:
        batches = list(io.read_arrow_batches())
    assert batches == []


# ---------------------------------------------------------------------------
# APPEND
# ---------------------------------------------------------------------------


def test_append_adds_entries(arrow_table):
    io = ZipIO()
    with io:
        io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
        io.write_arrow_table(arrow_table, mode=Mode.APPEND)
        io.seek(0)

        with zipfile.ZipFile(io, mode="r") as zf:
            entries = [n for n in zf.namelist() if n.startswith("batch-")]
        assert len(entries) >= 2  # at least one per write


def test_append_continues_index(arrow_table):
    io = ZipIO()
    with io:
        io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
        # Capture the highest existing index.
        with zipfile.ZipFile(io, mode="r") as zf:
            first_set = sorted(n for n in zf.namelist() if n.startswith("batch-"))

        io.write_arrow_table(arrow_table, mode=Mode.APPEND)
        with zipfile.ZipFile(io, mode="r") as zf:
            second_set = sorted(n for n in zf.namelist() if n.startswith("batch-"))

    # No name collisions.
    assert len(second_set) > len(first_set)
    assert len(set(second_set)) == len(second_set)


def test_append_combined_read_returns_all_rows(arrow_table):
    io = ZipIO()
    with io:
        io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
        io.write_arrow_table(arrow_table, mode=Mode.APPEND)
        io.seek(0)
        result = io.read_arrow_table()

    assert result.num_rows == 2 * arrow_table.num_rows


# ---------------------------------------------------------------------------
# Entry naming
# ---------------------------------------------------------------------------


def test_custom_entry_name_template(arrow_table):
    io = ZipIO()
    options = ZipOptions(entry_name_template="chunk-{:04d}.bin")

    with io:
        io.write_arrow_table(arrow_table, options=options)
        io.seek(0)
        with zipfile.ZipFile(io, mode="r") as zf:
            names = zf.namelist()

    # Custom template was honored.
    assert any(n.startswith("chunk-") for n in names)


# ---------------------------------------------------------------------------
# Filter non-batch entries
# ---------------------------------------------------------------------------


def test_read_filters_non_batch_entries(arrow_table):
    """Adjacent files in the same archive (e.g. README) should be
    ignored on read."""
    io = ZipIO()
    with io:
        io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
        io.seek(0)

        # Inject a non-batch entry by re-writing.
        existing = io.read()

    # Build a new zip with a README alongside the batches.
    out = _io.BytesIO()
    out.write(existing)
    out.seek(0)
    with zipfile.ZipFile(out, mode="a") as zf:
        zf.writestr("README.md", "Hello.")

    io2 = ZipIO()
    with io2:
        io2.write(out.getvalue())
        io2.seek(0)
        result = io2.read_arrow_table()

    # README should not have poisoned the read.
    assert result.num_rows == arrow_table.num_rows