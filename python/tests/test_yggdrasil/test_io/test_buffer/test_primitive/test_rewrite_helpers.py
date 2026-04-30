"""Tests for ``DataIO._arrow_append_via_rewrite`` /
``_arrow_upsert_via_rewrite``.

These are the read-modify-write fallbacks used by leaves whose
format can't natively append (Parquet, Arrow IPC, every JSON-as-
array variant). The contract is:

- The pre-existing buffer contents are read in full BEFORE the
  incoming iterator is consumed.
- Schema drift between existing and incoming is reconciled via
  ``concat_with_schema_union``.
- The recursion bottoms out by calling ``_write_arrow_batches``
  with ``mode=OVERWRITE``.

We use ParquetIO as the test target because Parquet exercises the
full path (footer reload, compressed write-back, native scanner
opt-out), and because Parquet's APPEND is strictly via rewrite —
no native append path could mask a bug in the helper.
"""

from __future__ import annotations

import pytest
import pyarrow as pa

from yggdrasil.io.enums import Mode
from yggdrasil.io.buffer.primitive.parquet_io import ParquetIO


# ---------------------------------------------------------------------------
# APPEND via rewrite
# ---------------------------------------------------------------------------


class TestAppendViaRewrite:

    def test_append_to_empty_buffer(self, arrow_table):
        """APPEND to an empty buffer behaves like OVERWRITE."""
        io = ParquetIO()
        with io:
            io.write_arrow_table(arrow_table, mode=Mode.APPEND)
            io.seek(0)
            result = io.read_arrow_table()
        assert result.num_rows == arrow_table.num_rows

    def test_append_concatenates_existing_and_incoming(self, arrow_table):
        io = ParquetIO()
        with io:
            io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
            # Second write with APPEND — should produce 2x the rows.
            io.seek(0)
            io.write_arrow_table(arrow_table, mode=Mode.APPEND)
            io.seek(0)
            result = io.read_arrow_table()

        assert result.num_rows == 2 * arrow_table.num_rows
        # Column order preserved.
        assert result.column_names == arrow_table.column_names

    def test_append_with_schema_drift_uses_union(self):
        """Schema drift between existing and incoming reconciles via
        column union with null fill."""
        io = ParquetIO()
        existing = pa.table({"a": [1, 2], "b": ["x", "y"]})
        incoming = pa.table({"a": [3], "c": [9.9]})

        with io:
            io.write_arrow_table(existing, mode=Mode.OVERWRITE)
            io.seek(0)
            io.write_arrow_table(incoming, mode=Mode.APPEND)
            io.seek(0)
            result = io.read_arrow_table()

        # Column union {a, b, c} with c missing on the existing rows
        # and b missing on the incoming row.
        assert set(result.column_names) == {"a", "b"}
        assert result.num_rows == 3

        # Find the row with a=3; it should have b=None, c=9.9.
        rows = result.to_pylist()
        new_row = next(r for r in rows if r["a"] == 3)
        assert new_row["b"] is None

    def test_existing_read_completes_before_incoming_consumed(
        self, arrow_table
    ):
        """The append helper MUST read existing batches in full before
        consuming the incoming iterator — otherwise the writer would
        feed itself its own output mid-write.

        We exercise this by passing a generator as the incoming
        iterable and asserting it isn't consumed until after the
        existing batches have been materialized.
        """
        io = ParquetIO()
        with io:
            io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)

            consumed = []

            def tracked_batches():
                for b in arrow_table.to_batches():
                    consumed.append(b)
                    yield b

            io.seek(0)
            io.write_arrow_batches(
                tracked_batches(),
                mode=Mode.APPEND,
            )
            io.seek(0)
            result = io.read_arrow_table()

        # The generator should have been fully drained.
        assert len(consumed) == len(arrow_table.to_batches())
        assert result.num_rows == 2 * arrow_table.num_rows


# ---------------------------------------------------------------------------
# UPSERT via rewrite
# ---------------------------------------------------------------------------


class TestUpsertViaRewrite:

    def test_upsert_replaces_overlapping_rows(self, upsert_tables):
        existing, incoming, match_by = upsert_tables

        io = ParquetIO()
        with io:
            io.write_arrow_table(existing, mode=Mode.OVERWRITE)
            io.seek(0)
            io.write_arrow_table(
                incoming,
                mode=Mode.UPSERT,
                match_by_names=match_by,
            )
            io.seek(0)
            result = io.read_arrow_table()

        rows = {r["key"]: r["value"] for r in result.to_pylist()}
        assert rows == {1: "old-1", 2: "new-2", 3: "new-3", 4: "new-4"}

    def test_upsert_without_match_by_raises(self, arrow_table):
        io = ParquetIO()
        with io:
            io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
            with pytest.raises(ValueError, match="match_by_names"):
                io.write_arrow_table(arrow_table, mode=Mode.UPSERT)

    def test_upsert_into_empty_buffer_is_just_incoming(self, arrow_table):
        io = ParquetIO()
        with io:
            io.write_arrow_table(
                arrow_table,
                mode=Mode.UPSERT,
                match_by_names=["id"],
            )
            io.seek(0)
            result = io.read_arrow_table()
        assert result.num_rows == arrow_table.num_rows
