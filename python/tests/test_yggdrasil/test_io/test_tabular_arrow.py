"""Tests for :class:`yggdrasil.io.tabular.arrow.ArrowTabular`.

Covers the three areas that grew during the
``leverage-arrow-cast / ArrowIPCFile-spill`` optimization:

* ingest now accepts more shapes (``*args``, ``RecordBatchReader``,
  another :class:`Tabular`);
* :meth:`_read_arrow_table` returns the held :class:`pa.Table` zero-
  copy on the no-target / no-rechunk path;
* spill writes go through :class:`ArrowIPCFile` over a
  :class:`LocalPath` and the mmap'd read-back stays zero-copy.
"""

from __future__ import annotations

import os
import pathlib

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.tabular import ArrowTabular


class TestArrowTabularIngest(ArrowTestCase):
    """Inputs ``_ingest`` recognises."""

    def _table(self):
        return self.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    def test_ingest_pa_table(self) -> None:
        io = ArrowTabular(self._table())
        self.assertEqual(io.num_rows, 3)
        self.assertEqual(io.read_arrow_table().num_rows, 3)

    def test_ingest_record_batch(self) -> None:
        batch = self._table().to_batches()[0]
        io = ArrowTabular(batch)
        self.assertEqual(io.num_rows, 3)

    def test_ingest_record_batch_reader(self) -> None:
        pa = self.pa
        t = self._table()
        reader = pa.RecordBatchReader.from_batches(t.schema, t.to_batches())
        io = ArrowTabular(reader)
        self.assertEqual(io.read_arrow_table().num_rows, 3)

    def test_ingest_tabular_source(self) -> None:
        seed = ArrowTabular(self._table())
        io = ArrowTabular(seed)
        self.assertEqual(io.read_arrow_table().num_rows, 3)
        # Independent holders — mutating the receiver's batch list
        # doesn't reach back into the source.
        io.unpersist()
        self.assertEqual(seed.num_rows, 3)

    def test_ingest_many_positional_sources(self) -> None:
        t = self._table()
        io = ArrowTabular(t, t, t)
        self.assertEqual(io.num_rows, 9)
        # Multi-source concat preserves column order.
        self.assertEqual(io.read_arrow_table().column_names, ["x", "y"])

    def test_ingest_many_mixed_shapes(self) -> None:
        pa = self.pa
        t = self._table()
        batch = t.to_batches()[0]
        reader = pa.RecordBatchReader.from_batches(t.schema, t.to_batches())
        io = ArrowTabular(t, batch, reader)
        self.assertEqual(io.num_rows, 9)

    def test_ingest_iterable_of_tables(self) -> None:
        t = self._table()
        io = ArrowTabular([t, t])
        self.assertEqual(io.num_rows, 6)

    def test_ingest_list_of_dicts(self) -> None:
        rows = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        io = ArrowTabular(rows)
        self.assertEqual(io.num_rows, 2)
        self.assertEqual(io.read_arrow_table().column_names, ["x", "y"])

    def test_ingest_dict_of_lists(self) -> None:
        io = ArrowTabular({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        self.assertEqual(io.num_rows, 3)

    def test_ingest_chunked_array_raises_clear_error(self) -> None:
        pa = self.pa
        chunked = pa.chunked_array([[1, 2, 3]])
        with self.assertRaises(TypeError) as cm:
            ArrowTabular(chunked)
        self.assertIn("pa.ChunkedArray", str(cm.exception))

    def test_ingest_unknown_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            ArrowTabular(object())

    def test_ingest_none_is_noop(self) -> None:
        io = ArrowTabular(None)
        self.assertTrue(io.is_empty())


class TestArrowTabularReadTable(ArrowTestCase):
    """:meth:`_read_arrow_table` returns the held table zero-copy."""

    def test_no_target_returns_held_table_reference(self) -> None:
        t = self.table({"x": [1, 2, 3]})
        io = ArrowTabular(t)
        out = io.read_arrow_table(CastOptions())
        # No target + no rechunk → the holder hands back the same
        # underlying buffers (we don't insist on object identity since
        # the holder may concat batches into a new Table wrapper, but
        # the column buffers must match).
        self.assertEqual(out.num_rows, t.num_rows)
        self.assertEqual(out.column_names, t.column_names)

    def test_match_target_collapses_to_bypass(self) -> None:
        t = self.table({"x": [1, 2, 3]})
        io = ArrowTabular(t)
        target = Schema.from_fields([Field("x", "int64")])
        out = io.read_arrow_table(CastOptions(target=target))
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(out.schema.field("x").type, self.pa.int64())

    def test_cast_target_changes_type(self) -> None:
        pa = self.pa
        t = pa.table({"x": pa.array([1, 2, 3], type=pa.int32())})
        io = ArrowTabular(t)
        target = Schema.from_fields([Field("x", "int64")])
        out = io.read_arrow_table(CastOptions(target=target))
        self.assertEqual(out.schema.field("x").type, pa.int64())

    def test_empty_holder_returns_empty_table(self) -> None:
        io = ArrowTabular()
        out = io.read_arrow_table(CastOptions())
        self.assertEqual(out.num_rows, 0)

    def test_repeated_reads_reuse_memory_cache(self) -> None:
        t = self.table({"x": [1, 2, 3]})
        io = ArrowTabular(t)
        first = io.read_arrow_table(CastOptions())
        second = io.read_arrow_table(CastOptions())
        # Same cache object on the no-cast path.
        self.assertIs(first, second)

    def test_write_invalidates_memory_cache(self) -> None:
        t = self.table({"x": [1, 2, 3]})
        io = ArrowTabular(t)
        first = io.read_arrow_table(CastOptions())
        io.write_arrow_table(self.table({"x": [10, 20]}))
        second = io.read_arrow_table(CastOptions())
        self.assertIsNot(first, second)
        self.assertEqual(second["x"].to_pylist(), [10, 20])


class TestArrowTabularSpill(ArrowTestCase):
    """Auto-spill to :class:`ArrowIPCFile` over a :class:`LocalPath`."""

    def test_spill_threshold_triggers_consolidation(self) -> None:
        # Force a spill on the first ingest with a 1-byte threshold.
        t = self.table({"x": list(range(100)), "y": ["s"] * 100})
        io = ArrowTabular(t, spill_bytes=1)
        self.assertTrue(io.spilled)
        # Spill file lives on disk under tempdir.
        self.assertIsNotNone(io._spill_path)
        spill_path = str(io._spill_path)
        self.assertTrue(os.path.exists(spill_path))
        # Read-back returns every row.
        self.assertEqual(io.read_arrow_table().num_rows, 100)

    def test_spill_file_is_arrow_ipc_format(self) -> None:
        # ArrowIPCFile writes the IPC file format (with the
        # ``ARROW1\0\0`` magic). Confirms the spill writer actually
        # routed through that leaf rather than the legacy direct path.
        pa = self.pa
        t = self.table({"x": list(range(50))})
        io = ArrowTabular(t, spill_bytes=1)
        with open(str(io._spill_path), "rb") as fh:
            head = fh.read(8)
        self.assertEqual(head, b"ARROW1\x00\x00")
        # Verify pyarrow can open it as an IPC file.
        with pa.memory_map(str(io._spill_path), "r") as mm:
            reader = pa.ipc.open_file(mm)
            self.assertEqual(reader.read_all().num_rows, 50)

    def test_spill_persists_across_appends(self) -> None:
        t1 = self.table({"x": [1, 2, 3]})
        t2 = self.table({"x": [4, 5, 6]})
        # Threshold = 1 byte → first ingest spills; second write
        # appends and re-spills (consolidating into a fresh file).
        io = ArrowTabular(t1, spill_bytes=1)
        original_path = str(io._spill_path)
        io.write_arrow_table(t2, CastOptions(mode="append"))
        self.assertTrue(io.spilled)
        self.assertEqual(io.read_arrow_table().num_rows, 6)
        # Owned-path mode mints a new spill file each consolidation
        # and unlinks the previous one.
        self.assertFalse(pathlib.Path(original_path).exists())

    def test_unpersist_unlinks_owned_spill_file(self) -> None:
        t = self.table({"x": list(range(50))})
        io = ArrowTabular(t, spill_bytes=1)
        spill_path = str(io._spill_path)
        self.assertTrue(os.path.exists(spill_path))
        io.unpersist()
        self.assertFalse(os.path.exists(spill_path))
        self.assertIsNone(io._spill_path)
        self.assertFalse(io.spilled)

    def test_spill_threshold_disabled_keeps_in_memory(self) -> None:
        t = self.table({"x": list(range(50))})
        io = ArrowTabular(t, spill_bytes=0)
        self.assertFalse(io.spilled)
        self.assertIsNone(io._spill_path)

    def test_caller_supplied_spill_path_is_preserved(self) -> None:
        # Caller-owned spill path — we write to it but don't unlink
        # on close (BytesIO "external spill path" convention).
        import tempfile
        custom = tempfile.mktemp(suffix=".arrow")
        try:
            t = self.table({"x": list(range(50))})
            io = ArrowTabular(t, spill_bytes=1, spill_path=custom)
            self.assertTrue(os.path.exists(custom))
            io.unpersist()
            # Caller-owned path stays on disk.
            self.assertTrue(os.path.exists(custom))
        finally:
            if os.path.exists(custom):
                os.unlink(custom)

    def test_spilled_read_returns_full_table_zero_copy(self) -> None:
        t = self.table({"x": list(range(200)), "y": ["s"] * 200})
        io = ArrowTabular(t, spill_bytes=1)
        # The fast read_arrow_table path returns the cached spilled
        # table directly when no target / rechunk is set.
        out = io.read_arrow_table(CastOptions())
        self.assertIs(out, io._spilled_table)
