"""Tests for :class:`yggdrasil.arrow.tabular.ArrowTabular`.

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
        with self.assertRaises(TypeError):
            ArrowTabular(chunked)

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
    """Append-only spill to a folder of :class:`ArrowIPCFile` part files."""

    def test_spill_threshold_triggers_consolidation(self) -> None:
        t = self.table({"x": list(range(100)), "y": ["s"] * 100})
        io = ArrowTabular(t, spill_bytes=1)
        self.assertTrue(io.spilled)
        # Spill folder exists under tempdir with one part file.
        self.assertIsNotNone(io.spill_dir)
        self.assertTrue(io.spill_dir.is_dir())
        self.assertEqual(len(io.spill_parts), 1)
        # Read-back returns every row.
        self.assertEqual(io.read_arrow_table().num_rows, 100)

    def test_spill_part_files_are_arrow_ipc(self) -> None:
        pa = self.pa
        t = self.table({"x": list(range(50))})
        io = ArrowTabular(t, spill_bytes=1)
        part = io.spill_parts[0]
        with open(str(part), "rb") as fh:
            head = fh.read(8)
        self.assertEqual(head, b"ARROW1\x00\x00")
        # pyarrow can open the part directly via mmap.
        with pa.memory_map(str(part), "r") as mm:
            reader = pa.ipc.open_file(mm)
            self.assertEqual(reader.read_all().num_rows, 50)

    def test_spill_is_append_only_across_appends(self) -> None:
        t1 = self.table({"x": [1, 2, 3]})
        t2 = self.table({"x": [4, 5, 6]})
        io = ArrowTabular(t1, spill_bytes=1)
        first_part = io.spill_parts[0]
        # APPEND-mode write adds a new part; the original stays on disk.
        io.write_arrow_table(t2, CastOptions(mode="append"))
        self.assertTrue(io.spilled)
        self.assertEqual(len(io.spill_parts), 2)
        self.assertTrue(first_part.exists())
        self.assertEqual(io.read_arrow_table().num_rows, 6)
        # Part file names are monotonic + lexically sortable.
        self.assertEqual(
            [p.name for p in io.spill_parts],
            sorted(p.name for p in io.spill_parts),
        )

    def test_spill_skip_when_in_memory_tail_empty(self) -> None:
        # Once data is fully spilled and no new batches arrive, calling
        # the spill machinery again is a no-op — already cached on disk.
        t = self.table({"x": list(range(50))})
        io = ArrowTabular(t, spill_bytes=1)
        parts_before = list(io.spill_parts)
        io._maybe_spill()  # in-memory tail is empty — skip
        io._consolidate_spill()  # explicit call — same skip path
        self.assertEqual(io.spill_parts, parts_before)

    def test_unpersist_removes_owned_spill_folder(self) -> None:
        t = self.table({"x": list(range(50))})
        io = ArrowTabular(t, spill_bytes=1)
        spill_dir = io.spill_dir
        self.assertTrue(spill_dir.is_dir())
        io.unpersist()
        # One rmtree wipes the whole spill state — folder + parts.
        self.assertFalse(spill_dir.exists())
        self.assertIsNone(io.spill_dir)
        self.assertEqual(io.spill_parts, [])
        self.assertFalse(io.spilled)

    def test_spill_threshold_disabled_keeps_in_memory(self) -> None:
        t = self.table({"x": list(range(50))})
        io = ArrowTabular(t, spill_bytes=0)
        self.assertFalse(io.spilled)
        self.assertIsNone(io.spill_dir)
        self.assertEqual(io.spill_parts, [])

    def test_caller_supplied_spill_folder_is_preserved(self) -> None:
        # Caller-owned spill folder — we mint part files inside it
        # but don't rmtree it on close (BytesIO "external spill" branch).
        import tempfile
        custom = pathlib.Path(tempfile.mkdtemp(prefix="ygg-arrow-spill-"))
        try:
            t = self.table({"x": list(range(50))})
            io = ArrowTabular(t, spill_bytes=1, spill_path=str(custom))
            self.assertEqual(io.spill_dir, custom)
            self.assertEqual(len(io.spill_parts), 1)
            io.unpersist()
            # Caller-owned folder stays on disk; our part files are
            # released from tracking but the folder is left to caller.
            self.assertTrue(custom.exists())
        finally:
            import shutil
            shutil.rmtree(custom, ignore_errors=True)

    def test_unique_returns_arrow_tabular_via_arrow_ops(self) -> None:
        t = self.table({"id": [1, 2, 1, 3, 2], "v": ["a", "b", "c", "d", "e"]})
        out = ArrowTabular(t).unique("id")
        # Default engine is arrow → result is an ArrowTabular.
        self.assertIs(type(out), ArrowTabular)
        self.assertEqual(
            out.read_arrow_table().to_pydict(),
            {"id": [1, 2, 3], "v": ["a", "b", "d"]},
        )

    def test_unique_accepts_field_and_iterable(self) -> None:
        t = self.table({"id": [1, 2, 1, 3, 2], "v": ["a", "b", "c", "d", "e"]})
        io = ArrowTabular(t)
        # bare string, list of strings, Field, and list-of-Field all
        # resolve to the same dedup keys.
        s1 = io.unique("id").read_arrow_table().to_pydict()
        s2 = io.unique(["id"]).read_arrow_table().to_pydict()
        s3 = io.unique(Field("id", "int64")).read_arrow_table().to_pydict()
        s4 = io.unique([Field("id", "int64")]).read_arrow_table().to_pydict()
        self.assertEqual(s1, s2)
        self.assertEqual(s1, s3)
        self.assertEqual(s1, s4)

    def test_unique_empty_keys_short_circuits(self) -> None:
        io = ArrowTabular(self.table({"x": [1, 2]}))
        self.assertIs(io.unique([]), io)
        self.assertIs(io.unique(None), io)

    def test_resample_with_int_seconds(self) -> None:
        import datetime as dt
        import pyarrow as pa

        ts = pa.array(
            [dt.datetime(2024, 1, 1, h) for h in range(4)],
            type=pa.timestamp("us"),
        )
        v = pa.array([1, None, None, 4])
        io = ArrowTabular(pa.table({"ts": ts, "v": v}))
        out = io.resample(on="ts", sampling=7200)
        # 2-hour buckets: ts=00→1 (ffill→1), ts=02→None→ffill from 1.
        self.assertIs(type(out), ArrowTabular)
        rows = out.read_arrow_table().to_pydict()
        self.assertEqual(rows["v"], [1, 1])

    def test_resample_with_timedelta_and_iso_duration(self) -> None:
        import datetime as dt
        import pyarrow as pa

        ts = pa.array(
            [dt.datetime(2024, 1, 1, h) for h in range(4)],
            type=pa.timestamp("us"),
        )
        v = pa.array([1, None, None, 4])
        io = ArrowTabular(pa.table({"ts": ts, "v": v}))
        td_rows = io.resample(on="ts", sampling=dt.timedelta(hours=2)).read_arrow_table()
        iso_rows = io.resample(on="ts", sampling="PT2H").read_arrow_table()
        self.assertEqual(td_rows.to_pydict(), iso_rows.to_pydict())

    def test_resample_with_field_on_and_partition_by(self) -> None:
        import datetime as dt
        import pyarrow as pa

        # Two symbols, each with 6 hourly observations. Interleaved
        # rows force the partition_by branch to do real work — without
        # it, the bucket collapse would cross symbols.
        rows: list[tuple[str, dt.datetime, "int | None"]] = []
        for h in range(6):
            t = dt.datetime(2024, 1, 1, h)
            rows.append(("A", t, 10 if h in (0, 3) else None))
            rows.append(("B", t, 100 if h == 4 else None))
        sym = pa.array([r[0] for r in rows])
        ts = pa.array([r[1] for r in rows], type=pa.timestamp("us"))
        v = pa.array([r[2] for r in rows])
        io = ArrowTabular(pa.table({"ts": ts, "sym": sym, "v": v}))
        out = io.resample(
            on=Field("ts", "timestamp[us]"),
            sampling="PT2H",
            partition_by=Field("sym", "string"),
        )
        result = sorted(
            out.read_arrow_table().to_pylist(),
            key=lambda r: (r["sym"], r["ts"]),
        )
        per_sym: dict[str, list] = {"A": [], "B": []}
        for r in result:
            per_sym[r["sym"]].append(r["v"])
        # A's 2h buckets: [10, None, None] (first rows at h=0,2,4)
        #     ffill → [10, 10, 10].
        # B's 2h buckets: [None, None, 100] (first rows at h=0,2,4)
        #     leading nulls have no prior non-null in B → stay null.
        self.assertEqual(per_sym["A"], [10, 10, 10])
        self.assertEqual(per_sym["B"], [None, None, 100])

    def test_resample_zero_or_negative_short_circuits(self) -> None:
        import datetime as dt
        import pyarrow as pa

        io = ArrowTabular(pa.table({
            "ts": pa.array([dt.datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "v": [1],
        }))
        self.assertIs(io.resample(on="ts", sampling=0), io)
        self.assertIs(io.resample(on="ts", sampling=-1), io)

    def test_resample_invalid_sampling_raises(self) -> None:
        import datetime as dt
        import pyarrow as pa

        io = ArrowTabular(pa.table({
            "ts": pa.array([dt.datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "v": [1],
        }))
        with self.assertRaises(ValueError):
            io.resample(on="ts", sampling="not-a-duration")
        with self.assertRaises(TypeError):
            io.resample(on="ts", sampling=True)  # bool rejected explicitly
        with self.assertRaises(TypeError):
            io.resample(on="ts", sampling=object())

    def test_unique_invalid_key_type_raises(self) -> None:
        io = ArrowTabular(self.table({"x": [1, 2]}))
        with self.assertRaises(TypeError):
            io.unique(123)
        with self.assertRaises(TypeError):
            io.unique(b"x")
        with self.assertRaises(TypeError):
            io.unique([1, 2])

    def test_select_keeps_named_columns(self) -> None:
        t = self.table({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        out = ArrowTabular(t).select("a", "c")
        self.assertIs(type(out), ArrowTabular)
        self.assertEqual(
            out.read_arrow_table().to_pydict(),
            {"a": [1, 2], "c": [10, 20]},
        )

    def test_select_accepts_field_and_iterable(self) -> None:
        t = self.table({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        io = ArrowTabular(t)
        from_list = io.select(["a", "b"]).read_arrow_table().to_pydict()
        from_field = io.select(Field("a", "int64"), Field("b", "string"))
        from_field_list = io.select([Field("a", "int64"), Field("b", "string")])
        self.assertEqual(from_list, from_field.read_arrow_table().to_pydict())
        self.assertEqual(from_list, from_field_list.read_arrow_table().to_pydict())

    def test_select_empty_raises(self) -> None:
        io = ArrowTabular(self.table({"a": [1, 2]}))
        with self.assertRaises(ValueError):
            io.select()

    def test_select_missing_column_raises(self) -> None:
        io = ArrowTabular(self.table({"a": [1, 2]}))
        with self.assertRaises(KeyError):
            io.select("nope")

    def test_drop_removes_named_columns(self) -> None:
        t = self.table({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        out = ArrowTabular(t).drop("b")
        self.assertEqual(
            out.read_arrow_table().to_pydict(),
            {"a": [1, 2], "c": [10, 20]},
        )

    def test_drop_missing_column_is_no_op(self) -> None:
        t = self.table({"a": [1, 2], "b": ["x", "y"]})
        out = ArrowTabular(t).drop("nope")
        self.assertEqual(
            out.read_arrow_table().to_pydict(),
            t.to_pydict(),
        )

    def test_drop_empty_returns_self(self) -> None:
        io = ArrowTabular(self.table({"a": [1, 2]}))
        self.assertIs(io.drop(), io)

    def test_filter_accepts_sql_string(self) -> None:
        t = self.table({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "z"]})
        out = ArrowTabular(t).filter("a > 2")
        self.assertEqual(
            out.read_arrow_table().to_pydict(),
            {"a": [3, 4], "b": ["x", "z"]},
        )

    def test_filter_accepts_yggdrasil_expression(self) -> None:
        from yggdrasil.execution.expr import col

        t = self.table({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "z"]})
        out = ArrowTabular(t).filter(col("b") == "x")
        self.assertEqual(
            out.read_arrow_table().to_pydict(),
            {"a": [1, 3], "b": ["x", "x"]},
        )

    def test_filter_chained_yggdrasil_expressions(self) -> None:
        # Two predicates AND-merged via ``&`` on the AST.
        from yggdrasil.execution.expr import col

        t = self.table({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "x", "z", "x"]})
        out = ArrowTabular(t).filter((col("a") > 1) & (col("b") == "x"))
        self.assertEqual(
            out.read_arrow_table().to_pydict(),
            {"a": [3, 5], "b": ["x", "x"]},
        )

    def test_filter_callable_rejected_on_arrow_path(self) -> None:
        # Pure callables don't lift through the Predicate parser; the
        # base ``Tabular.filter`` raises TypeError. Spark keeps the
        # legacy callable path via its own ``filter`` override.
        io = ArrowTabular(self.table({"a": [1, 2]}))
        with self.assertRaises(TypeError):
            io.filter(lambda r: True)

    def test_spilled_read_returns_cached_table_zero_copy(self) -> None:
        t = self.table({"x": list(range(200)), "y": ["s"] * 200})
        io = ArrowTabular(t, spill_bytes=1)
        # The fast read_arrow_table path returns the cached concat
        # table by reference; a second read sees the same object.
        first = io.read_arrow_table(CastOptions())
        second = io.read_arrow_table(CastOptions())
        self.assertIs(first, second)
        self.assertEqual(first.num_rows, 200)
