"""Tests for ``yggdrasil.io.buffer.nested.ygg``.

Covers the public surface of the ygg protocol:

- Manifest encode / decode round-trip (Arrow IPC, schema-level
  metadata, per-file column stats).
- Single-file ``write_manifest`` (no version history).
- ``YggIO`` write / read round-trip (flat + Hive-partitioned).
- Save modes: AUTO/OVERWRITE (hard delete), APPEND, UPSERT, IGNORE.
- ``ColumnStats`` computation at write time.
- :class:`Predicate` prefilter: file pruning + ``int64`` row indices.
"""
from __future__ import annotations


from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.io.buffer.nested.ygg import (
    MANIFEST_FILE_NAME,
    META_DIR_NAME,
    Between,
    ColumnStats,
    Eq,
    In,
    Manifest,
    ManifestEntry,
    YggIO,
    YggOptions,
    between,
    decode_manifest,
    encode_manifest,
    eq,
    filter_table,
    is_in,
    manifest_path,
    read_manifest,
    row_indices,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Manifest codec
# ---------------------------------------------------------------------------


class TestManifestCodec(ArrowTestCase):
    """Round-trip the on-disk manifest representation."""

    def _sample_manifest(self) -> Manifest:
        from yggdrasil.data.schema import Field, Schema

        schema = Schema.from_any_fields([
            Field(name="id", dtype="int64"),
            Field(name="val", dtype="string"),
        ])
        entries = (
            ManifestEntry(
                path="part-00000.arrow",
                size=1024,
                modification_time=1700000000000,
                num_rows=10,
                partition_values={},
                stats={"id": ColumnStats(min=1, max=10, null_count=0)},
            ),
            ManifestEntry(
                path="year=2025/part-00000.arrow",
                size=2048,
                modification_time=1700000001000,
                num_rows=20,
                partition_values={"year": "2025"},
                stats={"id": ColumnStats(min=20, max=40, null_count=2)},
            ),
        )
        return Manifest(
            timestamp=1700000002000,
            table_id="abc-123",
            partition_columns=("year",),
            primary_key_columns=("id",),
            data_schema=schema,
            engine_info="test/0",
            entries=entries,
        )

    def test_round_trip_preserves_all_fields(self):
        m = self._sample_manifest()
        decoded = decode_manifest(encode_manifest(m))

        self.assertEqual(decoded.timestamp, m.timestamp)
        self.assertEqual(decoded.table_id, m.table_id)
        self.assertEqual(decoded.partition_columns, m.partition_columns)
        self.assertEqual(decoded.primary_key_columns, m.primary_key_columns)
        self.assertEqual(decoded.engine_info, m.engine_info)
        self.assertEqual(decoded.protocol_version, m.protocol_version)
        self.assertEqual(len(decoded.entries), len(m.entries))

        for got, want in zip(decoded.entries, m.entries):
            self.assertEqual(got.path, want.path)
            self.assertEqual(got.size, want.size)
            self.assertEqual(got.modification_time, want.modification_time)
            self.assertEqual(got.num_rows, want.num_rows)
            self.assertEqual(dict(got.partition_values), dict(want.partition_values))
            # Stats round-trip including nulls.
            self.assertEqual(set(got.stats.keys()), set(want.stats.keys()))
            for col in want.stats:
                self.assertEqual(got.stats[col].min, want.stats[col].min)
                self.assertEqual(got.stats[col].max, want.stats[col].max)
                self.assertEqual(got.stats[col].null_count, want.stats[col].null_count)

    def test_round_trip_empty_entries(self):
        m = Manifest.empty(
            timestamp=1, table_id="empty",
            partition_columns=(),
            primary_key_columns=(),
        )
        decoded = decode_manifest(encode_manifest(m))
        self.assertEqual(decoded.entries, ())
        self.assertEqual(decoded.partition_columns, ())
        self.assertEqual(decoded.primary_key_columns, ())

    def test_round_trip_no_stats_entry(self):
        # An entry with no stats (e.g. table without primary keys)
        # must round-trip with an empty stats dict.
        from yggdrasil.data.schema import Schema
        m = Manifest(
            timestamp=0, table_id="t",
            partition_columns=(), primary_key_columns=(),
            data_schema=Schema.empty(), engine_info="t",
            entries=(ManifestEntry(
                path="p.arrow", size=1, modification_time=0,
                num_rows=1, partition_values={}, stats={},
            ),),
        )
        decoded = decode_manifest(encode_manifest(m))
        self.assertEqual(dict(decoded.entries[0].stats), {})

    def test_decode_rejects_empty_blob(self):
        with self.assertRaises(ValueError):
            decode_manifest(b"")

    def test_decode_rejects_missing_required_metadata(self):
        from yggdrasil.io.buffer.nested.ygg.manifest import MANIFEST_BODY_SCHEMA
        empty_batch = self.pa.RecordBatch.from_arrays(
            [self.pa.array([], type=f.type) for f in MANIFEST_BODY_SCHEMA],
            schema=MANIFEST_BODY_SCHEMA,
        )
        import io as _stdio
        buf = _stdio.BytesIO()
        with self.pa.ipc.new_file(buf, MANIFEST_BODY_SCHEMA) as w:
            w.write_batch(empty_batch)
        with self.assertRaises(ValueError) as cm:
            decode_manifest(buf.getvalue())
        self.assertIn("ygg.timestamp", str(cm.exception))


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------


class TestCommit(ArrowTestCase):
    """Single-file manifest writer behavior."""

    def test_read_manifest_returns_none_when_absent(self):
        from yggdrasil.io.fs import Path
        root = Path.from_(str(self.tmp_path))
        self.assertIsNone(read_manifest(root))

    def test_write_manifest_creates_single_file(self):
        from yggdrasil.io.fs import Path
        root = Path.from_(str(self.tmp_path))
        m = Manifest.empty(timestamp=1, table_id="t")
        target = write_manifest(root, m)

        self.assertTrue(target.exists())
        self.assertEqual(target.name, MANIFEST_FILE_NAME)
        self.assertEqual(manifest_path(root).name, MANIFEST_FILE_NAME)

    def test_write_manifest_overwrites_in_place(self):
        # Single-file model: a second write replaces the first.
        from yggdrasil.io.fs import Path
        root = Path.from_(str(self.tmp_path))
        write_manifest(root, Manifest.empty(timestamp=1, table_id="t"))
        write_manifest(root, Manifest.empty(timestamp=2, table_id="t"))
        m = read_manifest(root)
        self.assertEqual(m.timestamp, 2)


# ---------------------------------------------------------------------------
# YggIO end-to-end
# ---------------------------------------------------------------------------


class TestYggIORoundTrip(ArrowTestCase):
    """Flat tables, no partitions."""

    def test_write_then_read_roundtrip(self):
        path = str(self.tmp_path / "table")
        tbl = self.pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})

        io = YggIO(path=path)
        with io:
            io.write_arrow_table(tbl)

        io2 = YggIO(path=path)
        with io2:
            got = io2.read_arrow_table()

        self.assertEqual(got.column("id").to_pylist(), [1, 2, 3])
        self.assertEqual(got.column("val").to_pylist(), ["a", "b", "c"])

    def test_append_keeps_previous_files(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(self.pa.table({"id": [1]}))
        with io:
            io.write_arrow_table(self.pa.table({"id": [2]}), mode="append")

        with io:
            got = io.read_arrow_table().sort_by("id")
        self.assertEqual(got.column("id").to_pylist(), [1, 2])

    def test_overwrite_hard_deletes_previous_files(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(self.pa.table({"id": [1, 2, 3]}))

        # Find the old data file path so we can verify it's gone.
        from yggdrasil.io.fs import Path
        m = read_manifest(Path.from_(path))
        old_files = [Path.from_(path) / e.path for e in m.entries]
        self.assertTrue(all(p.exists() for p in old_files))

        with io:
            io.write_arrow_table(self.pa.table({"id": [99]}), mode="overwrite")

        # Old files must be physically removed (hard delete).
        for p in old_files:
            self.assertFalse(p.exists(), f"old file {p!r} should have been hard-deleted")

        with io:
            got = io.read_arrow_table()
        self.assertEqual(got.column("id").to_pylist(), [99])

    def test_metadata_directory_is_hidden_from_iter_children(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(self.pa.table({"id": [1]}))

        with io:
            child_paths = [c.path.name for c in io.iter_children()]
        self.assertNotIn(META_DIR_NAME, child_paths)
        self.assertTrue(any(n.startswith("part-") for n in child_paths))

    def test_collect_schema_from_manifest(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(
                self.pa.table({"id": [1, 2], "val": ["a", "b"]}),
            )

        io2 = YggIO(path=path)
        with io2:
            schema = io2.collect_schema()
        self.assertEqual(sorted(schema.keys()), ["id", "val"])

    def test_is_empty_on_fresh_path(self):
        io = YggIO(path=str(self.tmp_path / "fresh"))
        self.assertTrue(io.is_empty())

    def test_no_versions_directory(self):
        # Confirm the metadata folder holds exactly one manifest
        # file — no versions/ subdir, no _LATEST pointer.
        path = self.tmp_path / "table"
        io = YggIO(path=str(path))
        with io:
            io.write_arrow_table(self.pa.table({"id": [1]}))
        with io:
            io.write_arrow_table(self.pa.table({"id": [2]}), mode="append")

        meta_entries = sorted(p.name for p in (path / META_DIR_NAME).iterdir())
        self.assertEqual(meta_entries, [MANIFEST_FILE_NAME])


# ---------------------------------------------------------------------------
# Partitioned
# ---------------------------------------------------------------------------


class TestYggIOPartitioned(ArrowTestCase):
    """Hive-partitioned reads and writes."""

    def test_partitioned_write_creates_kv_directories(self):
        path = str(self.tmp_path / "trades")
        io = YggIO(path=path, partition_columns=["year"])
        with io:
            io.write_arrow_table(self.pa.table({
                "id": [1, 2, 3],
                "year": ["2024", "2024", "2025"],
            }))

        layout = sorted(
            p.name for p in (self.tmp_path / "trades").iterdir()
            if p.is_dir() and p.name != META_DIR_NAME
        )
        self.assertEqual(layout, ["year=2024", "year=2025"])

    def test_partitioned_read_injects_partition_columns(self):
        path = str(self.tmp_path / "trades")
        io = YggIO(path=path, partition_columns=["year"])
        with io:
            io.write_arrow_table(self.pa.table({
                "id": [1, 2, 3],
                "year": ["2024", "2024", "2025"],
            }))

        io2 = YggIO(path=path)  # partition columns from manifest
        with io2:
            got = io2.read_arrow_table().sort_by("id")
        self.assertEqual(set(got.column_names), {"id", "year"})
        self.assertEqual(got.column("year").to_pylist(),
                         ["2024", "2024", "2025"])


class TestYggIOUpsert(ArrowTestCase):
    """UPSERT via the inherited read-merge-overwrite helper."""

    def test_upsert_replaces_matching_rows(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(
                self.pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]}),
            )
        with io:
            io.write_arrow_table(
                self.pa.table({"id": [2, 4], "val": ["B", "D"]}),
                mode="upsert",
                match_by_names=["id"],
            )

        io2 = YggIO(path=path)
        with io2:
            got = io2.read_arrow_table().sort_by("id")
        self.assertEqual(got.column("id").to_pylist(), [1, 2, 3, 4])
        self.assertEqual(got.column("val").to_pylist(), ["a", "B", "c", "D"])


# ---------------------------------------------------------------------------
# Stats + predicate
# ---------------------------------------------------------------------------


class TestStatsComputation(ArrowTestCase):
    """Per-file stats are computed at write time for primary key columns."""

    def test_stats_recorded_for_primary_keys(self):
        path = str(self.tmp_path / "t")
        io = YggIO(path=path, primary_key_columns=["id"])
        with io:
            io.write_arrow_table(self.pa.table({"id": [10, 20, 30], "v": [1, 2, 3]}))
        with io:
            io.write_arrow_table(
                self.pa.table({"id": [100, 200], "v": [10, 20]}),
                mode="append",
            )

        from yggdrasil.io.fs import Path
        m = read_manifest(Path.from_(path))
        self.assertEqual(len(m.entries), 2)
        for e in m.entries:
            self.assertIn("id", e.stats)
            self.assertEqual(e.stats["id"].null_count, 0)
        # File 1 covers [10, 30], file 2 covers [100, 200].
        ranges = sorted((e.stats["id"].min, e.stats["id"].max) for e in m.entries)
        self.assertEqual(ranges, [(10, 30), (100, 200)])

    def test_no_stats_when_no_primary_keys_declared(self):
        path = str(self.tmp_path / "t")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(self.pa.table({"id": [1, 2, 3]}))

        from yggdrasil.io.fs import Path
        m = read_manifest(Path.from_(path))
        for e in m.entries:
            self.assertEqual(dict(e.stats), {})

    def test_primary_keys_persisted_across_appends(self):
        # A subsequent append without explicit primary_key_columns
        # picks the list back up from the live manifest.
        path = str(self.tmp_path / "t")
        io = YggIO(path=path, primary_key_columns=["id"])
        with io:
            io.write_arrow_table(self.pa.table({"id": [1]}))

        # Re-open without specifying primary_key_columns.
        io2 = YggIO(path=path)
        with io2:
            io2.write_arrow_table(
                self.pa.table({"id": [2]}), mode="append",
            )

        from yggdrasil.io.fs import Path
        m = read_manifest(Path.from_(path))
        self.assertEqual(m.primary_key_columns, ("id",))
        for e in m.entries:
            self.assertIn("id", e.stats)


class TestPredicateRowIndices(ArrowTestCase):
    """``row_indices`` and ``filter_table`` apply a predicate to a Table."""

    def test_eq_returns_int64_indices(self):
        t = self.pa.table({"id": [1, 2, 3, 2, 1], "v": ["a", "b", "c", "d", "e"]})
        idx = row_indices(eq("id", 2), t)
        self.assertEqual(idx.type, self.pa.int64())
        self.assertEqual(idx.to_pylist(), [1, 3])

    def test_in_returns_int64_indices(self):
        t = self.pa.table({"id": [1, 2, 3, 4, 5]})
        idx = row_indices(is_in("id", [2, 4, 9]), t)
        self.assertEqual(idx.to_pylist(), [1, 3])

    def test_between_open_bounds(self):
        t = self.pa.table({"id": [1, 5, 10, 15, 20]})
        # No lower bound.
        self.assertEqual(row_indices(between("id", hi=10), t).to_pylist(), [0, 1, 2])
        # No upper bound.
        self.assertEqual(row_indices(between("id", lo=10), t).to_pylist(), [2, 3, 4])
        # Both inclusive.
        self.assertEqual(row_indices(between("id", 5, 15), t).to_pylist(), [1, 2, 3])

    def test_and_combines_predicates(self):
        t = self.pa.table({"id": [1, 2, 3, 4, 5], "g": ["x", "y", "x", "y", "x"]})
        pred = eq("g", "x") & between("id", 2, 5)
        self.assertEqual(row_indices(pred, t).to_pylist(), [2, 4])

    def test_filter_table_returns_taken_slice(self):
        t = self.pa.table({"id": [1, 2, 3, 4]})
        out = filter_table(eq("id", 3), t)
        self.assertEqual(out.column("id").to_pylist(), [3])


class TestPredicateFilePruning(ArrowTestCase):
    """Manifest stats prune files before any data file is opened."""

    def _write_three_files(self):
        path = str(self.tmp_path / "t")
        io = YggIO(path=path, primary_key_columns=["id"])
        with io:
            io.write_arrow_table(self.pa.table({"id": [1, 2]}))
        with io:
            io.write_arrow_table(self.pa.table({"id": [10, 20]}), mode="append")
        with io:
            io.write_arrow_table(self.pa.table({"id": [100, 200]}), mode="append")
        return path

    def test_eq_prunes_unrelated_files(self):
        path = self._write_three_files()
        io = YggIO(path=path)
        with io:
            visited = list(io.iter_matching_indices(eq("id", 100)))
        self.assertEqual(len(visited), 1)
        child, idx = visited[0]
        self.assertEqual(idx.to_pylist(), [0])

    def test_between_keeps_overlapping_files(self):
        path = self._write_three_files()
        io = YggIO(path=path)
        with io:
            visited = list(io.iter_matching_indices(between("id", 5, 50)))
        # Files [1,2] excluded; files [10,20] and [100,200] *might*
        # overlap. The pruner keeps file [10,20] (range overlaps),
        # and prunes [100,200] (min=100 > 50). One survivor.
        self.assertEqual(len(visited), 1)
        _, idx = visited[0]
        self.assertEqual(idx.to_pylist(), [0, 1])

    def test_in_prunes_files_outside_value_set(self):
        path = self._write_three_files()
        io = YggIO(path=path)
        with io:
            visited = list(io.iter_matching_indices(is_in("id", [1, 200])))
        # Exactly two files contain values in the set.
        self.assertEqual(len(visited), 2)

    def test_partition_predicate_prunes_subtree(self):
        path = str(self.tmp_path / "t")
        io = YggIO(path=path, partition_columns=["year"], primary_key_columns=["id"])
        with io:
            io.write_arrow_table(self.pa.table({
                "id": [1, 2, 3, 4],
                "year": ["2024", "2024", "2025", "2025"],
            }))

        io2 = YggIO(path=path)
        with io2:
            got = io2.read_arrow_table(predicate=eq("year", "2025")).sort_by("id")
        self.assertEqual(got.column("id").to_pylist(), [3, 4])

    def test_read_arrow_table_with_predicate_returns_filtered_rows(self):
        path = self._write_three_files()
        io = YggIO(path=path)
        with io:
            got = io.read_arrow_table(predicate=between("id", 5, 50)).sort_by("id")
        self.assertEqual(got.column("id").to_pylist(), [10, 20])


class TestEqMatchesEntry(ArrowTestCase):
    """Stat-only matches_entry semantics, exercised directly."""

    def test_eq_in_range_keeps_file(self):
        e = ManifestEntry(
            path="p", size=0, modification_time=0, num_rows=10,
            partition_values={},
            stats={"id": ColumnStats(min=1, max=100, null_count=0)},
        )
        self.assertTrue(Eq("id", 50).matches_entry(e))

    def test_eq_outside_range_prunes_file(self):
        e = ManifestEntry(
            path="p", size=0, modification_time=0, num_rows=10,
            partition_values={},
            stats={"id": ColumnStats(min=1, max=100, null_count=0)},
        )
        self.assertFalse(Eq("id", 200).matches_entry(e))

    def test_eq_unknown_stats_fails_open(self):
        # No stats for the column → can't rule out, must keep.
        e = ManifestEntry(
            path="p", size=0, modification_time=0, num_rows=10,
            partition_values={},
            stats={},
        )
        self.assertTrue(Eq("id", 200).matches_entry(e))

    def test_in_outside_range_prunes(self):
        e = ManifestEntry(
            path="p", size=0, modification_time=0, num_rows=10,
            partition_values={},
            stats={"id": ColumnStats(min=10, max=20, null_count=0)},
        )
        self.assertFalse(In("id", (1, 2, 30)).matches_entry(e))

    def test_in_with_overlap_keeps(self):
        e = ManifestEntry(
            path="p", size=0, modification_time=0, num_rows=10,
            partition_values={},
            stats={"id": ColumnStats(min=10, max=20, null_count=0)},
        )
        self.assertTrue(In("id", (5, 15, 30)).matches_entry(e))

    def test_between_disjoint_prunes(self):
        e = ManifestEntry(
            path="p", size=0, modification_time=0, num_rows=10,
            partition_values={},
            stats={"id": ColumnStats(min=10, max=20, null_count=0)},
        )
        self.assertFalse(Between("id", 100, 200).matches_entry(e))

    def test_partition_eq_matches(self):
        e = ManifestEntry(
            path="p", size=0, modification_time=0, num_rows=10,
            partition_values={"year": "2025"},
            stats={},
        )
        self.assertTrue(Eq("year", "2025").matches_entry(e))
        self.assertTrue(Eq("year", 2025).matches_entry(e))  # int coerced
        self.assertFalse(Eq("year", "2024").matches_entry(e))


# ---------------------------------------------------------------------------
# YggOptions
# ---------------------------------------------------------------------------


class TestYggOptions(ArrowTestCase):
    def test_predicate_passed_via_options(self):
        path = str(self.tmp_path / "t")
        io = YggIO(path=path, primary_key_columns=["id"])
        with io:
            io.write_arrow_table(self.pa.table({"id": [1, 2, 3, 4]}))

        opts = YggOptions(predicate=between("id", 2, 3))
        with io:
            got = io.read_arrow_table(options=opts).sort_by("id")
        self.assertEqual(got.column("id").to_pylist(), [2, 3])
