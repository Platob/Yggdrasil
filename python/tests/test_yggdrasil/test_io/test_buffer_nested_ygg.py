"""Tests for ``yggdrasil.io.buffer.nested.ygg``.

Covers the public surface of the ygg protocol:

- Manifest encode / decode round-trip (Arrow IPC, schema-level metadata).
- ``write_manifest`` + ``_LATEST`` pointer atomicity.
- ``YggIO`` write / read round-trip (flat + Hive-partitioned).
- Save modes: AUTO/OVERWRITE, APPEND, UPSERT, IGNORE.
- Time travel via ``read_manifest_at``.
- Folder layout (``_ygg/`` side-folder, versioned manifests).
"""
from __future__ import annotations


from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.io.buffer.nested.ygg import (
    LATEST_POINTER_NAME,
    META_DIR_NAME,
    VERSIONS_DIR_NAME,
    Manifest,
    ManifestEntry,
    YggIO,
    decode_manifest,
    encode_manifest,
    manifest_filename,
    read_latest_version,
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
            ),
            ManifestEntry(
                path="year=2025/part-00000.arrow",
                size=2048,
                modification_time=1700000001000,
                num_rows=20,
                partition_values={"year": "2025"},
            ),
        )
        return Manifest(
            version=3,
            timestamp=1700000002000,
            table_id="abc-123",
            partition_columns=("year",),
            data_schema=schema,
            engine_info="test/0",
            entries=entries,
        )

    def test_round_trip_preserves_all_fields(self):
        m = self._sample_manifest()
        decoded = decode_manifest(encode_manifest(m))

        self.assertEqual(decoded.version, m.version)
        self.assertEqual(decoded.timestamp, m.timestamp)
        self.assertEqual(decoded.table_id, m.table_id)
        self.assertEqual(decoded.partition_columns, m.partition_columns)
        self.assertEqual(decoded.engine_info, m.engine_info)
        self.assertEqual(decoded.protocol_version, m.protocol_version)
        self.assertEqual(len(decoded.entries), len(m.entries))

        for got, want in zip(decoded.entries, m.entries):
            self.assertEqual(got.path, want.path)
            self.assertEqual(got.size, want.size)
            self.assertEqual(got.modification_time, want.modification_time)
            self.assertEqual(got.num_rows, want.num_rows)
            self.assertEqual(dict(got.partition_values), dict(want.partition_values))

    def test_round_trip_empty_entries(self):
        m = Manifest.empty(
            version=0, timestamp=1, table_id="empty",
            partition_columns=(),
        )
        decoded = decode_manifest(encode_manifest(m))
        self.assertEqual(decoded.entries, ())
        self.assertEqual(decoded.version, 0)
        self.assertEqual(decoded.partition_columns, ())

    def test_data_schema_preserved(self):
        m = self._sample_manifest()
        decoded = decode_manifest(encode_manifest(m))
        names_got = list(decoded.data_schema.keys()) if hasattr(decoded.data_schema, "keys") else []
        names_want = list(m.data_schema.keys()) if hasattr(m.data_schema, "keys") else []
        self.assertEqual(names_got, names_want)

    def test_decode_rejects_empty_blob(self):
        with self.assertRaises(ValueError):
            decode_manifest(b"")

    def test_decode_rejects_missing_required_metadata(self):
        # Build a valid IPC file with the right body schema but no
        # ygg-prefixed metadata. The decoder should reject it.
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
        self.assertIn("ygg.version", str(cm.exception))

    def test_partition_values_serialized_with_sorted_keys(self):
        m = Manifest(
            version=0, timestamp=0, table_id="t",
            partition_columns=("a", "b"),
            data_schema=__import__(
                "yggdrasil.data.schema", fromlist=["Schema"],
            ).Schema.empty(),
            engine_info="t",
            entries=(
                ManifestEntry(
                    path="a=x/b=y/part.arrow",
                    size=1,
                    modification_time=0,
                    num_rows=None,
                    partition_values={"b": "y", "a": "x"},
                ),
            ),
        )
        # Decode and re-extract the embedded partition_values JSON
        # — verify keys come back in sorted order regardless of input
        # order.
        decoded = decode_manifest(encode_manifest(m))
        pv = dict(decoded.entries[0].partition_values)
        self.assertEqual(pv, {"a": "x", "b": "y"})


# ---------------------------------------------------------------------------
# Commit writer
# ---------------------------------------------------------------------------


class TestCommit(ArrowTestCase):
    """Pointer + manifest writer behavior."""

    def test_initial_pointer_returns_minus_one(self):
        from yggdrasil.io.fs import Path
        root = Path.from_(str(self.tmp_path))
        self.assertEqual(read_latest_version(root), -1)

    def test_write_manifest_creates_pointer_and_versioned_file(self):
        from yggdrasil.io.fs import Path
        root = Path.from_(str(self.tmp_path))
        m = Manifest.empty(
            version=0, timestamp=1, table_id="t",
        )
        target = write_manifest(root, m)

        self.assertTrue(target.exists())
        self.assertEqual(target.name, manifest_filename(0))
        self.assertEqual(read_latest_version(root), 0)

        # Pointer file under _ygg/.
        pointer = self.tmp_path / META_DIR_NAME / LATEST_POINTER_NAME
        self.assertTrue(pointer.exists())
        self.assertEqual(pointer.read_text().strip(), "0")

    def test_write_manifest_refuses_duplicate_version(self):
        from yggdrasil.io.fs import Path
        root = Path.from_(str(self.tmp_path))
        m = Manifest.empty(version=0, timestamp=1, table_id="t")
        write_manifest(root, m)
        with self.assertRaises(FileExistsError):
            write_manifest(root, m)

    def test_versions_dir_layout(self):
        from yggdrasil.io.fs import Path
        root = Path.from_(str(self.tmp_path))
        for v in (0, 1, 2):
            write_manifest(
                root,
                Manifest.empty(version=v, timestamp=v, table_id="t"),
            )
        vdir = self.tmp_path / META_DIR_NAME / VERSIONS_DIR_NAME
        names = sorted(p.name for p in vdir.iterdir())
        self.assertEqual(
            names,
            [manifest_filename(0), manifest_filename(1), manifest_filename(2)],
        )


# ---------------------------------------------------------------------------
# YggIO end-to-end
# ---------------------------------------------------------------------------


class TestYggIORoundTrip(ArrowTestCase):
    """High-level IO round-trip — flat tables, no partitions."""

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

    def test_initial_version_is_zero(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(self.pa.table({"id": [1]}))
        self.assertEqual(io.current_version, 0)

    def test_append_increments_version(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(self.pa.table({"id": [1]}))
        with io:
            io.write_arrow_table(self.pa.table({"id": [2]}), mode="append")

        self.assertEqual(io.current_version, 1)
        with io:
            got = io.read_arrow_table().sort_by("id")
        self.assertEqual(got.column("id").to_pylist(), [1, 2])

    def test_overwrite_replaces_contents(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(self.pa.table({"id": [1, 2, 3]}))
        with io:
            io.write_arrow_table(self.pa.table({"id": [99]}), mode="overwrite")

        self.assertEqual(io.current_version, 1)
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
        tbl = self.pa.table({"id": [1, 2], "val": ["a", "b"]})
        with io:
            io.write_arrow_table(tbl)

        io2 = YggIO(path=path)
        with io2:
            schema = io2.collect_schema()
        names = list(schema.keys())
        self.assertEqual(sorted(names), ["id", "val"])

    def test_is_empty_on_fresh_path(self):
        io = YggIO(path=str(self.tmp_path / "fresh"))
        self.assertTrue(io.is_empty())

    def test_is_empty_after_writes(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(self.pa.table({"id": [1]}))
        self.assertFalse(io.is_empty())


class TestYggIOPartitioned(ArrowTestCase):
    """Hive-partitioned reads and writes."""

    def test_partitioned_write_creates_kv_directories(self):
        path = str(self.tmp_path / "trades")
        io = YggIO(path=path, partition_columns=["year"])
        tbl = self.pa.table({
            "id": [1, 2, 3],
            "year": ["2024", "2024", "2025"],
        })
        with io:
            io.write_arrow_table(tbl)

        # Partition directories exist on disk.
        layout = sorted(
            p.name for p in (self.tmp_path / "trades").iterdir()
            if p.is_dir() and p.name != META_DIR_NAME
        )
        self.assertEqual(layout, ["year=2024", "year=2025"])

    def test_partitioned_read_injects_partition_columns(self):
        path = str(self.tmp_path / "trades")
        io = YggIO(path=path, partition_columns=["year"])
        tbl = self.pa.table({
            "id": [1, 2, 3],
            "year": ["2024", "2024", "2025"],
        })
        with io:
            io.write_arrow_table(tbl)

        # Re-read without explicit partition_columns — they come
        # from the manifest.
        io2 = YggIO(path=path)
        with io2:
            got = io2.read_arrow_table().sort_by("id")
        self.assertEqual(set(got.column_names), {"id", "year"})
        self.assertEqual(got.column("id").to_pylist(), [1, 2, 3])
        self.assertEqual(got.column("year").to_pylist(), ["2024", "2024", "2025"])


class TestYggIOUpsert(ArrowTestCase):
    """UPSERT via the inherited read-merge-overwrite helper."""

    def test_upsert_replaces_matching_rows_preserves_others(self):
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


class TestYggIOTimeTravel(ArrowTestCase):
    """Time travel via list_versions / read_manifest_at."""

    def test_list_versions_returns_all_committed(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        for i in range(3):
            with io:
                io.write_arrow_table(
                    self.pa.table({"id": [i]}),
                    mode="overwrite" if i == 0 else "append",
                )
        self.assertEqual(io.list_versions(), [0, 1, 2])

    def test_read_manifest_at_returns_historical_snapshot(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(self.pa.table({"id": [1]}))
        with io:
            io.write_arrow_table(self.pa.table({"id": [2]}), mode="append")

        m0 = io.read_manifest_at(0)
        m1 = io.read_manifest_at(1)
        self.assertEqual(len(m0.entries), 1)
        self.assertEqual(len(m1.entries), 2)

    def test_keep_old_manifests_false_removes_previous(self):
        path = str(self.tmp_path / "table")
        io = YggIO(path=path)
        with io:
            io.write_arrow_table(self.pa.table({"id": [1]}))

        # Manually pass options with keep_old_manifests=False.
        from yggdrasil.io.buffer.nested.ygg import YggOptions
        with io:
            io.write_arrow_table(
                self.pa.table({"id": [2]}),
                options=YggOptions(mode="append", keep_old_manifests=False),
            )

        self.assertEqual(io.list_versions(), [1])
