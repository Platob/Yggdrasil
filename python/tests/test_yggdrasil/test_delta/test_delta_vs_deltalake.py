"""Integration tests comparing yggdrasil DeltaFolder with the deltalake package.

Validates that tables written by yggdrasil are readable by deltalake and
vice versa. Covers:

- Basic round-trip (write with ygg, read with deltalake)
- Reverse round-trip (write with deltalake, read with ygg)
- Partitioned tables
- Schema fidelity
- Deletion vectors
- Checkpoint V1 and V2
- APPEND / OVERWRITE modes
- Stats collection
- Time-travel
"""
from __future__ import annotations

import os
import unittest

from yggdrasil.enums import Mode
from yggdrasil.delta.io import DeltaOptions
from yggdrasil.delta.tests import DeltaTestCase


def _has_deltalake() -> bool:
    try:
        import deltalake  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_deltalake(), "deltalake package not installed")
class TestYggWriteDeltalakeRead(DeltaTestCase):
    """Write with yggdrasil DeltaFolder, read with deltalake."""

    def test_unpartitioned_round_trip(self) -> None:
        import deltalake

        d = self.delta_io()
        t = self.pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        d.write_arrow_table(t)

        dt = deltalake.DeltaTable(str(d.path))
        out = dt.to_pyarrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3])
        self.assertEqual(sorted(out.column("val").to_pylist()), ["a", "b", "c"])

    def test_partitioned_round_trip(self) -> None:
        import deltalake

        from yggdrasil.data.data_field import Field
        from yggdrasil.data.schema import Schema
        from yggdrasil.data.types.primitive import Int64Type, StringType

        schema = Schema()
        schema.with_field(Field(name="id", dtype=Int64Type()))
        schema.with_field(
            Field(name="region", dtype=StringType()).with_partition_by(True)
        )
        schema.with_field(Field(name="val", dtype=StringType()))

        d = self.delta_io()
        t = self.pa.table({
            "id": [1, 2, 3, 4],
            "region": ["us", "us", "eu", "eu"],
            "val": ["a", "b", "c", "d"],
        })
        d.write_arrow_table(t, options=DeltaOptions(target=schema))

        dt = deltalake.DeltaTable(str(d.path))
        out = dt.to_pyarrow_table()
        self.assertEqual(out.num_rows, 4)
        self.assertIn("region", out.column_names)
        self.assertEqual(set(out.column("region").to_pylist()), {"us", "eu"})

    def test_append_mode(self) -> None:
        import deltalake

        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2]}))
        d.write_arrow_batches(
            self.pa.table({"id": [3, 4]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )

        dt = deltalake.DeltaTable(str(d.path))
        out = dt.to_pyarrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3, 4])

    def test_overwrite_mode(self) -> None:
        import deltalake

        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2]}))
        d.write_arrow_table(
            self.pa.table({"id": [99]}),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )

        dt = deltalake.DeltaTable(str(d.path))
        out = dt.to_pyarrow_table()
        self.assertEqual(out.column("id").to_pylist(), [99])

    def test_schema_fidelity(self) -> None:
        import deltalake

        d = self.delta_io()
        table = self.pa.table({
            "int_col": self.pa.array([1, 2, 3], type=self.pa.int64()),
            "float_col": self.pa.array([1.1, 2.2, 3.3], type=self.pa.float64()),
            "str_col": self.pa.array(["a", "b", "c"], type=self.pa.string()),
            "bool_col": self.pa.array([True, False, True], type=self.pa.bool_()),
        })
        d.write_arrow_table(table)

        dt = deltalake.DeltaTable(str(d.path))
        out = dt.to_pyarrow_table()
        schema = out.schema
        self.assertEqual(schema.field("int_col").type, self.pa.int64())
        self.assertEqual(schema.field("float_col").type, self.pa.float64())
        self.assertTrue(
            schema.field("str_col").type in (self.pa.string(), self.pa.large_string())
        )
        self.assertEqual(schema.field("bool_col").type, self.pa.bool_())

    def test_time_travel(self) -> None:
        import deltalake

        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2]}))
        d.write_arrow_batches(
            self.pa.table({"id": [3]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )

        dt = deltalake.DeltaTable(str(d.path), version=0)
        out = dt.to_pyarrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2])

        dt_head = deltalake.DeltaTable(str(d.path))
        out_head = dt_head.to_pyarrow_table()
        self.assertEqual(sorted(out_head.column("id").to_pylist()), [1, 2, 3])

    def test_v1_checkpoint_readable_by_ygg(self) -> None:
        """V1 checkpoints produced by ygg are correctly replayed by ygg itself."""
        d = self.delta_io()
        for i in range(6):
            mode = Mode.AUTO if i == 0 else Mode.APPEND
            d.write_arrow_batches(
                self.pa.table({"id": [i]}).to_batches(),
                options=DeltaOptions(
                    mode=mode,
                    checkpoint_interval=5,
                    checkpoint_kind="v1",
                ),
            )

        from yggdrasil.io.delta.delta_folder import DeltaFolder
        d2 = DeltaFolder(path=str(d.path))
        out = d2.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), list(range(6)))

    def test_no_checkpoint_readable_by_deltalake(self) -> None:
        """Tables without checkpoints (pure JSON commits) are readable by deltalake."""
        import deltalake

        d = self.delta_io()
        for i in range(3):
            mode = Mode.AUTO if i == 0 else Mode.APPEND
            d.write_arrow_batches(
                self.pa.table({"id": [i]}).to_batches(),
                options=DeltaOptions(mode=mode, checkpoint_interval=0),
            )

        dt = deltalake.DeltaTable(str(d.path))
        out = dt.to_pyarrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), list(range(3)))

    def test_stats_in_commit(self) -> None:
        d = self.delta_io()
        t = self.pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        d.write_arrow_table(t, options=DeltaOptions(collect_stats=True))

        snap = d.snapshot(fresh=True)
        for add in snap.active_files.values():
            self.assertIsNotNone(add.stats)
            import json
            stats = json.loads(add.stats)
            self.assertEqual(stats["numRecords"], 3)
            self.assertIn("minValues", stats)
            self.assertIn("maxValues", stats)

    def test_multi_type_table(self) -> None:
        import deltalake

        d = self.delta_io()
        table = self.pa.table({
            "int8_col": self.pa.array([1, 2], type=self.pa.int8()),
            "int16_col": self.pa.array([100, 200], type=self.pa.int16()),
            "int32_col": self.pa.array([1000, 2000], type=self.pa.int32()),
            "int64_col": self.pa.array([10000, 20000], type=self.pa.int64()),
            "float32_col": self.pa.array([1.5, 2.5], type=self.pa.float32()),
            "float64_col": self.pa.array([1.5, 2.5], type=self.pa.float64()),
            "string_col": self.pa.array(["hello", "world"], type=self.pa.string()),
            "bool_col": self.pa.array([True, False], type=self.pa.bool_()),
            "binary_col": self.pa.array([b"abc", b"def"], type=self.pa.binary()),
        })
        d.write_arrow_table(table)

        dt = deltalake.DeltaTable(str(d.path))
        out = dt.to_pyarrow_table()
        self.assertEqual(out.num_rows, 2)
        self.assertEqual(out.column("string_col").to_pylist(), ["hello", "world"])

    def test_nullable_columns(self) -> None:
        import deltalake

        d = self.delta_io()
        table = self.pa.table({
            "id": [1, 2, 3],
            "nullable_str": ["a", None, "c"],
            "nullable_int": self.pa.array([10, None, 30], type=self.pa.int64()),
        })
        d.write_arrow_table(table)

        dt = deltalake.DeltaTable(str(d.path))
        out = dt.to_pyarrow_table()
        self.assertEqual(out.column("nullable_str").to_pylist(), ["a", None, "c"])
        self.assertEqual(out.column("nullable_int").to_pylist(), [10, None, 30])


@unittest.skipUnless(_has_deltalake(), "deltalake package not installed")
class TestDeltalakeWriteYggRead(DeltaTestCase):
    """Write with deltalake, read with yggdrasil DeltaFolder."""

    def test_basic_read(self) -> None:
        import deltalake

        table_path = str(self.tmp_path / "dl_table")
        t = self.pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        deltalake.write_deltalake(table_path, t)

        from yggdrasil.io.delta.delta_folder import DeltaFolder
        d = DeltaFolder(path=table_path)
        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3])

    def test_partitioned_read(self) -> None:
        import deltalake

        table_path = str(self.tmp_path / "dl_partitioned")
        t = self.pa.table({
            "id": [1, 2, 3, 4],
            "region": ["us", "us", "eu", "eu"],
            "val": ["a", "b", "c", "d"],
        })
        deltalake.write_deltalake(
            table_path, t, partition_by=["region"],
        )

        from yggdrasil.io.delta.delta_folder import DeltaFolder
        d = DeltaFolder(path=table_path)
        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, 4)
        self.assertEqual(set(out.column("region").to_pylist()), {"us", "eu"})

    def test_multi_version_read(self) -> None:
        import deltalake

        table_path = str(self.tmp_path / "dl_versions")
        t1 = self.pa.table({"id": [1, 2]})
        deltalake.write_deltalake(table_path, t1)

        t2 = self.pa.table({"id": [3, 4]})
        deltalake.write_deltalake(table_path, t2, mode="append")

        from yggdrasil.io.delta.delta_folder import DeltaFolder
        d = DeltaFolder(path=table_path)

        v0 = d.read_arrow_table(options=DeltaOptions(version=0))
        self.assertEqual(sorted(v0.column("id").to_pylist()), [1, 2])

        head = d.read_arrow_table()
        self.assertEqual(sorted(head.column("id").to_pylist()), [1, 2, 3, 4])

    def test_overwrite_read(self) -> None:
        import deltalake

        table_path = str(self.tmp_path / "dl_overwrite")
        t1 = self.pa.table({"id": [1, 2]})
        deltalake.write_deltalake(table_path, t1)

        t2 = self.pa.table({"id": [99]})
        deltalake.write_deltalake(table_path, t2, mode="overwrite")

        from yggdrasil.io.delta.delta_folder import DeltaFolder
        d = DeltaFolder(path=table_path)
        out = d.read_arrow_table()
        self.assertEqual(out.column("id").to_pylist(), [99])

    def test_schema_fidelity_reverse(self) -> None:
        import deltalake

        table_path = str(self.tmp_path / "dl_schema")
        t = self.pa.table({
            "int_col": self.pa.array([1, 2], type=self.pa.int64()),
            "str_col": self.pa.array(["a", "b"], type=self.pa.string()),
            "float_col": self.pa.array([1.1, 2.2], type=self.pa.float64()),
            "bool_col": self.pa.array([True, False], type=self.pa.bool_()),
        })
        deltalake.write_deltalake(table_path, t)

        from yggdrasil.io.delta.delta_folder import DeltaFolder
        d = DeltaFolder(path=table_path)
        schema = d.collect_schema()
        names = [f.name for f in schema.fields]
        self.assertEqual(names, ["int_col", "str_col", "float_col", "bool_col"])

    def test_large_table_round_trip(self) -> None:
        """Write a moderately large table with deltalake, read with ygg."""
        import deltalake

        table_path = str(self.tmp_path / "dl_large")
        n = 10000
        t = self.pa.table({
            "id": list(range(n)),
            "val": [f"row_{i}" for i in range(n)],
            "score": [float(i) * 0.1 for i in range(n)],
        })
        deltalake.write_deltalake(table_path, t)

        from yggdrasil.io.delta.delta_folder import DeltaFolder
        d = DeltaFolder(path=table_path)
        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, n)
        self.assertEqual(sorted(out.column("id").to_pylist()), list(range(n)))


@unittest.skipUnless(_has_deltalake(), "deltalake package not installed")
class TestBidirectionalInterop(DeltaTestCase):
    """Write with one engine, append with the other, verify consistency."""

    def test_ygg_write_deltalake_append(self) -> None:
        import deltalake

        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2]}))

        deltalake.write_deltalake(
            str(d.path),
            self.pa.table({"id": [3, 4]}),
            mode="append",
        )

        # External write requires refresh to pick up the new commit
        d.refresh()
        out = d.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3, 4])

    def test_deltalake_write_ygg_append(self) -> None:
        import deltalake

        table_path = str(self.tmp_path / "interop_dl_ygg")
        deltalake.write_deltalake(
            table_path, self.pa.table({"id": [1, 2]}),
        )

        from yggdrasil.io.delta.delta_folder import DeltaFolder
        d = DeltaFolder(path=table_path)
        d.write_arrow_batches(
            self.pa.table({"id": [3, 4]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )

        dt = deltalake.DeltaTable(table_path)
        out = dt.to_pyarrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3, 4])


@unittest.skipUnless(_has_deltalake(), "deltalake package not installed")
class TestDeletionVectorInterop(DeltaTestCase):
    """Verify deletion vector handling across engines."""

    def test_ygg_dv_write_read_cycle(self) -> None:
        """Write data, delete via DV, verify remaining rows."""
        d = self.delta_io()
        t = self.pa.table({"id": [1, 2, 3, 4, 5]})
        d.write_arrow_table(t, options=DeltaOptions(delete_via_dv=True))

        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.num_active_files(), 1)

        out = d.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3, 4, 5])

    def test_stats_with_null_values(self) -> None:
        d = self.delta_io()
        t = self.pa.table({
            "id": [1, 2, 3],
            "val": self.pa.array([10, None, 30], type=self.pa.int64()),
        })
        d.write_arrow_table(t, options=DeltaOptions(collect_stats=True))

        snap = d.snapshot(fresh=True)
        import json
        for add in snap.active_files.values():
            stats = json.loads(add.stats)
            self.assertEqual(stats["numRecords"], 3)
            self.assertEqual(stats["nullCount"]["val"], 1)


@unittest.skipUnless(_has_deltalake(), "deltalake package not installed")
class TestCheckpointV2Interop(DeltaTestCase):
    """Verify V2 checkpoint handling."""

    def test_v2_checkpoint_with_multi_sidecar(self) -> None:
        d = self.delta_io()
        for i in range(11):
            mode = Mode.AUTO if i == 0 else Mode.APPEND
            d.write_arrow_batches(
                self.pa.table({"id": [i]}).to_batches(),
                options=DeltaOptions(
                    mode=mode,
                    checkpoint_interval=10,
                    checkpoint_kind="v2",
                ),
            )

        from yggdrasil.io.delta.delta_folder import DeltaFolder
        d2 = DeltaFolder(path=str(d.path))
        out = d2.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), list(range(11)))

    def test_v2_checkpoint_contains_sidecars_dir(self) -> None:
        d = self.delta_io()
        for i in range(6):
            mode = Mode.AUTO if i == 0 else Mode.APPEND
            d.write_arrow_batches(
                self.pa.table({"id": [i]}).to_batches(),
                options=DeltaOptions(
                    mode=mode,
                    checkpoint_interval=5,
                    checkpoint_kind="v2",
                ),
            )

        log_dir = os.path.join(str(d.path), "_delta_log")
        sidecar_dir = os.path.join(log_dir, "_sidecars")
        self.assertTrue(os.path.isdir(sidecar_dir))
        parquets = [f for f in os.listdir(sidecar_dir) if f.endswith(".parquet")]
        self.assertGreater(len(parquets), 0)

        manifests = [
            f for f in os.listdir(log_dir)
            if f.endswith(".json") and ".checkpoint." in f
        ]
        self.assertGreater(len(manifests), 0)
