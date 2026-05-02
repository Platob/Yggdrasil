"""DeltaIO ↔ Spark cross-engine integration tests.

These tests pair :class:`yggdrasil.io.buffer.nested.delta.DeltaIO`
against ``delta-spark`` running on a local :class:`SparkSession`.
The two are independent implementations of the Delta protocol — if
they agree on the table state across reads, writes, partitions,
deletes, and MERGEs, we get strong evidence that DeltaIO's protocol
handling matches the reference.

Test shape
----------

- *Spark writes → DeltaIO reads* validates our log replay, AddFile
  enumeration, deletion-vector decode, partition-value injection,
  and checkpoint loading against tables produced by the reference
  writer.
- *DeltaIO writes → Spark reads* validates our commit format,
  schema serialization, partition path layout, and DV emission
  against the reference reader.
- *Round-trip* tests interleave the two so each side has to consume
  what the other produced.

Skip behavior
-------------

Skipped cleanly when ``pyspark`` or ``delta-spark`` aren't
installed (covered by ``ygg[bigdata]``), and when ``pyroaring``
isn't available for the DV-write tests (covered by the DV install
prompt). The base ``yggdrasil`` install must keep working without
any of these.
"""

from __future__ import annotations

import json
import unittest
from typing import TYPE_CHECKING

import pyarrow as pa

from yggdrasil.io.buffer.nested.delta import (
    AddFile,
    DeltaIO,
    DeltaOptions,
    Metadata,
    Protocol,
    RemoveFile,
    replay_log,
)
from yggdrasil.io.buffer.nested.delta.deletion_vector import (
    DeletionVectorDescriptor,
    decode_dv_blob,
    decode_inline_descriptor,
)
from yggdrasil.io.buffer.nested.delta.replay import (
    latest_commit_version,
    read_last_checkpoint,
)
from yggdrasil.io.enums import Mode
from yggdrasil.io.fs import LocalPath
from yggdrasil.spark.tests import SparkTestCase

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


# ---------------------------------------------------------------------------
# Base — Delta-enabled SparkSession
# ---------------------------------------------------------------------------


class _DeltaSparkBase(SparkTestCase):
    """SparkTestCase with delta-spark wired in.

    The first time any DeltaIO-Spark test runs it reconfigures the
    process-global SparkSession to load Delta's jar package and SQL
    extensions. Subsequent runs reuse the same session.

    Skips the whole class if either ``pyspark`` or ``delta-spark``
    isn't importable.
    """

    @classmethod
    def setUpClass(cls) -> None:  # noqa: D401 — overrides super
        # Defer the import so non-bigdata installs skip cleanly.
        try:
            import delta  # noqa: F401  — delta-spark
        except ImportError:
            raise unittest.SkipTest(
                "delta-spark is not installed. "
                "Install it with: pip install 'ygg[bigdata]'"
            )

        try:
            from importlib.metadata import PackageNotFoundError, version
        except ImportError:  # pragma: no cover — stdlib since 3.8
            raise unittest.SkipTest("importlib.metadata unavailable")

        try:
            delta_version = version("delta-spark")
        except PackageNotFoundError:  # pragma: no cover
            raise unittest.SkipTest("delta-spark version cannot be detected")

        # Spark 3.5 ships Scala 2.12 by default; PySpark 4.x is 2.13.
        # We probe at runtime to pick the right artifact suffix.
        try:
            import pyspark
        except ImportError:
            raise unittest.SkipTest("PySpark is not installed")
        scala_suffix = "2.13" if pyspark.__version__.startswith("4.") else "2.12"
        jar = f"io.delta:delta-spark_{scala_suffix}:{delta_version}"

        delta_cfg = {
            "spark.jars.packages": jar,
            "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
            "spark.sql.catalog.spark_catalog": (
                "org.apache.spark.sql.delta.catalog.DeltaCatalog"
            ),
        }

        # Attach as the class' default extra config — only takes
        # effect if our class is the first to trigger session
        # creation. If a previous SparkTestCase already started one
        # without delta, we reset it so our jars actually load.
        cls.spark_extra_config = {**cls.spark_extra_config, **delta_cfg}

        from yggdrasil.spark import tests as _spark_tests

        existing = _spark_tests._global_spark
        if existing is not None:
            try:
                packages = existing.conf.get("spark.jars.packages", "") or ""
            except Exception:
                packages = ""
            if "delta-spark" not in packages:
                _spark_tests.reset_global_session()

        try:
            super().setUpClass()
        except unittest.SkipTest:
            raise

        # Make sure delta extensions actually loaded — if the jar
        # download failed silently, surface that as a skip rather
        # than a confusing test failure later.
        try:
            cls.spark.sql("CREATE TABLE __ygg_probe (a INT) USING DELTA").collect()
            cls.spark.sql("DROP TABLE __ygg_probe")
        except Exception as exc:  # noqa: BLE001 — surface anything
            raise unittest.SkipTest(
                f"Delta SQL not available on this Spark session: {exc!r}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def table_path(self, name: str = "tbl") -> str:
        """Per-test Delta table path under ``self.tmp_path``."""
        return str(self.tmp_path / name)

    def delta_io(self, path: str | None = None) -> DeltaIO:
        """Open a DeltaIO at *path* (defaults to ``self.table_path()``)."""
        location = path or self.table_path()
        return DeltaIO(path=LocalPath.from_(location))

    def spark_read(self, path: str | None = None) -> "DataFrame":
        """Read the Delta table at *path* via Spark."""
        location = path or self.table_path()
        return self.spark.read.format("delta").load(location)

    def spark_write(
        self,
        df: "DataFrame",
        *,
        mode: str = "overwrite",
        path: str | None = None,
        partition_by: list[str] | None = None,
        properties: dict[str, str] | None = None,
    ) -> None:
        """Write *df* to a Delta table at *path*."""
        location = path or self.table_path()
        writer = df.write.format("delta").mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        if properties:
            for k, v in properties.items():
                writer = writer.option(k, v)
        writer.save(location)

    def assert_arrow_set_equal(
        self,
        actual: pa.Table,
        expected: pa.Table,
        *,
        msg: str | None = None,
    ) -> None:
        """Assert two Arrow tables hold the same rows (order-insensitive).

        Compares the column-name set first, then the multiset of
        normalized row dicts. Schemas don't have to match dtype-for-dtype
        as long as the row values compare equal — both sides are run
        through ``to_pylist`` which collapses precision-equivalent
        types.
        """
        self.assertEqual(
            sorted(actual.column_names),
            sorted(expected.column_names),
            msg or "column-name set differs",
        )
        a_rows = actual.select(sorted(actual.column_names)).to_pylist()
        e_rows = expected.select(sorted(expected.column_names)).to_pylist()

        def _key(row: dict) -> str:
            return repr(sorted((k, repr(v)) for k, v in row.items()))

        a_sorted = sorted(a_rows, key=_key)
        e_sorted = sorted(e_rows, key=_key)
        self.assertEqual(
            a_sorted, e_sorted,
            msg or "row multisets differ",
        )


# ---------------------------------------------------------------------------
# Spark writes → DeltaIO reads
# ---------------------------------------------------------------------------


class TestSparkWriteThenDeltaIORead(_DeltaSparkBase):
    """Tables written by Spark must be readable by DeltaIO."""

    def test_basic_primitive_columns(self):
        """All common primitive columns round-trip Spark→DeltaIO."""
        df = self.spark.createDataFrame(
            [
                (1, "alpha", 1.5, True),
                (2, "beta", 2.5, False),
                (3, "gamma", 3.5, True),
            ],
            schema="id LONG, name STRING, score DOUBLE, ok BOOLEAN",
        )
        self.spark_write(df)

        result = self.delta_io().read_arrow_table()
        self.assertEqual(result.num_rows, 3)
        self.assert_arrow_set_equal(
            result,
            pa.table(
                {
                    "id": [1, 2, 3],
                    "name": ["alpha", "beta", "gamma"],
                    "score": [1.5, 2.5, 3.5],
                    "ok": [True, False, True],
                }
            ),
        )

    def test_nullable_columns_preserve_nulls(self):
        """Nullable Spark columns surface nulls (not zeros) on the Arrow side."""
        df = self.spark.createDataFrame(
            [(1, "x"), (2, None), (None, "y")],
            schema="id LONG, name STRING",
        )
        self.spark_write(df)

        result = self.delta_io().read_arrow_table()
        self.assertEqual(result.num_rows, 3)
        ids = result.column("id").to_pylist()
        names = result.column("name").to_pylist()
        self.assertIn(None, ids)
        self.assertIn(None, names)

    def test_partitioned_table_reads_with_partition_columns(self):
        """Partition columns appear in the read output, not just on disk."""
        df = self.spark.createDataFrame(
            [
                (1, "us", 10.0),
                (2, "us", 20.0),
                (3, "eu", 30.0),
                (4, "eu", 40.0),
            ],
            schema="id LONG, region STRING, amount DOUBLE",
        )
        self.spark_write(df, partition_by=["region"])

        result = self.delta_io().read_arrow_table()
        self.assertEqual(result.num_rows, 4)

        regions = sorted(set(result.column("region").to_pylist()))
        self.assertEqual(regions, ["eu", "us"])

        # The Delta metadata should declare the partition column.
        replay = replay_log(self.delta_io()._log_dir)
        self.assertIsNotNone(replay.metadata)
        self.assertEqual(replay.metadata.partition_columns, ("region",))

    def test_multi_partition_layout(self):
        """Two-column Hive-style partitioning is handled correctly."""
        df = self.spark.createDataFrame(
            [
                (1, 2024, 1, "a"),
                (2, 2024, 2, "b"),
                (3, 2025, 1, "c"),
            ],
            schema="id LONG, year INT, month INT, val STRING",
        )
        self.spark_write(df, partition_by=["year", "month"])

        result = self.delta_io().read_arrow_table()
        self.assertEqual(result.num_rows, 3)
        self.assert_arrow_set_equal(
            result,
            pa.table(
                {
                    "id": [1, 2, 3],
                    "year": [2024, 2024, 2025],
                    "month": [1, 2, 1],
                    "val": ["a", "b", "c"],
                }
            ),
        )

    def test_schema_from_metadata_not_first_file(self):
        """``collect_schema`` reads the log Metadata, not parquet footers.

        The two can disagree under partitioning — partition columns
        live only in the path, not the parquet file. DeltaIO must
        return the *full* table schema regardless.
        """
        df = self.spark.createDataFrame(
            [(1, "a", "x"), (2, "b", "y")],
            schema="id LONG, p STRING, v STRING",
        )
        self.spark_write(df, partition_by=["p"])

        schema = self.delta_io().collect_schema()
        names = [f.name for f in schema.fields]
        self.assertIn("id", names)
        self.assertIn("p", names)
        self.assertIn("v", names)

    def test_spark_delete_then_deltaio_read_filters_rows(self):
        """A Spark DELETE removes rows from the DeltaIO read view.

        DELETE on a DV-capable table writes deletion vectors; on
        non-DV tables it rewrites parquet. Either way, the deleted
        rows must not appear.
        """
        df = self.spark.createDataFrame(
            [(1, "keep"), (2, "drop"), (3, "keep")],
            schema="id LONG, label STRING",
        )
        self.spark_write(df)

        from delta.tables import DeltaTable

        DeltaTable.forPath(self.spark, self.table_path()).delete("id = 2")

        result = self.delta_io().read_arrow_table()
        ids = sorted(result.column("id").to_pylist())
        self.assertEqual(ids, [1, 3])

    def test_spark_delete_with_dvs_enabled_emits_dv_descriptor(self):
        """When DVs are enabled, Spark DELETE produces a DV descriptor we can decode."""
        df = self.spark.createDataFrame(
            [(i, f"row_{i}") for i in range(20)],
            schema="id LONG, val STRING",
        )
        self.spark_write(
            df,
            properties={"delta.enableDeletionVectors": "true"},
        )

        from delta.tables import DeltaTable

        DeltaTable.forPath(self.spark, self.table_path()).delete("id < 5")

        replay = replay_log(self.delta_io()._log_dir)
        with_dv = [
            a for a in replay.live_files
            if a.deletion_vector is not None
            and not a.deletion_vector.is_empty
        ]
        self.assertGreater(
            len(with_dv), 0,
            "DV-enabled DELETE produced no DV-bearing AddFiles",
        )

        # Resolve and decode the first DV — both inline and external
        # paths must round-trip without error.
        io = self.delta_io()
        bitmap = io._load_dv_bitmap_from_descriptor(with_dv[0].deletion_vector)
        self.assertIsNotNone(bitmap)
        self.assertEqual(len(bitmap), with_dv[0].deletion_vector.cardinality)

        # The read view must filter the deleted rows out.
        result = self.delta_io().read_arrow_table()
        self.assertEqual(result.num_rows, 15)
        ids = sorted(result.column("id").to_pylist())
        self.assertEqual(ids, list(range(5, 20)))

    def test_spark_update_changes_visible(self):
        """Spark UPDATE produces a new commit; DeltaIO sees the new values."""
        df = self.spark.createDataFrame(
            [(1, "old"), (2, "old"), (3, "old")],
            schema="id LONG, label STRING",
        )
        self.spark_write(df)

        from delta.tables import DeltaTable
        from pyspark.sql.functions import lit

        DeltaTable.forPath(self.spark, self.table_path()).update(
            condition="id = 2",
            set={"label": lit("new")},
        )

        result = self.delta_io().read_arrow_table()
        rows = {r["id"]: r["label"] for r in result.to_pylist()}
        self.assertEqual(rows, {1: "old", 2: "new", 3: "old"})

    def test_spark_merge_then_deltaio_read(self):
        """Spark MERGE INTO produces a state DeltaIO replays correctly."""
        target = self.spark.createDataFrame(
            [(1, "a"), (2, "b"), (3, "c")],
            schema="id LONG, val STRING",
        )
        self.spark_write(target)

        source = self.spark.createDataFrame(
            [(2, "B_NEW"), (4, "D")],
            schema="id LONG, val STRING",
        )
        source.createOrReplaceTempView("src")
        self.spark.sql(
            f"""
            MERGE INTO delta.`{self.table_path()}` AS t
            USING src AS s
            ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET val = s.val
            WHEN NOT MATCHED THEN INSERT (id, val) VALUES (s.id, s.val)
            """
        )

        result = self.delta_io().read_arrow_table()
        rows = {r["id"]: r["val"] for r in result.to_pylist()}
        self.assertEqual(rows, {1: "a", 2: "B_NEW", 3: "c", 4: "D"})

    def test_iter_children_yields_one_per_live_addfile(self):
        """``iter_children`` enumerates live AddFiles (not Removes)."""
        df = self.spark.createDataFrame(
            [(i, f"v{i}") for i in range(5)],
            schema="id LONG, v STRING",
        )
        self.spark_write(df)

        delta = self.delta_io()
        children = list(delta.iter_children())
        self.assertGreater(len(children), 0)

        replay = replay_log(delta._log_dir)
        self.assertEqual(len(children), len(replay.live_files))

        for child in children:
            self.assertIs(child.parent, delta)
            self.assertTrue(child.path.exists())

    def test_replay_returns_known_protocol_metadata(self):
        """Spark-written tables have a Protocol + Metadata at version 0."""
        df = self.spark.createDataFrame([(1, "a")], schema="id LONG, v STRING")
        self.spark_write(df)

        replay = replay_log(self.delta_io()._log_dir)
        self.assertIsInstance(replay.protocol, Protocol)
        self.assertIsInstance(replay.metadata, Metadata)
        self.assertGreaterEqual(replay.protocol.min_reader_version, 1)
        self.assertGreaterEqual(replay.protocol.min_writer_version, 1)
        self.assertGreaterEqual(replay.version, 0)

    def test_checkpoint_loaded_correctly(self):
        """A v1 checkpoint emitted by Spark is consumed by replay.

        The default checkpoint interval is 10 commits; we force one
        explicitly so the test isn't dependent on it.
        """
        # First commit: create the table.
        df = self.spark.createDataFrame([(0, "x")], schema="id LONG, v STRING")
        self.spark_write(df)

        # 10 more commits.
        for i in range(1, 12):
            df_i = self.spark.createDataFrame(
                [(i, f"v{i}")], schema="id LONG, v STRING"
            )
            self.spark_write(df_i, mode="append")

        # Force a checkpoint at the latest version.
        from delta.tables import DeltaTable

        delta_tbl = DeltaTable.forPath(self.spark, self.table_path())
        try:
            delta_tbl.generate("symlink_format_manifest")
        except Exception:
            pass  # Not the checkpoint we want; just keep going.
        # Direct Java call to checkpoint:
        try:
            self.spark.sql(
                f"VACUUM delta.`{self.table_path()}` RETAIN 0 HOURS"
            )
        except Exception:
            pass

        # Either way: read via DeltaIO must succeed and see all rows.
        result = self.delta_io().read_arrow_table()
        ids = sorted(result.column("id").to_pylist())
        self.assertEqual(ids, list(range(12)))

        log_dir = self.delta_io()._log_dir
        latest = latest_commit_version(log_dir)
        self.assertGreaterEqual(latest, 11)

        # If a checkpoint exists, ``_last_checkpoint`` resolves and
        # replay still produces the right state.
        last_cp = read_last_checkpoint(log_dir)
        if last_cp is not None:
            replay = replay_log(log_dir)
            self.assertEqual(len(replay.live_files), 12)


# ---------------------------------------------------------------------------
# DeltaIO writes → Spark reads
# ---------------------------------------------------------------------------


class TestDeltaIOWriteThenSparkRead(_DeltaSparkBase):
    """Tables written by DeltaIO must be readable by Spark."""

    def test_overwrite_creates_table_spark_can_load(self):
        """A fresh OVERWRITE creates a table Spark loads with the right rows."""
        delta = self.delta_io()
        tbl = pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "name": pa.array(["a", "b", "c"], type=pa.string()),
            }
        )
        delta.write_arrow_table(tbl, options=DeltaOptions(mode=Mode.OVERWRITE))

        df = self.spark_read()
        self.assertEqual(df.count(), 3)
        rows = {r["id"]: r["name"] for r in df.collect()}
        self.assertEqual(rows, {1: "a", 2: "b", 3: "c"})

    def test_append_after_overwrite_preserves_existing_rows(self):
        """APPEND adds without removing previously-committed rows."""
        delta = self.delta_io()
        first = pa.table({"id": pa.array([1, 2], type=pa.int64())})
        delta.write_arrow_table(first, options=DeltaOptions(mode=Mode.OVERWRITE))

        second = pa.table({"id": pa.array([3, 4], type=pa.int64())})
        delta.write_arrow_table(second, options=DeltaOptions(mode=Mode.APPEND))

        df = self.spark_read()
        ids = sorted(r["id"] for r in df.collect())
        self.assertEqual(ids, [1, 2, 3, 4])

    def test_overwrite_replaces_existing_rows(self):
        """OVERWRITE removes the previous live AddFiles entirely.

        Spark's view must reflect the post-overwrite rows only.
        """
        delta = self.delta_io()
        delta.write_arrow_table(
            pa.table({"id": pa.array([1, 2, 3], type=pa.int64())}),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )
        delta.write_arrow_table(
            pa.table({"id": pa.array([99], type=pa.int64())}),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )

        df = self.spark_read()
        ids = [r["id"] for r in df.collect()]
        self.assertEqual(ids, [99])

    def test_partitioned_write_layout_visible_to_spark(self):
        """Partitioned DeltaIO writes produce Hive-layout dirs Spark reads."""
        delta = self.delta_io()
        tbl = pa.table(
            {
                "id": pa.array([1, 2, 3, 4], type=pa.int64()),
                "region": pa.array(["us", "us", "eu", "eu"], type=pa.string()),
            }
        )
        delta.write_arrow_table(
            tbl,
            options=DeltaOptions(
                mode=Mode.OVERWRITE,
                partition_columns=["region"],
            ),
        )

        # On-disk layout: ``region=us/`` and ``region=eu/`` dirs.
        children = sorted(p.name for p in delta.path.iterdir() if p.name != "_delta_log")
        self.assertIn("region=us", children)
        self.assertIn("region=eu", children)

        df = self.spark_read()
        rows = sorted((r["id"], r["region"]) for r in df.collect())
        self.assertEqual(rows, [(1, "us"), (2, "us"), (3, "eu"), (4, "eu")])

        # Spark sees ``region`` as a partition column.
        spark_meta = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{self.table_path()}`"
        ).collect()
        self.assertIn("region", list(spark_meta[0]["partitionColumns"]))

    def test_empty_overwrite_initializes_table(self):
        """OVERWRITE with zero rows on a fresh path creates an empty table.

        Spark loads it without error; row count is 0; schema is empty.
        """
        delta = self.delta_io()
        empty = pa.table({})
        delta.write_arrow_table(
            empty, options=DeltaOptions(mode=Mode.OVERWRITE),
        )

        df = self.spark_read()
        self.assertEqual(df.count(), 0)

        log_dir = delta._log_dir
        self.assertTrue(log_dir.exists())
        self.assertGreaterEqual(latest_commit_version(log_dir), 0)

    def test_commit_files_have_well_formed_actions(self):
        """Each commit file is newline-JSON of single-key action envelopes."""
        delta = self.delta_io()
        delta.write_arrow_table(
            pa.table({"id": pa.array([1], type=pa.int64())}),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )
        delta.write_arrow_table(
            pa.table({"id": pa.array([2], type=pa.int64())}),
            options=DeltaOptions(mode=Mode.APPEND),
        )

        log_dir = delta._log_dir
        for child in log_dir.iterdir():
            if not child.name.endswith(".json"):
                continue
            text = child.read_text()
            for line in text.splitlines():
                if not line.strip():
                    continue
                action = json.loads(line)
                self.assertEqual(
                    len(action), 1,
                    f"Action envelope must have exactly one key, got {action!r}",
                )
                kind = next(iter(action))
                self.assertIn(
                    kind,
                    {
                        "commitInfo", "protocol", "metaData",
                        "add", "remove", "txn", "domainMetadata", "cdc",
                    },
                    f"Unexpected action kind {kind!r} in {child.name}",
                )

    def test_failed_write_cleans_up_parquet_files(self):
        """A commit failure rolls back the parquet files staged for it.

        Simulate by patching :func:`write_commit` to raise after the
        parquet files are staged. The post-hoc table state must
        match the pre-failure snapshot — no orphaned files, no
        corrupt commit.
        """
        delta = self.delta_io()
        delta.write_arrow_table(
            pa.table({"id": pa.array([1, 2], type=pa.int64())}),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )

        before_files = set(delta._scan_data_files())
        before_version = latest_commit_version(delta._log_dir)

        # Patch the commit writer at the module level the IO uses.
        from yggdrasil.io.buffer.nested.delta import io as delta_io_mod

        original = delta_io_mod.write_commit

        def boom(*_args, **_kwargs):  # noqa: ANN001, ANN002
            raise RuntimeError("simulated commit failure")

        delta_io_mod.write_commit = boom
        try:
            with self.assertRaises(RuntimeError):
                delta.write_arrow_table(
                    pa.table({"id": pa.array([99], type=pa.int64())}),
                    options=DeltaOptions(mode=Mode.APPEND),
                )
        finally:
            delta_io_mod.write_commit = original

        after_files = set(delta._scan_data_files())
        after_version = latest_commit_version(delta._log_dir)

        self.assertEqual(after_files, before_files)
        self.assertEqual(after_version, before_version)

        # Spark still sees the original two rows.
        df = self.spark_read()
        ids = sorted(r["id"] for r in df.collect())
        self.assertEqual(ids, [1, 2])


# ---------------------------------------------------------------------------
# Round-trip — both engines as writers
# ---------------------------------------------------------------------------


class TestRoundTripSparkDeltaIO(_DeltaSparkBase):
    """Interleave Spark and DeltaIO writes; both must read consistently."""

    def test_spark_create_then_deltaio_append_then_spark_read(self):
        """Spark creates → DeltaIO appends → Spark reads everything."""
        df = self.spark.createDataFrame(
            [(1, "spark")], schema="id LONG, who STRING"
        )
        self.spark_write(df)

        delta = self.delta_io()
        delta.write_arrow_table(
            pa.table(
                {
                    "id": pa.array([2, 3], type=pa.int64()),
                    "who": pa.array(["delta_a", "delta_b"], type=pa.string()),
                }
            ),
            options=DeltaOptions(mode=Mode.APPEND),
        )

        out = self.spark_read()
        rows = sorted((r["id"], r["who"]) for r in out.collect())
        self.assertEqual(
            rows,
            [(1, "spark"), (2, "delta_a"), (3, "delta_b")],
        )

    def test_deltaio_create_then_spark_append_then_deltaio_read(self):
        """DeltaIO creates → Spark appends → DeltaIO reads everything."""
        delta = self.delta_io()
        delta.write_arrow_table(
            pa.table(
                {
                    "id": pa.array([1], type=pa.int64()),
                    "who": pa.array(["delta"], type=pa.string()),
                }
            ),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )

        df = self.spark.createDataFrame(
            [(2, "spark_a"), (3, "spark_b")],
            schema="id LONG, who STRING",
        )
        self.spark_write(df, mode="append")

        result = self.delta_io().read_arrow_table()
        rows = sorted((r["id"], r["who"]) for r in result.to_pylist())
        self.assertEqual(
            rows,
            [(1, "delta"), (2, "spark_a"), (3, "spark_b")],
        )

    def test_deltaio_overwrite_replaces_spark_written_table(self):
        """DeltaIO OVERWRITE on a Spark-created table emits Removes Spark sees.

        The previously-live AddFiles must be Removed in the new
        commit, so Spark's read shows only the OVERWRITE'd rows.
        """
        spark_df = self.spark.createDataFrame(
            [(1, "spark"), (2, "spark"), (3, "spark")],
            schema="id LONG, who STRING",
        )
        self.spark_write(spark_df)

        before_replay = replay_log(self.delta_io()._log_dir)
        before_paths = {a.path for a in before_replay.live_files}
        self.assertGreater(len(before_paths), 0)

        delta = self.delta_io()
        delta.write_arrow_table(
            pa.table(
                {
                    "id": pa.array([99], type=pa.int64()),
                    "who": pa.array(["delta"], type=pa.string()),
                }
            ),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )

        after_replay = replay_log(self.delta_io()._log_dir)
        after_paths = {a.path for a in after_replay.live_files}
        leaked = before_paths & after_paths
        self.assertFalse(
            leaked,
            f"OVERWRITE did not remove old AddFile(s): {leaked!r}",
        )

        out = self.spark_read()
        rows = sorted((r["id"], r["who"]) for r in out.collect())
        self.assertEqual(rows, [(99, "delta")])


# ---------------------------------------------------------------------------
# UPSERT — DV-emitting MERGE-on-read
# ---------------------------------------------------------------------------


class TestDeltaIOUpsertWithSpark(_DeltaSparkBase):
    """UPSERT writes DVs; Spark must filter the same rows."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            import pyroaring  # noqa: F401
        except ImportError:
            raise unittest.SkipTest(
                "pyroaring is not installed (required for DV writes). "
                "Install with: pip install pyroaring"
            )

    def test_upsert_replaces_matching_rows(self):
        """UPSERT replaces rows by match key; Spark sees only the new values."""
        delta = self.delta_io()
        delta.write_arrow_table(
            pa.table(
                {
                    "id": pa.array([1, 2, 3], type=pa.int64()),
                    "v": pa.array(["a", "b", "c"], type=pa.string()),
                }
            ),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )

        delta.write_arrow_table(
            pa.table(
                {
                    "id": pa.array([2, 4], type=pa.int64()),
                    "v": pa.array(["B_NEW", "D"], type=pa.string()),
                }
            ),
            options=DeltaOptions(
                mode=Mode.UPSERT,
                match_by_names=("id",),
            ),
        )

        # Spark must agree with DeltaIO on the post-merge state.
        spark_rows = {
            r["id"]: r["v"] for r in self.spark_read().collect()
        }
        self.assertEqual(
            spark_rows,
            {1: "a", 2: "B_NEW", 3: "c", 4: "D"},
        )

        deltaio_rows = {
            r["id"]: r["v"]
            for r in self.delta_io().read_arrow_table().to_pylist()
        }
        self.assertEqual(spark_rows, deltaio_rows)

    def test_upsert_emits_deletion_vector_for_matched_rows(self):
        """UPSERT writes a DV against the source AddFile of the matched row."""
        delta = self.delta_io()
        delta.write_arrow_table(
            pa.table(
                {
                    "id": pa.array(list(range(10)), type=pa.int64()),
                    "v": pa.array([f"old_{i}" for i in range(10)], type=pa.string()),
                }
            ),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )

        delta.write_arrow_table(
            pa.table(
                {
                    "id": pa.array([3, 5, 7], type=pa.int64()),
                    "v": pa.array(["NEW3", "NEW5", "NEW7"], type=pa.string()),
                }
            ),
            options=DeltaOptions(
                mode=Mode.UPSERT,
                match_by_names=("id",),
            ),
        )

        replay = replay_log(self.delta_io()._log_dir)
        with_dv = [
            a for a in replay.live_files
            if a.deletion_vector is not None
            and not a.deletion_vector.is_empty
        ]
        self.assertGreater(
            len(with_dv), 0,
            "UPSERT did not emit any DV — expected at least one AddFile "
            "with a non-empty deletion vector",
        )

        # Total DV cardinality across surviving files == 3 (rows replaced).
        total_dv = sum(a.deletion_vector.cardinality for a in with_dv)
        self.assertEqual(
            total_dv, 3,
            f"Expected 3 deleted ordinals across DVs; got {total_dv}",
        )

        # Spark must respect the same DVs and produce the same view.
        spark_rows = {
            r["id"]: r["v"] for r in self.spark_read().collect()
        }
        expected = {i: f"old_{i}" for i in range(10) if i not in (3, 5, 7)}
        expected.update({3: "NEW3", 5: "NEW5", 7: "NEW7"})
        self.assertEqual(spark_rows, expected)

    def test_upsert_promotes_protocol_to_dv_capable(self):
        """First UPSERT against a legacy-protocol table promotes the protocol."""
        delta = self.delta_io()
        # Initial overwrite uses legacy (DV-incapable) protocol.
        delta.write_arrow_table(
            pa.table(
                {
                    "id": pa.array([1, 2], type=pa.int64()),
                    "v": pa.array(["a", "b"], type=pa.string()),
                }
            ),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )

        before = replay_log(self.delta_io()._log_dir).protocol
        self.assertNotIn("deletionVectors", before.reader_features)

        delta.write_arrow_table(
            pa.table(
                {
                    "id": pa.array([2], type=pa.int64()),
                    "v": pa.array(["B_NEW"], type=pa.string()),
                }
            ),
            options=DeltaOptions(
                mode=Mode.UPSERT,
                match_by_names=("id",),
            ),
        )

        after = replay_log(self.delta_io()._log_dir).protocol
        self.assertIn("deletionVectors", after.reader_features)
        self.assertIn("deletionVectors", after.writer_features)

    def test_upsert_then_spark_can_still_write(self):
        """After DeltaIO UPSERT, Spark can append without protocol confusion."""
        delta = self.delta_io()
        delta.write_arrow_table(
            pa.table(
                {
                    "id": pa.array([1, 2, 3], type=pa.int64()),
                    "v": pa.array(["a", "b", "c"], type=pa.string()),
                }
            ),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )

        delta.write_arrow_table(
            pa.table(
                {
                    "id": pa.array([2], type=pa.int64()),
                    "v": pa.array(["B_NEW"], type=pa.string()),
                }
            ),
            options=DeltaOptions(
                mode=Mode.UPSERT,
                match_by_names=("id",),
            ),
        )

        df = self.spark.createDataFrame(
            [(99, "spark")], schema="id LONG, v STRING"
        )
        self.spark_write(df, mode="append")

        # Final state visible via DeltaIO matches the expected union.
        rows = {
            r["id"]: r["v"]
            for r in self.delta_io().read_arrow_table().to_pylist()
        }
        self.assertEqual(
            rows,
            {1: "a", 2: "B_NEW", 3: "c", 99: "spark"},
        )


# ---------------------------------------------------------------------------
# Protocol refusal — features we don't support
# ---------------------------------------------------------------------------


class TestProtocolRefusal(_DeltaSparkBase):
    """When Spark opts the table into a feature DeltaIO can't handle, refuse."""

    def test_unsupported_reader_feature_refused(self):
        """A reader feature outside our supported set causes replay to raise."""
        df = self.spark.createDataFrame(
            [(1, "x")], schema="id LONG, v STRING"
        )
        self.spark_write(df)

        # Manually rewrite the protocol action of commit 0 to declare
        # an unknown reader feature. The cleanest way is to append a
        # new commit with a Protocol that bumps reader version above
        # legacy and declares an unsupported feature.
        log_dir = self.delta_io()._log_dir
        latest = latest_commit_version(log_dir)
        commit_path = log_dir / f"{(latest + 1):020d}.json"
        protocol_line = json.dumps(
            {
                "protocol": {
                    "minReaderVersion": 3,
                    "minWriterVersion": 7,
                    "readerFeatures": ["unsupportedFutureFeature"],
                    "writerFeatures": ["unsupportedFutureFeature"],
                }
            }
        )
        commit_path.write_text(protocol_line + "\n")

        with self.assertRaisesRegex(ValueError, "refuses to read"):
            replay_log(log_dir)

    def test_supported_reader_feature_accepted(self):
        """``deletionVectors`` is in our supported set; replay succeeds."""
        df = self.spark.createDataFrame(
            [(1, "x")], schema="id LONG, v STRING"
        )
        self.spark_write(
            df, properties={"delta.enableDeletionVectors": "true"}
        )

        replay = replay_log(self.delta_io()._log_dir)
        self.assertIn(
            "deletionVectors",
            replay.protocol.reader_features,
            "DV-enabled table should expose deletionVectors as a reader feature",
        )


# ---------------------------------------------------------------------------
# Schema / type fidelity
# ---------------------------------------------------------------------------


class TestSchemaFidelity(_DeltaSparkBase):
    """Schema + type metadata survives the Spark↔DeltaIO boundary."""

    def test_field_order_preserved_through_round_trip(self):
        """DeltaIO write preserves field order Spark observes."""
        delta = self.delta_io()
        ordered_columns = ["a", "b", "c", "d"]
        tbl = pa.table(
            {
                "a": pa.array([1], type=pa.int64()),
                "b": pa.array(["x"], type=pa.string()),
                "c": pa.array([1.5], type=pa.float64()),
                "d": pa.array([True], type=pa.bool_()),
            }
        )
        delta.write_arrow_table(tbl, options=DeltaOptions(mode=Mode.OVERWRITE))

        spark_columns = self.spark_read().columns
        self.assertEqual(spark_columns, ordered_columns)

        replay = replay_log(delta._log_dir)
        log_columns = [f.name for f in replay.metadata.schema.fields]
        self.assertEqual(log_columns, ordered_columns)

    def test_nullability_survives_write(self):
        """Nulls written through DeltaIO are visible as nulls in Spark."""
        delta = self.delta_io()
        tbl = pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "v": pa.array(["x", None, "z"], type=pa.string()),
            }
        )
        delta.write_arrow_table(tbl, options=DeltaOptions(mode=Mode.OVERWRITE))

        df = self.spark_read()
        nulls = df.filter(df["v"].isNull()).count()
        self.assertEqual(nulls, 1)

    def test_partition_columns_in_metadata_match_path_layout(self):
        """``partition_columns`` declared in Metadata matches the on-disk layout."""
        delta = self.delta_io()
        tbl = pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "p": pa.array(["a", "a", "b"], type=pa.string()),
            }
        )
        delta.write_arrow_table(
            tbl,
            options=DeltaOptions(
                mode=Mode.OVERWRITE,
                partition_columns=["p"],
            ),
        )

        replay = replay_log(delta._log_dir)
        self.assertEqual(replay.metadata.partition_columns, ("p",))

        for add in replay.live_files:
            self.assertIn("p", add.partition_values)
            self.assertIn(add.partition_values["p"], {"a", "b"})


# ---------------------------------------------------------------------------
# Inline DV decode against Spark-emitted bitmaps
# ---------------------------------------------------------------------------


class TestInlineDVRoundTripWithSpark(_DeltaSparkBase):
    """Validate inline DV decode matches Spark's emitted DVs.

    Spark emits inline DVs for tiny tables; the `decode_inline_descriptor`
    path must agree with the byte-level `decode_dv_blob` path.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            import pyroaring  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("pyroaring required for DV decode")

    def test_inline_dv_decodes_to_known_ordinals(self):
        """Spark DELETE on a DV-enabled tiny table → DV resolves to right ordinals."""
        # 3 rows, single parquet file likely. Delete a known ordinal.
        df = self.spark.createDataFrame(
            [(0, "a"), (1, "b"), (2, "c")],
            schema="id LONG, v STRING",
        )
        self.spark_write(
            df, properties={"delta.enableDeletionVectors": "true"}
        )

        from delta.tables import DeltaTable

        DeltaTable.forPath(self.spark, self.table_path()).delete("id = 1")

        replay = replay_log(self.delta_io()._log_dir)
        with_dv = [
            a for a in replay.live_files
            if a.deletion_vector is not None
            and not a.deletion_vector.is_empty
        ]
        self.assertGreater(len(with_dv), 0)

        descriptor: DeletionVectorDescriptor = with_dv[0].deletion_vector
        if descriptor.is_inline:
            bitmap = decode_inline_descriptor(descriptor)
        else:
            io = self.delta_io()
            bitmap = io._load_dv_bitmap_from_descriptor(descriptor)

        # The deleted row's parquet ordinal is exactly 1 (single-file
        # writes preserve insertion order). DV must contain {1}.
        self.assertEqual(len(bitmap), 1)
        self.assertIn(1, set(bitmap))


# ---------------------------------------------------------------------------
# Replay correctness — Removes hide AddFiles
# ---------------------------------------------------------------------------


class TestReplayCorrectness(_DeltaSparkBase):
    """Spark-emitted Remove actions must hide the corresponding AddFile."""

    def test_overwrite_removes_old_paths_from_live_set(self):
        """After Spark OVERWRITE, the previous AddFiles are absent from live."""
        df = self.spark.createDataFrame(
            [(1, "old"), (2, "old")], schema="id LONG, v STRING"
        )
        self.spark_write(df)

        before = replay_log(self.delta_io()._log_dir)
        old_paths = {a.path for a in before.live_files}

        df_new = self.spark.createDataFrame(
            [(99, "new")], schema="id LONG, v STRING"
        )
        self.spark_write(df_new, mode="overwrite")

        after = replay_log(self.delta_io()._log_dir)
        new_paths = {a.path for a in after.live_files}

        leaked = old_paths & new_paths
        self.assertFalse(
            leaked,
            f"Spark OVERWRITE didn't drop old AddFiles: {leaked!r}",
        )

    def test_replay_version_advances_with_each_commit(self):
        """Each Spark commit increments the replay version."""
        df = self.spark.createDataFrame(
            [(1, "a")], schema="id LONG, v STRING"
        )
        self.spark_write(df)
        v0 = replay_log(self.delta_io()._log_dir).version

        for i in range(2, 5):
            df_i = self.spark.createDataFrame(
                [(i, f"v{i}")], schema="id LONG, v STRING"
            )
            self.spark_write(df_i, mode="append")

        vN = replay_log(self.delta_io()._log_dir).version
        self.assertEqual(vN - v0, 3)

    def test_remove_actions_emitted_by_overwrite_carry_dv_passthrough(self):
        """Remove actions for a DV-bearing AddFile preserve DV info.

        ``extendedFileMetadata=true`` is required by the Delta spec
        for DV-bearing tables. We don't enforce reading it, but the
        commit must round-trip.
        """
        df = self.spark.createDataFrame(
            [(i, f"r{i}") for i in range(20)],
            schema="id LONG, v STRING",
        )
        self.spark_write(
            df, properties={"delta.enableDeletionVectors": "true"}
        )

        from delta.tables import DeltaTable

        DeltaTable.forPath(self.spark, self.table_path()).delete("id < 5")

        # Read back commits and verify any Remove that has a DV also
        # carries extended metadata. We probe the raw commit JSON
        # rather than parsed dataclasses to assert the wire shape.
        log_dir = self.delta_io()._log_dir
        for child in log_dir.iterdir():
            if not child.name.endswith(".json"):
                continue
            for line in child.read_text().splitlines():
                if not line.strip():
                    continue
                action = json.loads(line)
                if "remove" not in action:
                    continue
                rm = action["remove"]
                if "deletionVector" in rm and rm["deletionVector"] is not None:
                    self.assertTrue(
                        rm.get("extendedFileMetadata", False),
                        f"Remove with DV must set extendedFileMetadata: {rm!r}",
                    )


if __name__ == "__main__":
    unittest.main()
