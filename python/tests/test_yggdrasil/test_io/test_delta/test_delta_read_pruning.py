"""Column-pushdown + snapshot-self-contained read tests for DeltaFolder.

These run fully offline against a local-filesystem Delta table
(:meth:`DeltaTestCase.new_table`). They lock in:

- ``read_arrow_table(columns=[...])`` projects *before* the parquet read
  (the leaf is opened with a ``columns=`` projection), not after — a
  regression that re-reads every column off disk fails the spy assertion.
- A predicate column that the projection dropped still survives the read,
  filters correctly, and is itself projected away from the output.
- ``scan_polars_frame`` is lazy, re-collectable, pins the snapshot version
  (a write after building the scan doesn't change what the earlier scan
  collects), and honours projection / predicate / head pushdown.
- The Spark frame (when pyspark is importable) reads the projected columns
  and the correct rows.
"""

from __future__ import annotations

import unittest

from yggdrasil.enums import Mode
from yggdrasil.execution.expr import col
from yggdrasil.io import parquet_file as _pfmod
from yggdrasil.io.delta.tests import DeltaTestCase


class _WideTableMixin:
    """Seed a wide local Delta table: a few narrow columns + a fat string."""

    def _wide_table(self, rows: int = 200):
        pa = self.pa
        return pa.table({
            "a": pa.array(range(rows)),
            "b": pa.array(range(rows, 2 * rows)),
            "fat": pa.array(["payload-" + "x" * 64 + f"-{i}" for i in range(rows)]),
            "c": pa.array([float(i) for i in range(rows)]),
        })


class TestArrowColumnPushdown(_WideTableMixin, DeltaTestCase):
    def test_single_column_returns_only_that_column(self) -> None:
        d = self.new_table(self._wide_table(), name="wide")
        out = d.read_arrow_table(columns=["a"])
        self.assertEqual(out.schema.names, ["a"])
        self.assertEqual(out.num_rows, 200)

    def test_projection_pushed_into_parquet_leaf(self) -> None:
        # Spy on the parquet projection helper: it must receive the ``a``
        # projection so only that column chunk is decoded off disk. A
        # regression that reshapes after a full read leaves this ``None``.
        d = self.new_table(self._wide_table(), name="wide")
        seen: list = []
        original = _pfmod.ParquetFile._projection_columns

        def spy(options, names):
            result = original(options, names)
            seen.append(result)
            return result

        _pfmod.ParquetFile._projection_columns = staticmethod(spy)
        try:
            d.read_arrow_table(columns=["a"])
        finally:
            _pfmod.ParquetFile._projection_columns = staticmethod(original)

        self.assertTrue(
            any(s == ["a"] for s in seen if s is not None),
            f"parquet leaf was not read with a columns= projection: {seen!r}",
        )

    def test_no_projection_reads_every_column(self) -> None:
        d = self.new_table(self._wide_table(), name="wide")
        out = d.read_arrow_table()
        self.assertEqual(set(out.schema.names), {"a", "b", "fat", "c"})
        self.assertEqual(out.num_rows, 200)

    def test_predicate_on_non_projected_column_survives_read(self) -> None:
        # Project only ``a`` but filter on ``b`` (dropped from the output):
        # ``b`` must survive the read so the row filter sees it, then be
        # projected away. ``b`` runs 200..399, so ``b > 350`` keeps 49 rows
        # (b in 351..399).
        d = self.new_table(self._wide_table(), name="wide")
        out = d.read_arrow_table(columns=["a"], predicate=col("b") > 350)
        self.assertEqual(out.schema.names, ["a"])
        self.assertEqual(out.num_rows, 49)
        # The surviving ``a`` values are exactly those whose ``b`` matched:
        # row i has a==i and b==200+i, so b>350 ⇒ a in 151..199.
        self.assertEqual(out.column("a").to_pylist(), list(range(151, 200)))


class TestPolarsAndPandasPruning(_WideTableMixin, DeltaTestCase):
    def test_read_polars_frame_projection(self) -> None:
        d = self.new_table(self._wide_table(), name="wide")
        frame = d.read_polars_frame(columns=["a"])
        self.assertEqual(frame.columns, ["a"])
        self.assertEqual(frame.height, 200)

    def test_read_pandas_frame_projection(self) -> None:
        d = self.new_table(self._wide_table(), name="wide")
        df = d.read_pandas_frame(columns=["c"])
        self.assertEqual(list(df.columns), ["c"])
        self.assertEqual(len(df), 200)


class TestScanPolarsSelfContained(_WideTableMixin, DeltaTestCase):
    def test_scan_is_lazy_and_recollectable(self) -> None:
        d = self.new_table(self._wide_table(), name="wide")
        scan = d.scan_polars_frame()
        # Re-collectable: two collects of the same scan agree.
        self.assertEqual(scan.collect().height, 200)
        self.assertEqual(scan.collect().height, 200)

    def test_scan_projection_pushdown(self) -> None:
        d = self.new_table(self._wide_table(), name="wide")
        scan = d.scan_polars_frame()
        self.assertEqual(scan.select("a").collect().columns, ["a"])

    def test_scan_head_pushdown(self) -> None:
        d = self.new_table(self._wide_table(), name="wide")
        scan = d.scan_polars_frame()
        self.assertEqual(scan.head(7).collect().height, 7)

    def test_scan_predicate_pushdown(self) -> None:
        from yggdrasil.lazy_imports import polars_module
        pl = polars_module()
        d = self.new_table(self._wide_table(), name="wide")
        scan = d.scan_polars_frame()
        kept = scan.filter(pl.col("b") > 350).collect()
        self.assertEqual(kept.height, 49)

    def test_scan_pins_snapshot_version(self) -> None:
        # Build the scan, then write more rows. The earlier-built scan must
        # still collect the *pinned* view (200 rows), proving it closed over
        # the snapshot, not the folder's live state. A freshly-built scan
        # sees the new data.
        d = self.new_table(self._wide_table(), name="wide")
        scan = d.scan_polars_frame()
        self.assertEqual(scan.collect().height, 200)

        d.write_arrow_table(self._wide_table(rows=10), mode=Mode.APPEND)

        self.assertEqual(scan.collect().height, 200)
        self.assertEqual(d.scan_polars_frame().collect().height, 210)


def _has_pyspark() -> bool:
    try:
        import pyspark  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_has_pyspark(), "pyspark not installed")
class TestSparkPruning(_WideTableMixin, DeltaTestCase):
    """Spark ``mapInArrow`` read path: projected columns + correct rows.

    Boots the shared local-JVM session via the spark test helper and passes
    it through ``spark_session=``. Skips cleanly when the session can't be
    created in this environment (e.g. a Connect-only setup missing grpcio).
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            from yggdrasil.spark.tests import _get_test_spark
            cls._spark = _get_test_spark(app_name="ygg-delta-pruning-test")
        except Exception as exc:  # pragma: no cover - env dependent
            raise unittest.SkipTest(f"no usable SparkSession: {exc}")

    def test_spark_reads_projected_columns(self) -> None:
        d = self.new_table(self._wide_table(rows=50), name="wide")
        frame = d.read_spark_frame(columns=["a"], spark_session=self._spark)
        self.assertEqual(frame.columns, ["a"])
        rows = frame.collect()
        self.assertEqual(len(rows), 50)
        self.assertEqual(sorted(r["a"] for r in rows), list(range(50)))

    def test_spark_projection_with_predicate_column(self) -> None:
        # A predicate on the non-projected ``b`` column must not break the
        # projected read: the leaf is read with ``columns=`` covering ``b``
        # (so data-skipping can run) while the result frame still narrows to
        # ``a``. The Delta spark path prunes at the file level, not the row
        # level, so every surviving row's ``a`` comes through.
        d = self.new_table(self._wide_table(rows=50), name="wide")
        frame = d.read_spark_frame(
            columns=["a"], predicate=col("b") > 75, spark_session=self._spark,
        )
        self.assertEqual(frame.columns, ["a"])
        rows = frame.collect()
        self.assertEqual(sorted(r["a"] for r in rows), list(range(50)))
