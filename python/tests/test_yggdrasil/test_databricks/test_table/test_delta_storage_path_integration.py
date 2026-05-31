"""Integration benchmark: SQL warehouse vs direct DeltaFolder over the storage path.

A Databricks-managed Delta table is, on disk, a folder of parquet files
plus a ``_delta_log`` transaction log. Two ways to read it:

1. **SQL warehouse path** — the canonical route. ``SELECT * FROM ...``
   over a serverless / pro warehouse, payload comes back as Arrow via
   the statement-result stream. Drives all of Databricks' compute
   features (auth, caching, photon-vectorised execution, row-level
   security, etc.) but pays one warehouse round trip per query.

2. **Direct storage read via** :class:`yggdrasil.io.delta.DeltaFolder`
   — point :class:`DeltaFolder` at the table's :attr:`storage_location`
   (an S3 / ABFS / GCS URI vended by Unity Catalog's temporary table
   credentials API) and open the parquet files directly. Skips the
   warehouse entirely and lets the predicate AST +
   ``extract_partition_filters`` drive partition pruning at the file
   level — same machinery the local-only delta tests exercise.

This file is **not** a microbenchmark — it's a correctness suite that
*also* prints wall-clock timings so a reviewer can see whether the
direct read is competitive with the warehouse on this workspace /
network. The numbers come back annotated with row counts and file
counts so they're comparable across runs.

Skip rules
----------

The whole module is gated by :class:`DatabricksIntegrationCase` —
:envvar:`DATABRICKS_HOST` must be set. Additionally:

- Reading the storage path requires temporary table credentials
  (managed tables only on most clouds, AWS-only on some). The fixture
  skips cleanly with :class:`unittest.SkipTest` when those creds can't
  be vended.

Cleanup
-------

The fixture creates one per-class throw-away table named
``yg_delta_path_<hex>`` partitioned by ``region``, inserts seed rows
in one shot, and drops the table cascade-style in ``tearDownClass``.
"""

from __future__ import annotations

import os
import secrets
import time
import unittest
from typing import Any, ClassVar

import pyarrow as pa
import pytest
from databricks.sdk.errors import DatabricksError, NotFound, PermissionDenied

from yggdrasil.data import Field
from yggdrasil.enums import Mode
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import Int64Type, StringType
from yggdrasil.databricks.table.table import Table
from yggdrasil.io.delta import DeltaFolder, DeltaOptions
from yggdrasil.execution.expr import col as expr_col

from .. import DatabricksIntegrationCase


__all__ = ["TestDeltaStoragePathBenchmark"]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


REGIONS = ("us", "eu", "uk", "ap", "br", "ca", "fr", "de")


def _resolve_catalog() -> str:
    name = os.environ.get(
        "DATABRICKS_INTEGRATION_CATALOG", "trading",
    ).strip()
    if not name:
        raise unittest.SkipTest(
            "DATABRICKS_INTEGRATION_CATALOG is empty — set it to a "
            "catalog the test identity has CREATE TABLE on."
        )
    return name


def _resolve_schema() -> str:
    name = os.environ.get(
        "DATABRICKS_INTEGRATION_SCHEMA", "unittest",
    ).strip()
    if not name:
        raise unittest.SkipTest(
            "DATABRICKS_INTEGRATION_SCHEMA is empty — set it to a "
            "schema the test identity has CREATE TABLE on."
        )
    return name


def _resolve_rows() -> int:
    """Row count for the seeded table.

    Pulled from :envvar:`DATABRICKS_DELTA_BENCH_ROWS` so a developer
    can crank it up locally without editing the test. Default 5_000
    keeps the CI cost small while still giving the SQL vs storage
    comparison meaningful signal.
    """
    raw = os.environ.get("DATABRICKS_DELTA_BENCH_ROWS", "5000")
    try:
        return max(1, int(raw))
    except ValueError:
        raise unittest.SkipTest(
            f"DATABRICKS_DELTA_BENCH_ROWS={raw!r} is not an integer."
        )


def _seed_table_arrow(rows: int) -> pa.Table:
    """Two columns + a partition column, deterministic so reads diff cleanly."""
    return pa.table(
        {
            "id": pa.array(range(rows), type=pa.int64()),
            "region": pa.array(
                [REGIONS[i % len(REGIONS)] for i in range(rows)], type=pa.string(),
            ),
            "val": pa.array(
                [f"row-{i}" for i in range(rows)], type=pa.string(),
            ),
        }
    )


def _partitioned_schema() -> Schema:
    s = Schema()
    s.with_field(Field(name="id", dtype=Int64Type(), nullable=False))
    s.with_field(
        Field(name="region", dtype=StringType()).with_partition_by(True)
    )
    s.with_field(Field(name="val", dtype=StringType()))
    return s


def _time_block(fn) -> tuple[Any, float]:
    """Run *fn* once, return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def _fmt_seconds(secs: float) -> str:
    if secs < 1e-3:
        return f"{secs * 1e6:.1f} us"
    if secs < 1:
        return f"{secs * 1e3:.1f} ms"
    return f"{secs:.3f} s"


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDeltaStoragePathBenchmark(DatabricksIntegrationCase):
    """Round-trip a partitioned Delta table and compare read paths.

    One table seeded per class — the test methods run in a fresh
    workspace state but share the same data so each comparison
    isn't biased by per-test write overhead.
    """

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    table_name: ClassVar[str]
    table: ClassVar[Table]
    rows: ClassVar[int]
    seed: ClassVar[pa.Table]
    storage_path: ClassVar[Any] = None  # yggdrasil Path

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = _resolve_catalog()
        cls.schema_name = _resolve_schema()
        cls.table_name = f"yg_delta_path_{secrets.token_hex(4)}"
        full_name = f"{cls.catalog_name}.{cls.schema_name}.{cls.table_name}"
        cls.rows = _resolve_rows()
        cls.seed = _seed_table_arrow(cls.rows)

        try:
            cls.table = cls.client.tables.table(full_name)
            cls.table.ensure_created(_partitioned_schema())
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Cannot create table {full_name}: {exc}. Override "
                f"DATABRICKS_INTEGRATION_CATALOG / "
                f"DATABRICKS_INTEGRATION_SCHEMA with a writable target."
            ) from exc

        # One-shot insert — partitioned by region, so the warehouse
        # lands one parquet per region. The Arrow stage handles the
        # cast to the table schema.
        try:
            cls.table.arrow_insert(cls.seed, mode=Mode.APPEND)
        except (DatabricksError, PermissionDenied) as exc:
            cls._safe_drop_table()
            raise unittest.SkipTest(
                f"Cannot insert seed rows into {full_name}: {exc}."
            ) from exc

        # Resolve the table's underlying cloud storage path. Skip the
        # whole benchmark when temporary-table-credentials aren't
        # available (Azure / GCP on workspaces that don't implement
        # the AWS-style flow, or identities without the grant).
        try:
            cls.storage_path = cls.table.storage_path()
        except NotImplementedError as exc:
            cls._safe_drop_table()
            raise unittest.SkipTest(
                f"storage_location not implemented on this workspace: {exc}."
            )
        except (DatabricksError, PermissionDenied) as exc:
            cls._safe_drop_table()
            raise unittest.SkipTest(
                f"Cannot resolve storage_location (likely non-AWS): {exc}."
            )
        except Exception as exc:
            cls._safe_drop_table()
            raise unittest.SkipTest(
                f"storage_location unavailable: {exc}."
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._safe_drop_table()
        super().tearDownClass()

    @classmethod
    def _safe_drop_table(cls) -> None:
        table = getattr(cls, "table", None)
        if table is None:
            return
        try:
            table.delete(missing_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers — both readers, normalised to a sorted pylist for diffing.
    # ------------------------------------------------------------------

    def _read_via_sql(self, where: str | None = None) -> pa.Table:
        """Warehouse SQL ``SELECT *`` (with optional ``WHERE``)."""
        full = self.table.full_name(safe=True)
        sql = f"SELECT id, region, val FROM {full}"
        if where:
            sql += f" WHERE {where}"
        return self.client.sql.execute(sql).to_arrow_table()

    def _read_via_storage(
        self,
        *,
        predicate: Any = None,
        prune_values: Any = None,
    ) -> pa.Table:
        """Direct :class:`DeltaFolder` read over the table's storage URI."""
        delta_io = DeltaFolder(path=str(self.storage_path.full_path()))
        opts = DeltaOptions(
            predicate=predicate, prune_values=prune_values,
        )
        return delta_io.read_arrow_table(options=opts)

    def _normalised(self, table: pa.Table) -> list[dict]:
        """Sort rows by ``id`` so the two readers' outputs compare."""
        if table.num_rows == 0:
            return []
        idx = pa.compute.sort_indices(table, sort_keys=[("id", "ascending")])
        return table.take(idx).to_pylist()

    # ------------------------------------------------------------------
    # Tests — each asserts correctness *and* logs timings.
    # ------------------------------------------------------------------

    def test_storage_path_exposes_delta_log(self) -> None:
        """Every Delta table has a ``_delta_log`` directory.

        Pins the contract that ``Table.storage_path()`` returns a
        directory containing the transaction log alongside the
        parquet data files. The test reaches for ``storage_path()``
        (not ``storage_location()``) so the canonical name stays the
        one tests + docs surface.
        """
        root = self.table.storage_path()
        children = [c.name for c in root.iterdir()]
        self.assertIn("_delta_log", children)

    def test_full_scan_round_trip(self) -> None:
        """Full table scan: SQL vs direct storage read return same rows."""
        sql_table, sql_secs = _time_block(lambda: self._read_via_sql())
        store_table, store_secs = _time_block(lambda: self._read_via_storage())

        # Sanity — both paths returned all seeded rows.
        self.assertEqual(sql_table.num_rows, self.rows)
        self.assertEqual(store_table.num_rows, self.rows)
        # Content matches (sorted by id since file order isn't stable).
        self.assertEqual(
            self._normalised(sql_table), self._normalised(store_table),
        )

        self._report(
            "full-scan",
            sql_secs=sql_secs, sql_rows=sql_table.num_rows,
            store_secs=store_secs, store_rows=store_table.num_rows,
        )

    def test_partition_filter_round_trip(self) -> None:
        """``region == 'us'``: warehouse + storage produce the same subset.

        On the storage side, the predicate routes through
        :func:`extract_partition_filters` — the parquet files for
        non-``us`` partitions never open.
        """
        sql_table, sql_secs = _time_block(
            lambda: self._read_via_sql(where="region = 'us'")
        )
        store_table, store_secs = _time_block(
            lambda: self._read_via_storage(
                predicate=(expr_col("region") == "us"),
            )
        )

        expected = sum(
            1 for i in range(self.rows)
            if REGIONS[i % len(REGIONS)] == "us"
        )

        self.assertEqual(sql_table.num_rows, expected)
        self.assertEqual(store_table.num_rows, expected)
        self.assertEqual(
            self._normalised(sql_table), self._normalised(store_table),
        )
        self.assertTrue(
            all(r == "us" for r in store_table.column("region").to_pylist())
        )

        self._report(
            "partition-eq",
            sql_secs=sql_secs, sql_rows=sql_table.num_rows,
            store_secs=store_secs, store_rows=store_table.num_rows,
        )

    def test_or_chain_round_trip(self) -> None:
        """``region == us | == eu | == uk``: OR-of-EQ on the same column.

        ``extract_partition_filters`` walks the OR and unions the per-
        operand value sets, so partition pruning still resolves the
        accepted values without an explicit ``is_in`` rewrite.
        Warehouse runs the equivalent ``IN`` clause.
        """
        target = ("us", "eu", "uk")
        sql_where = "region IN ('us', 'eu', 'uk')"
        predicate = (
            (expr_col("region") == "us")
            | (expr_col("region") == "eu")
            | (expr_col("region") == "uk")
        )

        sql_table, sql_secs = _time_block(
            lambda: self._read_via_sql(where=sql_where)
        )
        store_table, store_secs = _time_block(
            lambda: self._read_via_storage(predicate=predicate)
        )

        expected_set = set(target)
        self.assertEqual(sql_table.num_rows, store_table.num_rows)
        self.assertEqual(
            self._normalised(sql_table), self._normalised(store_table),
        )
        self.assertTrue(
            set(store_table.column("region").to_pylist()).issubset(expected_set)
        )

        self._report(
            "or-collapse-3-values",
            sql_secs=sql_secs, sql_rows=sql_table.num_rows,
            store_secs=store_secs, store_rows=store_table.num_rows,
        )

    def test_partition_and_row_filter_round_trip(self) -> None:
        """``region == 'us' AND id > N/2``: file prune + row filter together.

        The partition extractor pins ``region = us`` (file prune);
        the row-level predicate ``id > N/2`` runs as a
        ``pyarrow.compute`` filter on each batch. Warehouse runs the
        full ``WHERE`` server-side.
        """
        half = self.rows // 2
        sql_table, sql_secs = _time_block(
            lambda: self._read_via_sql(
                where=f"region = 'us' AND id > {half}"
            )
        )
        store_table, store_secs = _time_block(
            lambda: self._read_via_storage(
                predicate=(
                    (expr_col("region") == "us") & (expr_col("id") > half)
                ),
            )
        )

        self.assertEqual(sql_table.num_rows, store_table.num_rows)
        self.assertEqual(
            self._normalised(sql_table), self._normalised(store_table),
        )

        self._report(
            "partition+row",
            sql_secs=sql_secs, sql_rows=sql_table.num_rows,
            store_secs=store_secs, store_rows=store_table.num_rows,
        )

    # ------------------------------------------------------------------
    # Reporting — printed to capsys so ``pytest -s`` surfaces a table.
    # ------------------------------------------------------------------

    def _report(
        self,
        label: str,
        *,
        sql_secs: float,
        sql_rows: int,
        store_secs: float,
        store_rows: int,
    ) -> None:
        speedup = (sql_secs / store_secs) if store_secs > 0 else float("inf")
        # ``print`` so ``pytest -s`` shows the bench line. The test
        # itself only asserts correctness — timings are informational.
        print(
            f"\n[delta-storage-bench] {label:<24s}  "
            f"sql={_fmt_seconds(sql_secs)} ({sql_rows} rows)  "
            f"storage={_fmt_seconds(store_secs)} ({store_rows} rows)  "
            f"speedup={speedup:.2f}x"
        )
