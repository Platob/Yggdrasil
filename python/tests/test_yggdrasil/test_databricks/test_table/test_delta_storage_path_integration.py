"""Cross-engine Delta interop: yggdrasil :class:`DeltaFolder` ⇄ Databricks SQL.

A Databricks Delta table is, on disk, a folder of parquet files plus a
``_delta_log`` transaction log. This suite drives **both** engines against
the *same* table to prove the native yggdrasil Delta protocol is
byte-compatible with Databricks:

* **SQL writes → DeltaFolder reads.** Seed rows via the SQL/arrow insert,
  then read them back by pointing :class:`DeltaFolder` at the table's
  cloud :attr:`storage_location` (resolved through Unity Catalog temporary
  table credentials) — full scan + partition-pruned + row-filtered.
* **DeltaFolder writes → SQL reads.** Append rows by committing to the
  ``_delta_log`` directly through :class:`DeltaFolder`, then ``REFRESH`` +
  ``SELECT`` over the warehouse and confirm Databricks sees them.

Provisioning + skip rules
-------------------------

Gated by :class:`DatabricksIntegrationCase` (needs ``DATABRICKS_HOST``).
The table is created in an owned, throw-away ``ygg_integration_<hex>``
schema (via :meth:`scratch_schema`) under ``trading_tgp_dev`` so the
EXTERNAL USE SCHEMA grant that temporary-credential vending needs can be
self-granted; without it (non-AWS workspace, no grant) the suite skips.
The schema is dropped through the :meth:`safe_drop_schema` guard.
"""
from __future__ import annotations

import secrets
import unittest
from typing import Any, ClassVar

import pyarrow as pa
import pytest
from databricks.sdk.errors import DatabricksError, NotFound, PermissionDenied

from yggdrasil.data import Field
from yggdrasil.enums import Mode
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import Int64Type, StringType
from yggdrasil.io.delta import DeltaFolder, DeltaOptions
from yggdrasil.execution.expr import col as expr_col

from .. import DatabricksIntegrationCase


__all__ = ["TestDeltaSqlInterop"]


REGIONS = ("us", "eu", "uk", "ap")
SEED_ROWS = 2000


def _seed(rows: int, *, start: int = 0) -> pa.Table:
    return pa.table({
        "id": pa.array(range(start, start + rows), type=pa.int64()),
        "region": pa.array(
            [REGIONS[i % len(REGIONS)] for i in range(start, start + rows)],
            type=pa.string(),
        ),
        "val": pa.array([f"row-{i}" for i in range(start, start + rows)], type=pa.string()),
    })


def _partitioned_schema() -> Schema:
    s = Schema()
    s.with_field(Field(name="id", dtype=Int64Type(), nullable=False))
    s.with_field(Field(name="region", dtype=StringType()).with_partition_by(True))
    s.with_field(Field(name="val", dtype=StringType()))
    return s


@pytest.mark.integration
class TestDeltaSqlInterop(DatabricksIntegrationCase):
    """Round-trip a partitioned Delta table through both engines."""

    schema: ClassVar
    table: ClassVar
    full: ClassVar[str]
    storage: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.schema = cls.scratch_schema()
        name = f"yg_delta_{secrets.token_hex(4)}"
        cls.full = f"{cls.INTEGRATION_CATALOG}.{cls.schema.schema_name}.{name}"
        # An EXTERNAL Delta table at a writable location under the workspace's
        # registered external location. EXTERNAL (not managed) is required for
        # the DeltaFolder-write → SQL-read direction: UC only vends READ_WRITE
        # temporary credentials for external tables (managed tables vend READ,
        # so a direct ``_delta_log`` commit would 403).
        cls.location = cls.client.default_storage_location(
            f"ygg_integration/delta/{name}_{secrets.token_hex(4)}",
            catalog_name=cls.INTEGRATION_CATALOG,
        )
        try:
            cls.table = cls.client.tables.table(cls.full)
            # Create the EXTERNAL Delta table via SQL DDL with an explicit
            # LOCATION — Databricks initialises the ``_delta_log`` there and
            # registers it as external (so READ_WRITE temp creds are vended).
            cls.client.sql.execute(
                f"CREATE TABLE {cls.table.full_name(safe=True)} "
                f"(`id` BIGINT NOT NULL, `region` STRING, `val` STRING) "
                f"USING DELTA PARTITIONED BY (`region`) "
                f"LOCATION '{cls.location}'"
            )
            # SQL/arrow write — partitioned by region (one parquet per region).
            cls.table.arrow_insert(_seed(SEED_ROWS), mode=Mode.APPEND)
            # Resolve the cloud storage path (temporary table credentials —
            # EXTERNAL USE SCHEMA self-granted on the owned scratch schema).
            cls.storage = cls.table.storage_path()
        except (DatabricksError, PermissionDenied, NotFound, NotImplementedError) as exc:
            cls.safe_drop_schema(cls.schema)
            raise unittest.SkipTest(
                f"cannot provision / resolve storage for {cls.full}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 — any storage-resolve failure skips
            cls.safe_drop_schema(cls.schema)
            raise unittest.SkipTest(f"storage_location unavailable: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            # External-table data outlives DROP TABLE — purge the S3 location
            # while the table's temporary credentials are still vendable.
            storage = getattr(cls, "storage", None)
            if storage is not None:
                try:
                    storage.remove(recursive=True, missing_ok=True)
                except Exception:
                    pass
            table = getattr(cls, "table", None)
            if table is not None:
                try:
                    table.delete(missing_ok=True)
                except Exception:
                    pass
        finally:
            cls.safe_drop_schema(getattr(cls, "schema", None))
            super().tearDownClass()

    # -- helpers -------------------------------------------------------
    def _delta(self) -> DeltaFolder:
        # Built from the creds-carrying storage Path (Table.delta()), not a
        # re-parsed URI string — otherwise the S3 reads have no credentials.
        return self.table.delta()

    def _sql(self, where: "str | None" = None) -> pa.Table:
        sql = f"SELECT id, region, val FROM {self.table.full_name(safe=True)}"
        if where:
            sql += f" WHERE {where}"
        return self.client.sql.execute(sql).to_arrow_table()

    @staticmethod
    def _sorted(table: pa.Table) -> list:
        if table.num_rows == 0:
            return []
        idx = pa.compute.sort_indices(table, sort_keys=[("id", "ascending")])
        return table.take(idx).to_pylist()

    # -- SQL writes → DeltaFolder reads --------------------------------
    def test_storage_path_exposes_delta_log(self) -> None:
        children = [c.name for c in self.storage.iterdir()]
        self.assertIn("_delta_log", children)

    def test_sql_write_deltafolder_full_scan(self) -> None:
        # Both engines read the *same* table — assert they agree (rather than
        # a fixed count, since another test may have appended rows). At least
        # the seeded rows are present.
        sql = self._sql()
        store = self._delta().read_arrow_table()
        self.assertGreaterEqual(sql.num_rows, SEED_ROWS)
        self.assertEqual(sql.num_rows, store.num_rows)
        self.assertEqual(self._sorted(sql), self._sorted(store))

    def test_deltafolder_partition_pushdown(self) -> None:
        store = self._delta().read_arrow_table(
            options=DeltaOptions(predicate=(expr_col("region") == "us")),
        )
        sql = self._sql(where="region = 'us'")
        self.assertEqual(store.num_rows, sql.num_rows)
        self.assertTrue(all(r == "us" for r in store.column("region").to_pylist()))
        self.assertEqual(self._sorted(sql), self._sorted(store))

    def test_deltafolder_partition_and_row_filter(self) -> None:
        half = SEED_ROWS // 2
        store = self._delta().read_arrow_table(options=DeltaOptions(
            predicate=((expr_col("region") == "eu") & (expr_col("id") > half)),
        ))
        sql = self._sql(where=f"region = 'eu' AND id > {half}")
        self.assertEqual(self._sorted(sql), self._sorted(store))

    # -- DeltaFolder writes → SQL reads (the interop direction) --------
    def test_deltafolder_write_sql_read(self) -> None:
        """Append rows by committing to the ``_delta_log`` through yggdrasil,
        then confirm the warehouse reads them — proves yggdrasil's Delta
        writes are Databricks-readable."""
        extra = _seed(400, start=SEED_ROWS)  # ids 2000..2399, partitioned
        self._delta().refresh().write_arrow_table(extra, mode=Mode.APPEND)

        # Invalidate Databricks' cached snapshot, then read just the new ids.
        self.client.sql.execute(f"REFRESH TABLE {self.table.full_name(safe=True)}")
        got = self._sql(where=f"id >= {SEED_ROWS}")
        self.assertEqual(got.num_rows, 400)
        self.assertEqual(self._sorted(got), extra.to_pylist())
        # Total now reflects both writers.
        self.assertEqual(self._sql().num_rows, SEED_ROWS + 400)
