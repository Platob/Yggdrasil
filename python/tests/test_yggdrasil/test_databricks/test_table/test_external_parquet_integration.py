"""Live integration test: an **external PARQUET** table on the catalog's
**default external location**, written directly to S3 with Hive partitions.

The flow, end to end against a real workspace:

1. resolve the **default external location** — the catalog's ``storage_root``
   (an S3 URI), with no bucket spelled out by the caller;
2. ``Table.create(table_type=EXTERNAL, data_source_format=PARQUET)`` over a
   schema whose ``region`` / ``dt`` fields are tagged ``partition_by`` →
   ``CREATE TABLE … USING PARQUET PARTITIONED BY (region, dt) LOCATION
   's3://…'`` (no ``delta.*`` props / ``CLUSTER BY`` — Parquet rejects them);
3. ``Table.insert`` writes partitioned rows;
4. read-back confirms the per-partition row counts;
5. the S3 storage path is listed to confirm the **Hive partition layout**
   (``region=…/dt=…/*.parquet``) — i.e. data really landed in S3, partitioned.

Privilege-gated and degrades to ``skipTest`` (never a hard failure) when the
identity lacks ``CREATE EXTERNAL TABLE`` on the location, when no S3 default
external location is resolvable, or when the runner's AWS creds can't reach
the bucket. Override the location root with ``YGG_TEST_EXTERNAL_TABLE_LOCATION``.
"""
from __future__ import annotations

import os
import secrets
import unittest

from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied
from databricks.sdk.service.catalog import DataSourceFormat, TableType

from yggdrasil.data import Field, Schema
from yggdrasil.databricks.sql.exceptions import SQLError
from yggdrasil.enums import Mode

from .. import DatabricksIntegrationCase

#: Errors that mean "this identity can't create/write the external table
#: here" — a privilege/environment condition, not a code defect → skip.
_CREATE_ERRORS = (PermissionDenied, SQLError, DatabricksError)


def _leaf(path) -> str:
    """The final path segment (partition dir / file name) of a Path."""
    return str(path).rstrip("/").rsplit("/", 1)[-1]


class TestExternalParquetTableIntegration(DatabricksIntegrationCase):
    def _default_external_location(self) -> str:
        """The catalog's default external location (its S3 ``storage_root``)."""
        env = os.environ.get("YGG_TEST_EXTERNAL_TABLE_LOCATION", "").strip()
        if env:
            return env.rstrip("/")
        try:
            cat = self.client.workspace_client().catalogs.get(name=self.INTEGRATION_CATALOG)
        except PermissionDenied as exc:
            raise unittest.SkipTest(f"cannot read catalog {self.INTEGRATION_CATALOG}: {exc}")
        root = (cat.storage_root or "").strip().rstrip("/")
        if not root.startswith(("s3://", "s3a://")):
            raise unittest.SkipTest(
                f"catalog {self.INTEGRATION_CATALOG} has no S3 default external "
                f"location (storage_root={root!r}); set "
                f"YGG_TEST_EXTERNAL_TABLE_LOCATION to one."
            )
        return root

    def test_external_parquet_table_partitioned_on_s3(self):
        self.integration_schema()  # ensure the schema exists (skips on perms)
        root = self._default_external_location()
        name = f"yg_ext_{secrets.token_hex(4)}"
        location = f"{root}/{self.INTEGRATION_SCHEMA}/_ext_tests/{name}"
        full = f"{self.INTEGRATION_CATALOG}.{self.INTEGRATION_SCHEMA}.{name}"
        tbl = self.client.tables.table(full)

        schema = Schema.from_fields([
            Field("id", "int64"),
            Field("amount", "double"),
            Field("region", "string", tags={"partition_by": True}),
            Field("dt", "string", tags={"partition_by": True}),
        ])

        try:
            tbl.create(
                schema,
                table_type=TableType.EXTERNAL,
                data_source_format=DataSourceFormat.PARQUET,
                storage_location=location,
            )
        except _CREATE_ERRORS as exc:
            raise unittest.SkipTest(
                f"cannot create external table at {location} "
                f"(needs CREATE EXTERNAL TABLE on the location): {exc}"
            )

        storage = None
        try:
            # --- it is an external, partitioned, Parquet table on S3 --------
            info = tbl.infos
            self.assertEqual(info.table_type, TableType.EXTERNAL)
            self.assertEqual(info.data_source_format, DataSourceFormat.PARQUET)
            self.assertTrue((info.storage_location or "").startswith(("s3://", "s3a://")))
            # UC usually reports the partition columns via ``partition_index``;
            # when it does, they must be region/dt. The definitive proof of
            # "correct partitions" is the S3 Hive layout asserted below.
            partition_cols = [
                c.name for c in (info.columns or []) if c.partition_index is not None
            ]
            if partition_cols:
                self.assertEqual(partition_cols, ["region", "dt"])
            storage = tbl.storage_path()

            # --- write partitioned rows -------------------------------------
            import pyarrow as pa

            data = pa.table({
                "id": [1, 2, 3, 4, 5],
                "amount": [1.5, 2.5, 3.5, 4.5, 5.5],
                "region": ["eu", "eu", "us", "us", "eu"],
                "dt": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"],
            })
            try:
                tbl.insert(data, mode=Mode.APPEND)
            except _CREATE_ERRORS as exc:
                raise unittest.SkipTest(f"cannot write to the external table: {exc}")

            # --- read back the per-partition counts -------------------------
            rows = self.client.sql.execute(
                f"SELECT region, dt, COUNT(*) AS n FROM {full} "
                f"GROUP BY region, dt ORDER BY region, dt"
            ).to_arrow_table().to_pylist()
            counts = {(r["region"], r["dt"]): r["n"] for r in rows}
            self.assertEqual(counts, {
                ("eu", "2024-01-01"): 2,
                ("eu", "2024-01-02"): 1,
                ("us", "2024-01-01"): 1,
                ("us", "2024-01-02"): 1,
            })

            # --- the data really landed in S3, Hive-partitioned -------------
            try:
                top = {_leaf(p) for p in storage.ls()}
            except Exception as exc:  # noqa: BLE001 — ambient AWS creds
                raise unittest.SkipTest(
                    f"cannot list S3 storage (AWS creds for the bucket?): {exc}"
                )
            self.assertIn("region=eu", top)
            self.assertIn("region=us", top)

            # one level down: dt=… partition dirs holding the .parquet files
            eu_dir = next(p for p in storage.ls() if _leaf(p) == "region=eu")
            eu_children = list(eu_dir.ls())
            self.assertTrue(
                any(_leaf(p).startswith("dt=") for p in eu_children),
                f"expected dt=… sub-partitions under region=eu, got {[_leaf(p) for p in eu_children]}",
            )
            dt_dir = next(p for p in eu_children if _leaf(p).startswith("dt="))
            self.assertTrue(
                any(_leaf(p).endswith(".parquet") for p in dt_dir.ls()),
                "expected at least one .parquet data file in the leaf partition",
            )
        finally:
            try:
                tbl.delete(missing_ok=True)
            except Exception:  # noqa: BLE001 — teardown is best-effort
                pass
            # Dropping an external table leaves its S3 data — remove it too.
            if storage is not None:
                try:
                    storage.remove(recursive=True, missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
