"""Smoke test: clone a real 3mv table to an EXTERNAL POC table on the same
storage root and write into it through :class:`DeltaFolder`.

Seeded from the live ``trading.ba_3mv_polaris__d__volcano_output.css_prices_new``
table (MANAGED Delta, nested ``array<struct>`` columns, partitioned). The
suite:

* resolves the **3mv path** — the source schema's external storage root, the
  segment before Databricks' managed ``__unitystorage`` — and creates
  ``css_prices_poc`` as an EXTERNAL Delta table rooted there with the source's
  column schema, then
* writes a small sample (lifted from the source) straight to the POC table's
  ``_delta_log`` via :class:`DeltaFolder` and confirms both the warehouse and
  a native Delta read see the rows.

Provisioning + skip rules
-------------------------

Gated by :class:`DatabricksIntegrationCase` (needs ``DATABRICKS_HOST``). The
source table, target table and a sample-row limit are overridable via
``CSS_PRICES_SOURCE`` / ``CSS_PRICES_POC_TABLE`` / ``CSS_PRICES_POC_ROWS``.
The whole class skips when the source table is unreadable or the identity
can't create / write an external table on the 3mv root (no CREATE TABLE or
EXTERNAL location grant) — so it stays green on a workspace without 3mv access.
"""
from __future__ import annotations

import os
import unittest
import uuid
from typing import ClassVar

import pyarrow as pa
import pytest
from databricks.sdk.errors import DatabricksError, NotFound, PermissionDenied

from yggdrasil.data.schema import Schema
from yggdrasil.enums import Mode

from .. import DatabricksIntegrationCase


__all__ = ["TestCssPricesPocSmoke"]


def _canonical_utc(table: pa.Table) -> pa.Table:
    """Canonicalise ``Etc/UTC`` timestamp zones to ``UTC``.

    Databricks hands tz-aware timestamps back as ``Etc/UTC``; the Delta
    parquet file is created as ``UTC``, so the written batch must agree or the
    parquet writer rejects the (semantically identical) schema.
    """
    fields = []
    changed = False
    for f in table.schema:
        ty = f.type
        if pa.types.is_timestamp(ty) and ty.tz in ("Etc/UTC", "Etc/Universal"):
            ty = pa.timestamp(ty.unit, "UTC")
            changed = True
        fields.append(pa.field(f.name, ty, f.nullable))
    return table.cast(pa.schema(fields)) if changed else table


@pytest.mark.integration
class TestCssPricesPocSmoke(DatabricksIntegrationCase):
    """Create an EXTERNAL POC table on the 3mv path + DeltaFolder write."""

    SOURCE: ClassVar[str] = os.environ.get(
        "CSS_PRICES_SOURCE",
        "trading.ba_3mv_polaris__d__volcano_output.css_prices_new",
    )
    POC: ClassVar[str] = os.environ.get(
        "CSS_PRICES_POC_TABLE",
        "trading.ba_3mv_polaris__d__volcano_output.css_prices_poc",
    )
    ROWS: ClassVar[int] = int(os.environ.get("CSS_PRICES_POC_ROWS", "8"))

    source: ClassVar
    poc: ClassVar
    root: ClassVar[str]
    location: ClassVar[str]
    sample: ClassVar[pa.Table]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.source = cls.client.tables.table(cls.SOURCE)
        try:
            if cls.source.read_infos(default=None) is None:
                raise unittest.SkipTest(f"source table {cls.SOURCE} not found")
            # The 3mv path: the schema's external storage root, before the
            # managed ``__unitystorage`` segment. The POC table is rooted on it.
            cls.root = cls.source.schema_storage_location().split(
                "/__unitystorage",
            )[0].rstrip("/")
            # Unique per run: the Delta byte cache (yggdrasil.io.delta._cache)
            # is keyed by S3 path on the documented invariant that a
            # version-addressed ``_delta_log/<n>.json`` is immutable. Reusing a
            # fixed location across runs (drop + recreate writes *new* content to
            # the same ``00…00.json`` path) violates that, so the persistent
            # disk cache would serve a prior run's stale commit bytes — and the
            # DeltaFolder read would see 0 rows. A fresh location per run keeps
            # the invariant intact and the test hermetic.
            cls.location = f"{cls.root}/poc/css_prices_poc-{uuid.uuid4().hex[:12]}"

            cls.sample = _canonical_utc(
                cls.client.sql.execute(
                    f"SELECT * FROM {cls.SOURCE} LIMIT {cls.ROWS}",
                ).to_arrow_table()
            )
            if cls.sample.num_rows == 0:
                raise unittest.SkipTest(f"source table {cls.SOURCE} has no rows")

            cls.poc = cls.client.tables.table(cls.POC)
            cls.poc.delete(missing_ok=True)  # idempotent re-runs
            cls.poc.create(
                Schema.from_arrow(cls.sample.schema),
                storage_location=cls.location,
            )
        except unittest.SkipTest:
            raise
        except (DatabricksError, PermissionDenied, NotFound, NotImplementedError) as exc:
            cls._purge()
            raise unittest.SkipTest(
                f"cannot provision EXTERNAL {cls.POC} on the 3mv path: {exc}"
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls._purge()
        finally:
            super().tearDownClass()

    @classmethod
    def _purge(cls) -> None:
        poc = getattr(cls, "poc", None)
        if poc is None:
            return
        # External-table data outlives DROP TABLE — purge the S3 location while
        # the temporary credentials are still vendable, then drop the table.
        try:
            sp = poc.storage_path()
            if sp is not None:
                sp.remove(recursive=True, missing_ok=True)
        except Exception:
            pass
        try:
            poc.delete(missing_ok=True)
        except Exception:
            pass

    # -- tests ---------------------------------------------------------
    def test_poc_is_external_on_3mv_path(self) -> None:
        infos = self.poc.read_infos()
        self.assertEqual(infos.table_type.value, "EXTERNAL")
        self.assertTrue(
            infos.storage_location.rstrip("/").startswith(self.root),
            f"{infos.storage_location!r} is not under the 3mv root {self.root!r}",
        )

    def test_poc_schema_matches_source(self) -> None:
        src_cols = {c.name for c in self.source.columns}
        poc_cols = {c.name for c in self.poc.columns}
        self.assertEqual(poc_cols, src_cols)

    def test_deltafolder_write_then_read(self) -> None:
        # Commit the sample straight to the POC table's ``_delta_log`` through
        # DeltaFolder, then confirm both engines read it back.
        self.poc.delta(write=True).refresh().write_arrow_table(
            self.sample, mode=Mode.APPEND,
        )

        self.client.sql.execute(f"REFRESH TABLE {self.POC}")
        n = (
            self.client.sql.execute(f"SELECT COUNT(*) AS n FROM {self.POC}")
            .to_arrow_table()
            .column("n")
            .to_pylist()[0]
        )
        self.assertEqual(int(n), self.sample.num_rows)
        self.assertEqual(
            self.poc.delta(write=False).read_arrow_table().num_rows,
            self.sample.num_rows,
        )
