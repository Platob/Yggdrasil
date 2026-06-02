"""Shared fixture for the live SQL-engine integration suites.

Provisions the shared ``ygg_integration`` schema (via
:meth:`DatabricksIntegrationCase.integration_schema`) and a
:class:`SQLEngine` scoped to it, registers minted tables for
class-level cleanup, and exposes small sample-data builders. Not
collected by pytest (no ``Test`` prefix); the per-suite files subclass it.
"""
from __future__ import annotations

import logging
import secrets
from typing import ClassVar

import pyarrow as pa
from databricks.sdk.errors import DatabricksError

from yggdrasil.databricks.sql.engine import SQLEngine
from yggdrasil.databricks.table.table import Table

from .. import DatabricksIntegrationCase


LOGGER = logging.getLogger(__name__)

__all__ = ["SQLIntegrationCase"]


class SQLIntegrationCase(DatabricksIntegrationCase):
    """Shared fixture + helpers for the SQL integration suites.

    Not collected by pytest (no ``Test`` prefix). Provisions
    ``trading.unittest`` once per concrete subclass via
    ``ensure_created``, registers minted tables for class-level
    cleanup, and exposes small builders for sample schemas / data.
    """

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    engine: ClassVar[SQLEngine]
    created_tables: ClassVar[list[str]]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = cls.INTEGRATION_CATALOG
        cls.schema_name = cls.INTEGRATION_SCHEMA

        # Engine scoped to ``trading.unittest`` so every method that takes
        # ``catalog_name`` / ``schema_name`` keyword args inherits the right
        # default and unqualified table names resolve correctly.
        cls.engine = cls.client.sql(
            catalog_name=cls.catalog_name,
            schema_name=cls.schema_name,
        )
        cls.created_tables = []

        # Best-effort ensure_created on the catalog + schema. The engine /
        # table methods will retry the same path on per-test demand if a
        # transient miss surfaces later.
        cls._ensure_catalog_schema()

    @classmethod
    def _ensure_catalog_schema(cls) -> None:
        """Make sure ``trading.unittest`` exists; skip the suite if we
        can't get there (no permission to create the catalog, etc.)."""
        catalog = cls.engine.catalogs.catalog(cls.catalog_name)
        try:
            catalog.ensure_created(
                comment="yggdrasil integration-test catalog",
            )
        except DatabricksError as exc:
            # Most realistic miss: no metastore-admin grant. Skip the
            # suite cleanly so the rest of the local run isn't blocked.
            import unittest as _ut
            raise _ut.SkipTest(
                f"Cannot create or access catalog {cls.catalog_name!r}: "
                f"{exc}. Set DATABRICKS_INTEGRATION_CATALOG to a catalog "
                "the test identity can write to, or pre-provision "
                f"{cls.catalog_name}.{cls.schema_name}."
            ) from exc

        schema = cls.engine.schemas.schema(
            f"{cls.catalog_name}.{cls.schema_name}"
        )
        schema.ensure_created(
            comment="yggdrasil integration-test schema",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            for full_name in cls.created_tables:
                try:
                    cls.engine.table(full_name).delete(missing_ok=True)
                except DatabricksError:
                    pass
        finally:
            super().tearDownClass()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unique_table(self, prefix: str, seed: bool = True) -> Table:
        """Return a fresh :class:`Table` handle with a unique name.

        The handle is registered for class-level cleanup before any
        DDL runs, so a test that fails mid-flight still leaves the
        teardown loop with something to drop.
        """
        name = f"yg_{prefix}_{secrets.token_hex(4)}" if seed else f"yg_{prefix}"
        full_name = f"{self.catalog_name}.{self.schema_name}.{name}"
        type(self).created_tables.append(full_name)
        return self.engine.table(full_name)

    @staticmethod
    def _sample_schema() -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("label", pa.string()),
                pa.field("amount", pa.float64()),
            ]
        )

    @staticmethod
    def _sample_data() -> pa.Table:
        return pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "label": pa.array(["a", "b", "c"], type=pa.string()),
                "amount": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
            }
        )

    def _ensure_table(self, table: Table, definition: pa.Schema) -> Table:
        """Create the managed table if a probe says it's missing.

        Mirrors the "try operation; on miss, auto-create" contract the
        caller asked for: we read ``.exists`` and only call
        ``ensure_created`` when the answer is no.
        """
        if not table.exists():
            table.ensure_created(definition)
        return table


