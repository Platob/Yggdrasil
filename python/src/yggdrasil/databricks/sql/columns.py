"""
Collection-level service for Unity Catalog columns.

Mirrors the :class:`~yggdrasil.databricks.sql.tables.Tables` pattern:
context defaults (``catalog_name``, ``schema_name``, ``table_name``,
``column_name``) are carried on the service instance so callers only have to
supply the parts that differ from the default.

Typical usage::

    # Fully qualified
    col = client.columns.column("trading.unittest.orders.price")

    # Scoped service — provide only what differs
    cols = Columns(
        client=client,
        catalog_name="trading",
        schema_name="unittest",
        table_name="orders",
    )
    col  = cols.column("price")
    all_ = cols.list_columns()

    # Via Table (same defaults, no extra lookup)
    col = table.columns_service.column("price")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional, TYPE_CHECKING

from yggdrasil.databricks.client import DatabricksService

if TYPE_CHECKING:
    from .column import Column
    from .table import Table

__all__ = ["Columns"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Columns(DatabricksService):
    """Collection-level service for Unity Catalog columns.

    Carry default catalog / schema / table / column context so callers do not
    have to repeat them on every call::

        cols = client.columns(catalog_name="main", schema_name="sales", table_name="orders")
        col  = cols.column("price")
        for c in cols.list_columns():
            print(c.name)
    """

    catalog_name: str | None = None
    schema_name: str | None = None
    table_name: str | None = None
    column_name: str | None = None

    # -------------------------------------------------------------------------
    # Dict-like navigation — uses defaults on the service
    # -------------------------------------------------------------------------

    def __getitem__(self, name: str) -> "Column":
        """Resolve a :class:`Column` from a 1- to 4-part dotted name.

        Service defaults fill any missing leading parts.

        * ``columns["price"]``                         → needs catalog + schema + table defaults
        * ``columns["orders.price"]``                  → needs catalog + schema defaults
        * ``columns["sales.orders.price"]``            → needs catalog default
        * ``columns["main.sales.orders.price"]``       → fully qualified
        """
        return self.column(name)

    def __setitem__(self, name: str, new_name: str) -> None:
        """``columns[key] = "new"`` renames the resolved column."""
        self.column(name).rename(new_name)

    def __iter__(self) -> "Iterator[Column]":
        """Iterate over the columns of the default table."""
        return iter(self.list_columns())

    # -------------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------------

    def parse_location(
        self, location: str
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse a dotted location into ``(catalog, schema, table, column)``.

        Supported forms (backtick-quoted parts are stripped):

        ==========================================  =============================================
        Input                                       Result
        ==========================================  =============================================
        ``"price"``                                 ``(self.catalog, self.schema, self.table, "price")``
        ``"orders.price"``                          ``(self.catalog, self.schema, "orders", "price")``
        ``"sales.orders.price"``                    ``(self.catalog, "sales", "orders", "price")``
        ``"main.sales.orders.price"``               ``("main", "sales", "orders", "price")``
        ==========================================  =============================================

        Excess leading parts are dropped — only the rightmost four are used.
        """
        parts = [p.strip().strip("`") for p in location.split(".")]
        n = len(parts)
        if n == 1:
            return self.catalog_name, self.schema_name, self.table_name, parts[0]
        if n == 2:
            return self.catalog_name, self.schema_name, parts[0], parts[1]
        if n == 3:
            return self.catalog_name, parts[0], parts[1], parts[2]
        # 4+
        return parts[-4], parts[-3], parts[-2], parts[-1]

    def _resolve_table_parts(
        self,
        location: Optional[str],
        catalog_name: Optional[str],
        schema_name: Optional[str],
        table_name: Optional[str],
        column_name: Optional[str],
    ) -> tuple[str, str, str, Optional[str]]:
        """Merge *location* with explicit overrides and service defaults.

        Returns ``(catalog, schema, table, column_name_or_none)`` with
        assertions that the three-part table reference is present.
        """
        if location is not None:
            c, s, t, cn = self.parse_location(location)
            catalog_name = catalog_name or c
            schema_name = schema_name or s
            table_name = table_name or t
            column_name = column_name or cn

        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name
        table_name = table_name or self.table_name

        assert catalog_name, "No catalog_name: supply it explicitly or set it on the Columns service"
        assert schema_name, "No schema_name: supply it explicitly or set it on the Columns service"
        assert table_name, "No table_name: supply it explicitly or set it on the Columns service"

        return catalog_name, schema_name, table_name, column_name

    # -------------------------------------------------------------------------
    # Table lookup helper (lazy import to avoid circular dependency)
    # -------------------------------------------------------------------------

    def _find_table(
        self,
        catalog_name: str,
        schema_name: str,
        table_name: str,
    ) -> "Table":
        return self.client.tables.find_table(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def column(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        column_name: str | None = None,
    ) -> "Column":
        """Resolve and return a single :class:`~yggdrasil.databricks.sql.column.Column`.

        Args:
            location:     Dotted name (``"catalog.schema.table.col"`` or fewer parts).
            catalog_name: Override catalog (falls back to service default).
            schema_name:  Override schema (falls back to service default).
            table_name:   Override table (falls back to service default).
            column_name:  Override column name (falls back to last *location* part).

        Returns:
            The resolved :class:`Column`.

        Raises:
            AssertionError: When any of catalog / schema / table is missing.
            ValueError:     When the column does not exist in the table.
        """
        cat, sch, tbl, col = self._resolve_table_parts(
            location, catalog_name, schema_name, table_name, column_name
        )

        col = col or self.column_name
        assert col, "No column_name: supply it explicitly or include it in location"

        logger.debug("Columns.column: resolving %s.%s.%s.%s", cat, sch, tbl, col)
        return self._find_table(cat, sch, tbl).column(col)

    def list_columns(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
    ) -> list["Column"]:
        """Return all columns for a table.

        Args:
            location:     Dotted name (table or schema.table or fully qualified).
            catalog_name: Override catalog.
            schema_name:  Override schema.
            table_name:   Override table.

        Returns:
            List of :class:`Column` objects in schema-definition order.
        """
        cat, sch, tbl, _ = self._resolve_table_parts(
            location, catalog_name, schema_name, table_name, None
        )
        logger.debug("Columns.list_columns: %s.%s.%s", cat, sch, tbl)
        return self._find_table(cat, sch, tbl).columns

