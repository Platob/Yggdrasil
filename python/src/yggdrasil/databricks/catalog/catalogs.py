"""
Collection-level service for Unity Catalog catalogs and schemas.

Provides ``__getitem__``, ``catalog()``, ``schema()``, ``table()``, and
``list()`` against the Databricks catalog API.  Per-resource DDL/DML lives in
:class:`~yggdrasil.databricks.catalog.catalog.Catalog` and
:class:`~yggdrasil.databricks.schema.schema.Schema`.

Hierarchy
---------
::

    client.catalogs["main"]                            # Catalog
    client.catalogs["main"]["sales"]                   # Schema
    client.catalogs["main.sales"]                      # Schema  (dot-separated shorthand)
    client.catalogs["main"]["sales"]["orders"]         # Table
    client.catalogs["main.sales.orders"]               # Table
    client.catalogs["main"]["sales"]["orders"]["price"]  # Column
    client.catalogs["main.sales.orders.price"]         # Column
    client.catalogs.list_catalogs()                    # Iterator[Catalog]

Caching strategy
----------------
A module-level :class:`ExpiringDict` (``_CATALOG_INFO_CACHE``) keyed by
``"host|catalog"`` acts as a local cache (TTL = 5 min by default).
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional, Union

from databricks.sdk.service.catalog import CatalogInfo

from yggdrasil.databricks.client import DatabricksService
from yggdrasil.dataclasses.expiring import ExpiringDict

from yggdrasil.databricks.catalog.catalog import UCCatalog
from yggdrasil.databricks.column.column import Column
from yggdrasil.databricks.schema.schema import UCSchema
from yggdrasil.databricks.sql.sql_utils import name_matcher
from yggdrasil.databricks.table.table import Table

__all__ = ["Catalogs"]

logger = logging.getLogger(__name__)

# Module-level cache keyed by "host|catalog_name"; default TTL = 5 minutes.
_CATALOG_INFO_CACHE: ExpiringDict[str, CatalogInfo] = ExpiringDict(default_ttl=300.0)


class Catalogs(DatabricksService):
    """Collection-level service for Unity Catalog catalogs and schemas.

    Provides a dict-like interface for the full UC hierarchy::

        catalogs = client.catalogs

        # navigate by subscript — each step returns the next level
        catalog = catalogs["main"]                            # Catalog
        schema  = catalogs["main"]["sales"]                   # Schema
        table   = catalogs["main"]["sales"]["orders"]         # Table
        column  = catalogs["main"]["sales"]["orders"]["price"]  # Column

        # shorthand dot-notation string
        schema  = catalogs["main.sales"]                      # Schema
        table   = catalogs["main.sales.orders"]               # Table
        column  = catalogs["main.sales.orders.price"]         # Column

        # explicit factory methods
        catalog = catalogs.catalog("main")                    # Catalog
        schema  = catalogs.schema("main.sales")               # Schema
        table   = catalogs.table("main.sales.orders")         # Table

        # list
        for cat in catalogs.list_catalogs():
            for sch in cat.schemas():
                for tbl in sch.tables():
                    ...
    """

    # ── dict-like navigation ──────────────────────────────────────────────────

    def __getitem__(self, name: str) -> Union[UCCatalog, UCSchema, Table, Column]:
        """Route a 1-, 2-, 3-, or 4-part dotted name to the right resource.

        * ``catalogs["main"]``                     → :class:`Catalog`
        * ``catalogs["main.sales"]``               → :class:`Schema`
        * ``catalogs["main.sales.orders"]``        → :class:`Table`
        * ``catalogs["main.sales.orders.price"]``  → :class:`Column`
        """
        parts = [p.strip().strip("`") for p in name.split(".")]
        if len(parts) == 1:
            return self.catalog(parts[0])
        if len(parts) == 2:
            return self.schema(f"{parts[0]}.{parts[1]}")
        if len(parts) == 3:
            return self.table(".".join(parts))
        if len(parts) == 4:
            return self.client.columns.column(".".join(parts))
        raise KeyError(
            f"Expected a 1- to 4-part dotted name (catalog[.schema[.table[.column]]]),"
            f" got {name!r} with {len(parts)} parts"
        )

    def __setitem__(self, name: str, new_name: str) -> None:
        """``catalogs[key] = "new"`` renames the resource identified by *key*.

        *key* follows the same dotted routing as :meth:`__getitem__`; *new_name*
        is the unqualified new name for the leaf (catalog, schema, table, or column).
        """
        self[name].rename(new_name)

    def __iter__(self) -> Iterator[UCCatalog]:
        """Iterate over all visible catalogs (``self.list_catalogs()``)."""
        return self.list_catalogs()

    # ── factory methods ───────────────────────────────────────────────────────

    def catalog(self, name: str) -> UCCatalog:
        """Return a :class:`Catalog` bound to this service.

        Uses the module-level cache when available.

        Args:
            name: Catalog name.
        """
        cat = UCCatalog(service=self, catalog_name=name)
        key = self._cache_key(name)
        cached = _CATALOG_INFO_CACHE.get(key)
        if cached is not None:
            object.__setattr__(cat, "_infos", cached)
            object.__setattr__(cat, "_infos_fetched_at", time.time())
        return cat

    def schema(self, full_name: str) -> UCSchema:
        """Return a :class:`Schema` from a ``"catalog.schema"`` string.

        Args:
            full_name: Two-part dotted name (backtick-quoting is stripped).
        """
        parts = [p.strip().strip("`") for p in full_name.split(".")]
        if len(parts) != 2:
            raise ValueError(
                f"Expected a 'catalog.schema' two-part name, got {full_name!r}"
            )
        return UCSchema(service=self, catalog_name=parts[0], schema_name=parts[1])

    def table(self, location: str) -> Table:
        """Return a :class:`Table` from a three-part fully-qualified name.

        Args:
            location: Three-part dotted name ``"catalog.schema.table"``.
        """
        return self.client.tables.table(location=location)

    # ── listing ───────────────────────────────────────────────────────────────

    def list_catalogs(
        self,
        name: str | None = None,
        *,
        use_cache: bool = True,
    ) -> Iterator[UCCatalog]:
        """Iterate over visible catalogs, populating the local cache.

        Args:
            name:      Optional catalog-name filter.  When it contains ``*``,
                       matching uses a case-insensitive glob (e.g. ``"prod_*"``,
                       ``"*_raw"``, ``"*"``).  Without ``*`` it's an exact match.
            use_cache: Populate ``_CATALOG_INFO_CACHE`` from results.
        """
        logger.debug(
            "Listing catalogs (name_filter=%s, use_cache=%s)", name, use_cache,
        )
        matcher = name_matcher(name)
        yielded = 0
        for info in self.client.workspace_client().catalogs.list():
            if matcher is not None and not matcher(info.name):
                continue

            cat = UCCatalog(service=self, catalog_name=info.name)
            object.__setattr__(cat, "_infos", info)

            if use_cache:
                _CATALOG_INFO_CACHE[self._cache_key(info.name)] = info

            yielded += 1
            yield cat
        logger.debug(
            "Listed %d catalogs (name_filter=%s)", yielded, name,
        )

    # ── parse helper ──────────────────────────────────────────────────────────

    def parse_location(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse a 1-, 2-, or 3-part dotted name into ``(catalog, schema, table)``.

        Keyword overrides take precedence over parts extracted from *location*.
        """
        if location:
            parts = [p.strip().strip("`") for p in location.split(".")]
            if len(parts) >= 3:
                catalog_name = catalog_name or parts[-3]
                schema_name = schema_name or parts[-2]
                table_name = table_name or parts[-1]
            elif len(parts) == 2:
                catalog_name = catalog_name or parts[0]
                schema_name = schema_name or parts[1]
            elif len(parts) == 1:
                catalog_name = catalog_name or parts[0]
        return catalog_name, schema_name, table_name

    # ── cache helpers ─────────────────────────────────────────────────────────

    def _cache_key(self, catalog_name: str) -> str:
        # Use the dataclass ``host`` field directly — ``base_url.to_string()``
        # would parse a fresh URL on every call (~6 us). The host string is
        # already normalized in ``DatabricksClient.__post_init__``.
        host = (self.client.host if self.client else None) or "default"
        return f"{host}|{catalog_name}"

    @classmethod
    def invalidate_all(cls) -> None:
        """Clear the entire module-level catalog-info cache."""
        _CATALOG_INFO_CACHE.clear()

