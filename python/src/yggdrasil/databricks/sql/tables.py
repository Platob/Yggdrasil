"""
Collection-level service for Unity Catalog tables.

Provides ``find_table`` and ``list_tables`` against the Databricks catalog API.
Per-table DDL/DML lives in :class:`~yggdrasil.databricks.sql.table.Table`.

Caching strategy
----------------
A module-level :class:`ExpiringDict` (``_TABLE_INFO_CACHE``) keyed by
``"host|catalog.schema.table"`` acts as a fast *local* cache so the same
table lookup never hits the API twice within the TTL window.

    1. **Local** — check ``_TABLE_INFO_CACHE``; return immediately on hit.
    2. **Remote** — call ``_find_table_remote``; only reached on miss.
    3. **Update** — store the remote result back into the local cache.

Cache entries can be invalidated per-table via :meth:`Tables._invalidate`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Iterator

from databricks.sdk.errors import DatabricksError, ResourceDoesNotExist
from databricks.sdk.service.catalog import TableInfo
from yggdrasil.databricks.client import DatabricksService
from yggdrasil.databricks.sql.types import quote_ident
from yggdrasil.dataclasses.expiring import ExpiringDict

from .table import Table

__all__ = ["Tables"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level function cache
# Keyed by "host|catalog.schema.table"; default TTL = 5 minutes.
# ---------------------------------------------------------------------------
_TABLE_INFO_CACHE: ExpiringDict[str, TableInfo] = ExpiringDict(default_ttl=300.0)


@dataclass(frozen=True)
class Tables(DatabricksService):
    """Collection-level service for Unity Catalog tables.

    Attach default catalog / schema context so callers don't have to repeat
    them on every call::

        tables = client.tables(catalog_name="main", schema_name="sales")
        table  = tables.find_table("orders")
        for t in tables.list_tables():
            ...
    """

    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None
    table_name: Optional[str] = None

    def parse_catalog_schema_table_names(self, full_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        parts = [_.strip("`") for _ in full_name.split(".")]

        # Unreachable for .split("."), but harmless and keeps intent explicit.
        if len(parts) == 0:
            return self.catalog_name, self.schema_name, None
        if len(parts) == 1:
            return self.catalog_name, self.schema_name, parts[0]
        if len(parts) == 2:
            return self.catalog_name, parts[0], parts[1]

        catalog_name, schema_name, table_name = parts[-3], parts[-2], parts[-1]
        return catalog_name or self.catalog_name, schema_name or self.schema_name, table_name

    def parse_check_location_params(
        self,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        safe_chars: bool = True,
    ) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
        if location:
            c, s, t = self.parse_catalog_schema_table_names(location)
            catalog_name, schema_name, table_name = catalog_name or c, schema_name or s, table_name or t

        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        assert catalog_name, "No catalog name given"
        assert schema_name, "No schema name given"
        assert table_name, "No table name given"

        if safe_chars:
            location = f"{quote_ident(catalog_name)}.{quote_ident(schema_name)}.{quote_ident(table_name)}"
        else:
            location = f"{catalog_name}.{schema_name}.{table_name}"

        return location, catalog_name or self.catalog_name, schema_name or self.schema_name, table_name

    # -------------------------------------------------------------------------
    # Factory — create a Table resource bound to this service
    # -------------------------------------------------------------------------

    def table(
        self,
        location: Optional[str] = None,
        *,
        table_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> "Table":
        """Return a :class:`~yggdrasil.databricks.sql.table.Table` bound to this service."""

        return Table.parse_str(
            location=location,
            service=self,
            catalog_name=catalog_name or self.catalog_name,
            schema_name=schema_name or self.schema_name,
            table_name=table_name or self.table_name,
        )

    # -------------------------------------------------------------------------
    # Cache helpers
    # -------------------------------------------------------------------------

    def _cache_key(
        self,
        catalog_name: Optional[str],
        schema_name: Optional[str],
        table_name: Optional[str],
    ) -> str:
        """Build a stable, host-scoped cache key."""
        host = self.client.base_url.to_string() if self.client else "default"
        return f"{host}|{catalog_name}.{schema_name}.{table_name}"

    def _invalidate(
        self,
        catalog_name: Optional[str],
        schema_name: Optional[str],
        table_name: Optional[str],
    ) -> None:
        """Evict one entry from the module-level cache."""
        key = self._cache_key(catalog_name, schema_name, table_name)
        try:
            del _TABLE_INFO_CACHE[key]
        except KeyError:
            pass

    @classmethod
    def invalidate_all(cls) -> None:
        """Clear the entire module-level table-info cache."""
        _TABLE_INFO_CACHE.clear()

    # -------------------------------------------------------------------------
    # Remote fetch — single responsibility, no caching
    # -------------------------------------------------------------------------

    def find_table_remote(
        self,
        catalog_name: str,
        schema_name: str,
        table_name: Optional[str] = None,
        *,
        table_id: Optional[str] = None,
        raise_error: bool = True,
    ) -> Optional[TableInfo]:
        """Raw API lookup — three strategies in order, no cache.

        1. Search by ``table_id`` (full list scan, preferred when id is known).
        2. GET by fully-qualified name (fast path for normal lookups).
        3. Case-insensitive list scan (handles edge cases in naming / quoting).

        Returns ``None`` on miss when ``raise_error=False``.
        """
        uc = self.client.workspace_client().tables

        # --- Strategy 1: search by table_id ---------------------------------
        if table_id:
            logger.debug(
                "Remote fetch [strategy=table_id] id=%s catalog=%s schema=%s",
                table_id, catalog_name, schema_name,
            )
            try:
                for info in uc.list(catalog_name=catalog_name, schema_name=schema_name):
                    if info.table_id == table_id:
                        return info
            except DatabricksError as exc:
                if raise_error:
                    raise ResourceDoesNotExist(
                        f"Failed searching table_id={table_id} in"
                        f" {catalog_name}.{schema_name}"
                    ) from exc
                return None

            if raise_error:
                raise ResourceDoesNotExist(
                    f"Table id={table_id} not found in {catalog_name}.{schema_name}"
                )
            return None

        # --- Strategy 2: GET by full name (fast path) -----------------------
        full_name = (
            table_name
            if isinstance(table_name, str) and table_name.count(".") == 2
            else f"{catalog_name}.{schema_name}.{table_name}"
        )
        logger.debug("Remote fetch [strategy=get] full_name=%s", full_name)
        try:
            return uc.get(full_name=full_name)
        except DatabricksError:
            pass  # fall through to list scan
        except Exception as exc:
            if raise_error:
                raise ResourceDoesNotExist(f"Failed to GET table {full_name}") from exc
            return None

        # --- Strategy 3: case-insensitive list scan -------------------------
        logger.debug(
            "Remote fetch [strategy=list_scan] catalog=%s schema=%s name=%s",
            catalog_name, schema_name, table_name,
        )
        try:
            for info in uc.list(catalog_name=catalog_name, schema_name=schema_name):
                if info.name == table_name or info.name.lower() == str(table_name).lower():
                    return info
        except DatabricksError as exc:
            if raise_error:
                raise ResourceDoesNotExist(
                    f"Failed to list tables in {catalog_name}.{schema_name}"
                ) from exc
            return None

        if raise_error:
            raise ResourceDoesNotExist(f"Table {full_name} not found")
        return None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def find_table(
        self,
        location: Optional[str] = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        table_id: Optional[str] = None,
        raise_error: bool = True,
        use_cache: bool = True,
        cache_ttl: Optional[float] = None,
    ) -> Optional["Table"]:
        """Resolve a table by name or Unity Catalog ID.

        Caching is enabled by default (``use_cache=True``).  Set
        ``use_cache=False`` to force a fresh API call and refresh the entry.

        Args:
            location:     Full string location
            table_name:   Name.
            catalog_name: Override catalog (falls back to service default).
            schema_name:  Override schema (falls back to service default).
            table_id:     Unity Catalog table UUID — triggers id-based search.
            raise_error:  Raise :exc:`ResourceDoesNotExist` when not found.
            use_cache:    ``False`` bypasses and refreshes the cache entry.
            cache_ttl:    Entry TTL in seconds (``None`` → 5 min default).
        """
        _, catalog, schema, name = self.parse_check_location_params(
            location=location, catalog_name=catalog_name, schema_name=schema_name,
            table_name=table_name
        )
        cache_key = self._cache_key(catalog, schema, name)

        # 1. Check local cache -----------------------------------------------
        if use_cache:
            cached: Optional[TableInfo] = _TABLE_INFO_CACHE.get(cache_key)
            if cached is not None:
                logger.debug(
                    "Cache hit [Tables.find_table] key=%s table_id=%s",
                    cache_key, cached.table_id,
                )
                tb = Table(
                    service=self,
                    catalog_name=cached.catalog_name,
                    schema_name=cached.schema_name,
                    table_name=cached.name,
                )
                object.__setattr__(tb, "_infos", cached)
                return tb

        # 2. Fetch remote ----------------------------------------------------
        info = self.find_table_remote(
            catalog_name=catalog,
            schema_name=schema,
            table_name=name,
            table_id=table_id,
            raise_error=raise_error,
        )
        if info is None:
            return None

        # 3. Update local cache ----------------------------------------------
        if use_cache:
            if cache_ttl is not None:
                _TABLE_INFO_CACHE.set(cache_key, info, ttl=cache_ttl)
            else:
                _TABLE_INFO_CACHE.set(cache_key, info)

        tb = Table(
            service=self,
            catalog_name=info.catalog_name,
            schema_name=info.schema_name,
            table_name=info.name,
        )
        object.__setattr__(tb, "_infos", info)
        return tb

    def list_tables(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        *,
        use_cache: bool = True,
        cache_ttl: Optional[float] = None,
    ) -> Iterator["Table"]:
        """Iterate over all tables in a schema, populating the local cache.

        Args:
            catalog_name: Override catalog (falls back to service default).
            schema_name:  Override schema (falls back to service default).
            use_cache:    Populate ``_TABLE_INFO_CACHE`` from results.
            cache_ttl:    Entry TTL in seconds (``None`` → 5 min default).
        """

        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name
        uc = self.client.workspace_client().tables

        for info in uc.list(catalog_name=catalog_name, schema_name=schema_name):
            if use_cache:
                key = self._cache_key(info.catalog_name, info.schema_name, info.name)
                if _TABLE_INFO_CACHE.get(key) is None:
                    if cache_ttl is not None:
                        _TABLE_INFO_CACHE.set(key, info, ttl=cache_ttl)
                    else:
                        _TABLE_INFO_CACHE.set(key, info)

            tb = Table(
                service=self,
                catalog_name=info.catalog_name,
                schema_name=info.schema_name,
                table_name=info.name,
            )
            object.__setattr__(tb, "_infos", info)
            yield tb

