"""
Collection-level service for Unity Catalog schemas.

Provides ``__getitem__``, ``schema()``, ``table()``, ``catalog()``,
``find()``, and ``list()`` against the Databricks catalog API.  Per-resource
DDL/DML lives in :class:`~yggdrasil.databricks.sql.schema.Schema`.

Hierarchy
---------
::

    client.schemas                               # Schemas (root)
    client.schemas(catalog_name="main")          # Schemas scoped to "main"
    client.schemas["main.sales"]                 # Schema  (fully qualified)
    client.schemas["sales"]                      # Schema  (uses default catalog)
    client.schemas["main.sales.orders"]          # Table
    client.schemas["main.sales.orders.price"]    # Column
    client.schemas.list()                        # Iterator[Schema]
    client.schemas.list(catalog_name="main")     # Iterator[Schema] in "main"

Caching strategy
----------------
A module-level :class:`ExpiringDict` (``_SCHEMA_INFO_CACHE``) keyed by
``"host|catalog.schema"`` acts as a local cache (TTL = 5 min by default).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Union

from databricks.sdk.errors import DatabricksError, ResourceDoesNotExist
from databricks.sdk.service.catalog import SchemaInfo

from yggdrasil.databricks.client import DatabricksService
from yggdrasil.dataclasses.expiring import ExpiringDict

from .catalog import Catalog
from .column import Column
from .schema import Schema
from .sql_utils import is_glob_pattern, name_matcher
from .table import Table

__all__ = ["Schemas"]

logger = logging.getLogger(__name__)

# Module-level cache keyed by "host|catalog.schema"; default TTL = 5 minutes.
_SCHEMA_INFO_CACHE: ExpiringDict[str, SchemaInfo] = ExpiringDict(default_ttl=300.0)


@dataclass(frozen=True)
class Schemas(DatabricksService):
    """Collection-level service for Unity Catalog schemas.

    Attach a default ``catalog_name`` (and optional ``schema_name``) so
    callers don't have to repeat them on every call::

        schemas = client.schemas(catalog_name="main")

        # navigate by subscript
        schema = schemas["sales"]                    # Schema ("main.sales")
        schema = schemas["main.sales"]               # Schema  (fully qualified)
        table  = schemas["main.sales.orders"]        # Table

        # explicit factory methods
        schema  = schemas.schema("main.sales")       # Schema
        table   = schemas.table("main.sales.orders") # Table
        catalog = schemas.catalog("main")            # Catalog

        # list
        for sch in schemas.list():
            for tbl in sch.tables():
                ...
    """

    catalog_name: str | None = None
    schema_name: str | None = None

    # ── context rebind ────────────────────────────────────────────────────────

    def __call__(
        self,
        catalog_name: Optional[str] = "",
        schema_name: Optional[str] = "",
        *args,
        **kwargs,
    ):
        if catalog_name == "":
            catalog_name = self.catalog_name

        if schema_name == "":
            schema_name = self.schema_name

        if catalog_name and schema_name:
            return self.find(
                catalog_name=catalog_name,
                schema_name=schema_name,
            )

        return Schemas(
            client=self.client,
            catalog_name=catalog_name,
            schema_name=schema_name,
        )

    # ── dict-like navigation ──────────────────────────────────────────────────

    def __getitem__(self, name: str) -> Union[Schema, Table, Column]:
        """Route a 1-, 2-, 3-, or 4-part dotted name to the right resource.

        * ``schemas["sales"]``                      → :class:`Schema`  (uses ``self.catalog_name``)
        * ``schemas["main.sales"]``                 → :class:`Schema`
        * ``schemas["main.sales.orders"]``          → :class:`Table`
        * ``schemas["main.sales.orders.price"]``    → :class:`Column`
        """
        parts = [p.strip().strip("`") for p in name.split(".")]
        if len(parts) == 1:
            if not self.catalog_name:
                raise ValueError(
                    f"Cannot resolve one-part schema name {name!r} without a"
                    " default catalog — set ``catalog_name`` on the service or"
                    " use a two-part 'catalog.schema' name."
                )
            return self.schema(f"{self.catalog_name}.{parts[0]}")
        if len(parts) == 2:
            return self.schema(f"{parts[0]}.{parts[1]}")
        if len(parts) == 3:
            return self.table(".".join(parts))
        if len(parts) == 4:
            return self.client.columns.column(".".join(parts))
        raise KeyError(
            f"Expected a 1- to 4-part dotted name (schema[.table[.column]] or"
            f" catalog.schema[.table[.column]]), got {name!r} with {len(parts)} parts"
        )

    def __setitem__(self, name: str, new_name: str) -> None:
        """``schemas[key] = "new"`` renames the resource identified by *key*.

        Routing follows :meth:`__getitem__`; *new_name* is the unqualified new name.
        """
        self[name].rename(new_name)

    def __iter__(self) -> Iterator[Schema]:
        """Iterate over schemas in the resolved catalog scope (``self.list()``)."""
        return self.list()

    # ── factory methods ───────────────────────────────────────────────────────

    def schema(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ) -> Schema:
        """Return a :class:`Schema` bound to this service.

        Uses the module-level cache when available.

        Args:
            location:     Two-part dotted ``"catalog.schema"`` name (optional if
                          *catalog_name* / *schema_name* are provided).
            catalog_name: Override catalog (falls back to ``self.catalog_name``).
            schema_name:  Override schema (falls back to ``self.schema_name``).
        """
        c, s = self._resolve_parts(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
        )
        assert c, "No catalog_name: supply it explicitly or set it on the Schemas service"
        assert s, "No schema_name: supply it explicitly or set it on the Schemas service"

        sch = Schema(service=self, catalog_name=c, schema_name=s)
        cached = _SCHEMA_INFO_CACHE.get(self._cache_key(c, s))
        if cached is not None:
            object.__setattr__(sch, "_infos", cached)
            object.__setattr__(sch, "_infos_fetched_at", time.time())
        return sch

    def table(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
    ) -> Table:
        """Return a :class:`Table` via the client-level ``tables`` service.

        Args:
            location:     Three-part dotted name ``"catalog.schema.table"``
                          (or fewer parts, filled by service defaults).
            catalog_name: Override catalog.
            schema_name:  Override schema.
            table_name:   Override table.
        """
        return self.client.tables.table(
            location=location,
            catalog_name=catalog_name or self.catalog_name,
            schema_name=schema_name or self.schema_name,
            table_name=table_name,
        )

    def catalog(self, name: str | None = None) -> Catalog:
        """Return a :class:`Catalog` using this service's client.

        Args:
            name: Catalog name (falls back to ``self.catalog_name``).
        """
        catalog_name = name or self.catalog_name
        assert catalog_name, "No catalog_name: supply it explicitly or set it on the Schemas service"
        return self.client.catalogs.catalog(catalog_name)

    # ── listing ───────────────────────────────────────────────────────────────

    def list(
        self,
        name: str | None = None,
        *,
        catalog_name: str | None = None,
        use_cache: bool = True,
    ) -> Iterator[Schema]:
        """Iterate over visible schemas, optionally scoped to a catalog.

        Args:
            name:         Optional schema-name filter.  When it contains ``*``,
                          matching uses a case-insensitive glob (e.g.
                          ``"sales_*"``, ``"*_raw"``, ``"*"``).
            catalog_name: Catalog to list (falls back to ``self.catalog_name``).
                          When ``None`` the iteration fans out across every
                          visible catalog.  When it contains ``*`` the
                          iteration fans out across catalogs matching the glob.
            use_cache:    Populate ``_SCHEMA_INFO_CACHE`` from results.
        """
        catalog_name = catalog_name if catalog_name is not None else self.catalog_name

        if catalog_name is None or is_glob_pattern(catalog_name):
            for cat in self.client.catalogs.list_catalogs(name=catalog_name, use_cache=use_cache):
                yield from self.list(
                    name=name,
                    catalog_name=cat.catalog_name,
                    use_cache=use_cache,
                )
            return

        uc = self.client.workspace_client().schemas
        matcher = name_matcher(name)

        for info in uc.list(catalog_name=catalog_name):
            if matcher is not None and not matcher(info.name):
                continue

            sch = Schema(
                service=self,
                catalog_name=info.catalog_name,
                schema_name=info.name,
            )
            object.__setattr__(sch, "_infos", info)
            object.__setattr__(sch, "_infos_fetched_at", time.time())

            if use_cache:
                _SCHEMA_INFO_CACHE[self._cache_key(info.catalog_name, info.name)] = info

            yield sch

    # ── remote fetch ──────────────────────────────────────────────────────────

    def find_remote(
        self,
        catalog_name: str,
        schema_name: str,
        *,
        raise_error: bool = True,
    ) -> Optional[SchemaInfo]:
        """Raw API lookup — GET by fully-qualified name, no cache.

        Returns ``None`` on miss when ``raise_error=False``.
        """
        full_name = f"{catalog_name}.{schema_name}"
        logger.debug("Remote fetch [Schemas.find] full_name=%s", full_name)
        try:
            return self.client.workspace_client().schemas.get(full_name=full_name)
        except DatabricksError as exc:
            if raise_error:
                raise ResourceDoesNotExist(f"Schema {full_name} not found") from exc
            return None

    def find(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        raise_error: bool = True,
        cache_ttl: float | None = 300.0,
    ) -> Optional[Schema]:
        """Resolve a schema by name.

        Caching is controlled by ``cache_ttl``.  Set ``cache_ttl=None`` to
        bypass the cache for this lookup.

        Args:
            location:     Two-part dotted ``"catalog.schema"`` name.
            catalog_name: Override catalog (falls back to service default).
            schema_name:  Override schema (falls back to service default).
            raise_error:  Raise :exc:`ResourceDoesNotExist` when not found.
            cache_ttl:    Entry TTL in seconds (``None`` → bypass cache).
        """
        c, s = self._resolve_parts(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
        )
        assert c, "No catalog_name: supply it explicitly or set it on the Schemas service"
        assert s, "No schema_name: supply it explicitly or set it on the Schemas service"

        cache_key = self._cache_key(c, s)

        # 1. Check local cache ---------------------------------------------------
        if cache_ttl is not None:
            cached = _SCHEMA_INFO_CACHE.get(cache_key)
            if cached is not None:
                logger.debug(
                    "Cache hit [Schemas.find] key=%s schema=%s.%s",
                    cache_key, c, s,
                )
                sch = Schema(service=self, catalog_name=c, schema_name=s)
                object.__setattr__(sch, "_infos", cached)
                object.__setattr__(sch, "_infos_fetched_at", time.time())
                return sch

        # 2. Fetch remote --------------------------------------------------------
        info = self.find_remote(catalog_name=c, schema_name=s, raise_error=raise_error)
        if info is None:
            return None

        sch = Schema(
            service=self,
            catalog_name=info.catalog_name,
            schema_name=info.name,
        )
        object.__setattr__(sch, "_infos", info)
        object.__setattr__(sch, "_infos_fetched_at", time.time())

        # 3. Update local cache --------------------------------------------------
        if cache_ttl is not None:
            _SCHEMA_INFO_CACHE.set(cache_key, info, ttl=cache_ttl)
        return sch

    # ── parse helper ──────────────────────────────────────────────────────────

    def parse_location(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Parse a 1- or 2-part dotted name into ``(catalog, schema)``.

        Keyword overrides take precedence over parts extracted from *location*.
        Service defaults fill any remaining blanks.
        """
        if location:
            parts = [p.strip().strip("`") for p in location.split(".")]
            if len(parts) >= 2:
                catalog_name = catalog_name or parts[-2]
                schema_name = schema_name or parts[-1]
            elif len(parts) == 1:
                schema_name = schema_name or parts[0]

        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name
        return catalog_name, schema_name

    def _resolve_parts(
        self,
        location: str | None,
        catalog_name: str | None,
        schema_name: str | None,
    ) -> tuple[Optional[str], Optional[str]]:
        return self.parse_location(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
        )

    # ── cache helpers ─────────────────────────────────────────────────────────

    def _cache_key(self, catalog_name: str, schema_name: str) -> str:
        host = self.client.base_url.to_string() if self.client else "default"
        return f"{host}|{catalog_name}.{schema_name}"

    def invalidate(
        self,
        schema: Schema | str | None = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> None:
        """Evict one entry from the module-level schema-info cache."""
        if schema is not None:
            if isinstance(schema, Schema):
                catalog_name = schema.catalog_name
                schema_name = schema.schema_name
            else:
                catalog_name, schema_name = self.parse_location(schema)

        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        if catalog_name and schema_name:
            try:
                del _SCHEMA_INFO_CACHE[self._cache_key(catalog_name, schema_name)]
            except KeyError:
                pass

    @classmethod
    def invalidate_all(cls) -> None:
        """Clear the entire module-level schema-info cache."""
        _SCHEMA_INFO_CACHE.clear()
