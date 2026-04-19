"""
Collection-level service for Unity Catalog views.

Provides ``find_view`` and ``list_views`` against the Databricks catalog API.
Per-view DDL lives in :class:`~yggdrasil.databricks.sql.view.View`.

Unity Catalog stores views in the same ``tables`` endpoint as regular tables;
they are identified by ``table_type`` ∈ ``{VIEW, MATERIALIZED_VIEW,
METRIC_VIEW}``.  This service transparently filters for those subtypes.

Caching strategy
----------------
A module-level :class:`ExpiringDict` (``_VIEW_CACHE``) keyed by
``"host|catalog.schema.view"`` acts as a fast *local* cache so the same
view lookup never hits the API twice within the TTL window.

    1. **Local** — check ``_VIEW_CACHE``; return immediately on hit.
    2. **Remote** — call ``find_view_remote``; only reached on miss.
    3. **Update** — store the remote result back into the local cache.

Cache entries can be invalidated per-view via
:meth:`Views.invalidate_cached_view`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional, TYPE_CHECKING

from databricks.sdk.errors import DatabricksError, ResourceDoesNotExist
from databricks.sdk.service.catalog import TableInfo, TableType

from yggdrasil.databricks.client import DatabricksService
from yggdrasil.databricks.sql.sql_utils import is_glob_pattern, name_matcher, quote_ident
from yggdrasil.dataclasses.expiring import ExpiringDict

from .view import View

if TYPE_CHECKING:
    from .catalog import Catalog
    from .column import Column
    from .schema import Schema

__all__ = ["Views"]

logger = logging.getLogger(__name__)

# Module-level cache keyed by "host|catalog.schema.view"; default TTL = 5 minutes.
_VIEW_CACHE: ExpiringDict[str, View] = ExpiringDict(default_ttl=300.0)

_VIEW_TABLE_TYPES: frozenset[TableType] = frozenset(
    {TableType.VIEW, TableType.MATERIALIZED_VIEW, TableType.METRIC_VIEW}
)


def _is_view_info(info: TableInfo) -> bool:
    """Return ``True`` when *info* describes a view-like securable."""
    return info.table_type in _VIEW_TABLE_TYPES


@dataclass(frozen=True)
class Views(DatabricksService):
    """Collection-level service for Unity Catalog views.

    Attach default catalog / schema context so callers don't have to repeat
    them on every call::

        views = client.views(catalog_name="main", schema_name="sales")
        view  = views.find_view("orders_summary")
        for v in views.list_views():
            ...
    """

    catalog_name: str | None = None
    schema_name: str | None = None
    view_name: str | None = None

    # ── context rebind ────────────────────────────────────────────────────────

    def __call__(
        self,
        catalog_name: Optional[str] = "",
        schema_name: Optional[str] = "",
        view_name: Optional[str] = "",
        *args,
        **kwargs,
    ):
        if catalog_name == "":
            catalog_name = self.catalog_name
        if schema_name == "":
            schema_name = self.schema_name
        if view_name == "":
            view_name = self.view_name

        if catalog_name and schema_name and view_name:
            return self.find_view(
                catalog_name=catalog_name,
                schema_name=schema_name,
                view_name=view_name,
            )

        return Views(
            client=self.client,
            catalog_name=catalog_name,
            schema_name=schema_name,
            view_name=view_name,
        )

    # ── dict-like navigation — uses catalog/schema defaults ──────────────────

    def __getitem__(self, name: str) -> "View | Column":
        """Route a 1-, 2-, 3-, or 4-part dotted name to the right resource.

        Service defaults fill any missing leading parts.

        * ``views["orders_summary"]``                     → :class:`View` (needs ``catalog_name`` + ``schema_name`` defaults)
        * ``views["sales.orders_summary"]``               → :class:`View` (needs ``catalog_name`` default)
        * ``views["main.sales.orders_summary"]``          → :class:`View`
        * ``views["main.sales.orders_summary.price"]``    → :class:`Column`
        """
        parts = [p.strip().strip("`") for p in name.split(".")]
        n = len(parts)

        if n == 4:
            return self.client.columns.column(".".join(parts))

        if n == 1:
            if not (self.catalog_name and self.schema_name):
                raise ValueError(
                    f"Cannot resolve one-part view name {name!r} without"
                    " default catalog_name + schema_name — set them on the"
                    " service or pass a fully-qualified name."
                )
            return self.view(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                view_name=parts[0],
            )
        if n == 2:
            if not self.catalog_name:
                raise ValueError(
                    f"Cannot resolve two-part view name {name!r} without"
                    " a default catalog_name — set it on the service or"
                    " pass a three-part 'catalog.schema.view' name."
                )
            return self.view(
                catalog_name=self.catalog_name,
                schema_name=parts[0],
                view_name=parts[1],
            )
        if n == 3:
            return self.view(location=".".join(parts))

        raise KeyError(
            f"Expected a 1- to 4-part dotted name (view[.column] or"
            f" catalog.schema.view[.column]), got {name!r} with {n} parts"
        )

    def __setitem__(self, name: str, new_name: str) -> None:
        """``views[key] = "new"`` renames the resource identified by *key*.

        Routing follows :meth:`__getitem__`; *new_name* is the unqualified new name.
        """
        self[name].rename(new_name)

    def __iter__(self) -> Iterator[View]:
        """Iterate over views in the resolved catalog/schema scope."""
        return self.list_views()

    # ── name parsing ──────────────────────────────────────────────────────────

    def parse_catalog_schema_view_names(
        self,
        full_name: str,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        parts = [_.strip("`") for _ in full_name.split(".")]

        if len(parts) == 0:
            return self.catalog_name, self.schema_name, None
        if len(parts) == 1:
            return self.catalog_name, self.schema_name, parts[0]
        if len(parts) == 2:
            return self.catalog_name, parts[0], parts[1]

        catalog_name, schema_name, view_name = parts[-3], parts[-2], parts[-1]
        return (
            catalog_name or self.catalog_name,
            schema_name or self.schema_name,
            view_name,
        )

    def parse_check_location_params(
        self,
        location: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        view_name: str | None = None,
        safe_chars: bool = True,
    ) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
        if location:
            c, s, v = self.parse_catalog_schema_view_names(location)
            catalog_name = catalog_name or c
            schema_name = schema_name or s
            view_name = view_name or v

        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        assert catalog_name, "No catalog name given"
        assert schema_name, "No schema name given"
        assert view_name, "No view name given"

        if safe_chars:
            location = (
                f"{quote_ident(catalog_name)}"
                f".{quote_ident(schema_name)}"
                f".{quote_ident(view_name)}"
            )
        else:
            location = f"{catalog_name}.{schema_name}.{view_name}"

        return location, catalog_name, schema_name, view_name

    # ── factories — navigation into other resources ───────────────────────────

    def view(
        self,
        location: str | None = None,
        *,
        view_name: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ) -> View:
        """Return a :class:`View` bound to this service."""
        return View.parse_str(
            location=location,
            service=self,
            catalog_name=catalog_name or self.catalog_name,
            schema_name=schema_name or self.schema_name,
            view_name=view_name or self.view_name,
        )

    def catalog(self, name: str | None = None) -> "Catalog":
        from .catalog import Catalog as _Catalog
        from .catalogs import Catalogs
        return _Catalog(
            service=Catalogs(client=self.client),
            catalog_name=name or self.catalog_name,
        )

    def schema(
        self,
        name: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ) -> "Schema":
        from .schema import Schema as _Schema
        from .catalogs import Catalogs

        if name and "." in name:
            parts = [p.strip().strip("`") for p in name.split(".", 1)]
            catalog_name = catalog_name or parts[0]
            schema_name = schema_name or parts[1]

        return _Schema(
            service=Catalogs(client=self.client),
            catalog_name=catalog_name or self.catalog_name,
            schema_name=schema_name or self.schema_name,
        )

    # ── cache helpers ─────────────────────────────────────────────────────────

    def _cache_key(
        self,
        catalog_name: Optional[str],
        schema_name: Optional[str],
        view_name: Optional[str],
    ) -> str:
        host = self.client.base_url.to_string() if self.client else "default"
        return f"{host}|{catalog_name}.{schema_name}.{view_name}"

    def invalidate_cached_view(
        self,
        view: View | str | None = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        view_name: Optional[str] = None,
    ) -> None:
        if view is not None:
            if isinstance(view, View):
                catalog_name = view.catalog_name
                schema_name = view.schema_name
                view_name = view.view_name
            else:
                catalog_name, schema_name, view_name = self.parse_catalog_schema_view_names(view)

        self._invalidate_cached_view(catalog_name, schema_name, view_name)

    def _invalidate_cached_view(
        self,
        catalog_name: Optional[str],
        schema_name: Optional[str],
        view_name: Optional[str],
    ) -> None:
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name
        if not view_name:
            return

        key = self._cache_key(catalog_name, schema_name, view_name)
        try:
            del _VIEW_CACHE[key]
        except KeyError:
            pass

    @classmethod
    def invalidate_all(cls) -> None:
        """Clear the entire module-level view cache."""
        _VIEW_CACHE.clear()

    # ── remote fetch ──────────────────────────────────────────────────────────

    def find_view_remote(
        self,
        catalog_name: str,
        schema_name: str,
        view_name: str | None = None,
        *,
        raise_error: bool = True,
    ) -> Optional[TableInfo]:
        """Raw API lookup — GET by fully-qualified name, no cache.

        Returns ``None`` on miss when ``raise_error=False``.  Raises
        :class:`ResourceDoesNotExist` when the resource exists but isn't a view.
        """
        uc = self.client.workspace_client().tables
        full_name = (
            view_name
            if isinstance(view_name, str) and view_name.count(".") == 2
            else f"{catalog_name}.{schema_name}.{view_name}"
        )
        logger.debug("Remote fetch [Views.find] full_name=%s", full_name)

        try:
            info = uc.get(full_name=full_name)
        except DatabricksError as exc:
            # Fall back to a case-insensitive list scan over the schema — handles
            # quoting/casing edge cases.
            try:
                for candidate in uc.list(catalog_name=catalog_name, schema_name=schema_name):
                    name = candidate.name or ""
                    if name == view_name or name.casefold() == str(view_name).casefold():
                        if _is_view_info(candidate):
                            return candidate
                        break
            except DatabricksError:
                pass

            if raise_error:
                raise ResourceDoesNotExist(f"View {full_name} not found") from exc
            return None

        if info is not None and not _is_view_info(info):
            if raise_error:
                raise ResourceDoesNotExist(
                    f"{full_name} exists but is not a view (table_type={info.table_type})"
                )
            return None
        return info

    # ── public API ────────────────────────────────────────────────────────────

    def find_view(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        view_name: str | None = None,
        raise_error: bool = True,
        cache_ttl: float | None = 300.0,
    ) -> Optional[View]:
        """Resolve a view by name.

        Caching is controlled by ``cache_ttl``.  Set ``cache_ttl=None`` to
        bypass the cache for this lookup.
        """
        _, catalog, schema, name = self.parse_check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            view_name=view_name,
        )
        cache_key = self._cache_key(catalog, schema, name)

        # 1. Check local cache ----------------------------------------------
        if cache_ttl is not None:
            cached: Optional[View] = _VIEW_CACHE.get(cache_key)
            if cached is not None:
                logger.debug(
                    "Cache hit [Views.find_view] key=%s view=%s",
                    cache_key, cached.full_name(),
                )
                object.__setattr__(cached, "service", self)
                return cached

        # 2. Fetch remote ---------------------------------------------------
        info = self.find_view_remote(
            catalog_name=catalog,
            schema_name=schema,
            view_name=name,
            raise_error=raise_error,
        )
        if info is None:
            return None

        vw = View(
            service=self,
            catalog_name=info.catalog_name,
            schema_name=info.schema_name,
            view_name=info.name,
        )
        object.__setattr__(vw, "_infos", info)

        # 3. Update local cache ---------------------------------------------
        if cache_ttl is not None:
            _VIEW_CACHE.set(cache_key, vw, ttl=cache_ttl)
        return vw

    def list_views(
        self,
        name: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        *,
        table_types: Iterator[TableType] | tuple[TableType, ...] | None = None,
        cache_ttl: float | None = 300.0,
    ) -> Iterator[View]:
        """Iterate over views in the resolved catalog/schema scope.

        Any of ``name``, ``catalog_name``, or ``schema_name`` may be a
        case-insensitive glob (``"v_*"``, ``"*_summary"``, ``"*"``).  Globbed
        catalog/schema names fan out across the matching resources; ``None``
        still means "all" at that level.

        Args:
            name:         Optional view-name filter (exact or glob).
            catalog_name: Override catalog (falls back to service default).
                          Accepts a glob to fan out across catalogs.
                          When ``None``, iterates every visible catalog.
            schema_name:  Override schema (falls back to service default).
                          Accepts a glob to fan out across schemas.
                          When ``None``, iterates every schema in the scope.
            table_types:  Restrict which view subtypes to yield.  Defaults to
                          ``{VIEW, MATERIALIZED_VIEW, METRIC_VIEW}``.
            cache_ttl:    Entry TTL in seconds (``None`` → 5 min default).
        """
        catalog_name = self.catalog_name if catalog_name is None else catalog_name
        schema_name = self.schema_name if schema_name is None else schema_name

        allowed = (
            frozenset(table_types) if table_types is not None else _VIEW_TABLE_TYPES
        )

        if catalog_name is None or is_glob_pattern(catalog_name):
            from .catalogs import Catalogs

            for catalog in Catalogs(client=self.client).list_catalogs(name=catalog_name):
                yield from self.list_views(
                    name=name,
                    catalog_name=catalog.catalog_name,
                    schema_name=schema_name,
                    table_types=allowed,
                    cache_ttl=cache_ttl,
                )
            return

        if schema_name is None or is_glob_pattern(schema_name):
            schema_matcher = name_matcher(schema_name)
            for schema_info in self.client.workspace_client().schemas.list(
                catalog_name=catalog_name,
            ):
                if schema_matcher is not None and not schema_matcher(schema_info.name):
                    continue
                yield from self.list_views(
                    name=name,
                    catalog_name=catalog_name,
                    schema_name=schema_info.name,
                    table_types=allowed,
                    cache_ttl=cache_ttl,
                )
            return

        uc = self.client.workspace_client().tables
        matcher = name_matcher(name)

        for info in uc.list(catalog_name=catalog_name, schema_name=schema_name):
            if info.table_type not in allowed:
                continue

            if matcher is not None and not matcher(info.name):
                continue

            vw = View(
                service=self,
                catalog_name=info.catalog_name,
                schema_name=info.schema_name,
                view_name=info.name,
            )
            object.__setattr__(vw, "_infos", info)

            if cache_ttl is not None:
                key = self._cache_key(info.catalog_name, info.schema_name, info.name)
                if _VIEW_CACHE.get(key) is None:
                    _VIEW_CACHE.set(key, vw, ttl=cache_ttl)
            yield vw
