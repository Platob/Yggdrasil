"""
Collection-level service for Unity Catalog tables.

Provides ``find_table`` and ``list_tables`` against the Databricks catalog API.
Per-table DDL/DML lives in :class:`~yggdrasil.databricks.table.table.Table`.

Caching strategy
----------------
A module-level :class:`ExpiringDict` (``_TABLE_CACHE``) keyed by
``"host|catalog.schema.table"`` acts as a fast *local* cache so the same
table lookup never hits the API twice within the TTL window.

    1. **Local** — check ``_TABLE_INFO_CACHE``; return immediately on hit.
    2. **Remote** — call ``_find_table_remote``; only reached on miss.
    3. **Update** — store the remote result back into the local cache.

Cache entries can be invalidated per-table via :meth:`Tables._invalidate`.
"""

from __future__ import annotations

import logging
from typing import Optional, Iterator, TYPE_CHECKING, Any, Iterable

from databricks.sdk.errors import DatabricksError, ResourceDoesNotExist
from databricks.sdk.service.catalog import TableInfo, TableType

from yggdrasil.databricks.client import DatabricksService
from yggdrasil.databricks.sql.sql_utils import is_glob_pattern, name_matcher, quote_ident
from .table import Table, _VIEW_TABLE_TYPES
from ...data import Mode, ModeLike

if TYPE_CHECKING:
    from yggdrasil.databricks.catalog.catalog import Catalog
    from yggdrasil.databricks.column.column import Column
    from yggdrasil.databricks.schema.schema import Schema

__all__ = ["Tables"]

logger = logging.getLogger(__name__)


class Tables(DatabricksService):
    """Collection-level service for Unity Catalog tables.

    Attach default catalog / schema context so callers don't have to repeat
    them on every call::

        tables = client.tables(catalog_name="main", schema_name="sales")
        table  = tables.find_table("orders")
        for t in tables.list_tables():
            ...
    """

    def __init__(
        self,
        client=None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ):
        super().__init__(client=client)
        self.catalog_name = catalog_name
        self.schema_name = schema_name

    def __call__(
        self,
        catalog_name: Optional[str] = "",
        schema_name: Optional[str] = "",
        table_name: Optional[str] = "",
        *args,
        **kwargs
    ):
        if catalog_name == "":
            catalog_name = self.catalog_name

        if schema_name == "":
            schema_name = self.schema_name

        if catalog_name and schema_name and table_name:
            return self.find_table(
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name
            )

        return Tables(
            client=self.client,
            catalog_name=catalog_name,
            schema_name=schema_name,
        )

    def __getstate__(self):
        state = super().__getstate__()
        state["catalog_name"] = self.catalog_name
        state["schema_name"] = self.schema_name
        return state

    def __setstate__(self, state):
        object.__setattr__(self, "catalog_name", state["catalog_name"])
        object.__setattr__(self, "schema_name", state["schema_name"])
        super().__setstate__(state)

    # -------------------------------------------------------------------------
    # Dict-like navigation — uses catalog/schema defaults on the service
    # -------------------------------------------------------------------------

    def __getitem__(self, name: str) -> "Table | Column":
        """Route a 1-, 2-, 3-, or 4-part dotted name to the right resource.

        Service defaults fill any missing leading parts.

        * ``tables["orders"]``                      → :class:`Table` (needs ``catalog_name`` + ``schema_name`` defaults)
        * ``tables["sales.orders"]``                → :class:`Table` (needs ``catalog_name`` default)
        * ``tables["main.sales.orders"]``           → :class:`Table`
        * ``tables["main.sales.orders.price"]``     → :class:`Column`
        """
        parts = [p.strip().strip("`") for p in name.split(".")]
        n = len(parts)

        if n == 4:
            return self.client.columns.column(".".join(parts))

        if n == 1:
            if not (self.catalog_name and self.schema_name):
                raise ValueError(
                    f"Cannot resolve one-part table name {name!r} without"
                    " default catalog_name + schema_name — set them on the"
                    " service or pass a fully-qualified name."
                )
            return self.table(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_name=parts[0],
            )
        if n == 2:
            if not self.catalog_name:
                raise ValueError(
                    f"Cannot resolve two-part table name {name!r} without"
                    " a default catalog_name — set it on the service or"
                    " pass a three-part 'catalog.schema.table' name."
                )
            return self.table(
                catalog_name=self.catalog_name,
                schema_name=parts[0],
                table_name=parts[1],
            )
        if n == 3:
            return self.table(location=".".join(parts))

        raise KeyError(
            f"Expected a 1- to 4-part dotted name (table[.column] or"
            f" catalog.schema.table[.column]), got {name!r} with {n} parts"
        )

    def __setitem__(self, name: str, new_name: str) -> None:
        """``tables[key] = "new"`` renames the resource identified by *key*.

        Routing follows :meth:`__getitem__`; *new_name* is the unqualified new name.
        """
        self[name].rename(new_name)

    def __iter__(self) -> Iterator["Table"]:
        """Iterate over tables in the resolved catalog/schema scope."""
        return self.list_tables()

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
        location: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        safe_chars: bool = True,
    ) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
        if location:
            c, s, t = self.parse_catalog_schema_table_names(location)
            catalog_name, schema_name, table_name = c or catalog_name, s or schema_name, t or table_name

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
        location: Table | str | None = None,
        *,
        table_name: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ) -> "Table":
        """Return a :class:`~yggdrasil.databricks.table.table.Table` bound to this service."""

        return Table.from_(
            obj=location,
            service=self,
            catalog_name=catalog_name or self.catalog_name,
            schema_name=schema_name or self.schema_name,
            table_name=table_name,
        )

    def view(
        self,
        location: Table | str | None = None,
        *,
        table_name: str | None = None,
        view_name: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ) -> "Table":
        """Return a :class:`~yggdrasil.databricks.table.table.Table` bound to this service.

        Alias for :meth:`table` — Unity Catalog stores views in the same
        ``tables`` API as managed/external tables, so the returned
        :class:`Table` covers both. ``view_name`` is accepted as a
        convenience alias for ``table_name``.
        """
        return Table.from_(
            obj=location,
            service=self,
            catalog_name=catalog_name or self.catalog_name,
            schema_name=schema_name or self.schema_name,
            table_name=table_name or view_name,
        )

    def catalog(self, name: str | None = None) -> "Catalog":
        """Return a :class:`Catalog` using this service's client.

        Args:
            name: Catalog name (falls back to ``self.catalog_name``).
        """
        from yggdrasil.databricks.catalog.catalog import Catalog as _Catalog
        from yggdrasil.databricks.catalog.catalogs import Catalogs
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
        """Return a :class:`Schema` using this service's client.

        Args:
            name:         Two-part ``"catalog.schema"`` name (optional if
                          *catalog_name* / *schema_name* are provided).
            catalog_name: Override catalog (falls back to ``self.catalog_name``).
            schema_name:  Override schema (falls back to ``self.schema_name``).
        """
        from yggdrasil.databricks.schema.schema import Schema as _Schema
        from yggdrasil.databricks.catalog.catalogs import Catalogs

        if name and "." in name:
            parts = [p.strip().strip("`") for p in name.split(".", 1)]
            catalog_name = catalog_name or parts[0]
            schema_name = schema_name or parts[1]

        return _Schema(
            service=Catalogs(client=self.client),
            catalog_name=catalog_name or self.catalog_name,
            schema_name=schema_name or self.schema_name,
        )

    # -------------------------------------------------------------------------
    # Remote fetch — single responsibility, no caching
    # -------------------------------------------------------------------------

    def find_table_remote(
        self,
        catalog_name: str,
        schema_name: str,
        table_name: str | None = None,
        *,
        table_id: str | None = None,
        default: Any = ...,
    ) -> Optional[TableInfo]:
        """Raw API lookup — three strategies in order, no cache.

        1. Search by ``table_id`` (full list scan, preferred when id is known).
        2. GET by fully qualified name (fast path for normal lookups).
        3. Case-insensitive list scan (handles edge cases in naming / quoting).

        Returns ``None`` on miss when ``raise_error=False``.
        """
        uc = self.client.workspace_client().tables

        # --- Strategy 1: search by table_id ---------------------------------
        if table_id:
            logger.debug(
                "Fetching table by id=%s in %s.%s",
                table_id, catalog_name, schema_name,
            )
            try:
                for info in uc.list(catalog_name=catalog_name, schema_name=schema_name):
                    if info.table_id == table_id:
                        return info
            except DatabricksError as exc:
                if default is ...:
                    raise ResourceDoesNotExist(
                        f"Failed searching table_id={table_id} in"
                        f" {catalog_name}.{schema_name}"
                    ) from exc
                return default

            if default is ...:
                raise ResourceDoesNotExist(
                    f"Table id={table_id} not found in {catalog_name}.{schema_name}"
                )
            return default

        # --- Strategy 2: GET by full name (fast path) -----------------------
        full_name = (
            table_name
            if isinstance(table_name, str) and table_name.count(".") == 2
            else f"{catalog_name}.{schema_name}.{table_name}"
        )
        logger.debug("Fetching table %s via GET", full_name)
        try:
            return uc.get(full_name=full_name)
        except DatabricksError:
            pass  # fall through to list scan
        except Exception as exc:
            if default is ...:
                raise ResourceDoesNotExist(f"Failed to GET table {full_name}") from exc
            return default

        # --- Strategy 3: case-insensitive list scan -------------------------
        logger.debug(
            "Scanning %s.%s for table %r (case-insensitive list)",
            catalog_name, schema_name, table_name,
        )
        try:
            for info in uc.list(catalog_name=catalog_name, schema_name=schema_name):
                if info.name == table_name or info.name.lower() == str(table_name).lower():
                    return info
        except DatabricksError as exc:
            if default is ...:
                raise ResourceDoesNotExist(
                    f"Failed to list tables in {catalog_name}.{schema_name}"
                ) from exc
            return default

        if default is ...:
            raise ResourceDoesNotExist(f"Table {full_name} not found")
        return None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def get(
        self,
        location: str | Table | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        table_id: str | None = None,
        default: Any = None,
        cache_ttl: float | None = 300.0,
    ) -> Optional["Table"]:
        return self.find_table(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            table_id=table_id,
            default=default,
            cache_ttl=cache_ttl,
        )

    def find_table(
        self,
        location: str | Table | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        table_id: str | None = None,
        default: Any = ...,
        cache_ttl: float | None = 300.0,
    ) -> Optional["Table"]:
        """Resolve a table by name or Unity Catalog ID.

        Caching is controlled only by ``cache_ttl``. Set ``cache_ttl=None``
        to bypass the cache for this lookup.

        Args:
            location:     Full string location
            table_name:   Name.
            catalog_name: Override catalog (falls back to service default).
            schema_name:  Override schema (falls back to service default).
            table_id:     Unity Catalog table UUID — triggers id-based search.
            default:  Raise :exc:`ResourceDoesNotExist` when not found.
            cache_ttl:    Entry TTL in seconds (``None`` → 5 min default).
        """
        if isinstance(location, Table):
            return location

        _, catalog, schema, name = self.parse_check_location_params(
            location=location, catalog_name=catalog_name, schema_name=schema_name,
            table_name=table_name
        )

        # 2. Fetch remote ----------------------------------------------------
        info = self.find_table_remote(
            catalog_name=catalog,
            schema_name=schema,
            table_name=name,
            table_id=table_id,
            default=None,
        )
        if info is None:
            if default is ...:
                raise ResourceDoesNotExist(
                    f"Table {catalog}.{schema}.{name} not found"
                )
            return default

        tb = Table(
            service=self,
            catalog_name=info.catalog_name,
            schema_name=info.schema_name,
            table_name=info.name,
        )
        tb._store_infos(info)

        return tb

    def list_tables(
        self,
        name: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        *,
        table_types: Iterable[TableType] | None = None,
        cache_ttl: float | None = 300.0,
    ) -> Iterator["Table"]:
        """Iterate over tables in the resolved catalog/schema scope.

        Any of ``name``, ``catalog_name``, or ``schema_name`` may be a
        case-insensitive glob (``"sales_*"``, ``"*_raw"``, ``"prefix_*_table"``,
        ``"*"``).  Globbed catalog/schema names fan out across the matching
        resources; ``None`` still means "all" at that level.

        Args:
            name:         Optional table-name filter (exact or glob).
            catalog_name: Override catalog (falls back to service default).
                          Accepts a glob to fan out across catalogs.
            schema_name:  Override schema (falls back to service default).
                          Accepts a glob to fan out across schemas.
                          When ``None``, iterates every schema in the resolved
                          catalog scope.  When both catalog and schema are
                          ``None``, iterates every visible catalog and schema.
            table_types:  Restrict the yielded :attr:`Table.table_type`
                          set. ``None`` (default) yields every securable
                          UC reports — managed / external tables and
                          view-shaped securables alike. Pass
                          :data:`_VIEW_TABLE_TYPES` to filter to views,
                          or any subset to narrow further.
            cache_ttl:    Entry TTL in seconds (``None`` → 5 min default).
        """
        catalog_name = self.catalog_name if catalog_name is None else catalog_name
        schema_name = self.schema_name if schema_name is None else schema_name
        allowed = frozenset(table_types) if table_types is not None else None

        if catalog_name is None or is_glob_pattern(catalog_name):
            from yggdrasil.databricks.catalog.catalogs import Catalogs

            for catalog in Catalogs(client=self.client).list_catalogs(name=catalog_name):
                yield from self.list_tables(
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
                yield from self.list_tables(
                    name=name,
                    catalog_name=catalog_name,
                    schema_name=schema_info.name,
                    table_types=allowed,
                    cache_ttl=cache_ttl,
                )
            return

        uc = self.client.workspace_client().tables
        matcher = name_matcher(name)
        logger.debug(
            "Listing tables in %s.%s (name_filter=%s, table_types=%s)",
            catalog_name, schema_name, name,
            sorted(t.value for t in allowed) if allowed is not None else None,
        )

        yielded = 0
        for info in uc.list(catalog_name=catalog_name, schema_name=schema_name):
            if allowed is not None and info.table_type not in allowed:
                continue
            if matcher is not None and not matcher(info.name):
                continue

            tb = Table(
                service=self,
                catalog_name=info.catalog_name,
                schema_name=info.schema_name,
                table_name=info.name,
            )
            tb._store_infos(info)

            yielded += 1
            yield tb
        logger.debug(
            "Listed %d tables in %s.%s", yielded, catalog_name, schema_name,
        )

    def list_views(
        self,
        name: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        *,
        table_types: Iterable[TableType] | None = None,
        cache_ttl: float | None = 300.0,
    ) -> Iterator["Table"]:
        """Iterate over view-shaped securables only.

        Convenience wrapper around :meth:`list_tables` with
        ``table_types`` defaulted to ``{VIEW, MATERIALIZED_VIEW,
        METRIC_VIEW}``. Pass an explicit ``table_types`` to narrow
        further (e.g. only :data:`TableType.MATERIALIZED_VIEW`).
        """
        return self.list_tables(
            name=name,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_types=table_types if table_types is not None else _VIEW_TABLE_TYPES,
            cache_ttl=cache_ttl,
        )

    # ── concat_tables — create/update a UNION view over multiple tables ──────

    @staticmethod
    def _common_table_name_root(names: Iterable[str]) -> str:
        """Longest shared prefix across ``names``, stripped of trailing separators.

        Returns an empty string when the inputs share no leading characters.
        Trailing ``_ - . `` are trimmed so a prefix like ``"sales_"`` becomes
        a valid unqualified view name (``"sales"``).
        """
        name_list = [n for n in names if n]
        if not name_list:
            return ""
        if len(name_list) == 1:
            return name_list[0].rstrip("_-. ")

        prefix = name_list[0]
        for other in name_list[1:]:
            # Shrink until ``prefix`` is a prefix of ``other``.
            while not other.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix.rstrip("_-. ")

    def concat_tables(
        self,
        tables: Iterable["Table"],
        *,
        view_name: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        by_name: bool = True,
        cast: bool = True,
        comment: str | None = None,
        mode: ModeLike = Mode.OVERWRITE,
    ) -> "Table":
        """Create or update a view that concatenates *tables* with ``UNION ALL``.

        Resolves the view name + parent (deriving from the inputs' shared
        prefix and the first input's catalog/schema when not given) and
        delegates the actual DDL to :meth:`View.concat_tables` — which does
        the smart by-name + type-promotion projection when ``cast`` is
        ``True``.

        Args:
            tables:
                Iterable of :class:`Table` or :class:`View` instances to
                union.  At least one input is required.
            view_name:
                Unqualified view name.  When omitted, the longest shared
                prefix of the input table names (trimmed of trailing
                ``_ - . ``) is used.  Raises ``ValueError`` when the inputs
                share no common prefix.
            catalog_name, schema_name:
                Override the view location.  Fall back to the service
                defaults, then to the first input table's catalog/schema.
            by_name:
                Forwarded to :meth:`View.concat_tables`.  Only consulted
                when ``cast`` is ``False``.
            cast:
                Forwarded to :meth:`View.concat_tables` — enables smart
                column-name alignment with ``CAST(NULL AS <ddl>)`` fills
                for columns missing from a given input.  Default ``True``.
            comment:
                Optional ``COMMENT`` on the view.
            mode:
                Passed through to :meth:`View.create`.  Defaults to
                :attr:`Mode.OVERWRITE` so the view is created or
                replaced atomically.

        Returns:
            The :class:`View` that was created or updated.
        """
        tables_list = list(tables)
        if not tables_list:
            raise ValueError("concat_tables requires at least one Table")

        if not view_name:
            view_name = self._common_table_name_root(
                t.table_name for t in tables_list
            )
            if not view_name:
                input_names = [t.name for t in tables_list]
                raise ValueError(
                    "concat_tables could not derive a view name from "
                    f"{input_names!r}; the input tables share no common "
                    "prefix.  Pass view_name explicitly."
                )

        first = tables_list[0]
        catalog_name = catalog_name or self.catalog_name or first.catalog_name
        schema_name = schema_name or self.schema_name or first.schema_name

        view = self.table(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=view_name,
        )
        return view.concat_tables(
            tables_list,
            by_name=by_name,
            cast=cast,
            comment=comment,
            mode=mode,
        )