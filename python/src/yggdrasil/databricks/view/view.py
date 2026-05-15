"""
Per-view resource: DDL, schema introspection and grant helpers.

The :class:`View` dataclass wraps a single Unity Catalog view and exposes
instance-level methods only.  Collection operations (``find_view``,
``list_views``) live in :mod:`~yggdrasil.databricks.view.views`.

In Unity Catalog, views are stored in the same ``tables`` API as tables and
are distinguished by ``table_type`` (``VIEW``, ``MATERIALIZED_VIEW``, or
``METRIC_VIEW``).  They expose ``view_definition`` (the SELECT text) and
``view_dependencies`` on :class:`TableInfo`.

Caching strategy
----------------
``_infos`` and ``_columns`` are instance-level caches (``_infos`` is
TTL-guarded; ``_columns`` is derived from ``_infos``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import  field
from typing import Any, Iterable, Mapping, Optional, TYPE_CHECKING

from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import (
    SecurableType,
    TableInfo,
    TableType,
)

from yggdrasil.concurrent.threading import Job
from yggdrasil.databricks.client import DatabricksResource
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io import URL
from yggdrasil.data.enums.mode import ModeLike, Mode
from yggdrasil.databricks.column.column import Column
from yggdrasil.databricks.sql.sql_utils import (
    DEFAULT_TAG_COLLATION,
    _safe_str,
    databricks_tag_literal,
    escape_sql_string,
    quote_ident,
    quote_principal,
)

if TYPE_CHECKING:
    from yggdrasil.data.types import DataType

    from yggdrasil.databricks.catalog.catalog import Catalog
    from yggdrasil.databricks.sql.engine import SQLEngine
    from yggdrasil.databricks.schema.schema import Schema as UCSchema
    from yggdrasil.databricks.table.table import Table
    from .views import Views

__all__ = ["View"]

logger = logging.getLogger(__name__)

INFOS_TTL: float = 300.0


class View(DatabricksResource):
    """A single Unity Catalog view — DDL, schema, grants and tags."""

    catalog_name: str = "default"
    schema_name: str = "default"
    view_name: str = "default"

    _infos: Optional[TableInfo] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _infos_fetched_at: float | None = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _columns: Optional[list[Column]] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )

    def __init__(
        self,
        service: "Views | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        view_name: str | None = None,
        *,
        infos: TableInfo | None = None,
        infos_fetched_at: float | None = None,
        columns: list[Column] | None = None,
        **kwargs,
    ):
        if service is None:
            from .views import Views
            service = Views.current()

        super().__init__(service=service)
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.view_name = view_name
        self._infos = infos
        self._infos_fetched_at = infos_fetched_at
        self._columns = columns

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.view_name

    @property
    def table_name(self) -> str:
        """Alias so :class:`Column` (which references ``table.table_name``) works."""
        return self.view_name

    def schema_full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}"

    def full_name(self, safe: str | bool | None = None) -> str:
        if safe:
            q = safe if isinstance(safe, str) else "`"
            return (
                f"{q}{self.catalog_name}{q}"
                f".{q}{self.schema_name}{q}"
                f".{q}{self.view_name}{q}"
            )
        return f"{self.catalog_name}.{self.schema_name}.{self.view_name}"

    @property
    def url(self) -> URL:
        return (
            self.client.base_url
            .with_path(f"/explore/data/{self.catalog_name}/{self.schema_name}/{self.view_name}")
        )

    def __repr__(self) -> str:
        return f"View({self.url.to_string()!r})"

    def __str__(self) -> str:
        return self.full_name(safe=True)

    def __getitem__(self, item: str) -> Column:
        return self.column(item)

    def __setitem__(self, item: str, new_name: str) -> None:
        """``view["old_col"] = "new_col"`` renames a column.

        Renaming view columns requires the view to expose a persisted ``ALTER``
        path; Databricks supports this for :data:`TableType.VIEW` via
        ``ALTER VIEW … RENAME COLUMN``.
        """
        self.column(item).rename(new_name)

    def __iter__(self) -> Iterable[Column]:
        """Iterate over the columns of this view."""
        return iter(self.columns)

    # ── parsing ───────────────────────────────────────────────────────────────

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        view_name: str | None = None,
        service: "Views",
    ) -> "View":
        if isinstance(obj, cls):
            return obj
        return cls.parse_str(
            location=str(obj) if obj is not None else None,
            catalog_name=catalog_name,
            schema_name=schema_name,
            view_name=view_name,
            service=service,
        )

    @classmethod
    def parse_str(
        cls,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        view_name: str | None = None,
        service: "Views",
    ) -> "View":
        _, catalog_name, schema_name, view_name = service.parse_check_location_params(
            location=location,
            catalog_name=catalog_name or service.catalog_name,
            schema_name=schema_name or service.schema_name,
            view_name=view_name,
        )
        return View(
            service=service,
            catalog_name=catalog_name,
            schema_name=schema_name,
            view_name=view_name,
        )

    # ── navigation shorthands ─────────────────────────────────────────────────

    @property
    def sql(self) -> "SQLEngine":
        return self.client.sql(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )

    @property
    def catalog(self) -> "Catalog":
        from yggdrasil.databricks.catalog.catalog import Catalog as _Catalog
        from yggdrasil.databricks.catalog.catalogs import Catalogs
        return _Catalog(
            service=Catalogs(client=self.client),
            catalog_name=self.catalog_name,
        )

    @property
    def schema(self) -> "UCSchema":
        from yggdrasil.databricks.schema.schema import Schema as _Schema
        from yggdrasil.databricks.catalog.catalogs import Catalogs
        return _Schema(
            service=Catalogs(client=self.client),
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )

    # ── cache management ──────────────────────────────────────────────────────

    def _reset_cache(self, invalidate_cache: bool = False) -> None:
        if invalidate_cache:
            self.service.invalidate_cached_view(view=self)

        object.__setattr__(self, "_infos", None)
        object.__setattr__(self, "_infos_fetched_at", None)
        object.__setattr__(self, "_columns", None)

    def clear(self) -> "View":
        self._reset_cache()
        return self

    # ── infos / existence ─────────────────────────────────────────────────────

    def _cache_expired(self) -> bool:
        if self._infos is None:
            return True
        age = time.time() - (self._infos_fetched_at or 0.0)
        return age >= INFOS_TTL

    @property
    def infos(self) -> TableInfo:
        if self._cache_expired():
            logger.debug(
                "%r fetching remote infos",
                self,
            )
            infos = self.service.find_view_remote(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                view_name=self.view_name,
            )
            object.__setattr__(self, "_infos", infos)
            object.__setattr__(self, "_infos_fetched_at", time.time())
            object.__setattr__(self, "_columns", [
                Column.from_api(table=self, infos=col_info)
                for col_info in (infos.columns or [])
            ])
            logger.debug(
                "%r fetched remote infos: stored view=%s table_type=%s columns=%d",
                self, getattr(infos, "table_type", None),
                len(self._columns or ()),
            )
        else:
            age = time.time() - (self._infos_fetched_at or 0.0)
        return self._infos

    @property
    def exists(self) -> bool:
        try:
            _ = self.infos
            return True
        except NotFound:
            return False

    @property
    def view_id(self) -> str:
        return self.infos.table_id

    @property
    def view_definition(self) -> Optional[str]:
        """The SQL SELECT text that defines this view."""
        return self.infos.view_definition

    @property
    def view_dependencies(self):
        return self.infos.view_dependencies

    @property
    def table_type(self) -> Optional[TableType]:
        return self.infos.table_type

    @property
    def is_materialized(self) -> bool:
        return self.table_type == TableType.MATERIALIZED_VIEW

    @property
    def comment(self) -> Optional[str]:
        return self.infos.comment

    @property
    def owner(self) -> Optional[str]:
        return self.infos.owner

    # ── columns ───────────────────────────────────────────────────────────────

    @property
    def columns(self) -> list[Column]:
        if self._columns is None:
            _ = self.infos
        return self._columns or []

    def column(
        self,
        name: str,
        safe: bool = False,
        raise_error: bool = True,
    ) -> Optional[Column]:
        columns = self.columns
        for col in columns:
            if col.name == name:
                return col

        if not safe:
            case_folded = name.casefold()
            for col in columns:
                if col.name.casefold() == case_folded:
                    return col

        if raise_error:
            raise ValueError(f"Column {name!r} not found in {self!r}")
        return None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def create_ddl(
        self,
        query: str,
        *,
        or_replace: bool = False,
        if_not_exists: bool = False,
        columns: Iterable[str] | None = None,
        comment: str | None = None,
        properties: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Render a ``CREATE [OR REPLACE] VIEW [IF NOT EXISTS]`` DDL statement.

        Args:
            query:          The SELECT text that defines the view.
            or_replace:     Emit ``CREATE OR REPLACE VIEW``.
            if_not_exists:  Emit ``IF NOT EXISTS``.  Mutually exclusive with
                            ``or_replace``.
            columns:        Optional explicit column names to expose.
            comment:        Human-readable description.
            properties:     Extra TBLPROPERTIES key/value pairs.
        """
        if or_replace and if_not_exists:
            raise ValueError("Use either or_replace or if_not_exists, not both.")

        select_text = (query or "").strip().rstrip(";").strip()
        if not select_text:
            raise ValueError("View query (SELECT text) cannot be empty")

        if or_replace:
            create_kw = "CREATE OR REPLACE VIEW"
        elif if_not_exists:
            create_kw = "CREATE VIEW IF NOT EXISTS"
        else:
            create_kw = "CREATE VIEW"

        parts: list[str] = [f"{create_kw} {self.full_name(safe=True)}"]

        if columns:
            parts.append("(" + ", ".join(quote_ident(c) for c in columns) + ")")

        if comment:
            parts.append(f"COMMENT '{escape_sql_string(comment)}'")

        if properties:
            def _fmt(k: str, v: Any) -> str:
                if isinstance(v, bool):
                    return f"'{k}' = '{'true' if v else 'false'}'"
                if isinstance(v, str):
                    return f"'{k}' = '{escape_sql_string(v)}'"
                return f"'{k}' = {v}"

            parts.append(
                "TBLPROPERTIES ("
                + ", ".join(_fmt(k, v) for k, v in properties.items())
                + ")"
            )

        parts.append(f"AS {select_text}")
        return "\n".join(parts)

    def create(
        self,
        query: str,
        *,
        mode: ModeLike = None,
        or_replace: bool | None = None,
        if_not_exists: bool | None = None,
        columns: Iterable[str] | None = None,
        comment: str | None = None,
        properties: Optional[Mapping[str, Any]] = None,
        tags: Mapping[str, str] | None = None,
        wait_result: bool = True,
    ) -> "View":
        """Create (or replace) this view from a SELECT statement.

        When neither ``or_replace`` nor ``if_not_exists`` is provided, they are
        derived from ``mode``:

            * :data:`Mode.OVERWRITE` → ``or_replace=True``
            * :data:`Mode.AUTO` / :data:`Mode.APPEND` / :data:`Mode.UPSERT`
              / :data:`Mode.IGNORE` → ``if_not_exists=True``
            * :data:`Mode.ERROR_IF_EXISTS` → plain ``CREATE VIEW``
        """
        parsed_mode = Mode.from_(mode, default=Mode.AUTO)

        if or_replace is None and if_not_exists is None:
            if parsed_mode == Mode.OVERWRITE:
                or_replace = True
                if_not_exists = False
            elif parsed_mode == Mode.ERROR_IF_EXISTS:
                or_replace = False
                if_not_exists = False
            else:
                or_replace = False
                if_not_exists = True

        statement = self.create_ddl(
            query,
            or_replace=bool(or_replace),
            if_not_exists=bool(if_not_exists),
            columns=columns,
            comment=comment,
            properties=properties,
        )

        logger.debug(
            "View.create: view=%s or_replace=%s if_not_exists=%s mode=%s",
            self.full_name(), bool(or_replace), bool(if_not_exists), parsed_mode,
        )
        try:
            self.sql.execute(statement, wait=wait_result)
        except Exception as exc:
            if "SCHEMA_NOT_FOUND" in str(exc):
                logger.debug(
                    "View.create: parent schema missing for %s — auto-creating %s.%s",
                    self.full_name(), self.catalog_name, self.schema_name,
                )
                self.sql.execute(
                    f"CREATE SCHEMA IF NOT EXISTS "
                    f"{quote_ident(self.catalog_name)}.{quote_ident(self.schema_name)}",
                    wait=True,
                )
                self.sql.execute(statement, wait=wait_result)
            else:
                raise

        self._reset_cache(invalidate_cache=True)

        if tags:
            self.set_tags(tags)

        return self

    def ensure_created(
        self,
        query: str,
        *,
        comment: str | None = None,
        properties: Optional[Mapping[str, Any]] = None,
        columns: Iterable[str] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> "View":
        return self.create(
            query,
            mode=Mode.AUTO,
            comment=comment,
            properties=properties,
            columns=columns,
            tags=tags,
        )

    def delete_ddl(self, *, if_exists: bool = True) -> str:
        if_exists_clause = " IF EXISTS" if if_exists else ""
        keyword = "MATERIALIZED VIEW" if self._infos and self.is_materialized else "VIEW"
        return f"DROP {keyword}{if_exists_clause} {self.full_name(safe=True)}"

    def delete(
        self,
        *,
        if_exists: bool = True,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "View":
        statement = self.delete_ddl(if_exists=if_exists)
        logger.debug(
            "View.delete: view=%s if_exists=%s wait=%s",
            self.full_name(), if_exists, bool(wait),
        )
        if wait:
            try:
                self.sql.execute(statement, wait=True)
            except DatabricksError:
                if raise_error:
                    raise
        else:
            Job.make(self.sql.execute, statement).fire_and_forget()

        self._reset_cache(invalidate_cache=True)
        return self

    # ── rename ────────────────────────────────────────────────────────────────

    def rename(
        self,
        new_name: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        view_name: str | None = None,
    ) -> "View":
        """Rename this view in-place (``ALTER VIEW … RENAME TO …``).

        Accepts an unqualified name (``"new_orders_v"``), a two-part name
        (``"reporting.orders_v"`` → cross-schema move within the same catalog),
        or a three-part name (``"main.reporting.orders_v"``). Catalog/schema
        keyword overrides win over parts parsed from *new_name*.

        Unity Catalog allows cross-schema renames within the same catalog;
        moves across catalogs are rejected here with a clear error rather
        than letting the server return a generic failure.
        """
        if new_name is not None:
            parsed_c, parsed_s, parsed_v = self.service.parse_catalog_schema_view_names(new_name)
        else:
            parsed_c = parsed_s = parsed_v = None

        target_catalog = (catalog_name or parsed_c or self.catalog_name or "").strip().strip("`")
        target_schema = (schema_name or parsed_s or self.schema_name or "").strip().strip("`")
        target_view = (view_name or parsed_v or "").strip().strip("`")

        if not target_view:
            raise ValueError("Cannot rename view to an empty name")
        if not target_catalog or not target_schema:
            raise ValueError(
                f"Cannot rename {self.full_name()} — target needs a catalog and"
                f" schema (got catalog={target_catalog!r} schema={target_schema!r})"
            )
        if target_catalog != self.catalog_name:
            raise ValueError(
                f"Unity Catalog ALTER VIEW RENAME TO cannot move a view across"
                f" catalogs ({self.catalog_name!r} → {target_catalog!r}). Use"
                f" View.clone(...) to copy across catalogs instead."
            )
        if target_schema == self.schema_name and target_view == self.view_name:
            logger.debug(
                "View.rename: no-op — new name matches current %s",
                self.full_name(),
            )
            return self

        if target_schema == self.schema_name:
            rename_to = quote_ident(target_view)
        else:
            rename_to = f"{quote_ident(target_schema)}.{quote_ident(target_view)}"

        logger.debug(
            "View.rename: %s → %s.%s.%s",
            self.full_name(), target_catalog, target_schema, target_view,
        )
        self.sql.execute(
            f"ALTER VIEW {self.full_name(safe=True)} RENAME TO {rename_to}"
        )
        self._reset_cache(invalidate_cache=True)
        self.schema_name = target_schema
        self.view_name = target_view
        return self

    # ── clone ─────────────────────────────────────────────────────────────────

    def clone(
        self,
        target: "str | View | None" = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        view_name: str | None = None,
        query: str | None = None,
        replace: bool = False,
        if_not_exists: bool = False,
        comment: str | None = None,
        properties: Optional[Mapping[str, Any]] = None,
    ) -> "View":
        """Clone this view to *target* by re-emitting its SELECT definition.

        Delta CLONE is table-only, so the view-side analogue is a fresh
        ``CREATE [OR REPLACE] VIEW [IF NOT EXISTS] <target> AS <query>``.

        Args:
            target:        Target location — :class:`View`, a 1/2/3-part dotted
                           name, or ``None`` when *catalog_name* / *schema_name*
                           / *view_name* are passed explicitly.
            query:         Override SELECT text. Defaults to the source view's
                           own :attr:`view_definition`.
            replace:       Emit ``CREATE OR REPLACE VIEW``.
            if_not_exists: Emit ``CREATE VIEW IF NOT EXISTS``. Mutually
                           exclusive with *replace*.
            comment:       Optional ``COMMENT`` on the cloned view.
            properties:    Optional ``TBLPROPERTIES`` overrides.

        Returns:
            A :class:`View` bound to this service pointing at the target.
        """
        if replace and if_not_exists:
            raise ValueError("Use either replace=True or if_not_exists=True, not both.")

        views = self.service
        if isinstance(target, View):
            target_catalog = target.catalog_name
            target_schema = target.schema_name
            target_view = target.view_name
        else:
            parsed_c, parsed_s, parsed_v = (
                views.parse_catalog_schema_view_names(target) if target else (None, None, None)
            )
            target_catalog = catalog_name or parsed_c or self.catalog_name
            target_schema = schema_name or parsed_s or self.schema_name
            target_view = view_name or parsed_v

        if not (target_catalog and target_schema and target_view):
            raise ValueError(
                f"Cannot clone {self.full_name()} — target needs catalog +"
                f" schema + view (got catalog={target_catalog!r}"
                f" schema={target_schema!r} view={target_view!r})"
            )
        if (
            target_catalog == self.catalog_name
            and target_schema == self.schema_name
            and target_view == self.view_name
        ):
            raise ValueError(
                f"Cannot clone {self.full_name()} onto itself — choose a"
                f" different target catalog/schema/view."
            )

        select_text = (query or self.view_definition or "").strip().rstrip(";").strip()
        if not select_text:
            raise ValueError(
                f"Cannot clone {self.full_name()} — source has no view_definition"
                f" and no query override was provided."
            )

        cloned = View(
            service=views,
            catalog_name=target_catalog,
            schema_name=target_schema,
            view_name=target_view,
        )
        statement = cloned.create_ddl(
            select_text,
            or_replace=replace,
            if_not_exists=if_not_exists,
            comment=comment,
            properties=properties,
        )
        logger.debug(
            "View.clone: %s → %s.%s.%s replace=%s if_not_exists=%s",
            self.full_name(), target_catalog, target_schema, target_view,
            replace, if_not_exists,
        )
        self.sql.execute(statement)

        views.invalidate_cached_view(view=cloned)
        return cloned

    # ── tags ──────────────────────────────────────────────────────────────────

    def set_tags_ddl(
        self,
        tags: Mapping[str, str] | None,
        *,
        tag_collation: str | None = DEFAULT_TAG_COLLATION,
    ) -> str:
        pairs: list[str] = []
        for k, v in (tags or {}).items():
            key = _safe_str(k).strip() if k is not None else ""
            val = _safe_str(v).strip() if v is not None else ""
            if key and val:
                pairs.append(
                    f"{databricks_tag_literal(key, collation=tag_collation)} = "
                    f"{databricks_tag_literal(val, collation=tag_collation)}"
                )

        if not pairs:
            raise ValueError(f"Cannot set empty tags on {self!r}")

        return f"ALTER VIEW {self.full_name(safe=True)} SET TAGS ({', '.join(pairs)})"

    def set_tags(
        self,
        tags: Mapping[str, str] | None,
        *,
        tag_collation: str | None = DEFAULT_TAG_COLLATION,
    ) -> "View":
        if not tags:
            return self
        self.sql.execute(self.set_tags_ddl(tags, tag_collation=tag_collation))
        return self

    # ── concat_tables — UNION view across multiple tables/views ──────────────

    def concat_tables(
        self,
        tables: Iterable["Table | View"],
        *,
        by_name: bool = True,
        cast: bool = True,
        comment: str | None = None,
        mode: ModeLike = Mode.OVERWRITE,
    ) -> "View":
        """Create or replace this view as the ``UNION ALL`` of *tables*.

        When ``cast`` is ``True`` (default), the union is "smart": column
        names are aligned across inputs, types are promoted to the widest
        compatible :class:`DataType` via ``merge_with(upcast=True)``, each
        input projects the unified column list in order, and any column
        missing from a given input is emitted as ``CAST(NULL AS <ddl>)`` so
        the unified schema is preserved.  Present columns are selected
        as-is — Databricks reconciles compatible types across the union.

        When ``cast`` is ``False`` the method falls back to a plain
        ``SELECT * FROM <table> UNION ALL [BY NAME] ...`` — Databricks does
        the column matching at query time and any type mismatch surfaces as
        a SQL error.

        Args:
            tables:
                Iterable of :class:`Table` or :class:`View` instances to
                union.  At least one input is required.
            by_name:
                When ``True`` (default), align columns by name across
                inputs.  Ignored when ``cast`` is ``True`` because explicit
                projection always aligns by the unified column order.
            cast:
                Enable smart type promotion + explicit ``CAST`` projection.
                Set to ``False`` to emit a plain ``SELECT *`` union and let
                Databricks reconcile schemas.
            comment:
                Optional ``COMMENT`` for the view DDL.
            mode:
                Passed through to :meth:`create`.  Defaults to
                :attr:`Mode.OVERWRITE` so the view is created or replaced
                atomically.

        Returns:
            ``self`` after the view has been created or replaced.
        """
        tables_list = list(tables)
        if not tables_list:
            raise ValueError("concat_tables requires at least one table")

        if cast:
            query = self._build_smart_union_query(tables_list)
        else:
            separator = "\nUNION ALL BY NAME\n" if by_name else "\nUNION ALL\n"
            query = separator.join(
                f"SELECT * FROM {t.full_name(safe=True)}"
                for t in tables_list
            )

        return self.create(query, mode=mode, comment=comment)

    @staticmethod
    def _build_smart_union_query(
        tables_list: list["Table | View"],
    ) -> str:
        """Render a ``UNION ALL`` query projecting each input to a unified schema.

        Walks every input's ``columns``, accumulates a unified schema
        (first-seen column order, types promoted via ``merge_with(upcast=
        True)``), then projects each input to that column order —
        selecting present columns as-is and substituting
        ``CAST(NULL AS <ddl>)`` for absent ones.
        """
        from yggdrasil.data.enums.mode import Mode as _Mode

        column_order: list[str] = []
        unified: dict[str, "DataType"] = {}
        per_table: list[dict[str, "DataType"]] = []

        for tbl in tables_list:
            cols: dict[str, "DataType"] = {}
            for c in tbl.columns:
                cols[c.name] = c.field.dtype
                if c.name not in unified:
                    column_order.append(c.name)
                    unified[c.name] = c.field.dtype
                else:
                    unified[c.name] = unified[c.name].merge_with(
                        c.field.dtype, mode=_Mode.UPSERT, upcast=True,
                    )
            per_table.append(cols)

        if not column_order:
            raise ValueError(
                "concat_tables: input tables have no columns to union; "
                "ensure each input has been resolved against the catalog"
            )

        select_blocks: list[str] = []
        for tbl, cols in zip(tables_list, per_table):
            exprs: list[str] = []
            for name in column_order:
                qname = quote_ident(name)
                if name in cols:
                    exprs.append(qname)
                else:
                    ddl = unified[name].to_spark_name()
                    exprs.append(f"CAST(NULL AS {ddl}) AS {qname}")

            select_blocks.append(
                "SELECT\n  " + ",\n  ".join(exprs)
                + f"\nFROM {tbl.full_name(safe=True)}"
            )

        return "\nUNION ALL\n".join(select_blocks)

    # ── grants (GrantsMixin) ──────────────────────────────────────────────────

    def _grants_securable_type(self) -> SecurableType:
        return SecurableType.TABLE

    def _grants_full_name(self) -> str:
        return self.full_name()

    # ── GRANT DDL helpers ─────────────────────────────────────────────────────

    def grant_permissions_ddl(
        self,
        principal: str,
        privileges: str | Iterable[str],
    ) -> str:
        """Build a ``GRANT`` DDL statement for one principal on this view."""
        grant_privileges = self._normalize_view_privileges(privileges)
        return (
            f"GRANT {', '.join(grant_privileges)} "
            f"ON VIEW {self.full_name(safe=True)} "
            f"TO {quote_principal(principal)}"
        )

    @staticmethod
    def _normalize_view_privileges(
        privileges: str | Iterable[str] | None,
    ) -> tuple[str, ...]:
        if privileges is None:
            privileges = ("SELECT",)
        elif isinstance(privileges, str):
            privileges = (privileges,)

        normalized: list[str] = []
        for privilege in privileges:
            value = str(privilege).strip()
            if not value:
                continue
            value = " ".join(value.replace("_", " ").replace("-", " ").upper().split())
            if value not in normalized:
                normalized.append(value)

        if not normalized:
            raise ValueError("add_permissions requires at least one privilege")

        return tuple(normalized)
