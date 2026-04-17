"""
Per-view resource: DDL, schema introspection and grant helpers.

The :class:`View` dataclass wraps a single Unity Catalog view and exposes
instance-level methods only.  Collection operations (``find_view``,
``list_views``) live in :mod:`~yggdrasil.databricks.sql.views`.

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
from dataclasses import dataclass, field
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
from yggdrasil.io.enums.save_mode import SaveModeArg, SaveMode

from .column import Column
from .grants import GrantsMixin
from .sql_utils import (
    DEFAULT_TAG_COLLATION,
    _safe_str,
    databricks_tag_literal,
    escape_sql_string,
    quote_ident,
    quote_principal,
)

if TYPE_CHECKING:
    from .catalog import Catalog
    from .engine import SQLEngine
    from .schema import Schema as UCSchema
    from .views import Views

__all__ = ["View"]

logger = logging.getLogger(__name__)

INFOS_TTL: float = 300.0


@dataclass
class View(DatabricksResource, GrantsMixin):
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
        from .catalog import Catalog as _Catalog
        from .catalogs import Catalogs
        return _Catalog(
            service=Catalogs(client=self.client),
            catalog_name=self.catalog_name,
        )

    @property
    def schema(self) -> "UCSchema":
        from .schema import Schema as _Schema
        from .catalogs import Catalogs
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
            infos = self.service.find_view_remote(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                view_name=self.view_name,
            )
            self._reset_cache()
            object.__setattr__(self, "_infos", infos)
            object.__setattr__(self, "_infos_fetched_at", time.time())
            object.__setattr__(self, "_columns", [
                Column.from_api(table=self, infos=col_info)
                for col_info in (infos.columns or [])
            ])
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
        mode: SaveModeArg = None,
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

            * :data:`SaveMode.OVERWRITE` → ``or_replace=True``
            * :data:`SaveMode.AUTO` / :data:`SaveMode.APPEND` / :data:`SaveMode.UPSERT`
              / :data:`SaveMode.IGNORE` → ``if_not_exists=True``
            * :data:`SaveMode.ERROR_IF_EXISTS` → plain ``CREATE VIEW``
        """
        parsed_mode = SaveMode.parse(mode, SaveMode.AUTO)

        if or_replace is None and if_not_exists is None:
            if parsed_mode == SaveMode.OVERWRITE:
                or_replace = True
                if_not_exists = False
            elif parsed_mode == SaveMode.ERROR_IF_EXISTS:
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

        try:
            self.sql.execute(statement, wait=wait_result)
        except Exception as exc:
            if "SCHEMA_NOT_FOUND" in str(exc):
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
            mode=SaveMode.AUTO,
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
