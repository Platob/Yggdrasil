"""
Per-table resource: DDL, DML, schema introspection and storage helpers.

The :class:`Table` dataclass wraps a single Unity Catalog table and exposes
instance-level methods only.  Collection operations (``find_table``,
``list_tables``) live in :mod:`~yggdrasil.databricks.sql.tables`.

Caching strategy
----------------
Expensive Unity Catalog lookups (basic ``TableInfo``, entity-tag
assignments on the table and on each column) are cached on the instance
with a shared TTL and loaded lazily on first access.

    1. **Local** — return the cached value immediately when fresh.
    2. **Remote** — hit the Databricks API only on miss / expiry.
    3. **Update** — store the fetched value so the next access is free.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union, TYPE_CHECKING, Mapping, Iterable

import pyarrow as pa
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import (
    DataSourceFormat,
    Privilege,
    SecurableType,
    TableInfo,
    TableOperation,
    TableType,
)
from pyarrow.fs import FileSystem, S3FileSystem

from yggdrasil.concurrent.threading import Job
from yggdrasil.data import Field
from yggdrasil.data.schema import Schema as DataSchema
from yggdrasil.databricks.iam import IAMUser, IAMGroup
from yggdrasil.dataclasses.expiring import Expiring, RefreshResult
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io import URL
from yggdrasil.io.enums.save_mode import SaveModeArg, SaveMode
from .column import Column
from .grants import GrantsMixin
from .sql_utils import (
    DEFAULT_TAG_COLLATION,
    _qualify_fk_ref,
    _safe_constraint_name,
    _safe_str,
    databricks_tag_literal,
    quote_ident,
    quote_principal,
    quote_qualified_ident,
    sql_literal, escape_sql_string,
)
from .types import PrimaryKeySpec, ForeignKeySpec

if TYPE_CHECKING:
    import delta
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from yggdrasil.databricks.sql.engine import SQLEngine
    from yggdrasil.databricks.sql.tables import Tables
    from yggdrasil.databricks.sql.catalog import Catalog
    from yggdrasil.databricks.sql.columns import Columns
    from yggdrasil.databricks.sql.schema import Schema as UCSchema, Schema

__all__ = ["Table"]

logger = logging.getLogger(__name__)

_INVALID_COL_CHARS = set(" ,;{}()\n\t=")


def _needs_column_mapping(col_name: str) -> bool:
    return any(ch in _INVALID_COL_CHARS for ch in col_name)


GRANTS_METADATA_KEY: bytes = b"grants"


def parse_grants_principals(value: bytes | str | Iterable[Any] | None) -> list[str]:
    """Parse a ``b"grants"`` metadata value into a list of principal names.

    Accepts a bytes/str payload (JSON array of strings or comma-separated)
    or any iterable of principal-like values.  Whitespace is stripped and
    empty entries are skipped while preserving original order.
    """
    if value is None:
        return []

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed if str(p).strip()]
        return [p.strip() for p in text.split(",") if p.strip()]

    if isinstance(value, Iterable):
        return [str(p).strip() for p in value if str(p).strip()]

    raise TypeError(f"Unsupported grants metadata value: {value!r}")


INFOS_TTL: float = 300.0


# ===========================================================================
# Table — per-table resource
# ===========================================================================

@dataclass
class Table(GrantsMixin):
    """A single Unity Catalog table — DDL, DML, schema, storage helpers."""

    catalog_name: str = "default"
    schema_name: str = "default"
    table_name: str = "default"

    @property
    def name(self):
        return self.table_name

    _infos: Optional[TableInfo] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _infos_fetched_at: float | None = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _columns: Optional[list[Column]] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _tags: Optional[tuple[Any, ...]] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _tags_fetched_at: float | None = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _column_tags: Optional[dict[str, tuple[Any, ...]]] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _column_tags_fetched_at: float | None = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )

    @property
    def url(self) -> URL:
        return (
            self.client.base_url
            .with_path(f"/explore/data/{self.catalog_name}/{self.schema_name}/{self.table_name}")
        )

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        service: "Tables"
    ):
        if isinstance(obj, cls):
            return obj

        return cls.parse_str(
            location=str(obj) if obj is not None else None,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            service=service
        )

    @classmethod
    def parse_str(
        cls,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        service: "Tables",
    ):
        _, catalog_name, schema_name, table_name = service.parse_check_location_params(
            location=location,
            catalog_name=catalog_name or service.catalog_name,
            schema_name=schema_name or service.schema_name,
            table_name=table_name,
        )

        return Table(
            service=service,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name
        )

    # =========================================================================
    # Convenience shorthand — service delegates
    # =========================================================================

    @property
    def sql(self) -> "SQLEngine":
        return self.client.sql(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name
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

    # =========================================================================
    # Identity / repr
    # =========================================================================

    def schema_full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}"

    def full_name(self, safe: str | bool | None = None) -> str:
        if safe:
            q = safe if isinstance(safe, str) else "`"
            return (
                f"{q}{self.catalog_name}{q}"
                f".{q}{self.schema_name}{q}"
                f".{q}{self.table_name}{q}"
            )
        return f"{self.catalog_name}.{self.schema_name}.{self.table_name}"

    def __repr__(self) -> str:
        return f"Table({self.url.to_string()!r})"

    def __str__(self):
        return self.full_name(safe=True)

    def __getitem__(self, item: str) -> Column:
        return self.column(item)

    def __setitem__(self, item: str, new_name: str) -> None:
        """``table["old_col"] = "new_col"`` renames a column."""
        self.column(item).rename(new_name)

    def __iter__(self) -> Iterable[Column]:
        """Iterate over the columns of this table."""
        return iter(self.columns)

    # =========================================================================
    # Cache management
    # =========================================================================

    def _reset_cache(self, invalidate_cache: bool = False) -> None:
        if invalidate_cache:
            self.sql.tables.invalidate_cached_table(table=self)

        object.__setattr__(self, "_infos", None)
        object.__setattr__(self, "_infos_fetched_at", None)
        object.__setattr__(self, "_columns", None)
        object.__setattr__(self, "_tags", None)
        object.__setattr__(self, "_tags_fetched_at", None)
        object.__setattr__(self, "_column_tags", None)
        object.__setattr__(self, "_column_tags_fetched_at", None)

    def clear(self) -> "Table":
        self._reset_cache()
        return self

    # =========================================================================
    # Databricks SDK — lazy-loaded properties
    # =========================================================================

    @property
    def exists(self) -> bool:
        try:
            _ = self.infos
            return True
        except NotFound:
            return False

    @property
    def table_id(self) -> str:
        return self.infos.table_id

    @staticmethod
    def _is_fresh(fetched_at: float | None) -> bool:
        if fetched_at is None:
            return False
        return (time.time() - fetched_at) < INFOS_TTL

    def _store_infos(self, infos: TableInfo) -> TableInfo:
        """Populate the lightweight infos + columns caches."""
        object.__setattr__(self, "_infos", infos)
        object.__setattr__(self, "_infos_fetched_at", time.time())
        object.__setattr__(self, "_columns", [
            Column.from_api(table=self, infos=col_info)
            for col_info in (infos.columns or [])
        ])
        return infos

    @property
    def infos(self) -> TableInfo:
        """Basic :class:`TableInfo` — TTL-cached."""
        if self._infos is not None and self._is_fresh(self._infos_fetched_at):
            return self._infos

        info = self.client.tables.find_table_remote(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=self.table_name,
        )
        self._store_infos(info)
        return info

    # =========================================================================
    # Entity-tag assignments — lazy, TTL-cached
    # =========================================================================

    def _list_tag_assignments(
        self,
        entity_type: str,
        entity_name: str,
    ) -> tuple[Any, ...]:
        """Return entity-tag assignments for ``entity_name``.

        Missing SDK API, permission errors, or disabled workspace features
        are logged and produce an empty tuple; the caller can treat the
        absence of tags as "none known".
        """
        tags_api = getattr(self.client.workspace_client(), "entity_tag_assignments", None)
        if tags_api is None:
            return ()
        try:
            return tuple(tags_api.list(entity_type=entity_type, entity_name=entity_name))
        except Exception:
            logger.warning(
                "Failed to list %s tag assignments for %r",
                entity_type, entity_name, exc_info=True,
            )
            return ()

    @property
    def tags(self) -> tuple[Any, ...]:
        """Table-level entity-tag assignments — TTL-cached, lazy-loaded."""
        if self._tags is not None and self._is_fresh(self._tags_fetched_at):
            return self._tags

        assignments = self._list_tag_assignments("tables", self.full_name())
        object.__setattr__(self, "_tags", assignments)
        object.__setattr__(self, "_tags_fetched_at", time.time())
        return assignments

    @property
    def column_tags(self) -> Mapping[str, tuple[Any, ...]]:
        """Per-column entity-tag assignments — TTL-cached, lazy-loaded.

        Only columns that have at least one tag assignment appear in the
        returned mapping.
        """
        if self._column_tags is not None and self._is_fresh(self._column_tags_fetched_at):
            return self._column_tags

        full_name = self.full_name()
        assignments: dict[str, tuple[Any, ...]] = {}
        for col_info in (self.infos.columns or []):
            col_name = col_info.name
            if not col_name:
                continue
            col_assignments = self._list_tag_assignments(
                "columns", f"{full_name}.{col_name}",
            )
            if col_assignments:
                assignments[col_name] = col_assignments

        object.__setattr__(self, "_column_tags", assignments)
        object.__setattr__(self, "_column_tags_fetched_at", time.time())
        return assignments

    # =========================================================================
    # Arrow schema introspection
    # =========================================================================

    @property
    def columns(self) -> list[Column]:
        if self._columns is None:
            # Refresh the cache if needed to ensure we have the latest column infos.
            _ = self.infos
        return self._columns

    def column(
        self,
        name: str,
        safe: bool = False,
        raise_error: bool = True
    ) -> Column:
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

    @property
    def data_schema(self) -> DataSchema:
        return DataSchema.from_any_fields(
            [c.field for c in self.columns],
            metadata={
                b"name": self.table_name.encode(),
                b"engine": b"databricks",
                b"catalog_name": self.catalog_name.encode(),
                b"schema_name": self.schema_name.encode(),
                b"table_name": self.table_name.encode(),
            }
        )

    @property
    def data_field(self):
        return self.data_schema.to_field()

    @property
    def arrow_fields(self) -> list[pa.Field]:
        return [c.field.to_arrow_field() for c in self.columns]

    @property
    def arrow_schema(self) -> pa.Schema:
        return self.data_schema.to_arrow_schema()

    @property
    def arrow_field(self) -> pa.Field:
        return self.data_field.to_arrow_field()

    # =========================================================================
    # Constraint helpers — public ALTER TABLE DDL builders
    # =========================================================================

    def add_primary_key_ddl(
        self,
        columns: "list[str] | str",
        *,
        constraint_name: str | None = None,
        rely: bool = False,
        timeseries: str | None = None,
    ) -> str:
        if isinstance(columns, str):
            columns = [columns]

        col_exprs = [
            f"`{c}` TIMESERIES" if timeseries == c else f"`{c}`"
            for c in columns
        ]
        cname = _safe_constraint_name(
            constraint_name or f"{self.table_name}_{'_'.join(columns)}_pk"
        )
        rely_clause = " RELY" if rely else ""
        return (
            f"ALTER TABLE {self.full_name(safe=True)} "
            f"ADD CONSTRAINT `{cname}` "
            f"PRIMARY KEY ({', '.join(col_exprs)}) NOT ENFORCED"
            f"{rely_clause}"
        )

    def set_primary_key(
        self,
        columns: "list[str] | str",
        *,
        constraint_name: str | None = None,
        rely: bool = False,
        timeseries: str | None = None,
    ) -> "Table":
        """Create a PRIMARY KEY constraint via the Unity Catalog API.

        See https://docs.databricks.com/api/gcp/workspace/tableconstraints
        for the ``TableConstraintsAPI.create`` endpoint.
        """
        from .constraints_api import apply_primary_key

        if isinstance(columns, str):
            columns = [columns]

        apply_primary_key(
            self,
            PrimaryKeySpec(
                columns=list(columns),
                constraint_name=constraint_name,
                rely=rely,
                timeseries=timeseries,
            ),
        )
        self._reset_cache(invalidate_cache=True)
        return self

    def drop_primary_key_ddl(
        self,
        *,
        if_exists: bool = True,
        cascade: bool = False,
    ) -> str:
        if_exists_clause = " IF EXISTS" if if_exists else ""
        cascade_clause = " CASCADE" if cascade else ""
        return (
            f"ALTER TABLE {self.full_name(safe=True)} "
            f"DROP PRIMARY KEY{if_exists_clause}{cascade_clause}"
        )

    def drop_primary_key(
        self,
        *,
        if_exists: bool = True,
        cascade: bool = False,
    ) -> "Table":
        """Delete the table's PRIMARY KEY constraint via the UC API.

        The API deletes by constraint name, so we resolve the current PK
        constraint name from ``TableInfo.table_constraints`` and call
        ``TableConstraintsAPI.delete``.
        """
        from .constraints_api import delete_constraint

        name = self._primary_key_constraint_name()
        if name is None:
            if not if_exists:
                raise ValueError(
                    f"{self!r} has no PRIMARY KEY constraint to drop"
                )
            return self

        delete_constraint(
            self, name, cascade=cascade, if_exists=if_exists,
        )
        self._reset_cache(invalidate_cache=True)
        return self

    def _primary_key_constraint_name(self) -> str | None:
        """Return the current PK constraint name from ``TableInfo``, if any."""
        for constraint in (getattr(self.infos, "table_constraints", None) or ()):
            pk = getattr(constraint, "primary_key_constraint", None)
            if pk is not None and getattr(pk, "name", None):
                return pk.name
        return None

    def _foreign_key_constraint_name(self, column: str) -> str | None:
        """Return the current FK constraint name on *column*, if any."""
        for constraint in (getattr(self.infos, "table_constraints", None) or ()):
            fk = getattr(constraint, "foreign_key_constraint", None)
            if fk is None:
                continue
            child_columns = getattr(fk, "child_columns", None) or ()
            if column in child_columns:
                return getattr(fk, "name", None)
        return None

    def set_foreign_keys(
        self,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | Any | None" = None,
        *,
        schema: Any = None,
        raise_error: bool = False,
    ) -> "Table":
        """Apply foreign-key constraints via the UC ``table_constraints`` API.

        ``foreign_keys`` accepts anything :meth:`ForeignKeySpec.from_any` handles
        (a dict ``{column: ref}``, a list of specs, a single spec, a schema, …).
        When ``foreign_keys`` is omitted, the constraints are read from
        ``schema`` (each field's ``foreign_key`` tag).

        Partial refs such as ``"ref_table.col"`` are resolved against this
        table's catalog/schema. Failures are logged and skipped unless
        ``raise_error`` is set.
        """
        from .constraints_api import apply_foreign_key

        fk_specs = ForeignKeySpec.from_any(foreign_keys, schema=schema)

        for fk in fk_specs:
            ref = _qualify_fk_ref(
                fk.ref,
                default_catalog=self.catalog_name,
                default_schema=self.schema_name,
            )
            fk_with_ref = ForeignKeySpec(
                column=fk.column,
                ref=ref,
                constraint_name=fk.constraint_name,
                rely=fk.rely,
                match_full=fk.match_full,
                on_update_no_action=fk.on_update_no_action,
                on_delete_no_action=fk.on_delete_no_action,
            )
            try:
                apply_foreign_key(self, fk_with_ref)
                logger.debug(
                    "Applied FOREIGN KEY %r → %r on %s",
                    fk.column, ref, self.full_name(),
                )
            except Exception:
                if raise_error:
                    raise
                logger.warning(
                    "Failed to apply FOREIGN KEY %r → %r on %s",
                    fk.column, ref, self.full_name(),
                    exc_info=True,
                )

        return self

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

        return f"ALTER TABLE {self.full_name(safe=True)} SET TAGS ({', '.join(pairs)})"

    def set_tags(
        self,
        tags: Mapping[str, str] | None,
        *,
        tag_collation: str | None = DEFAULT_TAG_COLLATION,
    ) -> "Table":
        if not tags:
            return self
        self.sql.execute(self.set_tags_ddl(tags, tag_collation=tag_collation))
        return self

    # =========================================================================
    # Constraint helpers — inline CREATE TABLE constraint rendering
    # =========================================================================

    # =========================================================================
    # Lifecycle — create / ensure / delete
    # =========================================================================

    def ensure_created(
        self,
        definition: Union[pa.Schema, Any, None],
        *,
        mode: SaveMode | str | None = None,
        **options
    ) -> "Table":
        return self.create(
            definition=definition,
            mode=mode,
            **options,
        )

    def _columns_service(self) -> "Columns":
        """Columns service scoped to this table's catalog/schema/table defaults."""
        from .columns import Columns

        return Columns(
            client=self.client,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=self.table_name,
        )

    def _resolve_fk_field(
        self,
        data_field: Field,
    ) -> tuple[Field, "ForeignKeySpec | None"]:
        """Detect the ``catalog.schema.table.column`` FK pattern on ``data_field.name``.

        When ``data_field.name`` contains ``.``, it is parsed through the
        :class:`Columns` service (scoped to this table's defaults) as a
        foreign-key reference:

        =============================================  =======================================
        Input name                                     Resolved as
        =============================================  =======================================
        ``"ref_table.col"``                            ``<this catalog>.<this schema>.ref_table.col``
        ``"ref_schema.ref_table.col"``                 ``<this catalog>.ref_schema.ref_table.col``
        ``"catalog.schema.ref_table.col"``             fully qualified
        =============================================  =======================================

        The referenced column is looked up via the Columns service so the
        local column inherits its dtype (keeping the FK type-compatible).
        The returned field is renamed to the leaf column name and the
        returned :class:`ForeignKeySpec` carries the fully-qualified ref
        plus a stable, FK-specific constraint name.

        Safe fall-back: if the pattern can't be fully resolved, or the
        reference points at this same table, the field is simply renamed
        to the leaf name (dotted names are never valid column names) and
        ``None`` is returned for the FK spec.
        """
        name = data_field.name or ""
        if "." not in name:
            return data_field, None

        columns_service = self._columns_service()
        cat, sch, tbl, col = columns_service.parse_location(name)

        if not (cat and sch and tbl and col):
            return data_field, None

        local_field = data_field.with_name(col, inplace=False)

        if (
            cat == self.catalog_name
            and sch == self.schema_name
            and tbl == self.table_name
        ):
            return local_field, None

        try:
            ref_column = columns_service.column(
                catalog_name=cat,
                schema_name=sch,
                table_name=tbl,
                column_name=col,
            )
        except Exception:
            logger.warning(
                "Foreign key reference %r could not be resolved; skipping constraint.",
                name, exc_info=True,
            )
            return local_field, None

        local_field = local_field.with_dtype(ref_column.field.dtype, inplace=False)

        fk_spec = ForeignKeySpec(
            column=col,
            ref=f"{cat}.{sch}.{tbl}.{col}",
            constraint_name=_safe_constraint_name(
                self.table_name, col, tbl, col, "fk",
            ),
        )
        return local_field, fk_spec

    def update_column(
        self,
        column: Field,
        *,
        mode: SaveMode | str | None = None,
    ):
        return self.update_columns(
            [column],
            mode=mode,
        )

    def update_columns(
        self,
        columns: Iterable[Field],
        *,
        mode: SaveMode | str | None = None,
    ):
        """Evolve this table's schema to match *columns*.

        Matching is case-insensitive: an input field whose name differs only
        in case from an existing column is treated as the same column and
        renamed to the supplied casing.

        Mode semantics
        --------------
        ``UPSERT`` (and ``AUTO``)
            Rename case-insensitive matches, add new columns, update the data
            type of existing columns whose dtype no longer matches the input.
        ``OVERWRITE``
            Everything ``UPSERT`` does, plus drop columns that do not appear
            in *columns* so the resulting schema matches the input exactly.
        Any other mode
            Only adds new columns and renames case-insensitive matches, so
            the call remains non-destructive.
        """
        mode = SaveMode.parse(mode, SaveMode.AUTO)
        alter_table = f"ALTER TABLE {self.full_name(safe=True)}"
        update_dtype = mode in (SaveMode.UPSERT, SaveMode.OVERWRITE)
        drop_missing = mode == SaveMode.OVERWRITE

        rename_statements: list[str] = []
        type_statements: list[str] = []
        add_columns: list[str] = []
        pending_fks: list[ForeignKeySpec] = []
        matched_existing: set[str] = set()

        for column in columns:
            data_field = Field.from_any(column)
            data_field, fk = self._resolve_fk_field(data_field)

            existing = self.column(name=data_field.name, safe=False, raise_error=False)

            if existing is None:
                add_columns.append(
                    f"`{data_field.name}` {data_field.dtype.to_databricks_ddl()}"
                )
                if fk is not None:
                    pending_fks.append(fk)
                continue

            matched_existing.add(existing.name)
            current_name = existing.name

            if existing.name != data_field.name:
                rename_statements.append(
                    f"{alter_table} RENAME COLUMN `{existing.name}` "
                    f"TO `{data_field.name}`"
                )
                current_name = data_field.name

            if update_dtype:
                existing_ddl = existing.field.dtype.to_databricks_ddl()
                new_ddl = data_field.dtype.to_databricks_ddl()
                if existing_ddl != new_ddl:
                    type_statements.append(
                        f"{alter_table} ALTER COLUMN `{current_name}` "
                        f"TYPE {new_ddl}"
                    )

        drop_names: list[str] = []
        if drop_missing:
            drop_names = [
                col.name for col in self.columns
                if col.name not in matched_existing
            ]

        add_col_statement: str | None = None
        if add_columns:
            add_col_statement = (
                f"{alter_table} ADD COLUMNS ({', '.join(add_columns)})"
            )

        # Renames must complete before we reference the new names in ALTER
        # COLUMN TYPE; split those into an earlier phase when both exist.
        needs_phase_split = bool(rename_statements and type_statements)

        first_phase: list[str] = []
        second_phase: list[str] = []

        if needs_phase_split:
            first_phase.extend(rename_statements)
        else:
            second_phase.extend(rename_statements)

        second_phase.extend(type_statements)
        if drop_names:
            second_phase.append(
                f"{alter_table} DROP COLUMNS "
                + "(" + ", ".join(f"`{n}`" for n in drop_names) + ")"
            )
        if add_col_statement is not None:
            second_phase.append(add_col_statement)

        executed = False
        if first_phase:
            self.sql.execute_many(first_phase, parallel=True)
            executed = True
        if second_phase:
            self.sql.execute_many(second_phase, parallel=True)
            executed = True

        if pending_fks:
            # FKs are applied through the UC ``table_constraints`` API after
            # any ADD COLUMN has materialized the referencing column.
            self._apply_constraints(pk_spec=None, fk_specs=pending_fks)
            executed = True

        if executed:
            self._reset_cache(invalidate_cache=True)

        return self

    def create(
        self,
        definition: Union[pa.Schema, Any],
        *,
        mode: SaveMode | str | None = None,
        partition_by: Optional[list[str]] = None,
        storage_location: str | None = None,
        comment: str | None = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        if_not_exists: bool = True,
        primary_keys: "list[str] | str | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
    ) -> "Table":
        """Create the table, routing to SQL DDL or the Unity Catalog REST API.

        SQL path:
            - inlines PK/FK constraints directly in CREATE TABLE

        API path:
            - creates the table first
            - then applies PK/FK via ALTER TABLE fallback
        """
        mode = SaveMode.parse(mode, SaveMode.AUTO)
        schema = DataSchema.from_(definition)

        if self.exists:
            if mode == SaveMode.ERROR_IF_EXISTS:
                raise ValueError(f"Table {self!r} already exists")
            elif mode == SaveMode.IGNORE:
                return self
            return self.update_columns(
                schema.fields,
                mode=mode,
            )

        if table_type is None:
            table_type = TableType.EXTERNAL if storage_location else TableType.MANAGED

        if table_type == TableType.MANAGED:
            result = self.sql_create(
                definition,
                comment=comment,
                partition_by=partition_by,
                if_not_exists=if_not_exists,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
            )
        else:
            if table_type == TableType.EXTERNAL and not storage_location:
                storage_location = (
                    self.schema_storage_location(table_type=table_type)
                    + "/tables/%s" % self.table_name
                )
            result = self.api_create(
                definition=definition,
                storage_location=storage_location,
                comment=comment,
                properties=properties,
                table_type=table_type,
                data_source_format=data_source_format,
                if_not_exists=if_not_exists,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
            )

        self._apply_grants_from_metadata(schema.metadata)
        return result

    def _apply_grants_from_metadata(
        self,
        metadata: Mapping[bytes | str, Any] | None,
    ) -> None:
        """Grant the principals listed under ``b"grants"`` access to this table.

        Each principal receives the default privilege chain required to read
        the table: ``USE_CATALOG`` on the catalog, ``USE_SCHEMA`` on the
        schema, and ``SELECT`` on the table itself.  Failures are logged and
        do not abort table creation.
        """
        if not metadata:
            return

        raw = metadata.get(GRANTS_METADATA_KEY) or metadata.get(
            GRANTS_METADATA_KEY.decode("utf-8")
        )
        principals = parse_grants_principals(raw)
        if not principals:
            return

        catalog = self.catalog
        schema = self.schema

        for principal in principals:
            try:
                catalog.grant(principal, [Privilege.USE_CATALOG])
                schema.grant(principal, [Privilege.USE_SCHEMA])
                self.grant(principal, [Privilege.SELECT])
            except Exception:
                logger.warning(
                    "Failed to apply default grants for principal %r on %s",
                    principal, self.full_name(),
                    exc_info=True,
                )

    def sql_create(
        self,
        description: Union[pa.Field, pa.Schema, Any],
        *,
        storage_location: str | None = None,
        partition_by: Optional[list[str]] = None,
        cluster_by: Optional[list[str]] = None,
        comment: str | None = None,
        properties: Optional[dict[str, Any]] = None,
        if_not_exists: bool = True,
        or_replace: bool = False,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        optimize_write: bool = True,
        auto_compact: bool = True,
        enable_cdf: bool | None = None,
        enable_deletion_vectors: bool | None = None,
        target_file_size: int | None = None,
        column_mapping_mode: str | None = None,
        wait_result: bool = True,
        auto_tag: bool = True,
        primary_keys: "list[str] | str | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
    ) -> "Table":
        """Generate and execute a CREATE TABLE DDL statement.

        PK columns are forced to NOT NULL in the column definition to match
        Databricks requirements for primary keys. The PK/FK constraints
        themselves are applied after the table exists, via the Unity Catalog
        ``TableConstraintsAPI`` — see
        https://docs.databricks.com/api/gcp/workspace/tableconstraints.
        """
        schema_info = DataSchema.from_any(description).autotag()
        partition_by = partition_by or schema_info.partition_by
        cluster_by = cluster_by or schema_info.cluster_by
        primary_keys = primary_keys or schema_info.primary_key_names
        comment = comment or schema_info.comment

        pk_spec = PrimaryKeySpec.from_any(primary_keys, schema=schema_info)
        fk_specs = list(ForeignKeySpec.from_any(foreign_keys, schema=schema_info))

        effective_fields: list[Field] = []
        column_definitions: list[str] = []
        for f in schema_info.children_fields:
            f, inferred_fk = self._resolve_fk_field(f)
            if inferred_fk is not None:
                fk_specs.append(inferred_fk)
            if f.primary_key and f.nullable:
                f = f.with_nullable(False)
            effective_fields.append(f)
            column_definitions.append(f.to_databricks_ddl())

        any_invalid = any(_needs_column_mapping(f.name) for f in effective_fields)
        if column_mapping_mode is None:
            column_mapping_mode = "name" if any_invalid else "none"

        table_definitions = column_definitions

        if or_replace and if_not_exists:
            raise ValueError("Use either or_replace or if_not_exists, not both.")

        if or_replace:
            create_kw = "CREATE OR REPLACE TABLE"
        elif if_not_exists:
            create_kw = "CREATE TABLE IF NOT EXISTS"
        else:
            create_kw = "CREATE TABLE"

        sql_parts: list[str] = [
            f"{create_kw} {self.full_name(safe=True)} (",
            "  " + ",\n  ".join(table_definitions),
            ")",
            f"USING {data_source_format.value}",
        ]

        if partition_by:
            sql_parts.append(
                "PARTITIONED BY (" + ", ".join(quote_ident(c.name) for c in partition_by) + ")"
            )
        elif cluster_by:
            sql_parts.append(
                "CLUSTER BY (" + ", ".join(quote_ident(c.name) for c in cluster_by) + ")"
            )
        else:
            sql_parts.append("CLUSTER BY AUTO")

        if comment:
            sql_parts.append(f"COMMENT '{escape_sql_string(comment)}'")

        if storage_location:
            sql_parts.append(f"LOCATION '{escape_sql_string(storage_location)}'")

        props: dict[str, Any] = {
            "delta.autoOptimize.optimizeWrite": bool(optimize_write),
            "delta.autoOptimize.autoCompact": bool(auto_compact),
        }
        if enable_cdf is not None:
            props["delta.enableChangeDataFeed"] = bool(enable_cdf)
        if enable_deletion_vectors is not None:
            props["delta.enableDeletionVectors"] = bool(enable_deletion_vectors)
        if target_file_size is not None:
            props["delta.targetFileSize"] = int(target_file_size)
        if column_mapping_mode != "none":
            props["delta.columnMapping.mode"] = column_mapping_mode
            props["delta.minReaderVersion"] = 2
            props["delta.minWriterVersion"] = 5
        if properties:
            props.update(properties)

        if props:
            def _fmt(k: str, v: Any) -> str:
                if isinstance(v, str):
                    return f"'{k}' = '{escape_sql_string(v)}'"
                if isinstance(v, bool):
                    return f"'{k}' = '{'true' if v else 'false'}'"
                return f"'{k}' = {v}"

            sql_parts.append(
                "TBLPROPERTIES (" + ", ".join(_fmt(k, v) for k, v in props.items()) + ")"
            )

        statement = "\n".join(sql_parts)

        try:
            self.sql.execute(statement, wait=wait_result)
        except Exception as exc:
            if "SCHEMA_NOT_FOUND" in str(exc):
                self.sql.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {quote_ident(self.catalog_name)}.{quote_ident(self.schema_name)}",
                    wait=True,
                )
                self.sql.execute(statement, wait=wait_result)
            else:
                raise

        self._reset_cache(invalidate_cache=True)

        # Apply PK/FK via the UC table_constraints API — CREATE TABLE no
        # longer carries inline constraint DDL.
        self._apply_constraints(pk_spec, fk_specs)

        if schema_info.tags:
            self.set_tags(schema_info.tags)

        for f in effective_fields:
            if f.tags:
                self.column(f.name).set_tags(f.tags)

        return self

    def api_create(
        self,
        definition: Union[pa.Schema, Any],
        *,
        storage_location: str | None = None,
        comment: str | None = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        if_not_exists: bool = False,
        primary_keys: "list[str] | str | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
    ) -> "Table":
        # TODO: support for table_type=TableType.EXTERNAL
        raise NotImplementedError("API path not implemented for SQL tables.")

    def _apply_constraints(
        self,
        pk_spec: "PrimaryKeySpec | None",
        fk_specs: "list[ForeignKeySpec]",
    ) -> None:
        """Apply PK then FK constraints via the UC ``table_constraints`` API.

        Failures are logged and do not abort the successful table create.
        """
        from .constraints_api import apply_foreign_key, apply_primary_key

        if pk_spec and pk_spec.columns:
            try:
                apply_primary_key(self, pk_spec)
                logger.debug(
                    "Applied PRIMARY KEY %r on %s",
                    pk_spec.columns, self.full_name(),
                )
            except Exception:
                logger.warning(
                    "Failed to apply PRIMARY KEY %r on %s — "
                    "the table was created; add the constraint manually.",
                    pk_spec.columns, self.full_name(),
                    exc_info=True,
                )

        for fk in fk_specs:
            try:
                apply_foreign_key(self, fk)
                logger.debug(
                    "Applied FOREIGN KEY %r → %r on %s",
                    fk.column, fk.ref, self.full_name(),
                )
            except Exception:
                logger.warning(
                    "Failed to apply FOREIGN KEY %r → %r on %s — "
                    "add the constraint manually.",
                    fk.column, fk.ref, self.full_name(),
                    exc_info=True,
                )

    def delete(
        self,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Table":
        uc = self.client.workspace_client().tables

        if wait:
            try:
                uc.delete(full_name=self.full_name())
            except DatabricksError:
                if raise_error:
                    raise
        else:
            Job.make(uc.delete, self.full_name()).fire_and_forget()

        self._reset_cache(invalidate_cache=True)
        return self

    # =========================================================================
    # Rename
    # =========================================================================

    def rename(self, new_name: str) -> "Table":
        """Rename this table in-place (``ALTER TABLE … RENAME TO …``).

        The catalog/schema parent is unchanged; *new_name* is the unqualified name.
        """
        new_name = (new_name or "").strip().strip("`")
        if not new_name:
            raise ValueError("Cannot rename table to an empty name")
        if new_name == self.table_name:
            return self

        self.sql.execute(
            f"ALTER TABLE {self.full_name(safe=True)} "
            f"RENAME TO {quote_ident(new_name)}"
        )
        self._reset_cache(invalidate_cache=True)
        self.table_name = new_name
        return self

    # =========================================================================
    # Spark / Delta integration
    # =========================================================================

    def delta_spark(
        self,
        spark_session: "SparkSession | None" = None,
    ) -> "delta.tables.DeltaTable":  # noqa
        from delta.tables import DeltaTable  # noqa

        session = spark_session or PyEnv.spark_session(
            create=True, import_error=True, install_spark=False,
        )
        return DeltaTable.forName(sparkSession=session, tableOrViewName=self.full_name(safe=True))

    def to_spark(
        self,
        spark_session: Optional["SparkSession"] = None,
    ) -> "SparkDataFrame":
        return self.delta_spark(spark_session=spark_session).toDF()

    # =========================================================================
    # Data I/O
    # =========================================================================

    def to_arrow_dataset(
        self,
        *,
        filters: Optional[list[tuple[str, str, str]]] = None,
        wait: WaitingConfigArg = True,
        cache_for: WaitingConfigArg = None,
    ):
        statement = f"SELECT * FROM {self.full_name(safe=True)}"
        if filters:
            predicates = [_build_predicate(c, o, v) for c, o, v in filters]
            statement += " WHERE " + " AND ".join(predicates)

        result = self.sql.execute(
            statement,
            wait=wait,
            cache_for=cache_for,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )
        return result.to_arrow_dataset()

    def insert(
        self,
        data: Any,
        *,
        mode: SaveModeArg = None,
        match_by: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        spark_session: Optional["SparkSession"] = None,
    ) -> None:
        return self.sql.insert_into(
            data,
            mode=mode,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=self.table_name,
            table=self,
            match_by=match_by,
            wait=wait,
            raise_error=raise_error,
            spark_session=spark_session,
        )

    # =========================================================================
    # Storage & credentials
    # =========================================================================

    def schema_storage_location(
        self,
        table_type: Optional[TableType] = None,
    ) -> str:
        infos = self.client.workspace_client().schemas.get(
            full_name=self.schema_full_name()
        )
        if not infos.storage_location:
            raise NotImplementedError

        if table_type == TableType.EXTERNAL and "/__unitystorage" in infos.storage_location:
            root = infos.storage_location.split("/__unitystorage")[0]
            return root + "/catalogs/%s/schemas/%s" % (
                self.catalog_name or "default",
                self.schema_name or "default",
            )

        return infos.storage_location

    @property
    def storage_location(self) -> str:
        return self.infos.storage_location

    def credentials(self, operation: TableOperation = TableOperation.READ):
        return (
            self.client.workspace_client()
            .temporary_table_credentials
            .generate_temporary_table_credentials(
                table_id=self.table_id,
                operation=operation,
            )
        )

    def arrow_filesystem(
        self,
        *,
        operation: TableOperation = TableOperation.READ_WRITE,
        expiring: bool = False,
    ) -> Union[FileSystem, "TableFilesystem"]:
        created_at = time.time_ns()
        creds = self.credentials(operation=operation)
        assert creds.aws_temp_credentials, "Cannot get AWS credentials"
        aws = creds.aws_temp_credentials

        base = S3FileSystem(
            access_key=aws.access_key_id,
            secret_key=aws.secret_access_key,
            session_token=aws.session_token,
            region="eu-west-1",
        )

        if not expiring:
            return base

        ttl_ns = 3_600_000_000_000
        return TableFilesystem.create(
            value=base,
            created_at=created_at,
            ttl=ttl_ns,
            expires_at=created_at + ttl_ns,
            table=self,
            operation=operation,
        )

    # =========================================================================
    # Grants — Unity Catalog REST API helpers
    # =========================================================================

    def _grants_securable_type(self) -> SecurableType:
        return SecurableType.TABLE

    def _grants_full_name(self) -> str:
        return self.full_name()

    # =========================================================================
    # Permissions — SQL DDL helpers
    # =========================================================================

    def add_permissions(
        self,
        iam_id: str | list[str] | None = None,
        *,
        users: Iterable[IAMUser | str] | None = None,
        groups: Iterable[IAMGroup | str] | None = None,
        group: Iterable[IAMGroup | str] | None = None,
        privileges: str | Iterable[str] | None = None,
    ) -> "Table":
        principals = self._normalize_permission_principals(
            iam_id=iam_id,
            users=users,
            groups=groups,
            group=group,
        )
        grant_privileges = self._normalize_table_privileges(privileges)

        if not principals:
            raise ValueError("add_permissions requires at least one user, group, or iam_id")

        for principal in principals:
            self.sql.execute(self.grant_permissions_ddl(principal, grant_privileges))

        return self

    @staticmethod
    def _normalize_table_privileges(
        privileges: str | Iterable[str] | None,
    ) -> tuple[str, ...]:
        """Normalize table privilege names to Databricks SQL GRANT syntax."""
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

    @staticmethod
    def _principal_from_user(user: IAMUser | str) -> str:
        if isinstance(user, str):
            principal = user.strip()
        else:
            principal = user.username or user.email or user.name or user.id or ""

        if not principal:
            raise ValueError("User principal must have a username, email, name, or id")

        return principal

    @staticmethod
    def _principal_from_group(group: IAMGroup | str) -> str:
        if isinstance(group, str):
            principal = group.strip()
        else:
            principal = group.name or group.id or ""

        if not principal:
            raise ValueError("Group principal must have a name or id")

        return principal

    def _normalize_permission_principals(
        self,
        *,
        iam_id: str | list[str] | None,
        users: Iterable[IAMUser | str] | None,
        groups: Iterable[IAMGroup | str] | None,
        group: Iterable[IAMGroup | str] | None,
    ) -> tuple[str, ...]:
        """Collect principals from supported user/group inputs preserving order."""
        seen: set[str] = set()
        out: list[str] = []

        def add(v: str) -> None:
            principal = v.strip()
            if principal and principal not in seen:
                seen.add(principal)
                out.append(principal)

        if isinstance(iam_id, str):
            add(iam_id)
        elif iam_id:
            for value in iam_id:
                add(str(value))

        for user in users or ():
            add(self._principal_from_user(user))

        for entry in groups or ():
            add(self._principal_from_group(entry))

        for entry in group or ():
            add(self._principal_from_group(entry))

        return tuple(out)

    def grant_permissions_ddl(
        self,
        principal: str,
        privileges: str | Iterable[str],
    ) -> str:
        """Build a ``GRANT`` DDL statement for one principal on this table."""
        grant_privileges = self._normalize_table_privileges(privileges)
        return (
            f"GRANT {', '.join(grant_privileges)} "
            f"ON TABLE {self.full_name(safe=True)} "
            f"TO {quote_principal(principal)}"
        )


# ===========================================================================
# TableFilesystem — auto-refreshing S3 filesystem credentials
# ===========================================================================

@dataclass
class TableFilesystem(Expiring[FileSystem]):
    """Expiring wrapper around S3FileSystem that auto-refreshes UC credentials."""

    table: Optional[Table] = field(default=None)
    operation: TableOperation = TableOperation.READ

    def _refresh(self) -> RefreshResult[FileSystem]:
        value = self.table.arrow_filesystem(operation=self.operation, expiring=False)
        created_ns = time.time_ns()
        ttl_ns = 3_600_000_000_000
        return RefreshResult(
            value=value,
            created_at_ns=created_ns,
            ttl_ns=ttl_ns,
            expires_at_ns=created_ns + ttl_ns,
        )


# ===========================================================================
# SQL filter helpers  (used by to_arrow_dataset)
# ===========================================================================

def _build_predicate(col: str, op: str, val: Any) -> str:
    """Build a single SQL WHERE predicate from a ``(col, op, val)`` tuple."""
    _ALLOWED_OPS = {
        "=", "!=", "<>", ">", ">=", "<", "<=",
        "LIKE", "NOT LIKE", "IS", "IS NOT", "IN", "NOT IN",
    }
    op_norm = op.strip().upper()
    if op_norm == "==":
        op_norm = "="
    elif op_norm not in _ALLOWED_OPS:
        raise ValueError(f"Unsupported filter operator: {op!r}")

    col_sql = quote_qualified_ident(col)

    if val is None:
        return f"{col_sql} IS NULL"

    if op_norm in ("IS", "IS NOT"):
        return f"{col_sql} {op_norm} {sql_literal(val)}"

    if op_norm in ("IN", "NOT IN"):
        raw = str(val).strip()
        if raw.lower().startswith("sql:"):
            return f"{col_sql} {op_norm} {sql_literal(raw)}"
        inner = raw.strip("()")
        items = [x.strip() for x in inner.split(",") if x.strip()]
        if not items:
            return "FALSE" if op_norm == "IN" else "TRUE"
        return f"{col_sql} {op_norm} ({', '.join(sql_literal(x) for x in items)})"

    return f"{col_sql} {op_norm} {sql_literal(val)}"
