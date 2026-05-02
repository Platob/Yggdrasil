"""
Per-table resource: DDL, DML, schema introspection and storage helpers.

The :class:`Table` dataclass wraps a single Unity Catalog table and exposes
instance-level methods only.  Collection operations (``find_table``,
``list_tables``) live in :mod:`~yggdrasil.databricks.sql.tables`.

Caching strategy
----------------
``TableInfo`` (and the derived columns list) is cached on the instance
with a shared TTL and loaded lazily on first access.

Entity-tag assignments — both table-level and per-column — are *not*
cached on the instance.  They route through
:attr:`DatabricksClient.entity_tags`, whose module-level
:class:`ExpiringDict` is host-scoped and authoritative; surgical patches
on write keep the cache fresh without fan-out invalidation.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, TYPE_CHECKING, Mapping, Iterable, Iterator, Literal

import pyarrow as pa
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import (
    ColumnInfo,
    ColumnTypeName,
    DataSourceFormat,
    TableInfo,
    TableOperation,
    TableType, EntityTagAssignment,
)
from pyarrow.fs import FileSystem, S3FileSystem

from yggdrasil.concurrent.threading import Job
from yggdrasil.data import Field
from yggdrasil.data.expr import Expr, Predicate
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema as DataSchema
from yggdrasil.data.statement import PreparedStatement, StatementResult
from yggdrasil.databricks.client import DatabricksResource
from yggdrasil.dataclasses.expiring import Expiring, RefreshResult
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io import URL
from yggdrasil.io.buffer.primitive import ParquetIO
from yggdrasil.io.enums import MimeTypes, MimeType
from yggdrasil.io.enums.mode import ModeLike, Mode
from yggdrasil.io.buffer.base import TabularIO
from .column import Column
from .sql_utils import (
    quote_ident,
    quote_qualified_ident,
    sql_literal, escape_sql_string,
)
from .table_options import TableOptions
from ...lazy_imports import aws_config_class

if TYPE_CHECKING:
    import delta
    import pandas
    import polars
    import pyspark
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from yggdrasil.databricks.sql.engine import SQLEngine
    from yggdrasil.databricks.sql.tables import Tables
    from yggdrasil.databricks.sql.catalog import Catalog
    from yggdrasil.databricks.sql.columns import Columns
    from yggdrasil.databricks.sql.schema import Schema as UCSchema
    from yggdrasil.aws.client import AWSClient
    from yggdrasil.databricks.warehouse import WarehousePreparedStatement

__all__ = ["Table", "TableOptions"]

logger = logging.getLogger(__name__)

_INVALID_COL_CHARS = set(" ,;{}()\n\t=")


def _needs_column_mapping(col_name: str) -> bool:
    return any(ch in _INVALID_COL_CHARS for ch in col_name)


INFOS_TTL: float = 300.0
_ALIAS_TMPSRC = "__tmpsrc__"


# ---------------------------------------------------------------------------
# DML statement helpers — shared by every insert path.
#
# All three insert paths (arrow / spark / sql) feed through the same DML
# generator below.  They differ only in how the *source* is prepared:
#
#   arrow → stage Parquet to a UC Volume; reference it via ``{__tmpsrc__}``
#   spark → register a temp view; reference it by quoted name
#   sql   → wrap caller's query + CAST projection
#
# Save modes:
#   - ``append``    insert-only; with ``match_by`` only non-matching rows
#   - ``overwrite`` drop, then insert
#   - ``truncate``  in-place wipe + insert; with ``match_by`` targeted DELETE
#   - ``auto``      default; with ``match_by`` upsert
#
# Merge ``ON`` is built null-safe (``<=>``) so NULL matches NULL.
#
# Retry semantics: caller-supplied ``retry`` (a ``WaitingConfig`` arg) is
# applied only to DML statements (INSERT/MERGE/DELETE/UPDATE).
# TRUNCATE/OPTIMIZE/VACUUM stay non-retryable on purpose: re-running
# TRUNCATE after a successful INSERT is dangerous, and
# OPTIMIZE/VACUUM are best-effort maintenance.
# ---------------------------------------------------------------------------


_DML_HEAD_RE = re.compile(
    r"\A(?:\s+|--[^\n]*\n|--[^\n]*\Z|/\*.*?\*/)*"
    r"(?P<kw>[A-Za-z]+)",
    re.DOTALL,
)
_DML_KEYWORDS: frozenset[str] = frozenset({"INSERT", "MERGE", "DELETE", "UPDATE"})


def _classify_dml(sql: str) -> bool:
    """True when ``sql`` looks like an INSERT/MERGE/DELETE/UPDATE."""
    if not sql:
        return False
    m = _DML_HEAD_RE.match(sql)
    if not m:
        return False
    return m.group("kw").upper() in _DML_KEYWORDS


def _apply_retry_to_warehouse_statement(
    stmt: "WarehousePreparedStatement",
    retry: Optional[WaitingConfigArg],
) -> None:
    """Install ``retry`` on a warehouse statement, in place."""
    if retry is None:
        return
    if retry is False:
        stmt.retry = None
        return
    stmt.retry = WaitingConfig.from_(retry)


def _build_match_condition(
    match_by: list[str],
    *,
    left_alias: str,
    right_alias: str,
    null_safe: bool = True,
    extra_predicates: Optional[Iterable[str]] = None,
) -> str:
    """Build a merge ``ON`` expression from key columns and optional extras."""
    op = "<=>" if null_safe else "="
    clauses = [
        f"{left_alias}.{quote_ident(k)} {op} {right_alias}.{quote_ident(k)}"
        for k in match_by
    ]
    if extra_predicates:
        clauses.extend(p for p in extra_predicates if p)
    return " AND ".join(clauses)


def _build_prune_predicates(
    prune_values: Mapping[str, Iterable[Any]],
    *,
    target_alias: str,
) -> list[str]:
    """Convert ``{column: [values]}`` into target-side ``IN`` predicates."""
    predicates: list[str] = []
    for col, vals in prune_values.items():
        materialized = tuple(vals)
        if not materialized:
            continue
        pred = Expr(col, flavor="databricks", alias=target_alias).in_(materialized)
        sql = pred.to_sql()
        if pred.kind != "leaf":
            sql = f"({sql})"
        predicates.append(sql)
    return predicates


def _wrap_user_predicate(pred: Predicate, *, target_alias: str) -> str:
    """Render a user predicate aliased to ``target_alias`` (parens if compound)."""
    aliased = pred.with_table_alias(target_alias)
    sql = aliased.to_sql()
    if aliased.kind != "leaf":
        sql = f"({sql})"
    return sql


def _collect_prune_values_polars(
    buffer: ParquetIO,
    prune_by: list[str],
) -> dict[str, tuple[Any, ...]]:
    df = buffer.scan_polars().select(*prune_by).unique().collect()
    return {col: tuple(df.get_column(col).to_list()) for col in prune_by}


def _collect_prune_values_spark(
    data_df: Any,
    prune_by: list[str],
) -> dict[str, tuple[Any, ...]]:
    rows = data_df.select(*prune_by).distinct().collect()
    return {col: tuple(row[col] for row in rows) for col in prune_by}


def _resolve_prune_by(
    prune_by: list[str] | str | None,
    fallback_partition_fields: Iterable[Any],
) -> Optional[list[str]]:
    if prune_by == "auto":
        return [f.name for f in fallback_partition_fields] or None
    if prune_by:
        return list(prune_by)
    return None


def _build_dml_statements(
    *,
    target_location: str,
    source_sql: str,
    columns: list[str],
    mode: Mode,
    match_by: Optional[list[str]],
    update_cols: Optional[list[str]],
    prune_predicates: list[str],
    zorder_by: Optional[list[str]] = None,
    optimize_after_merge: bool = False,
    vacuum_hours: Optional[int] = None,
) -> list[str]:
    """Generate INSERT / MERGE / DELETE / OPTIMIZE / VACUUM SQL.

    Path-agnostic — feeds from any source-SQL fragment that names the
    rows to merge.  ``prune_predicates`` are pre-rendered & parenthesized;
    they get AND-stitched onto ``ON`` clauses but NOT plain INSERT.
    """
    cols_quoted = ", ".join(quote_ident(c) for c in columns)
    statements: list[str] = []

    if mode in (Mode.TRUNCATE, Mode.OVERWRITE):
        if mode == Mode.TRUNCATE and match_by:
            key_cols = ", ".join(quote_ident(k) for k in match_by)
            on_condition = _build_match_condition(
                match_by, left_alias="T", right_alias="S",
                null_safe=True, extra_predicates=prune_predicates,
            )
            statements.extend([
                (
                    f"DELETE FROM {target_location} AS T\n"
                    f"USING (\n"
                    f"  SELECT DISTINCT {key_cols} FROM ({source_sql}) AS src\n"
                    f") AS S\n"
                    f"ON {on_condition}"
                ),
                f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}",
            ])
        elif mode == Mode.TRUNCATE:
            statements.extend([
                f"TRUNCATE TABLE {target_location}",
                f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}",
            ])
        else:
            statements.append(
                f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}"
            )

    elif match_by:
        on_condition = _build_match_condition(
            match_by, left_alias="T", right_alias="S",
            null_safe=True, extra_predicates=prune_predicates,
        )
        insert_clause = (
            f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) "
            f"VALUES ({', '.join(f'S.{quote_ident(c)}' for c in columns)})"
        )

        if mode == Mode.APPEND:
            statements.append(
                f"MERGE INTO {target_location} AS T\n"
                f"USING (\n{source_sql}\n) AS S\n"
                f"ON {on_condition}\n"
                f"{insert_clause}"
            )
        else:
            update_cols_effective = (
                update_cols
                if update_cols is not None
                else [c for c in columns if c not in match_by]
            )
            update_clause = ""
            if update_cols_effective:
                update_set = ", ".join(
                    f"T.{quote_ident(c)} = S.{quote_ident(c)}"
                    for c in update_cols_effective
                )
                update_clause = f"WHEN MATCHED THEN UPDATE SET {update_set}\n"

            statements.append(
                f"MERGE INTO {target_location} AS T\n"
                f"USING (\n{source_sql}\n) AS S\n"
                f"ON {on_condition}\n"
                f"{update_clause}"
                f"{insert_clause}"
            )
    else:
        statements.append(
            f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}"
        )

    if zorder_by:
        zorder_cols = ", ".join(quote_ident(c) for c in zorder_by)
        statements.append(f"OPTIMIZE {target_location} ZORDER BY ({zorder_cols})")
    if optimize_after_merge and match_by:
        statements.append(f"OPTIMIZE {target_location}")
    if vacuum_hours is not None:
        statements.append(f"VACUUM {target_location} RETAIN {int(vacuum_hours)} HOURS")

    return statements


def _delta_conf_for(
    overwrite_schema: bool | None,
    spark_options: Optional[Dict[str, Any]],
) -> dict[str, str]:
    """Translate caller-facing knobs into Spark session conf keys."""
    out: dict[str, str] = {}
    if overwrite_schema or (spark_options and spark_options.get("overwriteSchema")):
        out["spark.databricks.delta.schema.autoMerge.enabled"] = "true"
    return out


# Mapping from yggdrasil-recognised type-id keywords to UC SDK
# ``ColumnTypeName`` enum entries.  Used by :meth:`Table.api_create` to
# build ``ColumnInfo`` objects from a ``DataSchema``.
_DDL_TO_COLUMN_TYPE_NAME: dict[str, ColumnTypeName] = {
    "BIGINT": ColumnTypeName.LONG,
    "LONG": ColumnTypeName.LONG,
    "INT": ColumnTypeName.INT,
    "INTEGER": ColumnTypeName.INT,
    "SMALLINT": ColumnTypeName.SHORT,
    "SHORT": ColumnTypeName.SHORT,
    "TINYINT": ColumnTypeName.BYTE,
    "BYTE": ColumnTypeName.BYTE,
    "FLOAT": ColumnTypeName.FLOAT,
    "DOUBLE": ColumnTypeName.DOUBLE,
    "DECIMAL": ColumnTypeName.DECIMAL,
    "BOOLEAN": ColumnTypeName.BOOLEAN,
    "BOOL": ColumnTypeName.BOOLEAN,
    "STRING": ColumnTypeName.STRING,
    "BINARY": ColumnTypeName.BINARY,
    "DATE": ColumnTypeName.DATE,
    "TIMESTAMP": ColumnTypeName.TIMESTAMP,
    "TIMESTAMP_NTZ": ColumnTypeName.TIMESTAMP_NTZ,
    "INTERVAL": ColumnTypeName.INTERVAL,
    "ARRAY": ColumnTypeName.ARRAY,
    "MAP": ColumnTypeName.MAP,
    "STRUCT": ColumnTypeName.STRUCT,
    "VARIANT": ColumnTypeName.VARIANT,
    "NULL": ColumnTypeName.NULL,
}


def _column_type_name_from_ddl(ddl: str) -> ColumnTypeName:
    """Pick the UC ``ColumnTypeName`` enum for a Databricks DDL fragment."""
    head = ddl.strip().split("(", 1)[0].split("<", 1)[0].strip().upper()
    return _DDL_TO_COLUMN_TYPE_NAME.get(head, ColumnTypeName.STRING)


# ===========================================================================
# Table — per-table resource
# ===========================================================================

class Table(DatabricksResource, TabularIO[TableOptions]):
    """A single Unity Catalog table — DDL, DML, schema, storage helpers."""

    @classmethod
    def options_class(cls) -> type[TableOptions]:
        return TableOptions

    def __init__(
        self,
        service: "Tables | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        *,
        infos: TableInfo | None = None,
        infos_fetched_at: float | None = None,
        columns: list[Column] | None = None,
    ):
        if service is None:
            from .tables import Tables
            service = Tables.current()

        super().__init__(service=service)
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.table_name = table_name
        self._infos = infos
        self._infos_fetched_at = infos_fetched_at
        self._columns = columns

    def __getstate__(self):
        state = super().__getstate__()
        state["catalog_name"] = self.catalog_name
        state["schema_name"] = self.schema_name
        state["table_name"] = self.table_name
        state["_infos"] = self._infos
        state["_infos_fetched_at"] = self._infos_fetched_at
        state["_columns"] = self._columns

        return state

    def __setstate__(self, state):
        object.__setattr__(self, "catalog_name", state["catalog_name"])
        object.__setattr__(self, "schema_name", state["schema_name"])
        object.__setattr__(self, "table_name", state["table_name"])
        object.__setattr__(self, "_infos", state["_infos"])
        object.__setattr__(self, "_infos_fetched_at", state["_infos_fetched_at"])
        object.__setattr__(self, "_columns", state["_columns"])
        super().__setstate__(state)

    # ------------------------------------
    # TabularIO
    # ------------------------------------

    @classmethod
    def default_mime_type(cls) -> MimeType:
        return MimeTypes.DATABRICKS_UNITY_CATALOG_TABLE

    @property
    def cached(self) -> bool:
        return True

    def unpersist(self) -> None:
        pass

    def persist(self, engine: Literal["arrow", "polars", "spark", "auto"] = "auto", *,
                data: Any | None = None) -> "TabularIO":
        return self

    def _read_arrow_batches(self, options: TableOptions) -> Iterator[pa.RecordBatch]:
        options = options.with_source(source=self.collect_schema())
        safe_char = "`"
        names = ",".join(
            safe_char + name + safe_char
            for name in options.column_names or [c.name for c in self.columns]
        )
        query = f"SELECT {names}"
        if options.where:
            query += f" WHERE {options.where.with_flavor("databricks")}"

        for batch in self.execute(query).read_arrow_batches(options=options):
            yield batch

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: TableOptions
    ) -> None:
        options = options.with_target(self.collect_schema(options))

        return self.insert(
            batches,
            mode=options.mode,
            match_by=options.match_by_names,
            update_cols=options.update_cols,
            wait=options.wait,
            zorder_by=options.zorder_by,
            optimize_after_merge=options.optimize_after_merge,
            vacuum_hours=options.vacuum_hours,
            where=options.where,
            prune_by=options.prune_by,
            prune_values=options.prune_values,
            retry=options.retry,
        )
    
    # Properties
    
    @property
    def name(self):
        return self.table_name

    @property
    def url(self) -> URL:
        return (
            self.client.base_url
            .with_path(f"/explore/data/{self.catalog_name}/{self.schema_name}/{self.table_name}")
        )

    @classmethod
    def from_(
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
    
    def execute(
        self,
        statement: str | PreparedStatement,
        *args,
        **kwargs
    ):
        return self.sql.execute(
            statement=statement,
            *args,
            **kwargs
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

    def column_full_name(self, column_name: str) -> str:
        """Fully-qualified column name suitable for ``entity_tag_assignments``."""
        return f"{self.full_name()}.{column_name}"

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
            # Also drop entity-tag entries for this table and its columns —
            # a structural change (rename / drop / recreate) means the
            # ``entity_name`` keys themselves are stale.
            self._invalidate_entity_tag_cache()

        object.__setattr__(self, "_infos", None)
        object.__setattr__(self, "_infos_fetched_at", None)
        object.__setattr__(self, "_columns", None)

    def _invalidate_entity_tag_cache(self) -> None:
        """Drop cached tag lists for this table and every cached column."""
        tags = self.client.entity_tags
        tags.invalidate_cached_tags("tables", self.full_name())
        # Use the still-cached columns list (if any) — refusing to refetch
        # ``infos`` here keeps invalidation cheap and safe inside teardown.
        for col in (self._columns or ()):
            tags.invalidate_cached_tags("columns", self.column_full_name(col.name))

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
        """Populate the ``infos`` + ``columns`` caches."""
        self._infos_fetched_at = time.time()
        self._infos = infos
        self._columns = [
            Column.from_api(table=self, infos=col_info)
            for col_info in (infos.columns or [])
        ]
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
    # Entity-tag assignments — delegated to client.entity_tags
    # =========================================================================

    @property
    def tags(self) -> tuple[EntityTagAssignment, ...]:
        """Table-level entity-tag assignments — served from ``client.entity_tags``."""
        return tuple(
            self.client.entity_tags.entity_tags(
                "tables", self.full_name(), default=()
            ) or ()
        )

    @property
    def column_tags(self) -> Mapping[str, tuple[EntityTagAssignment, ...]]:
        """Per-column entity-tag assignments.

        Fan-out is parallelised so wide tables pay one aggregate wall-clock
        round trip rather than N sequential ones; cache hits inside
        ``client.entity_tags`` short-circuit each leg.
        """
        tags = self.client.entity_tags
        full = self.full_name()
        jobs: dict[str, Any] = {}
        for col_info in (self.infos.columns or []):
            col_name = col_info.name
            if not col_name:
                continue
            jobs[col_name] = Job.make(
                tags.entity_tags,
                "columns",
                f"{full}.{col_name}",
                default=(),
            ).fire_and_forget()

        result: dict[str, tuple[Any, ...]] = {}
        for col_name, job in jobs.items():
            assignments = tuple(job.wait() or ())
            if assignments:
                result[col_name] = assignments
        return result

    # =========================================================================
    # Arrow schema introspection
    # =========================================================================

    @property
    def columns(self) -> list[Column]:
        if self._columns is None:
            _ = self.infos  # populates _columns as a side effect
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

    def _collect_schema(self, options: CastOptions) -> DataSchema:
        """Return the field schema, optionally enriched with UC metadata."""
        metadata: dict[bytes, bytes] = {
            b"name": self.table_name.encode(),
            b"engine": b"databricks",
            b"catalog_name": self.catalog_name.encode(),
            b"schema_name": self.schema_name.encode(),
            b"table_name": self.table_name.encode(),
        }

        col_tags = self.column_tags

        fields: list[Field] = []
        for column in self.columns:
            base = column.field
            extra_tags: dict[bytes, bytes] = {}

            for assignment in col_tags.get(column.name, ()):
                key = getattr(assignment, "tag_key", None)
                if not key:
                    continue
                value = getattr(assignment, "tag_value", None) or ""
                extra_tags[key.encode("utf-8")] = str(value).encode("utf-8")

            if extra_tags:
                fields.append(
                    base.copy(
                        metadata=dict(base.metadata or {}),
                        tags=extra_tags,
                    )
                )
            else:
                fields.append(base)

        for assignment in self.tags:
            key = getattr(assignment, "tag_key", None)
            if not key:
                continue
            value = getattr(assignment, "tag_value", None) or ""
            metadata[f"tag:{key}".encode("utf-8")] = str(value).encode("utf-8")

        return DataSchema.from_any_fields(fields, metadata=metadata)

    def collect_data_field(self, safe: bool = False) -> Field:
        return self.collect_schema(safe=safe).to_field()

    @property
    def arrow_fields(self) -> list[pa.Field]:
        return [c.field.to_arrow_field() for c in self.columns]

    @property
    def arrow_schema(self) -> pa.Schema:
        return self.collect_schema().to_arrow_schema()

    @property
    def arrow_field(self) -> pa.Field:
        return self.collect_data_field().to_arrow_field()

    def set_tags(
        self,
        tags: Mapping[str, str] | None,
    ) -> "Table":
        """Apply table-level tags via the UC ``entity_tag_assignments`` API.

        ``tag_collation`` is accepted for API compatibility and ignored —
        collations only matter for the legacy DDL literal form.
        """
        if not tags:
            return self

        self.client.entity_tags.update_entity_tags(
            tags=tags,
            entity_type="tables",
            entity_name=self.full_name(),
        )
        return self

    def unset_tags(
        self,
        tag_keys: Iterable[str],
        *,
        if_exists: bool = True,
    ) -> "Table":
        """Delete table-level tag assignments by key."""
        self.client.entity_tags.delete_entity_tags(
            entity_type="tables",
            entity_name=self.full_name(),
            tag_keys=tag_keys,
            if_exists=if_exists,
        )
        return self

    # =========================================================================
    # Lifecycle — create / ensure / delete
    # =========================================================================

    def ensure_created(
        self,
        definition: Union[pa.Schema, Any, None],
        *,
        mode: Mode | str | None = None,
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

    def with_column(
        self,
        column: Field,
        *,
        mode: Mode | str | None = None,
    ):
        return self.with_columns([column], mode=mode)

    def with_columns(
        self,
        columns: Iterable[Field],
        *,
        mode: Mode | str | None = None,
    ):
        mode = Mode.from_(mode, default=Mode.AUTO)
        alter_table = f"ALTER TABLE {self.full_name(safe=True)}"
        update_dtype = mode in (Mode.UPSERT, Mode.OVERWRITE)
        drop_missing = mode == Mode.OVERWRITE

        rename_statements: list[str] = []
        type_statements: list[str] = []
        add_columns: list[str] = []
        matched_existing: set[str] = set()

        for column in columns:
            data_field = Field.from_any(column)
            existing = self.column(name=data_field.name, safe=False, raise_error=False)

            if existing is None:
                add_columns.append(
                    f"`{data_field.name}` {data_field.dtype.to_databricks_ddl()}"
                )
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

        if executed:
            self._reset_cache(invalidate_cache=True)

        return self

    def create(
        self,
        definition: Union[pa.Schema, Any],
        *,
        mode: Mode | str | None = None,
        partition_by: Optional[list[str]] = None,
        storage_location: str | None = None,
        comment: str | None = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        if_not_exists: bool = True,
    ) -> "Table":
        mode = Mode.from_(mode, default=Mode.AUTO)
        schema = DataSchema.from_(definition)

        if self.exists:
            if mode == Mode.ERROR_IF_EXISTS:
                raise ValueError(f"Table {self!r} already exists")
            elif mode == Mode.IGNORE:
                return self
            return self.with_columns(schema.fields, mode=mode)

        if table_type is None:
            table_type = TableType.EXTERNAL if storage_location else TableType.MANAGED

        if table_type == TableType.MANAGED:
            result = self.sql_create(
                definition,
                comment=comment,
                if_not_exists=if_not_exists,
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
            )

        return result

    def sql_create(
        self,
        description: DataSchema,
        *,
        storage_location: str | None = None,
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
    ) -> "Table":
        schema_info = DataSchema.from_any(description).autotag()
        comment = comment or schema_info.comment
        effective_fields: list[Field] = []
        column_definitions: list[str] = []
        partition_by = schema_info.partition_fields
        cluster_by = schema_info.cluster_fields
        primary_keys = schema_info.primary_keys

        for f in schema_info.children_fields:
            if f.primary_key and f.nullable:
                f = f.with_nullable(False, inplace=True)

            effective_fields.append(f)
            column_definitions.append(f.to_databricks_ddl())

        any_invalid = any(_needs_column_mapping(f.name) for f in effective_fields)
        if column_mapping_mode is None:
            column_mapping_mode = "name" if any_invalid else "none"

        # Inline-PK constraint: a single named PRIMARY KEY clause covering
        # every primary-key field.  Delta requires PK columns to be NOT
        # NULL — already enforced by the with_nullable(False) loop above.
        # FK / CHECK constraints can't be expressed inline against an
        # arbitrary parent table (the SDK ``table_constraints`` API does
        # the cross-table reference); they're applied post-create below.
        constraint_clauses = self._build_inline_constraints(primary_keys)

        table_definitions = column_definitions + constraint_clauses

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

        # Apply remaining constraints (FK / CHECK) via the SDK post-create.
        # Inline PK was already emitted in DDL — skip it here.
        self._apply_post_create_constraints(schema_info)

        if schema_info.tags:
            self.set_tags(schema_info.tags)

        for f in effective_fields:
            if f.tags:
                self.column(f.name).set_tags(f.tags)

        return self

    @staticmethod
    def _build_inline_constraints(primary_keys: Iterable[Field]) -> list[str]:
        """Render inline DDL ``CONSTRAINT … PRIMARY KEY(…)`` clauses.

        FK / CHECK aren't emitted inline: FK needs a parent reference that
        only the constraint :class:`Field` (or the SDK call) carries, and
        CHECK predicates aren't part of this layer.  Those go through
        :meth:`_apply_post_create_constraints`.
        """
        from yggdrasil.data.data_utils import safe_constraint_name

        pk_fields = [f for f in primary_keys if f and f.primary_key]
        if not pk_fields:
            return []

        col_names = [f.name for f in pk_fields]
        constraint_name = safe_constraint_name(col_names, prefix="pk")
        cols = ", ".join(quote_ident(n) for n in col_names)
        return [f"CONSTRAINT {quote_ident(constraint_name)} PRIMARY KEY ({cols})"]

    def _apply_post_create_constraints(self, schema_info: DataSchema) -> None:
        """Push FK / CHECK constraint Fields through the SDK constraints API.

        Inline-PK constraints already landed in the CREATE TABLE DDL — the
        primary-key fields on ``schema_info`` are intentionally skipped
        here to avoid a duplicate-name collision.
        """
        constraint_fields = [
            f for f in (schema_info.constraints or [])
            if f.foreign_key or (f.constraint_key and not f.primary_key)
        ]
        if not constraint_fields:
            return

        try:
            from yggdrasil.databricks.constraints.service import TableConstraints
        except ImportError:
            logger.debug(
                "yggdrasil.databricks.constraints not available; "
                "skipping post-create constraints on %s", self.full_name(),
            )
            return

        constraints_service = TableConstraints(client=self.client)
        for cf in constraint_fields:
            try:
                constraints_service.create_constraint(self, cf)
            except Exception:
                logger.warning(
                    "Failed to create constraint %r on %s",
                    cf.name, self.full_name(), exc_info=True,
                )

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
    ) -> "Table":
        """Create the table via the Unity Catalog ``tables.create`` REST API.

        Targets EXTERNAL tables — the SDK ``tables.create`` endpoint
        requires an explicit ``storage_location``.  For MANAGED tables,
        prefer :meth:`sql_create`, which is also the only path that
        exposes Delta-specific knobs (``CLUSTER BY``, ``OPTIMIZE``,
        ``TBLPROPERTIES``, column mapping mode, …).

        ``comment`` and constraints (PK / FK / CHECK) carried by the
        schema are applied post-create — the SDK call itself only takes
        columns + storage + properties — so the behaviour ends up
        symmetric with :meth:`sql_create`.
        """
        if if_not_exists and self.exists:
            return self

        schema_info = DataSchema.from_any(definition).autotag()
        comment = comment or schema_info.comment

        effective_fields: list[Field] = []
        column_infos: list[ColumnInfo] = []
        for position, f in enumerate(schema_info.children_fields):
            if f.constraint_key:
                continue
            if f.primary_key and f.nullable:
                f = f.with_nullable(False, inplace=True)
            effective_fields.append(f)
            column_infos.append(self._field_to_column_info(f, position=position))

        if table_type is None:
            table_type = TableType.EXTERNAL if storage_location else TableType.MANAGED

        if not storage_location:
            raise ValueError(
                "api_create requires an explicit storage_location — the UC "
                "tables.create endpoint won't materialise a MANAGED table for you. "
                "Use sql_create for managed Delta tables."
            )

        merged_properties: dict[str, str] = {str(k): str(v) for k, v in (properties or {}).items()}

        try:
            self.client.workspace_client().tables.create(
                name=self.table_name,
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_type=table_type,
                data_source_format=data_source_format,
                storage_location=storage_location,
                columns=column_infos,
                properties=merged_properties or None,
            )
        except DatabricksError as exc:
            if if_not_exists and "already exists" in str(exc).lower():
                self._reset_cache(invalidate_cache=True)
                return self
            raise

        self._reset_cache(invalidate_cache=True)

        # The SDK endpoint doesn't accept a comment — set it via ALTER
        # TABLE so the behaviour matches sql_create (which embeds COMMENT
        # in the CREATE DDL).
        if comment:
            self.sql.execute(
                f"ALTER TABLE {self.full_name(safe=True)} "
                f"SET TBLPROPERTIES ('comment' = '{escape_sql_string(comment)}')",
                wait=True,
            )

        self._apply_post_create_constraints(schema_info)

        if schema_info.tags:
            self.set_tags(schema_info.tags)

        for f in effective_fields:
            if f.tags:
                col = self.column(f.name, raise_error=False)
                if col is not None:
                    col.set_tags(f.tags)

        return self

    @staticmethod
    def _field_to_column_info(f: Field, *, position: int) -> ColumnInfo:
        """Translate a :class:`Field` into a UC SDK :class:`ColumnInfo`."""
        ddl = f.dtype.to_databricks_ddl()
        type_name = _column_type_name_from_ddl(ddl)
        comment_bytes = (f.metadata or {}).get(b"comment") if f.metadata else None
        comment = comment_bytes.decode("utf-8") if isinstance(comment_bytes, bytes) else None
        return ColumnInfo(
            name=f.name,
            type_text=ddl,
            type_name=type_name,
            type_json=None,
            nullable=bool(f.nullable),
            position=position,
            comment=comment,
            partition_index=position if f.partition_by else None,
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

    # =========================================================================
    # Data I/O
    # =========================================================================

    def to_arrow_dataset(
        self,
        *,
        filters: Optional[list[tuple[str, str, str]]] = None,
        wait: WaitingConfigArg = True,
    ):
        statement = f"SELECT * FROM {self.full_name(safe=True)}"
        if filters:
            predicates = [_build_predicate(c, o, v) for c, o, v in filters]
            statement += " WHERE " + " AND ".join(predicates)

        result = self.sql.execute(
            statement,
            wait=wait,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )
        return result.to_arrow_dataset()

    def insert(
        self,
        data: Any,
        *,
        mode: ModeLike = None,
        match_by: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        spark_session: Optional["SparkSession"] = None,
        **kwargs
    ) -> None:
        """Insert *data* into this table — thin wrapper over :meth:`insert_into`."""
        return self.insert_into(
            data,
            mode=mode,
            match_by=match_by,
            wait=wait,
            raise_error=raise_error,
            spark_session=spark_session,
            **kwargs,
        )

    # =========================================================================
    # insert_into — top-level dispatcher (arrow / spark / sql paths)
    # =========================================================================

    def insert_into(
        self,
        data: Union[
            pa.Table, pa.RecordBatch, pa.RecordBatchReader,
            dict, list, str,
            PreparedStatement, StatementResult,
            "pandas.DataFrame", "polars.DataFrame", "pyspark.sql.DataFrame",
        ],
        *,
        mode: Mode | str | None = None,
        schema_mode: Mode | str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_cols: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_session: Optional["pyspark.sql.SparkSession"] = None,
        spark_options: Optional[Dict[str, Any]] = None,
        where: Predicate | None = None,
        prune_by: list[str] | str | None = None,
        prune_values: dict[str, tuple[Any]] | None = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> None:
        """Insert *data* into this table using the most appropriate backend.

        Routing:

        - Query-shaped sources (str, ``PreparedStatement``,
          ``StatementResult``) → :meth:`sql_insert`
        - Spark DataFrame (or anything when a ``SparkSession`` is reachable)
          → :meth:`spark_insert`
        - Otherwise → :meth:`arrow_insert` (warehouse path with Volume staging)
        """
        common = dict(
            mode=mode,
            match_by=match_by,
            update_cols=update_cols,
            wait=wait,
            raise_error=raise_error,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            where=where,
            prune_by=prune_by,
            prune_values=prune_values,
            retry=retry,
        )

        if isinstance(data, (PreparedStatement, StatementResult)) or PreparedStatement.looks_like_query(data):
            return self.sql_insert(data, spark_session=spark_session, **common)

        if spark_session is None:
            session_attr = getattr(data, "sparkSession", None)
            spark_session = session_attr if session_attr is not None else self.sql.spark.resolve_session(create=False)

        if spark_session is not None:
            return self.spark_insert(
                data=data,
                schema_mode=schema_mode,
                cast_options=cast_options,
                overwrite_schema=overwrite_schema,
                spark_options=spark_options,
                spark_session=spark_session,
                **common,
            )

        return self.arrow_insert(
            data=data,
            schema_mode=schema_mode,
            cast_options=cast_options,
            overwrite_schema=overwrite_schema,
            **common,
        )

    # =========================================================================
    # arrow_insert — warehouse path, Volume staging
    # =========================================================================

    def arrow_insert(
        self,
        data,
        *,
        mode: Mode | str | None = None,
        schema_mode: Mode | str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_cols: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        where: Predicate | None = None,
        prune_by: list[str] | str | None = None,
        prune_values: Mapping[str, list[Any]] | None = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> None:
        """Insert through the warehouse SQL path with staged Parquet."""
        from yggdrasil.databricks.fs import VolumePath
        from yggdrasil.databricks.warehouse import WarehousePreparedStatement

        if isinstance(data, (PreparedStatement, StatementResult)) or PreparedStatement.looks_like_query(data):
            return self.sql_insert(
                data,
                mode=mode,
                match_by=match_by, update_cols=update_cols,
                wait=wait, raise_error=raise_error,
                zorder_by=zorder_by, optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                where=where, prune_by=prune_by,
                retry=retry,
            )

        mode_enum = Mode.from_(mode, default=Mode.AUTO)

        if mode_enum == Mode.OVERWRITE and not match_by:
            self.delete(wait=True, raise_error=False)

        target = self.create(data, mode=schema_mode)
        target_location = target.full_name(safe=True)
        existing_schema = target.collect_schema()
        cast_options = CastOptions.check(options=cast_options).check_target(
            existing_schema.to_field(),
        )

        if match_by == "auto":
            match_by = [f.name for f in existing_schema.primary_fields] or None
        prune_by = _resolve_prune_by(prune_by, existing_schema.partition_fields)

        wait_cfg = WaitingConfig.from_(wait)

        staging = VolumePath.staging_path(
            client=self.client,
            catalog_name=target.catalog_name,
            schema_name=target.schema_name,
            resource_name=target.table_name,
            max_lifetime=3600,
            temporary=bool(wait_cfg),
        )

        prune_values = prune_values or {}
        with ParquetIO() as buffer:
            buffer.write_table(data, cast_options)
            buffer.seek(0)
            if prune_by:
                prune_values = _collect_prune_values_polars(buffer, prune_by)
                logger.debug(
                    "Arrow pruning %s -> %s",
                    prune_by, {k: len(v) for k, v in prune_values.items()},
                )
            buffer.seek(0)
            staging.write_stream(buffer)

        buffer.clear()
        prune_predicates = _build_prune_predicates(prune_values, target_alias="T") if prune_values else []
        if where is not None:
            prune_predicates.append(_wrap_user_predicate(where, target_alias="T"))

        columns = list(existing_schema.field_names())
        cols_quoted = ", ".join(quote_ident(c) for c in columns)
        source_sql = f"SELECT {cols_quoted} FROM {{{_ALIAS_TMPSRC}}}"

        sql_texts = _build_dml_statements(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            mode=mode_enum,
            match_by=match_by,
            update_cols=update_cols,
            prune_predicates=prune_predicates,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
        )

        retry_active = retry is not None
        prepared: list[WarehousePreparedStatement] = []
        for sql in sql_texts:
            external_data = (
                {_ALIAS_TMPSRC: staging}
                if (f"{{{_ALIAS_TMPSRC}}}" in sql)
                else None
            )
            stmt = WarehousePreparedStatement.prepare(
                sql,
                external_data=external_data,
                catalog_name=target.catalog_name,
                schema_name=target.schema_name,
            )
            if retry_active and _classify_dml(sql):
                _apply_retry_to_warehouse_statement(stmt, retry)
            prepared.append(stmt)

        logger.debug(
            "Arrow insert -> %s | mode=%s match_by=%s prune_by=%s statements=%d retry=%s",
            target_location, mode_enum, match_by, prune_by, len(prepared),
            retry_active,
        )

        return self.sql.execute_many(prepared, wait=wait_cfg, raise_error=raise_error, engine="api")

    # =========================================================================
    # spark_insert — Spark path, temp-view source
    # =========================================================================

    def spark_insert(
        self,
        data: Any,
        *,
        mode: Mode | str | None = None,
        schema_mode: Mode | str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_cols: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_options: Optional[Dict[str, Any]] = None,
        where: Predicate | None = None,
        prune_by: list[str] | str | None = None,
        prune_values: dict[str, tuple[Any, ...]] | None = None,
        spark_session: Optional["pyspark.sql.SparkSession"] = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> None:
        """Insert into this table using Spark.

        ``retry`` is accepted for API symmetry but ignored — Spark uses
        driver-side retry, not the warehouse ``StatementResult.retry()``
        loop.
        """
        if isinstance(data, (PreparedStatement, StatementResult)) or PreparedStatement.looks_like_query(data):
            return self.sql_insert(
                data,
                mode=mode,
                match_by=match_by, update_cols=update_cols,
                wait=wait, raise_error=raise_error,
                zorder_by=zorder_by, optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                where=where, prune_by=prune_by,
                spark_session=spark_session,
                retry=retry,
            )

        if retry is not None:
            logger.debug(
                "Ignoring retry on spark_insert — Spark statements use "
                "driver-side retry."
            )

        from yggdrasil.spark.cast import any_to_spark_dataframe
        from yggdrasil.spark.statement import SparkPreparedStatement

        mode_enum = Mode.from_(mode, default=Mode.AUTO)

        # TODO: Fix async databricks notebook.
        wait = True if PyEnv.in_databricks() else wait

        if mode_enum == Mode.OVERWRITE and not match_by:
            self.delete(wait=True, raise_error=False)

        target = self.create(data, mode=schema_mode)
        target_location = target.full_name(safe=True)
        existing_schema = target.collect_schema()
        cast_options = CastOptions.check(options=cast_options).check_target(
            target.collect_data_field(),
        )

        sql_engine = self.sql
        session = spark_session or sql_engine.spark.resolve_session(create=True)
        data_df = any_to_spark_dataframe(data, cast_options)

        if match_by == "auto":
            match_by = [f.name for f in existing_schema.primary_fields] or None
        prune_by = _resolve_prune_by(prune_by, existing_schema.partition_fields)

        prune_values = prune_values or {}
        if prune_by:
            prune_values = _collect_prune_values_spark(data_df, prune_by)
            logger.debug(
                "Spark pruning %s -> %s",
                prune_by, {k: len(v) for k, v in prune_values.items()},
            )

        prune_predicates = _build_prune_predicates(prune_values, target_alias="T") if prune_values else []
        if where is not None:
            prune_predicates.append(_wrap_user_predicate(where, target_alias="T"))

        view_name = f"_yg_src_{uuid.uuid4().hex}"
        data_df.createOrReplaceTempView(view_name)

        columns = list(existing_schema.field_names())
        cols_quoted = ", ".join(quote_ident(c) for c in columns)
        source_sql = f"SELECT {cols_quoted} FROM {quote_ident(view_name)}"

        sql_texts = _build_dml_statements(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            mode=mode_enum,
            match_by=match_by,
            update_cols=update_cols,
            prune_predicates=prune_predicates,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
        )

        prepared = [SparkPreparedStatement(text=sql, spark_session=session) for sql in sql_texts]

        logger.info(
            "Spark insert -> %s | mode=%s match_by=%s prune_by=%s statements=%d",
            target_location, mode_enum, match_by, prune_by, len(prepared),
        )

        applied_conf = _delta_conf_for(overwrite_schema, spark_options)

        try:
            with sql_engine.spark.scoped_spark_conf(session, applied_conf):
                return sql_engine.execute_many(prepared, wait=wait, raise_error=raise_error, engine="spark")
        finally:
            try:
                session.catalog.dropTempView(view_name)
            except Exception:
                logger.debug("Failed to drop temp view %r; continuing.", view_name, exc_info=True)

    # =========================================================================
    # sql_insert — query source, no staging
    # =========================================================================

    def sql_insert(
        self,
        statement: "PreparedStatement | StatementResult | str",
        *,
        mode: Mode | str | None = None,
        match_by: Optional[list[str]] = None,
        update_cols: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_session: Optional["pyspark.sql.SparkSession"] = None,
        where: Predicate | None = None,
        prune_by: list[str] | str | None = None,
        prune_values: dict[str, tuple[Any]] | None = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> None:
        """Insert into this table from a SQL source query.

        Smart dispatch:

        1. Cached :class:`StatementResult` → reuse the materialised frame
           via :meth:`insert_into` (no re-execution).
        2. SparkSession reachable → run via :meth:`spark_insert`.
        3. Otherwise → warehouse-side ``INSERT … SELECT`` /
           ``MERGE … USING (q)`` with a CAST projection aligning the
           user's query schema to the target.
        """
        common = dict(
            mode=mode,
            match_by=match_by, update_cols=update_cols,
            wait=wait, raise_error=raise_error,
            zorder_by=zorder_by, optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            where=where, prune_by=prune_by, prune_values=prune_values,
            retry=retry,
        )

        if isinstance(statement, StatementResult) and statement.cached:
            spark_df = getattr(statement, "spark_dataframe", None)
            cached = spark_df if spark_df is not None else statement.to_arrow_table()
            return self.insert_into(data=cached, spark_session=spark_session, **common)

        if spark_session is None:
            spark_session = self.sql.spark.resolve_session(create=False)
        if spark_session is not None:
            text = (
                statement.statement.text
                if isinstance(statement, StatementResult)
                else (statement.text if isinstance(statement, PreparedStatement) else str(statement))
            )
            df = spark_session.sql(text)
            return self.spark_insert(data=df, spark_session=spark_session, **common)

        return self._sql_insert_warehouse_fallback(statement, **common)

    def _sql_insert_warehouse_fallback(
        self,
        statement: "PreparedStatement | StatementResult | str",
        *,
        mode: Mode | str | None,
        match_by: Optional[list[str]],
        update_cols: Optional[list[str]],
        wait: WaitingConfigArg,
        raise_error: bool,
        zorder_by: Optional[list[str]],
        optimize_after_merge: bool,
        vacuum_hours: int | None,
        where: Predicate | None,
        prune_by: list[str] | str | None,
        prune_values: dict[str, tuple[Any]] | None = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> None:
        """Warehouse fallback for :meth:`sql_insert`."""
        from yggdrasil.databricks.warehouse import WarehousePreparedStatement

        base = statement.statement if isinstance(statement, StatementResult) else statement
        source_prepared = WarehousePreparedStatement.from_(base)

        mode_enum = Mode.from_(mode, default=Mode.AUTO)

        if mode_enum == Mode.OVERWRITE and not match_by:
            self.delete(wait=True, raise_error=False)

        if not self.exists:
            raise ValueError(
                "sql_insert requires the target table to exist; "
                f"{self.full_name()!r} was not found."
            )

        target_location = self.full_name(safe=True)
        existing_schema = self.collect_schema()
        fields = list(existing_schema.fields)
        columns = [f.name for f in fields]

        if match_by == "auto":
            match_by = [f.name for f in existing_schema.primary_fields] or None

        cast_projection = ", ".join(
            (
                f"CAST(raw_src.{quote_ident(f.name)} AS "
                f"{f.to_databricks_ddl(put_name=False, put_not_null=False, put_comment=False)})"
                f" AS {quote_ident(f.name)}"
            )
            for f in fields
        )
        source_sql = (
            f"SELECT {cast_projection} FROM (\n{source_prepared.text}\n) AS raw_src"
        )

        prune_predicates: list[str] = []
        if where is not None:
            prune_predicates.append(_wrap_user_predicate(where, target_alias="T"))
        if prune_by:
            logger.debug(
                "prune_by %s ignored on warehouse-fallback sql_insert "
                "(would require re-executing source query)", prune_by,
            )

        sql_texts = _build_dml_statements(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            mode=mode_enum,
            match_by=match_by,
            update_cols=update_cols,
            prune_predicates=prune_predicates,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
        )

        retry_active = retry is not None

        prepared: list[WarehousePreparedStatement] = []
        for sql in sql_texts:
            stmt = WarehousePreparedStatement.prepare(
                sql,
                parameters=source_prepared.parameters,
                external_volume_paths=source_prepared.external_volume_paths,
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
            )
            if retry_active and _classify_dml(sql):
                _apply_retry_to_warehouse_statement(stmt, retry)
            prepared.append(stmt)

        logger.info(
            "SQL insert -> %s | mode=%s match_by=%s statements=%d retry=%s",
            target_location, mode_enum, match_by, len(prepared), retry_active,
        )

        if prepared:
            self.sql.execute_many(prepared, wait=wait, raise_error=raise_error, engine="api")

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

    def storage_location(self, operation: TableOperation = TableOperation.READ) -> str:
        if operation == TableOperation.READ_WRITE and self.infos.table_type == TableType.MANAGED:
            operation = TableOperation.READ

        return self.aws(
            operation=operation
        ).s3.path(self.infos.storage_location)

    def aws(self, operation: TableOperation = TableOperation.READ) -> "AWSClient":
        credentials = self.temporary_credentials(operation=operation)
        return aws_config_class()(
            access_key_id=credentials.aws_temp_credentials.access_key_id,
            secret_access_key=credentials.aws_temp_credentials.secret_access_key,
            session_token=credentials.aws_temp_credentials.session_token,
            region="eu-central-1",
        ).to_client()

    def temporary_credentials(self, operation: TableOperation = TableOperation.READ):
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
        creds = self.temporary_credentials(operation=operation)
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
