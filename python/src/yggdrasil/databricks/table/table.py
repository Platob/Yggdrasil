"""
Per-table resource: DDL, DML, schema introspection and storage helpers.

The :class:`Table` dataclass wraps a single Unity Catalog table and exposes
instance-level methods only.  Collection operations (``find_table``,
``list_tables``) live in :mod:`~yggdrasil.databricks.table.tables`.

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

import datetime as _dt
import logging
import re
import time
import uuid
from typing import Any, Dict, Optional, Union, TYPE_CHECKING, Mapping, Iterable, Iterator, Literal, ClassVar

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
from yggdrasil.concurrent.threading import Job
from yggdrasil.data import Field
from yggdrasil.data.data_utils import safe_constraint_name
from yggdrasil.data.enums import MimeTypes, MimeType, MediaType, MediaTypes, ModeLike, Mode, Scheme
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema as DataSchema, Schema
from yggdrasil.data.statement import PreparedStatement, StatementResult
from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.column.column import Column
from yggdrasil.databricks.path import DatabricksPath
from yggdrasil.databricks.sql.sql_utils import (
    MAX_TABLE_NAME_LEN,
    quote_ident,
    quote_qualified_ident,
    safe_table_name,
    sql_literal, escape_sql_string,
)
from yggdrasil.dataclasses import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.io import URL
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.io.path import Path
from yggdrasil.io.primitive import ParquetFile
from yggdrasil.io.tabular import Tabular, O
from yggdrasil.execution.expr import (
    Expression,
    InList,
    Logical,
    LogicalOp,
    Predicate,
    col as expr_col,
)
from yggdrasil.execution.expr.backends.sql import Dialect, to_sql as expr_to_sql

from ..fs import VolumePath
from ..volume import Volume

if TYPE_CHECKING:
    import pandas
    import polars
    import pyspark
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from yggdrasil.databricks.sql.engine import SQLEngine
    from yggdrasil.databricks.table.tables import Tables
    from yggdrasil.databricks.catalog.catalog import UCCatalog
    from yggdrasil.databricks.column.columns import Columns
    from yggdrasil.databricks.schema.schema import UCSchema
    from yggdrasil.aws.client import AWSClient
    from yggdrasil.databricks.aws import AWSDatabricksTableCredentials
    from yggdrasil.databricks.warehouse import WarehousePreparedStatement
    from yggdrasil.databricks.jobs.job import Job as DatabricksJob
    from yggdrasil.databricks.table.async_job import AsyncApplierTaskType
    from yggdrasil.databricks.table.async_write import AsyncInsert
    from yggdrasil.data.statement import StatementBatch

_READ_ONLY_MODES = frozenset({Mode.AUTO})

# Unity Catalog ``table_type`` tokens that identify view-shaped securables.
# Used by ``Table`` to dispatch view-specific DDL (``ALTER VIEW`` /
# ``DROP VIEW`` / ``CREATE VIEW``) and by ``Tables`` to filter list output.
_VIEW_TABLE_TYPES: frozenset[TableType] = frozenset({
    TableType.VIEW,
    TableType.MATERIALIZED_VIEW,
    TableType.METRIC_VIEW,
})


def _coerce_tag_str(value: Any) -> str:
    """Coerce a tag key/value to a UTF-8 string.

    PyArrow stores schema/field metadata as ``bytes``, so the
    auto-tags propagated from :data:`yggdrasil.data.Schema` flow
    through here as ``b"primary_key"`` / ``b"true"``. ``str(b"x")``
    would render the literal ``"b'x'"`` — what the Databricks API
    actually receives — so decode bytes-shaped tags before forwarding.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _resolve_table_operation(
    operation: "TableOperation | ModeLike | None",
    table_type: "TableType | None",
) -> TableOperation:
    """Resolve a user-facing operation hint into a UC :class:`TableOperation`.

    ``None`` defaults to READ for managed tables, READ_WRITE for
    external. A :class:`TableOperation` passes through. A
    :class:`Mode` / mode-like string is mapped read-only-or-not, then
    collapsed to READ for managed tables (UC won't vend write creds
    for those).
    """
    if isinstance(operation, TableOperation):
        op = operation
    elif operation is None:
        op = (
            TableOperation.READ if table_type == TableType.MANAGED
            else TableOperation.READ_WRITE
        )
    else:
        mode = Mode.from_(operation, default=Mode.AUTO)
        op = (
            TableOperation.READ if mode in _READ_ONLY_MODES
            else TableOperation.READ_WRITE
        )

    if op == TableOperation.READ_WRITE and table_type == TableType.MANAGED:
        op = TableOperation.READ
    return op

__all__ = [
    "Table",
    "YGG_SCHEMA_FIELD_PREFIX",
    "YGG_SCHEMA_FIELD_SUFFIX",
]

logger = logging.getLogger(__name__)

_INVALID_COL_CHARS = set(" ,;{}()\n\t=")

# URL path / free-text → identifier: collapse anything outside ``[0-9A-Za-z]``
# to ``_``. Compiled once so ``Table.safe_name`` is a single ``re.sub`` call.
_PATH_TO_IDENT_RE: re.Pattern[str] = re.compile(r"[^0-9A-Za-z]+")


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
#   - ``auto``      default; with ``match_by`` only non-matching rows
#                   (lightweight append: target is probed by the match
#                   keys only, no full read)
#   - ``upsert`` /
#     ``merge``     MERGE INTO with full update-and-insert; the
#                   keyed-DELETE + INSERT pair is emitted as a
#                   fallback when the engine MERGE fails.
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


def _apply_retry_to_statement(
    stmt: "PreparedStatement",
    retry: Optional[WaitingConfigArg],
) -> None:
    """Install ``retry`` on any prepared statement (warehouse or Spark)."""
    if retry is None:
        return
    if retry is False:
        stmt.retry = None
        return
    stmt.retry = WaitingConfig.from_(retry)


def _resolve_retry(retry: Optional[WaitingConfigArg]) -> Optional[WaitingConfig]:
    """Normalize a caller-supplied retry arg to a :class:`WaitingConfig`.

    ``None`` and ``False`` both disable explicit pre-installation — the
    statement-level auto-promote on transient failures still runs.  Any
    other value is coerced through :meth:`WaitingConfig.from_`.
    """
    if retry is None or retry is False:
        return None
    return WaitingConfig.from_(retry)


def _execute_dml(
    sql_engine: "SQLEngine",
    *,
    statements: list,
    wait: WaitingConfigArg,
    raise_error: bool,
    engine: Literal["api", "spark"],
):
    """Submit DML statements through *sql_engine* and surface failures.

    Replaces the legacy MERGE-fallback funnel: there's no fallback
    factory, no per-batch retry shuffle, no auto-promote dance —
    statement-level retry policies (set by the caller via
    :class:`WaitingConfig`) still fire inside
    :class:`SparkPreparedStatement` / :class:`WarehousePreparedStatement`,
    but the table layer no longer second-guesses them.

    On a failed batch we route through :meth:`StatementBatch.retry`
    rather than :meth:`raise_for_status` so a transient Delta
    concurrent-append (a race between sibling MERGE / DELETE + INSERT
    writers on overlapping keys) gets auto-promoted and retried
    instead of bubbling straight up.  Non-transient failures still
    surface — ``batch.retry`` re-raises through ``raise_for_status``
    once the budget is exhausted or the failure isn't retryable.
    """
    batch = sql_engine.execute_many(
        statements, wait=wait, raise_error=False, engine=engine,
    )
    if raise_error and batch.failed:
        batch.retry(wait=wait, raise_error=True)
    elif raise_error:
        batch.raise_for_status()
    return batch


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


def _build_prune_predicate(
    prune_values: Mapping[str, Iterable[Any]] | None,
    where: Predicate | None,
    *,
    target_alias: str,
) -> list[str]:
    """Combine ``prune_values`` + ``where`` into a single target-side SQL clause.

    Builds one AST: per-column ``InList`` from ``prune_values`` AND'd
    with the user's ``where``, target-aliased. ``InList.__post_init__``
    dedupes per-column values and ``Logical.__post_init__`` flattens
    same-op nesting so the rendered SQL is already tight without an
    explicit normalisation pass.

    Return shape stays ``list[str]`` (0 or 1 elements) so the
    downstream :func:`_build_dml_statements` / :func:`_build_merge_statement`
    / :func:`_build_delete_insert_statements` / :func:`_build_anti_join_insert`
    contract — *list of clauses AND'd together by the consumer* —
    is unchanged. A top-level ``OR`` in the final SQL gets a paren
    wrap so callers can splice it into an AND chain without
    precedence bleed.
    """
    parts: list[Expression] = []
    if prune_values:
        for column_name, vals in prune_values.items():
            materialized = tuple(vals)
            if not materialized:
                continue
            parts.append(expr_col(column_name).is_in(materialized))
    if where is not None:
        parts.append(where)
    if not parts:
        return []
    if len(parts) == 1:
        combined: Expression = parts[0]
    else:
        combined = Logical(LogicalOp.AND, tuple(parts))
    # Alias once over the combined tree — single walk instead of one
    # per part.
    aliased = _alias_columns(combined, target_alias)
    sql = expr_to_sql(aliased, dialect=Dialect.DATABRICKS)
    # Top-level OR (e.g. a single ``InList`` with ``includes_null=True``
    # renders as ``T.x IN (...) OR T.x IS NULL``) needs parens before
    # the consumer concatenates it with AND. A top-level AND is
    # already what the consumer would build anyway, so no wrap.
    if isinstance(aliased, Logical) and aliased.op is LogicalOp.OR:
        sql = f"({sql})"
    elif isinstance(aliased, InList) and aliased.includes_null:
        sql = f"({sql})"
    return [sql]


def _alias_columns(expr, alias: str):
    """Walk *expr* and stamp every :class:`Column` with ``alias``.

    Used by :func:`_wrap_user_predicate` to lift a user predicate
    onto the target-side of a MERGE so ``foo`` becomes ``T.foo``.
    Returns a new tree — the AST is immutable so we never mutate
    the caller's predicate.
    """
    from yggdrasil.execution.expr.nodes import (
        Arithmetic,
        Between,
        Cast,
        Column,
        Comparison,
        InList,
        IsNull,
        Like,
        Logical,
        Not,
    )

    if isinstance(expr, Column):
        return type(expr)(
            name=expr.name,
            field=expr.field,
            alias=expr.alias,
            qualifier=alias,
        )
    if isinstance(expr, Comparison):
        return Comparison(
            _alias_columns(expr.left, alias),
            expr.op,
            _alias_columns(expr.right, alias),
        )
    if isinstance(expr, Logical):
        return Logical(
            expr.op,
            tuple(_alias_columns(o, alias) for o in expr.operands),
        )
    if isinstance(expr, Not):
        return Not(_alias_columns(expr.operand, alias))
    if isinstance(expr, Between):
        return Between(
            _alias_columns(expr.target, alias),
            _alias_columns(expr.low, alias),
            _alias_columns(expr.high, alias),
            negated=expr.negated,
        )
    if isinstance(expr, InList):
        return InList(
            target=_alias_columns(expr.target, alias),
            values=expr.values,
            negated=expr.negated,
            includes_null=expr.includes_null,
        )
    if isinstance(expr, IsNull):
        return IsNull(_alias_columns(expr.target, alias), negated=expr.negated)
    if isinstance(expr, Like):
        return Like(
            target=_alias_columns(expr.target, alias),
            pattern=expr.pattern,
            case_insensitive=expr.case_insensitive,
            negated=expr.negated,
        )
    if isinstance(expr, Cast):
        return Cast(_alias_columns(expr.operand, alias), expr.dtype)
    if isinstance(expr, Arithmetic):
        return Arithmetic(
            expr.op,
            _alias_columns(expr.left, alias),
            _alias_columns(expr.right, alias),
        )
    return expr


def _collect_prune_values_polars(
    buffer: ParquetFile,
    prune_by: list[str],
) -> dict[str, tuple[Any, ...]]:
    df = buffer.scan_polars_frame().select(*prune_by).unique().collect()
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


def _build_column_projection(
    fields: "Iterable[Field]",
    *,
    source_alias: "str | None" = None,
) -> str:
    """Build a plain column-reference projection list for INSERT/MERGE.

    Each :class:`Field` contributes one bare (or alias-qualified)
    quoted column reference — no per-column ``CAST(... AS <ddl>)``
    wrapper.  The data has already been aligned to the target schema
    upstream: ``arrow_insert`` writes through
    :meth:`CastOptions.cast_arrow_tabular` before staging,
    ``spark_insert`` aligns the DataFrame via
    :func:`any_to_spark_dataframe`, and the warehouse INSERT itself
    performs implicit coercion at the column boundary, so the engine
    accepts the rows as-is.  Skipping the explicit CAST keeps the SQL
    short — important for wide / deeply nested schemas where the
    spelled-out DDL bloated statements past the warehouse text
    limits — and avoids paying for redundant per-column validation.

    Used by :meth:`Table.arrow_insert`, :meth:`Table.spark_insert`,
    and :meth:`Table.sql_insert` (the latter passes
    ``source_alias="raw_src"`` so columns resolve against the wrapped
    user query).
    """
    parts: list[str] = []
    for f in fields:
        col = quote_ident(f.name)
        parts.append(f"{source_alias}.{col}" if source_alias else col)
    return ", ".join(parts)


def _build_delete_insert_statements(
    *,
    target_location: str,
    source_sql: str,
    columns: list[str],
    match_by: list[str],
    prune_predicates: list[str],
) -> list[str]:
    """Build the keyed-DELETE + INSERT pair used by upsert and the merge fallback.

    Always emits exactly two statements: a key-scoped ``DELETE`` against
    target rows whose match keys appear in ``source_sql``, followed by a
    plain ``INSERT INTO ... SELECT``.  ``columns`` is the projection that
    bridges source → target — callers narrow it to the intersection of
    source-side columns when feeding a fallback so non-matching columns
    are filtered out.

    Databricks/Spark SQL doesn't accept ``DELETE FROM target USING source``;
    the keyed delete is rendered as ``DELETE FROM target T WHERE EXISTS
    (...)`` so it parses on Delta.  ``prune_predicates`` lift onto the
    outer ``WHERE`` so the target scan is bounded before the EXISTS
    subquery runs.
    """
    cols_quoted = ", ".join(quote_ident(c) for c in columns)
    key_cols = ", ".join(quote_ident(k) for k in match_by)
    key_match = " AND ".join(
        f"T.{quote_ident(k)} <=> S.{quote_ident(k)}"
        for k in match_by
    )
    exists_clause = (
        f"EXISTS (\n"
        f"  SELECT 1 FROM (\n"
        f"    SELECT DISTINCT {key_cols} FROM ({source_sql}) AS src\n"
        f"  ) AS S\n"
        f"  WHERE {key_match}\n"
        f")"
    )
    where_clauses = [*prune_predicates, exists_clause]
    where_sql = "\n  AND ".join(where_clauses)
    return [
        f"DELETE FROM {target_location} AS T\nWHERE {where_sql}",
        f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}",
    ]


def _build_merge_statement(
    *,
    target_location: str,
    source_sql: str,
    columns: list[str],
    match_by: list[str],
    update_column_names: Optional[list[str]],
    prune_predicates: list[str],
    insert_only: bool = False,
) -> str:
    """Render a single ``MERGE INTO ...`` statement.

    Used by :func:`_build_dml_statements` when ``safe_merge=False``
    (the default) — Databricks / Delta supports MERGE natively and
    plans the keyed dedup once instead of twice (one delete + one
    insert) the way the safe path does.

    ``insert_only=True`` emits a MERGE with only the ``WHEN NOT
    MATCHED THEN INSERT`` clause — the keyed-APPEND shape. Without
    it, the full update-and-insert MERGE runs.
    """
    cols_quoted = ", ".join(quote_ident(c) for c in columns)
    on_condition = _build_match_condition(
        match_by, left_alias="T", right_alias="S",
        null_safe=True, extra_predicates=prune_predicates,
    )
    insert_clause = (
        f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) "
        f"VALUES ({', '.join(f'S.{quote_ident(c)}' for c in columns)})"
    )
    if insert_only:
        return (
            f"MERGE INTO {target_location} AS T\n"
            f"USING (\n{source_sql}\n) AS S\n"
            f"ON {on_condition}\n"
            f"{insert_clause}"
        )

    update_column_names_effective = (
        update_column_names
        if update_column_names is not None
        else [c for c in columns if c not in match_by]
    )
    update_clause = ""
    if update_column_names_effective:
        update_set = ", ".join(
            f"T.{quote_ident(c)} = S.{quote_ident(c)}"
            for c in update_column_names_effective
        )
        update_clause = f"WHEN MATCHED THEN UPDATE SET {update_set}\n"

    return (
        f"MERGE INTO {target_location} AS T\n"
        f"USING (\n{source_sql}\n) AS S\n"
        f"ON {on_condition}\n"
        f"{update_clause}"
        f"{insert_clause}"
    )


def _spark_filter_existing_keys(
    *,
    session: Any,
    data_df: Any,
    target_location: str,
    match_by: list[str],
):
    """Drop rows from *data_df* whose ``match_by`` tuple already exists in target.

    The Spark fast path for keyed APPEND. Reads only the
    ``match_by`` columns from the target via
    ``session.table(target_location).select(*match_by).distinct()``
    and left-anti-joins them against the incoming DataFrame.
    Catalyst pushes the join down to the Delta files, so the
    target side reads only its key columns — much cheaper than
    the SQL ``NOT EXISTS`` shape used on the warehouse path.

    Returns a tuple ``(filtered_df, ok)``:

    * ``ok=True`` — the anti-join succeeded; ``filtered_df`` is
      the survivor DataFrame ready for a plain INSERT.
    * ``ok=False`` — target doesn't exist yet (first write) or the
      session can't see it; caller falls through to the SQL
      ``NOT EXISTS`` path which handles empty / missing targets.
    """
    try:
        target_df = session.table(target_location)
        key_df = target_df.select(*match_by).distinct()
        return data_df.join(key_df, list(match_by), "left_anti"), True
    except Exception:
        return data_df, False


def _build_anti_join_insert(
    *,
    target_location: str,
    source_sql: str,
    columns: list[str],
    match_by: list[str],
    prune_predicates: list[str],
) -> list[str]:
    """Build a keyed APPEND that filters out rows already in the target.

    Renders one statement of the shape::

        INSERT INTO target (cols)
        SELECT cols FROM (source) AS S
        WHERE NOT EXISTS (
          SELECT 1 FROM target AS T
          WHERE T.k1 <=> S.k1 AND ... [AND prune_predicates]
        )

    No ``MERGE``, no fallback, no retry — Databricks / Spark / any
    SQL engine that supports correlated ``EXISTS`` runs this
    natively. The ``<=>`` null-safe comparison matches the legacy
    MERGE behavior so rows with NULL key columns line up.

    ``prune_predicates`` (target-aliased) narrow the EXISTS scan
    to the matching partitions, so the keyed dedup doesn't read
    the whole target.
    """
    cols_quoted = ", ".join(quote_ident(c) for c in columns)
    key_match = " AND ".join(
        f"T.{quote_ident(k)} <=> S.{quote_ident(k)}"
        for k in match_by
    )
    exists_where = "\n    AND ".join([key_match, *prune_predicates])
    exists_clause = (
        f"NOT EXISTS (\n"
        f"  SELECT 1 FROM {target_location} AS T\n"
        f"  WHERE {exists_where}\n"
        f")"
    )
    return [
        f"INSERT INTO {target_location} ({cols_quoted})\n"
        f"SELECT {cols_quoted} FROM (\n{source_sql}\n) AS S\n"
        f"WHERE {exists_clause}"
    ]


def _append_maintenance_statements(
    statements: list[str],
    *,
    target_location: str,
    zorder_by: Optional[list[str]],
    optimize_after_merge: bool,
    keyed: bool,
    vacuum_hours: Optional[int],
) -> None:
    """Tack on OPTIMIZE / VACUUM statements when configured."""
    if zorder_by:
        zorder_cols = ", ".join(quote_ident(c) for c in zorder_by)
        statements.append(f"OPTIMIZE {target_location} ZORDER BY ({zorder_cols})")
    if optimize_after_merge and keyed:
        statements.append(f"OPTIMIZE {target_location}")
    if vacuum_hours is not None:
        statements.append(f"VACUUM {target_location} RETAIN {int(vacuum_hours)} HOURS")


def _build_dml_statements(
    *,
    target_location: str,
    source_sql: str,
    columns: list[str],
    mode: Mode,
    match_by: Optional[list[str]],
    update_column_names: Optional[list[str]],
    prune_predicates: list[str],
    zorder_by: Optional[list[str]] = None,
    optimize_after_merge: bool = False,
    vacuum_hours: Optional[int] = None,
    safe_merge: bool = False,
) -> list[str]:
    """Generate INSERT / MERGE / DELETE / OPTIMIZE / VACUUM SQL.

    Mode dispatch when ``match_by`` is set:

    * **safe_merge=False (default)** — emit a single ``MERGE INTO``
      statement. Databricks / Delta plans the keyed dedup once;
      :attr:`Mode.UPSERT` / :attr:`Mode.MERGE` get the full
      update-and-insert MERGE, :attr:`Mode.APPEND` /
      :attr:`Mode.AUTO` get the insert-only variant.
    * **safe_merge=True** — sidestep MERGE entirely.
      :attr:`Mode.UPSERT` / :attr:`Mode.MERGE` run a keyed ``DELETE``
      followed by ``INSERT`` (incoming wins on overlap).
      :attr:`Mode.APPEND` / :attr:`Mode.AUTO` run
      ``INSERT ... WHERE NOT EXISTS (...)`` so existing rows are
      filtered at INSERT time. Useful for backends without native
      MERGE, for callers that want "do exactly the dedup you wrote
      down" semantics, or for the Spark fast-path
      (:func:`_spark_filter_existing_keys`) which pre-filters the
      DataFrame before submission and downgrades to plain INSERT.

    Mode without keys:

    * :attr:`Mode.TRUNCATE` with ``match_by`` → DELETE + INSERT
      (truncate-by-key).
    * :attr:`Mode.TRUNCATE` no keys → ``TRUNCATE TABLE`` + INSERT.
    * :attr:`Mode.OVERWRITE` → plain INSERT (the caller already
      cleared the target up front).
    """
    cols_quoted = ", ".join(quote_ident(c) for c in columns)
    statements: list[str] = []

    if mode in (Mode.TRUNCATE, Mode.OVERWRITE):
        if mode == Mode.TRUNCATE and match_by:
            statements.extend(_build_delete_insert_statements(
                target_location=target_location,
                source_sql=source_sql,
                columns=columns,
                match_by=match_by,
                prune_predicates=prune_predicates,
            ))
        elif mode == Mode.TRUNCATE:
            statements.extend([
                f"TRUNCATE TABLE {target_location}",
                f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}",
            ])
        else:
            statements.append(
                f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}"
            )

    elif match_by and not safe_merge:
        # Native MERGE INTO — Databricks / Delta plans the dedup
        # once. Insert-only for APPEND/AUTO; full update + insert
        # for UPSERT/MERGE.
        statements.append(_build_merge_statement(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            match_by=match_by,
            update_column_names=update_column_names,
            prune_predicates=prune_predicates,
            insert_only=mode in (Mode.APPEND, Mode.AUTO),
        ))

    elif match_by and mode in (Mode.UPSERT, Mode.MERGE):
        # safe_merge=True + UPSERT — keyed DELETE then INSERT.
        # Same outcome as MERGE but without the engine-specific
        # syntax; the staged source reads twice, which is the trade
        # the caller opted into.
        statements.extend(_build_delete_insert_statements(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            match_by=match_by,
            prune_predicates=prune_predicates,
        ))

    elif match_by:
        # safe_merge=True + AUTO/APPEND — INSERT NOT EXISTS so
        # existing rows are filtered at INSERT time. The Spark
        # fast path replaces this with a DataFrame anti-join one
        # layer up in :meth:`Table.spark_insert`.
        statements.extend(_build_anti_join_insert(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            match_by=match_by,
            prune_predicates=prune_predicates,
        ))

    else:
        statements.append(
            f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}"
        )

    _append_maintenance_statements(
        statements,
        target_location=target_location,
        zorder_by=zorder_by,
        optimize_after_merge=optimize_after_merge,
        keyed=bool(match_by),
        vacuum_hours=vacuum_hours,
    )
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


# ---------------------------------------------------------------------------
# ``ygg.schema[<field_name>]`` TBLPROPERTIES — shared by sql_create and
# api_create. Brackets wrap the field name so identifiers containing
# ``.`` don't collide with the property-namespace separator
# (``ygg.schema.user.first_name`` would otherwise be ambiguous between
# a field named ``user.first_name`` and a nested ``user`` field with a
# ``first_name`` child). Everything else a reader could want
# (table_type, storage_location, partition/cluster/primary keys,
# created_at, data_source_format) is already first-class on UC's
# ``TableInfo`` — re-stamping it on TBLPROPERTIES is dead weight.
# ---------------------------------------------------------------------------

YGG_SCHEMA_FIELD_PREFIX = "ygg.schema["
YGG_SCHEMA_FIELD_SUFFIX = "]"


def _ygg_schema_key(name: str) -> str:
    """Build the ``ygg.schema[<name>]`` TBLPROPERTIES key for a field."""
    return f"{YGG_SCHEMA_FIELD_PREFIX}{name}{YGG_SCHEMA_FIELD_SUFFIX}"


def _build_ygg_properties(schema_info: DataSchema) -> dict[str, str]:
    """Build the ``ygg.schema[<field>]`` TBLPROPERTIES yggdrasil stamps on create.

    Emitted by both create paths (:meth:`Table.sql_create` and
    :meth:`Table.api_create`) so the two surfaces stay symmetric.

    Top-level data fields are dumped one-per-property under
    ``ygg.schema[<field_name>]`` (each value is a JSON document for
    that field) rather than as a single ``ygg.schema_json`` blob.
    Per-field keys keep individual TBLPROPERTIES values comfortably
    under Databricks' per-property size budget on wide schemas, and
    let readers fetch only the columns they care about. Constraint-only
    fields (FK/CHECK rows on ``schema.constraints``) are skipped: they're
    applied via the SDK constraints API and aren't columns the table
    actually carries.
    """
    props: dict[str, str] = {}
    seen: set[str] = set()
    for f in schema_info.children:
        if getattr(f, "constraint_key", False):
            continue
        name = f.name
        # Defensive de-dup: schemas constructed by hand can repeat names;
        # later definitions would silently shadow earlier ones in a dict.
        if not name or name in seen:
            continue
        seen.add(name)
        props[_ygg_schema_key(name)] = f.to_json(to_bytes=False)
    return props


# ===========================================================================
# Table — per-table resource
# ===========================================================================

class Table(DatabricksPath):
    """A single Unity Catalog table — DDL, DML, schema, storage helpers.

    Registers under :attr:`Scheme.DATABRICKS_TABLE` (``dbfs+table://``)
    so a URL of the shape
    ``dbfs+table://[creds@]host/<catalog>/<schema>/<table>?…`` round-trips
    a Table through :meth:`from_url` / :meth:`to_url`. Reads and writes
    flow through the active :class:`SQLEngine` via the existing
    :class:`Tabular` hooks (``_read_arrow_batches`` /
    ``_write_arrow_batches``); the byte-level :class:`Holder`
    primitives are intentionally not implemented because a SQL table
    is not a positional byte buffer — callers should use the Tabular
    surface (``read_arrow_table`` / ``write_arrow_table`` / …).

    Identity is ``(client, catalog_name, schema_name, table_name)``:
    two callers asking for the same fully-qualified table under the
    same client collapse onto one instance via the :class:`Singleton`
    cache, so the cached :class:`TableInfo` / columns / staging
    volume slot are shared across views into the same UC resource.
    """

    # Per-class singleton cache — keeps Table singletons separate
    # from the rest of the project's :class:`Singleton` users.
    _INSTANCES: ClassVar = Singleton._INSTANCES.__class__(default_ttl=None)
    # Cache every Table under the singleton convention; the cached
    # ``TableInfo`` / columns / staging-volume slot are worth keeping
    # for the process lifetime so navigation through
    # ``schema['<table>']`` doesn't keep refetching.
    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        service: "Tables | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        # Key on the bound :class:`DatabricksClient` *instance* + the
        # three-part name. Same convention as :class:`Catalog` /
        # :class:`Schema`. ``safe_table_name`` is applied here so two
        # callers with semantically-equivalent but textually-distinct
        # names (long names, suffix-trimmed forms) collapse correctly.
        client = None
        try:
            client = service.client if service is not None else None
        except Exception:
            client = None
        if catalog_name is None and service is not None:
            catalog_name = getattr(service, "catalog_name", None)
        if schema_name is None and service is not None:
            schema_name = getattr(service, "schema_name", None)
        return (cls, client, catalog_name, schema_name, safe_table_name(table_name))

    def __new__(
        cls,
        service: "Tables | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        *,
        singleton_ttl: "int | None" = ...,
        **kwargs: Any,
    ):
        # Mirror :class:`Catalog` / :class:`Schema`'s opt-in cache
        # contract: per-call ``singleton_ttl`` overrides
        # ``_SINGLETON_TTL``; ``...`` on both sides means "don't
        # register" and every call allocates a fresh instance. Cache
        # lookup runs BEFORE the :class:`RemotePath` /
        # :class:`Holder` construction chain so a hit skips
        # allocation entirely; ``object.__new__`` keeps the MRO's
        # :class:`Singleton.__new__` from re-keying with empty args.
        if singleton_ttl is ...:
            singleton_ttl = cls._SINGLETON_TTL

        def _allocate() -> "Table":
            return object.__new__(cls)

        if singleton_ttl is ...:
            return _allocate()

        key = cls._singleton_key(
            service,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )
        with cls._INSTANCES_LOCK:
            existing = cls._INSTANCES.get(key)
            if existing is not None:
                return existing
            instance = _allocate()
            try:
                object.__setattr__(instance, "_singleton_key_", key)
            except AttributeError:
                pass
            ttl_arg = (
                float(singleton_ttl)
                if isinstance(singleton_ttl, int) and not isinstance(singleton_ttl, bool)
                else singleton_ttl
            )
            cls._INSTANCES.set(key, instance, ttl=ttl_arg)
            return instance

    def _stat_uncached(self) -> IOStats:
        return IOStats(
            size=0,
            mtime=0,
            kind=IOKind.DIRECTORY if self.exists else IOKind.MISSING,
            media_type=MediaTypes.DATABRICKS_UNITY_CATALOG_TABLE
        )

    def _bwrite(self, data: BytesIO, pos: int, mode: Mode) -> int:
        raise NotImplementedError("Table is a read-only resource")

    def _bread(self, n: int, pos: int, mode: Mode) -> BytesIO:
        raise NotImplementedError("Table is a read-only resource")

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        del parents, exist_ok
        if not self.exists:
            raise NotImplementedError("Table is a read-only resource")

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["Path"]:
        del recursive, singleton_ttl
        return iter(())

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        self.delete(wait=wait, missing_ok=missing_ok)

    def _remove_dir(self, recursive: bool, missing_ok: bool, wait: WaitingConfig) -> None:
        del recursive
        self.delete(wait=wait, missing_ok=missing_ok)

    def full_path(self) -> str:
        return self.full_name()

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_TABLE

    @classmethod
    def options_class(cls) -> type[CastOptions]:
        return CastOptions

    @classmethod
    def safe_name(cls, raw: str | None) -> str:
        """Build a Unity-Catalog-safe table name from any raw string.

        Centralized "raw string → table name" builder so every caller
        (URL paths, free-text in user code, composed names from upstream
        metadata) lands on the same identifier without duplicating the
        sanitization logic.

        Pipeline:

        1. Lowercase the input, collapse every run of non-alphanumeric
           characters to a single ``_`` (``/``, ``.``, query-string
           punctuation, whitespace, non-ASCII all fold to the same
           separator).
        2. Strip surrounding ``_``; substitute ``"root"`` for the empty
           result so ``"/"`` / ``""`` / ``None`` still yield a legal
           identifier.
        3. Hand off to :func:`safe_table_name` for the 255-char UC
           ceiling — overflow tokens are joined and BLAKE2b-hashed
           into a 32-char suffix so distinct overflows stay distinct.

        When the returned name differs from *raw* (sanitization or
        truncation kicked in), a :class:`logging.WARNING` is emitted
        on this module's logger so the rewrite is visible in the wall
        of logs that any pipeline already collects. An identifier
        that's already safe round-trips silently — no warning churn
        for the steady-state case.
        """
        original = raw or ""
        cleaned = _PATH_TO_IDENT_RE.sub("_", original.lower()).strip("_")
        if not cleaned:
            cleaned = "root"
        name = safe_table_name(cleaned)
        assert name is not None and len(name) <= MAX_TABLE_NAME_LEN, (
            f"Table.safe_name: derived name {name!r} "
            f"({len(name) if name else 0} chars) exceeds Unity Catalog's "
            f"{MAX_TABLE_NAME_LEN}-char limit — safe_table_name contract broken."
        )
        if original and name != original:
            logger.warning(
                "Sanitized table name %r -> %r (reason=%s)",
                original, name,
                "truncated" if len(original) > MAX_TABLE_NAME_LEN else "non-identifier-chars",
            )
        return name

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
        url: URL | None = None,
        temporary: bool = False,
        singleton_ttl: "int | None" = ...,
    ):
        # ``singleton_ttl`` is consumed by ``__new__``; accept it here
        # too so Python's auto-call after ``__new__`` doesn't trip on
        # an unexpected kwarg.
        del singleton_ttl
        # Singleton-cached re-entry: a second ``Table(service=…,
        # catalog_name=…, schema_name=…, table_name=…)`` call returns
        # the live instance via ``__new__``; skip the second pass so
        # the cached ``_infos`` / columns / staging volume don't get
        # reset under the caller.
        if getattr(self, "_initialized", False):
            return

        if service is None:
            from .tables import Tables
            service = Tables.current()

        # Build a canonical ``dbfs+table://...`` URL so :class:`Holder`
        # has a real URL to bind ``self._url`` to. The host comes from
        # the underlying client (when available) so the URL alone
        # round-trips through :meth:`from_url`.
        if url is None:
            host = ""
            try:
                base = service.client.base_url
                host = base.host or ""
            except Exception:
                host = ""
            path_parts = [
                p for p in (catalog_name, schema_name, table_name) if p
            ]
            url = URL(
                scheme=type(self).scheme.value,
                host=host,
                path="/" + "/".join(path_parts) if path_parts else "/",
            )

        super().__init__(service=service, url=url, temporary=temporary)
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        # Unity Catalog caps identifiers at 255 chars. Normalize once at the
        # boundary so every downstream SQL/URL/cache key sees the safe form.
        self.table_name = safe_table_name(table_name)
        self._infos = infos
        self._infos_fetched_at = infos_fetched_at
        self._columns = columns
        self._staging_volume: Volume | None = None
        self._async_job: "DatabricksJob | None" = None
        self._initialized = True

    # ------------------------------------
    # Tabular
    # ------------------------------------

    @classmethod
    def default_media_type(cls) -> MimeType:
        return MimeTypes.DATABRICKS_UNITY_CATALOG_TABLE

    # ------------------------------------------------------------------
    # Holder primitives — Table is *logical*, not byte-shaped.
    # The Tabular surface (``read_arrow_table`` / ``write_arrow_table``
    # / …) is the supported way to move rows; the byte-level
    # primitives raise so a misuse fails loudly with a hint at the
    # right surface.
    # ------------------------------------------------------------------

    @property
    def is_memory(self) -> bool:
        return False

    @property
    def is_local_path(self) -> bool:
        return False

    @property
    def is_remote_path(self) -> bool:
        # The table is *logical* — neither local nor remote in the
        # filesystem sense. The Databricks-side identity lives in the
        # warehouse, not at a file URL we can hand to ``is_remote_path``.
        return False

    @property
    def size(self) -> int:
        # A SQL table has no positional byte size. Return 0 so
        # ``BytesIO(holder=table)``-style code sees an empty buffer
        # instead of crashing; the byte primitives still raise on
        # actual read/write attempts.
        return 0

    def stat(self) -> IOStats:
        return self._stat()

    def _stat(self) -> IOStats:
        return IOStats(
            size=0, mtime=0.0, kind=IOKind.MISSING,
            media_type=type(self).default_media_type(),
        )

    def _read_mv(self, n: int, pos: int) -> memoryview:
        raise NotImplementedError(
            f"{type(self).__name__} is a logical Unity Catalog table, "
            f"not a positional byte buffer. Use the Tabular surface "
            f"(``read_arrow_table()``, ``read_pandas_frame()``, etc.) "
            f"to materialize rows."
        )

    def _write_mv(self, data: memoryview, pos: int) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} is a logical Unity Catalog table, "
            f"not a positional byte buffer. Use ``insert(...)`` / "
            f"``write_arrow_table(...)`` to write rows."
        )

    def reserve(self, n: int) -> None:
        # No capacity layer to pre-grow; honor the contract by
        # rejecting only nonsense inputs.
        if n < 0:
            raise ValueError(f"reserve size must be >= 0, got {n!r}")

    def truncate(self, n: int) -> int:
        raise NotImplementedError(
            f"{type(self).__name__}.truncate is byte-shaped and does "
            f"not apply to a SQL table. Use ``insert(..., mode='overwrite')`` "
            f"or ``execute('TRUNCATE TABLE ...')`` for the SQL equivalent."
        )

    def _clear(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}._clear is byte-shaped and does "
            f"not apply to a SQL table. Use ``execute('DROP TABLE ...')`` "
            f"or ``insert(..., mode='overwrite')`` for the SQL equivalent."
        )

    # ------------------------------------------------------------------
    # URLBased — ``dbfs+table://[creds@]host/<cat>/<sch>/<tbl>``
    # ------------------------------------------------------------------

    @classmethod
    def from_url(cls, url: "URL | str", **kwargs: Any) -> "Table":
        """Build a :class:`Table` from a ``dbfs+table://...`` URL.

        Reads the catalog / schema / table from the URL path
        (``/catalog/schema/table``) and, when ``service`` is not
        passed in *kwargs*, infers the underlying
        :class:`DatabricksClient` from the URL via
        :meth:`DatabricksClient.from_url` — userinfo carries the PAT
        / OAuth secret, the URL host is the workspace, and remaining
        query items are forwarded as DatabricksClient init kwargs.
        Then a :class:`Tables` service is built on top of that
        client.

        Caller-supplied ``service`` / ``catalog_name`` /
        ``schema_name`` / ``table_name`` overrides anything the URL
        provided.
        """
        u = URL.from_(url)
        path = (u.path or "/").strip("/")
        parts = path.split("/") if path else []
        cat = parts[0] if len(parts) > 0 else None
        sch = parts[1] if len(parts) > 1 else None
        tbl = parts[2] if len(parts) > 2 else None

        service = kwargs.pop("service", None)
        if service is None:
            client = kwargs.pop("client", None)
            if client is None:
                # Coerce the URL through ``DatabricksClient.from_url``
                # so the same userinfo / host / query knobs that work
                # on ``dbks://`` work on ``dbfs+table://`` too.
                client = DatabricksClient.from_url(u)
            from .tables import Tables
            service = Tables(client=client)
        else:
            kwargs.pop("client", None)

        return cls(
            service=service,
            catalog_name=kwargs.pop("catalog_name", None) or cat,
            schema_name=kwargs.pop("schema_name", None) or sch,
            table_name=kwargs.pop("table_name", None) or tbl,
            **kwargs,
        )

    def to_url(self) -> URL:
        """Render this Table as a ``dbfs+table://...`` URL.

        Layers the table's ``/catalog/schema/table`` path on top of
        :meth:`DatabricksClient.to_url` so credentials / profile /
        account_id ride along the same URL — symmetric with
        :meth:`from_url`.
        """
        try:
            client_url = self.client.to_url(scheme=type(self).scheme.value)
        except Exception:
            # No usable client — fall back to a bare logical URL.
            client_url = URL(scheme=type(self).scheme.value)
        path_parts = [
            p for p in (self.catalog_name, self.schema_name, self.table_name) if p
        ]
        return client_url.with_path("/" + "/".join(path_parts) if path_parts else "/")

    def _options_to_sql(self, options: CastOptions):
        safe_char = "`"
        names = ",".join(
            safe_char + name + safe_char
            for name in options.column_names or [c.name for c in self.columns]
        )
        query = f"SELECT {names} FROM {self.full_name(safe=True)}"

        if options.predicate is not None:
            query += (
                f" WHERE "
                f"{expr_to_sql(options.predicate, dialect=Dialect.DATABRICKS)}"
            )

        if options.row_limit:
            query += f" LIMIT {options.row_limit}"

        return query

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        options = options.with_source(source=self.collect_schema())
        query = self._options_to_sql(options)

        try:
            execution = self.sql.execute(query)
        except Exception:
            if not self.exists and options.target:
                self.create(options.target)
                s: pa.Schema = options.target.to_arrow_schema()
                yield pa.RecordBatch.from_pylist([], schema=s)
                return
            else:
                raise

        for batch in execution.read_arrow_batches(options=options):
            yield batch

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions
    ) -> None:
        options = options.with_target(self.collect_schema(options))

        return self.insert(
            batches,
            mode=options.mode,
            match_by=options.match_by_keys,
            update_column_names=options.update_column_names,
            wait=options.wait,
            zorder_by=options.zorder_by,
            optimize_after_merge=options.optimize_after_merge,
            vacuum_hours=options.vacuum_hours,
            # Write-side filter — the unified ``predicate`` survives
            # the MERGE / UPDATE planning so callers can scope the
            # destination rewrite. The same predicate is consulted by
            # the read path; backends decide which scope applies.
            where=options.predicate,
            prune_by=options.prune_by,
            prune_values=options.prune_values,
            retry=options.retry,
            return_data=options.return_data,
            safe_merge=options.safe_merge,
        )

    def _read_spark_frame(self, options: O) -> "SparkDataFrame":
        options = options.with_source(source=self.collect_schema(options))
        query = self._options_to_sql(options)

        try:
            execution = self.sql.execute(query)
        except Exception:
            if not self.exists and options.target:
                self.create(options.target)
                s: pa.Schema = options.target.to_spark_schema()
                return options.get_spark_session().createDataFrame([], schema=s)
            else:
                raise

        return execution.read_spark_frame(options)

    def _write_spark_frame(
        self,
        frame: "SparkDataFrame",
        options: O,
    ) -> None:
        return self.spark_insert(
            frame,
            mode=options.mode,
            match_by=options.match_by_keys,
            wait=options.wait,
            return_data=options.return_data,
            safe_merge=options.safe_merge,
            spark_session=getattr(options, "spark_session", None),
        )

    # Properties
    
    @property
    def name(self):
        return self.table_name

    @property
    def explore_url(self) -> URL:
        """Workspace UI deep-link for this table (``/explore/data/...``).

        Mirrors :attr:`Catalog.explore_url` / :attr:`Schema.explore_url`.
        The canonical addressable URL for this table lives on
        :attr:`url` (inherited from :class:`Holder`); ``explore_url``
        is the human-friendly Catalog Explorer link.
        """
        return (
            self.client.base_url
            .with_path(f"/explore/data/{self.catalog_name}/{self.schema_name}/{self.table_name}")
        )

    @classmethod
    def from_(
        cls,
        obj: Any,
        *,
        media_type: MediaType | None = None,
        default: Any = ...,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        service: "Tables",
        **kwargs,
    ):
        if isinstance(obj, cls):
            return obj

        return cls.from_str(
            location=str(obj) if obj is not None else None,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            service=service
        )

    @classmethod
    def from_str(
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
    def catalog(self) -> "UCCatalog":
        """Navigate up to the parent :class:`UCCatalog`.

        Returns the singleton-cached :class:`UCCatalog` for this
        client + catalog name — repeated calls hand back the same
        instance with shared :class:`CatalogInfo` cache.
        """
        from yggdrasil.databricks.catalog.catalog import UCCatalog as _Catalog
        return _Catalog(
            service=self.client.catalogs,
            catalog_name=self.catalog_name,
        )

    @property
    def schema(self) -> "UCSchema":
        """Navigate up to the parent :class:`UCSchema`.

        Returns the singleton-cached :class:`UCSchema` for this
        client + (catalog, schema) — repeated calls hand back the
        same instance with shared :class:`SchemaInfo` cache.
        """
        from yggdrasil.databricks.schema.schema import UCSchema as _Schema
        return _Schema(
            service=self.client.schemas,
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

    def invalidate_singleton(self, remove_global: bool = False) -> None:
        object.__setattr__(self, "_infos", None)
        object.__setattr__(self, "_infos_fetched_at", None)
        object.__setattr__(self, "_columns", None)
        self._invalidate_entity_tag_cache()
        super().invalidate_singleton(remove_global=remove_global)

    def _invalidate_entity_tag_cache(self) -> None:
        """Drop cached tag lists for this table and every cached column."""
        tags = self.client.entity_tags
        tags.invalidate_cached_tags("tables", self.full_name())
        # Use the still-cached columns list (if any) — refusing to refetch
        # ``infos`` here keeps invalidation cheap and safe inside teardown.
        for col in (self._columns or ()):
            tags.invalidate_cached_tags("columns", self.column_full_name(col.name))

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
        logger.debug(
            "Stored info for table %r (id=%s, columns=%d, type=%s)",
            self, getattr(infos, "table_id", None),
            len(self._columns), getattr(infos, "table_type", None),
        )
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
    # View-shaped tables — Unity Catalog stores views in the same ``tables``
    # API as managed/external tables, distinguished by ``table_type``.
    # =========================================================================

    @property
    def table_type(self) -> Optional[TableType]:
        """:class:`TableType` from the cached ``infos``.

        Returns ``None`` when the table hasn't been resolved against
        Unity Catalog yet — the property never triggers a network
        round trip on its own. Callers that need a guaranteed-fresh
        answer should access ``self.infos.table_type`` directly.
        """
        cached = self._infos
        return cached.table_type if cached is not None else None

    @property
    def is_view(self) -> bool:
        """True for ``VIEW`` / ``MATERIALIZED_VIEW`` / ``METRIC_VIEW`` securables.

        Reads the cached :attr:`table_type`; returns ``False`` until
        the table's ``infos`` has been resolved at least once.
        """
        return self.table_type in _VIEW_TABLE_TYPES

    @property
    def is_materialized_view(self) -> bool:
        return self.table_type == TableType.MATERIALIZED_VIEW

    @property
    def is_metric_view(self) -> bool:
        return self.table_type == TableType.METRIC_VIEW

    @property
    def view_definition(self) -> Optional[str]:
        """The SQL ``SELECT`` text for a view; ``None`` for non-views.

        Reads the cached ``infos``; does not trigger a remote fetch.
        """
        cached = self._infos
        return cached.view_definition if cached is not None else None

    @property
    def view_dependencies(self):
        """Upstream dependencies declared by a view (cached only)."""
        cached = self._infos
        return cached.view_dependencies if cached is not None else None

    # ── view name aliases — old ``view_name`` callers stay working ───────────

    @property
    def view_name(self) -> str:
        """Alias for :attr:`table_name` so view-style call sites keep working."""
        return self.table_name

    @view_name.setter
    def view_name(self, value: str) -> None:
        self.table_name = value

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
        logger.debug(
            "Collecting schema for table %r (columns=%d)", self, len(self.columns),
        )
        metadata: dict[bytes, bytes] = {
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

        schema = DataSchema.from_fields(fields, metadata=metadata, name=self.table_name, nullable=False)
        self._persist_schema(schema)
        logger.debug(
            "Built schema for table %r (fields=%d, metadata_keys=%d)",
            self, len(fields), len(metadata),
        )
        return schema

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
        from yggdrasil.databricks.column.columns import Columns

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
        update_dtype = mode in (Mode.UPSERT, Mode.MERGE, Mode.OVERWRITE)
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
                    f"`{data_field.name}` {data_field.dtype.to_spark_name()}"
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
                existing_ddl = existing.field.dtype.to_spark_name()
                new_ddl = data_field.dtype.to_spark_name()
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
            self.invalidate_singleton(remove_global=True)

        return self

    def create(
        self,
        definition: Schema,
        *,
        mode: Mode | str | None = None,
        storage_location: str | None = None,
        comment: str | None = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        missing_ok: bool = True,
        wait: WaitingConfigArg = True,
        or_replace: bool = False,
        record_ygg_properties: bool = True,
    ) -> "Table":
        mode = Mode.from_(mode, default=Mode.AUTO)

        # ``or_replace=True`` — one-shot atomic replacement via
        # ``CREATE OR REPLACE TABLE ... USING <format>``. Saves a round
        # trip versus delete + recreate and removes the intermediate
        # "table missing" window the warehouse used to see between the
        # two calls. OR REPLACE is supported for managed Delta tables;
        # external / explicit-storage paths fall through to the legacy
        # drop + recreate (UC's tables.create API has no replace verb).
        if or_replace:
            is_managed_delta = (
                (table_type is None or table_type == TableType.MANAGED)
                and storage_location is None
                and data_source_format == DataSourceFormat.DELTA
            )
            if is_managed_delta:
                result = self.sql_create(
                    definition,
                    comment=comment,
                    missing_ok=False,
                    or_replace=True,
                    wait=wait,
                    properties=properties,
                    data_source_format=data_source_format,
                    record_ygg_properties=record_ygg_properties,
                )
                self.invalidate_singleton(remove_global=True)
                return result
            self.delete(wait=True, missing_ok=True, delete_staging=False, delete_job=False)

        if self.exists:
            if mode == Mode.ERROR_IF_EXISTS:
                raise ValueError(f"Table {self!r} already exists")
            elif mode in (Mode.IGNORE, Mode.AUTO):
                return self

            schema = DataSchema.from_(definition)
            return self.with_columns(schema.fields, mode=mode)

        if table_type is None:
            table_type = TableType.EXTERNAL if storage_location else TableType.MANAGED

        if table_type == TableType.MANAGED:
            result = self.sql_create(
                definition,
                comment=comment,
                missing_ok=missing_ok,
                properties=properties,
                record_ygg_properties=record_ygg_properties,
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
                missing_ok=missing_ok,
                record_ygg_properties=record_ygg_properties,
            )

        return result

    def sql_create(
        self,
        description: DataSchema,
        *,
        storage_location: str | None = None,
        comment: str | None = None,
        properties: Optional[dict[str, Any]] = None,
        missing_ok: bool = True,
        or_replace: bool = False,
        wait: WaitingConfigArg = True,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        optimize_write: bool = True,
        auto_compact: bool = True,
        enable_cdf: bool | None = None,
        enable_deletion_vectors: bool | None = None,
        target_file_size: int | None = None,
        column_mapping_mode: str | None = None,
        auto_tag: bool = True,
        record_ygg_properties: bool = True,
    ) -> "Table":
        schema_info = DataSchema.from_any(description)
        if auto_tag:
            schema_info = schema_info.autotag()
        comment = comment or schema_info.comment
        effective_fields: list[Field] = []
        column_definitions: list[str] = []
        partition_by = schema_info.partition_fields
        cluster_by = schema_info.cluster_fields
        primary_keys = schema_info.primary_fields

        for f in schema_info.children:
            effective_fields.append(f)
            column_definitions.append(f.to_spark_name())

        any_invalid = any(_needs_column_mapping(f.name) for f in effective_fields)
        if column_mapping_mode is None:
            column_mapping_mode = "name" if any_invalid else "none"

        # Inline-PK constraint: a single named PRIMARY KEY clause covering
        # every primary-key field.  Delta requires PK columns to be NOT
        # NULL — already enforced by the with_nullable(False) loop above.
        # FK / CHECK constraints can't be expressed inline against an
        # arbitrary parent table (the SDK ``table_constraints`` API does
        # the cross-table reference); they're applied post-create below.
        constraint_clauses = self._build_inline_constraints(
            self.full_name(safe=False), primary_keys
        )

        table_definitions = column_definitions + constraint_clauses

        if or_replace and missing_ok:
            raise ValueError("Use either or_replace or missing_ok, not both.")

        if or_replace:
            create_kw = "CREATE OR REPLACE TABLE"
        elif missing_ok:
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
        if record_ygg_properties:
            props.update(_build_ygg_properties(schema_info))
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
        logger.debug(
            "Creating table %r via SQL (or_replace=%s, missing_ok=%s, "
            "columns=%d, partition_by=%d, cluster_by=%d, primary_keys=%d, "
            "data_source_format=%s, column_mapping_mode=%s)",
            self, or_replace, missing_ok,
            len(column_definitions), len(partition_by or ()),
            len(cluster_by or ()), len(primary_keys or ()),
            data_source_format, column_mapping_mode,
        )

        try:
            self.sql.execute(statement, wait=wait)
        except Exception as exc:
            if "SCHEMA_NOT_FOUND" in str(exc):
                logger.debug(
                    "Parent schema missing for table %r — auto-creating %s.%s and retrying",
                    self, self.catalog_name, self.schema_name,
                )
                self.sql.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {quote_ident(self.catalog_name)}.{quote_ident(self.schema_name)}",
                    wait=True,
                )
                self.sql.execute(statement, wait=wait)
            elif "CONSTRAINT_ALREADY_EXISTS_IN_SCHEMA" in str(exc):
                logger.debug(
                    "Constraint already exists on table %r — ignoring", self,
                )
            else:
                raise

        # Apply remaining constraints (FK / CHECK) via the SDK post-create.
        # Inline PK was already emitted in DDL — skip it here.
        self._apply_post_create_constraints(schema_info)

        if schema_info.tags:
            self.set_tags(schema_info.tags)

            # Per-column tags in one parallelised pass rather than N sequential
            # round-trips. validate=False: column names are authoritative here
            # (we just emitted the DDL from these same fields).
        column_tag_batches = {
            f.name: f.tags for f in effective_fields if f.tags
        }
        if column_tag_batches:
            self.update_columns_tags(column_tag_batches, validate=False)

        return self

    def update_columns_tags(
        self,
        tags_by_column: Mapping[str, Mapping[str, str] | list[EntityTagAssignment]] | None,
        *,
        mode: ModeLike | None = None,
        parallel_columns: int | bool | None = None,
        parallel_per_column: int | bool | None = None,
        cache_ttl: float | None = 300.0,
        continue_on_error: bool = True,
        validate: bool = True,
    ) -> dict[str, BaseException | None]:
        """Apply tag batches to many columns of this table in parallel.

        Per-column counterpart of :meth:`set_tags`. Each column's batch is
        routed through :meth:`EntityTags.update_entities_tags` with the same
        *mode* and *cache_ttl*; columns are processed concurrently up to
        *parallel_columns*.

        Args:
            tags_by_column:
                Mapping of column name to its tag batch. Each batch may be a
                ``{tag_key: tag_value}`` dict or a list of
                :class:`EntityTagAssignment` (entity addressing on the
                assignments is filled in here — callers don't need to set it).
            mode:
                Batch mode applied per column. See
                :meth:`EntityTags.update_entity_tags` for semantics.
            parallel_columns:
                Outer concurrency — columns processed at once. Defaults to 4.
            parallel_per_column:
                Inner concurrency — writes within a single column's batch.
                Defaults to 1; bump only when the workspace can absorb the
                extra load (rate limits are workspace-wide).
            cache_ttl:
                TTL for the per-column tag-list cache reads used to diff
                before writing. ``None`` bypasses the cache.
            continue_on_error:
                When ``True`` (default), per-column failures are returned in
                the result rather than aborting the whole call. With
                ``False``, the first exception propagates.
            validate:
                When ``True`` (default), unknown column names raise
                :class:`ValueError` before any write goes out. Turn off when
                applying tags speculatively against a partially-known schema.

        Returns:
            ``{column_name: None | BaseException}``. ``None`` denotes success.
        """
        if not tags_by_column:
            return {}

        # ---- validate column names against the table schema --------------
        # Cheap local check — saves a round trip on the typo case where the
        # API would otherwise return an opaque "entity not found".
        if validate:
            known = {c.name for c in self.columns}
            unknown = [name for name in tags_by_column if name not in known]
            if unknown:
                raise ValueError(
                    f"Unknown column(s) on {self.full_name()}: {sorted(unknown)}. "
                    f"Pass validate=False to apply tags anyway."
                )

        # ---- normalise into the {(et, en): batch} shape ------------------
        # Each column is its own UC entity; we build entity names eagerly so
        # the assignments carry the right identity and update_entities_tags
        # can group/dispatch directly without re-deriving them.
        full = self.full_name()
        grouped: dict[tuple[str, str], list[EntityTagAssignment]] = {}

        for col_name, batch in tags_by_column.items():
            entity_name = f"{full}.{col_name}"
            key = ("columns", entity_name)

            if not batch:
                # OVERWRITE with an empty batch clears all tags for that
                # column; other modes drop the entry. update_entities_tags
                # does this filter itself, but normalising here keeps the
                # column→entity mapping symmetric on the way back out.
                grouped[key] = []
                continue

            if isinstance(batch, Mapping):
                assignments = [
                    EntityTagAssignment(
                        entity_type="columns",
                        entity_name=entity_name,
                        tag_key=_coerce_tag_str(k),
                        tag_value=_coerce_tag_str(v) if v is not None else "",
                    )
                    for k, v in batch.items()
                ]
            else:
                # List of EntityTagAssignment — stamp our entity addressing
                # over whatever the caller put on them, since we own the
                # routing here. Copying via from_dict/to_dict keeps the
                # frozen-ness intact without touching SDK internals.
                assignments = []
                for a in batch:
                    if not isinstance(a, EntityTagAssignment):
                        raise TypeError(
                            f"update_columns_tags: expected EntityTagAssignment "
                            f"in list batch for column {col_name!r}, got {type(a)}"
                        )
                    d = a.as_dict() if hasattr(a, "as_dict") else dict(a.__dict__)
                    d["entity_type"] = "columns"
                    d["entity_name"] = entity_name
                    assignments.append(EntityTagAssignment.from_dict(d))

            grouped[key] = assignments

        # ---- dispatch via the multi-entity service -----------------------
        raw_results = self.client.entity_tags.update_entities_tags(
            tags_by_entity=grouped,
            mode=mode,
            parallel_entities=parallel_columns,
            parallel_per_entity=parallel_per_column,
            cache_ttl=cache_ttl,
            continue_on_error=continue_on_error,
        )

        # ---- pivot results back to column-name keyspace ------------------
        # Strip the ``"columns"`` entity_type and the table prefix from the
        # entity_name so callers don't have to know we routed through the
        # multi-entity API underneath.
        prefix = f"{full}."
        out: dict[str, BaseException | None] = {}
        for (entity_type, entity_name), err in raw_results.items():
            col_name = (
                entity_name[len(prefix):]
                if entity_name.startswith(prefix) else entity_name
            )
            out[col_name] = err
        return out

    @staticmethod
    def _build_inline_constraints(prefix: str, primary_keys: Iterable[Field]) -> list[str]:
        """Render inline DDL ``CONSTRAINT … PRIMARY KEY(…)`` clauses.

        FK / CHECK aren't emitted inline: FK needs a parent reference that
        only the constraint :class:`Field` (or the SDK call) carries, and
        CHECK predicates aren't part of this layer.  Those go through
        :meth:`_apply_post_create_constraints`.
        """
        pk_fields = [f for f in primary_keys if f and f.primary_key]
        if not pk_fields:
            return []

        col_names = [f.name for f in pk_fields]
        constraint_name = safe_constraint_name(col_names, prefix="pk_" + prefix)
        cols = ", ".join(quote_ident(n) for n in col_names)
        return [f"CONSTRAINT {quote_ident(constraint_name)} PRIMARY KEY ({cols}) RELY"]

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
                "yggdrasil.databricks.constraints not available — "
                "skipping post-create constraints on table %r", self,
            )
            return

        constraints_service = TableConstraints(client=self.client)
        for cf in constraint_fields:
            try:
                constraints_service.create_constraint(self, cf)
            except Exception:
                logger.warning(
                    "Failed to create constraint %r on table %r",
                    cf.name, self, exc_info=True,
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
        missing_ok: bool = False,
        record_ygg_properties: bool = True,
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
        if missing_ok and self.exists:
            return self

        schema_info = DataSchema.from_any(definition).autotag()
        comment = comment or schema_info.comment

        effective_fields: list[Field] = []
        column_infos: list[ColumnInfo] = []
        for position, f in enumerate(schema_info.children):
            if f.constraint_key:
                continue
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

        merged_properties: dict[str, str] = {}
        if record_ygg_properties:
            merged_properties.update(_build_ygg_properties(schema_info))
        if properties:
            merged_properties.update({str(k): str(v) for k, v in properties.items()})

        logger.debug(
            "Creating table %r via API (table_type=%s, data_source_format=%s, "
            "storage_location=%s, columns=%d, properties=%d)",
            self, table_type, data_source_format,
            storage_location, len(column_infos), len(merged_properties),
        )
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
            if missing_ok and "already exists" in str(exc).lower():
                logger.debug(
                    "Table %r already exists — soft-resetting cache", self,
                )
                return self
            raise

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
        ddl = f.dtype.to_spark_name()
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

    # =========================================================================
    # View DDL — same securable family, different create / drop keywords
    # =========================================================================

    def create_view_ddl(
        self,
        query: str,
        *,
        or_replace: bool = False,
        missing_ok: bool = False,
        columns: Iterable[str] | None = None,
        comment: str | None = None,
        properties: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Render a ``CREATE [OR REPLACE] VIEW [IF NOT EXISTS]`` DDL statement.

        Mirrors the legacy :meth:`View.create_ddl` shape; ``or_replace``
        and ``missing_ok`` are mutually exclusive, and the SELECT
        text is required.
        """
        if or_replace and missing_ok:
            raise ValueError("Use either or_replace or missing_ok, not both.")

        select_text = (query or "").strip().rstrip(";").strip()
        if not select_text:
            raise ValueError("View query (SELECT text) cannot be empty")

        if or_replace:
            create_kw = "CREATE OR REPLACE VIEW"
        elif missing_ok:
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

    def create_view(
        self,
        query: str,
        *,
        mode: ModeLike = None,
        or_replace: bool | None = None,
        missing_ok: bool | None = None,
        columns: Iterable[str] | None = None,
        comment: str | None = None,
        properties: Optional[Mapping[str, Any]] = None,
        tags: Mapping[str, str] | None = None,
        wait: WaitingConfigArg = True,
    ) -> "Table":
        """Create (or replace) this Table as a Unity Catalog view.

        When neither ``or_replace`` nor ``missing_ok`` is provided
        the keywords are derived from ``mode``:

        * :data:`Mode.OVERWRITE` → ``or_replace=True``
        * :data:`Mode.AUTO` / :data:`Mode.APPEND` / :data:`Mode.UPSERT`
          / :data:`Mode.IGNORE` → ``missing_ok=True``
        * :data:`Mode.ERROR_IF_EXISTS` → plain ``CREATE VIEW``
        """
        parsed_mode = Mode.from_(mode, default=Mode.AUTO)

        if or_replace is None and missing_ok is None:
            if parsed_mode == Mode.OVERWRITE:
                or_replace = True
                missing_ok = False
            elif parsed_mode == Mode.ERROR_IF_EXISTS:
                or_replace = False
                missing_ok = False
            else:
                or_replace = False
                missing_ok = True

        statement = self.create_view_ddl(
            query,
            or_replace=bool(or_replace),
            missing_ok=bool(missing_ok),
            columns=columns,
            comment=comment,
            properties=properties,
        )

        logger.debug(
            "Creating view %r (or_replace=%s, missing_ok=%s, mode=%s)",
            self, bool(or_replace), bool(missing_ok), parsed_mode.name,
        )
        try:
            self.sql.execute(statement, wait=wait)
        except Exception as exc:
            if "SCHEMA_NOT_FOUND" in str(exc):
                logger.debug(
                    "Parent schema missing for view %r — auto-creating %s.%s",
                    self, self.catalog_name, self.schema_name,
                )
                self.sql.execute(
                    f"CREATE SCHEMA IF NOT EXISTS "
                    f"{quote_ident(self.catalog_name)}.{quote_ident(self.schema_name)}",
                    wait=True,
                )
                self.sql.execute(statement, wait=wait)
            else:
                raise

        if tags:
            self.set_tags(tags)

        return self

    def concat_tables(
        self,
        tables: Iterable["Table"],
        *,
        by_name: bool = True,
        cast: bool = True,
        comment: str | None = None,
        mode: ModeLike = Mode.OVERWRITE,
    ) -> "Table":
        """Create or replace this Table as the ``UNION ALL`` of *tables*.

        When ``cast`` is ``True`` (default), the union is "smart": column
        names are aligned across inputs, types are promoted to the widest
        compatible :class:`DataType` via ``merge_with(upcast=True)``,
        each input projects the unified column list in order, and any
        column missing from a given input is emitted as
        ``CAST(NULL AS <ddl>)`` so the unified schema is preserved.

        When ``cast`` is ``False`` the method falls back to a plain
        ``SELECT * FROM <table> UNION ALL [BY NAME] ...`` and lets
        Databricks reconcile the schemas at query time.
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

        return self.create_view(query, mode=mode, comment=comment)

    @staticmethod
    def _build_smart_union_query(tables_list: list["Table"]) -> str:
        """Render a ``UNION ALL`` query projecting each input to a unified schema.

        Walks every input's ``columns``, accumulates a unified schema
        (first-seen column order, types promoted via
        ``merge_with(upcast=True)``), then projects each input to that
        column order — selecting present columns as-is and substituting
        ``CAST(NULL AS <ddl>)`` for absent ones.
        """
        from yggdrasil.data.enums.mode import Mode as _Mode

        column_order: list[str] = []
        unified: dict[str, Any] = {}
        per_table: list[dict[str, Any]] = []

        for tbl in tables_list:
            cols: dict[str, Any] = {}
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

    def delete(
        self,
        *,
        wait: WaitingConfigArg = True,
        missing_ok: bool = False,
        delete_staging: bool = True,
        delete_job: bool = True
    ) -> "Table":
        # ``delete_staging=False`` keeps the staging volume around for
        # internal drop-and-recreate flows (OVERWRITE) where the very
        # next step uploads a fresh parquet to the same volume — the
        # background ``VolumesAPI.delete`` would otherwise race the
        # upload and surface as PATH_NOT_FOUND on the warehouse INSERT.
        uc = self.client.workspace_client().tables
        logger.debug(
            "Deleting table %r (wait=%s, delete_staging=%s, delete_job=%s)",
            self, bool(wait), delete_staging, delete_job
        )

        if wait:
            try:
                uc.delete(full_name=self.full_name())

                if delete_staging and self._staging_volume:
                    self._staging_volume.delete(wait=False)

                if delete_job and self._async_job:
                    self._async_job.delete(wait=False)
            except DatabricksError:
                if not missing_ok:
                    raise
        else:
            Job.make(self.delete, delete_staging=delete_staging, delete_job=delete_job).fire_and_forget()

        self.invalidate_singleton(remove_global=True)
        logger.info("Deleted table %r", self)
        return self

    # =========================================================================
    # Rename
    # =========================================================================

    def rename(
        self,
        new_name: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
    ) -> "Table":
        """Rename this table in-place (``ALTER TABLE … RENAME TO …``).

        Accepts an unqualified name (``"new_orders"``), a two-part name
        (``"sales.new_orders"`` → cross-schema move within the same catalog),
        or a three-part name (``"main.sales.new_orders"``). Catalog/schema
        keyword overrides win over parts parsed from *new_name*.

        Unity Catalog allows cross-schema renames within the same catalog;
        moves across catalogs are rejected here with a clear error rather
        than letting the server return a generic failure.
        """
        if new_name is not None:
            parsed_c, parsed_s, parsed_t = self.sql.tables.parse_catalog_schema_table_names(new_name)
        else:
            parsed_c = parsed_s = parsed_t = None

        target_catalog = (catalog_name or parsed_c or self.catalog_name or "").strip().strip("`")
        target_schema = (schema_name or parsed_s or self.schema_name or "").strip().strip("`")
        target_table = (table_name or parsed_t or "").strip().strip("`")

        if not target_table:
            raise ValueError("Cannot rename table to an empty name")
        if not target_catalog or not target_schema:
            raise ValueError(
                f"Cannot rename {self.full_name()} — target needs a catalog and"
                f" schema (got catalog={target_catalog!r} schema={target_schema!r})"
            )
        if target_catalog != self.catalog_name:
            raise ValueError(
                f"Unity Catalog ALTER TABLE RENAME TO cannot move a table across"
                f" catalogs ({self.catalog_name!r} → {target_catalog!r}). Use"
                f" Table.clone(...) to copy across catalogs instead."
            )
        if target_schema == self.schema_name and target_table == self.table_name:
            logger.debug(
                "Skipping rename of table %r — new name matches current", self,
            )
            return self

        if target_schema == self.schema_name:
            rename_to = quote_ident(target_table)
        else:
            rename_to = f"{quote_ident(target_schema)}.{quote_ident(target_table)}"

        keyword = "VIEW" if self.is_view else "TABLE"
        logger.debug(
            "Renaming %s %r → %s.%s.%s",
            keyword, self, target_catalog, target_schema, target_table,
        )
        self.sql.execute(
            f"ALTER {keyword} {self.full_name(safe=True)} RENAME TO {rename_to}"
        )
        self.invalidate_singleton(remove_global=True)
        self.schema_name = target_schema
        self.table_name = target_table
        return self

    # =========================================================================
    # Clone
    # =========================================================================

    def clone(
        self,
        target: "str | Table | None" = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        deep: bool = True,
        replace: bool = False,
        missing_ok: bool = False,
        properties: Mapping[str, Any] | None = None,
        location: str | None = None,
        version: int | None = None,
        timestamp: "str | _dt.datetime | _dt.date | None" = None,
    ) -> "Table":
        """Clone this table to *target* via Delta ``CREATE TABLE … CLONE``.

        Emits one of::

            CREATE TABLE [IF NOT EXISTS] <target> [SHALLOW|DEEP] CLONE <source>
                [TBLPROPERTIES (...)] [LOCATION '...']
            CREATE OR REPLACE TABLE <target> [SHALLOW|DEEP] CLONE <source> ...

        Args:
            target:        Target location — :class:`Table`, a 1/2/3-part dotted
                           name, or ``None`` when *catalog_name* / *schema_name*
                           / *table_name* are passed explicitly.
            deep:          ``True`` (default) → DEEP CLONE (independent copy);
                           ``False`` → SHALLOW CLONE (metadata only, shares files).
            replace:       Emit ``CREATE OR REPLACE TABLE``.
            missing_ok: Emit ``CREATE TABLE IF NOT EXISTS``. Mutually
                           exclusive with *replace*.
            properties:    Optional ``TBLPROPERTIES`` overrides.
            location:      External storage path for the target.
            version:       Delta source version (``VERSION AS OF``).
            timestamp:     Delta source timestamp (``TIMESTAMP AS OF``).

        Returns:
            A :class:`Table` bound to this service pointing at the target.
        """
        if replace and missing_ok:
            raise ValueError("Use either replace=True or missing_ok=True, not both.")
        if version is not None and timestamp is not None:
            raise ValueError(
                "Pass either version or timestamp to clone, not both — Delta"
                " accepts one temporal anchor on the source."
            )

        tables = self.sql.tables
        if isinstance(target, Table):
            target_catalog = target.catalog_name
            target_schema = target.schema_name
            target_table = target.table_name
        else:
            parsed_c, parsed_s, parsed_t = (
                tables.parse_catalog_schema_table_names(target) if target else (None, None, None)
            )
            target_catalog = catalog_name or parsed_c or self.catalog_name
            target_schema = schema_name or parsed_s or self.schema_name
            target_table = table_name or parsed_t

        if not (target_catalog and target_schema and target_table):
            raise ValueError(
                f"Cannot clone {self.full_name()} — target needs catalog +"
                f" schema + table (got catalog={target_catalog!r}"
                f" schema={target_schema!r} table={target_table!r})"
            )
        if (
            target_catalog == self.catalog_name
            and target_schema == self.schema_name
            and target_table == self.table_name
        ):
            raise ValueError(
                f"Cannot clone {self.full_name()} onto itself — choose a"
                f" different target catalog/schema/table."
            )

        # Views can't ride the Delta ``CLONE`` path — re-emit the
        # source's ``view_definition`` as a fresh ``CREATE [OR REPLACE]
        # VIEW [IF NOT EXISTS]`` against the target, mirroring the
        # legacy :meth:`View.clone` shape.
        if self.is_view:
            select_text = (self.view_definition or "").strip().rstrip(";").strip()
            if not select_text:
                raise ValueError(
                    f"Cannot clone {self.full_name()} — source has no"
                    f" view_definition. Run ``create_view(query=...)``"
                    f" against the target directly with explicit SQL."
                )
            cloned = Table(
                service=tables,
                catalog_name=target_catalog,
                schema_name=target_schema,
                table_name=target_table,
            )
            statement = cloned.create_view_ddl(
                select_text,
                or_replace=replace,
                missing_ok=missing_ok,
                properties=properties,
            )
            logger.debug(
                "Cloning view %r → %s.%s.%s (replace=%s, missing_ok=%s)",
                self, target_catalog, target_schema, target_table,
                replace, missing_ok,
            )
            self.sql.execute(statement)
            return cloned

        target_full = (
            f"{quote_ident(target_catalog)}.{quote_ident(target_schema)}."
            f"{quote_ident(target_table)}"
        )

        if replace:
            create_kw = "CREATE OR REPLACE TABLE"
        elif missing_ok:
            create_kw = "CREATE TABLE IF NOT EXISTS"
        else:
            create_kw = "CREATE TABLE"

        source_full = self.full_name(safe=True)
        if version is not None:
            source_clause = f"{source_full} VERSION AS OF {int(version)}"
        elif timestamp is not None:
            if isinstance(timestamp, (_dt.datetime, _dt.date)):
                ts_lit = f"'{timestamp.isoformat()}'"
            else:
                ts_lit = f"'{escape_sql_string(str(timestamp))}'"
            source_clause = f"{source_full} TIMESTAMP AS OF {ts_lit}"
        else:
            source_clause = source_full

        clone_kw = "DEEP CLONE" if deep else "SHALLOW CLONE"
        sql_parts: list[str] = [
            f"{create_kw} {target_full} {clone_kw} {source_clause}",
        ]
        if properties:
            sql_parts.append(
                "TBLPROPERTIES ("
                + ", ".join(
                    f"'{escape_sql_string(str(k))}' = {sql_literal(v)}"
                    for k, v in properties.items()
                )
                + ")"
            )
        if location:
            sql_parts.append(f"LOCATION '{escape_sql_string(location)}'")

        statement = " ".join(sql_parts)
        logger.debug(
            "Cloning table %r → %s.%s.%s (deep=%s, replace=%s, missing_ok=%s)",
            self, target_catalog, target_schema, target_table,
            deep, replace, missing_ok,
        )
        self.sql.execute(statement)

        cloned = Table(
            service=tables,
            catalog_name=target_catalog,
            schema_name=target_schema,
            table_name=target_table,
        )

        return cloned

    def insert(
        self,
        data: Any,
        *,
        mode: ModeLike = None,
        match_by: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        spark_session: Optional["SparkSession"] = None,
        return_data: bool = False,
        **kwargs
    ) -> "Tabular | None":
        """Insert *data* into this table — thin wrapper over :meth:`insert_into`.

        For the deferred / drop-and-apply-later flow, see
        :meth:`async_insert`.
        """
        return self.insert_into(
            data,
            mode=mode,
            match_by=match_by,
            wait=wait,
            raise_error=raise_error,
            spark_session=spark_session,
            return_data=return_data,
            **kwargs,
        )

    def async_job(
        self,
        *,
        applier: Any = ...,
        task_type: "AsyncApplierTaskType" = "notebook",
        force: bool = False,
        **overrides: Any,
    ) -> "DatabricksJob":
        """Get-or-create the per-table applier :class:`Job` for async inserts.

        One Databricks Job per ``(catalog, schema, table)`` triple,
        watching this table's own
        ``<table>/.sql/async/insert/data/`` folder via a
        file-arrival trigger. ``**overrides`` flow into
        :meth:`AsyncInsertJob.settings` for per-deploy knobs
        (``schedule=``, ``file_arrival_trigger=``, ``parameters=``,
        …).

        :meth:`AsyncInsertJob.settings` auto-stages
        :func:`AsyncInsertJob.apply_records` as the default task —
        the source is uploaded under
        ``/Workspace/Shared/.ygg/jobs/<key>/main-<digest>.py`` and a
        matching :class:`JobEnvironment` lands on ``environments``.
        ``task_type`` picks the flavour:

        * ``"notebook"`` (default) — Databricks notebook with cells
          (imports + metadata, captured locals, the ``@checkargs``
          body, widget-driven invocation that pulls
          ``catalog_name`` / ``schema_name`` / ``table_name`` from
          the Job's parameters via ``dbutils.widgets.get``). The UI
          shows stdout / ``LOGGER`` lines under the cell that
          produced them.
        * ``"spark"`` — flat ``SparkPythonTask`` script wired with
          ``parameters=["{{job.parameters.<name>}}", …]`` so the
          rendered ``sys.argv`` reads still pick up the Job's
          parameters at run time. Single-stream logs.

        Pass ``applier=my_func`` to stage a custom callable instead,
        or ``applier=None`` to leave the job tasks-less.

        By default a pre-existing Job with the matching name
        short-circuits the deploy — useful when the same table is
        wired up from multiple processes. Pass ``force=True`` to
        always re-stage the applier and push the rebuilt settings
        through :meth:`Jobs.create_or_update` instead — the call to
        make after upgrading ``yggdrasil`` so the staged task picks
        up the latest renderer (e.g. the notebook conversion replaces
        a previously-staged ``SparkPythonTask`` whose ``apply_records()``
        invocation can't see the job's ``{{job.parameters.*}}``
        bindings).
        """
        if self._async_job is not None:
            return self._async_job

        from .async_job import AsyncInsertJob

        jobs = self.client.jobs

        if not force:
            # Pre-check before staging the applier — an existing job
            # short-circuits the workspace write entirely.
            prelim_name = AsyncInsertJob.job_name(self)
            found = jobs.find(name=prelim_name)
            if found is not None:
                return found

        settings = AsyncInsertJob.settings(
            self, applier=applier, task_type=task_type, **overrides,
        )
        name = settings.pop("name")
        self._async_job = jobs.create_or_update(name=name, **settings)
        return self._async_job

    def async_insert(
        self,
        data: Any,
        *,
        mode: ModeLike = None,
        match_by: Optional[list[str]] = None,
        require_job: bool = True,
        **kwargs,
    ) -> "AsyncInsert":
        """Stage *data* as an async insert and return the metadata record.

        Rows are cast to the target schema and dropped (alongside a
        JSON metadata file describing the operation) under the
        table's ``<table>/.sql/async/insert`` staging folder for a
        downstream applier to pick up; the SQL insert is *not*
        executed. The constructed :class:`AsyncInsert` is itself a
        :class:`WarehouseStatementBatch`, so binding an executor and
        submitting is a single ``.execute(engine)`` call. The caller
        can also ``merge_with`` peers, or schedule the apply via the
        per-table :class:`AsyncInsertJob` in :mod:`.async_job`. See
        :mod:`.async_write` for the wire format.

        ``require_job`` (default ``True``) ensures the per-table
        applier Job exists before any staging round trip — without one,
        staged payloads sit in the table's
        ``<table>/.sql/async/insert/`` folder forever with no
        consumer. The check rides through :meth:`Table.async_job`,
        whose :meth:`Jobs.find` lookup caches the ``name → job_id``
        mapping for 60 s; the steady-state cost is sub-millisecond.
        A missing job is auto-deployed via :meth:`Table.async_job`
        with default settings — pass ``require_job=False`` to skip the
        check entirely (e.g. seeding payloads before
        :meth:`Table.async_job` deploys a tuned applier from a
        different process).
        """
        from .async_write import stage_async_insert

        if require_job:
            self.async_job()

        return stage_async_insert(
            self,
            data,
            mode=mode,
            match_by=match_by,
            lazy=True,
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
        update_column_names: Optional[list[str]] = None,
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
        return_data: bool = False,
        safe_merge: bool = False,
    ) -> "StatementBatch | Tabular | None":
        """Insert *data* into this table using the most appropriate backend.

        Routing:

        - Query-shaped sources (str, ``PreparedStatement``,
          ``StatementResult``) → :meth:`sql_insert`
        - Spark DataFrame (or anything when a ``SparkSession`` is reachable)
          → :meth:`spark_insert`
        - Otherwise → :meth:`arrow_insert` (warehouse path with Volume staging)

        Returns the submitted :class:`StatementBatch` by default. With
        ``return_data=True`` the backend that ran the write hands back
        its source payload as a :class:`Tabular` —
        :class:`ArrowTabular` from :meth:`arrow_insert`,
        :class:`Dataset` from :meth:`spark_insert`, the input
        :class:`StatementResult` from :meth:`sql_insert` — for
        downstream chaining without re-querying the target.
        """
        common = dict(
            mode=mode,
            match_by=match_by,
            update_column_names=update_column_names,
            wait=wait,
            raise_error=raise_error,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            where=where,
            prune_by=prune_by,
            prune_values=prune_values,
            retry=retry,
            return_data=return_data,
            safe_merge=safe_merge,
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

    @property
    def staging_volume(self):
        if self._staging_volume is None:
            if not self.catalog_name or not self.schema_name or not self.table_name:
                raise ValueError(f"Table {self} is missing required catalog, schema, or table name")

            self._staging_volume = Volume(
                service=self.service.volumes,
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                volume_name=self.client.safe_tag_value(self.table_name, repl="_").lower()
            )
        return self._staging_volume

    def staging_folder(
        self,
        temporary: bool = False,
        async_write: bool = False,
    ) -> VolumePath:
        """Return the staging folder for this table."""
        if async_write:
            return self.staging_volume.path(".sql/async/insert", temporary=temporary)
        else:
            return self.staging_volume.path(".sql/tmp", temporary=temporary)

    def insert_volume_path(
        self,
        target: "Table | None" = None,
        *,
        temporary: bool = True,
    ) -> VolumePath:
        """Mint a fresh Parquet staging path under the target table's
        :attr:`staging_volume`.

        Roots the file at ``<staging_volume>/.sql/tmp/tmp-<epoch_ms>-<seed>.parquet``
        (same shape as :meth:`staging_folder` but with a unique leaf
        per call). ``target`` defaults to ``self``; pass another
        :class:`Table` when the staging hierarchy needs to live next
        to a different table (e.g. dispatch fan-out). Lifted out of
        :meth:`arrow_insert` so callers — and tests — can pre-mint or
        swap the staging location without driving the full insert.
        """
        target = target if target is not None else self
        seed = uuid.uuid4().hex[:8]
        leaf = f"tmp-{int(time.time() * 1000)}-{seed}.parquet"
        return target.staging_volume.path(
            f".sql/tmp/{leaf}",
            temporary=temporary,
        )

    def arrow_insert(
        self,
        data,
        *,
        engine: Literal["api", "spark"] | None = None,
        mode: Mode | str | None = None,
        schema_mode: Mode | str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_column_names: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        where: Predicate | None = None,
        prune_by: list[str] | str | None = None,
        prune_values: Mapping[str, list[Any]] | None = None,
        retry: Optional[WaitingConfigArg] = None,
        return_data: bool = False,
        safe_merge: bool = False,
    ) -> "StatementBatch | Tabular | None":
        """Insert through the warehouse SQL path with staged Parquet.

        ``safe_merge`` controls keyed-write strategy:

        * ``safe_merge=False`` (default) — emits a single ``MERGE
          INTO`` statement. Databricks / Delta plans the keyed dedup
          once.
        * ``safe_merge=True`` — sidesteps MERGE: keyed APPEND becomes
          ``INSERT ... WHERE NOT EXISTS (...)``, keyed UPSERT becomes
          ``DELETE`` matching keys then ``INSERT``. Useful for
          backends without native MERGE or callers that want explicit
          dedup semantics.

        Returns the submitted :class:`StatementBatch` by default. With
        ``return_data=True``, returns an :class:`ArrowTabular` wrapping
        the staged source rows so callers can chain on the payload
        without re-reading from the target.
        """
        from yggdrasil.databricks.warehouse import WarehousePreparedStatement

        if isinstance(data, (PreparedStatement, StatementResult)) or PreparedStatement.looks_like_query(data):
            return self.sql_insert(
                data,
                mode=mode,
                match_by=match_by, update_column_names=update_column_names,
                wait=wait, raise_error=raise_error,
                zorder_by=zorder_by, optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                where=where, prune_by=prune_by,
                retry=retry,
                return_data=return_data,
            )

        mode_enum = Mode.from_(mode, default=Mode.AUTO)

        target = self.create(
            data,
            mode=schema_mode,
            or_replace=(mode_enum == Mode.OVERWRITE and not match_by),
        )
        target_location = target.full_name(safe=True)
        existing_schema = target.collect_schema()
        cast_options = CastOptions.check(options=cast_options).with_target(existing_schema)

        if match_by == "auto":
            match_by = [f.name for f in existing_schema.primary_fields] or None
        prune_by = _resolve_prune_by(prune_by, existing_schema.partition_fields)

        wait = WaitingConfig.from_(wait)
        staging = self.insert_volume_path(target, temporary=bool(wait))

        prune_values = prune_values or {}
        output_data: "Tabular | None" = None
        staging.write_table(data, cast_options, mode=Mode.OVERWRITE)
        if return_data:
            output_data = staging.read_arrow_table()

        prune_predicates = _build_prune_predicate(
            prune_values, where, target_alias="T",
        )

        columns = list(existing_schema.field_names())
        # Plain column projection — the staged Parquet was already
        # cast to the target schema in :meth:`CastOptions.cast_arrow_tabular`
        # before the write, and the warehouse INSERT applies the
        # column-boundary coercion on top.  No per-column CAST needed.
        source_projection = _build_column_projection(existing_schema.fields)
        source_sql = f"SELECT {source_projection} FROM {{{_ALIAS_TMPSRC}}}"

        sql_texts = _build_dml_statements(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            mode=mode_enum,
            match_by=match_by,
            update_column_names=update_column_names,
            prune_predicates=prune_predicates,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            safe_merge=safe_merge,
        )

        retry_active = retry is not None

        def _prepare_batch(texts: list[str]) -> list[WarehousePreparedStatement]:
            out: list[WarehousePreparedStatement] = []
            for sql in texts:
                external_data = (
                    {_ALIAS_TMPSRC: staging}
                    if (f"{{{_ALIAS_TMPSRC}}}" in sql)
                    else None
                )
                stmt = WarehousePreparedStatement.prepare(
                    sql,
                    client=self.client,
                    external_data=external_data,
                    catalog_name=target.catalog_name,
                    schema_name=target.schema_name,
                )
                if retry_active and _classify_dml(sql):
                    _apply_retry_to_warehouse_statement(stmt, retry)
                out.append(stmt)
            return out

        prepared = _prepare_batch(sql_texts)

        logger.debug(
            "Arrow insert into table %r (mode=%s, match_by=%s, prune_by=%s, "
            "statements=%d, retry=%s)",
            target_location, mode_enum.name, match_by, prune_by, len(prepared),
            retry_active,
        )

        batch = self.sql.execute_many(
            statements=prepared,
            wait=wait,
            raise_error=raise_error,
            engine=engine,
        )

        return output_data if return_data else batch

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
        update_column_names: Optional[list[str]] = None,
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
        return_data: bool = False,
        safe_merge: bool = False,
    ) -> "StatementBatch | Tabular | None":
        """Insert into this table using Spark.

        ``retry`` is applied to DML statements (INSERT/MERGE/DELETE/UPDATE)
        only — TRUNCATE/OPTIMIZE/VACUUM stay non-retryable.
        :class:`SparkStatementResult` already auto-promotes transient
        Delta failures (``ConcurrentAppendException``, …) to retryable;
        passing ``retry=True`` (or any :class:`WaitingConfig` arg) makes
        the policy explicit instead of relying on auto-promote.

        Returns the submitted :class:`StatementBatch` by default. With
        ``return_data=True``, returns a :class:`Dataset` wrapping
        the materialised source DataFrame — handy for chaining
        downstream transforms without re-querying the target.
        """
        if isinstance(data, (PreparedStatement, StatementResult)) or PreparedStatement.looks_like_query(data):
            return self.sql_insert(
                data,
                mode=mode,
                match_by=match_by, update_column_names=update_column_names,
                wait=wait, raise_error=raise_error,
                zorder_by=zorder_by, optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                where=where, prune_by=prune_by,
                spark_session=spark_session,
                retry=retry,
                return_data=return_data,
            )

        from yggdrasil.spark.cast import any_to_spark_dataframe
        from yggdrasil.spark.statement import SparkPreparedStatement

        mode_enum = Mode.from_(mode, default=Mode.AUTO)

        target = self.create(
            data,
            mode=schema_mode,
            or_replace=(mode_enum == Mode.OVERWRITE and not match_by),
        )
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
            # Cache before the distinct().collect() so the temp view backing
            # the MERGE doesn't re-execute the source plan from scratch.
            data_df = data_df.cache()
            prune_values = _collect_prune_values_spark(data_df, prune_by)
            logger.debug(
                "Spark pruning %s -> %s",
                prune_by, {k: len(v) for k, v in prune_values.items()},
            )

        prune_predicates = _build_prune_predicate(
            prune_values, where, target_alias="T",
        )

        # Spark fast path for keyed APPEND under ``safe_merge=True``
        # (see :func:`_spark_filter_existing_keys`). Catalyst's
        # anti-join only reads the target's key columns from disk,
        # so this is dramatically cheaper than the SQL NOT EXISTS
        # shape used on the warehouse path. ``safe_merge=False``
        # leaves the work to a native MERGE INTO statement.
        anti_join_handled = False
        if (
            safe_merge
            and match_by
            and mode_enum in (Mode.APPEND, Mode.AUTO)
        ):
            data_df, anti_join_handled = _spark_filter_existing_keys(
                session=session,
                data_df=data_df,
                target_location=target_location,
                match_by=list(match_by),
            )

        view_name = f"_yg_src_{uuid.uuid4().hex}"
        data_df.createOrReplaceTempView(view_name)

        columns = list(existing_schema.field_names())
        cols_quoted = ", ".join(quote_ident(c) for c in columns)
        # Plain column projection — :func:`any_to_spark_dataframe`
        # already aligned the DataFrame to the target schema, and the
        # INSERT itself applies the column-boundary coercion, so the
        # SQL stays free of per-column CASTs.
        source_sql = f"SELECT {cols_quoted} FROM {quote_ident(view_name)}"

        # The DataFrame anti-join already dedup'd; emit a plain INSERT
        # so we don't pay for the SQL-side NOT EXISTS twice.
        effective_mode = (
            Mode.OVERWRITE if anti_join_handled else mode_enum
        )
        effective_match_by = None if anti_join_handled else match_by
        # OVERWRITE-with-no-match_by would normally trigger a target
        # delete up front; skip that — the fast path is *append*.
        if anti_join_handled:
            sql_texts = [
                f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}"
            ]
            _append_maintenance_statements(
                sql_texts,
                target_location=target_location,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                keyed=True,
                vacuum_hours=vacuum_hours,
            )
        else:
            sql_texts = _build_dml_statements(
                target_location=target_location,
                source_sql=source_sql,
                columns=columns,
                mode=effective_mode,
                match_by=effective_match_by,
                update_column_names=update_column_names,
                prune_predicates=prune_predicates,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                safe_merge=safe_merge,
            )

        retry_cfg = _resolve_retry(retry)

        def _prepare_spark_batch(texts: list[str]) -> list[SparkPreparedStatement]:
            out: list[SparkPreparedStatement] = []
            for sql in texts:
                stmt = SparkPreparedStatement(text=sql, spark_session=session)
                if retry_cfg is not None and _classify_dml(sql):
                    _apply_retry_to_statement(stmt, retry_cfg)
                out.append(stmt)
            return out

        prepared = _prepare_spark_batch(sql_texts)

        logger.debug(
            "Inserting via Spark into table %r (mode=%s, match_by=%s, prune_by=%s, "
            "statements=%d, retry=%s, anti_join=%s)",
            target_location, mode_enum.name, match_by, prune_by, len(prepared),
            retry_cfg is not None, anti_join_handled,
        )

        applied_conf = _delta_conf_for(overwrite_schema, spark_options)

        primary_batch = None
        try:
            with sql_engine.spark.scoped_spark_conf(session, applied_conf):
                primary_batch = _execute_dml(
                    sql_engine,
                    statements=prepared,
                    wait=wait,
                    raise_error=raise_error,
                    engine="spark",
                )
            logger.info(
                "Inserted via Spark into table %r (mode=%s, match_by=%s, "
                "prune_by=%s, statements=%d, anti_join=%s)",
                target_location, mode_enum.name, match_by, prune_by, len(prepared),
                anti_join_handled,
            )
        finally:
            try:
                session.catalog.dropTempView(view_name)
            except Exception:
                logger.debug("Failed to drop temp view %r; continuing.", view_name, exc_info=True)
            if prune_by and not return_data:
                # Keep the cached source alive when the caller asked
                # for it back — :class:`Dataset` is the consumer
                # and unpersisting here would force a re-execution
                # downstream.
                try:
                    data_df.unpersist()
                except Exception:
                    logger.debug("Failed to unpersist cached source; continuing.", exc_info=True)

        if return_data:
            from yggdrasil.spark.tabular import Dataset
            return Dataset(data_df)
        return primary_batch

    # =========================================================================
    # sql_insert — query source, no staging
    # =========================================================================

    def sql_insert(
        self,
        statement: "PreparedStatement | StatementResult | str",
        *,
        mode: Mode | str | None = None,
        match_by: Optional[list[str]] = None,
        update_column_names: Optional[list[str]] = None,
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
        return_data: bool = False,
        safe_merge: bool = False,
    ) -> "StatementBatch | Tabular | None":
        """Insert into this table from a SQL source query.

        Smart dispatch:

        1. Cached :class:`StatementResult` → reuse the materialised frame
           via :meth:`insert_into` (no re-execution).
        2. SparkSession reachable → run via :meth:`spark_insert`.
        3. Otherwise → warehouse-side ``INSERT … SELECT`` /
           ``MERGE … USING (q)`` with a CAST projection aligning the
           user's query schema to the target.

        Returns the submitted :class:`StatementBatch` by default
        (warehouse fallback) or the chosen backend's batch (Arrow /
        Spark). With ``return_data=True``, hands back the underlying
        :class:`StatementResult` (or the materialised frame from a
        cached one) so callers can stream the same rows the warehouse
        just inserted.
        """
        common = dict(
            mode=mode,
            match_by=match_by, update_column_names=update_column_names,
            wait=wait, raise_error=raise_error,
            zorder_by=zorder_by, optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            where=where, prune_by=prune_by, prune_values=prune_values,
            retry=retry,
        )

        if isinstance(statement, StatementResult):
            spark_df = getattr(statement, "spark_dataframe", None)
            cached = spark_df if spark_df is not None else statement.to_arrow_table()
            return self.insert_into(
                data=cached, spark_session=spark_session,
                return_data=return_data, **common,
            )

        if spark_session is None:
            spark_session = self.sql.spark.resolve_session(create=False)
        if spark_session is not None:
            text = (
                statement.statement.text
                if isinstance(statement, StatementResult)
                else (statement.text if isinstance(statement, PreparedStatement) else str(statement))
            )
            df = spark_session.sql(text)
            return self.spark_insert(
                data=df, spark_session=spark_session,
                return_data=return_data, **common,
            )

        batch = self._sql_insert_warehouse_fallback(statement, **common)
        if return_data and isinstance(statement, StatementResult):
            # The warehouse path doesn't materialise rows on its own,
            # but the caller's :class:`StatementResult` is already a
            # :class:`Tabular` over the same source query — hand it
            # back so ``return_data=True`` stays consistent across paths.
            return statement
        return batch

    def _sql_insert_warehouse_fallback(
        self,
        statement: "PreparedStatement | StatementResult | str",
        *,
        engine: Optional[Literal["api", "spark"]] = None,
        mode: Mode | str | None,
        match_by: Optional[list[str]],
        update_column_names: Optional[list[str]],
        wait: WaitingConfigArg,
        raise_error: bool,
        zorder_by: Optional[list[str]],
        optimize_after_merge: bool,
        vacuum_hours: int | None,
        where: Predicate | None,
        prune_by: list[str] | str | None,
        prune_values: dict[str, tuple[Any]] | None = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> "StatementBatch | None":
        """Warehouse fallback for :meth:`sql_insert`."""
        from yggdrasil.databricks.warehouse import WarehousePreparedStatement

        base = statement.statement if isinstance(statement, StatementResult) else statement
        source_prepared = WarehousePreparedStatement.from_(base)

        mode_enum = Mode.from_(mode, default=Mode.AUTO)

        if mode_enum == Mode.OVERWRITE and not match_by:
            self.delete(wait=True, missing_ok=True, delete_staging=False, delete_job=False)

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

        source_projection = _build_column_projection(fields, source_alias="raw_src")
        source_sql = (
            f"SELECT {source_projection} FROM (\n{source_prepared.text}\n) AS raw_src"
        )

        prune_predicates = _build_prune_predicate(
            None, where, target_alias="T",
        )
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
            update_column_names=update_column_names,
            prune_predicates=prune_predicates,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
        )

        retry_active = retry is not None

        def _prepare_batch(texts: list[str]) -> list[WarehousePreparedStatement]:
            out: list[WarehousePreparedStatement] = []
            for sql in texts:
                stmt = WarehousePreparedStatement.prepare(
                    sql,
                    parameters=source_prepared.parameters,
                    external_volume_paths=source_prepared.external_volume_paths,
                    catalog_name=self.catalog_name,
                    schema_name=self.schema_name,
                )
                if retry_active and _classify_dml(sql):
                    _apply_retry_to_warehouse_statement(stmt, retry)
                out.append(stmt)
            return out

        prepared = _prepare_batch(sql_texts)

        logger.info(
            "SQL insert into table %r (mode=%s, match_by=%s, statements=%d, retry=%s)",
            target_location, mode_enum.name, match_by, len(prepared), retry_active,
        )

        if not prepared:
            return None

        return _execute_dml(
            self.sql,
            statements=prepared,
            wait=wait,
            raise_error=raise_error,
            engine=engine,
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

    def storage_location(
        self,
        operation: "TableOperation | ModeLike | None" = None,
    ) -> "Path":
        """Return the table's backing storage location as an addressable :class:`Path`.

        ``operation`` accepts a :class:`TableOperation`, a :class:`Mode`,
        a :class:`Mode`-like string (``"read"``, ``"overwrite"``,
        ``"append"``, …), or ``None``.

        Resolution:

        - ``None`` (the default) picks ``READ`` for managed tables
          (Unity Catalog only ever vends read creds for those) and
          ``READ_WRITE`` for external tables.
        - A :class:`TableOperation` is used as-is.
        - A :class:`Mode` / string is mapped to ``READ`` for read-only
          modes and ``READ_WRITE`` otherwise; managed tables still
          collapse to ``READ`` because Unity Catalog will refuse
          to vend write credentials for them.
        """
        op = _resolve_table_operation(operation, self.infos.table_type)
        return self.aws(operation=op).s3.path(self.infos.storage_location)

    def aws(
        self,
        operation: "TableOperation | ModeLike | None" = None,
        *,
        region: Optional[str] = None,
    ) -> "AWSClient":
        """Return an :class:`AWSClient` whose credentials self-refresh
        from Unity Catalog's ``temporary_table_credentials`` API.

        Routes through :meth:`credentials_refresher` — every
        :class:`Table` instance pointing at the same UC table id
        collapses to one provider that handles both read and write
        modes internally. The provider caches its :class:`AWSClient`
        per ``(mode, region)`` so the boto session,
        :class:`RefreshableCredentials`, connection pool, and STS
        vending are shared across every caller on the same scope.

        ``operation`` accepts a :class:`TableOperation`, a
        :class:`Mode` / mode-like string, or ``None`` (defaults to the
        right operation for this table's type).
        """
        op = _resolve_table_operation(operation, self.infos.table_type)
        mode = Mode.READ_ONLY if op == TableOperation.READ else Mode.OVERWRITE
        return self.credentials_refresher().aws_client(mode=mode, region=region)

    def credentials_refresher(self) -> "AWSDatabricksTableCredentials":
        """Return the process-wide singleton credentials provider for
        this table.

        Keyed by ``table_id``; handles both read and write modes
        internally via :meth:`AWSDatabricksTableCredentials.get_credentials`.
        """
        from yggdrasil.databricks.aws import AWSDatabricksTableCredentials

        return AWSDatabricksTableCredentials(
            table_id=self.table_id,
            client=self.client,
        )

    def temporary_credentials(self, operation: TableOperation = TableOperation.READ):
        return (
            self.client.workspace_client()
            .temporary_table_credentials
            .generate_temporary_table_credentials(
                table_id=self.table_id,
                operation=operation,
            )
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
