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

import datetime as _dt
import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union, TYPE_CHECKING, Mapping, Iterable, Iterator, Literal

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
from yggdrasil.data.data_utils import safe_constraint_name
from yggdrasil.io.tabular.execution.expr import Predicate, col as expr_col
from yggdrasil.io.tabular.execution.expr.backends.sql import Dialect, to_sql as expr_to_sql
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema as DataSchema
from yggdrasil.data.statement import PreparedStatement, StatementResult
from yggdrasil.databricks.client import DatabricksClient, DatabricksResource
from yggdrasil.dataclasses.expiring import Expiring, RefreshResult
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io import URL
from yggdrasil.io.holder import Holder
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.io.tabular import Tabular, O
from yggdrasil.data.enums import Scheme
from yggdrasil.io.primitive import ParquetIO
from yggdrasil.data.enums import MimeTypes, MimeType, MediaType
from yggdrasil.data.enums.mode import ModeLike, Mode
from yggdrasil.lazy_imports import aws_config_class

from .column import Column
from .sql_utils import (
    quote_ident,
    quote_qualified_ident,
    safe_table_name,
    sql_literal, escape_sql_string,
)
from ..fs import VolumePath

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
    from yggdrasil.io.nested import DeltaIO
    from yggdrasil.io.path import Path


# Modes that only need read credentials. Anything outside this set
# (OVERWRITE, APPEND, UPSERT, MERGE, TRUNCATE, ERROR_IF_EXISTS) needs
# READ_WRITE — UC will refuse the write path on a managed table, but
# we still ask for the right scope so an external table gets one.
_READ_ONLY_MODES = frozenset({Mode.AUTO})


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
    "YGG_PROPERTY_PREFIX",
    "YGG_SCHEMA_FIELD_PREFIX",
    "YGG_SCHEMA_FIELD_SUFFIX",
]

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
    engine_name: Literal["api", "spark"],
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
        statements, wait=wait, raise_error=False, engine=engine_name,
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


def _build_prune_predicates(
    prune_values: Mapping[str, Iterable[Any]],
    *,
    target_alias: str,
) -> list[str]:
    """Convert ``{column: [values]}`` into target-side ``IN`` predicates."""
    predicates: list[str] = []
    for column_name, vals in prune_values.items():
        materialized = tuple(vals)
        if not materialized:
            continue
        pred = expr_col(column_name, alias=target_alias).is_in(materialized)
        sql = expr_to_sql(pred, dialect=Dialect.DATABRICKS)
        # Compound predicates get wrapped so they don't bleed into
        # the surrounding AND/OR structure when concatenated by
        # the caller. The new ``InList`` is always a single leaf,
        # but empty / null-aware variants render to a compound
        # ``... OR ... IS NULL`` form — wrap defensively.
        if " OR " in sql or " AND " in sql:
            sql = f"({sql})"
        predicates.append(sql)
    return predicates


def _wrap_user_predicate(pred: Predicate, *, target_alias: str) -> str:
    """Render a user predicate aliased to ``target_alias`` (parens if compound)."""
    aliased = _alias_columns(pred, target_alias)
    sql = expr_to_sql(aliased, dialect=Dialect.DATABRICKS)
    if " OR " in sql or " AND " in sql:
        sql = f"({sql})"
    return sql


def _alias_columns(expr, alias: str):
    """Walk *expr* and stamp every :class:`Column` with ``alias``.

    Used by :func:`_wrap_user_predicate` to lift a user predicate
    onto the target-side of a MERGE so ``foo`` becomes ``T.foo``.
    Returns a new tree — the AST is immutable so we never mutate
    the caller's predicate.
    """
    from yggdrasil.io.tabular.execution.expr.nodes import (
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
        return type(expr)(name=expr.name, field=expr.field, alias=alias)
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
    buffer: ParquetIO,
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


# ---------------------------------------------------------------------------
# table_dispatch — multi-target insert fan-out
#
# A ``table_dispatch`` mapping lets a single insert call also push the same
# source rows into additional Delta tables, with each target getting its own
# row-level :class:`Predicate` applied to the source as a ``WHERE`` filter.
# Resolution accepts either a built :class:`Table` or a dotted location
# string for keys, and a :class:`Predicate`, raw SQL string, or any other
# engine expression :meth:`Predicate.from_` knows how to lift for values.
# ---------------------------------------------------------------------------


def _resolve_dispatch_targets(
    dispatch: "Mapping[Table | str, Predicate | str] | None",
    *,
    primary: "Table",
) -> list[tuple["Table", Predicate]]:
    """Normalize ``table_dispatch`` into ``[(Table, Predicate), ...]``.

    String keys go through :meth:`Table.from_` against the primary's
    own ``Tables`` service so callers can pass either a fully-built
    handle or a dotted location like ``"cat.sch.name"``. String values
    (or anything else :meth:`Predicate.from_` can lift — Polars / Arrow /
    Spark expressions) are coerced to a yggdrasil :class:`Predicate`
    here so the rendering path always sees the canonical AST.
    A self-dispatch (an entry that resolves to the primary itself)
    raises — the primary insert already covers that target and
    silently double-inserting would surprise everyone.
    """
    if not dispatch:
        return []
    primary_loc = primary.full_name() if primary.table_name else None
    out: list[tuple[Table, Predicate]] = []
    for key, predicate in dispatch.items():
        if isinstance(key, str):
            target = Table.from_(obj=key, service=primary.service)
        elif isinstance(key, Table):
            target = key
        else:
            raise TypeError(
                f"table_dispatch keys must be Table or str (location); "
                f"got {type(key).__name__}. "
                f"Pass a Table handle or a dotted 'cat.sch.tbl' location."
            )
        if (
            primary_loc
            and target.table_name
            and target.full_name() == primary_loc
        ):
            raise ValueError(
                f"table_dispatch entry {key!r} resolves to the primary "
                f"target {primary_loc!r}; the primary insert already "
                f"covers it. Drop the entry or point it at a different table."
            )
        # Coerce SQL strings and engine-native expressions into the
        # canonical Predicate AST. ``Predicate.from_`` is forgiving on
        # input: existing Predicate passes through, str → from_sql,
        # polars/pyarrow/pyspark → matching from_*. Anything it can't
        # lift surfaces with a clear "give me one of: …" error.
        if not isinstance(predicate, Predicate):
            predicate = Predicate.from_(predicate)
        out.append((target, predicate))
    return out


def _render_source_predicate(predicate: "Predicate | None") -> str:
    """Render a source-side ``WHERE`` fragment for a dispatch predicate.

    Columns render unaliased so the fragment composes onto whatever
    source projection the caller built (staged Parquet read, temp view,
    sub-query). Compound expressions get parenthesized so AND/OR
    nesting stays explicit when the fragment is concatenated.
    Returns ``""`` when the predicate is ``None``.
    """
    if predicate is None:
        return ""
    sql = expr_to_sql(predicate, dialect=Dialect.DATABRICKS)
    if " OR " in sql or " AND " in sql:
        sql = f"({sql})"
    return sql


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
# ``ygg.*`` TBLPROPERTIES — shared by sql_create and api_create
# ---------------------------------------------------------------------------

YGG_PROPERTY_PREFIX = "ygg."
# Per-field schema dump key shape: ``ygg.schema[<field_name>]``. Brackets
# wrap the field name so identifiers containing ``.`` don't collide with
# the property-namespace separator (``ygg.schema.user.first_name`` would
# otherwise be ambiguous between a field named ``user.first_name`` and a
# nested ``user`` field with a ``first_name`` child).
YGG_SCHEMA_FIELD_PREFIX = "ygg.schema["
YGG_SCHEMA_FIELD_SUFFIX = "]"


def _ygg_schema_key(name: str) -> str:
    """Build the ``ygg.schema[<name>]`` TBLPROPERTIES key for a field."""
    return f"{YGG_SCHEMA_FIELD_PREFIX}{name}{YGG_SCHEMA_FIELD_SUFFIX}"


def _resolve_format_mime(
    data_source_format: "DataSourceFormat | str | None",
) -> MimeType:
    """Map a Databricks ``DataSourceFormat`` onto a yggdrasil :class:`MimeType`.

    Databricks ``DataSourceFormat`` is the storage flavor (DELTA,
    PARQUET, AVRO, …); yggdrasil already carries IANA-ish ``MimeType``
    descriptors for the ones with a recognized type. We prefer the
    yggdrasil categorization on table properties so the full ygg stack
    (loaders, codecs, format dispatch) can speak one vocabulary.

    Unresolvable formats — Databricks-specific connectors like
    ``UNITY_CATALOG`` / ``DELTASHARING`` / ``BIGQUERY_FORMAT`` —
    collapse to :data:`MimeTypes.DATABRICKS_UNITY_CATALOG_TABLE`,
    which is also :meth:`Table.default_media_type`.
    """
    if data_source_format is None:
        return MimeTypes.DATABRICKS_UNITY_CATALOG_TABLE
    name = (
        data_source_format.value
        if hasattr(data_source_format, "value")
        else str(data_source_format)
    )
    return MimeType.from_(name, default=None) or MimeTypes.DATABRICKS_UNITY_CATALOG_TABLE


def _build_ygg_properties(
    schema_info: DataSchema,
    *,
    engine: Literal["sql", "api"],
    data_source_format: "DataSourceFormat | str | None" = None,
    table_type: "TableType | str | None" = None,
    storage_location: str | None = None,
) -> dict[str, str]:
    """Build the ``ygg.*`` TBLPROPERTIES that yggdrasil stamps on every create.

    The same dict is emitted by both create paths
    (:meth:`Table.sql_create` and :meth:`Table.api_create`) so the two
    surfaces stay observable in the same way.

    Top-level data fields are dumped one-per-property under
    ``ygg.schema[<field_name>]`` (each value is a JSON document for that
    field) rather than as a single ``ygg.schema_json`` blob. Per-field
    keys keep individual TBLPROPERTIES values comfortably under
    Databricks' per-property size budget on wide schemas, and let
    readers fetch only the columns they care about. The bracket wrap
    keeps names containing ``.`` unambiguous.

    The fingerprint is a short blake2b digest of the *full* schema
    JSON so a reader can detect schema drift without re-assembling the
    per-field payloads.

    The storage flavor goes out as ``ygg.mime_type`` — resolved via
    :func:`_resolve_format_mime` so a yggdrasil :class:`MimeType` is
    preferred over the raw Databricks ``DataSourceFormat`` enum string.
    """
    from yggdrasil.version import __version__ as ygg_version

    digest = hashlib.blake2b(
        schema_info.to_json(to_bytes=False).encode("utf-8"),
        digest_size=16,
    ).hexdigest()

    data_fields = [
        f for f in schema_info.children_fields
        if not getattr(f, "constraint_key", False)
    ]

    mime = _resolve_format_mime(data_source_format)

    props: dict[str, str] = {
        "ygg.version": str(ygg_version),
        "ygg.created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "ygg.engine": engine,
        "ygg.mime_type": mime.value,
        "ygg.field_count": str(len(data_fields)),
        "ygg.schema_fingerprint": digest,
    }
    if table_type is not None:
        props["ygg.table_type"] = (
            table_type.value if hasattr(table_type, "value") else str(table_type)
        )
    if storage_location:
        props["ygg.storage_location"] = str(storage_location)

    partition_names = [f.name for f in schema_info.partition_fields]
    cluster_names = [f.name for f in schema_info.cluster_fields]
    primary_names = [f.name for f in schema_info.primary_fields]

    if partition_names:
        props["ygg.partition_columns"] = ",".join(partition_names)
    if cluster_names:
        props["ygg.cluster_columns"] = ",".join(cluster_names)
    if primary_names:
        props["ygg.primary_keys"] = ",".join(primary_names)

    comment = schema_info.comment
    if comment:
        props["ygg.comment"] = comment

    # Per-field JSON entries — one TBLPROPERTIES key per top-level data
    # field. Constraint-only fields (FK/CHECK rows on ``schema.constraints``)
    # are skipped: they're applied via the SDK constraints API and aren't
    # columns the table actually carries.
    seen: set[str] = set()
    for f in data_fields:
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

class Table(DatabricksResource, Holder):
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
    """

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_TABLE

    @classmethod
    def options_class(cls) -> type[CastOptions]:
        return CastOptions

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
    ):
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
    # Tabular
    # ------------------------------------

    @classmethod
    def default_media_type(cls) -> MimeType:
        return MimeTypes.DATABRICKS_UNITY_CATALOG_TABLE

    @property
    def cached(self) -> bool:
        return True

    def unpersist(self) -> None:
        pass

    def persist(self, engine: Literal["arrow", "polars", "spark", "auto"] = "auto", *,
                data: Any | None = None) -> "Tabular":
        return self

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

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        options = options.with_source(source=self.collect_schema())
        safe_char = "`"
        names = ",".join(
            safe_char + name + safe_char
            for name in options.column_names or [c.name for c in self.columns]
        )
        query = f"SELECT {names}"
        # The unified ``predicate`` on CastOptions becomes a SQL
        # ``WHERE`` so the warehouse drops non-matching rows before
        # they reach the cast pipeline. Pushdown is the whole point —
        # round-tripping rows through Arrow just to filter them on
        # the driver wastes a round trip per batch.
        if options.predicate is not None:
            query += (
                f" WHERE "
                f"{expr_to_sql(options.predicate, dialect=Dialect.DATABRICKS)}"
            )

        for batch in self.execute(query).read_arrow_batches(options=options):
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
        return self.sql.execute(
            f"SELECT * FROM {self.full_name(safe=True)}"
        ).read_spark_frame(options)

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
        storage_location: str | None = None,
        comment: str | None = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        if_not_exists: bool = True,
        record_ygg_properties: bool = True,
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
                if_not_exists=if_not_exists,
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

        for f in schema_info.children_fields:
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
        constraint_clauses = self._build_inline_constraints(
            self.full_name(safe=False), primary_keys
        )

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
        if record_ygg_properties:
            props.update(_build_ygg_properties(
                schema_info,
                engine="sql",
                data_source_format=data_source_format,
                table_type=TableType.EXTERNAL if storage_location else TableType.MANAGED,
                storage_location=storage_location,
            ))
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
            elif "CONSTRAINT_ALREADY_EXISTS_IN_SCHEMA" in str(exc):
                pass
            else:
                raise

        self._reset_cache(invalidate_cache=True)

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
        if if_not_exists and self.exists:
            return self

        schema_info = DataSchema.from_any(definition).autotag()
        comment = comment or schema_info.comment

        effective_fields: list[Field] = []
        column_infos: list[ColumnInfo] = []
        for position, f in enumerate(schema_info.children_fields):
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
            merged_properties.update(_build_ygg_properties(
                schema_info,
                engine="api",
                data_source_format=data_source_format,
                table_type=table_type,
                storage_location=storage_location,
            ))
        if properties:
            merged_properties.update({str(k): str(v) for k, v in properties.items()})

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
        return_data: bool = False,
        table_dispatch: "Mapping[Table | str, Predicate | str] | None" = None,
        **kwargs
    ) -> "Tabular | None":
        """Insert *data* into this table — thin wrapper over :meth:`insert_into`."""
        return self.insert_into(
            data,
            mode=mode,
            match_by=match_by,
            wait=wait,
            raise_error=raise_error,
            spark_session=spark_session,
            return_data=return_data,
            table_dispatch=table_dispatch,
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
        table_dispatch: "Mapping[Table | str, Predicate | str] | None" = None,
    ) -> "Tabular | None":
        """Insert *data* into this table using the most appropriate backend.

        Routing:

        - Query-shaped sources (str, ``PreparedStatement``,
          ``StatementResult``) → :meth:`sql_insert`
        - Spark DataFrame (or anything when a ``SparkSession`` is reachable)
          → :meth:`spark_insert`
        - Otherwise → :meth:`arrow_insert` (warehouse path with Volume staging)

        Returns ``None`` by default. With ``return_data=True`` the
        backend that ran the write hands back its source payload as a
        :class:`Tabular` — :class:`ArrowTabular` from
        :meth:`arrow_insert`, :class:`SparkTabular` from
        :meth:`spark_insert`, the input :class:`StatementResult` from
        :meth:`sql_insert` — for downstream chaining without
        re-querying the target.

        ``table_dispatch`` fans the same source rows into additional
        Delta tables in the same call. Each entry is
        ``target_or_location -> Predicate``; the predicate is rendered
        as a source-side ``WHERE`` so only matching rows reach that
        target. The primary insert (into ``self``) keeps its full mode
        / match_by / MERGE-fallback behavior; dispatch entries always
        run as plain ``INSERT INTO ... SELECT ... WHERE`` against the
        same staged source. Primary runs first; dispatch entries run as
        a single parallel batch only if the primary succeeds, so a
        partial fan-out can't leave you with rows in dispatch targets
        that never made it into the primary.
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
            table_dispatch=table_dispatch,
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

    def insert_volume_path(
        self,
        target: "Table | None" = None,
        *,
        temporary: bool = True,
        max_lifetime: float | None = 3600,
    ) -> VolumePath:
        """Mint the staging :class:`VolumePath` used by :meth:`arrow_insert`.

        Wraps :meth:`VolumePath.staging_path` with this table's
        catalog/schema/name and workspace client. Lifted out of
        :meth:`arrow_insert` so callers — and tests — can pre-mint
        or swap the staging location without driving the full
        insert. ``target`` defaults to ``self``; pass another
        :class:`Table` when the staging hierarchy needs to live next
        to a different table (e.g. dispatch fan-out).
        """
        target = target if target is not None else self
        return VolumePath.staging_path(
            client=self.client,
            catalog_name=target.catalog_name,
            schema_name=target.schema_name,
            resource_name=target.table_name,
            max_lifetime=max_lifetime,
            temporary=temporary,
        )

    def arrow_insert(
        self,
        data,
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
        where: Predicate | None = None,
        prune_by: list[str] | str | None = None,
        prune_values: Mapping[str, list[Any]] | None = None,
        retry: Optional[WaitingConfigArg] = None,
        return_data: bool = False,
        safe_merge: bool = False,
        table_dispatch: "Mapping[Table | str, Predicate | str] | None" = None,
    ) -> "Tabular | None":
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

        With ``return_data=True``, returns an :class:`ArrowTabular`
        wrapping the staged source rows so callers can chain on the
        payload without re-reading from the target.

        ``table_dispatch`` fans the staged Parquet into additional
        Delta tables. Each ``target -> Predicate`` entry runs as a
        plain ``INSERT INTO target (cols) SELECT cast_proj FROM
        {staging} WHERE <predicate>`` against the same staging volume,
        submitted as a single parallel batch after the primary insert
        lands.
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
                table_dispatch=table_dispatch,
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

        staging = self.insert_volume_path(target, temporary=bool(wait_cfg))

        prune_values = prune_values or {}
        output_data: "Tabular | None" = None
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
            # Capture the staged payload as a ArrowTabular before
            # the buffer is cleared.  Read straight off the spilled
            # Parquet so the holder shares the same row chunking the
            # warehouse will see.
            if return_data:
                from yggdrasil.io.tabular import ArrowTabular
                buffer.seek(0)
                output_data = ArrowTabular(buffer.read_arrow_table())

        prune_predicates = _build_prune_predicates(prune_values, target_alias="T") if prune_values else []
        if where is not None:
            prune_predicates.append(_wrap_user_predicate(where, target_alias="T"))

        columns = list(existing_schema.field_names())
        # Explicit per-column CAST to the target field's DDL coerces the
        # staged Parquet rows to the target table's schema. Mirrors the
        # projection used by :meth:`sql_insert`. ``to_databricks_ddl``
        # never emits ``NOT NULL`` on struct children, so the source
        # side is always nullable and matches the parquet reader's view
        # — Spark/Delta refuse the implicit cast otherwise
        # (``DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION``).
        cast_projection = ", ".join(
            (
                f"CAST({quote_ident(f.name)} AS "
                f"{f.to_databricks_ddl(with_name=False, with_nullable=False, with_comment=False)})"
                f" AS {quote_ident(f.name)}"
            )
            for f in existing_schema.fields
        )
        source_sql = f"SELECT {cast_projection} FROM {{{_ALIAS_TMPSRC}}}"

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
            "Arrow insert -> %s | mode=%s match_by=%s prune_by=%s statements=%d retry=%s",
            target_location, mode_enum, match_by, prune_by, len(prepared),
            retry_active,
        )

        _execute_dml(
            self.sql,
            statements=prepared,
            wait=wait_cfg,
            raise_error=raise_error,
            engine_name="api",
        )

        dispatch_entries = _resolve_dispatch_targets(table_dispatch, primary=target)
        if dispatch_entries:
            extra_prepared: list[WarehousePreparedStatement] = []
            for extra_target, predicate in dispatch_entries:
                extra_target.create(data, mode=schema_mode)
                extra_schema = extra_target.collect_schema()
                extra_columns = list(extra_schema.field_names())
                extra_cast_proj = ", ".join(
                    (
                        f"CAST({quote_ident(f.name)} AS "
                        f"{f.to_databricks_ddl(with_name=False, with_nullable=False, with_comment=False)})"
                        f" AS {quote_ident(f.name)}"
                    )
                    for f in extra_schema.fields
                )
                where_frag = _render_source_predicate(predicate)
                where_clause = f"\nWHERE {where_frag}" if where_frag else ""
                extra_source_sql = (
                    f"SELECT {extra_cast_proj} FROM {{{_ALIAS_TMPSRC}}}"
                    f"{where_clause}"
                )
                extra_texts = _build_dml_statements(
                    target_location=extra_target.full_name(safe=True),
                    source_sql=extra_source_sql,
                    columns=extra_columns,
                    mode=mode_enum,
                    match_by=match_by,
                    update_column_names=update_column_names,
                    prune_predicates=[],
                )
                extra_prepared.extend(_prepare_batch(extra_texts))
            if extra_prepared:
                logger.debug(
                    "Arrow dispatch fan-out -> %d target(s) | statements=%d",
                    len(dispatch_entries), len(extra_prepared),
                )
                extra_batch = self.sql.execute_many(
                    extra_prepared,
                    wait=wait_cfg,
                    raise_error=False,
                    engine="api",
                    parallel=max(1, len(dispatch_entries)),
                )
                if raise_error:
                    extra_batch.raise_for_status()

        return output_data

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
        table_dispatch: "Mapping[Table | str, Predicate | str] | None" = None,
    ) -> "Tabular | None":
        """Insert into this table using Spark.

        ``retry`` is applied to DML statements (INSERT/MERGE/DELETE/UPDATE)
        only — TRUNCATE/OPTIMIZE/VACUUM stay non-retryable.
        :class:`SparkStatementResult` already auto-promotes transient
        Delta failures (``ConcurrentAppendException``, …) to retryable;
        passing ``retry=True`` (or any :class:`WaitingConfig` arg) makes
        the policy explicit instead of relying on auto-promote.

        With ``return_data=True``, returns a :class:`SparkTabular`
        wrapping the materialised source DataFrame — handy for
        chaining downstream transforms without re-querying the
        target.

        ``table_dispatch`` fans the same registered temp view into
        additional Delta tables. Each ``target -> Predicate`` entry
        runs as a plain ``INSERT INTO target SELECT ... FROM view
        WHERE <predicate>`` against the same view, submitted as a
        single parallel batch after the primary insert lands.
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
                table_dispatch=table_dispatch,
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
            # Cache before the distinct().collect() so the temp view backing
            # the MERGE doesn't re-execute the source plan from scratch.
            data_df = data_df.cache()
            prune_values = _collect_prune_values_spark(data_df, prune_by)
            logger.debug(
                "Spark pruning %s -> %s",
                prune_by, {k: len(v) for k, v in prune_values.items()},
            )

        prune_predicates = _build_prune_predicates(prune_values, target_alias="T") if prune_values else []
        if where is not None:
            prune_predicates.append(_wrap_user_predicate(where, target_alias="T"))

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

        logger.info(
            "Spark insert -> %s | mode=%s match_by=%s prune_by=%s "
            "statements=%d retry=%s anti_join=%s",
            target_location, mode_enum, match_by, prune_by, len(prepared),
            retry_cfg is not None, anti_join_handled,
        )

        applied_conf = _delta_conf_for(overwrite_schema, spark_options)

        try:
            with sql_engine.spark.scoped_spark_conf(session, applied_conf):
                _execute_dml(
                    sql_engine,
                    statements=prepared,
                    wait=wait,
                    raise_error=raise_error,
                    engine_name="spark",
                )

                dispatch_entries = _resolve_dispatch_targets(
                    table_dispatch, primary=target,
                )
                if dispatch_entries:
                    extra_prepared: list[SparkPreparedStatement] = []
                    for extra_target, predicate in dispatch_entries:
                        extra_target.create(data, mode=schema_mode)
                        extra_schema = extra_target.collect_schema()
                        extra_columns = list(extra_schema.field_names())
                        extra_cols_quoted = ", ".join(
                            quote_ident(c) for c in extra_columns
                        )
                        where_frag = _render_source_predicate(predicate)
                        where_clause = f" WHERE {where_frag}" if where_frag else ""
                        extra_source_sql = (
                            f"SELECT {extra_cols_quoted} "
                            f"FROM {quote_ident(view_name)}{where_clause}"
                        )
                        extra_texts = _build_dml_statements(
                            target_location=extra_target.full_name(safe=True),
                            source_sql=extra_source_sql,
                            columns=extra_columns,
                            mode=mode_enum,
                            match_by=match_by,
                            update_column_names=update_column_names,
                            prune_predicates=[],
                        )
                        extra_prepared.extend(_prepare_spark_batch(extra_texts))
                    if extra_prepared:
                        logger.debug(
                            "Spark dispatch fan-out -> %d target(s) | statements=%d",
                            len(dispatch_entries), len(extra_prepared),
                        )
                        extra_batch = sql_engine.execute_many(
                            extra_prepared,
                            wait=wait,
                            raise_error=False,
                            engine="spark",
                            parallel=max(1, len(dispatch_entries)),
                        )
                        if raise_error:
                            extra_batch.raise_for_status()
        finally:
            try:
                session.catalog.dropTempView(view_name)
            except Exception:
                logger.debug("Failed to drop temp view %r; continuing.", view_name, exc_info=True)
            if prune_by and not return_data:
                # Keep the cached source alive when the caller asked
                # for it back — :class:`SparkTabular` is the consumer
                # and unpersisting here would force a re-execution
                # downstream.
                try:
                    data_df.unpersist()
                except Exception:
                    logger.debug("Failed to unpersist cached source; continuing.", exc_info=True)

        if return_data:
            from yggdrasil.io.tabular.spark import SparkTabular
            return SparkTabular(data_df)
        return None

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
        table_dispatch: "Mapping[Table | str, Predicate | str] | None" = None,
    ) -> "Tabular | None":
        """Insert into this table from a SQL source query.

        Smart dispatch:

        1. Cached :class:`StatementResult` → reuse the materialised frame
           via :meth:`insert_into` (no re-execution).
        2. SparkSession reachable → run via :meth:`spark_insert`.
        3. Otherwise → warehouse-side ``INSERT … SELECT`` /
           ``MERGE … USING (q)`` with a CAST projection aligning the
           user's query schema to the target.

        With ``return_data=True``, hands back the underlying
        :class:`StatementResult` (or the materialised frame from a
        cached one) so callers can stream the same rows the warehouse
        just inserted.

        ``table_dispatch`` flows through to whichever backend handles
        the write. On the warehouse fallback path, each
        ``target -> Predicate`` entry runs as a plain
        ``INSERT INTO target SELECT cast_proj FROM (source) AS raw_src
        WHERE <predicate>`` against the same wrapped source query,
        submitted as a single parallel batch after the primary insert
        lands.
        """
        common = dict(
            mode=mode,
            match_by=match_by, update_column_names=update_column_names,
            wait=wait, raise_error=raise_error,
            zorder_by=zorder_by, optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            where=where, prune_by=prune_by, prune_values=prune_values,
            retry=retry,
            table_dispatch=table_dispatch,
        )

        if isinstance(statement, StatementResult) and statement.cached:
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

        self._sql_insert_warehouse_fallback(statement, **common)
        if return_data and isinstance(statement, StatementResult):
            # The warehouse path doesn't materialise rows on its own,
            # but the caller's :class:`StatementResult` is already a
            # :class:`Tabular` over the same source query — hand it
            # back so ``return_data=True`` stays consistent across paths.
            return statement
        return None

    def _sql_insert_warehouse_fallback(
        self,
        statement: "PreparedStatement | StatementResult | str",
        *,
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
        table_dispatch: "Mapping[Table | str, Predicate | str] | None" = None,
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
                f"{f.to_databricks_ddl(with_name=False, with_nullable=False, with_comment=False)})"
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
            "SQL insert -> %s | mode=%s match_by=%s statements=%d retry=%s",
            target_location, mode_enum, match_by, len(prepared), retry_active,
        )

        if prepared:
            _execute_dml(
                self.sql,
                statements=prepared,
                wait=wait,
                raise_error=raise_error,
                engine_name="api",
            )

        dispatch_entries = _resolve_dispatch_targets(table_dispatch, primary=self)
        if dispatch_entries:
            extra_prepared: list[WarehousePreparedStatement] = []
            for extra_target, predicate in dispatch_entries:
                if not extra_target.exists:
                    raise ValueError(
                        "table_dispatch target "
                        f"{extra_target.full_name()!r} was not found. "
                        "sql_insert dispatch targets must already exist — "
                        "create them ahead of time or use arrow_insert / "
                        "spark_insert which can create on the fly."
                    )
                extra_schema = extra_target.collect_schema()
                extra_fields = list(extra_schema.fields)
                extra_columns = [f.name for f in extra_fields]
                extra_cast_proj = ", ".join(
                    (
                        f"CAST(raw_src.{quote_ident(f.name)} AS "
                        f"{f.to_databricks_ddl(with_name=False, with_nullable=False, with_comment=False)})"
                        f" AS {quote_ident(f.name)}"
                    )
                    for f in extra_fields
                )
                where_frag = _render_source_predicate(predicate)
                where_clause = f"\nWHERE {where_frag}" if where_frag else ""
                extra_source_sql = (
                    f"SELECT {extra_cast_proj} "
                    f"FROM (\n{source_prepared.text}\n) AS raw_src{where_clause}"
                )
                extra_texts = _build_dml_statements(
                    target_location=extra_target.full_name(safe=True),
                    source_sql=extra_source_sql,
                    columns=extra_columns,
                    mode=mode_enum,
                    match_by=match_by,
                    update_column_names=update_column_names,
                    prune_predicates=[],
                )
                extra_prepared.extend(_prepare_batch(extra_texts))
            if extra_prepared:
                logger.debug(
                    "SQL dispatch fan-out -> %d target(s) | statements=%d",
                    len(dispatch_entries), len(extra_prepared),
                )
                extra_batch = self.sql.execute_many(
                    extra_prepared,
                    wait=wait,
                    raise_error=False,
                    engine="api",
                    parallel=max(1, len(dispatch_entries)),
                )
                if raise_error:
                    extra_batch.raise_for_status()

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

    def tabular_io(
        self,
        operation: "TableOperation | ModeLike | None" = None,
    ) -> "DeltaIO":
        """Open this table's underlying Delta storage as a :class:`DeltaIO`.

        Convenience over :meth:`storage_location` + ``DeltaIO.from_path``:
        resolves the right credential :class:`TableOperation` from the
        table type and the requested mode, then binds a DeltaIO to the
        backing :class:`S3Path`. The returned IO carries the
        auto-refreshing credentials from :meth:`aws`, so reads survive
        STS token rotation without caller-side re-binding.
        """
        from yggdrasil.io.nested import DeltaIO

        return DeltaIO.from_path(self.storage_location(operation=operation))

    def aws(self, operation: TableOperation = TableOperation.READ) -> "AWSClient":
        """Return an :class:`AWSClient` whose credentials self-refresh
        from :meth:`temporary_credentials`.

        The returned client carries a botocore
        :class:`RefreshableCredentials`-backed session: every signing
        request that runs after the token's near-expiry window
        re-invokes :meth:`temporary_credentials` and rotates the
        underlying creds in place. No caller-side refresh dance.
        """
        from yggdrasil.aws.config import AwsCredentials

        def _refresh() -> AwsCredentials:
            creds = self.temporary_credentials(operation=operation)
            aws = creds.aws_temp_credentials
            expiration = getattr(creds, "expiration_time", None) / 1000
            return AwsCredentials(
                access_key_id=aws.access_key_id,
                secret_access_key=aws.secret_access_key,
                session_token=aws.session_token,
                expiration=(
                    expiration.isoformat()
                    if expiration is not None and hasattr(expiration, "isoformat")
                    else (str(expiration) if expiration is not None else None)
                ),
            )

        return (
            aws_config_class()
            .from_refresher(_refresh, region="eu-central-1")
            .to_client()
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
