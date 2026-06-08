"""Databricks table insert — the one place insert DML is generated.

This module centralizes *everything* about loading rows into a Unity Catalog
table for the synchronous warehouse / Spark / SQL paths:

- :class:`DatabricksTableInsert` is the full description of one insert
  operation — target, mode, the staged source, plus the keyed-write surface
  (``match_by``, ``update_column_names``, ``predicate``, ``zorder_by``,
  maintenance flags, ``safe_merge``). It is :class:`Awaitable` and
  self-executing: :meth:`~DatabricksTableInsert.execute` renders its own
  INSERT / MERGE statement list and runs it via ``execute_many``.
- :func:`make_sql_select` renders the per-op ``SELECT`` over its staged source.
- :func:`make_sql_insert` renders the full INSERT / MERGE / DELETE+INSERT /
  TRUNCATE / OPTIMIZE / VACUUM statement list.

``Table.arrow_insert`` stages the rows as Parquet to a Volume, builds a
:class:`DatabricksTableInsert` over that staged file, and calls
:meth:`~DatabricksTableInsert.execute`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional

from yggdrasil.databricks.sql.sql_utils import quote_ident
from yggdrasil.dataclasses.awaitable import Awaitable
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.enums.mode import Mode
from yggdrasil.enums.state import State
from yggdrasil.saga.expr.backends.sql import Dialect, to_sql as expr_to_sql
from yggdrasil.path import Path

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.data import Field
    from yggdrasil.databricks.table.table import Table
    from yggdrasil.saga.expr import Predicate
    from yggdrasil.io.tabular.base import Tabular

__all__ = [
    "ALIAS_TMPSRC",
    "DatabricksTableInsert",
    "make_sql_select",
    "make_sql_insert",
]

logger = logging.getLogger(__name__)

#: Placeholder a staged Parquet source is referenced by on the warehouse path —
#: substituted for the concrete external-data ``VolumePath`` at prepare time.
ALIAS_TMPSRC = "__tmpsrc__"


class _InsertExecution(Awaitable):
    """:class:`Awaitable` execution for :class:`DatabricksTableInsert`.

    The op builds its own INSERT / MERGE DML and runs it — callers don't render
    SQL or drive a warehouse batch externally. :meth:`execute` (and the
    :class:`Awaitable` ``start`` / ``wait`` / ``await``) drive an inner
    :class:`StatementBatch`: :meth:`_submit` renders the statement list and
    runs it through the target's SQL engine via ``execute_many``. The Awaitable
    mirrors that inner batch's state, so ``op.start(wait=False)`` fires the load
    and ``await op`` / ``op.wait()`` block on it.
    """

    #: The inner :class:`StatementBatch` (or backend result) the load drives.
    _inner: Any = None
    #: Per-run knobs stashed by :meth:`execute` for :meth:`_submit` to read.
    _engine: Any = None
    _retry: Any = None

    def _submit(self, *, wait: bool, raise_error: bool) -> Any:
        """Render this op's DML and submit it — return the inner batch."""
        raise NotImplementedError

    def execute(
        self,
        *,
        target: Any = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        engine: Any = None,
        retry: WaitingConfigArg | None = None,
    ) -> "_InsertExecution":
        """Build and run this insert. ``target`` rebinds the destination
        :class:`Table`; ``engine`` forces the ``"api"`` / ``"spark"`` backend,
        ``retry`` a per-statement retry policy. With ``wait`` (default) blocks
        until the statements finish; ``wait=False`` fires them and returns
        immediately (poll via :meth:`wait` / ``await``).
        """
        if target is not None:
            self.target = target
        self._engine = engine
        self._retry = retry
        return self.start(wait=wait, raise_error=raise_error)

    def _run_dml(
        self,
        target: Any,
        texts: "list[str]",
        *,
        staging: Any = None,
        source_ref: "str | None" = None,
        wait: bool,
        raise_error: bool,
    ) -> Any:
        """Prepare *texts* as warehouse statements and run them via
        ``execute_many``.

        When a Volume *staging* file backs the source, it is registered on
        the source-reading statement (matched by *source_ref* in the text)
        so its temporary scratch is reclaimed after the load.
        """
        from yggdrasil.databricks.warehouse import WarehousePreparedStatement

        prepared = []
        for sql in texts:
            external = (
                {ALIAS_TMPSRC: staging}
                if (staging is not None and source_ref and source_ref in sql)
                else None
            )
            prepared.append(
                WarehousePreparedStatement.prepare(
                    sql,
                    client=target.client,
                    external_volume_paths=external,
                    catalog_name=target.catalog_name,
                    schema_name=target.schema_name,
                )
            )
        if not prepared:
            return None
        return target.sql.execute_many(
            statements=prepared,
            wait=wait,
            raise_error=raise_error,
            engine=self._engine,
            retry=self._retry,
        )

    @property
    def result(self) -> Any:
        """The inner :class:`StatementBatch` driving the load (``None`` until
        :meth:`start` / :meth:`execute`)."""
        return self._inner

    # -- Awaitable hooks ------------------------------------------------------

    def _start(self) -> None:
        self._inner = self._submit(wait=False, raise_error=False)
        self._sync_from_inner()

    def _poll(self) -> None:
        inner = self._inner
        if inner is not None and hasattr(inner, "wait"):
            inner.wait(wait=False, raise_error=False)
        self._sync_from_inner()

    def _sync_from_inner(self) -> None:
        inner = self._inner
        if inner is None:
            # Nothing to run — treat as done.
            self._state = State.SUCCEEDED
            return
        st = getattr(inner, "state", None)
        if st is not None:
            try:
                self._state = State.from_(st)
                return
            except (TypeError, ValueError):
                pass
        # A backend result without a coercible state (or a non-Awaitable
        # one) is taken as done once it reports so.
        if getattr(inner, "is_done", True):
            self._state = State.SUCCEEDED

    def _error_for_status(self) -> "BaseException | None":
        inner = self._inner
        return getattr(inner, "error", None) if inner is not None else None

    @property
    def retryable(self) -> bool:
        inner = getattr(self, "_inner", None)
        return bool(getattr(inner, "retryable", False))


@dataclass
class DatabricksTableInsert(_InsertExecution):
    """One insert operation — the full ``arrow_insert`` surface in one object.

    Carries the ``target`` table, the save ``mode``, the staged ``data``
    location (a :class:`Path` / :class:`VolumePath`, or a uniform-URL string),
    and the keyed-write surface (``schema``, ``predicate``, ``match_by``,
    ``update_column_names``, ``schema_mode``, ``zorder_by``,
    ``optimize_after_merge``, ``vacuum_hours``, ``safe_merge``).

    ``target`` may be a :class:`Table` or its full name; ``data`` is the staged
    Parquet source — a :class:`Path` / :class:`VolumePath`, or its uniform URL
    as a string (reconstructed through the bound client at execute time).
    """

    target: "Table | str"
    mode: Mode
    data: "Tabular | Path | str"
    client: "DatabricksClient | None" = None

    #: the data's schema (the same :class:`Field` / :class:`Schema` type
    #: ``Table.collect_schema()`` returns); drives the SELECT projection.
    schema: "Field | None" = None
    #: target-row prune / source-row filter, folded in from ``cast_options``.
    predicate: "Predicate | None" = None
    match_by: "list[str] | None" = None
    update_column_names: "list[str] | None" = None
    schema_mode: "Mode | str | None" = None
    zorder_by: "list[str] | None" = None
    optimize_after_merge: bool = False
    vacuum_hours: "int | None" = None
    safe_merge: bool = False

    def __post_init__(self) -> None:
        # The :class:`StatementBatch` this op's execution drives — set by
        # :meth:`_start` (the :class:`Awaitable` lifecycle).
        self._inner: Any = None
        self.mode = Mode.from_(self.mode, default=Mode.APPEND)
        # The bare op shape (no ``schema`` / ``match_by``) only knows the
        # unkeyed OVERWRITE / APPEND modes; a keyed op carries the richer
        # surface and supports every mode.
        if (
            self.mode not in (Mode.OVERWRITE, Mode.APPEND)
            and self.schema is None
            and not self.match_by
        ):
            raise ValueError(
                f"insert supports only OVERWRITE / APPEND for a bare op, "
                f"got {self.mode.name}"
            )

    # -- normalized views -----------------------------------------------------

    @property
    def target_name(self) -> str:
        """``catalog.schema.table`` — the resolved target name."""
        full_name = getattr(self.target, "full_name", None)
        return full_name() if callable(full_name) else str(self.target)

    def data_path(self, client: Any = None) -> "Path":
        """Resolve the staged **file** ``data`` to a concrete :class:`Path`.

        Already a :class:`Path` → returned as-is; a uniform-URL string →
        reconstructed through the bound (or supplied) client so the warehouse
        can read it wherever it landed.
        """
        if isinstance(self.data, Path):
            return self.data
        from yggdrasil.databricks.path import DatabricksPath

        return DatabricksPath.from_(self.data, client=client or self.client)

    def staged_source(self, client: Any = None) -> Any:
        """Rebuild the staged ``data`` into the concrete :class:`Path` the
        warehouse reads. A live :class:`Path` is returned unchanged; a
        serialized URL is rebuilt through the bound (or supplied) client."""
        if isinstance(self.data, str):
            return self.data_path(client)
        return self.data

    def select_sql(self, client: Any = None) -> str:
        """Back-compat alias for :func:`make_sql_select` over this op."""
        return make_sql_select(self, client=client)

    def _target_table(self) -> "Table":
        """The :class:`Table` to run the load against. The producer constructs
        the op with the bound :class:`Table`; a bare-string ``target`` carries
        no client-resolvable surface."""
        target = self.target
        if hasattr(target, "full_name") and hasattr(target, "catalog_name"):
            return target  # a Table
        raise TypeError(
            "Running an insert needs a Table `target` "
            f"(got {type(target).__name__!r}); build the op with the bound "
            "Table so its sql engine can be reached."
        )

    def _submit(self, *, wait: bool, raise_error: bool) -> Any:
        """Render this op's INSERT / MERGE statement list and run it via
        ``execute_many`` — the specialized table load, fully self-contained.

        The staged Parquet file is referenced as ``parquet.`<path>``` and
        registered for post-load cleanup. The target must be a :class:`Table`
        (pass it to :meth:`execute` when the op only carries the name)."""
        from yggdrasil.databricks.warehouse import WarehousePreparedStatement

        target = self._target_table()
        schema = self.schema if (self.schema and self.schema.fields) else target.collect_schema()
        columns = [f.name for f in schema.fields] if schema and schema.fields else None

        staged = self.staged_source(self.client)
        staging = None
        source_ref = None
        source_sql = None
        if getattr(staged, "full_path", None) is not None:
            staging = staged
            source_ref = WarehousePreparedStatement.volume_path_text_value(staged)
            source_sql = make_sql_select(self, source=source_ref)

        texts = make_sql_insert(
            self,
            target_location=target.full_name(safe=True),
            source_sql=source_sql,
            columns=columns,
            client=self.client,
        )
        return self._run_dml(
            target, texts,
            staging=staging, source_ref=source_ref,
            wait=wait, raise_error=raise_error,
        )


# ===========================================================================
# DML generation — the single source of truth for every insert path.
#
# ``make_sql_insert`` reproduces the arrow / spark / sql sync-path DML exactly:
#
#   - ``overwrite`` (no keys)  → plain ``INSERT OVERWRITE``
#   - ``truncate`` (no keys)   → ``TRUNCATE TABLE`` + ``INSERT``
#   - ``truncate``/``overwrite`` with keys → keyed ``DELETE`` + ``INSERT``
#   - keyed, ``safe_merge=False`` → single ``MERGE INTO`` (insert-only for
#     APPEND/AUTO, full update+insert for UPSERT/MERGE)
#   - keyed UPSERT/MERGE, ``safe_merge=True`` → keyed ``DELETE`` + ``INSERT``
#   - keyed APPEND/AUTO, ``safe_merge=True`` → ``INSERT … WHERE NOT EXISTS``
#   - no keys                  → plain ``INSERT INTO``
#   - maintenance tail         → ``OPTIMIZE [ZORDER]`` / ``VACUUM``
#
# Merge ``ON`` is built null-safe (``<=>``) so NULL matches NULL.
# ===========================================================================


def make_sql_select(
    op: "DatabricksTableInsert",
    *,
    client: Any = None,
    source: "str | None" = None,
) -> str:
    """The atomic per-op ``SELECT`` over the op's staged source.

    Two source shapes:

    * **default** — render ``SELECT * FROM parquet.`<warehouse path>``` over
      the op's staged Parquet (resolved from its :class:`Path` / uniform URL).
    * **explicit *source*** — when the caller already has a source reference
      (the ``{__tmpsrc__}`` placeholder, which is substituted for the
      external-data ``VolumePath`` at prepare time), project the op's schema
      columns from it: ``SELECT <projection> FROM <source>``.
    """
    if source is not None:
        if op.schema is not None and op.schema.fields:
            projection = _build_column_projection(op.schema.fields)
        else:
            projection = "*"
        return f"SELECT {projection} FROM {source}"

    path = op.data_path(client)
    full_path = getattr(path, "full_path", None)
    ref = full_path() if callable(full_path) else str(op.data)
    return f"SELECT * FROM parquet.`{ref}`"


def make_sql_insert(
    op: "DatabricksTableInsert",
    *,
    target_location: "str | None" = None,
    source_sql: "str | None" = None,
    columns: "list[str] | None" = None,
    client: Any = None,
) -> list[str]:
    """Render the full statement list for one insert.

    Yields the INSERT / MERGE / DELETE+INSERT / TRUNCATE / OPTIMIZE / VACUUM
    statement list for the op.

    ``target_location`` / ``source_sql`` / ``columns`` let the synchronous
    paths supply their own source reference (the ``{__tmpsrc__}`` placeholder,
    a Spark temp-view name, or a wrapped user query) and pre-resolved target
    location; when omitted they're derived from the op's ``target`` and staged
    data.
    """
    location = _resolve_target_location(op.target, target_location)
    if columns is None:
        columns = (
            [f.name for f in op.schema.fields]
            if op.schema is not None and op.schema.fields
            else []
        )
    if source_sql is None:
        source_sql = make_sql_select(op, client=client)
    prune_predicates = _build_where_predicates(op.predicate, target_alias="T")
    return _build_dml_statements(
        target_location=location,
        source_sql=source_sql,
        columns=columns,
        mode=op.mode,
        match_by=op.match_by,
        update_column_names=op.update_column_names,
        prune_predicates=prune_predicates,
        zorder_by=op.zorder_by,
        optimize_after_merge=op.optimize_after_merge,
        vacuum_hours=op.vacuum_hours,
        safe_merge=op.safe_merge,
    )


def _resolve_target_location(target: Any, fallback: "str | None") -> str:
    if fallback is not None:
        return fallback
    full_name = getattr(target, "full_name", None)
    if callable(full_name):
        return full_name(safe=True)
    return str(target)


# ---------------------------------------------------------------------------
# DML builders — the shared insert SQL generator.
# ---------------------------------------------------------------------------


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


def _build_where_predicates(
    where: "Predicate | None",
    *,
    target_alias: str,
) -> list[str]:
    """Render *where* as a target-aliased SQL clause for MERGE / DELETE.

    Returns a ``list[str]`` (0 or 1 elements) so the downstream DML
    builders can splice it into an AND chain.
    """
    if where is None:
        return []
    aliased = _alias_columns(where, target_alias)
    sql = expr_to_sql(aliased, dialect=Dialect.DATABRICKS)
    return [sql]


def _alias_columns(expr, alias: str):
    """Walk *expr* and stamp every :class:`Column` with ``alias``.

    Lifts a user predicate onto the target-side of a MERGE so ``foo`` becomes
    ``T.foo``. Returns a new tree — the AST is immutable so we never mutate
    the caller's predicate.
    """
    from yggdrasil.saga.expr.nodes import (
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


def _build_column_projection(
    fields: "Iterable[Field]",
    *,
    source_alias: "str | None" = None,
) -> str:
    """Build a plain column-reference projection list for INSERT/MERGE.

    Each :class:`Field` contributes one bare (or alias-qualified) quoted
    column reference — no per-column ``CAST(... AS <ddl>)`` wrapper. The data
    has already been aligned to the target schema upstream (the arrow cast
    pipeline / Spark dataframe coercion / the warehouse's implicit
    column-boundary cast), so the engine accepts the rows as-is. Skipping the
    explicit CAST keeps the SQL short — important for wide / deeply nested
    schemas where the spelled-out DDL bloated statements past the warehouse
    text limits.
    """
    parts: list[str] = []
    alias = quote_ident(source_alias) if source_alias else None
    for f in fields:
        col = quote_ident(f.name)
        parts.append(f"{alias}.{col}" if alias else col)
    return ", ".join(parts)


def _build_cast_column_projection(
    target_fields: "Iterable[Field]",
    *,
    source: "Field | None" = None,
    source_alias: str,
) -> str:
    """Build a SELECT projection that CASTs source columns to target Spark types.

    For each target field:

    * **present in source, same type** — bare ``alias.`col``` (no CAST)
    * **present in source, different type** —
      ``CAST(alias.`col` AS <spark_type>)``
    * **missing from source** — ``CAST(NULL AS <spark_type>) AS `col```

    *source* is the :class:`Field` describing the source schema. Child lookup
    uses :meth:`Field.get` — no intermediate dict. When *source* is ``None``
    every target column is assumed present with an unknown type (always CAST).
    """
    alias = quote_ident(source_alias)
    parts: list[str] = []
    for f in target_fields:
        col = quote_ident(f.name)
        target_spark = f.to_spark_name(
            with_name=False, with_nullable=False, with_comment=False,
        )
        src = source.get(f.name) if source is not None else ...

        if src is None:
            parts.append(f"CAST(NULL AS {target_spark}) AS {col}")
        elif src is ...:
            parts.append(f"CAST({alias}.{col} AS {target_spark})")
        else:
            source_spark = src.to_spark_name(
                with_name=False, with_nullable=False, with_comment=False,
            )
            if source_spark == target_spark:
                parts.append(f"{alias}.{col}")
            else:
                parts.append(f"CAST({alias}.{col} AS {target_spark})")
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
    target rows whose match keys appear in ``source_sql``, followed by a plain
    ``INSERT INTO ... SELECT``. Databricks/Spark SQL doesn't accept ``DELETE
    FROM target USING source``; the keyed delete is rendered as ``DELETE FROM
    target T WHERE EXISTS (...)`` so it parses on Delta. ``prune_predicates``
    lift onto the outer ``WHERE`` so the target scan is bounded before the
    EXISTS subquery runs.
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

    Used by :func:`_build_dml_statements` when ``safe_merge=False`` (the
    default) — Databricks / Delta supports MERGE natively and plans the keyed
    dedup once instead of twice (one delete + one insert) the way the safe path
    does.

    ``insert_only=True`` emits a MERGE with only the ``WHEN NOT MATCHED THEN
    INSERT`` clause — the keyed-APPEND shape. Without it, the full
    update-and-insert MERGE runs.
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

    No ``MERGE``, no fallback, no retry. The ``<=>`` null-safe comparison
    matches the MERGE behavior so rows with NULL key columns line up.
    ``prune_predicates`` (target-aliased) narrow the EXISTS scan to the
    matching partitions, so the keyed dedup doesn't read the whole target.
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

    * **safe_merge=False (default)** — emit a single ``MERGE INTO`` statement.
      :attr:`Mode.UPSERT` / :attr:`Mode.MERGE` get the full update-and-insert
      MERGE, :attr:`Mode.APPEND` / :attr:`Mode.AUTO` get the insert-only
      variant.
    * **safe_merge=True** — sidestep MERGE entirely.
      :attr:`Mode.UPSERT` / :attr:`Mode.MERGE` run a keyed ``DELETE`` followed
      by ``INSERT`` (incoming wins on overlap). :attr:`Mode.APPEND` /
      :attr:`Mode.AUTO` run ``INSERT ... WHERE NOT EXISTS (...)`` so existing
      rows are filtered at INSERT time.

    Mode without keys:

    * :attr:`Mode.TRUNCATE` with ``match_by`` → DELETE + INSERT.
    * :attr:`Mode.TRUNCATE` no keys → ``TRUNCATE TABLE`` + INSERT.
    * :attr:`Mode.OVERWRITE` → plain INSERT (the caller already cleared the
      target up front).
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
        elif match_by:
            statements.extend(_build_delete_insert_statements(
                target_location=target_location,
                source_sql=source_sql,
                columns=columns,
                match_by=match_by,
                prune_predicates=prune_predicates,
            ))
        else:
            statements.append(
                f"INSERT OVERWRITE {target_location} ({cols_quoted})\n{source_sql}"
            )

    elif match_by and not safe_merge:
        # Native MERGE INTO — Databricks / Delta plans the dedup once.
        # Insert-only for APPEND/AUTO; full update + insert for UPSERT/MERGE.
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
        statements.extend(_build_delete_insert_statements(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            match_by=match_by,
            prune_predicates=prune_predicates,
        ))

    elif match_by:
        # safe_merge=True + AUTO/APPEND — INSERT NOT EXISTS so existing rows
        # are filtered at INSERT time.
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
