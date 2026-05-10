"""SQL execution — Polars fast path with an Arrow fallback.

Two backends, picked at runtime by what's installed:

- :class:`PolarsSqlExecutor` (preferred) — registers each
  referenced source on a :class:`polars.SQLContext` and lets polars
  evaluate the query end-to-end. Covers joins, aggregations,
  window functions, CTEs, ORDER BY, LIMIT, UNION, EXCEPT — i.e.
  the full Spark-SQL shape on top of Arrow batches with zero
  extra glue.
- :class:`ArrowSqlExecutor` (fallback) — used when polars isn't
  installed. Handles the
  ``SELECT cols FROM <single source> [WHERE pred] [LIMIT n]``
  shape by lifting the predicate to our :class:`Expression` AST,
  rendering it back to a :class:`pyarrow.compute.Expression` for
  pushdown, and projecting via :class:`yggdrasil.data.data_field.Field`.
  Anything beyond that surface raises a clean "install polars"
  error so the user gets a single-line fix.

Both backends materialize the result into a :class:`Tabular`
holder (:class:`ArrowTabular` by default, :class:`ParquetIO`
folder when :attr:`SqlPreparedStatement.persist == "path"`) and
return it alongside the row count. The :class:`SqlStatementResult`
hangs the holder off ``_persisted_data`` so subsequent reads short-
circuit through the cache.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Iterable

import pyarrow as pa

from yggdrasil.io.tabular.execution.expr import (
    Column,
    Expression,
    Logical,
    LogicalOp,
)
from yggdrasil.data.data_field import Field
from yggdrasil.data.executor import StatementExecutor
from yggdrasil.io.tabular import Tabular
from yggdrasil.io.tabular import ArrowTabular

from .catalog import SqlContext, default_context
from yggdrasil.lazy_imports import has_polars, sqlglot_expressions
from .parser import SqlParseError, extract_sources, is_query, parse
from .statement import SqlPreparedStatement, SqlStatementResult

if TYPE_CHECKING:
    pass


__all__ = [
    "SqlExecutor",
    "PolarsSqlExecutor",
    "ArrowSqlExecutor",
    "resolve_executor",
]


class SqlExecutor(StatementExecutor, ABC):
    """Backend-agnostic base.

    Subclasses implement :meth:`run` returning ``(holder, row_count)``
    — the materialized payload plus its row count. The
    :class:`SqlStatementResult` lifecycle calls ``run`` exactly
    once per started statement; persisting / re-reading is the
    holder's job.

    The standard :class:`StatementExecutor` contract
    (:meth:`_submit_statement`, :meth:`execute`, :meth:`execute_many`)
    is satisfied by building a fresh :class:`SqlStatementResult` and
    handing it back — the result drives this executor's :meth:`run`
    on first read via its own ``start`` lifecycle.
    """

    _PREPARED_STATEMENT_CLASS: ClassVar[type[SqlPreparedStatement]] = SqlPreparedStatement
    _STATEMENT_RESULT_CLASS: ClassVar[type[SqlStatementResult]] = SqlStatementResult

    def __init__(self, context: "SqlContext | None" = None) -> None:
        super().__init__()
        self.context: SqlContext = context or default_context

    # ------------------------------------------------------------------
    # StatementExecutor contract
    # ------------------------------------------------------------------

    def _submit_statement(
        self,
        statement: SqlPreparedStatement,
    ) -> SqlStatementResult:
        """Build a fresh result tied to this executor.

        ``execute()`` (inherited from :class:`StatementExecutor`)
        will then call ``result.wait()`` if asked, which our
        synchronous :meth:`SqlStatementResult.start` handles in
        one shot — there's no separate "submit then wait" round-
        trip for in-process SQL.
        """
        result = self._STATEMENT_RESULT_CLASS(
            statement, executor=self, context=self.context,
        )
        result.start(raise_error=False)
        return result

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        statement: SqlPreparedStatement,
        *,
        context: "SqlContext | None" = None,
    ) -> "tuple[Tabular, int]":
        """Execute *statement* and return ``(holder, row_count)``."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _resolve_sources(
        self,
        statement: SqlPreparedStatement,
        context: "SqlContext | None",
    ) -> "dict[str, Tabular]":
        """Return ``{name: Tabular}`` for every base table the SQL touches.

        Statement-bound sources win (they were snapshotted at
        prepare time) over the live context — that mirrors the
        contract documented on :class:`SqlPreparedStatement`.
        """
        ctx = context or self.context
        try:
            root = parse(statement.text, dialect=statement.dialect)
        except SqlParseError:
            raise
        if not is_query(root):
            sge = sqlglot_expressions()
            raise NotImplementedError(
                f"yggdrasil.sql executes row-producing statements only; "
                f"got {type(root).__name__} via {statement.text!r}. "
                "DDL/DML against the in-process catalog is not supported "
                "— register a fresh source instead via ctx.register(...)."
            )

        bound = dict(statement.sources)
        sources: "dict[str, Tabular]" = {}
        for name in extract_sources(root):
            if name in bound:
                sources[name] = bound[name]
                continue
            hit = ctx.get(name)
            if hit is None and "." in name:
                # Try the leaf-only name when a qualified ref didn't
                # match — analysts often register ``trades`` and
                # query ``main.warehouse.trades``.
                leaf = name.rsplit(".", 1)[-1]
                hit = ctx.get(leaf)
            if hit is None:
                available = ctx.names() + list(bound.keys())
                raise KeyError(
                    f"SQL source {name!r} is not registered. Available: "
                    f"{available!r}. Register via "
                    "yggdrasil.sql.register(name, source) or pass "
                    "sources={name: io} to sql()."
                )
            sources[name] = hit
        return sources

    def _build_holder(
        self,
        statement: SqlPreparedStatement,
        table: pa.Table,
    ) -> Tabular:
        """Materialize *table* into the requested persistence target.

        ``persist == "memory"`` → :class:`ArrowTabular` (zero copy
        — the holder just keeps a reference to the batches).
        ``persist == "path"`` → spill to a parquet file/folder under
        ``statement.path`` and return the :class:`ParquetIO`
        wrapping it. ``persist is None`` collapses to memory; the
        statement just won't keep the holder alive past the first
        read.
        """
        target = statement.persist or "memory"
        if target == "memory":
            return ArrowTabular(table)
        if target == "path":
            if not statement.path:
                raise ValueError(
                    "persist='path' requires a non-empty `path` on the "
                    "SqlPreparedStatement (or pass path='...' to sql(...))."
                )
            return _spill_to_path(table, statement.path)
        raise ValueError(
            f"Unknown persist target {target!r}. Valid: 'memory', "
            "'path', or None (= memory)."
        )


def _spill_to_path(table: pa.Table, path: str) -> Tabular:
    """Write *table* to a parquet target and return a :class:`Tabular`.

    A path ending in ``.parquet`` writes a single file; anything
    else is treated as a folder (created on demand) so multi-batch
    results stay streamable. The path is *owned* by the caller —
    we don't unlink it on close, mirroring the convention used by
    the rest of the IO stack for caller-supplied paths.
    """
    import pyarrow.parquet as pq

    parent = os.path.dirname(path) if path.endswith(".parquet") else path
    if parent:
        os.makedirs(parent, exist_ok=True)
    if path.endswith(".parquet"):
        pq.write_table(table, path)
    else:
        # Folder: one file per batch keeps the writer streaming.
        os.makedirs(path, exist_ok=True)
        for index, batch in enumerate(table.to_batches()):
            pq.write_table(
                pa.Table.from_batches([batch]),
                os.path.join(path, f"part-{index:08d}.parquet"),
            )
    return Tabular.from_path(path)


# ---------------------------------------------------------------------------
# Polars backend — preferred when polars is installed
# ---------------------------------------------------------------------------


class PolarsSqlExecutor(SqlExecutor):
    """SQL via :class:`polars.SQLContext`.

    Each referenced source is wrapped as a :class:`polars.LazyFrame`
    and registered on a fresh :class:`polars.SQLContext`. We
    register under both the original name and (when the original
    name is dotted) its leaf form, so a query against
    ``main.warehouse.trades`` resolves to a source registered as
    ``trades``.

    A statement-level ``predicate`` (set via
    :meth:`SqlPreparedStatement.with_predicate` or the ``where=``
    kwarg of :func:`yggdrasil.sql.sql`) is applied to the
    LazyFrame *after* the SQL — so the SQL stays unmodified and
    the predicate composes via :meth:`Expression.to_polars`. A
    statement-level ``select`` projection is applied last.
    """

    def run(
        self,
        statement: SqlPreparedStatement,
        *,
        context: "SqlContext | None" = None,
    ) -> "tuple[Tabular, int]":
        import polars as pl

        sources = self._resolve_sources(statement, context)
        ctx = pl.SQLContext()
        for name, io in sources.items():
            lf = io.scan_polars_frame() if hasattr(io, "scan_polars_frame") \
                else pl.from_arrow(io.read_arrow_table()).lazy()
            ctx.register(name, lf)
            # Convenience: register leaf alias too so qualified
            # references resolve cleanly when polars-SQL strips
            # the qualifier.
            if "." in name:
                leaf = name.rsplit(".", 1)[-1]
                if leaf not in sources and leaf not in ctx.tables():
                    ctx.register(leaf, lf)

        try:
            lazy = ctx.execute(statement.text)
        except Exception as exc:
            raise RuntimeError(
                f"polars failed to execute SQL {statement.text!r} "
                f"(dialect={statement.dialect.value}): {exc}. "
                "If this is valid Databricks/Postgres SQL that polars "
                "doesn't support yet, fall back to ArrowSqlExecutor or "
                "evaluate against a Spark / Postgres backend."
            ) from exc

        if statement.predicate is not None:
            lazy = lazy.filter(statement.predicate.to_polars())

        if statement.select is not None:
            lazy = lazy.select([
                _selector_to_polars(s) for s in statement.select
            ])

        frame = lazy.collect()
        table = frame.to_arrow()
        holder = self._build_holder(statement, table)
        return holder, table.num_rows


def _selector_to_polars(spec: Any) -> Any:
    """Translate a projection entry to a polars expression.

    Strings → bare column reference. :class:`Field` →
    cast-on-select with rename — :attr:`Field.alias` (when
    different from :attr:`Field.name`) is the source-side label
    and :attr:`Field.name` is the projected output name; the
    target dtype comes from :attr:`Field.dtype`. Anything else
    falls through to :meth:`Expression.to_polars` so a builder-
    side ``col(...).alias("...")`` chain works directly.
    """
    import polars as pl

    if isinstance(spec, str):
        return pl.col(spec)
    if isinstance(spec, Field):
        source = spec.alias if spec.has_alias else spec.name
        expr = pl.col(source)
        if spec.dtype is not None and hasattr(spec.dtype, "to_polars"):
            try:
                expr = expr.cast(spec.dtype.to_polars())
            except Exception:
                pass
        return expr.alias(spec.name) if spec.name and spec.name != source else expr
    if isinstance(spec, Column):
        return pl.col(spec.name)
    if isinstance(spec, Expression):
        return spec.to_polars()
    raise TypeError(
        f"Unsupported select entry {type(spec).__name__}: {spec!r}. "
        "Pass column names, Field, Column, or Expression."
    )


# ---------------------------------------------------------------------------
# Arrow-only fallback
# ---------------------------------------------------------------------------


class ArrowSqlExecutor(SqlExecutor):
    """Polars-free fallback for the simple ``SELECT cols FROM src`` shape.

    Limited but useful: covers single-source ``SELECT cols FROM src
    [WHERE pred] [LIMIT n]`` plus ``SELECT *``. Predicates lift to
    :class:`Expression` and render to
    :class:`pyarrow.compute.Expression` for pushdown via
    :class:`pyarrow.dataset`. Anything else (joins, ``GROUP BY``,
    ``ORDER BY``, subqueries, set ops) raises with a clean
    "install polars" hint so the user gets a one-line fix.

    Intended for base installs; on a real workload you almost
    always want :class:`PolarsSqlExecutor`.
    """

    def run(
        self,
        statement: SqlPreparedStatement,
        *,
        context: "SqlContext | None" = None,
    ) -> "tuple[Tabular, int]":
        sources = self._resolve_sources(statement, context)

        sge = sqlglot_expressions()
        root = parse(statement.text, dialect=statement.dialect)

        if not isinstance(root, sge.Select):
            raise NotImplementedError(
                "ArrowSqlExecutor handles SELECT only — install polars "
                "(pip install polars) for joins, CTEs, aggregations, "
                "set operations, and ORDER BY."
            )

        # sqlglot 30+ renamed the args key to ``from_`` to avoid the
        # Python keyword clash; older versions used ``from``. We try
        # both so the executor stays version-tolerant.
        from_clause = root.args.get("from_") or root.args.get("from")
        if from_clause is None:
            raise NotImplementedError(
                "ArrowSqlExecutor requires a FROM clause — install polars "
                "(pip install polars) for VALUES-only / scalar SELECTs."
            )
        froms = list(from_clause.expressions or [from_clause.this])
        if len(froms) != 1 or root.args.get("joins"):
            raise NotImplementedError(
                "ArrowSqlExecutor handles single-source SELECT only — "
                "install polars (pip install polars) for joins."
            )
        for unsupported in ("group", "having", "order", "windows", "qualify"):
            if root.args.get(unsupported):
                raise NotImplementedError(
                    f"ArrowSqlExecutor does not support {unsupported.upper()} "
                    "— install polars (pip install polars) to enable it."
                )

        # Resolve the single source.
        table_node = froms[0]
        if isinstance(table_node, sge.Alias):
            table_node = table_node.this
        if not isinstance(table_node, sge.Table):
            raise NotImplementedError(
                "ArrowSqlExecutor expects a base table in FROM; subqueries "
                "require polars."
            )
        name = ".".join(filter(None, [
            getattr(table_node, "catalog", None),
            getattr(table_node, "db", None),
            table_node.name,
        ]))
        io = sources.get(name) or sources.get(table_node.name)
        if io is None:
            raise KeyError(
                f"Source {name!r} unresolved in ArrowSqlExecutor — internal "
                "bug, _resolve_sources should have raised earlier."
            )

        # Build the predicate: SQL WHERE composed with statement.predicate.
        where = root.args.get("where")
        sql_predicate: "Expression | None" = None
        if where is not None:
            sql_predicate = Expression.from_sql(
                where.this.sql(dialect=statement.dialect.value),
                dialect=statement.dialect.value,
            )
        composed = _and_merge(sql_predicate, statement.predicate)

        # Read with predicate pushdown when possible.
        from yggdrasil.data.options import CastOptions

        options = CastOptions(predicate=composed) if composed is not None \
            else CastOptions()
        table = io.read_arrow_table(options=options)

        # Project columns (SELECT list).
        table = _project_arrow(table, root, statement)

        # LIMIT.
        limit = root.args.get("limit")
        if limit is not None:
            n = int(limit.expression.this if hasattr(limit.expression, "this")
                    else limit.expression)
            table = table.slice(0, n)

        holder = self._build_holder(statement, table)
        return holder, table.num_rows


def _and_merge(
    a: "Expression | None",
    b: "Expression | None",
) -> "Expression | None":
    if a is None:
        return b
    if b is None:
        return a
    return Logical(LogicalOp.AND, (a, b))


def _project_arrow(
    table: pa.Table,
    root: Any,
    statement: SqlPreparedStatement,
) -> pa.Table:
    """Apply the SELECT list (and any statement-level ``select``).

    Handles the simple cases: ``SELECT *``, ``SELECT col1, col2``,
    ``SELECT col AS alias``. Computed expressions in the SELECT
    list (``UPPER(col)``, arithmetic) raise NotImplementedError —
    polars-backed execution handles those without our help.
    """
    sge = sqlglot_expressions()
    selects = list(root.expressions or [])
    if not selects or any(isinstance(s, sge.Star) for s in selects):
        if statement.select is not None:
            return _apply_statement_select(table, statement.select)
        return table

    out_names: list[str] = []
    out_columns: list[pa.ChunkedArray] = []
    for expr in selects:
        alias = None
        node = expr
        if isinstance(expr, sge.Alias):
            alias = expr.alias
            node = expr.this
        if isinstance(node, sge.Column):
            name = node.name
            if name not in table.column_names:
                raise KeyError(
                    f"SELECT references unknown column {name!r}. "
                    f"Available: {table.column_names!r}."
                )
            out_names.append(alias or name)
            out_columns.append(table[name])
        else:
            raise NotImplementedError(
                f"ArrowSqlExecutor SELECT supports columns and *; got "
                f"{type(node).__name__} ({expr.sql()!r}). Install polars "
                "(pip install polars) for computed projections."
            )
    projected = pa.Table.from_arrays(out_columns, names=out_names)
    if statement.select is not None:
        projected = _apply_statement_select(projected, statement.select)
    return projected


def _apply_statement_select(
    table: pa.Table,
    select: "Iterable[Any]",
) -> pa.Table:
    """Apply a Python-side ``select`` list (names / Field / Column)."""
    out_names: list[str] = []
    out_columns: list[Any] = []
    for spec in select:
        if isinstance(spec, str):
            out_names.append(spec)
            out_columns.append(table[spec])
            continue
        if isinstance(spec, Field):
            source = spec.alias if spec.has_alias else spec.name
            col = table[source]
            if spec.dtype is not None and hasattr(spec.dtype, "to_arrow"):
                try:
                    col = col.cast(spec.dtype.to_arrow())
                except Exception:
                    pass
            out_names.append(spec.name)
            out_columns.append(col)
            continue
        if isinstance(spec, Column):
            out_names.append(spec.alias or spec.name)
            out_columns.append(table[spec.name])
            continue
        raise TypeError(
            f"Unsupported select entry {type(spec).__name__}: {spec!r}. "
            "Pass column names, Field, or Column."
        )
    return pa.Table.from_arrays(out_columns, names=out_names)


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def resolve_executor(context: "SqlContext | None" = None) -> SqlExecutor:
    """Pick the best available executor for the runtime.

    Polars wins when installed (full SQL surface). Otherwise the
    Arrow fallback covers the simple ``SELECT cols FROM src
    [WHERE] [LIMIT]`` shape — anything else raises a clean
    "install polars" hint at execution time.
    """
    if has_polars():
        return PolarsSqlExecutor(context=context)
    return ArrowSqlExecutor(context=context)
