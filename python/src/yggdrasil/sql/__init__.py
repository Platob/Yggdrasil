"""In-process SQL on top of :class:`yggdrasil.io.buffer.Tabular`.

The single line you'll write 95% of the time::

    import yggdrasil.sql as ysql

    ysql.register("trades", trades_io)              # name → Tabular
    result = ysql.sql("SELECT symbol, SUM(qty) AS total "
                       "FROM trades GROUP BY symbol "
                       "ORDER BY total DESC LIMIT 10")
    table = result.read_arrow_table()                # one read…
    polars_df = result.read_polars_frame()           # …or many; cached.

What this module is
-------------------

A thin SQL surface that:

- Parses SQL via :mod:`sqlglot`. Default flavor is **Databricks**
  (Spark-SQL — backticks, ``ILIKE``, very close to the Postgres
  SELECT shape). ``dialect='postgres'`` / ``'sqlite'`` / ``'mysql'``
  / ``'ansi'`` switch flavors when you need them.
- Resolves table references against a :class:`SqlContext` — a dict
  of ``name → Tabular`` (or anything :func:`coerce_source`
  knows how to lift: pyarrow Table/Batch, polars / pandas / Spark
  DataFrame, ``list[dict]``, path string).
- Executes through :class:`polars.SQLContext` when polars is
  installed (full SQL surface — joins, aggregations, CTEs, window
  functions). Falls back to a minimal Arrow-only path for the
  ``SELECT cols FROM src [WHERE] [LIMIT]`` shape on base installs.
- Returns a :class:`SqlStatementResult` — a real
  :class:`yggdrasil.data.statement.StatementResult` (so it's a
  full :class:`Tabular` with ``read_arrow_table`` /
  ``read_polars_frame`` / ``read_pandas_frame`` /
  ``read_spark_frame`` / ``to_records`` / …).
- Persists the materialized result through
  :attr:`Tabular._persisted_data` so repeat reads are cache
  hits. Default holder is :class:`MemoryArrowIO`; pass
  ``persist='path', path='...'`` to spill to a parquet file or
  folder instead.

Why this shape
--------------

Existing pieces in the repo do most of the work — we just stitch
them together:

- :class:`yggdrasil.io.buffer.Tabular` is the universal source
  surface. Anything that yields Arrow batches plugs in.
- :class:`yggdrasil.data.expr.Expression` is the cross-engine
  predicate AST. The ``where=`` kwarg accepts one and composes
  it with the SQL ``WHERE`` via AND, so a builder-side filter
  doesn't have to be re-stringified.
- :class:`yggdrasil.data.expr.Selector` is the cross-engine
  projection. The ``select=`` kwarg accepts a list of names /
  selectors / columns to apply on the way out.
- :class:`yggdrasil.data.statement.StatementResult` is the
  lifecycle + Arrow-IO base.
- :class:`yggdrasil.io.buffer.memory.MemoryArrowIO` is the
  zero-copy in-memory holder. ParquetIO handles the on-disk
  spill case.

That keeps :mod:`yggdrasil.sql` itself small and focused on the
SQL-specific bits: parse, resolve, execute, persist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping

from yggdrasil.data.expr import Expression

from .catalog import (
    SqlContext,
    coerce_source,
    default_context,
    deregister,
    register,
    registered,
)
from .dialect import DEFAULT_DIALECT, Dialect, resolve_dialect
from .executor import (
    ArrowSqlExecutor,
    PolarsSqlExecutor,
    SqlExecutor,
    resolve_executor,
)
from .lib import has_polars, has_sqlglot
from .parser import (
    SqlParseError,
    extract_sources,
    is_query,
    parse,
    parse_many,
    parse_predicate,
)
from .statement import PersistTarget, SqlPreparedStatement, SqlStatementResult
from .utils import (
    is_valid_identifier,
    parse_dotted_name,
    quote_ident,
    quote_qualified_ident,
    split_qualified_ident,
    sql_literal,
)

if TYPE_CHECKING:
    pass


__all__ = [
    "sql",
    "register",
    "deregister",
    "registered",
    "default_context",
    "SqlContext",
    "SqlExecutor",
    "PolarsSqlExecutor",
    "ArrowSqlExecutor",
    "SqlPreparedStatement",
    "SqlStatementResult",
    "PersistTarget",
    "Dialect",
    "DEFAULT_DIALECT",
    "resolve_dialect",
    "coerce_source",
    "parse",
    "parse_many",
    "parse_predicate",
    "extract_sources",
    "is_query",
    "SqlParseError",
    "quote_ident",
    "quote_qualified_ident",
    "split_qualified_ident",
    "parse_dotted_name",
    "sql_literal",
    "is_valid_identifier",
    "has_polars",
    "has_sqlglot",
    "resolve_executor",
]


def sql(
    query: str,
    *,
    sources: "Mapping[str, Any] | None" = None,
    context: "SqlContext | None" = None,
    where: "Expression | str | None" = None,
    select: "Iterable[Any] | None" = None,
    persist: PersistTarget = "memory",
    path: "str | None" = None,
    dialect: "Dialect | str | None" = None,
    executor: "SqlExecutor | None" = None,
    wait: Any = True,
    raise_error: bool = True,
    **kwargs: Any,
) -> SqlStatementResult:
    """Execute *query* against the registered sources, return the result handle.

    Parameters
    ----------
    query
        The SQL text. Default dialect is **Databricks** (Spark-SQL,
        backtick identifiers, ``ILIKE`` available); pass
        ``dialect='postgres'`` etc. to switch.
    sources
        Optional ``{name: source}`` overrides scoped to this
        execution. Each value is lifted to a :class:`Tabular`
        via :func:`coerce_source` (accepts pyarrow / polars /
        pandas / Spark frames, ``list[dict]``, path strings, or
        an existing :class:`Tabular`). Wins over the
        :class:`SqlContext` for the duration of the call.
    context
        :class:`SqlContext` to resolve unbound names against.
        Default: the process-wide :data:`default_context` (the
        one :func:`register` writes to).
    where
        Optional :class:`Expression` (or SQL string lifted via
        :func:`parse_predicate`) AND-merged into the parsed
        ``WHERE``. Lets a caller compose builder-side filters
        without re-stringifying.
    select
        Optional projection list applied **after** the SQL — a
        sequence of column names, :class:`Selector`, or
        :class:`Column` entries. Useful for renaming / casting on
        the way out without touching the SQL text.
    persist
        Where to land the materialized result. ``"memory"`` (the
        default) keeps it as :class:`MemoryArrowIO`. ``"path"``
        spills to a parquet file/folder under ``path``. ``None``
        skips persistence — re-reads will re-execute the engine,
        which is only what you want for one-shot consumption.
    path
        Required when ``persist='path'``. ``foo.parquet`` writes a
        single file; anything else is a folder (created on demand)
        with one parquet part per output batch.
    dialect
        SQL flavor (``'databricks'`` / ``'postgres'`` / ``'sqlite'``
        / ``'mysql'`` / ``'ansi'``). Default: Databricks.
    executor
        Override the auto-resolved backend. Default: polars when
        installed, Arrow fallback otherwise.
    wait, raise_error
        Forwarded to :meth:`SqlStatementResult.start`.

    Returns
    -------
    SqlStatementResult
        Lifecycle handle and a full :class:`Tabular` over the
        materialized result. Read via :meth:`read_arrow_table`,
        :meth:`read_polars_frame`, :meth:`read_pandas_frame`,
        :meth:`read_spark_frame`, :meth:`to_records`, etc.

    Examples
    --------

    .. code-block:: python

        import pyarrow as pa
        import yggdrasil.sql as ysql

        trades = pa.table({
            "symbol": ["AAPL", "GOOG", "AAPL"],
            "qty": [10, 5, 7],
        })
        ysql.register("trades", trades)

        result = ysql.sql(
            "SELECT symbol, SUM(qty) AS total FROM trades "
            "GROUP BY symbol ORDER BY total DESC"
        )
        result.read_polars_frame()

        # Persist to disk for big result sets:
        ysql.sql(
            "SELECT * FROM trades WHERE qty > 5",
            persist="path",
            path="/tmp/big-result",
        )
    """
    ctx = context if context is not None else default_context
    if sources:
        ctx = ctx.child(sources)

    composed_predicate: "Expression | None" = None
    if where is not None:
        composed_predicate = (
            where if isinstance(where, Expression)
            else parse_predicate(where, dialect=dialect)
        )

    prepared = SqlPreparedStatement(
        text=query,
        dialect=dialect,
        sources=ctx.snapshot(),
        predicate=composed_predicate,
        select=select,
        persist=persist,
        path=path,
        **kwargs,
    )
    chosen = executor or resolve_executor(ctx)
    return chosen.execute(prepared, wait=wait, raise_error=raise_error)
