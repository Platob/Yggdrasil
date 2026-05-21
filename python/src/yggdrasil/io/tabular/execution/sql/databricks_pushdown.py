"""Whole-query pushdown into a Databricks SQL warehouse.

When every source the SQL touches resolves to a
:class:`yggdrasil.databricks.table.table.Table` (and they all share one
:class:`DatabricksClient`), it's almost always faster to ship the
SQL straight to the warehouse than to pull batches over the wire and
re-do joins / aggregates in pyarrow on the driver. This module
implements that "ship the whole query" pushdown.

How it works
------------

1. Parse the query with sqlglot.
2. Walk the AST, for each :class:`sqlglot.expressions.Table`
   reference look the name up in the engine's catalog.
3. If every hit is a Databricks ``Table`` and they all share one
   client, rewrite each table reference to its
   ``<catalog>.<schema>.<name>`` form and render the SQL back.
4. Submit the rewritten SQL through the shared client's
   :class:`SQLEngine` and return its :class:`StatementResult`.

Returns ``None`` when:

- Any referenced name isn't registered (caller will fall through to
  the planner, which raises a friendly error).
- Any referenced name resolves to a non-Databricks Tabular.
- Two referenced tables come from different clients (no obvious
  common warehouse).
- sqlglot couldn't render the rewritten AST in the requested dialect.

Why duck-typed instead of ``isinstance(t, Table)``
--------------------------------------------------

Importing :mod:`yggdrasil.databricks.table.table` pulls in the
Databricks SDK and a few transitive dependencies. We don't want a
plain ``import yggdrasil.io.tabular.execution.sql`` to require any of that on a base
install. Instead, the dispatcher matches the *class identity*
through ``type().__module__`` + ``type().__name__`` — same effect,
zero import cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.io.tabular.execution.expr import Expression, Predicate
from yggdrasil.io.tabular import Tabular

from yggdrasil.io.tabular.execution.sql.dialect import Dialect, resolve_dialect


if TYPE_CHECKING:
    from yggdrasil.data.statement import StatementResult
    from yggdrasil.io.tabular.execution.sql.catalog import SqlContext


__all__ = ["try_databricks_pushdown", "is_databricks_table"]


_DBR_TABLE_MODULE = "yggdrasil.databricks.table.table"
_DBR_TABLE_CLASSNAME = "Table"


def is_databricks_table(obj: Any) -> bool:
    """Duck-type check for a :class:`yggdrasil.databricks.table.table.Table`.

    Uses class identity (module + qualname) so we don't have to
    import the Databricks SDK on a base install. Subclasses are
    accepted as long as one of their MRO entries matches.
    """
    cls = type(obj)
    for base in cls.__mro__:
        if (
            getattr(base, "__module__", "") == _DBR_TABLE_MODULE
            and base.__name__ == _DBR_TABLE_CLASSNAME
        ):
            return True
    return False


def try_databricks_pushdown(
    query: str,
    *,
    dialect: "Dialect | str | None" = None,
    catalog: "SqlContext",
    where: "Expression | str | None" = None,
) -> "Optional[StatementResult]":
    """Try to push *query* down to a Databricks warehouse, return its result.

    ``where`` (when given) is AND-merged into the SQL ``WHERE`` before
    pushdown, mirroring the composition shape :meth:`Engine.execute`
    uses on the Arrow path. Returns ``None`` when pushdown is not
    applicable (mixed sources, unknown name, render failure, etc.) —
    the caller falls back to the in-process planner.
    """
    d = resolve_dialect(dialect)
    try:
        from sqlglot import expressions as sge
        import sqlglot
    except ImportError:
        return None

    try:
        root = sqlglot.parse_one(query, read=d.value)
    except Exception:
        return None
    if root is None:
        return None

    # Resolve every Table node to a registered Tabular. If any
    # resolves to non-Databricks, abort.
    resolved: list[tuple[Any, Any]] = []
    client_id: "Optional[int]" = None
    shared_client: Any = None

    for tbl_node in root.find_all(sge.Table):
        name = _qualified(tbl_node)
        source = catalog.get(name)
        if source is None and "." in name:
            source = catalog.get(name.rsplit(".", 1)[-1])
        if source is None or not is_databricks_table(source):
            return None
        # All Tables must share one client. Compare by id() — clients
        # don't override __eq__ and we don't want a multi-client mix
        # to silently route to whichever client we saw first.
        client = getattr(source, "client", None)
        if client is None:
            return None
        cid = id(client)
        if client_id is None:
            client_id, shared_client = cid, client
        elif cid != client_id:
            return None
        resolved.append((tbl_node, source))

    if not resolved or shared_client is None:
        return None

    # Optional caller-side WHERE composition.
    if where is not None:
        predicate = (
            where if isinstance(where, Expression)
            else Expression.from_sql(where, dialect=d.value)
        )
        if not isinstance(predicate, Predicate):
            return None
        _and_merge_where(root, predicate, dialect=d, sge=sge)

    # Rewrite every Table node to its fully-qualified Databricks form.
    for tbl_node, source in resolved:
        catalog_name = getattr(source, "catalog_name", None)
        schema_name = getattr(source, "schema_name", None)
        table_name = getattr(source, "table_name", None) or getattr(source, "name", None)
        if not (catalog_name and schema_name and table_name):
            # Incomplete table identity — refuse to push rather than
            # send a SQL with NULLs in the FROM.
            return None
        tbl_node.set("catalog", sge.to_identifier(catalog_name))
        tbl_node.set("db", sge.to_identifier(schema_name))
        tbl_node.set("this", sge.to_identifier(table_name))

    try:
        rewritten = root.sql(dialect=d.value)
    except Exception:
        return None

    # Submit through the shared client's SQL engine. The returned
    # StatementResult is itself a Tabular — caller chains
    # ``.read_arrow_table()`` / ``.read_polars_frame()`` / ... as usual.
    try:
        sql_engine = shared_client.sql()
    except TypeError:
        # Some SQLEngine factories accept (catalog, schema) kwargs;
        # fall back to a positional call when the bare form fails.
        sql_engine = shared_client.sql(catalog_name=None, schema_name=None)

    return sql_engine.execute(rewritten)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qualified(node: Any) -> str:
    parts: list[str] = []
    for attr in ("catalog", "db"):
        v = getattr(node, attr, None)
        if v:
            parts.append(str(v))
    parts.append(node.name)
    return ".".join(parts)


def _and_merge_where(root: Any, predicate: Predicate, *, dialect: Dialect, sge: Any) -> None:
    """AND-merge *predicate* into the SELECT's existing WHERE.

    Mirrors the composition the legacy yggdrasil.sql.sql does: the
    builder-side predicate composes with whatever the SQL string
    already says, without re-stringifying the original.
    """
    if not isinstance(root, sge.Select):
        return
    extra_sql = predicate.to_sql(flavor=dialect.value)
    extra_node = sge.condition(extra_sql, dialect=dialect.value)
    existing_where = root.args.get("where")
    if existing_where is not None and existing_where.this is not None:
        merged = sge.And(this=existing_where.this, expression=extra_node)
        root.set("where", sge.Where(this=merged))
    else:
        root.set("where", sge.Where(this=extra_node))
