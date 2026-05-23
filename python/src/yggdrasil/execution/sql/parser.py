"""SQL parser — sqlglot wrapper with a couple of yggdrasil-specific knobs.

Two entry points cover ~everything the executor needs to know:

- :func:`parse` — full statement → sqlglot AST root. Handles
  multi-statement scripts (returns the first; warns when more)
  and surfaces the offending dialect on a parse error rather than
  the bare sqlglot message.
- :func:`extract_sources` — walk the AST and yield every base table
  reference. Used by the executor to figure out which contexts to
  pin into a polars :class:`SQLContext` (or which to scan in the
  Arrow-only fallback) without having to learn every SQL feature
  individually.

We *also* re-export :func:`Expression.from_sql` shaped for predicate
fragments — callers that just want a ``WHERE``-style boolean lift
from a string to our :class:`yggdrasil.data.expr.Expression` AST
should reach for :func:`parse_predicate` here so they don't have
to remember which ``from_sql`` lives on which class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from yggdrasil.execution.expr import Expression

from .dialect import Dialect, resolve_dialect
from yggdrasil.lazy_imports import sqlglot_expressions, sqlglot_module

if TYPE_CHECKING:
    pass

__all__ = [
    "parse",
    "parse_many",
    "parse_predicate",
    "extract_sources",
    "is_query",
    "SqlParseError",
]


class SqlParseError(ValueError):
    """Raised when sqlglot fails to parse the input.

    Wraps the sqlglot exception so callers can distinguish "this
    SQL is broken" from "the executor doesn't understand this
    feature yet" (which raises :class:`NotImplementedError`).
    """


def parse(
    query: str,
    *,
    dialect: "Dialect | str | None" = None,
) -> Any:
    """Parse *query* and return the first statement's sqlglot AST root.

    Multi-statement scripts log a soft note and return the first;
    callers that need every statement should use :func:`parse_many`
    instead. Empty input raises with a useful message rather than
    sqlglot's terse default.
    """
    if not query or not query.strip():
        raise SqlParseError(
            "Cannot parse an empty SQL query. Pass a SELECT / WITH / "
            "VALUES / SHOW / DESCRIBE statement."
        )
    statements = parse_many(query, dialect=dialect)
    if not statements:
        raise SqlParseError(
            f"sqlglot parsed {query!r} into zero statements; this usually "
            "means the input was only comments or whitespace."
        )
    return statements[0]


def parse_many(
    query: str,
    *,
    dialect: "Dialect | str | None" = None,
) -> "list[Any]":
    """Parse *query* and return every statement.

    sqlglot's ``parse`` returns a list with ``None`` placeholders
    for empty trailing statements (a stray semicolon, a final
    comment block); we strip them so callers don't have to.
    """
    sg = sqlglot_module()
    d = resolve_dialect(dialect)
    try:
        parsed = sg.parse(query, read=d.value)
    except sg.errors.ParseError as exc:  # type: ignore[attr-defined]
        raise SqlParseError(
            f"sqlglot failed to parse {query!r} as {d.value} SQL: {exc}. "
            "Check the dialect (default 'databricks') matches your "
            "syntax — pass dialect='postgres' / 'mysql' / 'sqlite' / "
            "'ansi' if needed."
        ) from exc
    return [s for s in parsed if s is not None]


def parse_predicate(
    sql: str,
    *,
    dialect: "Dialect | str | None" = None,
) -> Expression:
    """Lift a SQL boolean fragment (``a > 5 AND b IS NOT NULL``) to our AST.

    Thin wrapper around :meth:`Expression.from_sql` so callers that
    are already in the :mod:`yggdrasil.sql` namespace don't have to
    cross-import. Useful for composing a ``WHERE`` clause from a
    string with builder-side ``col(...).is_in([...])`` predicates
    via ``Expression.merge_with``.
    """
    return Expression.from_sql(sql, dialect=resolve_dialect(dialect).value)


def extract_sources(node: Any) -> "list[str]":
    """Names of every base table referenced by *node*, in tree-walk order.

    De-duplicated, preserves first-seen order. CTE definitions
    register their alias as a source too — callers compose the
    polars :class:`SQLContext` from this list and rely on the SQL
    engine itself to wire CTEs to their definitions.
    """
    seen: dict[str, None] = {}
    for name in _walk_tables(node):
        seen.setdefault(name, None)
    return list(seen.keys())


def _walk_tables(node: Any) -> Iterator[str]:
    sge = sqlglot_expressions()
    if isinstance(node, sge.Table):
        # ``schema.table`` / ``catalog.schema.table`` get joined with
        # dots so the executor can match either fully-qualified or
        # leaf-only registrations.
        parts = []
        if getattr(node, "catalog", None):
            parts.append(node.catalog)
        if getattr(node, "db", None):
            parts.append(node.db)
        parts.append(node.name)
        yield ".".join(parts)
        # Don't yield the leaf again as a separate name when the
        # qualified form was emitted; one entry per physical
        # reference is what the caller wants.
        return
    for child in node.args.values() if hasattr(node, "args") else ():
        if isinstance(child, list):
            for c in child:
                if c is not None:
                    yield from _walk_tables(c)
        elif child is not None and hasattr(child, "args"):
            yield from _walk_tables(child)


def is_query(node: Any) -> bool:
    """Whether *node* is a row-producing statement (``SELECT`` / ``WITH`` / ``UNION`` / …).

    Used by the executor to refuse to run side-effecting DDL/DML
    against the in-process catalog — those wouldn't make sense on
    a :class:`Tabular` source anyway.
    """
    sge = sqlglot_expressions()
    return isinstance(
        node,
        (sge.Select, sge.Subquery, sge.Union, sge.SetOperation)
        if hasattr(sge, "SetOperation")
        else (sge.Select, sge.Subquery, sge.Union),
    ) or _has_query_root(node, sge)


def _has_query_root(node: Any, sge: Any) -> bool:
    # ``WITH cte AS (...) SELECT ...`` parses as a Select with a
    # ``with`` arg — the root isn't a CTE per se but it IS a query.
    if isinstance(node, sge.Select):
        return True
    if hasattr(sge, "Values") and isinstance(node, sge.Values):
        return True
    return False
