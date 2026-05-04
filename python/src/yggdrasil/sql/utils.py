"""Identifier quoting + literal helpers for :mod:`yggdrasil.sql`.

Thin layer on top of the per-dialect quote chars used by the
expression SQL emitter — same source of truth so a ``WHERE`` clause
rendered via :meth:`Expression.to_sql` and a CTE name quoted here
agree on escape rules.

Single-source: identifier quote chars come from
``yggdrasil.data.expr.backends.sql._IDENT_QUOTES``. Literal rendering
defers to the same emitter's ``_render_literal`` so date / bytes /
NULL handling stays consistent.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from yggdrasil.data.expr.backends.sql import (
    _IDENT_QUOTES,
    _render_literal,
)
from .dialect import Dialect, resolve_dialect

__all__ = [
    "quote_ident",
    "quote_qualified_ident",
    "split_qualified_ident",
    "sql_literal",
    "parse_dotted_name",
    "is_valid_identifier",
]


# Same shape as ``yggdrasil.postgres.sql_utils._DOTTED_PART_RE`` but
# accepts both double-quoted (ANSI / Postgres / SQLite) and
# backtick-quoted (Databricks / MySQL) segments. Bare identifiers
# follow the SQL rule: leading letter or underscore, then alnum /
# underscore / dollar.
_DOTTED_PART_RE = re.compile(
    r'"((?:[^"]|"")*)"'           # "double-quoted"
    r"|`((?:[^`]|``)*)`"          # `backtick-quoted`
    r"|([A-Za-z_][A-Za-z0-9_$]*)"  # bare-identifier
)

_BARE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def quote_ident(ident: str, dialect: "Dialect | str | None" = None) -> str:
    """Quote *ident* for the chosen dialect, escaping embedded quotes.

    Postgres / SQLite / ANSI use ``"…"`` and double the embedded
    ``"``. Databricks / MySQL use backticks and double the embedded
    backtick. ``None`` resolves through :func:`resolve_dialect`, so
    the project default (Databricks) wins.
    """
    if ident is None:
        raise ValueError("Cannot quote None as a SQL identifier.")
    d = resolve_dialect(dialect)
    quote, escaped = _IDENT_QUOTES.get(d, _IDENT_QUOTES[Dialect.ANSI])
    return f"{quote}{str(ident).replace(quote, escaped)}{quote}"


def quote_qualified_ident(
    name: "str | Iterable[str]",
    dialect: "Dialect | str | None" = None,
) -> str:
    """Quote each segment of a dotted (or pre-split) identifier.

    ``"public.users"`` → ``"public"."users"`` on Postgres,
    ``` `public`.`users` ``` on Databricks. Pre-split list/tuple
    inputs skip parsing and just get quoted segment-by-segment.
    """
    if isinstance(name, (list, tuple)):
        parts: Sequence[str] = name
    else:
        parts = split_qualified_ident(str(name))
    return ".".join(quote_ident(p, dialect) for p in parts if p)


def split_qualified_ident(name: str) -> list[str]:
    """Split a dotted identifier preserving quoted segments.

    Handles ``catalog.schema.table`` and quoted variants like
    ``"My Catalog"."schema"."tbl"`` / ```` `my cat`.`schema`.`tbl` ````
    where dots inside the quoted segment are not separators.
    """
    if not name:
        return []
    parts: list[str] = []
    for match in _DOTTED_PART_RE.finditer(name):
        dq, bq, bare = match.group(1), match.group(2), match.group(3)
        if dq is not None:
            parts.append(dq.replace('""', '"'))
        elif bq is not None:
            parts.append(bq.replace("``", "`"))
        elif bare is not None:
            parts.append(bare)
    return parts


def parse_dotted_name(
    name: "str | None",
    *,
    catalog_name: "str | None" = None,
    schema_name: "str | None" = None,
    table_name: "str | None" = None,
) -> "tuple[str | None, str | None, str | None]":
    """Resolve a 1-, 2-, or 3-part dotted name with explicit overrides.

    Overrides win when non-None — callers pre-binding a
    catalog/schema only need to fill in what's missing from
    ``name``. Returns ``(catalog, schema, table)``; any segment
    not derivable is ``None`` (the caller decides whether to raise).
    """
    parts = split_qualified_ident(name) if name else []
    if len(parts) > 3:
        raise ValueError(
            f"SQL dotted name must have 1, 2, or 3 segments; got "
            f"{len(parts)} in {name!r}. Examples: 'tbl', 'schema.tbl', "
            "'catalog.schema.tbl'."
        )
    parsed_catalog: "str | None" = None
    parsed_schema: "str | None" = None
    parsed_table: "str | None" = None
    if len(parts) == 3:
        parsed_catalog, parsed_schema, parsed_table = parts
    elif len(parts) == 2:
        parsed_schema, parsed_table = parts
    elif len(parts) == 1:
        parsed_table = parts[0]
    return (
        catalog_name if catalog_name is not None else parsed_catalog,
        schema_name if schema_name is not None else parsed_schema,
        table_name if table_name is not None else parsed_table,
    )


def is_valid_identifier(name: str) -> bool:
    """Whether *name* matches the bare SQL identifier shape (no quoting needed)."""
    return bool(_BARE_IDENT_RE.match(name or ""))


def sql_literal(value: Any, dialect: "Dialect | str | None" = None) -> str:
    """Render a Python value as a SQL literal for the chosen dialect.

    Defers to the expression backend's ``_render_literal`` so date /
    timestamp / bytes / NULL formatting matches what
    :meth:`Expression.to_sql` emits — one source of truth keeps
    round-trips clean.
    """
    return _render_literal(value, resolve_dialect(dialect))
