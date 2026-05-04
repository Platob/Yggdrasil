"""SQL string helpers for the Postgres backend.

Postgres identifiers are quoted with double quotes (vs. Databricks'
backticks) and embedded ``"`` chars get doubled. Literals follow the
SQL standard: single quotes around the value with ``'`` doubled.

All helpers here are pure — no DB calls, no caching. They're shared
by the resource objects (:class:`Catalog`, :class:`Schema`,
:class:`Table`) and the statement / executor layer for DDL string
assembly.
"""

from __future__ import annotations

import base64
import datetime as _dt
import decimal as _dec
import re
from typing import Any, Iterable, Sequence

__all__ = [
    "quote_ident",
    "quote_qualified_ident",
    "split_qualified_ident",
    "escape_sql_string",
    "sql_literal",
    "parse_dotted_name",
    "DEFAULT_SCHEMA",
]


DEFAULT_SCHEMA = "public"


_DOTTED_PART_RE = re.compile(
    # Either a quoted "..." segment (with "" escapes inside) or a
    # bare identifier of [A-Za-z0-9_$]+. Anything else is parsed as
    # a literal ``.`` separator.
    r'"((?:[^"]|"")*)"|([A-Za-z_][A-Za-z0-9_$]*)'
)


def quote_ident(ident: str) -> str:
    """Double-quote a Postgres identifier, escaping embedded quotes."""
    if ident is None:
        raise ValueError("Cannot quote None as a Postgres identifier")
    escaped = str(ident).replace('"', '""')
    return f'"{escaped}"'


def quote_qualified_ident(name: str | Iterable[str]) -> str:
    """Quote each segment of a dotted (or pre-split) identifier.

    ``"public.users"`` → ``"public"."users"`` — segments are split on
    unquoted dots, so an already-quoted ``'"foo.bar"'`` stays atomic.
    Pass a list/tuple to skip parsing entirely.
    """
    if isinstance(name, (list, tuple)):
        parts: Sequence[str] = name
    else:
        parts = split_qualified_ident(str(name))
    return ".".join(quote_ident(p) for p in parts if p)


def split_qualified_ident(name: str) -> list[str]:
    """Split a dotted identifier preserving quoted segments.

    Handles ``database.schema.table`` and the quoted variant
    ``"My DB"."public"."users"`` where dots inside the quotes are
    not separators.
    """
    if not name:
        return []
    parts: list[str] = []
    for match in _DOTTED_PART_RE.finditer(name):
        quoted, bare = match.group(1), match.group(2)
        if quoted is not None:
            parts.append(quoted.replace('""', '"'))
        elif bare is not None:
            parts.append(bare)
    return parts


def parse_dotted_name(
    name: str | None,
    *,
    catalog_name: str | None = None,
    schema_name: str | None = None,
    table_name: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve a 1-, 2-, or 3-part dotted name with explicit overrides.

    The overrides win when they are non-None — callers pre-bind a
    schema/catalog from a parent resource and only need to fill in
    the missing pieces from ``name``. Returns ``(catalog, schema,
    table)``; any missing segment is ``None`` (the caller decides
    whether to raise).
    """
    if name:
        parts = split_qualified_ident(name)
    else:
        parts = []
    if len(parts) > 3:
        raise ValueError(
            f"Postgres dotted name must have 1, 2, or 3 segments; "
            f"got {len(parts)} in {name!r}."
        )
    parsed_catalog: str | None = None
    parsed_schema: str | None = None
    parsed_table: str | None = None
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


def escape_sql_string(s: str) -> str:
    """Escape a Python string for embedding inside ``'...'`` SQL literals."""
    return s.replace("'", "''")


def sql_literal(value: Any) -> str:
    """Render a Python value as a Postgres SQL literal string.

    Supports the partition-key shaped subset: ``None`` / bool / int /
    float / Decimal / date / datetime / bytes / str. Anything else
    falls back to a quoted ``str(value)`` — safe but the type may
    not match the column.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, _dec.Decimal)):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, _dt.datetime):
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'"
    if isinstance(value, _dt.date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, (bytes, bytearray, memoryview)):
        # bytea hex literal; standard-conforming connections accept
        # the ``E'\\x...'`` form too, but ``'\x...'`` is the canonical
        # ``bytea`` output format and parses everywhere.
        b64 = base64.b16encode(bytes(value)).decode("ascii").lower()
        return f"'\\x{b64}'"
    return "'" + escape_sql_string(str(value)) + "'"
