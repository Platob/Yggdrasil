"""Shared SQL string helpers for the Databricks SQL package."""

from __future__ import annotations

import base64
import re
from typing import Any
import hashlib

__all__ = [
    "DEFAULT_TAG_COLLATION",
    "_build_fk_constraint_sql",
    "_build_pk_constraint_sql",
    "_build_table_constraints_sql",
    "databricks_tag_literal",
    "normalize_databricks_collation",
    "_safe_constraint_name",
    "_safe_str",
    "_sql_str",
    "escape_sql_string",
    "quote_ident",
    "quote_qualified_ident",
    "quote_principal",
    "sql_literal",
]

DEFAULT_TAG_COLLATION = None
_COLLATION_SUFFIXES = ("CI", "AI", "RTRIM")
_LOCALE_COLLATION_BASE_RE = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$")


def _safe_constraint_name(*parts: object, max_length: int = 255) -> str:
    """Build a stable Databricks-safe constraint name.

    The visible prefix is normalized for readability. A short hash derived from
    the raw input parts guarantees stability and avoids collisions caused by
    normalization.
    """
    raw_parts: tuple[str, ...] = tuple("" if part is None else str(part) for part in parts)

    tokens: list[str] = []
    for text in raw_parts:
        text = text.strip()
        if not text:
            continue
        text = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in text)
        text = "_".join(piece for piece in text.split("_") if piece)
        if text:
            tokens.append(text)

    prefix = "_".join(tokens) if tokens else "constraint"
    digest = hashlib.sha256("\x1f".join(raw_parts).encode("utf-8")).hexdigest()[:12]

    name = f"{prefix}_{digest}"
    if len(name) <= max_length:
        return name

    prefix_max = max_length - len(digest) - 1
    if prefix_max <= 0:
        return digest[:max_length]

    prefix = prefix[:prefix_max].rstrip("_")
    if not prefix:
        return digest[:max_length]

    return f"{prefix}_{digest}"


def _safe_str(value: Any) -> str:
    if not value:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (bytes, memoryview, bytearray)):
        return value.decode()

    return str(value)


def _sql_str(value: Any) -> str:
    return "'" + _safe_str(value).replace("'", "''") + "'"


def normalize_databricks_collation(collation: str | None) -> str | None:
    """Normalize and validate a Databricks collation name.

    Databricks supports the built-in `UNICODE*`, `UTF8_BINARY*`, and
    `UTF8_LCASE*` families plus ICU locale collations with optional `_CI`,
    `_AI`, and `_RTRIM` suffixes.
    """
    if collation is None:
        return None

    raw = str(collation).strip()
    if not raw:
        raise ValueError("Collation cannot be empty")

    parts = raw.split("_")
    suffixes: list[str] = []
    while parts and parts[-1].upper() in _COLLATION_SUFFIXES:
        suffix = parts.pop().upper()
        if suffix in suffixes:
            raise ValueError(f"Invalid Databricks collation: {collation!r}")
        suffixes.append(suffix)

    suffixes.reverse()
    base = "_".join(parts)
    if not base:
        raise ValueError(f"Invalid Databricks collation: {collation!r}")

    base_upper = base.upper()
    suffix_set = set(suffixes)

    if base_upper == "UTF8_BINARY":
        if suffix_set - {"RTRIM"}:
            raise ValueError(f"Invalid Databricks collation: {collation!r}")
        return "_".join([base_upper, *suffixes]) if suffixes else base_upper

    if base_upper == "UTF8_LCASE":
        if suffix_set - {"RTRIM"}:
            raise ValueError(f"Invalid Databricks collation: {collation!r}")
        return "_".join([base_upper, *suffixes]) if suffixes else base_upper

    if base_upper == "UNICODE":
        ordered = [part for part in ("CI", "AI", "RTRIM") if part in suffix_set]
        return "_".join([base_upper, *ordered]) if ordered else base_upper

    if not _LOCALE_COLLATION_BASE_RE.match(base):
        raise ValueError(f"Invalid Databricks collation: {collation!r}")

    ordered = [part for part in ("CI", "AI", "RTRIM") if part in suffix_set]
    return "_".join([base, *ordered]) if ordered else base


def databricks_tag_literal(
    value: Any,
    *,
    collation: str | None = DEFAULT_TAG_COLLATION,
) -> str:
    """Build a SQL string literal for Databricks tag DDL.

    Tag keys and values can be annotated with an explicit collation to avoid
    dependence on the session or object default collation.
    """
    literal = _sql_str(value)
    normalized = normalize_databricks_collation(collation)
    if normalized is None:
        return literal
    return f"{literal} COLLATE {normalized}"


def _parse_fk_ref(ref: str) -> tuple[str, list[str]]:
    if not ref or not isinstance(ref, str):
        raise ValueError(f"Invalid foreign key ref: {ref!r}")

    parts = [part.strip() for part in ref.split(".") if part.strip()]
    if len(parts) < 4:
        raise ValueError(
            "Foreign key ref must be "
            "'catalog.schema.table.column' or "
            "'catalog.schema.table.col1,col2,...', "
            f"got {ref!r}"
        )

    ref_table = ".".join(parts[:3])
    ref_cols_expr = ".".join(parts[3:])
    ref_cols = [col.strip() for col in ref_cols_expr.split(",") if col.strip()]
    if not ref_cols:
        raise ValueError(f"No referenced columns found in foreign key ref: {ref!r}")

    return ref_table, ref_cols


def _build_pk_constraint_sql(pk_spec: Any) -> str | None:
    if not pk_spec or not getattr(pk_spec, "columns", None):
        return None

    cols: list[str] = []
    for column in pk_spec.columns:
        col_sql = quote_ident(column)
        if getattr(pk_spec, "timeseries", None) == column:
            col_sql += " TIMESERIES"
        cols.append(col_sql)

    parts: list[str] = []
    if getattr(pk_spec, "constraint_name", None):
        parts.append(f"CONSTRAINT {quote_ident(pk_spec.constraint_name)}")

    parts.append(f"PRIMARY KEY ({', '.join(cols)})")
    parts.append("NOT ENFORCED")

    if getattr(pk_spec, "rely", False):
        parts.append("RELY")

    return " ".join(parts)


def _build_fk_constraint_sql(table_name: str, fk: Any) -> str:
    ref_table, ref_columns = _parse_fk_ref(fk.ref)

    parts: list[str] = []
    cname = _safe_constraint_name(
        getattr(fk, "constraint_name", None)
        or f"{table_name}_{fk.column}_{ref_table}_{'_'.join(ref_columns)}_fk"
    )
    parts.append(f"CONSTRAINT {quote_ident(cname)}")
    parts.append(f"FOREIGN KEY ({quote_ident(fk.column)})")
    parts.append(f"REFERENCES {quote_qualified_ident(ref_table)}")

    if ref_columns:
        parts.append("(" + ", ".join(quote_ident(column) for column in ref_columns) + ")")

    parts.append("NOT ENFORCED")

    if getattr(fk, "rely", False):
        parts.append("RELY")
    if getattr(fk, "match_full", False):
        parts.append("MATCH FULL")
    if getattr(fk, "on_update_no_action", False):
        parts.append("ON UPDATE NO ACTION")
    if getattr(fk, "on_delete_no_action", False):
        parts.append("ON DELETE NO ACTION")

    return " ".join(parts)


def _build_table_constraints_sql(
    table_name: str,
    pk_spec: Any,
    fk_specs: list[Any],
) -> list[str]:
    constraints: list[str] = []

    pk_sql = _build_pk_constraint_sql(pk_spec)
    if pk_sql:
        constraints.append(pk_sql)

    for fk in fk_specs:
        constraints.append(_build_fk_constraint_sql(table_name, fk))

    return constraints


def escape_sql_string(s: str) -> str:
    """Escape a Python string for safe embedding in a single-quoted SQL literal."""
    return s.replace("'", "''")


def quote_ident(ident: str) -> str:
    """Quote a SQL identifier using backticks, escaping embedded backticks."""
    escaped = ident.replace("`", "``")
    return f"`{escaped}`"


def quote_qualified_ident(name: str) -> str:
    """Backtick-quote each segment of a dotted identifier."""
    parts = [part.strip() for part in str(name).split(".") if part.strip()]
    return ".".join(quote_ident(part) for part in parts)


def quote_principal(name: str) -> str:
    """Backtick-quote a Databricks principal name as a single identifier."""
    return quote_ident(str(name).strip())


def sql_literal(value: Any) -> str:
    """Convert a Python value into a Databricks SQL literal string."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        b64 = base64.b64encode(bytes(value)).decode("ascii")
        return f"'{b64}'"

    text = str(value).strip()
    lowered = text.lower()
    if lowered.startswith("sql:"):
        return text[4:].strip()
    if lowered in ("null", "none"):
        return "NULL"
    if lowered in ("true", "false"):
        return lowered.upper()
    try:
        float(text)
        return text
    except ValueError:
        pass
    return "'" + escape_sql_string(text) + "'"
