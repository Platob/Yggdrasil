"""Shared SQL string helpers for the Databricks SQL package."""

from __future__ import annotations

import base64
import hashlib
import re
from fnmatch import fnmatchcase
from typing import Any, Callable, Optional

__all__ = [
    "DEFAULT_TAG_COLLATION",
    "MAX_TABLE_NAME_LEN",
    "_qualify_fk_ref",
    "databricks_tag_literal",
    "is_glob_pattern",
    "name_matcher",
    "normalize_databricks_collation",
    "safe_table_name",
    "_safe_str",
    "_sql_str",
    "escape_sql_string",
    "quote_ident",
    "quote_qualified_ident",
    "quote_principal",
    "sql_literal",
]


# Unity Catalog rejects identifiers longer than 255 chars. Generated table
# names (e.g. derived from user input or composed keys) can blow past that,
# so any path that builds a name needs a length-safe normalizer.
MAX_TABLE_NAME_LEN = 255


def safe_table_name(name: str | None, *, limit: int = MAX_TABLE_NAME_LEN) -> str | None:
    """Return *name* unchanged if it fits, otherwise truncate + hash.

    Databricks Unity Catalog caps identifiers at 255 chars. When a caller
    hands in something longer, keep a deterministic prefix and append a
    BLAKE2b digest of the full name. The result fits the limit, stays
    stable for the same input, and uses only identifier-safe hex chars.

    ``None`` and empty strings pass through unchanged so this stays safe to
    call before defaults have been resolved.
    """
    if not name or len(name) <= limit:
        return name

    # 16-byte digest → 32 hex chars. Collision-safe for practical table-name
    # workloads, and leaves plenty of room for a recognizable prefix.
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=16).hexdigest()
    keep = limit - len(digest) - 1  # 1 for the underscore separator
    if keep <= 0:
        return digest[:limit]
    return f"{name[:keep]}_{digest}"


def is_glob_pattern(value: Any) -> bool:
    """Return ``True`` when *value* is a string carrying a ``*`` wildcard."""
    return isinstance(value, str) and "*" in value


def name_matcher(pattern: str | None) -> Optional[Callable[[str | None], bool]]:
    """Build a predicate for filtering resource names with optional globbing.

    * ``None`` → ``None`` (caller skips filtering entirely).
    * No ``*`` → exact-match predicate (case-sensitive).
    * Contains ``*`` → case-insensitive :func:`fnmatch.fnmatchcase` predicate.

    The predicate accepts ``None`` defensively and treats it as an empty name.
    """
    if pattern is None:
        return None
    if "*" in pattern:
        folded = pattern.casefold()
        return lambda name: fnmatchcase((name or "").casefold(), folded)
    return lambda name: name == pattern

DEFAULT_TAG_COLLATION = None
_COLLATION_SUFFIXES = ("CI", "AI", "RTRIM")
_LOCALE_COLLATION_BASE_RE = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$")


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


def _parse_fk_ref(
    ref: str,
    *,
    default_catalog: str | None = None,
    default_schema: str | None = None,
) -> tuple[str, list[str]]:
    """Parse a foreign-key ref into ``(ref_table, ref_columns)``.

    Accepts 2-, 3-, or 4-part dotted refs; missing catalog/schema segments are
    filled from ``default_catalog`` / ``default_schema`` when available:

        - ``"table.col"``                          (needs both defaults)
        - ``"schema.table.col"``                   (needs default_catalog)
        - ``"catalog.schema.table.col"``           (fully qualified)
        - ``"catalog.schema.table.col1,col2,..."`` (composite ref)
    """
    if not ref or not isinstance(ref, str):
        raise ValueError(f"Invalid foreign key ref: {ref!r}")

    parts = [part.strip() for part in ref.split(".") if part.strip()]
    if len(parts) == 2 and default_catalog and default_schema:
        parts = [default_catalog, default_schema, *parts]
    elif len(parts) == 3 and default_catalog:
        parts = [default_catalog, *parts]

    if len(parts) < 4:
        raise ValueError(
            "Foreign key ref must resolve to "
            "'catalog.schema.table.column' or "
            "'catalog.schema.table.col1,col2,...'; "
            f"got {ref!r}"
        )

    ref_table = ".".join(parts[:3])
    ref_cols_expr = ".".join(parts[3:])
    ref_cols = [col.strip() for col in ref_cols_expr.split(",") if col.strip()]
    if not ref_cols:
        raise ValueError(f"No referenced columns found in foreign key ref: {ref!r}")

    return ref_table, ref_cols


def _qualify_fk_ref(
    ref: str,
    *,
    default_catalog: str | None = None,
    default_schema: str | None = None,
) -> str:
    """Normalize a foreign-key ref to the ``catalog.schema.table.col[,col]`` form."""
    ref_table, ref_cols = _parse_fk_ref(
        ref,
        default_catalog=default_catalog,
        default_schema=default_schema,
    )
    return f"{ref_table}.{','.join(ref_cols)}"


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


def _render_sql_literal(value: Any) -> str:
    """Render a Python value as a Databricks SQL literal.

    Used to inline distinct partition keys into the merge ON clause so
    Delta can prune files instead of acquiring a whole-table OCC read set.
    Only the narrow set of types Delta actually allows as partition columns
    is supported — strings, bools, ints/floats, dates, datetimes, decimals,
    bytes.  Anything else falls back to a quoted string repr, which is
    safe but may not match.
    """
    import datetime as _dt
    import decimal as _dec

    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, _dec.Decimal)):
        return str(value)
    if isinstance(value, float):
        # NaN/Inf aren't valid partition values; let Delta reject if seen.
        return repr(value)
    if isinstance(value, _dt.datetime):
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'"
    if isinstance(value, _dt.date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "X'" + bytes(value).hex() + "'"
    # Strings and fallback: single-quote and escape embedded quotes.
    return "'" + str(value).replace("'", "''") + "'"


