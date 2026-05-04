"""Postgres ↔ Arrow type mapping.

Two directions:

* :func:`arrow_to_postgres_type` — render a :class:`pyarrow.DataType`
  as the Postgres column type used in ``CREATE TABLE`` / ``ALTER
  TABLE`` DDL. Defaults are conservative: no length caps on ``text``,
  ``timestamptz`` for tz-aware timestamps, ``numeric`` for arbitrary-
  precision decimals.
* :func:`postgres_to_arrow_type` — parse a Postgres type name (as
  returned from ``information_schema.columns.data_type`` /
  ``pg_attribute``) into a :class:`pyarrow.DataType`. Handles the
  ``timestamp(p) with time zone`` / ``numeric(p, s)`` / ``character
  varying(n)`` shapes.

Use :func:`arrow_to_postgres_field` / :func:`postgres_to_arrow_field`
when you also need ``NULL`` / ``NOT NULL`` rendering or a yggdrasil
:class:`Field` round-trip.
"""

from __future__ import annotations

import re
from typing import Any

import pyarrow as pa

from yggdrasil.data import Field

__all__ = [
    "arrow_to_postgres_type",
    "postgres_to_arrow_type",
    "arrow_to_postgres_field",
    "postgres_to_arrow_field",
    "arrow_schema_to_postgres_columns",
]


# ---------------------------------------------------------------------------
# Arrow → Postgres
# ---------------------------------------------------------------------------


def arrow_to_postgres_type(dtype: pa.DataType) -> str:
    """Render *dtype* as a Postgres column type string.

    Type families:

    ============  ================================
    Arrow          Postgres
    ============  ================================
    bool           boolean
    int8 / int16   smallint
    int32          integer
    int64          bigint
    uint*          (promoted to next signed family)
    float16        real
    float32        real
    float64        double precision
    decimal        numeric(p, s)
    string / utf8  text
    binary         bytea
    date32 / 64    date
    time           time
    timestamp      timestamp [with time zone]
    duration       interval
    list / map     jsonb       (rendered as JSON)
    struct         jsonb
    ============  ================================

    Unsigned integers promote one rank up to keep the value range
    representable (``uint32`` → ``bigint``, ``uint64`` → ``numeric``).
    """
    if pa.types.is_boolean(dtype):
        return "boolean"
    if pa.types.is_int8(dtype) or pa.types.is_int16(dtype):
        return "smallint"
    if pa.types.is_int32(dtype):
        return "integer"
    if pa.types.is_int64(dtype):
        return "bigint"
    if pa.types.is_uint8(dtype) or pa.types.is_uint16(dtype):
        return "integer"
    if pa.types.is_uint32(dtype):
        return "bigint"
    if pa.types.is_uint64(dtype):
        return "numeric(20, 0)"
    if pa.types.is_float16(dtype) or pa.types.is_float32(dtype):
        return "real"
    if pa.types.is_float64(dtype):
        return "double precision"
    if pa.types.is_decimal(dtype):
        precision = getattr(dtype, "precision", None)
        scale = getattr(dtype, "scale", None)
        if precision is None:
            return "numeric"
        if scale is None:
            return f"numeric({precision})"
        return f"numeric({precision}, {scale})"
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "text"
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype) or pa.types.is_fixed_size_binary(dtype):
        return "bytea"
    if pa.types.is_date(dtype):
        return "date"
    if pa.types.is_time(dtype):
        return "time"
    if pa.types.is_timestamp(dtype):
        tz = getattr(dtype, "tz", None)
        return "timestamp with time zone" if tz else "timestamp"
    if pa.types.is_duration(dtype):
        return "interval"
    if pa.types.is_null(dtype):
        # No first-class NULL type in Postgres; ``text`` keeps the
        # column nullable and accepts any string when the source
        # later resolves a real dtype.
        return "text"
    # Nested types: list / large_list / fixed_size_list / map / struct
    # all round-trip through JSONB. Postgres has no first-class
    # equivalent, and JSONB preserves the shape with index support.
    if (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
        or pa.types.is_map(dtype)
        or pa.types.is_struct(dtype)
    ):
        return "jsonb"
    raise TypeError(f"No Postgres type mapping for Arrow type {dtype!r}")


def arrow_to_postgres_field(field: pa.Field | Field) -> str:
    """Render a single ``pa.Field`` as ``"name" type [NOT NULL]``."""
    if isinstance(field, Field):
        arrow_field = field.to_arrow_field()
    else:
        arrow_field = field
    from .sql_utils import quote_ident
    parts = [quote_ident(arrow_field.name), arrow_to_postgres_type(arrow_field.type)]
    if not arrow_field.nullable:
        parts.append("NOT NULL")
    return " ".join(parts)


def arrow_schema_to_postgres_columns(schema: pa.Schema | Field) -> list[str]:
    """Render every field in *schema* as a Postgres column DDL fragment."""
    if isinstance(schema, Field):
        schema = schema.to_arrow_schema()
    return [arrow_to_postgres_field(schema.field(i)) for i in range(len(schema))]


# ---------------------------------------------------------------------------
# Postgres → Arrow
# ---------------------------------------------------------------------------


_NUMERIC_RE = re.compile(r"^numeric(?:\(\s*(\d+)(?:\s*,\s*(-?\d+))?\s*\))?$", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(
    r"^timestamp(?:\s*\(\s*\d+\s*\))?(?:\s+(with(?:out)?\s+time\s+zone))?$",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"^time(?:\s*\(\s*\d+\s*\))?(?:\s+(with(?:out)?\s+time\s+zone))?$",
    re.IGNORECASE,
)
_VARCHAR_RE = re.compile(
    r"^(?:character\s+varying|varchar|character|char|bpchar|text)(?:\s*\(\s*\d+\s*\))?$",
    re.IGNORECASE,
)


def postgres_to_arrow_type(name: str) -> pa.DataType:
    """Parse a Postgres type name into a :class:`pyarrow.DataType`.

    Accepts the canonical lower-case forms returned by
    ``information_schema.columns`` (``"timestamp without time zone"``,
    ``"character varying"``, ``"numeric"``) as well as the short
    aliases (``"int"``, ``"int4"``, ``"int8"``, ``"float4"``,
    ``"float8"``, ``"bool"``).

    Unknown / extension types (``"hstore"``, ``"ltree"``, …) fall
    back to ``string`` so callers can still materialize the rows;
    callers that need exact fidelity should override the schema.
    """
    raw = (name or "").strip().lower()
    if not raw:
        raise ValueError("Empty Postgres type name")

    # Strip a trailing ``[]`` array suffix — Postgres arrays come
    # back from the wire as Python lists; the Arrow type for the
    # column is a list of the element type.
    is_array = raw.endswith("[]")
    if is_array:
        elem_type = postgres_to_arrow_type(raw[:-2].strip())
        return pa.list_(elem_type)

    # Direct table for fixed-name types.
    direct = _DIRECT_TYPES.get(raw)
    if direct is not None:
        return direct

    m = _NUMERIC_RE.match(raw)
    if m:
        precision = int(m.group(1)) if m.group(1) else 38
        scale = int(m.group(2)) if m.group(2) else 0
        precision = max(1, min(precision, 38))
        scale = max(-precision, min(scale, precision))
        return pa.decimal128(precision, scale)

    m = _TIMESTAMP_RE.match(raw)
    if m:
        with_tz = bool(m.group(1)) and "without" not in m.group(1)
        return pa.timestamp("us", tz="UTC") if with_tz else pa.timestamp("us")

    m = _TIME_RE.match(raw)
    if m:
        return pa.time64("us")

    m = _VARCHAR_RE.match(raw)
    if m:
        return pa.string()

    # Unknown — treat as text. Logged at the caller site if needed.
    return pa.string()


_DIRECT_TYPES: dict[str, pa.DataType] = {
    "boolean": pa.bool_(),
    "bool": pa.bool_(),
    "smallint": pa.int16(),
    "int2": pa.int16(),
    "integer": pa.int32(),
    "int": pa.int32(),
    "int4": pa.int32(),
    "bigint": pa.int64(),
    "int8": pa.int64(),
    "real": pa.float32(),
    "float4": pa.float32(),
    "double precision": pa.float64(),
    "float8": pa.float64(),
    "bytea": pa.binary(),
    "date": pa.date32(),
    "interval": pa.duration("us"),
    "uuid": pa.string(),
    "json": pa.string(),
    "jsonb": pa.string(),
    "xml": pa.string(),
    "name": pa.string(),
    "oid": pa.int64(),
    "money": pa.decimal128(38, 4),
    "inet": pa.string(),
    "cidr": pa.string(),
    "macaddr": pa.string(),
    "macaddr8": pa.string(),
}


def postgres_to_arrow_field(
    name: str,
    type_name: str,
    *,
    nullable: bool = True,
) -> pa.Field:
    """Build a ``pa.Field`` from a column name + Postgres type name."""
    return pa.field(name, postgres_to_arrow_type(type_name), nullable=nullable)
