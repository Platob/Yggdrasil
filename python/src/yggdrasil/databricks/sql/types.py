"""Type utilities for Databricks SQL metadata and Arrow."""

import json
import re
from typing import Union

import pyarrow as pa
from databricks.sdk.service.catalog import ColumnInfo as CatalogColumnInfo, ColumnTypeName
from databricks.sdk.service.sql import ColumnInfo as SQLColumnInfo

from yggdrasil.types import is_arrow_type_list_like
from yggdrasil.types.cast.arrow_cast import ArrowDataType

STRING_TYPE_MAP = {
    # boolean
    "BOOL": pa.bool_(),
    "BOOLEAN": pa.bool_(),

    # string / text
    "CHAR": pa.string(),
    "NCHAR": pa.string(),
    "VARCHAR": pa.string(),
    "NVARCHAR": pa.string(),
    "STRING": pa.string(),
    "TEXT": pa.large_string(),
    "LONGTEXT": pa.large_string(),

    # integers
    "TINYINT": pa.int8(),
    "BYTE": pa.int8(),
    "SMALLINT": pa.int16(),
    "SHORT": pa.int16(),
    "INT2": pa.int16(),

    "INT": pa.int32(),
    "INTEGER": pa.int32(),
    "INT4": pa.int32(),

    "BIGINT": pa.int64(),
    "LONG": pa.int64(),
    "INT8": pa.int64(),

    # unsigned → widen (Arrow has no unsigned for many)
    "UNSIGNED TINYINT": pa.int16(),
    "UNSIGNED SMALLINT": pa.int32(),
    "UNSIGNED INT": pa.int64(),
    "UNSIGNED BIGINT": pa.uint64() if hasattr(pa, "uint64") else pa.int64(),

    # floats
    "FLOAT": pa.float32(),
    "REAL": pa.float32(),
    "DOUBLE": pa.float64(),
    "DOUBLE PRECISION": pa.float64(),

    # numeric/decimal — regex handles DECIMAL(p,s); bare form fallback
    "NUMERIC": pa.decimal128(38, 18),
    "DECIMAL": pa.decimal128(38, 18),

    # date/time/timestamp
    "DATE": pa.date32(),
    "TIME": pa.time64("ns"),
    "TIMESTAMP": pa.timestamp("us", "Etc/UTC"),
    "TIMESTAMP_NTZ": pa.timestamp("us"),
    "DATETIME": pa.timestamp("us", "Etc/UTC"),

    # binary
    "BINARY": pa.binary(),
    "VARBINARY": pa.binary(),
    "BLOB": pa.binary(),

    # json-like
    "JSON": pa.string(),
    "JSONB": pa.string(),

    # other structured text
    "UUID": pa.string(),
    "XML": pa.string(),

    # explicit arrow large types
    "LARGE_STRING": pa.large_string(),
    "LARGE_BINARY": pa.large_binary(),

    "VARIANT": pa.string(),
}

ARROW_TYPE_MAP = {
    # null / strings
    pa.null(): ColumnTypeName.NULL,
    pa.string(): ColumnTypeName.STRING,
    pa.large_string(): ColumnTypeName.STRING,

    # booleans
    pa.bool_(): ColumnTypeName.BOOLEAN,

    # integers
    pa.int8(): ColumnTypeName.BYTE,
    pa.int16(): ColumnTypeName.SHORT,
    pa.int32(): ColumnTypeName.INT,
    pa.int64(): ColumnTypeName.LONG,

    pa.uint8(): ColumnTypeName.SHORT,   # closest (no unsigned byte)
    pa.uint16(): ColumnTypeName.INT,    # closest
    pa.uint32(): ColumnTypeName.LONG,   # closest
    pa.uint64(): ColumnTypeName.DECIMAL,  # safest-ish fallback for very large unsigned (see note)

    # floats
    pa.float16(): ColumnTypeName.FLOAT,   # closest
    pa.float32(): ColumnTypeName.FLOAT,
    pa.float64(): ColumnTypeName.DOUBLE,

    # decimal
    pa.decimal128(1, 0): ColumnTypeName.DECIMAL,  # canonical key; match by type.id in code
    pa.decimal256(1, 0): ColumnTypeName.DECIMAL,

    # binary
    pa.binary(): ColumnTypeName.BINARY,
    pa.large_binary(): ColumnTypeName.BINARY,

    # date / time
    pa.date32(): ColumnTypeName.DATE,
    pa.date64(): ColumnTypeName.DATE,      # closest (date64 is ms since epoch)

    pa.time32("s"): ColumnTypeName.STRING,  # Catalog usually doesn't have TIME; store as string
    pa.time32("ms"): ColumnTypeName.STRING,
    pa.time64("us"): ColumnTypeName.STRING,
    pa.time64("ns"): ColumnTypeName.STRING,

    # timestamp
    pa.timestamp("s"): ColumnTypeName.TIMESTAMP,
    pa.timestamp("ms"): ColumnTypeName.TIMESTAMP,
    pa.timestamp("us"): ColumnTypeName.TIMESTAMP,
    pa.timestamp("ns"): ColumnTypeName.TIMESTAMP,
    pa.timestamp("us", tz="UTC"): ColumnTypeName.TIMESTAMP,
    pa.timestamp("ns", tz="UTC"): ColumnTypeName.TIMESTAMP,

    # interval / duration -> string (Catalog typically lacks these)
    pa.duration("s"): ColumnTypeName.LONG,   # durations are numeric; store as long of unit ticks
    pa.duration("ms"): ColumnTypeName.LONG,
    pa.duration("us"): ColumnTypeName.LONG,
    pa.duration("ns"): ColumnTypeName.LONG,
}

_decimal_re = re.compile(r"^DECIMAL\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)$", re.IGNORECASE)
_array_re   = re.compile(r"^ARRAY\s*<\s*(.+)\s*>$", re.IGNORECASE)
_map_re     = re.compile(r"^MAP\s*<\s*(.+?)\s*,\s*(.+)\s*>$", re.IGNORECASE)
_struct_re  = re.compile(r"^STRUCT\s*<\s*(.+)\s*>$", re.IGNORECASE)

__all__ = [
    "STRING_TYPE_MAP",
    "parse_sql_type_to_pa",
    "column_info_to_arrow_field",
    "arrow_type_to_column_type_name",
    "arrow_type_to_type_text",
    "arrow_field_to_column_info",
    "arrow_field_to_type_json",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_top_level_commas(s: str) -> list[str]:
    """Split a type string by commas, respecting nested angle brackets."""
    parts, cur, depth = [], [], 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _safe_bytes(obj) -> bytes:
    """Convert an object to UTF-8 bytes, with safe handling for None."""
    if isinstance(obj, bytes):
        return obj
    if not obj:
        return b""
    if not isinstance(obj, str):
        obj = str(obj)
    return obj.encode("utf-8")


# ---------------------------------------------------------------------------
# SQL/Catalog type string → Arrow  (existing, inbound direction)
# ---------------------------------------------------------------------------

def parse_sql_type_to_pa(type_str: str) -> pa.DataType:
    """Parse a Databricks/Spark SQL type string into a PyArrow DataType.

    Handles DECIMAL(p,s), ARRAY<…>, MAP<k,v>, STRUCT<…> recursively,
    then falls back to STRING_TYPE_MAP for scalar types.

    Args:
        type_str: SQL/catalogString type text, e.g. ``"DECIMAL(18,6)"``,
                  ``"ARRAY<STRING>"``, ``"MAP<STRING,DOUBLE>"``.

    Returns:
        Corresponding ``pa.DataType``.

    Raises:
        ValueError: If *type_str* is empty.
    """
    if not type_str:
        raise ValueError("Empty type string")

    raw = str(type_str).strip()

    m = _decimal_re.match(raw)
    if m:
        return pa.decimal128(int(m.group(1)), int(m.group(2)))

    m = _array_re.match(raw)
    if m:
        return pa.list_(parse_sql_type_to_pa(m.group(1).strip()))

    m = _map_re.match(raw)
    if m:
        return pa.map_(
            parse_sql_type_to_pa(m.group(1).strip()),
            parse_sql_type_to_pa(m.group(2).strip()),
        )

    m = _struct_re.match(raw)
    if m:
        fields = []
        for part in _split_top_level_commas(m.group(1).strip()):
            if ":" not in part:
                fields.append(pa.field(part, pa.string(), nullable=True))
            else:
                fname, ftype_raw = part.split(":", 1)
                fields.append(
                    pa.field(fname.strip(), parse_sql_type_to_pa(ftype_raw.strip()), nullable=True)
                )
        return pa.struct(fields)

    # strip trailing size/precision suffix: VARCHAR(255) → VARCHAR
    base = re.sub(r"\(.*\)\s*$", "", raw).strip().upper()
    if base in STRING_TYPE_MAP:
        return STRING_TYPE_MAP[base]

    # unknown type — degrade gracefully to string
    return pa.string()


# ---------------------------------------------------------------------------
# Arrow → Spark SQL type text  (new, outbound direction)
# ---------------------------------------------------------------------------

def arrow_type_to_type_text(arrow_type: ArrowDataType) -> str:
    """Return a Spark-compatible SQL type string for a PyArrow DataType.

    This is the inverse of ``parse_sql_type_to_pa`` and produces strings
    accepted by the Databricks Unity Catalog Tables API.

    Args:
        arrow_type: Any ``pa.DataType``.

    Returns:
        Spark SQL type string, e.g. ``"DECIMAL(18,6)"``, ``"ARRAY<DOUBLE>"``.

    Raises:
        ValueError: If the type cannot be mapped.
    """
    if pa.types.is_boolean(arrow_type):
        return "BOOLEAN"
    if pa.types.is_int8(arrow_type):
        return "BYTE"
    if pa.types.is_int16(arrow_type):
        return "SHORT"
    if pa.types.is_int32(arrow_type):
        return "INT"
    if pa.types.is_int64(arrow_type):
        return "LONG"
    if pa.types.is_float16(arrow_type) or pa.types.is_float32(arrow_type):
        return "FLOAT"
    if pa.types.is_float64(arrow_type):
        return "DOUBLE"
    if pa.types.is_decimal(arrow_type):
        return f"DECIMAL({arrow_type.precision},{arrow_type.scale})"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "STRING"
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return "BINARY"
    if pa.types.is_date(arrow_type):
        return "DATE"
    if pa.types.is_time(arrow_type):
        return "STRING"  # Spark has no TIME; store as STRING
    if pa.types.is_timestamp(arrow_type):
        return "TIMESTAMP" if arrow_type.tz is not None else "TIMESTAMP_NTZ"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return f"ARRAY<{arrow_type_to_type_text(arrow_type.value_type)}>"
    if pa.types.is_map(arrow_type):
        return (
            f"MAP<{arrow_type_to_type_text(arrow_type.key_type)},"
            f"{arrow_type_to_type_text(arrow_type.item_type)}>"
        )
    if pa.types.is_struct(arrow_type):
        fields = ",".join(
            f"{arrow_type.field(i).name}:{arrow_type_to_type_text(arrow_type.field(i).type)}"
            for i in range(arrow_type.num_fields)
        )
        return f"STRUCT<{fields}>"
    if pa.types.is_uint8(arrow_type):
        return "SHORT"   # widen — Spark has no UINT8
    if pa.types.is_uint16(arrow_type):
        return "INT"
    if pa.types.is_uint32(arrow_type):
        return "LONG"
    if pa.types.is_uint64(arrow_type):
        return "DECIMAL(20,0)"  # widest safe mapping
    raise ValueError(
        f"Cannot convert Arrow type '{arrow_type}' to a Spark SQL type string."
    )


# ---------------------------------------------------------------------------
# Arrow → ColumnTypeName enum  (new, outbound direction)
# ---------------------------------------------------------------------------

def arrow_type_to_column_type_name(arrow_type: ArrowDataType) -> ColumnTypeName:
    """Map a PyArrow DataType to the Databricks ``ColumnTypeName`` enum.

    Handles timestamps (tz-aware → TIMESTAMP, naive → TIMESTAMP_NTZ),
    decimals, unsigned integers (widened), and all common scalar/nested types.

    Args:
        arrow_type: Any ``pa.DataType``.

    Returns:
        The closest ``ColumnTypeName`` variant.

    Raises:
        ValueError: If the type has no reasonable mapping.
    """
    found = ARROW_TYPE_MAP.get(arrow_type)

    if found is not None:
        return found

    if pa.types.is_decimal(arrow_type):
        return ColumnTypeName.DECIMAL
    if pa.types.is_timestamp(arrow_type):
        return ColumnTypeName.TIMESTAMP if arrow_type.tz is not None else ColumnTypeName.TIMESTAMP_NTZ
    if is_arrow_type_list_like(arrow_type):
        return ColumnTypeName.ARRAY
    if pa.types.is_map(arrow_type):
        return ColumnTypeName.MAP
    if pa.types.is_struct(arrow_type):
        return ColumnTypeName.STRUCT

    raise ValueError(
        f"Unsupported Arrow type '{arrow_type}' — "
        "add an explicit mapping to _ARROW_STR_TO_COLUMN_TYPE."
    )


# ---------------------------------------------------------------------------
# Arrow field → ColumnInfo  (new, outbound direction)
# ---------------------------------------------------------------------------

def arrow_field_to_column_info(
    field: pa.Field,
    position: int,
) -> CatalogColumnInfo:
    """Convert a PyArrow Field to a Databricks Unity Catalog ``ColumnInfo``.

    This is the inverse of ``column_info_to_arrow_field`` and is used when
    creating or updating UC tables via the SDK.

    Column-level ``comment`` is sourced from the field's Arrow metadata under
    the key ``b"comment"`` if present.

    Args:
        field:    PyArrow field to convert.
        position: Zero-based ordinal position in the table schema.

    Returns:
        A populated ``CatalogColumnInfo`` ready to pass to ``tables.create()``.

    Example::

        fields = [
            pa.field("trade_date", pa.date32(),       nullable=False),
            pa.field("commodity",  pa.string(),        nullable=False),
            pa.field("price",      pa.float64()),
            pa.field("volume",     pa.int64()),
            pa.field("notional",   pa.decimal128(18, 6)),
        ]
        columns = [arrow_field_to_column_info(f, i) for i, f in enumerate(fields)]
    """
    arrow_type = field.type
    type_name  = arrow_type_to_column_type_name(arrow_type)
    type_text  = arrow_type_to_type_text(arrow_type)
    type_json  = arrow_field_to_type_json(field)

    precision: int | None = None
    scale: int | None = None
    if pa.types.is_decimal(arrow_type):
        precision = arrow_type.precision
        scale = arrow_type.scale
    # uint64 widened to DECIMAL(20,0)
    if pa.types.is_uint64(arrow_type):
        precision, scale = 20, 0

    comment: str | None = None
    if field.metadata:
        raw = field.metadata.get(b"comment") or field.metadata.get(b"description")
        if raw:
            comment = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    return CatalogColumnInfo(
        name=field.name,
        position=position,
        nullable=field.nullable,
        type_name=type_name,
        type_text=type_text,
        type_json=type_json,
        type_precision=precision,
        type_scale=scale,
        comment=comment,
    )


# ---------------------------------------------------------------------------
# Catalog/SQL ColumnInfo → Arrow field  (existing, inbound direction)
# ---------------------------------------------------------------------------

def column_info_to_arrow_field(col: Union[SQLColumnInfo, CatalogColumnInfo]) -> pa.Field:
    """Convert a Databricks SQL or Catalog ``ColumnInfo`` into a PyArrow Field.

    Args:
        col: ``ColumnInfo`` from the SQL Statement Execution API or
             Unity Catalog Tables API.

    Returns:
        An Arrow ``Field`` preserving the column name, type, nullability,
        and any UC metadata stored in ``type_json``.

    Raises:
        TypeError: If *col* is not a recognised ``ColumnInfo`` variant.
    """
    arrow_type = parse_sql_type_to_pa(col.type_text)

    if isinstance(col, CatalogColumnInfo):
        parsed = json.loads(col.type_json)
        md = parsed.get("metadata", {}) or {}
        md = {_safe_bytes(k): _safe_bytes(v) for k, v in md.items()}
        nullable = col.nullable
    elif isinstance(col, SQLColumnInfo):
        md = {}
        nullable = True
    else:
        raise TypeError(f"Cannot build arrow field from {col.__class__}")

    return pa.field(col.name, arrow_type, nullable=nullable, metadata=md)

def arrow_field_to_type_json(field: pa.Field) -> str:
    """
    Build Databricks-style `type_json` for a single Arrow field.

    Output looks like:
      {"name":"origin","type":"string","nullable":false,"metadata":{}}

    Notes:
    - `type` is Spark SQL type text (same as `arrow_type_to_type_text`), lowercased to match
      common UC payloads.
    - `metadata` is serialized as a string->string dict (bytes decoded as UTF-8).
    """
    md: dict[str, str] = {}
    if field.metadata:
        for k, v in field.metadata.items():
            kk = k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
            vv = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
            md[kk] = vv

    payload = {
        "name": field.name,
        "type": arrow_type_to_type_text(field.type).lower(),
        "nullable": bool(field.nullable),
        "metadata": md,
    }
    # compact JSON (no spaces) to match typical API payloads
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
