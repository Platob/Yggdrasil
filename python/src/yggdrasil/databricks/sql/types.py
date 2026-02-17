"""Type utilities for Databricks SQL metadata and Arrow."""

import json
import re
from typing import Union

import pyarrow as pa
from databricks.sdk.service.catalog import ColumnInfo as CatalogColumnInfo
from databricks.sdk.service.sql import ColumnInfo as SQLColumnInfo

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
    "SMALLINT": pa.int16(),
    "INT2": pa.int16(),

    "INT": pa.int32(),
    "INTEGER": pa.int32(),
    "INT4": pa.int32(),

    "BIGINT": pa.int64(),
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

    # numeric/decimal — regex later for DECIMAL(p,s)
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
}

_decimal_re = re.compile(r"^DECIMAL\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)$", re.IGNORECASE)
_array_re = re.compile(r"^ARRAY\s*<\s*(.+)\s*>$", re.IGNORECASE)
_map_re = re.compile(r"^MAP\s*<\s*(.+?)\s*,\s*(.+)\s*>$", re.IGNORECASE)
_struct_re = re.compile(r"^STRUCT\s*<\s*(.+)\s*>$", re.IGNORECASE)


def _split_top_level_commas(s: str):
    """Split a type string by commas, respecting nested angle brackets.

    Args:
        s: Type string to split.

    Returns:
        A list of top-level comma-separated parts.
    """
    parts, cur, depth = [], [], 0
    for ch in s:
        if ch == '<':
            depth += 1
        elif ch == '>':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur).strip())
    return parts


def _safe_bytes(obj):
    """Convert an object to UTF-8 bytes, with safe handling for None.

    Args:
        obj: Value to convert.

    Returns:
        UTF-8 encoded bytes.
    """
    if not isinstance(obj, bytes):
        if not obj:
            return b""

        if not isinstance(obj, str):
            obj = str(obj)

        return obj.encode("utf-8")
    return obj


def parse_sql_type_to_pa(type_str: str) -> pa.DataType:
    """
    Adapted parser that:
      - looks up base types in STRING_TYPE_MAP (expects uppercase keys)
      - supports DECIMAL(p,s), ARRAY<...>, MAP<k,v>, STRUCT<...> recursively
      - raises ValueError if it cannot map the provided type string

    Args:
        type_str: SQL type string to parse.

    Returns:
        The corresponding Arrow DataType.
    """
    if not type_str:
        raise ValueError("Empty type string")

    raw = str(type_str).strip()

    # DECIMAL(p,s)
    m = _decimal_re.match(raw)
    if m:
        precision = int(m.group(1)); scale = int(m.group(2))
        return pa.decimal128(precision, scale)

    # ARRAY<...>
    m = _array_re.match(raw)
    if m:
        inner = m.group(1).strip()
        return pa.list_(parse_sql_type_to_pa(inner))

    # MAP<k,v>
    m = _map_re.match(raw)
    if m:
        key_raw = m.group(1).strip()
        val_raw = m.group(2).strip()
        key_type = parse_sql_type_to_pa(key_raw)
        val_type = parse_sql_type_to_pa(val_raw)
        return pa.map_(key_type, val_type)

    # STRUCT<...>
    m = _struct_re.match(raw)
    if m:
        inner = m.group(1).strip()
        parts = _split_top_level_commas(inner)
        fields = []
        for p in parts:
            if ':' not in p:
                # defensive fallback
                fname = p
                ftype = pa.string()
            else:
                fname, ftype_raw = p.split(':', 1)
                fname = fname.strip()
                ftype = parse_sql_type_to_pa(ftype_raw.strip())
            fields.append(pa.field(fname, ftype, nullable=True))
        return pa.struct(fields)

    # normalize and strip size/precision suffixes: e.g. VARCHAR(255) -> VARCHAR
    base = re.sub(r"\(.*\)\s*$", "", raw).strip().upper()

    # direct lookup in provided map
    if base in STRING_TYPE_MAP:
        return STRING_TYPE_MAP[base]

    # nothing matched — raise so caller knows it's unknown
    raise ValueError(f"Cannot convert string data type '{type_str}' to arrow")


def column_info_to_arrow_field(col: Union[SQLColumnInfo, CatalogColumnInfo]):
    """Convert Databricks SQL/Catalog column info into an Arrow field.

    Args:
        col: ColumnInfo from SQL or Catalog APIs.

    Returns:
        An Arrow Field for the column.
    """
    arrow_type = parse_sql_type_to_pa(col.type_text)

    if isinstance(col, CatalogColumnInfo):
        parsed = json.loads(col.type_json)
        md = parsed.get("metadata", {}) or {}
        md = {
            _safe_bytes(k): _safe_bytes(v)
            for k, v in md.items()
        }
        nullable = col.nullable
    elif isinstance(col, SQLColumnInfo):
        md = {}
        nullable = True
    else:
        raise TypeError(f"Cannot build arrow field from {col.__class__}")

    return pa.field(
        col.name,
        arrow_type,
        nullable=nullable,
        metadata=md
    )
