"""Type utilities for Databricks SQL metadata and Arrow."""

import json
import re
from typing import Union

import pyarrow as pa
from databricks.sdk.service.catalog import ColumnInfo as CatalogColumnInfo, ColumnTypeName
from databricks.sdk.service.sql import ColumnInfo as SQLColumnInfo

from yggdrasil.arrow.cast import (
    ArrowDataType,
    is_arrow_type_binary_like,
    is_arrow_type_list_like,
    is_arrow_type_string_like,
)

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
    "TIMESTAMP": pa.timestamp("us", "UTC"),
    "TIMESTAMP_NTZ": pa.timestamp("us"),
    "DATETIME": pa.timestamp("us", "UTC"),

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

    # integers - using standard Databricks SQL type names
    pa.int8(): ColumnTypeName.BYTE,     # maps to TINYINT in SQL
    pa.int16(): ColumnTypeName.SHORT,   # maps to SMALLINT in SQL
    pa.int32(): ColumnTypeName.INT,
    pa.int64(): ColumnTypeName.LONG,    # maps to BIGINT in SQL

    # unsigned integers (widened to signed)
    pa.uint8(): ColumnTypeName.SHORT,   # widened (no unsigned BYTE)
    pa.uint16(): ColumnTypeName.INT,    # widened
    pa.uint32(): ColumnTypeName.LONG,   # widened
    pa.uint64(): ColumnTypeName.DECIMAL,  # widened to DECIMAL(20,0) for safety

    # floats
    pa.float16(): ColumnTypeName.FLOAT,
    pa.float32(): ColumnTypeName.FLOAT,
    pa.float64(): ColumnTypeName.DOUBLE,

    # decimal (canonical keys; matched by type.id in code)
    pa.decimal128(1, 0): ColumnTypeName.DECIMAL,
    pa.decimal256(1, 0): ColumnTypeName.DECIMAL,

    # binary
    pa.binary(): ColumnTypeName.BINARY,
    pa.large_binary(): ColumnTypeName.BINARY,

    # date / time
    pa.date32(): ColumnTypeName.DATE,
    pa.date64(): ColumnTypeName.DATE,

    # time types (no native support; store as STRING)
    pa.time32("s"): ColumnTypeName.STRING,
    pa.time32("ms"): ColumnTypeName.STRING,
    pa.time64("us"): ColumnTypeName.STRING,
    pa.time64("ns"): ColumnTypeName.STRING,

    # timestamp (various units and timezones)
    pa.timestamp("s"): ColumnTypeName.TIMESTAMP,
    pa.timestamp("ms"): ColumnTypeName.TIMESTAMP,
    pa.timestamp("us"): ColumnTypeName.TIMESTAMP,
    pa.timestamp("ns"): ColumnTypeName.TIMESTAMP,
    pa.timestamp("us", tz="UTC"): ColumnTypeName.TIMESTAMP,
    pa.timestamp("ns", tz="UTC"): ColumnTypeName.TIMESTAMP,

    # duration → LONG (store tick count as BIGINT in SQL)
    pa.duration("s"): ColumnTypeName.LONG,
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
    "quote_ident",
    "escape_sql_string",
    "arrow_field_to_ddl"
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
                  ``"ARRAY<STRING>"``, ``"MAP<STRING,DOUBLE>"``, ``"VOID"``.

    Returns:
        Corresponding ``pa.DataType``.

    Raises:
        ValueError: If *type_str* is empty.

    Example::

        >>> parse_sql_type_to_pa("DECIMAL(18,6)")
        Decimal128Type(decimal128(18, 6))

        >>> parse_sql_type_to_pa("ARRAY<STRING>")
        ListType(list<item: string>)

        >>> parse_sql_type_to_pa("MAP<STRING,DOUBLE>")
        MapType(map<string, double>)
    """
    if not type_str:
        raise ValueError("Empty type string")

    raw = str(type_str).strip()

    # Handle VOID/NULL types explicitly
    if raw.upper() in ("VOID", "NULL"):
        return pa.null()

    # DECIMAL(precision, scale)
    m = _decimal_re.match(raw)
    if m:
        return pa.decimal128(int(m.group(1)), int(m.group(2)))

    # ARRAY<element_type>
    m = _array_re.match(raw)
    if m:
        return pa.list_(parse_sql_type_to_pa(m.group(1).strip()))

    # MAP<key_type, value_type>
    m = _map_re.match(raw)
    if m:
        return pa.map_(
            parse_sql_type_to_pa(m.group(1).strip()),
            parse_sql_type_to_pa(m.group(2).strip()),
        )

    # STRUCT<field1:type1, field2:type2, ...>
    m = _struct_re.match(raw)
    if m:
        fields = []
        for part in _split_top_level_commas(m.group(1).strip()):
            if ":" not in part:
                # Field without explicit type - default to string
                fields.append(pa.field(part.strip(), pa.string(), nullable=True))
            else:
                fname, ftype_raw = part.split(":", 1)
                fields.append(
                    pa.field(fname.strip(), parse_sql_type_to_pa(ftype_raw.strip()), nullable=True)
                )
        return pa.struct(fields)

    # Strip trailing size/precision suffix: VARCHAR(255) → VARCHAR
    base = re.sub(r"\(.*\)\s*$", "", raw).strip().upper()
    if base in STRING_TYPE_MAP:
        return STRING_TYPE_MAP[base]

    # Unknown type — degrade gracefully to string
    return pa.string()


# ---------------------------------------------------------------------------
# Arrow → Spark SQL type text  (new, outbound direction)
# ---------------------------------------------------------------------------

def arrow_type_to_type_text(arrow_type: ArrowDataType) -> str:
    """Return a Spark-compatible SQL type string for a PyArrow DataType.

    This is the inverse of ``parse_sql_type_to_pa`` and produces strings
    accepted by the Databricks Unity Catalog Tables API.

    Uses Databricks SQL / Spark SQL standard type names:
    - Integer types: TINYINT, SMALLINT, INT, BIGINT (not BYTE, SHORT, LONG)
    - Unsigned integers are widened to signed types
    - Timestamps: TIMESTAMP (tz-aware) vs TIMESTAMP_NTZ (naive)
    - Durations: stored as BIGINT (tick count)

    Args:
        arrow_type: Any ``pa.DataType``.

    Returns:
        Spark SQL type string, e.g. ``"DECIMAL(18,6)"``, ``"ARRAY<DOUBLE>"``.

    Raises:
        ValueError: If the type cannot be mapped.
    """
    # Boolean
    if pa.types.is_boolean(arrow_type):
        return "BOOLEAN"

    # Integers (using standard SQL names)
    if pa.types.is_int8(arrow_type):
        return "TINYINT"
    if pa.types.is_int16(arrow_type):
        return "SMALLINT"
    if pa.types.is_int32(arrow_type):
        return "INT"
    if pa.types.is_int64(arrow_type):
        return "BIGINT"

    # Unsigned integers (widened to signed)
    if pa.types.is_uint8(arrow_type):
        return "SMALLINT"  # TINYINT is signed only
    if pa.types.is_uint16(arrow_type):
        return "INT"
    if pa.types.is_uint32(arrow_type):
        return "BIGINT"
    if pa.types.is_uint64(arrow_type):
        return "DECIMAL(20,0)"  # widest safe mapping

    # Floats
    if pa.types.is_float16(arrow_type) or pa.types.is_float32(arrow_type):
        return "FLOAT"
    if pa.types.is_float64(arrow_type):
        return "DOUBLE"

    # Decimal
    if pa.types.is_decimal(arrow_type):
        return f"DECIMAL({arrow_type.precision},{arrow_type.scale})"

    # String types (using helper)
    if is_arrow_type_string_like(arrow_type):
        return "STRING"

    # Binary types (using helper)
    if is_arrow_type_binary_like(arrow_type):
        return "BINARY"

    # Date
    if pa.types.is_date(arrow_type):
        return "DATE"

    # Time (no native support in Databricks)
    if pa.types.is_time(arrow_type):
        return "STRING"  # Spark has no TIME; store as STRING

    # Timestamp (tz-aware vs naive)
    if pa.types.is_timestamp(arrow_type):
        return "TIMESTAMP" if arrow_type.tz is not None else "TIMESTAMP_NTZ"

    # Duration (stored as tick count)
    if pa.types.is_duration(arrow_type):
        return "BIGINT"  # durations are numeric; store as BIGINT of unit ticks

    # Nested types (recursive)
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

    # Null type
    if pa.types.is_null(arrow_type):
        return "VOID"  # Databricks representation for null type

    raise ValueError(
        f"Cannot convert Arrow type '{arrow_type}' to a Spark SQL type string."
    )


# ---------------------------------------------------------------------------
# Arrow → ColumnTypeName enum  (new, outbound direction)
# ---------------------------------------------------------------------------

def arrow_type_to_column_type_name(arrow_type: ArrowDataType) -> ColumnTypeName:
    """Map a PyArrow DataType to the Databricks ``ColumnTypeName`` enum.

    Handles timestamps (tz-aware → TIMESTAMP, naive → TIMESTAMP_NTZ),
    decimals, unsigned integers (widened), durations (as LONG),
    and all common scalar/nested types.

    Args:
        arrow_type: Any ``pa.DataType``.

    Returns:
        The closest ``ColumnTypeName`` variant.

    Raises:
        ValueError: If the type has no reasonable mapping.
    """
    # Check exact match in map first
    found = ARROW_TYPE_MAP.get(arrow_type)
    if found is not None:
        return found

    # Handle types that vary by parameters
    if pa.types.is_decimal(arrow_type):
        return ColumnTypeName.DECIMAL
    if pa.types.is_timestamp(arrow_type):
        return ColumnTypeName.TIMESTAMP if arrow_type.tz is not None else ColumnTypeName.TIMESTAMP_NTZ
    if pa.types.is_duration(arrow_type):
        return ColumnTypeName.LONG  # durations stored as tick count
    if is_arrow_type_list_like(arrow_type):
        return ColumnTypeName.ARRAY
    if pa.types.is_map(arrow_type):
        return ColumnTypeName.MAP
    if pa.types.is_struct(arrow_type):
        return ColumnTypeName.STRUCT

    raise ValueError(
        f"Unsupported Arrow type '{arrow_type}' — "
        "add an explicit mapping to ARROW_TYPE_MAP."
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

    Preserves column metadata from Unity Catalog's ``type_json`` field,
    including comments and custom metadata. For SQL execution results,
    assumes nullable=True and no metadata.

    Args:
        col: ``ColumnInfo`` from the SQL Statement Execution API or
             Unity Catalog Tables API.

    Returns:
        An Arrow ``Field`` preserving the column name, type, nullability,
        and any UC metadata stored in ``type_json``.

    Raises:
        TypeError: If *col* is not a recognised ``ColumnInfo`` variant.

    Example::

        from databricks.sdk.service.catalog import ColumnInfo, ColumnTypeName

        col = ColumnInfo(
            name="price",
            type_text="DECIMAL(18,6)",
            type_name=ColumnTypeName.DECIMAL,
            type_json='{"name":"price","type":"decimal(18,6)","nullable":false,"metadata":{"comment":"USD"}}',
            nullable=False,
            position=0,
        )
        field = column_info_to_arrow_field(col)
        # pa.field("price", pa.decimal128(18, 6), nullable=False, metadata={b"comment": b"USD"})
    """
    arrow_type = parse_sql_type_to_pa(col.type_text)

    if isinstance(col, CatalogColumnInfo):
        # Parse metadata from type_json
        parsed = json.loads(col.type_json)
        md = parsed.get("metadata", {}) or {}
        md = {_safe_bytes(k): _safe_bytes(v) for k, v in md.items()}

        # Add comment from col.comment if present and not in metadata
        if col.comment and b"comment" not in md:
            md[b"comment"] = _safe_bytes(col.comment)

        nullable = col.nullable
    elif isinstance(col, SQLColumnInfo):
        # SQL execution results: no metadata, assume nullable
        md = {}
        nullable = True
    else:
        raise TypeError(f"Cannot build arrow field from {col.__class__}")

    return pa.field(col.name, arrow_type, nullable=nullable, metadata=md)



def escape_sql_string(s: str) -> str:
    """Escape a Python string for safe embedding in a single-quoted SQL literal.

    Args:
        s: Input string to escape.

    Returns:
        SQL-escaped string where single quotes are doubled.

    Example::

        >>> escape_sql_string("O'Reilly")
        "O''Reilly"
    """
    return s.replace("'", "''")


def quote_ident(ident: str) -> str:
    """Quote a SQL identifier using backticks, escaping embedded backticks.

    Args:
        ident: Identifier to quote (catalog/schema/table/column/etc).

    Returns:
        Backtick-quoted identifier with embedded backticks doubled.

    Example::

        >>> quote_ident("my_table")
        "`my_table`"
        >>> quote_ident("table`with`backticks")
        "`table``with``backticks`"
    """
    escaped = ident.replace("`", "``")
    return f"`{escaped}`"


def arrow_field_to_type_json(field: pa.Field) -> str:
    """Build Databricks-style ``type_json`` for a single Arrow field.

    Creates a JSON string matching the format used by Unity Catalog's Tables API.
    The type field is lowercased to match typical Databricks API responses.

    Args:
        field: Arrow field to convert.

    Returns:
        Compact JSON string with name, type, nullable, and metadata fields.

    Example::

        >>> field = pa.field("price", pa.decimal128(18, 6), nullable=False,
        ...                  metadata={b"comment": b"Trade price in USD"})
        >>> arrow_field_to_type_json(field)
        '{"name":"price","type":"decimal(18,6)","nullable":false,"metadata":{"comment":"Trade price in USD"}}'

    Notes:
        - ``type`` is Spark SQL type text (same as ``arrow_type_to_type_text``), lowercased
        - ``metadata`` is serialized as a string→string dict (bytes decoded as UTF-8)
        - Output uses compact JSON (no spaces) to match typical API payloads
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


def arrow_field_to_ddl(
    field: pa.Field,
    put_name: bool = True,
    put_not_null: bool = True,
    put_comment: bool = True,
) -> str:
    """Convert an Arrow field to a Databricks SQL column DDL fragment.

    Supports primitives plus nested types (STRUCT, MAP, ARRAY).
    Uses ``arrow_type_to_sql_type`` for consistent type mapping.

    Args:
        field: Arrow field to convert.
        put_name: If True, include the column name (backtick-quoted).
        put_not_null: If True, emit NOT NULL when ``field.nullable`` is False.
        put_comment: If True, emit COMMENT from field metadata key ``b"comment"``.

    Returns:
        SQL DDL fragment for a column definition.

    Raises:
        TypeError: If a nested Arrow type cannot be represented.
        ValueError: If a primitive Arrow type cannot be mapped to SQL.

    Example::

        field = pa.field("price", pa.decimal128(18, 6), nullable=False,
                         metadata={b"comment": b"Trade price in USD"})
        ddl = arrow_field_to_ddl(field)
        # "`price` DECIMAL(18, 6) NOT NULL COMMENT 'Trade price in USD'"
    """
    name_str = f"{quote_ident(field.name)} " if put_name else ""
    nullable_str = " NOT NULL" if put_not_null and not field.nullable else ""

    comment_str = ""
    if put_comment and field.metadata and b"comment" in field.metadata:
        comment = field.metadata[b"comment"].decode("utf-8")
        comment_str = f" COMMENT '{escape_sql_string(comment)}'"

    # Handle primitive (non-nested) types
    if not pa.types.is_nested(field.type):
        sql_type = arrow_type_to_sql_type(field.type)
        return f"{name_str}{sql_type}{nullable_str}{comment_str}"

    # Handle STRUCT<...>
    if pa.types.is_struct(field.type):
        struct_body = ", ".join([arrow_field_to_ddl(child) for child in field.type])
        return f"{name_str}STRUCT<{struct_body}>{nullable_str}{comment_str}"

    # Handle MAP<k,v>
    if pa.types.is_map(field.type):
        map_type: pa.MapType = field.type
        key_type = arrow_field_to_ddl(
            map_type.key_field,
            put_name=False,
            put_comment=False,
            put_not_null=False,
        )
        val_type = arrow_field_to_ddl(
            map_type.item_field,
            put_name=False,
            put_comment=False,
            put_not_null=False,
        )
        return f"{name_str}MAP<{key_type}, {val_type}>{nullable_str}{comment_str}"

    # Handle ARRAY<elem>
    if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
        list_type: pa.ListType = field.type
        elem_type = arrow_field_to_ddl(
            list_type.value_field,
            put_name=False,
            put_comment=False,
            put_not_null=False,
        )
        return f"{name_str}ARRAY<{elem_type}>{nullable_str}{comment_str}"

    raise TypeError(f"Cannot make DDL field from nested type: {field.type}")

def arrow_type_to_sql_type(arrow_type: ArrowDataType) -> str:
    """Convert an Arrow data type to a Databricks SQL type string.

    This function is used for DDL generation and is harmonized with
    ``arrow_type_to_type_text`` for consistent type mapping.

    Args:
        arrow_type: Arrow type instance to convert.

    Returns:
        Databricks SQL type string.

    Raises:
        ValueError: If the Arrow type is unsupported.
    """
    # Boolean
    if pa.types.is_boolean(arrow_type):
        return "BOOLEAN"
    
    # Integers (using standard SQL names)
    if pa.types.is_int8(arrow_type):
        return "TINYINT"
    if pa.types.is_int16(arrow_type):
        return "SMALLINT"
    if pa.types.is_int32(arrow_type):
        return "INT"
    if pa.types.is_int64(arrow_type):
        return "BIGINT"
    
    # Unsigned integers (widened to signed)
    if pa.types.is_uint8(arrow_type):
        return "SMALLINT"
    if pa.types.is_uint16(arrow_type):
        return "INT"
    if pa.types.is_uint32(arrow_type):
        return "BIGINT"
    if pa.types.is_uint64(arrow_type):
        return "DECIMAL(20, 0)"

    # Floats
    if pa.types.is_float16(arrow_type) or pa.types.is_float32(arrow_type):
        return "FLOAT"
    if pa.types.is_float64(arrow_type):
        return "DOUBLE"
    
    # Decimal
    if pa.types.is_decimal(arrow_type):
        return f"DECIMAL({arrow_type.precision}, {arrow_type.scale})"

    # String types (using helper for consistency)
    if is_arrow_type_string_like(arrow_type):
        return "STRING"

    # Binary types (using helper for consistency)
    if is_arrow_type_binary_like(arrow_type):
        return "BINARY"
    
    # Date
    if pa.types.is_date(arrow_type):
        return "DATE"
    
    # Time (no native support)
    if pa.types.is_time(arrow_type):
        return "STRING"

    # Timestamp (tz-aware vs naive)
    if pa.types.is_timestamp(arrow_type):
        return "TIMESTAMP" if arrow_type.tz is not None else "TIMESTAMP_NTZ"

    # Duration (stored as tick count)
    if pa.types.is_duration(arrow_type):
        return "BIGINT"

    # Null type
    if pa.types.is_null(arrow_type):
        return "VOID"
    
    raise ValueError(f"Cannot convert Arrow type '{arrow_type}' to SQL type string")
