"""Spark <-> Arrow casting helpers and converters.

This module provides bidirectional type-mapping and data-casting between
Apache Spark (PySpark) and Apache Arrow (PyArrow).  Every public function
accepts an optional ``CastOptions`` argument that carries the target schema
**and** a ``safe`` flag:

- ``safe=True``  (default)  – best-effort: invalid casts become ``null`` /
  empty values; missing columns are filled with type-appropriate defaults.
- ``safe=False`` – strict: any type mismatch or missing column raises
  immediately.

The module is structured in three layers:

1. **Type converters** – stateless, pure functions that map Arrow ↔ Spark
   type objects without touching any data.
2. **Field / schema converters** – wrap type converters to also carry name,
   nullability, and metadata.
3. **Data converters** – operate on live Spark DataFrames / Columns using the
   type information from layers 1-2.

JSON decoding for compound targets
-----------------------------------
When the *source* column is ``StringType`` or ``BinaryType`` and the *target*
type is a compound type (struct, array, or map), the casters automatically
attempt to parse the column as JSON using ``from_json`` **before** applying
field-level / element-level casting.  This handles the common commodity-data
pattern where nested structures are stored as JSON strings in a flat column
(e.g. a ``STRING`` column containing ``'{"bid":99.5,"ask":100.0}'``).

- ``BinaryType`` sources are decoded to UTF-8 string first via ``CAST(... AS STRING)``.
- Spark's ``from_json`` silently returns ``null`` for rows whose JSON is
  malformed or structurally incompatible — consistent with ``safe=True``
  semantics.  In ``safe=False`` mode callers can enforce non-nullability on
  the result via the standard null-fill machinery.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import pyarrow as pa
import pyarrow.types as pat
import pyspark.sql.functions as F
import pyspark.sql.types as T

from yggdrasil.spark.lib import pyspark_sql
from ..pyutils.serde import ObjectSerde
from ..types.cast.arrow_cast import (
    arrow_field_to_field,
    arrow_field_to_schema,
    arrow_type_to_field,
    cast_arrow_tabular, ArrowDataType,
)
from ..types.cast.cast_options import CastOptions, CastOptionsArg
from ..types.cast.registry import register_converter
from ..types.python_defaults import default_arrow_scalar, default_python_scalar

__all__ = [
    # Type-level converters
    "arrow_type_to_spark_type",
    "spark_type_to_arrow_type",
    # Field-level converters
    "arrow_field_to_spark_field",
    "spark_field_to_arrow_field",
    # Schema-level converters
    "arrow_schema_to_spark_schema",
    "spark_schema_to_arrow_schema",
    # Data converters – column
    "cast_spark_column",
    "cast_spark_column_to_struct",
    "cast_spark_column_to_list",
    "cast_spark_column_to_map",
    # Data converters – dataframe
    "cast_spark_dataframe",
    "spark_dataframe_to_arrow_table",
    "arrow_table_to_spark_dataframe",
    # Utility converters
    "any_to_spark_dataframe",
    "any_spark_to_arrow_field",
]

# ---------------------------------------------------------------------------
# Primitive type lookup tables
# ---------------------------------------------------------------------------

#: Direct Arrow → Spark primitive mappings.
#: For compound / parameterised types (decimal, timestamp, list, struct, map)
#: see :func:`arrow_type_to_spark_type`.
ARROW_TO_SPARK: dict[pa.DataType, T.DataType] = {
    pa.null():    T.NullType(),
    pa.bool_():   T.BooleanType(),

    pa.int8():    T.ByteType(),
    pa.int16():   T.ShortType(),
    pa.int32():   T.IntegerType(),
    pa.int64():   T.LongType(),

    # Spark has no unsigned integer types; widen to the next signed type.
    pa.uint8():   T.ShortType(),
    pa.uint16():  T.IntegerType(),
    pa.uint32():  T.LongType(),
    pa.uint64():  T.LongType(),   # risk of overflow; DecimalType is safer

    pa.float16(): T.FloatType(),  # best-effort: Spark has no float16
    pa.float32(): T.FloatType(),
    pa.float64(): T.DoubleType(),

    pa.string():  T.StringType(),
    getattr(pa, "string_view",  pa.string)():  T.StringType(),
    getattr(pa, "large_string", pa.string)():  T.StringType(),

    pa.binary():  T.BinaryType(),
    getattr(pa, "binary_view",  pa.binary)():  T.BinaryType(),
    getattr(pa, "large_binary", pa.binary)():  T.BinaryType(),

    pa.date32():  T.DateType(),
    pa.date64():  T.DateType(),   # time-of-day component is silently dropped

    # Canonical UTC microsecond timestamp maps to Spark TimestampType.
    pa.timestamp("us", "UTC"): T.TimestampType(),
}

#: Reverse mapping – used as a fast path in :func:`spark_type_to_arrow_type`.
SPARK_TO_ARROW: dict[T.DataType, pa.DataType] = {
    v: k for k, v in ARROW_TO_SPARK.items()
}


# ---------------------------------------------------------------------------
# Type converters
# ---------------------------------------------------------------------------

def arrow_type_to_spark_type(
    arrow_type: Union[pa.DataType, pa.Decimal128Type, pa.ListType, pa.MapType],
    cast_options: Optional[CastOptionsArg] = None,
) -> T.DataType:
    """Convert a :class:`pyarrow.DataType` to a :class:`pyspark.sql.types.DataType`.

    The mapping follows these rules in order:

    1. Exact hit in ``ARROW_TO_SPARK`` lookup table.
    2. Decimal → :class:`~pyspark.sql.types.DecimalType` with same precision/scale.
    3. Timestamp (with or without timezone).
    4. List / LargeList / FixedSizeList → :class:`~pyspark.sql.types.ArrayType`.
    5. Struct → :class:`~pyspark.sql.types.StructType` (recursive).
    6. Map → :class:`~pyspark.sql.types.MapType` (recursive).
    7. Duration → ``LongType`` (nanoseconds/microseconds as integer).
    8. Numeric fallback: integer → ``LongType``, float → ``DoubleType``.
    9. Binary / string families.
    10. Raises :class:`TypeError` for unsupported types.

    Args:
        arrow_type:    Arrow data type to convert.
        cast_options:  Optional :class:`CastOptions` (currently unused at this
                       level but forwarded to recursive calls for consistency).

    Returns:
        Equivalent Spark SQL data type.

    Raises:
        TypeError: If no mapping exists for *arrow_type*.
    """
    # Fast path: exact primitive mapping
    spark_type = ARROW_TO_SPARK.get(arrow_type)
    if spark_type is not None:
        return spark_type

    if pat.is_decimal(arrow_type):
        return T.DecimalType(precision=arrow_type.precision, scale=arrow_type.scale)

    if pat.is_timestamp(arrow_type):
        # tz-aware → TimestampType (UTC-normalised by Spark)
        # tz-naive → TimestampNTZType (wall-clock; Spark 3.4+)
        return T.TimestampType() if getattr(arrow_type, "tz", None) else T.TimestampNTZType()

    if pat.is_list(arrow_type) or pat.is_large_list(arrow_type):
        element_spark = arrow_type_to_spark_type(arrow_type.value_type, cast_options)
        return T.ArrayType(elementType=element_spark, containsNull=True)

    if pat.is_fixed_size_list(arrow_type):
        # Fixed-size lists have no direct Spark equivalent; treat as variable array.
        element_spark = arrow_type_to_spark_type(arrow_type.value_type, cast_options)
        return T.ArrayType(elementType=element_spark, containsNull=True)

    if pat.is_struct(arrow_type):
        fields = [arrow_field_to_spark_field(f, cast_options) for f in arrow_type]
        return T.StructType(fields)

    if pat.is_map(arrow_type):
        key_spark   = arrow_type_to_spark_type(arrow_type.key_type,  cast_options)
        value_spark = arrow_type_to_spark_type(arrow_type.item_type, cast_options)
        return T.MapType(keyType=key_spark, valueType=value_spark, valueContainsNull=True)

    if pat.is_duration(arrow_type):
        # Store as signed 64-bit integer (epoch units depend on arrow_type.unit).
        return T.LongType()

    # Numeric fallbacks (handles exotic / extension types)
    if pat.is_integer(arrow_type):
        return T.LongType()
    if pat.is_floating(arrow_type):
        return T.DoubleType()

    # Binary / string fallbacks
    if pat.is_binary(arrow_type) or pat.is_large_binary(arrow_type):
        return T.BinaryType()
    if pat.is_string(arrow_type) or pat.is_large_string(arrow_type):
        return T.StringType()

    raise TypeError(
        f"Unsupported or unknown Arrow type for Spark conversion: {arrow_type!r}"
    )


def spark_type_to_arrow_type(
    spark_type: T.DataType,
    cast_options: Any = None,
) -> pa.DataType:
    """Convert a :class:`pyspark.sql.types.DataType` to a :class:`pyarrow.DataType`.

    Covers all built-in Spark SQL types.  Compound types (Struct, Array, Map)
    are converted recursively.

    Args:
        spark_type:   Spark SQL data type to convert.
        cast_options: Optional :class:`CastOptions` forwarded to recursive
                      calls.

    Returns:
        Equivalent Arrow data type.

    Raises:
        TypeError: If no mapping exists for *spark_type*.
    """
    # Primitive fast path
    arrow_type = SPARK_TO_ARROW.get(spark_type)
    if arrow_type is not None:
        return arrow_type

    # Explicit isinstance checks for parameterised types
    if isinstance(spark_type, T.BooleanType):    return pa.bool_()
    if isinstance(spark_type, T.ByteType):       return pa.int8()
    if isinstance(spark_type, T.ShortType):      return pa.int16()
    if isinstance(spark_type, T.IntegerType):    return pa.int32()
    if isinstance(spark_type, T.LongType):       return pa.int64()
    if isinstance(spark_type, T.FloatType):      return pa.float32()
    if isinstance(spark_type, T.DoubleType):     return pa.float64()
    if isinstance(spark_type, T.StringType):     return pa.string()
    if isinstance(spark_type, T.BinaryType):     return pa.binary()
    if isinstance(spark_type, T.DateType):       return pa.date32()
    if isinstance(spark_type, T.TimestampType):  return pa.timestamp("us", "UTC")
    if isinstance(spark_type, T.TimestampNTZType): return pa.timestamp("us")
    if isinstance(spark_type, T.NullType):       return pa.null()

    if isinstance(spark_type, T.DecimalType):
        return pa.decimal128(spark_type.precision, spark_type.scale)

    if isinstance(spark_type, T.ArrayType):
        element_arrow = spark_type_to_arrow_type(spark_type.elementType, cast_options)
        return pa.list_(element_arrow)

    if isinstance(spark_type, T.MapType):
        key_arrow   = spark_type_to_arrow_type(spark_type.keyType,   cast_options)
        value_arrow = spark_type_to_arrow_type(spark_type.valueType, cast_options)
        return pa.map_(key_arrow, value_arrow)

    if isinstance(spark_type, T.StructType):
        arrow_fields = [
            spark_field_to_arrow_field(f, cast_options) for f in spark_type.fields
        ]
        return pa.struct(arrow_fields)

    raise TypeError(
        f"Unsupported or unknown Spark type for Arrow conversion: {spark_type!r}"
    )


# ---------------------------------------------------------------------------
# Arrow type → Spark metadata helpers (lossless roundtrip)
# ---------------------------------------------------------------------------

#: Metadata key used to store Arrow type information inside a Spark StructField.
#: The value is a JSON object produced by :func:`_arrow_type_to_metadata` and
#: consumed by :func:`_arrow_type_from_metadata`.
_ARROW_META_KEY = "__arrow__"

#: Characters that are invalid in JSON string values and must be escaped.
_JSON_SAFE = str  # alias for clarity; json.dumps handles escaping


def _arrow_type_to_metadata(arrow_type: ArrowDataType) -> dict[str, str]:
    """Extract Arrow-specific type attributes into a flat ``str → str`` dict.

    Spark's type system is a lossy projection of Arrow's — e.g. all timestamp
    units collapse to ``TimestampType``, ``float16`` becomes ``FloatType``,
    ``duration[ns]`` becomes ``LongType``, ``fixed_size_binary(16)`` becomes
    ``BinaryType``, and unsigned integers are widened to the next signed type.
    This function captures every attribute that is lost in that projection so
    :func:`_arrow_type_from_metadata` can restore the original type exactly.

    The returned dict is intended to be JSON-serialised and stored under
    :data:`_ARROW_META_KEY` in a Spark ``StructField``'s metadata dict.

    Supported attributes:

    - ``arrow_type_id``  – canonical ``str(arrow_type)`` representation used
      as a last-resort fallback (e.g. ``"timestamp[us, tz=UTC]"``).
    - ``unit``           – temporal unit for timestamp / duration / time types
      (``"s"``, ``"ms"``, ``"us"``, ``"ns"``).
    - ``tz``             – timezone string for timestamp types (``"UTC"``,
      ``"America/New_York"``, etc.).  Absent for timezone-naive timestamps.
    - ``precision``      – decimal precision.
    - ``scale``          – decimal scale.
    - ``byte_width``     – byte width for ``fixed_size_binary`` types.
    - ``list_size``      – fixed number of elements for ``fixed_size_list``.
    - ``value_type``     – ``str(value_type)`` for list / large_list /
      fixed_size_list.  Compound child types are represented as their Arrow
      string form; roundtrip recovery uses ``pa.lib.ensure_type``.
    - ``key_type``       – ``str(key_type)`` for map types.
    - ``item_type``      – ``str(item_type)`` for map types.
    - ``signed``         – ``"false"`` for unsigned integer types (``uint8``
      through ``uint64``); absent for signed integers.

    Args:
        arrow_type: Arrow data type to inspect.

    Returns:
        Flat ``str → str`` metadata dict.  Empty dict if no Arrow-specific
        attributes are needed (e.g. plain ``int64`` maps perfectly to
        ``LongType`` without any loss).
    """

    meta: dict[str, str] = {"arrow_type_id": str(arrow_type)}

    # Always store the canonical string so we have a last-resort fallback.

    # --- Temporal types ---
    if pat.is_timestamp(arrow_type):
        meta["unit"] = arrow_type.unit
        if arrow_type.tz:
            meta["tz"] = arrow_type.tz

    elif pat.is_duration(arrow_type):
        meta["unit"] = arrow_type.unit

    elif pat.is_time(arrow_type):
        meta["unit"] = arrow_type.unit

    # --- Decimal ---
    elif pat.is_decimal(arrow_type):
        meta["precision"] = str(arrow_type.precision)
        meta["scale"]     = str(arrow_type.scale)

    # --- Fixed-size binary ---
    elif pat.is_fixed_size_binary(arrow_type):
        meta["byte_width"] = str(arrow_type.byte_width)

    # --- Fixed-size list ---
    elif pat.is_fixed_size_list(arrow_type):
        meta["list_size"]  = str(arrow_type.list_size)
        meta["value_type"] = str(arrow_type.value_type)

    # --- Variable list / large list ---
    elif pat.is_list(arrow_type) or pat.is_large_list(arrow_type):
        meta["value_type"] = str(arrow_type.value_type)
        if pat.is_large_list(arrow_type):
            meta["large"] = "true"

    # --- Map ---
    elif pat.is_map(arrow_type):
        meta["key_type"]  = str(arrow_type.key_type)
        meta["item_type"] = str(arrow_type.item_type)

    # --- Unsigned integers (Spark widens these; record sign) ---
    elif arrow_type in (pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()):
        meta["signed"] = "false"

    # --- float16 (Spark has no float16; stores as float32) ---
    elif arrow_type == pa.float16():
        meta["float_width"] = "16"

    # --- String / binary variants ---
    elif pat.is_large_string(arrow_type):
        meta["large"] = "true"
        meta["kind"]  = "string"
    elif pat.is_large_binary(arrow_type):
        meta["large"] = "true"
        meta["kind"]  = "binary"
    elif hasattr(pa, "string_view") and arrow_type == pa.string_view():
        meta["view"] = "true"
        meta["kind"] = "string"
    elif hasattr(pa, "binary_view") and arrow_type == pa.binary_view():
        meta["view"] = "true"
        meta["kind"] = "binary"

    return meta


def _arrow_type_from_metadata(
    spark_type: T.DataType,
    meta: dict[str, str],
) -> pa.DataType:
    """Reconstruct an Arrow :class:`~pyarrow.DataType` from Spark metadata.

    This is the inverse of :func:`_arrow_type_to_metadata`.  Given the Spark
    type (needed as a structural hint for compound types) and the metadata dict
    stored under :data:`_ARROW_META_KEY`, it returns the most precise Arrow
    type possible.

    Recovery strategy (applied in order):

    1. **``arrow_type_id`` fast path** – ``pa.lib.ensure_type`` can parse the
       canonical Arrow type string for all primitive and most parameterised
       types (e.g. ``"timestamp[us, tz=UTC]"``, ``"decimal128(18, 6)"``).
       Used unless the type string is known to be ambiguous or lossy.
    2. **Attribute-by-attribute reconstruction** – for types where the string
       form may not survive a round-trip through ``ensure_type`` (older PyArrow
       versions), individual attributes (``unit``, ``tz``, ``precision``, …)
       are used to build the type explicitly.
    3. **Spark type fallback** – :func:`spark_type_to_arrow_type` is used when
       no metadata is available or recovery fails.

    Args:
        spark_type: Spark data type (structural hint, especially for compound
                    types whose Arrow children are encoded in metadata).
        meta:       Flat ``str → str`` metadata dict produced by
                    :func:`_arrow_type_to_metadata`.

    Returns:
        Reconstructed Arrow data type.
    """
    # Fast path: try pa.lib.ensure_type on the canonical string.
    arrow_type_id = meta.get("arrow_type_id")
    if arrow_type_id:
        try:
            return pa.lib.ensure_type(arrow_type_id)  # type: ignore[attr-defined]
        except Exception:
            pass  # fall through to attribute reconstruction

    # --- Attribute-by-attribute reconstruction ---

    unit = meta.get("unit")
    tz   = meta.get("tz")

    # Timestamp
    if isinstance(spark_type, (T.TimestampType, T.TimestampNTZType)) and unit:
        return pa.timestamp(unit, tz=tz or None)

    # Duration (stored as LongType in Spark)
    if isinstance(spark_type, T.LongType) and unit:
        return pa.duration(unit)

    # Decimal
    if isinstance(spark_type, T.DecimalType) and "precision" in meta:
        return pa.decimal128(int(meta["precision"]), int(meta["scale"]))

    # Fixed-size binary
    if isinstance(spark_type, T.BinaryType) and "byte_width" in meta:
        return pa.binary(int(meta["byte_width"]))

    # Fixed-size list
    if isinstance(spark_type, T.ArrayType) and "list_size" in meta:
        value_type = _parse_arrow_type_str(meta.get("value_type"), spark_type.elementType)
        return pa.list_(value_type, meta.get("list_size") and int(meta["list_size"]))  # type: ignore[call-arg]

    # Large list
    if isinstance(spark_type, T.ArrayType) and meta.get("large") == "true" and not meta.get("list_size"):
        value_type = _parse_arrow_type_str(meta.get("value_type"), spark_type.elementType)
        return pa.large_list(value_type)

    # Map
    if isinstance(spark_type, T.MapType) and "key_type" in meta:
        key_type  = _parse_arrow_type_str(meta.get("key_type"),  spark_type.keyType)
        item_type = _parse_arrow_type_str(meta.get("item_type"), spark_type.valueType)
        return pa.map_(key_type, item_type)

    # Unsigned integers
    if meta.get("signed") == "false":
        _unsigned = {
            T.ShortType:   pa.uint8,
            T.IntegerType: pa.uint16,
            T.LongType:    pa.uint32,
        }
        factory = _unsigned.get(type(spark_type))
        if factory:
            return factory()
        # uint64 → LongType (ambiguous with int64); use arrow_type_id hint
        if isinstance(spark_type, T.LongType):
            return pa.uint64()

    # float16
    if meta.get("float_width") == "16":
        return pa.float16()

    # Large string / binary variants
    kind = meta.get("kind")
    if meta.get("large") == "true":
        if kind == "string":
            return pa.large_string()
        if kind == "binary":
            return pa.large_binary()
    if meta.get("view") == "true":
        if kind == "string" and hasattr(pa, "string_view"):
            return pa.string_view()
        if kind == "binary" and hasattr(pa, "binary_view"):
            return pa.binary_view()

    # Fallback: derive from Spark type
    return spark_type_to_arrow_type(spark_type)


def _parse_arrow_type_str(
    type_str: Optional[str],
    spark_fallback: T.DataType,
) -> pa.DataType:
    """Parse an Arrow type string, falling back to a Spark-derived type.

    Used internally by :func:`_arrow_type_from_metadata` to recover child
    types stored as strings (e.g. ``"int32"``, ``"utf8"``).

    Args:
        type_str:       Arrow type string (may be ``None``).
        spark_fallback: Spark type used when *type_str* is absent or
                        unparseable.

    Returns:
        Arrow data type.
    """
    if type_str:
        try:
            return pa.lib.ensure_type(type_str)  # type: ignore[attr-defined]
        except Exception:
            pass
    return spark_type_to_arrow_type(spark_fallback)


# ---------------------------------------------------------------------------
# Field converters
# ---------------------------------------------------------------------------

@register_converter(pa.Field, T.StructField)
def arrow_field_to_spark_field(
    field: pa.Field,
    options: Optional[CastOptions] = None,
) -> T.StructField:
    """Convert a :class:`pyarrow.Field` to a :class:`pyspark.sql.types.StructField`.

    In addition to converting the Arrow type to its nearest Spark equivalent
    via :func:`arrow_type_to_spark_type`, this function **preserves all
    Arrow-specific type attributes** that have no direct Spark representation
    by serialising them as JSON under the reserved metadata key
    ``"__arrow__"``.  This enables :func:`spark_field_to_arrow_field` to
    restore the original Arrow type exactly (lossless roundtrip).

    Attributes captured include (but are not limited to):

    - Timestamp ``unit`` and ``tz`` (all Spark timestamps are ``us``; the
      original unit and timezone are preserved).
    - ``duration`` unit (stored as ``LongType`` in Spark).
    - Decimal ``precision`` and ``scale``.
    - ``fixed_size_binary`` byte-width.
    - ``fixed_size_list`` size and element type.
    - ``large_list`` / ``large_string`` / ``large_binary`` markers.
    - ``uint8`` … ``uint64`` unsigned flag (Spark widens to next signed type).
    - ``float16`` width marker (Spark stores as ``FloatType``).
    - ``string_view`` / ``binary_view`` view markers.

    Existing Arrow field metadata (user-defined ``bytes → bytes`` dict) is
    decoded to ``str → str`` and merged into the Spark metadata **before** the
    ``__arrow__`` key is added.  The ``__arrow__`` key always takes precedence
    and is never overwritten by user metadata.

    Args:
        field:   Arrow field to convert.
        options: Optional :class:`CastOptions` forwarded to type conversion.

    Returns:
        Spark :class:`~pyspark.sql.types.StructField` with ``__arrow__``
        metadata attached.
    """
    import json

    field = arrow_field_to_field(field, options)
    spark_type = arrow_type_to_spark_type(field.type, options)

    # Start with user-defined Arrow metadata (bytes → str decode)
    metadata: dict[str, str] = {}
    if field.metadata:
        metadata = {
            k.decode("utf-8", errors="replace"): v.decode("utf-8", errors="replace")
            for k, v in field.metadata.items()
        }

    # Capture Arrow-specific type attributes that Spark cannot represent natively.
    arrow_meta = _arrow_type_to_metadata(field.type)
    if arrow_meta:
        # Store as a compact JSON string; Spark metadata values must be strings.
        metadata[_ARROW_META_KEY] = json.dumps(arrow_meta, separators=(",", ":"))

    return T.StructField(
        name=field.name,
        dataType=spark_type,
        nullable=field.nullable,
        metadata=metadata,
    )


def spark_field_to_arrow_field(
    field: T.StructField,
    cast_options: Optional[CastOptions] = None,
) -> pa.Field:
    """Convert a :class:`pyspark.sql.types.StructField` to a :class:`pyarrow.Field`.

    If the Spark field carries ``"__arrow__"`` metadata (written by
    :func:`arrow_field_to_spark_field`), the original Arrow type is
    reconstructed from it via :func:`_arrow_type_from_metadata`.  This
    enables a **lossless roundtrip** for all Arrow types that have no direct
    Spark equivalent.

    Recovery order:

    1. ``__arrow__`` metadata present → :func:`_arrow_type_from_metadata`
       (exact reconstruction).
    2. No ``__arrow__`` metadata → :func:`spark_type_to_arrow_type`
       (best-effort structural mapping).

    Spark field metadata is re-encoded to ``bytes → bytes`` Arrow metadata,
    **excluding** the ``__arrow__`` key (which is an internal implementation
    detail and not useful to Arrow consumers).

    Args:
        field:        Spark StructField to convert.
        cast_options: Optional :class:`CastOptions`.

    Returns:
        Arrow :class:`~pyarrow.Field` with the most precise type recoverable.
    """
    import json

    field_metadata = field.metadata or {}

    # Attempt lossless reconstruction from stored Arrow type metadata.
    arrow_type: pa.DataType
    arrow_meta_str = field_metadata.get(_ARROW_META_KEY)
    if arrow_meta_str:
        try:
            arrow_meta = json.loads(arrow_meta_str)
            arrow_type = _arrow_type_from_metadata(field.dataType, arrow_meta)
        except Exception:
            # Corrupt / unrecognised metadata – fall back to structural mapping.
            arrow_type = spark_type_to_arrow_type(field.dataType, cast_options)
    else:
        arrow_type = spark_type_to_arrow_type(field.dataType, cast_options)

    # Re-encode user metadata to bytes, stripping the internal __arrow__ key.
    arrow_field_meta: Optional[dict[bytes, bytes]] = None
    user_meta = {k: v for k, v in field_metadata.items() if k != _ARROW_META_KEY}
    if user_meta:
        arrow_field_meta = {
            k.encode(): str(v).encode()
            for k, v in user_meta.items()
        }

    return pa.field(
        name=field.name,
        type=arrow_type,
        nullable=field.nullable,
        metadata=arrow_field_meta,
    )


# ---------------------------------------------------------------------------
# Schema converters
# ---------------------------------------------------------------------------

@register_converter(T.StructType, pa.Schema)
def spark_schema_to_arrow_schema(
    schema: T.StructType,
    options: Optional[CastOptions] = None,
) -> pa.Schema:
    """Convert a Spark :class:`~pyspark.sql.types.StructType` to a :class:`pyarrow.Schema`.

    Args:
        schema:  Spark schema (StructType) to convert.
        options: Optional :class:`CastOptions`.

    Returns:
        Arrow :class:`~pyarrow.Schema`.
    """
    opts = CastOptions.check_arg(options)
    return pa.schema([
        spark_field_to_arrow_field(field, opts)
        for field in schema.fields
    ])


@register_converter(pa.Schema, T.StructType)
def arrow_schema_to_spark_schema(
    schema: pa.Schema,
    options: Optional[CastOptions] = None,
) -> T.StructType:
    """Convert a :class:`pyarrow.Schema` to a Spark :class:`~pyspark.sql.types.StructType`.

    Args:
        schema:  Arrow schema to convert.
        options: Optional :class:`CastOptions`.

    Returns:
        Spark :class:`~pyspark.sql.types.StructType`.
    """
    opts = CastOptions.check_arg(options)
    return T.StructType([
        arrow_field_to_spark_field(field, opts)
        for field in schema
    ])


# ---------------------------------------------------------------------------
# Column-level safe-cast helpers
# ---------------------------------------------------------------------------

def _safe_cast_primitive(
    column: pyspark_sql.Column,
    target_type: T.DataType,
    safe: bool,
) -> pyspark_sql.Column:
    """Attempt a primitive Spark column cast, falling back to ``null`` when safe.

    PySpark's ``Column.cast`` raises a runtime AnalysisException for certain
    incompatible type pairs (e.g. binary → timestamp).  In safe mode we wrap
    the attempt in a ``try`` pattern using ``F.try_add`` is unavailable so we
    use a workaround: cast to string first, then to the target type via
    ``to_timestamp`` / numeric coercion helpers where possible.

    For most type pairs ``Column.cast`` simply returns ``null`` on bad data at
    execution time (it does not raise), so this helper is mainly needed for
    AnalysisException-prone conversions.

    Args:
        column:      Spark column expression to cast.
        target_type: Target Spark data type.
        safe:        If ``True``, coerce cast failures to ``null``.
                     If ``False``, let Spark raise on incompatible types.

    Returns:
        Spark column expression, typed as *target_type*.
    """
    if not safe:
        return column.cast(target_type)

    # For timestamp targets, route through string to avoid AnalysisException
    # when the source is, e.g., binary or boolean.
    if isinstance(target_type, (T.TimestampType, T.TimestampNTZType)):
        return F.try_to_timestamp(column.cast(T.StringType()))  # type: ignore[attr-defined]

    # For numeric targets, an intermediate string cast is resilient.
    if isinstance(target_type, (
        T.ByteType, T.ShortType, T.IntegerType, T.LongType,
        T.FloatType, T.DoubleType, T.DecimalType,
    )):
        return column.cast(T.StringType()).cast(target_type)

    # Default: rely on Spark's own null-on-error behaviour during execution.
    try:
        return column.cast(target_type)
    except Exception:
        # If the cast raises at *planning* time, return a typed null literal.
        return F.lit(None).cast(target_type)


def _fill_null_default(
    column: pyspark_sql.Column,
    target_field: T.StructField,
    source_nullable: bool,
) -> pyspark_sql.Column:
    """Fill null values when the target field is declared non-nullable.

    Args:
        column:          Column expression (already cast to target type).
        target_field:    Target Spark StructField; checked for nullability.
        source_nullable: Whether the source field allows nulls.

    Returns:
        Column expression with nulls replaced by the type's default value
        when *target_field* is non-nullable.
    """
    if source_nullable and not target_field.nullable:
        default_val = default_python_scalar(target_field)
        return F.when(column.isNull(), F.lit(default_val)).otherwise(column)
    return column


# ---------------------------------------------------------------------------
# JSON-parse helpers for string/binary → nested type coercion
# ---------------------------------------------------------------------------

def _is_string_or_binary(spark_type: T.DataType) -> bool:
    """Return ``True`` if *spark_type* is a string or binary variant.

    These are the two source types for which JSON parsing is attempted before
    casting to a compound target (struct / array / map).

    Args:
        spark_type: Spark data type to test.

    Returns:
        ``True`` for :class:`~pyspark.sql.types.StringType` and
        :class:`~pyspark.sql.types.BinaryType`; ``False`` otherwise.
    """
    return isinstance(spark_type, (T.StringType, T.BinaryType))


def _try_json_parse(
    column: pyspark_sql.Column,
    source_spark_type: T.DataType,
    target_spark_type: T.DataType,
    safe: bool,
) -> tuple[pyspark_sql.Column, T.DataType]:
    """Attempt to parse a string or binary column as JSON into *target_spark_type*.

    When the source is ``BinaryType`` the bytes are first decoded to a UTF-8
    string via ``CAST(col AS STRING)``; Spark's ``from_json`` then parses the
    resulting JSON text against the target schema.

    The function returns **both** the (possibly transformed) column *and* the
    effective source type after transformation so callers can decide whether
    the source is now structurally compatible with the target.

    Behaviour under ``safe``:

    - ``safe=True``:  ``from_json`` silently returns ``null`` for rows whose
      JSON is malformed or structurally incompatible – this is Spark's default
      behaviour and requires no extra wrapping.
    - ``safe=False``: Spark still returns ``null`` for bad rows at execution
      time (it cannot raise per-row), but the schema of the expression is
      enforced.  Callers in strict mode should validate output nullability
      after the fact when required.

    Args:
        column:            Source Spark column (``StringType`` or
                           ``BinaryType``).
        source_spark_type: Actual Spark type of *column*.
        target_spark_type: Compound target Spark type (struct / array / map).
        safe:              Whether to suppress structural errors.

    Returns:
        A ``(column, effective_source_type)`` 2-tuple.  When JSON parsing is
        applied the effective source type is set to *target_spark_type* so the
        caller skips the "wrong source type" guard.  When the source is not a
        string/binary type the column and its original type are returned
        unchanged.
    """
    if not _is_string_or_binary(source_spark_type):
        return column, source_spark_type

    # Binary → decode bytes to UTF-8 string first
    str_col = (
        column.cast(T.StringType())
        if isinstance(source_spark_type, T.BinaryType)
        else column
    )

    # Spark's from_json returns null on parse failure – safe by default.
    parsed = F.from_json(str_col, target_spark_type)

    return parsed, target_spark_type


# ---------------------------------------------------------------------------
# Column cast – compound types
# ---------------------------------------------------------------------------

def cast_spark_column_to_list(
    column: pyspark_sql.Column,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.Column:
    """Cast a Spark ``Column`` to an :class:`~pyspark.sql.types.ArrayType`.

    If the source column is ``StringType`` or ``BinaryType``, the value is
    first parsed as JSON via ``from_json(col, target_array_type)`` before
    per-element casting is applied.  This handles cases where array data has
    been serialised to a JSON string (e.g. ``"[1,2,3]"``).  Malformed JSON
    rows are decoded to ``null``; downstream element-level casting then applies
    normally to successfully parsed rows.

    Each element is cast individually using the Arrow-aware machinery so that
    nested types (e.g. array-of-structs) are handled correctly.

    Args:
        column:  Spark column backed by an ``ArrayType``-, ``StringType``-, or
                 ``BinaryType``-compatible value.
        options: Cast options including the target Arrow field and ``safe``
                 flag.

    Returns:
        Spark column expression typed as ``ArrayType``.

    Raises:
        ValueError: If the source column is not an ``ArrayType`` (and JSON
                    parsing was not applicable) and ``options.safe`` is
                    ``False``.
    """
    opts = CastOptions.check_arg(options)

    target_arrow_field = opts.target_arrow_field
    target_spark_field = opts.target_spark_field

    if target_arrow_field is None:
        return column

    source_spark_field = opts.source_spark_field
    source_spark_type  = source_spark_field.dataType

    # When the source is string/binary, attempt JSON decoding into the target
    # ArrayType before falling through to per-element casting.  from_json
    # returns null for malformed rows; valid JSON arrays are decoded in-place.
    column, source_spark_type = _try_json_parse(
        column, source_spark_type, target_spark_field.dataType, opts.safe
    )

    if not isinstance(source_spark_type, T.ArrayType):
        if opts.safe:
            # Return a null array literal typed correctly.
            return F.lit(None).cast(target_spark_field.dataType)
        raise ValueError(
            f"Cannot cast non-array field {source_spark_field!r} to {target_spark_field!r}"
        )

    element_opts = opts.copy(
        source_field=opts.source_child_arrow_field(index=0),
        target_field=opts.target_child_arrow_field(index=0),
    )

    casted = F.transform(column, lambda x: cast_spark_column(x, element_opts))
    return casted.cast(target_spark_field.dataType)


def cast_spark_column_to_struct(
    column: pyspark_sql.Column,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.Column:
    """Cast a Spark ``Column`` to a :class:`~pyspark.sql.types.StructType`.

    If the source column is ``StringType`` or ``BinaryType``, the value is
    first parsed as JSON via ``from_json(col, target_struct_type)`` before
    field-level casting is applied.  This handles cases where struct data has
    been serialised to a JSON string (e.g. ``'{"price":1.5,"qty":10}'``).
    Malformed JSON rows are decoded to ``null``; ``getField`` calls on
    successfully parsed rows then work as normal.

    Fields are matched by name (case-insensitive unless
    ``options.strict_match_names`` is set).  Missing source fields are filled
    with type-appropriate defaults in safe mode; they raise in strict mode.

    Args:
        column:  Spark column backed by a ``StructType``-, ``StringType``-, or
                 ``BinaryType``-compatible value.
        options: Cast options including the target Arrow field and ``safe``
                 flag.

    Returns:
        Spark column expression typed as ``StructType``.

    Raises:
        ValueError: In strict mode, if the source is not a struct (and JSON
                    parsing was not applicable) or a required field is absent.
    """
    opts = CastOptions.check_arg(options)

    target_arrow_field = opts.target_field
    target_spark_field = opts.target_spark_field

    if target_arrow_field is None:
        return column

    target_spark_type: T.StructType = target_spark_field.dataType
    source_spark_field = opts.source_spark_field
    source_spark_type  = source_spark_field.dataType

    # When the source is string/binary, attempt JSON decoding into the target
    # StructType before field-level casting.  from_json returns null for
    # malformed rows; valid JSON objects are decoded into the target struct
    # schema so getField() calls below work normally.
    column, source_spark_type = _try_json_parse(
        column, source_spark_type, target_spark_type, opts.safe
    )

    if not isinstance(source_spark_type, T.StructType):
        if opts.safe:
            return F.lit(None).cast(target_spark_type)
        raise ValueError(
            f"Cannot cast non-struct field {source_spark_field!r} to {target_spark_field!r}"
        )

    source_spark_fields: list[T.StructField] = list(source_spark_type.fields)
    source_arrow_fields: list[pa.Field]      = [
        spark_field_to_arrow_field(f) for f in source_spark_fields
    ]
    target_arrow_fields: list[pa.Field]      = list(target_arrow_field.type)
    target_spark_fields: list[T.StructField] = list(target_spark_type.fields)

    # Build name → index lookup (optionally case-insensitive)
    name_to_index: dict[str, int] = {
        f.name: idx for idx, f in enumerate(source_spark_fields)
    }
    if not opts.strict_match_names:
        name_to_index.update({
            f.name.casefold(): idx for idx, f in enumerate(source_spark_fields)
        })

    children: list[pyspark_sql.Column] = []
    found_source_names: set[str] = set()

    for child_idx, child_target_spark_field in enumerate(target_spark_fields):
        child_target_arrow_field: pa.Field = target_arrow_fields[child_idx]

        find_name  = (
            child_target_spark_field.name
            if opts.strict_match_names
            else child_target_spark_field.name.casefold()
        )
        source_idx = name_to_index.get(find_name)

        if source_idx is None:
            if not opts.safe and not opts.add_missing_columns:
                raise ValueError(
                    f"Missing struct field {child_target_arrow_field!r} in source; "
                    f"available: {[f.name for f in source_spark_fields]}"
                )
            # Safe / add_missing_columns: fill with typed default literal.
            dv = default_arrow_scalar(
                dtype=child_target_arrow_field.type,
                nullable=child_target_arrow_field.nullable,
            )
            casted_col = F.lit(dv.as_py()).cast(child_target_spark_field.dataType)
        else:
            child_source_arrow_field = source_arrow_fields[source_idx]
            child_source_spark_field = source_spark_fields[source_idx]
            found_source_names.add(child_source_spark_field.name)

            casted_col = cast_spark_column(
                column.getField(child_source_arrow_field.name),
                opts.copy(
                    source_field=child_source_arrow_field,
                    target_field=child_target_arrow_field,
                ),
            )

        children.append(casted_col.alias(child_target_spark_field.name))

    return F.struct(*children)


def cast_spark_column_to_map(
    column: pyspark_sql.Column,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.Column:
    """Cast a Spark ``Column`` to a :class:`~pyspark.sql.types.MapType`.

    If the source column is ``StringType`` or ``BinaryType``, the value is
    first parsed as JSON via ``from_json(col, target_map_type)`` before
    entry-level casting is applied.  Spark's ``from_json`` interprets a JSON
    object (``{"key":"value",...}``) as a ``MapType`` when the schema is a
    ``MapType``.  Malformed JSON rows become ``null``.

    Keys and values are cast individually using the Arrow-aware machinery.
    Map entries are accessed via ``map_entries``; the transformed entries are
    rebuilt with ``map_from_entries``.

    Args:
        column:  Spark column backed by a ``MapType``-, ``StringType``-, or
                 ``BinaryType``-compatible value.
        options: Cast options including the target Arrow field and ``safe``
                 flag.

    Returns:
        Spark column expression typed as ``MapType``.

    Raises:
        ValueError: In strict mode, if the source is not a map (and JSON
                    parsing was not applicable), or if the Arrow target type
                    is not a map type.
    """
    opts = CastOptions.check_arg(options)

    target_arrow_field = opts.target_field
    target_spark_field = opts.target_spark_field

    if target_arrow_field is None:
        return column

    target_spark_type: T.MapType = target_spark_field.dataType
    source_spark_field = opts.source_spark_field
    source_spark_type  = source_spark_field.dataType

    # When the source is string/binary, attempt JSON decoding into the target
    # MapType before entry-level casting.  Spark's from_json parses JSON
    # objects ({"k":"v",...}) into MapType when the schema is a MapType.
    # Malformed rows become null.
    column, source_spark_type = _try_json_parse(
        column, source_spark_type, target_spark_type, opts.safe
    )

    if not isinstance(source_spark_type, T.MapType):
        if opts.safe:
            return F.lit(None).cast(target_spark_type)
        raise ValueError(
            f"Cannot cast non-map field {source_spark_field!r} to {target_spark_field!r}"
        )

    target_map_type = target_arrow_field.type
    if not pat.is_map(target_map_type):
        if opts.safe:
            return F.lit(None).cast(target_spark_type)
        raise ValueError(
            f"Expected Arrow map type for {target_arrow_field!r}, got {target_map_type!r}"
        )

    # Build Arrow / Spark field wrappers for key and value
    target_key_arrow_field:   pa.Field = target_map_type.key_field
    target_value_arrow_field: pa.Field = target_map_type.item_field

    source_key_spark_field = T.StructField(
        name=f"{source_spark_field.name}_key",
        dataType=source_spark_type.keyType,
        nullable=False,   # Spark map keys are always non-null
    )
    source_value_spark_field = T.StructField(
        name=f"{source_spark_field.name}_value",
        dataType=source_spark_type.valueType,
        nullable=source_spark_type.valueContainsNull,
    )
    source_key_arrow_field   = spark_field_to_arrow_field(source_key_spark_field)
    source_value_arrow_field = spark_field_to_arrow_field(source_value_spark_field)

    key_opts   = opts.copy(
        source_field=source_key_arrow_field,
        target_field=target_key_arrow_field,
    )
    value_opts = opts.copy(
        source_field=source_value_arrow_field,
        target_field=target_value_arrow_field,
    )

    # Transform entries: array<struct<key, value>>
    entries        = F.map_entries(column)
    casted_entries = F.transform(
        entries,
        lambda entry: F.struct(
            cast_spark_column(entry["key"],   key_opts).alias("key"),
            cast_spark_column(entry["value"], value_opts).alias("value"),
        ),
    )
    casted_map = F.map_from_entries(casted_entries)

    # Enforce exact target MapType (includes valueContainsNull)
    return casted_map.cast(target_spark_type)


# ---------------------------------------------------------------------------
# Column cast – main entry point
# ---------------------------------------------------------------------------

@register_converter(pyspark_sql.Column, pyspark_sql.Column)
def cast_spark_column(
    column: pyspark_sql.Column,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.Column:
    """Cast a single Spark ``Column`` to an Arrow-specified target type.

    This is the primary column-level cast function.  It dispatches to the
    appropriate compound-type handler (struct / list / map) or falls back to
    a primitive Spark ``cast``.

    Behaviour is controlled by ``options.safe``:

    - **safe=True** (default): incompatible casts coerce the value to
      ``null``; missing fields receive type-appropriate defaults.
    - **safe=False**: any incompatible cast or missing field raises immediately.

    Args:
        column:  Spark column expression to cast.
        options: :class:`CastOptions` carrying the target Arrow field
                 (``target_arrow_field`` or a ``pa.Field`` / ``pa.Schema``
                 shorthand) and the ``safe`` flag.

    Returns:
        Spark column expression typed as the target and renamed to the
        target field name.

    Raises:
        AssertionError: If ``options.source_spark_field`` is not set.
        ValueError:     In strict mode, if a compound-type cast is
                        impossible.
    """
    opts = CastOptions.check_arg(options)
    target_spark_field = opts.target_spark_field

    if target_spark_field is None:
        # No target specification → pass through unchanged.
        return column

    target_spark_type  = target_spark_field.dataType
    source_spark_field = opts.source_spark_field
    assert source_spark_field is not None, (
        "cast_spark_column requires options.source_spark_field to be set"
    )

    # Dispatch to compound-type handlers
    if isinstance(target_spark_type, T.StructType):
        casted = cast_spark_column_to_struct(column, opts)
    elif isinstance(target_spark_type, T.ArrayType):
        casted = cast_spark_column_to_list(column, opts)
    elif isinstance(target_spark_type, T.MapType):
        casted = cast_spark_column_to_map(column, opts)
    else:
        casted = _safe_cast_primitive(column, target_spark_type, safe=opts.safe)

    # Fill nulls for non-nullable targets
    casted = _fill_null_default(
        casted,
        target_field=target_spark_field,
        source_nullable=(source_spark_field.nullable if source_spark_field else True),
    )

    return casted.alias(target_spark_field.name)


# ---------------------------------------------------------------------------
# DataFrame cast
# ---------------------------------------------------------------------------

@register_converter(pyspark_sql.DataFrame, pyspark_sql.DataFrame)
def cast_spark_dataframe(
    dataframe: pyspark_sql.DataFrame,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.DataFrame:
    """Cast a Spark ``DataFrame`` to a target Arrow schema **without collecting data**.

    The function:

    1. Maps the target Arrow schema to an equivalent Spark schema.
    2. For each target column, locates the corresponding source column by name
       (case-insensitive unless ``options.strict_match_names``).
    3. Casts source → target using :func:`cast_spark_column`.
    4. Fills missing columns with type-appropriate default literals
       (``options.add_missing_columns=True``) **or** raises
       (``safe=False`` + ``add_missing_columns=False``).
    5. Optionally preserves extra source columns
       (``options.allow_add_columns=True``).

    Behaviour is controlled by ``options.safe``:

    - **safe=True**: missing columns → defaults; bad casts → ``null``.
    - **safe=False**: any mismatch raises immediately.

    Args:
        dataframe: Input Spark DataFrame.
        options:   :class:`CastOptions` including ``target_arrow_schema``,
                   ``safe``, ``strict_match_names``, ``add_missing_columns``,
                   and ``allow_add_columns``.

    Returns:
        New Spark DataFrame with the target schema applied.
    """
    opts = CastOptions.check_arg(options)
    target_arrow_schema = opts.target_arrow_schema

    if target_arrow_schema is None:
        return dataframe

    source_spark_fields = dataframe.schema
    source_arrow_fields = [spark_field_to_arrow_field(f) for f in source_spark_fields]

    target_arrow_fields: list[pa.Field]      = list(target_arrow_schema)
    target_spark_fields: list[T.StructField] = [
        arrow_field_to_spark_field(f) for f in target_arrow_fields
    ]
    target_spark_schema: T.StructType        = arrow_schema_to_spark_schema(
        target_arrow_schema, None
    )

    # Name → index lookup for source columns
    source_name_to_index: dict[str, int] = {
        f.name: idx for idx, f in enumerate(source_arrow_fields)
    }
    if not opts.strict_match_names:
        source_name_to_index.update({
            f.name.casefold(): idx for idx, f in enumerate(source_arrow_fields)
        })

    casted_columns: list[tuple[T.StructField, pyspark_sql.Column]] = []
    found_source_names: set[str] = set()

    for tgt_idx, child_target_spark_field in enumerate(target_spark_fields):
        child_target_arrow_field = target_arrow_fields[tgt_idx]

        find_name  = (
            child_target_spark_field.name
            if opts.strict_match_names
            else child_target_spark_field.name.casefold()
        )
        source_idx = source_name_to_index.get(find_name)

        if source_idx is None:
            # --- Missing source column ---
            if not opts.safe and not opts.add_missing_columns:
                raise ValueError(
                    f"Column '{child_target_spark_field.name}' is missing from the "
                    f"source DataFrame and safe=False / add_missing_columns=False"
                )
            dv = default_arrow_scalar(
                dtype=child_target_arrow_field.type,
                nullable=child_target_arrow_field.nullable,
            )
            casted_col = F.lit(dv.as_py()).cast(child_target_spark_field.dataType)
        else:
            # --- Present source column ---
            child_source_arrow_field = source_arrow_fields[source_idx]
            child_source_spark_field = source_spark_fields[source_idx]
            found_source_names.add(child_source_spark_field.name)

            casted_col = cast_spark_column(
                dataframe[child_source_spark_field.name],
                opts.copy(
                    source_field=child_source_arrow_field,
                    target_field=child_target_arrow_field,
                ),
            )

        casted_columns.append((child_target_spark_field, casted_col))

    # Optionally keep extra source columns not present in the target schema
    if opts.allow_add_columns:
        for src_field in source_spark_fields:
            if src_field.name not in found_source_names:
                casted_columns.append(
                    (src_field, dataframe[src_field.name])
                )

    result = dataframe.select(*[col for _, col in casted_columns])

    # Re-apply exact target schema to fix nullability metadata
    return result.sparkSession.createDataFrame(result.rdd, schema=target_spark_schema)


# ---------------------------------------------------------------------------
# DataFrame ↔ Arrow Table converters
# ---------------------------------------------------------------------------

@register_converter(pyspark_sql.DataFrame, pa.Table)
def spark_dataframe_to_arrow_table(
    dataframe: pyspark_sql.DataFrame,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """Convert a Spark ``DataFrame`` to a :class:`pyarrow.Table`.

    If ``options.target_arrow_schema`` is provided, the DataFrame is first
    cast via :func:`cast_spark_dataframe`; the resulting Arrow schema is taken
    from the options rather than inferred from Spark.

    .. warning::
        This function calls :py:meth:`~pyspark.sql.DataFrame.toArrow` which
        **collects** the entire DataFrame to the driver.  Use only on small-to-
        medium datasets or after filtering/aggregating in Spark first.

    Args:
        dataframe: Input Spark DataFrame.
        options:   Optional :class:`CastOptions`.

    Returns:
        Arrow Table (all data on driver).
    """
    opts = CastOptions.check_arg(options)

    if opts.target_arrow_schema is not None:
        dataframe   = cast_spark_dataframe(dataframe, opts)
        arrow_schema = opts.target_arrow_schema
    else:
        arrow_schema = pa.schema([
            spark_field_to_arrow_field(f, options)
            for f in dataframe.schema
        ])

    return cast_arrow_tabular(
        dataframe.toArrow(),
        CastOptions.check_arg(arrow_schema),
    )


@register_converter(pa.Table, pyspark_sql.DataFrame)
def arrow_table_to_spark_dataframe(
    table: pa.Table,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.DataFrame:
    """Convert a :class:`pyarrow.Table` to a Spark ``DataFrame``.

    If a target schema is supplied, :func:`cast_arrow_tabular` is applied
    before creating the Spark DataFrame so that Arrow-level casting (e.g.
    dictionary decoding, temporal coercions) is performed first.

    Args:
        table:   Arrow Table to convert.
        options: Optional :class:`CastOptions`.

    Returns:
        Spark DataFrame backed by the Arrow data.

    Raises:
        RuntimeError: If no active :class:`~pyspark.sql.SparkSession` is found.
    """
    opts = CastOptions.check_arg(options)

    if opts.target_arrow_schema is not None:
        table = cast_arrow_tabular(table, opts)

    spark = pyspark_sql.SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(
            "An active SparkSession is required to convert Arrow data to Spark. "
            "Call SparkSession.builder.getOrCreate() first."
        )

    spark_schema = arrow_schema_to_spark_schema(table.schema, None)
    return spark.createDataFrame(table, schema=spark_schema)


# ---------------------------------------------------------------------------
# Generic converters
# ---------------------------------------------------------------------------

@register_converter(Any, pyspark_sql.DataFrame)
def any_to_spark_dataframe(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.DataFrame:
    """Convert any supported object to a Spark ``DataFrame``.

    Supported input types (resolved in this order):

    - ``pyspark.sql.DataFrame`` – cast directly via :func:`cast_spark_dataframe`.
    - ``None`` – returns an empty DataFrame typed according to
      ``options.target_spark_schema``.
    - Anything else – converted to Polars first (via internal helpers), then
      to an Arrow Table, then to Spark.

    Args:
        obj:     Input object.
        options: Optional :class:`CastOptions`.

    Returns:
        Spark DataFrame.

    Raises:
        RuntimeError: If no active SparkSession is found.
    """
    spark = pyspark_sql.SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(
            "An active SparkSession is required.  "
            "Call SparkSession.builder.getOrCreate() first."
        )

    if isinstance(obj, pyspark_sql.DataFrame):
        return cast_spark_dataframe(obj, options)

    opts = CastOptions.check_arg(options)

    if obj is None:
        return spark.createDataFrame([], schema=opts.target_spark_schema)

    namespace = ObjectSerde.full_namespace(obj)

    if namespace.startswith("pyarrow"):
        if isinstance(obj, pa.RecordBatch):
            obj = pa.Table.from_batches([obj], schema=obj.schema) # type: ignore
        elif hasattr(obj, "to_table"):
            obj = obj.to_table()

        if isinstance(obj, pa.Table):
            spark_schema = arrow_schema_to_spark_schema(obj.schema, None)
            df = spark.createDataFrame(obj, schema=spark_schema)
        else:
            raise ValueError(
                f"Cannot convert {type(obj)} to pyspark.sql.DataFrame"
            )
    else:
        # Route through Polars as the intermediate representation for arbitrary inputs.
        from ..polars.cast import any_to_polars_dataframe, polars_dataframe_to_arrow_table

        arrow_table = polars_dataframe_to_arrow_table(
            any_to_polars_dataframe(obj, opts), opts
        )
        spark_schema = arrow_schema_to_spark_schema(arrow_table.schema, None)
        df = spark.createDataFrame(arrow_table, schema=spark_schema)

    return cast_spark_dataframe(df, opts)


def any_spark_to_arrow_field(
    obj: Any,
    options: Optional[CastOptions],
) -> pa.Field:
    """Derive an Arrow :class:`~pyarrow.Field` from a heterogeneous Spark object.

    Accepts:

    - :class:`pyarrow.Field` – returned as-is (after passing through
      :func:`arrow_field_to_field` normalisation).
    - :class:`~pyspark.sql.DataFrame` – schema wrapped in a struct field
      named ``"root"``.
    - :class:`~pyspark.sql.types.StructField` – converted via
      :func:`spark_field_to_arrow_field`.
    - :class:`~pyspark.sql.types.DataType` – wrapped in an anonymous field.
    - :class:`~pyspark.sql.Column` – requires ``options.source_arrow_field`` to
      be pre-set in the options.

    Args:
        obj:     Input object.
        options: Optional :class:`CastOptions`.

    Returns:
        Arrow field.

    Raises:
        TypeError: If the input type cannot be mapped to an Arrow field.
    """
    if isinstance(obj, pa.Field):
        return obj

    if isinstance(obj, pyspark_sql.DataFrame):
        obj = obj.schema

    if isinstance(obj, T.StructField):
        return spark_field_to_arrow_field(obj, options)

    if isinstance(obj, T.DataType):
        return arrow_type_to_field(spark_type_to_arrow_type(obj), options)

    opts = CastOptions.check_arg(options)

    if isinstance(obj, pyspark_sql.Column):
        if opts.source_arrow_field is not None:
            return opts.source_arrow_field
        raise TypeError(
            "Cannot derive an Arrow field from a Spark Column without "
            "options.source_arrow_field being set."
        )

    raise TypeError(
        f"Cannot convert {type(obj).__qualname__!r} to pyarrow.Field; "
        f"expected DataFrame, StructField, DataType, Column, or Field."
    )


# ---------------------------------------------------------------------------
# Registry wrappers (thin shims so the converter registry stays tidy)
# ---------------------------------------------------------------------------

@register_converter(pyspark_sql.DataFrame, T.DataType)
def spark_dataframe_to_spark_type(
    df: pyspark_sql.DataFrame,
    options: Optional[CastOptions] = None,
) -> T.DataType:
    """Return the Spark schema of *df* as a :class:`~pyspark.sql.types.DataType`.

    Args:
        df:      Spark DataFrame.
        options: Unused; present for registry signature consistency.

    Returns:
        Spark :class:`~pyspark.sql.types.StructType` schema.
    """
    return df.schema


@register_converter(pyspark_sql.DataFrame, T.StructField)
def spark_dataframe_to_spark_field(
    df: pyspark_sql.DataFrame,
    options: Optional[CastOptions] = None,
) -> T.StructField:
    """Wrap the Spark DataFrame schema in a named :class:`~pyspark.sql.types.StructField`.

    The field name is the DataFrame alias when set, otherwise ``"root"``.

    Args:
        df:      Spark DataFrame.
        options: Unused; present for registry signature consistency.

    Returns:
        Spark :class:`~pyspark.sql.types.StructField`.
    """
    return T.StructField(
        name=df.getAlias() or "root",
        dataType=df.schema,
        nullable=False,
    )


@register_converter(pyspark_sql.DataFrame, pa.Field)
def spark_dataframe_to_arrow_field(
    df: pyspark_sql.DataFrame,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Return an Arrow field representation of the DataFrame schema.

    Args:
        df:      Spark DataFrame.
        options: Optional :class:`CastOptions`.

    Returns:
        Arrow :class:`~pyarrow.Field`.
    """
    return spark_field_to_arrow_field(
        spark_dataframe_to_spark_field(df, options), options
    )


@register_converter(pyspark_sql.DataFrame, pa.Schema)
def spark_dataframe_to_arrow_schema(
    df: pyspark_sql.DataFrame,
    options: Optional[CastOptions] = None,
) -> pa.Schema:
    """Return an Arrow schema representation of the DataFrame.

    Args:
        df:      Spark DataFrame.
        options: Optional :class:`CastOptions`.

    Returns:
        Arrow :class:`~pyarrow.Schema`.
    """
    return arrow_field_to_schema(
        spark_dataframe_to_arrow_field(df, options), options
    )


@register_converter(pa.DataType, T.DataType)
def _arrow_type_to_spark_type_reg(
    dtype: pa.DataType,
    options: Optional[CastOptions] = None,
) -> T.DataType:
    return arrow_type_to_spark_type(dtype, options)


@register_converter(pa.Field, T.StructField)
def _arrow_field_to_spark_field_reg(
    field: pa.Field,
    options: Optional[CastOptions] = None,
) -> T.StructField:
    return arrow_field_to_spark_field(field, options)


@register_converter(T.DataType, pa.DataType)
def _spark_type_to_arrow_type_reg(
    dtype: T.DataType,
    options: Optional[CastOptions] = None,
) -> pa.DataType:
    return spark_type_to_arrow_type(dtype, options)


@register_converter(T.StructField, pa.Field)
def _spark_field_to_arrow_field_reg(
    field: T.StructField,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    return spark_field_to_arrow_field(field, options)


@register_converter(T.StructField, pa.Schema)
def _spark_struct_field_to_arrow_schema_reg(
    schema: T.StructType,
    options: Optional[CastOptions] = None,
) -> pa.Schema:
    return spark_schema_to_arrow_schema(schema, options)