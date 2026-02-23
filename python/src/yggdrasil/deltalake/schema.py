"""
schema.py — PyArrow schema → Delta ``schemaString`` serialisation.

Delta Lake stores table schemas as a JSON blob (``schemaString``) inside the
``metaData`` log action.  This module converts PyArrow schemas and types into
that format so new tables and schema-evolution commits can be written without
a Spark runtime.

Public API
----------
arrow_schema_to_schema_string(schema)   PyArrow schema → JSON schemaString

Internal helpers (not exported)
--------------------------------
_arrow_type_to_delta(arrow_type)        pa.DataType → Delta type (str or dict)
_arrow_field_to_delta(field)            pa.Field    → Delta field dict
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

__all__ = ["arrow_schema_to_schema_string"]


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------

def _arrow_type_to_delta(arrow_type: pa.DataType) -> Any:
    """Recursively convert a PyArrow ``DataType`` to a Delta schema type.

    Scalar types become JSON strings (``"integer"``, ``"double"``, …).
    Complex types become nested dicts matching the Delta schema JSON spec.

    Unsigned integer types are widened to the next larger signed type because
    Spark / Delta has no unsigned integer concept:

    * ``uint8``  → ``"short"``  (int16)
    * ``uint16`` → ``"integer"`` (int32)
    * ``uint32`` → ``"long"``   (int64)
    * ``uint64`` → ``"decimal(20,0)"``

    Args:
        arrow_type: Any ``pa.DataType`` instance.

    Returns:
        A JSON-serialisable Delta type — a ``str`` for scalars or a ``dict``
        for ``array``, ``map``, and ``struct`` complex types.

    Raises:
        ValueError: For Arrow types with no Delta equivalent.
    """
    # --- Boolean ---
    if pa.types.is_boolean(arrow_type):
        return "boolean"

    # --- Signed integers ---
    if pa.types.is_int8(arrow_type):
        return "byte"
    if pa.types.is_int16(arrow_type):
        return "short"
    if pa.types.is_int32(arrow_type):
        return "integer"
    if pa.types.is_int64(arrow_type):
        return "long"

    # --- Unsigned integers (widened) ---
    if pa.types.is_uint8(arrow_type):
        return "short"
    if pa.types.is_uint16(arrow_type):
        return "integer"
    if pa.types.is_uint32(arrow_type):
        return "long"
    if pa.types.is_uint64(arrow_type):
        return "decimal(20,0)"

    # --- Floating point ---
    if pa.types.is_float16(arrow_type) or pa.types.is_float32(arrow_type):
        return "float"
    if pa.types.is_float64(arrow_type):
        return "double"

    # --- Decimal ---
    if pa.types.is_decimal(arrow_type):
        return f"decimal({arrow_type.precision},{arrow_type.scale})"

    # --- String / binary ---
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return "binary"

    # --- Temporal ---
    if pa.types.is_date(arrow_type):
        return "date"
    if pa.types.is_timestamp(arrow_type):
        # tz-aware → TIMESTAMP; naive → TIMESTAMP_NTZ (Delta spec §timestamp_ntz)
        return "timestamp" if arrow_type.tz is not None else "timestamp_ntz"
    if pa.types.is_time(arrow_type):
        # Spark has no TIME type; serialise as string for round-trip safety.
        return "string"

    # --- Complex types ---
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return {
            "type":         "array",
            "elementType":  _arrow_type_to_delta(arrow_type.value_type),
            "containsNull": True,
        }
    if pa.types.is_map(arrow_type):
        return {
            "type":              "map",
            "keyType":           _arrow_type_to_delta(arrow_type.key_type),
            "valueType":         _arrow_type_to_delta(arrow_type.item_type),
            "valueContainsNull": True,
        }
    if pa.types.is_struct(arrow_type):
        return {
            "type":   "struct",
            "fields": [
                _arrow_field_to_delta(arrow_type.field(i))
                for i in range(arrow_type.num_fields)
            ],
        }

    raise ValueError(
        f"Cannot convert Arrow type '{arrow_type}' to a Delta schema type."
    )


def _arrow_field_to_delta(f: pa.Field) -> dict[str, Any]:
    """Convert a PyArrow ``Field`` to a Delta schema field dict.

    Extracts an optional ``comment`` from the field's Arrow metadata
    (key ``b"comment"``) and places it in the Delta ``metadata`` sub-dict.

    Args:
        f: A ``pa.Field`` instance.

    Returns:
        ``{"name": ..., "type": ..., "nullable": ..., "metadata": {...}}``.
    """
    comment: str | None = None
    if f.metadata:
        raw = f.metadata.get(b"comment")
        if raw:
            comment = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    return {
        "name":     f.name,
        "type":     _arrow_type_to_delta(f.type),
        "nullable": f.nullable,
        "metadata": {"comment": comment} if comment else {},
    }


# ---------------------------------------------------------------------------
# Public serialiser
# ---------------------------------------------------------------------------

def arrow_schema_to_schema_string(schema: pa.Schema) -> str:
    """Serialise a PyArrow schema to a Delta ``schemaString`` JSON blob.

    The resulting string is suitable for the ``schemaString`` field of a
    Delta ``metaData`` log action.

    Args:
        schema: PyArrow schema to serialise.  Field-level comments stored in
                Arrow metadata (``b"comment"`` key) are preserved in the Delta
                ``metadata`` sub-dict.

    Returns:
        Compact JSON string, e.g.::

            '{"type":"struct","fields":[{"name":"price","type":"double",...}]}'

    Example::

        schema = pa.schema([
            pa.field("trade_date", pa.date32(),        nullable=False),
            pa.field("commodity",  pa.string(),         nullable=False),
            pa.field("price",      pa.float64()),
            pa.field("notional",   pa.decimal128(18, 6)),
        ])
        schema_string = arrow_schema_to_schema_string(schema)
    """
    delta_schema = {
        "type":   "struct",
        "fields": [_arrow_field_to_delta(schema.field(i)) for i in range(len(schema))],
    }
    return json.dumps(delta_schema, separators=(",", ":"))