"""Arrow ↔ Delta (Spark-flavoured) schema-string conversion.

Delta stores the table schema on the wire as a Spark
``StructType.json()`` payload — the same shape :func:`pyspark.sql.types.StructType.fromJson`
parses. We map between that JSON and :class:`pyarrow.Schema` here so
:class:`yggdrasil.delta.io.DeltaIO` doesn't need pyspark to serialize a
write target.

Coverage
--------

Primitives (``string``, ``long``, ``integer``, ``short``, ``byte``,
``double``, ``float``, ``boolean``, ``binary``, ``date``,
``timestamp``, ``timestamp_ntz``, fixed-precision ``decimal(p,s)``)
plus the three complex shapes (``struct``, ``array``, ``map``) cover
everything a portable Delta table holds. Unknown Spark-side types
fall back to ``string`` on the way in (best-effort load) and
:class:`pa.binary` on the way out — a deliberate "don't crash"
choice; pinning the unmapped shape to ``OBJECT`` would have the same
effect when the Arrow type later round-trips through the
yggdrasil :class:`Field`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

import pyarrow as pa


__all__ = [
    "arrow_schema_to_spark_json",
    "spark_json_to_arrow_schema",
]


# ---------------------------------------------------------------------------
# Arrow → Spark JSON
# ---------------------------------------------------------------------------


def arrow_schema_to_spark_json(schema: pa.Schema) -> str:
    """Render a pyarrow schema as a Delta-compatible schemaString."""
    fields = [_arrow_field_to_spark(f) for f in schema]
    payload: Dict[str, Any] = {"type": "struct", "fields": fields}
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _arrow_field_to_spark(field: pa.Field) -> Dict[str, Any]:
    metadata = {}
    if field.metadata:
        for k, v in field.metadata.items():
            try:
                metadata[k.decode("utf-8")] = v.decode("utf-8")
            except (AttributeError, UnicodeDecodeError):
                metadata[str(k)] = str(v)
    return {
        "name": field.name,
        "type": _arrow_type_to_spark(field.type),
        "nullable": bool(field.nullable),
        "metadata": metadata,
    }


def _arrow_type_to_spark(t: pa.DataType) -> Any:
    # Primitives
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    if pa.types.is_int64(t) or pa.types.is_uint64(t):
        return "long"
    if pa.types.is_int32(t) or pa.types.is_uint32(t):
        return "integer"
    if pa.types.is_int16(t) or pa.types.is_uint16(t):
        return "short"
    if pa.types.is_int8(t) or pa.types.is_uint8(t):
        return "byte"
    if pa.types.is_float64(t):
        return "double"
    if pa.types.is_float32(t):
        return "float"
    if pa.types.is_boolean(t):
        return "boolean"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_fixed_size_binary(t):
        return "binary"
    if pa.types.is_date32(t) or pa.types.is_date64(t):
        return "date"
    if pa.types.is_timestamp(t):
        # Delta distinguishes ``timestamp`` (with tz, stored as UTC)
        # from ``timestamp_ntz`` (no zone). Arrow encodes the same
        # split as the timestamp's ``tz`` attribute.
        return "timestamp_ntz" if t.tz is None else "timestamp"
    if pa.types.is_decimal(t):
        return f"decimal({t.precision},{t.scale})"
    if pa.types.is_null(t):
        return "void"

    # Complex
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return {
            "type": "array",
            "elementType": _arrow_type_to_spark(t.value_type),
            "containsNull": True,
        }
    if pa.types.is_map(t):
        return {
            "type": "map",
            "keyType": _arrow_type_to_spark(t.key_type),
            "valueType": _arrow_type_to_spark(t.item_type),
            "valueContainsNull": True,
        }
    if pa.types.is_struct(t):
        return {
            "type": "struct",
            "fields": [_arrow_field_to_spark(t.field(i)) for i in range(t.num_fields)],
        }

    # Unknown / unsupported → binary fallback. The intent is "round-
    # trips, doesn't crash"; a richer mapping can land later.
    return "binary"


# ---------------------------------------------------------------------------
# Spark JSON → Arrow
# ---------------------------------------------------------------------------


def spark_json_to_arrow_schema(schema_string: str) -> pa.Schema:
    """Parse a Delta schemaString back into a :class:`pa.Schema`."""
    if not schema_string:
        return pa.schema([])
    payload = json.loads(schema_string)
    if payload.get("type") != "struct":
        raise ValueError(
            f"Expected a struct-typed schemaString; got {payload.get('type')!r}."
        )
    fields = [_spark_field_to_arrow(f) for f in payload.get("fields", [])]
    return pa.schema(fields)


def _spark_field_to_arrow(field: Mapping[str, Any]) -> pa.Field:
    name = str(field["name"])
    nullable = bool(field.get("nullable", True))
    arrow_type = _spark_type_to_arrow(field["type"])
    metadata = field.get("metadata") or {}
    encoded: Dict[bytes, bytes] = {
        str(k).encode("utf-8"): str(v).encode("utf-8") for k, v in metadata.items()
    }
    return pa.field(name, arrow_type, nullable=nullable, metadata=encoded or None)


def _spark_type_to_arrow(t: Any) -> pa.DataType:
    if isinstance(t, str):
        return _spark_primitive_to_arrow(t)
    if isinstance(t, Mapping):
        kind = t.get("type")
        if kind == "struct":
            return pa.struct([_spark_field_to_arrow(f) for f in t.get("fields", [])])
        if kind == "array":
            return pa.list_(_spark_type_to_arrow(t["elementType"]))
        if kind == "map":
            return pa.map_(
                _spark_type_to_arrow(t["keyType"]),
                _spark_type_to_arrow(t["valueType"]),
            )
    return pa.string()


def _spark_primitive_to_arrow(name: str) -> pa.DataType:
    n = name.strip().lower()
    if n == "string":
        return pa.string()
    if n == "long":
        return pa.int64()
    if n == "integer":
        return pa.int32()
    if n == "short":
        return pa.int16()
    if n == "byte":
        return pa.int8()
    if n == "double":
        return pa.float64()
    if n == "float":
        return pa.float32()
    if n == "boolean":
        return pa.bool_()
    if n == "binary":
        return pa.binary()
    if n == "date":
        return pa.date32()
    if n == "timestamp":
        return pa.timestamp("us", tz="UTC")
    if n == "timestamp_ntz":
        return pa.timestamp("us")
    if n == "void" or n == "null":
        return pa.null()
    if n.startswith("decimal"):
        # ``decimal(10,2)`` — best-effort parse; fall back to (38, 18)
        # which is Spark's default unbounded.
        try:
            inner = n[n.index("(") + 1: n.index(")")]
            p, s = (int(x.strip()) for x in inner.split(","))
            return pa.decimal128(p, s)
        except Exception:
            return pa.decimal128(38, 18)
    return pa.string()
