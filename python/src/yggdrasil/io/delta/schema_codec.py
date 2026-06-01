"""Delta schema-string codec using yggdrasil's Field/DataType system.

Delta stores schemas as Spark ``StructType.json()``. This module
converts between that wire format and yggdrasil's type system
(:class:`Schema`, :class:`Field`, :class:`DataType`), which then
projects to any target (Arrow, Spark, Polars, pandas) on demand.
Arrow is not privileged — it's just another projection target.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.base import DataType
from yggdrasil.pickle import json as ygg_json

__all__ = [
    "arrow_schema_to_spark_json", "spark_json_to_arrow_schema",
    "schema_to_spark_json", "spark_json_to_schema",
]

_DDL_TO_DELTA = {"INT": "integer", "BIGINT": "long"}


# ---------------------------------------------------------------------------
# Spark JSON → yggdrasil Schema
# ---------------------------------------------------------------------------

def spark_json_to_schema(schema_string: str) -> Schema:
    if not schema_string:
        return Schema.empty()
    payload = ygg_json.loads(schema_string)
    if payload.get("type") != "struct":
        raise ValueError(f"Expected struct schemaString; got {payload.get('type')!r}.")
    return Schema(fields=[_parse_field(f) for f in payload.get("fields", [])])


def _parse_field(raw: Mapping[str, Any]) -> Field:
    meta = raw.get("metadata") or None
    metadata: "dict[bytes, bytes] | None" = None
    if isinstance(meta, Mapping) and meta:
        # Spark metadata is string-keyed JSON. Round-trip back into
        # Arrow's bytes-keyed convention so the field's metadata is
        # available via the same API on both sides.
        metadata = {}
        for k, v in meta.items():
            key = k.encode("utf-8") if isinstance(k, str) else bytes(k)
            value = v.encode("utf-8") if isinstance(v, str) else (
                v if isinstance(v, bytes) else str(v).encode("utf-8")
            )
            metadata[key] = value
    return Field(
        name=str(raw["name"]),
        dtype=_parse_type(raw["type"]),
        nullable=bool(raw.get("nullable", True)),
        metadata=metadata,
    )


def _parse_type(t: Any) -> DataType:
    if isinstance(t, str):
        n = t.strip().lower()
        if n == "void":
            from yggdrasil.data.types.primitive import BinaryType
            return BinaryType()
        return DataType.from_str(n)
    if isinstance(t, Mapping):
        kind = t.get("type")
        if kind == "struct":
            from yggdrasil.data.types.nested import StructType
            return StructType(fields=[_parse_field(f) for f in t.get("fields", [])])
        if kind == "array":
            from yggdrasil.data.types.nested import ArrayType
            return ArrayType(item_field=Field(
                name="item", dtype=_parse_type(t["elementType"]),
                nullable=t.get("containsNull", True),
            ))
        if kind == "map":
            from yggdrasil.data.types.nested import MapType, StructType
            return MapType(item_field=Field(name="entries", dtype=StructType(fields=[
                Field(name="key", dtype=_parse_type(t["keyType"]), nullable=False),
                Field(name="value", dtype=_parse_type(t["valueType"]),
                      nullable=t.get("valueContainsNull", True)),
            ])))
    from yggdrasil.data.types.primitive import StringType
    return StringType()


# ---------------------------------------------------------------------------
# yggdrasil Schema → Spark JSON
# ---------------------------------------------------------------------------

def schema_to_spark_json(schema: Schema) -> str:
    return ygg_json.dumps(
        {"type": "struct", "fields": [_field_to_spark(f) for f in schema.fields]},
        ensure_ascii=False, separators=(",", ":"), to_bytes=False,
    )


def _field_to_spark(field: Field) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    raw = field.metadata
    if raw:
        # Field.metadata is a bytes→bytes dict (Arrow convention);
        # Spark expects JSON-serializable string keys + values. Decode
        # both sides as UTF-8, falling back to repr() for values that
        # don't decode (e.g. encoded type-json blobs).
        for k, v in raw.items():
            try:
                key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            except UnicodeDecodeError:
                key = repr(k)
            if isinstance(v, bytes):
                try:
                    value = v.decode("utf-8")
                except UnicodeDecodeError:
                    value = repr(v)
            else:
                value = v
            metadata[key] = value
    return {
        "name": field.name,
        "type": _type_to_spark(field.dtype),
        "nullable": bool(field.nullable),
        "metadata": metadata,
    }


def _type_to_spark(dt: DataType) -> Any:
    from yggdrasil.data.types.nested import StructType, ArrayType, MapType
    if isinstance(dt, StructType):
        return {"type": "struct", "fields": [_field_to_spark(f) for f in dt.fields]}
    if isinstance(dt, ArrayType):
        return {"type": "array", "elementType": _type_to_spark(dt.item_field.dtype),
                "containsNull": dt.item_field.nullable}
    if isinstance(dt, MapType):
        kf, vf = dt.item_field.dtype.fields[0], dt.item_field.dtype.fields[1]
        return {"type": "map", "keyType": _type_to_spark(kf.dtype),
                "valueType": _type_to_spark(vf.dtype),
                "valueContainsNull": vf.nullable}
    # Primitive: DDL name → Delta lowercase convention
    spark_name = dt.as_spark().to_spark_name()
    paren = spark_name.find("(")
    if paren == -1:
        return _DDL_TO_DELTA.get(spark_name, spark_name.lower())
    head, tail = spark_name[:paren], spark_name[paren:]
    return f"{_DDL_TO_DELTA.get(head, head.lower())}{tail.replace(' ', '').lower()}"


# ---------------------------------------------------------------------------
# Arrow bridges (convenience — projects through Schema)
# ---------------------------------------------------------------------------

def arrow_schema_to_spark_json(schema: pa.Schema) -> str:
    return schema_to_spark_json(Schema.from_arrow(schema))


def spark_json_to_arrow_schema(schema_string: str) -> pa.Schema:
    return spark_json_to_schema(schema_string).to_arrow_schema()
