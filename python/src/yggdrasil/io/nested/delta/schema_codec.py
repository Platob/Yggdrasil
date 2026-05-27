"""Arrow / yggdrasil <-> Delta (Spark-flavoured) schema-string conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping

import pyarrow as pa

from yggdrasil.pickle import json as ygg_json

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema

__all__ = [
    "arrow_schema_to_spark_json", "spark_json_to_arrow_schema",
    "schema_to_spark_json", "spark_json_to_schema",
]

_DDL_HEAD_TO_DELTA = {"INT": "integer", "BIGINT": "long"}


# ---------------------------------------------------------------------------
# Arrow -> Spark JSON
# ---------------------------------------------------------------------------

def arrow_schema_to_spark_json(schema: pa.Schema) -> str:
    return ygg_json.dumps(
        {"type": "struct", "fields": [_arrow_field_to_spark(f) for f in schema]},
        ensure_ascii=False, separators=(",", ":"), to_bytes=False,
    )

def _arrow_field_to_spark(field: pa.Field) -> Dict[str, Any]:
    metadata: Dict[str, str] = {}
    if field.metadata:
        for k, v in field.metadata.items():
            try: metadata[k.decode("utf-8")] = v.decode("utf-8")
            except (AttributeError, UnicodeDecodeError): metadata[str(k)] = str(v)
    return {"name": field.name, "type": _arrow_type_to_spark(field.type),
            "nullable": bool(field.nullable), "metadata": metadata}

def _arrow_type_to_spark(t: pa.DataType) -> Any:
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return {"type": "array", "elementType": _arrow_type_to_spark(t.value_type), "containsNull": True}
    if pa.types.is_map(t):
        return {"type": "map", "keyType": _arrow_type_to_spark(t.key_type),
                "valueType": _arrow_type_to_spark(t.item_type), "valueContainsNull": True}
    if pa.types.is_struct(t):
        return {"type": "struct", "fields": [_arrow_field_to_spark(t.field(i)) for i in range(t.num_fields)]}
    from yggdrasil.data.types.base import DataType
    try: ddl = DataType.from_arrow_type(t).as_spark().to_spark_name()
    except Exception: return "binary"
    paren = ddl.find("(")
    if paren == -1: return _DDL_HEAD_TO_DELTA.get(ddl, ddl.lower())
    head, tail = ddl[:paren], ddl[paren:]
    return f"{_DDL_HEAD_TO_DELTA.get(head, head.lower())}{tail.replace(' ', '').lower()}"


# ---------------------------------------------------------------------------
# Spark JSON -> Arrow
# ---------------------------------------------------------------------------

def spark_json_to_arrow_schema(schema_string: str) -> pa.Schema:
    if not schema_string: return pa.schema([])
    payload = ygg_json.loads(schema_string)
    if payload.get("type") != "struct":
        raise ValueError(f"Expected struct schemaString; got {payload.get('type')!r}.")
    return pa.schema([_spark_field_to_arrow(f) for f in payload.get("fields", [])])

def _spark_field_to_arrow(field: Mapping[str, Any]) -> pa.Field:
    metadata = field.get("metadata") or {}
    encoded = {str(k).encode("utf-8"): str(v).encode("utf-8") for k, v in metadata.items()}
    return pa.field(str(field["name"]), _spark_type_to_arrow(field["type"]),
                    nullable=bool(field.get("nullable", True)), metadata=encoded or None)

def _spark_type_to_arrow(t: Any) -> pa.DataType:
    if isinstance(t, str):
        from yggdrasil.data.types.base import DataType
        n = t.strip().lower()
        if n == "void": return pa.null()
        try: return DataType.from_str(n).to_arrow()
        except Exception: return pa.string()
    if isinstance(t, Mapping):
        kind = t.get("type")
        if kind == "struct":
            return pa.struct([_spark_field_to_arrow(f) for f in t.get("fields", [])])
        if kind == "array": return pa.list_(_spark_type_to_arrow(t["elementType"]))
        if kind == "map":
            return pa.map_(_spark_type_to_arrow(t["keyType"]), _spark_type_to_arrow(t["valueType"]))
    return pa.string()


# ---------------------------------------------------------------------------
# yggdrasil.data.Schema bridges
# ---------------------------------------------------------------------------

def schema_to_spark_json(schema: "Schema") -> str:
    return arrow_schema_to_spark_json(schema.to_arrow_schema())

def spark_json_to_schema(schema_string: str) -> "Schema":
    from yggdrasil.data.schema import Schema
    return Schema.from_arrow(spark_json_to_arrow_schema(schema_string))
