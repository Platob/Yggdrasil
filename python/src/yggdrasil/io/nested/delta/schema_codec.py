"""Arrow / yggdrasil ↔ Delta (Spark-flavoured) schema-string conversion.

Delta stores the table schema on the wire as a Spark
``StructType.json()`` payload — the same shape :func:`pyspark.sql.types.StructType.fromJson`
parses. We map between that JSON and :class:`pyarrow.Schema` /
:class:`yggdrasil.data.Schema` here so the rest of the package never
hand-rolls the codec.

Coverage
--------

Primitives (``string``, ``long``, ``integer``, ``short``, ``byte``,
``double``, ``float``, ``boolean``, ``binary``, ``date``,
``timestamp``, ``timestamp_ntz``, fixed-precision ``decimal(p,s)``)
plus the three complex shapes (``struct``, ``array``, ``map``) cover
everything a portable Delta table holds. Unknown Spark-side types
fall back to ``string`` on the way in (best-effort load) and
:class:`pa.binary` on the way out — a deliberate "don't crash" choice;
pinning the unmapped shape to ``OBJECT`` would have the same effect
when the Arrow type later round-trips through the yggdrasil
:class:`Field`.

The :func:`schema_to_spark_json` / :func:`spark_json_to_schema` pair
operate on :class:`yggdrasil.data.Schema` — they delegate to the
Arrow-shaped helpers via ``Schema.to_arrow_schema`` /
``Schema.from_arrow_schema`` so callers that already have a yggdrasil
schema don't have to round-trip through Arrow themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping

import pyarrow as pa

from yggdrasil.pickle import json as ygg_json

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema


__all__ = [
    "arrow_schema_to_spark_json",
    "spark_json_to_arrow_schema",
    "schema_to_spark_json",
    "spark_json_to_schema",
]


# ---------------------------------------------------------------------------
# Arrow → Spark JSON
# ---------------------------------------------------------------------------


def arrow_schema_to_spark_json(schema: pa.Schema) -> str:
    """Render a pyarrow schema as a Delta-compatible schemaString."""
    fields = [_arrow_field_to_spark(f) for f in schema]
    payload: Dict[str, Any] = {"type": "struct", "fields": fields}
    # Compact separators match the spec's ``schemaString`` shape on
    # disk — Delta tooling parses both, but the canonical writer
    # output is always whitespace-free.
    return ygg_json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        to_bytes=False,
    )


def _arrow_field_to_spark(field: pa.Field) -> Dict[str, Any]:
    metadata: Dict[str, str] = {}
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


#: DDL → Delta schemaString primitive name. ``DataType.to_spark_name``
#: produces the uppercase SQL DDL form (``BIGINT`` / ``INT`` / ``SHORT``
#: / ``DECIMAL(p, s)`` / …); Delta's wire format is the lowercase
#: ``typeName`` Spark uses inside ``StructType.json()``. Most heads
#: are just lowercase, but the integer family renames (``INT`` →
#: ``integer``, ``BIGINT`` → ``long``) so they need an explicit slot.
_DDL_HEAD_TO_DELTA = {
    "INT": "integer",
    "BIGINT": "long",
}


def _arrow_type_to_spark(t: pa.DataType) -> Any:
    # Complex types stay JSON-shaped — they recurse through
    # :func:`_arrow_type_to_spark` on their child types, so the
    # canonical-name routing below kicks in for the leaves too.
    if (
        pa.types.is_list(t)
        or pa.types.is_large_list(t)
        or pa.types.is_fixed_size_list(t)
    ):
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

    # Primitives — route through the canonical surface. ``to_spark_name``
    # returns the SQL DDL form (``BIGINT`` / ``DECIMAL(10, 2)`` / …);
    # Delta's wire vocabulary differs from DDL in two slots
    # (``INT`` / ``BIGINT``), so :data:`_DDL_HEAD_TO_DELTA` patches
    # those before the lowercase pass.
    #
    # Unsigned integers go through ``as_spark`` first so we *don't*
    # widen them — ``uint8`` lands as ``byte`` (int8), ``uint64`` as
    # ``long`` (int64). Values that don't fit the signed range round-
    # trip via two's-complement when the parquet-write path casts
    # ``uint`` → same-width signed (``DeltaFolder._coerce_uints``); the
    # Delta schema and the parquet payload then agree on width and
    # signedness. The default ``as_spark`` widening (which would land
    # ``uint64`` at ``DECIMAL(20, 0)``) costs storage and breaks the
    # bit-identical round-trip property — keep ``self`` for ints by
    # the as_spark override on :class:`IntegerType`.
    from yggdrasil.data.types.base import DataType

    try:
        ddl = DataType.from_arrow_type(t).as_spark().to_spark_name()
    except Exception:
        # Unknown / unsupported → binary fallback. The intent is
        # "round-trips, doesn't crash"; a richer mapping can land
        # later if a real caller hits this.
        return "binary"

    paren = ddl.find("(")
    if paren == -1:
        head, tail = ddl, ""
    else:
        head, tail = ddl[:paren], ddl[paren:]
    base = _DDL_HEAD_TO_DELTA.get(head, head.lower())
    if not tail:
        return base
    # Decimal is the only parametric primitive Delta carries. Strip
    # whitespace + lowercase the tail so ``DECIMAL(10, 2)`` ends up
    # as ``decimal(10,2)`` — matches Spark's ``simpleString()`` and
    # the schemaString shape every Delta engine writes.
    return f"{base}{tail.replace(' ', '').lower()}"


# ---------------------------------------------------------------------------
# Spark JSON → Arrow
# ---------------------------------------------------------------------------


def spark_json_to_arrow_schema(schema_string: str) -> pa.Schema:
    """Parse a Delta schemaString back into a :class:`pa.Schema`."""
    if not schema_string:
        return pa.schema([])
    payload = ygg_json.loads(schema_string)
    if payload.get("type") != "struct":
        raise ValueError(
            f"Expected a struct-typed schemaString; got {payload.get('type')!r}. "
            f"Delta tables always store the top-level schema as a struct — "
            f"check that the metadata action's schemaString is well-formed."
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
    """Resolve a Spark/Delta primitive name to a pyarrow type.

    Delegates to :meth:`DataType.from_str` — the canonical parser
    already knows every Spark primitive (``long`` / ``short`` / ``byte``
    / ``timestamp_ntz`` / ``decimal(p,s)`` / …). The two Delta-spec
    quirks the canonical parser doesn't speak natively are handled
    inline:

    - ``void`` — Spark's name for the null type. The canonical parser
      maps it to :class:`ObjectType` (large_binary), which is right
      for free-form Python objects but not for Delta.
    - Unknown / unmapped tokens fall back to ``string`` (the existing
      "round-trips, doesn't crash" contract of this module).
    """
    from yggdrasil.data.types.base import DataType

    n = name.strip().lower()
    if n == "void":
        return pa.null()
    try:
        return DataType.from_str(n).to_arrow()
    except Exception:
        return pa.string()


# ---------------------------------------------------------------------------
# yggdrasil.data.Schema bridges
# ---------------------------------------------------------------------------


def schema_to_spark_json(schema: "Schema") -> str:
    """Serialize a :class:`yggdrasil.data.Schema` to Delta's schemaString.

    Goes through Arrow because the Spark JSON shape lines up tightly
    with the Arrow type system; round-tripping via the yggdrasil
    :class:`DataType` would lose Delta-specific intent (``timestamp``
    vs ``timestamp_ntz``, decimal precision/scale) we already emit
    correctly from Arrow. Field-level metadata survives because
    :meth:`Schema.to_arrow_schema` carries it into the Arrow field's
    metadata dict.
    """
    arrow_schema = schema.to_arrow_schema()
    return arrow_schema_to_spark_json(arrow_schema)


def spark_json_to_schema(schema_string: str) -> "Schema":
    """Parse a Delta schemaString into a :class:`yggdrasil.data.Schema`.

    Lazy import because :mod:`yggdrasil.data.schema` pulls a chunk of
    the typesystem with it, and the snapshot read path doesn't always
    need a yggdrasil schema — only callers that asked for one.
    """
    from yggdrasil.data.schema import Schema

    arrow_schema = spark_json_to_arrow_schema(schema_string)
    return Schema.from_arrow(arrow_schema)
