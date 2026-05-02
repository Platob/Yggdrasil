"""Bridge between yggdrasil :class:`Schema` and Delta's schemaString.

Delta serializes a table's schema as Spark's StructType JSON. The
shape::

    {
        "type": "struct",
        "fields": [
            {"name": "id", "type": "long", "nullable": false, "metadata": {}},
            {"name": "ts", "type": "timestamp", "nullable": true, "metadata": {}},
            {"name": "tags", "type": {"type": "array", "elementType": "string", "containsNull": true}, ...}
        ]
    }

We rely on the dtype layer (your ``DataType`` hierarchy) to render
itself in this JSON form via a ``to_spark_json`` method. The Spark
catalyst format and Delta's are identical here — Spark is the
reference. If a dtype doesn't implement ``to_spark_json``, we raise
rather than guess; the alternative (a hand-rolled fallback table) is
exactly the kind of plausibly-wrong code we don't want.

Round-trip discipline
---------------------

Read direction (Delta JSON → Schema): we call
:meth:`Schema.from_any` on the parsed JSON dict. Your converter
registry already maps Spark JSON dicts to Schema; this is a single
call here on purpose, so that any future improvements to the dtype
parser benefit Delta automatically.

Write direction (Schema → Delta JSON): we walk fields and call
``field.dtype.to_spark_json()`` per field. The result is a JSON
value (str for primitives, dict for nested types) embedded in a
StructField wrapper.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from yggdrasil.data.schema import Field, Schema


__all__ = [
    "delta_schema_string_to_schema",
    "schema_to_delta_schema_string",
]


def delta_schema_string_to_schema(schema_string: str) -> Schema:
    """Parse Delta's ``Metadata.schemaString`` into a :class:`Schema`.

    The string is JSON of a Spark StructType. We delegate to
    :meth:`Schema.from_any`, which routes through the converter
    registry. Failures bubble up as-is — the caller (replay) treats
    them as table corruption.
    """
    parsed = json.loads(schema_string)
    return Schema.from_any(parsed)


def schema_to_delta_schema_string(schema: Schema) -> str:
    """Render a :class:`Schema` as a Delta-compatible schemaString.

    Walks ``schema.fields`` (excluding constraints — Delta has its
    own constraint surface via ``checkConstraints`` writer feature)
    and emits StructField JSON for each. Field-level metadata is
    passed through as-is when it's a JSON-compatible dict; otherwise
    we emit ``{}`` to keep the wire format strict.

    Failure mode: if any dtype lacks ``to_spark_json``, we raise
    with a pointer at the offending field. Means a Delta write
    fails fast on a non-portable type rather than silently writing
    a placeholder that a downstream Spark reader would reject.
    """
    spark_struct = {
        "type": "struct",
        "fields": [_field_to_spark_json(f) for f in schema.fields],
    }
    return json.dumps(spark_struct)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _field_to_spark_json(field: Field) -> Mapping[str, Any]:
    """Render a single :class:`Field` as Spark StructField JSON."""
    dtype = field.dtype
    render = getattr(dtype, "to_spark_json", None)
    if not callable(render):
        raise TypeError(
            f"Cannot serialize field {field.name!r} for Delta: its "
            f"dtype {type(dtype).__name__} has no to_spark_json() "
            "method. Implement DataType.to_spark_json on this type, "
            "or change the field's dtype before writing to Delta."
        )

    try:
        type_value = render()
    except Exception as exc:  # pragma: no cover - exercised via tests
        raise ValueError(
            f"Field {field.name!r}: dtype {type(dtype).__name__} "
            f"failed to render as Spark JSON: {exc!r}."
        ) from exc

    metadata = _field_metadata_to_spark_json(field)

    return {
        "name": field.name,
        "type": type_value,
        "nullable": bool(field.nullable),
        "metadata": metadata,
    }


def _field_metadata_to_spark_json(field: Field) -> Mapping[str, Any]:
    """Coerce a Field's metadata into a JSON-safe dict.

    Spark's StructField metadata is a JSON object with string keys
    and JSON-typed values. yggdrasil's :class:`Field` carries
    bytes-keyed metadata (the same convention as Arrow), so we
    decode keys to strings and JSON-decode any byte values that
    look like JSON. Bytes that aren't valid UTF-8 are dropped — we
    can't round-trip them through a JSON wire format.
    """
    raw = getattr(field, "metadata", None)
    if not raw:
        return {}

    out: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            str_key = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
        except UnicodeDecodeError:
            continue

        out[str_key] = _coerce_metadata_value(value)
    return out


def _coerce_metadata_value(value: Any) -> Any:
    """Coerce a single metadata value to JSON-safe form."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError:
            return ""
        # If it parses as JSON, prefer the parsed form so dicts /
        # lists round-trip cleanly.
        try:
            return json.loads(decoded)
        except (ValueError, json.JSONDecodeError):
            return decoded
    if isinstance(value, (list, tuple)):
        return [_coerce_metadata_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_metadata_value(v) for k, v in value.items()}
    # Fallback: stringify. Lossy but valid JSON.
    return str(value)
