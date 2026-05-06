"""BSON ↔ Arrow / yggdrasil :class:`DataType` mapping.

MongoDB has no first-class column schema — every document is free-shape
BSON. To plug it into the yggdrasil tabular abstractions we go through
two conversion layers:

* :func:`bson_to_arrow_type` / :func:`arrow_to_bson_type_name` map a
  single BSON type code to a :class:`pyarrow.DataType`. The "BSON
  type" is identified either by its canonical alias (``"objectId"``,
  ``"decimal"``, ``"date"``) or by the integer BSON type byte (1 for
  ``double``, 7 for ``objectId``, …).
* :func:`infer_arrow_schema_from_documents` walks a sample of
  documents and unifies their per-key types into a single
  :class:`pyarrow.Schema` — the canonical recipe used by every
  fallback (non-pymongoarrow) read path.

For the Arrow-native path the schema *is* the contract and pymongoarrow
returns Arrow types directly; this module then wraps them into
yggdrasil :class:`Field` / :class:`Schema` so the rest of the
yggdrasil surface (cast registry, options, typed records) lights up.

What gets preserved
-------------------

* ``ObjectId`` → :class:`pa.binary(12)` with a ``"bson:type=objectId"``
  metadata tag, so :func:`arrow_field_to_bson_extra` can round-trip
  back to ``ObjectId`` on writes.
* ``Decimal128`` → :class:`pa.decimal128(38, scale)` (we keep the full
  precision; scale is detected from the sampled value, falling back to
  scale=10 when no values are seen).
* ``Binary`` → :class:`pa.binary()` with the BSON subtype stashed in
  field metadata under ``"bson:subtype"``.
* ``Date`` (BSON 9, the millisecond UTC instant) →
  :class:`pa.timestamp("ms", tz="UTC")`.
* ``Document`` (embedded subdoc) → :class:`pa.struct(...)` recursively.
* ``Array`` → :class:`pa.list_(...)`.
* ``Null`` / missing key → :class:`pa.null()` (only when no other type
  is observed for the key — anything else outranks null).
* ``MinKey`` / ``MaxKey`` / ``Undefined`` / ``Symbol`` / ``Code`` /
  ``CodeWScope`` / ``DBRef`` / ``Timestamp`` (the BSON internal one,
  not the wall-clock Date) → :class:`pa.string()` carrying the
  ``str(...)`` of the value, with the original BSON type stashed in
  metadata. These are rare in user data and the loss is tagged.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Iterable, Mapping

import pyarrow as pa

from yggdrasil.data import Field
from yggdrasil.data.schema import Schema

__all__ = [
    "BSON_METADATA_KEY",
    "BSON_SUBTYPE_METADATA_KEY",
    "OBJECT_ID_BYTES",
    "bson_to_arrow_type",
    "arrow_to_bson_type_name",
    "arrow_field_to_bson_extra",
    "infer_arrow_schema_from_documents",
    "infer_schema_from_documents",
    "documents_to_arrow_table",
    "arrow_table_to_documents",
    "encode_value_for_bson",
    "decode_value_from_bson",
]


#: Field-metadata key that carries a BSON type alias the dtype can't
#: encode on its own (``objectId``, ``regex``, ``timestamp``, …).
BSON_METADATA_KEY = b"bson:type"

#: Field-metadata key for the BSON binary subtype (an integer 0-255 for
#: :class:`bson.Binary` columns).
BSON_SUBTYPE_METADATA_KEY = b"bson:subtype"

#: BSON ``ObjectId`` is exactly 12 bytes — encoded as a fixed-size
#: binary so consumers can index into it without per-row length checks.
OBJECT_ID_BYTES = 12


# ---------------------------------------------------------------------------
# BSON aliases
# ---------------------------------------------------------------------------

# Official MongoDB BSON type aliases — the strings you see in
# ``$type`` query operators. Maps both numeric codes and aliases.
# https://www.mongodb.com/docs/manual/reference/bson-types/
_BSON_ALIAS_BY_CODE: dict[int, str] = {
    1: "double",
    2: "string",
    3: "object",
    4: "array",
    5: "binData",
    6: "undefined",
    7: "objectId",
    8: "bool",
    9: "date",
    10: "null",
    11: "regex",
    12: "dbPointer",
    13: "javascript",
    14: "symbol",
    15: "javascriptWithScope",
    16: "int",
    17: "timestamp",
    18: "long",
    19: "decimal",
    -1: "minKey",
    127: "maxKey",
}


def _bson_alias(code_or_name: Any) -> str:
    if isinstance(code_or_name, str):
        return code_or_name
    if isinstance(code_or_name, int):
        return _BSON_ALIAS_BY_CODE.get(code_or_name, f"bson_{code_or_name}")
    return type(code_or_name).__name__


# ---------------------------------------------------------------------------
# BSON → Arrow
# ---------------------------------------------------------------------------


def bson_to_arrow_type(bson_type: Any) -> pa.DataType:
    """Map a BSON type alias (or numeric code) to a :class:`pa.DataType`.

    Accepts the canonical string aliases (``"double"``, ``"objectId"``,
    ``"decimal"``) and the numeric BSON type bytes (1, 7, 19). Unknown
    inputs fall back to :class:`pa.string` so callers can still
    materialise the rows; the original alias is recoverable from the
    field metadata via :data:`BSON_METADATA_KEY`.
    """
    alias = _bson_alias(bson_type)
    handler = _BSON_TO_ARROW.get(alias)
    if handler is not None:
        return handler()
    return pa.string()


def _objectid_type() -> pa.DataType:
    return pa.binary(OBJECT_ID_BYTES)


def _decimal128_type() -> pa.DataType:
    # BSON Decimal128 is a 34-digit IEEE 754-2008 decimal. Arrow caps
    # at decimal128(38, scale); we pick scale=10 as a sensible default
    # when no concrete value has been observed. ``infer_*`` adjusts
    # scale per-column.
    return pa.decimal128(38, 10)


_BSON_TO_ARROW: dict[str, Any] = {
    "double": pa.float64,
    "string": pa.string,
    "object": lambda: pa.struct([]),
    "array": lambda: pa.list_(pa.null()),
    "binData": pa.binary,
    "undefined": pa.null,
    "objectId": _objectid_type,
    "bool": pa.bool_,
    "date": lambda: pa.timestamp("ms", tz="UTC"),
    "null": pa.null,
    "regex": pa.string,
    "dbPointer": pa.string,
    "javascript": pa.string,
    "symbol": pa.string,
    "javascriptWithScope": pa.string,
    "int": pa.int32,
    "timestamp": lambda: pa.struct(
        [pa.field("t", pa.uint32()), pa.field("i", pa.uint32())]
    ),
    "long": pa.int64,
    "decimal": _decimal128_type,
    "minKey": pa.string,
    "maxKey": pa.string,
}


# ---------------------------------------------------------------------------
# Arrow → BSON aliases
# ---------------------------------------------------------------------------


def arrow_to_bson_type_name(dtype: pa.DataType) -> str:
    """Best-effort BSON alias for an Arrow type — used in ``$type`` filters."""
    if pa.types.is_boolean(dtype):
        return "bool"
    if pa.types.is_integer(dtype):
        return "long" if pa.types.is_int64(dtype) or pa.types.is_uint32(dtype) or pa.types.is_uint64(dtype) else "int"
    if pa.types.is_floating(dtype):
        return "double"
    if pa.types.is_decimal(dtype):
        return "decimal"
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "string"
    if pa.types.is_binary(dtype) or pa.types.is_fixed_size_binary(dtype) or pa.types.is_large_binary(dtype):
        return "binData"
    if pa.types.is_date(dtype) or pa.types.is_timestamp(dtype):
        return "date"
    if pa.types.is_struct(dtype):
        return "object"
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        return "array"
    if pa.types.is_null(dtype):
        return "null"
    return "string"


def arrow_field_to_bson_extra(field: pa.Field | Field) -> dict[bytes, bytes]:
    """Render the BSON-specific metadata for a yggdrasil/Arrow field.

    Used to stamp ``bson:type`` / ``bson:subtype`` onto fields that
    carry information the Arrow type alone can't express (``objectId``,
    ``binary subtype``, ``regex``…).
    """
    if isinstance(field, Field):
        arrow_field = field.to_arrow_field()
    else:
        arrow_field = field
    out: dict[bytes, bytes] = {}
    md = arrow_field.metadata or {}
    for key in (BSON_METADATA_KEY, BSON_SUBTYPE_METADATA_KEY):
        if key in md:
            out[key] = md[key]
    return out


# ---------------------------------------------------------------------------
# Document inference
# ---------------------------------------------------------------------------


_TYPE_RANK: dict[str, int] = {
    "null": 0,
    "bool": 10,
    "int": 20,
    "long": 21,
    "double": 22,
    "decimal": 23,
    "date": 30,
    "objectId": 31,
    "binData": 32,
    "string": 40,
    "array": 50,
    "object": 51,
}


def _classify(value: Any) -> tuple[str, Any]:
    """Return ``(bson_alias, value)`` — the discriminator + raw value."""
    if value is None:
        return "null", value
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, int):
        # MongoDB distinguishes 32-bit ``int`` from 64-bit ``long`` —
        # we widen anything out of int32 range automatically.
        if -(2 ** 31) <= value < 2 ** 31:
            return "int", value
        return "long", value
    if isinstance(value, float):
        return "double", value
    if isinstance(value, dt.datetime):
        return "date", value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "binData", bytes(value)
    if isinstance(value, str):
        return "string", value
    if isinstance(value, dict):
        return "object", value
    if isinstance(value, (list, tuple)):
        return "array", list(value)
    cls_name = type(value).__name__
    if cls_name == "ObjectId":
        return "objectId", value
    if cls_name == "Decimal128":
        return "decimal", value
    if cls_name == "Binary":
        return "binData", value
    if cls_name == "Regex":
        return "regex", value
    if cls_name == "Timestamp":
        return "timestamp", value
    if cls_name == "Code":
        return "javascript", value
    return "string", value


def _promote(left: str, right: str) -> str:
    """Pick the more permissive of two BSON type aliases."""
    if left == right:
        return left
    if left == "null":
        return right
    if right == "null":
        return left
    rl = _TYPE_RANK.get(left, 99)
    rr = _TYPE_RANK.get(right, 99)
    if rl == rr:
        return left
    # Numeric promotion ladder.
    if {left, right} <= {"int", "long", "double", "decimal"}:
        return max((left, right), key=lambda a: _TYPE_RANK[a])
    # Mixing scalar with object / array → fall back to string for
    # safety; downstream callers can override with a typed schema.
    if rl < rr:
        return right
    return left


def _decimal_scale(value: Any) -> int:
    """Best-effort scale extraction from a :class:`bson.Decimal128`."""
    text = str(value)
    if "." in text:
        scale = len(text.split(".", 1)[1].split("E", 1)[0].rstrip("0"))
        return min(max(scale, 0), 38)
    return 0


def _infer_arrow_type(
    alias: str,
    samples: list[Any],
) -> pa.DataType:
    """Materialise an Arrow type for a column observed as ``alias``.

    For ``object`` and ``array`` we recurse on the samples; for
    ``decimal`` we compute the maximum scale across samples.
    """
    if alias == "object":
        nested: dict[str, list[Any]] = {}
        observed_aliases: dict[str, str] = {}
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            for key, val in sample.items():
                a, _ = _classify(val)
                observed = observed_aliases.get(key)
                observed_aliases[key] = a if observed is None else _promote(observed, a)
                nested.setdefault(key, []).append(val)
        fields = []
        for key, sub_samples in nested.items():
            sub_alias = observed_aliases[key]
            sub_type = _infer_arrow_type(sub_alias, sub_samples)
            md = {BSON_METADATA_KEY: sub_alias.encode("utf-8")}
            fields.append(pa.field(key, sub_type, nullable=True, metadata=md))
        return pa.struct(fields)
    if alias == "array":
        flat: list[Any] = []
        observed: str | None = None
        for sample in samples:
            if not isinstance(sample, (list, tuple)):
                continue
            for item in sample:
                a, v = _classify(item)
                observed = a if observed is None else _promote(observed, a)
                flat.append(v)
        if observed is None:
            return pa.list_(pa.null())
        return pa.list_(_infer_arrow_type(observed, flat))
    if alias == "decimal":
        scale = max((_decimal_scale(v) for v in samples if v is not None), default=10)
        return pa.decimal128(38, scale)
    return bson_to_arrow_type(alias)


def infer_arrow_schema_from_documents(
    documents: Iterable[Mapping[str, Any]],
    *,
    sample_size: int | None = None,
) -> pa.Schema:
    """Infer a :class:`pa.Schema` from a sample of MongoDB documents.

    Walks up to ``sample_size`` documents (``None`` = all), unioning
    every key's observed BSON types via :data:`_TYPE_RANK`. Each
    output field carries the resolved BSON alias under
    :data:`BSON_METADATA_KEY` so downstream code that needs the
    original BSON discrimination (e.g. round-tripping back to a write)
    has it on hand.

    Pure-Python — no pymongo / bson import required, so this also
    works for unit tests against in-memory document lists.
    """
    aliases: dict[str, str] = {}
    samples: dict[str, list[Any]] = {}
    seen = 0
    for doc in documents:
        if not isinstance(doc, Mapping):
            continue
        for key, value in doc.items():
            alias, normalized = _classify(value)
            existing = aliases.get(key)
            aliases[key] = alias if existing is None else _promote(existing, alias)
            samples.setdefault(key, []).append(normalized)
        seen += 1
        if sample_size is not None and seen >= sample_size:
            break

    fields: list[pa.Field] = []
    for key, alias in aliases.items():
        dtype = _infer_arrow_type(alias, samples[key])
        md = {BSON_METADATA_KEY: alias.encode("utf-8")}
        fields.append(pa.field(key, dtype, nullable=True, metadata=md))
    return pa.schema(fields)


def infer_schema_from_documents(
    documents: Iterable[Mapping[str, Any]],
    *,
    sample_size: int | None = None,
) -> Schema:
    """Same as :func:`infer_arrow_schema_from_documents`, lifted to yggdrasil :class:`Schema`."""
    return Schema.from_arrow(infer_arrow_schema_from_documents(documents, sample_size=sample_size))


# ---------------------------------------------------------------------------
# Value codec — Python <-> BSON
# ---------------------------------------------------------------------------


def encode_value_for_bson(value: Any, *, target_type: pa.DataType | None = None) -> Any:
    """Coerce a Python value into something pymongo can write directly.

    Used by the row-write fallback path. The Arrow-native path
    (pymongoarrow) handles encoding internally and skips this helper.

    * ``bytes`` of length 12 with ``target_type=binary(12)`` → ObjectId.
    * decimal-typed numerics → :class:`bson.Decimal128`.
    * ``date`` / ``datetime`` are passed through (pymongo handles
      ``datetime.datetime`` natively as BSON Date).
    """
    if value is None:
        return None
    cls_name = type(value).__name__
    # Already-BSON values pass through.
    if cls_name in {"ObjectId", "Decimal128", "Binary", "Regex", "Code", "Timestamp", "DBRef"}:
        return value

    if target_type is not None:
        if pa.types.is_fixed_size_binary(target_type) and target_type.byte_width == OBJECT_ID_BYTES:
            from .lib import bson_module
            return bson_module().ObjectId(value)
        if pa.types.is_decimal(target_type):
            from .lib import bson_module
            return bson_module().Decimal128(str(value))

    return value


def decode_value_from_bson(value: Any) -> Any:
    """Convert a BSON-typed value to a yggdrasil-friendly Python value.

    * :class:`bson.ObjectId` → 12-byte ``bytes``.
    * :class:`bson.Decimal128` → :class:`decimal.Decimal`.
    * :class:`bson.Binary` → ``bytes``.
    * :class:`bson.Regex` → ``str(pattern)``.
    * Everything else passes through unchanged.
    """
    if value is None:
        return None
    cls_name = type(value).__name__
    if cls_name == "ObjectId":
        return bytes(value.binary)
    if cls_name == "Decimal128":
        return value.to_decimal()
    if cls_name == "Binary":
        return bytes(value)
    if cls_name == "Regex":
        return getattr(value, "pattern", str(value))
    if cls_name == "Code":
        return str(value)
    if cls_name == "Timestamp":
        return {"t": int(getattr(value, "time", 0)), "i": int(getattr(value, "inc", 0))}
    if isinstance(value, list):
        return [decode_value_from_bson(v) for v in value]
    if isinstance(value, dict):
        return {k: decode_value_from_bson(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Documents <-> Arrow tables (fallback path)
# ---------------------------------------------------------------------------


def documents_to_arrow_table(
    documents: Iterable[Mapping[str, Any]],
    *,
    schema: pa.Schema | None = None,
    sample_size: int | None = None,
) -> pa.Table:
    """Lift an iterable of MongoDB documents into a :class:`pa.Table`.

    Used by the row-fallback path (no pymongoarrow available). Two
    phases: materialise the iterable into a list (we need it twice,
    once to infer the schema if absent and once to project rows), then
    project each row through :func:`decode_value_from_bson` so BSON-
    only values round-trip into Arrow-friendly Python.
    """
    docs = list(documents)
    if schema is None:
        schema = infer_arrow_schema_from_documents(docs, sample_size=sample_size)
    if not docs:
        return schema.empty_table()
    rows: list[dict[str, Any]] = []
    keys = [field.name for field in schema]
    for doc in docs:
        row: dict[str, Any] = {}
        for key in keys:
            row[key] = decode_value_from_bson(doc.get(key)) if isinstance(doc, Mapping) else None
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=schema)


def arrow_table_to_documents(
    table: pa.Table,
    *,
    encode_object_ids: bool = True,
) -> list[dict[str, Any]]:
    """Materialise a :class:`pa.Table` as MongoDB-ready documents.

    For columns annotated with :data:`BSON_METADATA_KEY` we re-encode
    the Python value back into the matching BSON wrapper (ObjectId,
    Decimal128, Binary). Plain types pass through.
    """
    field_types: list[pa.DataType] = [field.type for field in table.schema]
    field_metadata: list[Mapping[bytes, bytes] | None] = [
        field.metadata for field in table.schema
    ]
    out: list[dict[str, Any]] = []
    for row in table.to_pylist():
        encoded: dict[str, Any] = {}
        for idx, (name, value) in enumerate(row.items()):
            md = field_metadata[idx] or {}
            alias = md.get(BSON_METADATA_KEY) if md else None
            if value is None:
                encoded[name] = None
                continue
            if encode_object_ids and alias == b"objectId":
                from .lib import bson_module
                encoded[name] = bson_module().ObjectId(value)
                continue
            if alias == b"decimal":
                from .lib import bson_module
                encoded[name] = bson_module().Decimal128(str(value))
                continue
            if alias == b"binData":
                from .lib import bson_module
                subtype = int.from_bytes(md.get(BSON_SUBTYPE_METADATA_KEY, b"\x00"), "big") if md else 0
                encoded[name] = bson_module().Binary(value, subtype)
                continue
            encoded[name] = encode_value_for_bson(value, target_type=field_types[idx])
        out.append(encoded)
    return out
