"""JSON-string cast helpers shared by nested types.

Parsing from string/binary to nested structures goes through vectorised
JSON decoding so we avoid per-row Python loops.  The reverse direction
(nested → string/binary) goes through vectorised JSON encoding.
"""
from __future__ import annotations

import datetime as _dt
import decimal as _decimal
import json
import re
from typing import TYPE_CHECKING, Any

import orjson
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as paj

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.support import get_polars, get_spark_sql
from yggdrasil.exceptions import CastError

if TYPE_CHECKING:
    from yggdrasil.data.options import CastOptions


# PyArrow's JSON parser emits errors like ``JSON parse error: ... in row 5``.
# We extract the row index to surface a row-pointing message without
# re-walking the rows in Python.
_PYARROW_JSON_ROW_RE = re.compile(r"in row (\d+)\b")

# Reused string scalars for the NDJSON wrapper.  Allocated once at module
# import so each cast call doesn't rebuild them.
_NDJSON_PREFIX = pa.scalar('{"v":', type=pa.string())
_NDJSON_SUFFIX = pa.scalar("}", type=pa.string())
_NDJSON_EMPTY_SEP = pa.scalar("", type=pa.string())
_NDJSON_LINE_SEP = pa.scalar("\n", type=pa.string())
_NDJSON_NULL = pa.scalar("null", type=pa.string())
_NDJSON_COMMA_SEP = pa.scalar(",", type=pa.string())

# Scalars for the empty-source pre-cleanup. ``_NULL_STRING_SCALAR`` is a
# typed null so ``pc.if_else`` can splice it directly into a string array
# without an intermediate cast.
_EMPTY_STRING_SCALAR = pa.scalar("", type=pa.string())
_NULL_STRING_SCALAR = pa.scalar(None, type=pa.string())


__all__ = [
    "is_json_string_source",
    "is_json_string_target",
    "is_json_nested_source",
    "cast_arrow_json_string_array",
    "cast_polars_json_string_expr",
    "cast_polars_json_string_series",
    "cast_spark_json_string_column",
    "cast_arrow_json_encode_array",
    "cast_polars_json_encode_series",
    "cast_spark_json_encode_column",
]


# Decoding direction (string/binary -> nested) only fires for sources
# that explicitly declare JSON intent.  Plain STRING / BINARY columns
# are NOT auto-parsed: callers who want JSON decode must surface the
# column as :class:`SJsonType` / :class:`BJsonType` (or pass that as
# ``target_field``'s source). Removing the implicit STRING -> nested
# JSON parse avoids silently parsing arbitrary text and forces the
# intent to be visible in the schema.
_JSON_DECODE_SOURCE_TYPES = frozenset(
    {DataTypeId.SJSON, DataTypeId.BJSON}
)

# Encoding direction (nested -> string/binary) keeps the broader set:
# producing JSON text for a STRING / BINARY target is unambiguous (no
# competing semantics) and matches what most sinks expect.
_JSON_ENCODE_TARGET_TYPES = frozenset(
    {
        DataTypeId.STRING,
        DataTypeId.BINARY,
        DataTypeId.SJSON,
        DataTypeId.BJSON,
    }
)
_JSON_NESTED_TYPES = frozenset(
    {DataTypeId.ARRAY, DataTypeId.MAP, DataTypeId.STRUCT}
)


def is_json_string_source(source_type_id: DataTypeId) -> bool:
    return source_type_id in _JSON_DECODE_SOURCE_TYPES


def is_json_string_target(target_type_id: DataTypeId) -> bool:
    return target_type_id in _JSON_ENCODE_TARGET_TYPES


def is_json_nested_source(source_type_id: DataTypeId) -> bool:
    return source_type_id in _JSON_NESTED_TYPES


def _arrow_to_utf8(
    array: pa.Array | pa.ChunkedArray,
    memory_pool: pa.MemoryPool | None = None,
) -> pa.Array:
    if isinstance(array, pa.ChunkedArray):
        # ``combine_chunks`` copies every chunk into a fresh contiguous
        # buffer — needed for the NDJSON path downstream, but a waste
        # when the input is already a single chunk. Unwrap zero-copy
        # in that case.
        if array.num_chunks == 1:
            array = array.chunks[0]
        else:
            array = array.combine_chunks()

    src_type = array.type
    if (
        pa.types.is_binary(src_type)
        or pa.types.is_large_binary(src_type)
        or pa.types.is_binary_view(src_type)
    ):
        return pc.cast(array, pa.string(), memory_pool=memory_pool)

    if (
        pa.types.is_string(src_type)
        or pa.types.is_large_string(src_type)
        or pa.types.is_string_view(src_type)
    ):
        if not pa.types.is_string(src_type):
            return pc.cast(array, pa.string(), memory_pool=memory_pool)
        return array

    raise pa.ArrowInvalid(
        f"JSON cast expects a string/binary source, got {src_type!r}"
    )


def _null_out_empty(
    array: pa.Array,
    memory_pool: pa.MemoryPool | None = None,
) -> pa.Array:
    """Replace empty / whitespace-only strings with nulls.

    The NDJSON / json.loads paths downstream would otherwise reject
    these rows with ``invalid JSON`` errors. Treating them as null
    pre-cast saves a guaranteed-fail parse and keeps strict mode from
    blowing up on rows that simply have no payload — same intent as
    a missing cell carrying ``null`` natively.

    Whitespace-only is included because callers exporting CSV / Excel
    / Power Query often surface "no value" as ``" "`` rather than the
    empty string; both shapes mean the same thing here.
    """
    # ``utf8_trim_whitespace`` is a zero-copy view-trim in the C++ kernel.
    trimmed = pc.utf8_trim_whitespace(array, memory_pool=memory_pool)
    is_empty = pc.equal(trimmed, _EMPTY_STRING_SCALAR)
    return pc.if_else(is_empty, _NULL_STRING_SCALAR, array, memory_pool=memory_pool)


def cast_arrow_json_string_array(
    array: pa.Array | pa.ChunkedArray,
    options: "CastOptions",
) -> pa.Array:
    """Parse a string/binary Arrow array as JSON into the target arrow type.

    The hot path stays inside the C++ runtime: rows are wrapped as
    ``{"v":<row>}`` via Arrow compute, joined with newlines into a
    single NDJSON buffer, and decoded in one ``pyarrow.json.read_json``
    call against an ``explicit_schema`` that names the target type.

    Map targets — ``pa.map_(...)`` — bypass pyarrow.json (which cannot
    emit ``map`` arrays) and route through a single ``orjson.loads``
    call on the whole NDJSON-as-list blob built via pyarrow.compute;
    no per-row Python loop runs in either branch.

    Both strict (``options.safe=True``) and permissive
    (``options.safe=False``) modes share the same vectorised path —
    pyarrow.json has no "skip bad rows" option, so a malformed row
    fails the batch under either mode. Pre-cleanup turns empty /
    whitespace-only rows into nulls (see :func:`_null_out_empty`) so
    the common "no payload" shape decodes cleanly; genuinely invalid
    JSON should be filtered upstream.
    """
    target_field = options.target_field
    if target_field is None:
        return array

    target_arrow_type = target_field.dtype.to_arrow()
    memory_pool = options.arrow_memory_pool

    normalized = _arrow_to_utf8(array, memory_pool=memory_pool)
    # Empty / whitespace-only rows would otherwise fail the JSON parser
    # with ``invalid JSON`` — coerce them to null up front so strict mode
    # treats them the same as a natively-null source row.
    normalized = _null_out_empty(normalized, memory_pool=memory_pool)

    if len(normalized) == 0:
        return pa.array([], type=target_arrow_type)

    if _pyarrow_json_supports_target(target_arrow_type):
        try:
            return _vectorized_parse_json(
                normalized, target_arrow_type, memory_pool=memory_pool,
            )
        except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            if options.safe:
                raise _enrich_pyarrow_json_error(
                    exc, normalized, options=options
                ) from exc
            # Permissive: null out bad rows per row so the rest of the
            # batch lands. Mirrors polars' ``map_elements`` permissive
            # branch — strict callers stay on the vectorised path above.
            return _parse_per_row_permissive(
                normalized, target_arrow_type, memory_pool=memory_pool,
            )

    # Map targets (and anything else pyarrow.json cannot emit) take the
    # orjson-blob route — one C-level decode per batch, no per-row loop
    # in the happy path.
    try:
        return _parse_via_orjson_blob(
            normalized, target_arrow_type, memory_pool=memory_pool,
        )
    except (orjson.JSONDecodeError, ValueError, TypeError, pa.ArrowInvalid) as exc:
        if options.safe:
            raise CastError(
                f"failed to decode JSON batch: {exc}",
                source_field=options.source_field,
                target_field=target_field,
                original=exc,
            ) from exc
        # Permissive map path: per-row null-out fallback.
        return _parse_per_row_permissive(
            normalized, target_arrow_type, memory_pool=memory_pool,
        )


def _pyarrow_json_supports_target(t: pa.DataType) -> bool:
    """Does ``pyarrow.json.read_json`` know how to materialise ``t``?

    PyArrow's reader handles primitives, structs, list/large_list and
    fixed-size lists, but not ``map``.  We recurse so a ``map`` buried
    inside a struct or list also routes to the Python fallback.
    """
    if pa.types.is_map(t):
        return False
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return _pyarrow_json_supports_target(t.value_type)
    if pa.types.is_fixed_size_list(t):
        return _pyarrow_json_supports_target(t.value_type)
    if pa.types.is_struct(t):
        return all(
            _pyarrow_json_supports_target(t.field(i).type)
            for i in range(t.num_fields)
        )
    return True


def _vectorized_parse_json(
    normalized: pa.Array,
    target_arrow_type: pa.DataType,
    memory_pool: pa.MemoryPool | None = None,
) -> pa.Array:
    """Decode every row in a single C++ NDJSON pass.

    Steps stay inside Arrow compute / C++ kernels:

    1. ``fill_null`` swaps null entries for the literal ``null``.
    2. ``replace_substring_regex`` collapses any embedded ``\\r\\n``
       (which would otherwise split NDJSON records mid-row).  Valid
       JSON strings cannot legally contain literal newlines — only
       ``\\n`` escapes — so this only normalises framing whitespace.
    3. ``binary_join_element_wise`` wraps each row as ``{"v":<row>}``.
    4. A one-element ``list<string>`` plus ``binary_join`` glues all
       rows together with ``\\n`` separators into a single NDJSON
       buffer accessible as Arrow raw bytes.
    5. ``pyarrow.json.read_json`` parses the buffer against an
       ``explicit_schema`` declaring the target type for ``v``.

    The only Python crossings are the function calls themselves — no
    per-row iteration.
    """
    n = len(normalized)

    filled = pc.fill_null(normalized, _NDJSON_NULL)
    cleaned = pc.replace_substring_regex(filled, r"[\r\n]+", " ")

    wrapped = pc.binary_join_element_wise(
        _NDJSON_PREFIX, cleaned, _NDJSON_SUFFIX, _NDJSON_EMPTY_SEP,
    )

    list_arr = pa.ListArray.from_arrays(
        pa.array([0, n], pa.int32()),
        wrapped,
    )
    joined = pc.binary_join(list_arr, _NDJSON_LINE_SEP)
    # Underlying utf-8 bytes of the single concatenated scalar — no
    # Python string round-trip needed.
    buffer = joined.buffers()[2]

    schema = pa.schema([pa.field("v", target_arrow_type)])
    table = paj.read_json(
        pa.BufferReader(buffer),
        parse_options=paj.ParseOptions(
            explicit_schema=schema,
            unexpected_field_behavior="ignore",
        ),
        read_options=paj.ReadOptions(use_threads=True),
    )
    column = table.column("v")
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    return column


def _enrich_pyarrow_json_error(
    exc: BaseException,
    normalized: pa.Array,
    options: "CastOptions",
) -> CastError:
    """Promote a raw pyarrow.json error to a row-pointing :class:`CastError`.

    The pyarrow message already names the failing row (``... in row N``);
    we lift that index out via regex, pluck the offending source value
    with an Arrow scalar lookup (no Python loop) and append a truncated
    preview. Wrapped in :class:`CastError` so callers get the source +
    target field context — which would otherwise be lost on a bare
    :class:`pyarrow.ArrowInvalid`.
    """
    msg = str(exc)
    match = _PYARROW_JSON_ROW_RE.search(msg)
    target_field = options.target_field
    source_field = options.source_field
    if match is None:
        return CastError(
            f"invalid JSON: {msg}",
            source_field=source_field,
            target_field=target_field,
            original=exc,
        )

    idx = int(match.group(1))
    preview: Any = None
    if 0 <= idx < len(normalized):
        scalar = normalized[idx]
        if scalar.is_valid:
            value = scalar.as_py()
            preview = value if len(value) <= 120 else value[:117] + "..."

    return CastError(
        f"invalid JSON at row {idx}: {msg}. Value: {preview!r}",
        source_field=source_field,
        target_field=target_field,
        original=exc,
    )


def _parse_per_row_permissive(
    normalized: pa.Array,
    target_arrow_type: pa.DataType,
    memory_pool: pa.MemoryPool | None,
) -> pa.Array:
    """Permissive (``safe=False``) fallback: bad rows null out.

    Used only when the caller has explicitly opted out of strict
    parsing — the strict path stays on the fully-vectorised
    pyarrow.json / orjson-blob branches. Each cell that fails to parse
    via ``orjson.loads`` (or that the Arrow builder rejects) becomes
    null in the output so the rest of the batch lands.
    """
    pylist = normalized.to_pylist()
    decoded: list[Any] = [None] * len(pylist)
    for idx, blob in enumerate(pylist):
        if blob is None:
            continue
        try:
            decoded[idx] = orjson.loads(blob)
        except (orjson.JSONDecodeError, ValueError, TypeError):
            decoded[idx] = None

    try:
        return pa.array(decoded, type=target_arrow_type, memory_pool=memory_pool)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, ValueError):
        # A row decoded fine as JSON but doesn't fit the Arrow target
        # (e.g. wrong shape). Null those cells too.
        cells: list[Any] = [None] * len(decoded)
        for idx, value in enumerate(decoded):
            if value is None:
                continue
            try:
                pa.array([value], type=target_arrow_type, memory_pool=memory_pool)
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, ValueError):
                continue
            cells[idx] = value
        return pa.array(cells, type=target_arrow_type, memory_pool=memory_pool)


def _parse_via_orjson_blob(
    normalized: pa.Array,
    target_arrow_type: pa.DataType,
    memory_pool: pa.MemoryPool | None,
) -> pa.Array:
    """One-shot JSON parse for targets pyarrow.json cannot emit (maps).

    pyarrow.json has no MapArray builder, so we fall back to a single
    ``orjson.loads`` call — *not* per-row Python decoding. The whole
    batch is wrapped into ``"[<row1>,<row2>,...]"`` via pyarrow.compute
    (no Python row loop) and handed to orjson once; the resulting
    Python list is materialised back to Arrow via ``pa.array(...)``,
    which iterates internally in C++.

    Both strict and permissive callers share this path; a malformed
    row fails the whole batch (the empty-row pre-cleanup upstream
    already absorbs the common "no payload" shape).
    """
    n = len(normalized)
    if n == 0:
        return pa.array([], type=target_arrow_type, memory_pool=memory_pool)

    # Replace nulls with the literal ``null`` token so the joined blob
    # stays valid JSON without a Python crossing per row.
    filled = pc.fill_null(normalized, _NDJSON_NULL)
    list_arr = pa.ListArray.from_arrays(
        pa.array([0, n], pa.int32()),
        filled,
    )
    # One scalar string ``"r1,r2,...,rn"`` — read its raw utf-8 bytes
    # without round-tripping through a Python ``str``.
    joined = pc.binary_join(list_arr, _NDJSON_COMMA_SEP)
    inner_bytes = joined.buffers()[2].to_pybytes()
    blob = b"[" + inner_bytes + b"]"

    parsed = orjson.loads(blob)
    return pa.array(parsed, type=target_arrow_type, memory_pool=memory_pool)


def cast_polars_json_string_expr(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = get_polars()

    target_field = options.target_field
    if target_field is None:
        return expr

    target_polars_type = target_field.dtype.to_polars()

    # BJSON is the only binary-flavoured JSON source after the
    # decode-source narrowing (SJSON / BJSON); cast to String so
    # ``str.json_decode`` can read it.
    if options.source_field.dtype.type_id == DataTypeId.BJSON:
        expr = expr.cast(pl.String)

    # Coerce empty / whitespace-only rows to null so the parser sees
    # null where the source carried "no payload". Matches the Arrow
    # path's ``_null_out_empty`` pre-cleanup.
    expr = (
        pl.when(expr.is_null() | (expr.str.strip_chars().str.len_bytes() == 0))
        .then(None)
        .otherwise(expr)
    )

    if options.safe:
        # Strict: ``str.json_decode`` raises on any bad row — the
        # collection surfaces it as ``polars.exceptions.ComputeError``.
        return expr.str.json_decode(target_polars_type)

    # Permissive: polars' ``str.json_decode`` has no ``strict=False`` /
    # ``ignore_errors`` switch in 1.x, so route bad rows to null via a
    # one-shot ``map_elements`` with a per-row try/except. Acceptable
    # here because permissive is the explicit opt-out for batch
    # tolerance — strict callers stay on the fully-vectorised path
    # above.
    return expr.map_elements(
        _polars_json_decode_permissive,
        return_dtype=target_polars_type,
        skip_nulls=True,
    )


def _polars_json_decode_permissive(value: Any) -> Any:
    if value is None:
        return None
    try:
        return orjson.loads(value)
    except (orjson.JSONDecodeError, ValueError, TypeError):
        return None


def cast_polars_json_string_series(
    series: Any,
    options: "CastOptions",
) -> Any:
    pl = get_polars()
    expr = cast_polars_json_string_expr(pl.col(series.name), options).alias(
        options.target_field.name
    )
    return pl.DataFrame({series.name: series}).select(expr).to_series()


def cast_spark_json_string_column(
    column: Any,
    options: "CastOptions",
) -> Any:
    spark = get_spark_sql()
    F = spark.functions

    target_field = options.target_field
    if target_field is None:
        return column

    target_ddl = target_field.dtype.to_spark_name()

    # BJSON is the only binary-flavoured JSON source after the
    # decode-source narrowing (SJSON / BJSON); cast to string for
    # ``from_json``.
    if options.source_field.dtype.type_id == DataTypeId.BJSON:
        column = column.cast("string")

    # Empty / whitespace-only rows become null before ``from_json`` sees
    # them — FAILFAST mode would otherwise abort the query on rows that
    # carry no payload, matching the Arrow / Polars pre-cleanup.
    column = F.when(
        column.isNull() | (F.length(F.trim(column)) == 0),
        F.lit(None),
    ).otherwise(column)

    # Spark's ``from_json`` defaults to PERMISSIVE (bad records null
    # out). Strict (``options.safe=True``) opts into FAILFAST so the
    # behaviour matches the Arrow / Polars strict paths — a malformed
    # record fails the query instead of silently nulling.
    if options.safe:
        return F.from_json(column, target_ddl, {"mode": "FAILFAST"})
    return F.from_json(column, target_ddl)


# ---------------------------------------------------------------------------
# Encode: nested -> string/binary via JSON
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Fallback JSON encoder for values ``json.dumps`` doesn't natively handle.

    Kept deliberately narrow — Decimal/bytes/datetime/timedelta — because
    anything else slipping through suggests a schema mismatch the caller
    should see.
    """
    if isinstance(obj, _decimal.Decimal):
        return str(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj).decode("utf-8", errors="replace")
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, _dt.timedelta):
        return obj.total_seconds()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def _encode_row(value: Any) -> Any:
    if value is None:
        return None
    return json.dumps(value, default=_json_default, ensure_ascii=False)


def _encode_map_row(value: Any) -> Any:
    if value is None:
        return None
    return json.dumps(dict(value), default=_json_default, ensure_ascii=False)


def cast_arrow_json_encode_array(
    array: pa.Array | pa.ChunkedArray,
    options: "CastOptions",
) -> pa.Array:
    """Serialise a nested Arrow array to a string/binary array using JSON.

    ``json.dumps`` is invoked per row via ``map()`` so the iteration
    runs as a C-level CPython ``CALL_FUNCTION`` loop instead of Python
    bytecode — a list comprehension would be the same speed but more
    Python-frame overhead.
    """
    target_field = options.target_field
    if target_field is None:
        return array

    target_arrow_type = target_field.dtype.to_arrow()
    memory_pool = options.arrow_memory_pool

    if isinstance(array, pa.ChunkedArray):
        # Single-chunk inputs unwrap zero-copy — combine_chunks would
        # otherwise allocate + copy the full nested buffer for nothing.
        if array.num_chunks == 1:
            array = array.chunks[0]
        else:
            array = array.combine_chunks()

    if len(array) == 0:
        return pa.array([], type=target_arrow_type)

    encoder = _encode_map_row if pa.types.is_map(array.type) else _encode_row
    serialized = list(map(encoder, array.to_pylist()))

    string_arr = pa.array(
        serialized,
        type=pa.string(),
        memory_pool=memory_pool,
    )

    if target_arrow_type == pa.string():
        return string_arr

    return pc.cast(
        string_arr,
        target_arrow_type,
        memory_pool=memory_pool,
    )


def cast_polars_json_encode_series(
    series: Any,
    options: "CastOptions",
) -> Any:
    """Serialise a nested polars Series to string/binary using JSON.

    Polars has a native ``struct.json_encode`` but no equivalent for
    lists / maps, so the implementation simply roundtrips through
    Arrow's vectorised path and drops back into polars.
    """
    pl = get_polars()

    target_field = options.target_field
    if target_field is None:
        return series

    arrow_input = series.to_arrow()
    arrow_output = cast_arrow_json_encode_array(arrow_input, options=options)

    out = pl.from_arrow(arrow_output)

    if hasattr(out, "rename"):
        out = out.rename(target_field.name)

    return out


def cast_spark_json_encode_column(
    column: Any,
    options: "CastOptions",
) -> Any:
    spark = get_spark_sql()
    F = spark.functions

    target_field = options.target_field
    if target_field is None:
        return column

    json_col = F.to_json(column)

    target_type_id = target_field.dtype.type_id
    if target_type_id == DataTypeId.BINARY:
        return json_col.cast(target_field.dtype.to_spark())

    return json_col
