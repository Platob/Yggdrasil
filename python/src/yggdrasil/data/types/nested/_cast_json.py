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

# ``orjson.JSONDecodeError`` subclasses :class:`json.JSONDecodeError`, so
# anything that does ``except json.JSONDecodeError`` keeps catching the
# orjson failures too — including the ``msg`` / ``lineno`` / ``colno``
# attributes we use for row-pointing error messages downstream.

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.lazy_imports import polars_module, spark_sql_module

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


_JSON_STRING_SOURCE_TYPES = frozenset(
    {
        DataTypeId.STRING,
        DataTypeId.BINARY,
        # Storage-pinned JSON variants — same wire shape as STRING / BINARY,
        # so the same vectorised decoder applies.
        DataTypeId.SJSON,
        DataTypeId.BJSON,
    }
)
_JSON_NESTED_TYPES = frozenset(
    {DataTypeId.ARRAY, DataTypeId.MAP, DataTypeId.STRUCT}
)


def is_json_string_source(source_type_id: DataTypeId) -> bool:
    return source_type_id in _JSON_STRING_SOURCE_TYPES


def is_json_string_target(target_type_id: DataTypeId) -> bool:
    return target_type_id in _JSON_STRING_SOURCE_TYPES


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


def cast_arrow_json_string_array(
    array: pa.Array | pa.ChunkedArray,
    options: "CastOptions",
) -> pa.Array:
    """Parse a string/binary Arrow array as JSON into the target arrow type.

    The hot path stays inside the C++ runtime: rows are wrapped as
    ``{"v":<row>}`` via Arrow compute, joined with newlines into a
    single NDJSON buffer, and decoded in one ``pyarrow.json.read_json``
    call against an ``explicit_schema`` that names the target type.
    No per-row Python loop runs on success.

    Strict mode (``options.safe=True``, the default) re-raises any
    parse failure as :class:`pa.ArrowInvalid`, enriched with the row
    index PyArrow reports plus a value preview pulled from the source
    array.

    Permissive mode (``options.safe=False``) falls back to per-row
    Python decoding *only* when the vectorised pass fails, so bad
    rows can null-out individually instead of failing the batch.

    Map targets — ``pa.map_(...)`` — bypass the NDJSON path because
    PyArrow's JSON reader cannot emit ``map`` arrays directly; for
    those we go straight to the Python path, which builds the map
    via ``pa.array(list_of_dicts, type=map_type)``.
    """
    target_field = options.target
    if target_field is None:
        return array

    target_arrow_type = target_field.dtype.to_arrow()
    memory_pool = options.arrow_memory_pool

    normalized = _arrow_to_utf8(array, memory_pool=memory_pool)

    if len(normalized) == 0:
        return pa.array([], type=target_arrow_type)

    if _pyarrow_json_supports_target(target_arrow_type):
        try:
            return _vectorized_parse_json(
                normalized, target_arrow_type, memory_pool=memory_pool,
            )
        except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            if options.safe:
                raise _enrich_pyarrow_json_error(exc, normalized) from exc
            # Permissive: drop into per-row fallback so bad rows null
            # out individually instead of failing the whole batch.
        except pa.ArrowNotImplementedError:
            # Predicate should already filter unsupported targets;
            # treat any leakage as a signal to use the Python path.
            pass

    return _parse_via_python(
        normalized, target_arrow_type, memory_pool=memory_pool, safe=options.safe,
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
) -> pa.ArrowInvalid:
    """Promote a raw pyarrow.json error to a row-pointing ArrowInvalid.

    The pyarrow message already names the failing row (``... in row N``);
    we lift that index out via regex, pluck the offending source value
    with an Arrow scalar lookup (no Python loop) and append a truncated
    preview plus the standard ``safe=False`` hint.
    """
    msg = str(exc)
    match = _PYARROW_JSON_ROW_RE.search(msg)
    if match is None:
        return pa.ArrowInvalid(
            f"Invalid JSON: {msg}. "
            f"Pass safe=False on CastOptions to coerce bad rows to null."
        )

    idx = int(match.group(1))
    preview: Any = None
    if 0 <= idx < len(normalized):
        scalar = normalized[idx]
        if scalar.is_valid:
            value = scalar.as_py()
            preview = value if len(value) <= 120 else value[:117] + "..."

    return pa.ArrowInvalid(
        f"Invalid JSON at row {idx}: {msg}. "
        f"Value: {preview!r}. "
        f"Pass safe=False on CastOptions to coerce bad rows to null."
    )


def _parse_via_python(
    normalized: pa.Array,
    target_arrow_type: pa.DataType,
    memory_pool: pa.MemoryPool | None,
    safe: bool,
) -> pa.Array:
    """Python-side fallback for targets pyarrow.json cannot emit (maps)
    and for permissive-mode batches that need per-row null-on-failure.

    Strict mode keeps the fast single-blob ``json.loads`` path and only
    drops to per-row on failure to surface a row-pointing error.
    """
    pylist = normalized.to_pylist()

    if safe:
        try:
            blob = "[" + ",".join(
                "null" if s is None else s for s in pylist
            ) + "]"
            parsed = orjson.loads(blob)
        except json.JSONDecodeError:
            parsed = _parse_json_rows_strict(pylist)
        return pa.array(parsed, type=target_arrow_type, memory_pool=memory_pool)

    parsed = _parse_json_rows_permissive(pylist)
    try:
        return pa.array(parsed, type=target_arrow_type, memory_pool=memory_pool)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, ValueError):
        return _build_arrow_per_row(parsed, target_arrow_type, memory_pool)


def _parse_json_rows_strict(pylist: list[str | None]) -> list[Any]:
    out: list[Any] = [None] * len(pylist)
    for idx, blob in enumerate(pylist):
        if blob is None:
            continue
        try:
            out[idx] = orjson.loads(blob)
        except json.JSONDecodeError as exc:
            preview = blob if len(blob) <= 120 else blob[:117] + "..."
            raise pa.ArrowInvalid(
                f"Invalid JSON at row {idx}: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno}). "
                f"Value: {preview!r}. "
                f"Pass safe=False on CastOptions to coerce bad rows to null."
            ) from exc
    return out


def _parse_json_rows_permissive(pylist: list[str | None]) -> list[Any]:
    out: list[Any] = [None] * len(pylist)
    for idx, blob in enumerate(pylist):
        if blob is None:
            continue
        try:
            out[idx] = orjson.loads(blob)
        except (json.JSONDecodeError, ValueError):
            out[idx] = None
    return out


def _build_arrow_per_row(
    parsed: list[Any],
    target_arrow_type: pa.DataType,
    memory_pool: pa.MemoryPool | None,
) -> pa.Array:
    cells: list[Any] = [None] * len(parsed)
    for idx, value in enumerate(parsed):
        if value is None:
            continue
        try:
            pa.array([value], type=target_arrow_type, memory_pool=memory_pool)
            cells[idx] = value
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, ValueError):
            cells[idx] = None
    return pa.array(cells, type=target_arrow_type, memory_pool=memory_pool)


def cast_polars_json_string_expr(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = polars_module()

    target_field = options.target
    if target_field is None:
        return expr

    target_polars_type = target_field.dtype.to_polars()

    src_id = options.source.dtype.type_id
    src_dtype = options.source.dtype.to_polars()
    if src_id == DataTypeId.BINARY or src_id == DataTypeId.BJSON or src_dtype == pl.Binary:
        expr = expr.cast(pl.String)

    if not options.safe:
        # Polars ``json_decode`` raises on any row it can't parse; the
        # permissive contract is per-row null-on-failure. Route through
        # ``map_elements`` with a per-row try/except so bad rows null
        # out instead of failing the whole frame.
        return expr.map_elements(
            _polars_json_decode_permissive,
            return_dtype=target_polars_type,
            skip_nulls=True,
        )

    return expr.str.json_decode(target_polars_type)


def _polars_json_decode_permissive(value: Any) -> Any:
    if value is None:
        return None
    try:
        return orjson.loads(value)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def cast_polars_json_string_series(
    series: Any,
    options: "CastOptions",
) -> Any:
    pl = polars_module()
    expr = cast_polars_json_string_expr(pl.col(series.name), options).alias(
        options.target.name
    )
    return pl.DataFrame({series.name: series}).select(expr).to_series()


def cast_spark_json_string_column(
    column: Any,
    options: "CastOptions",
) -> Any:
    spark = spark_sql_module()
    F = spark.functions

    target_field = options.target
    if target_field is None:
        return column

    target_ddl = target_field.dtype.to_spark_name()

    src_id = options.source.dtype.type_id
    if src_id == DataTypeId.BINARY or src_id == DataTypeId.BJSON:
        column = column.cast("string")

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
    # orjson is ~10x faster than stdlib json for common types (int/float/str/bool/
    # list/dict/None).  It returns bytes; decode to UTF-8 once per row.  Fall back
    # to stdlib for types orjson rejects (Decimal, raw bytes, timedelta) so the
    # _json_default coercions still fire for edge-case values.
    try:
        return orjson.dumps(value).decode("utf-8")
    except TypeError:
        return json.dumps(value, default=_json_default, ensure_ascii=False)


def _encode_map_row(value: Any) -> Any:
    if value is None:
        return None
    d = dict(value)
    try:
        return orjson.dumps(d).decode("utf-8")
    except TypeError:
        return json.dumps(d, default=_json_default, ensure_ascii=False)


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
    target_field = options.target
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
    pl = polars_module()

    target_field = options.target
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
    spark = spark_sql_module()
    F = spark.functions

    target_field = options.target
    if target_field is None:
        return column

    json_col = F.to_json(column)

    target_type_id = target_field.dtype.type_id
    if target_type_id == DataTypeId.BINARY:
        return json_col.cast(target_field.dtype.to_spark())

    return json_col
