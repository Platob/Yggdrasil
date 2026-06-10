"""JSON-shaped primitive datatypes — :class:`SJsonType` and :class:`BJsonType`.

These are storage-pinned siblings of :class:`StringType` / :class:`BinaryType`
that carry "this column holds JSON" intent through the schema. Their
physical encoding is identical to the underlying string/binary type
(Arrow / Polars / Spark all lack a true native JSON dtype outside
extension types we deliberately don't pull in here), but the dtype
itself is distinct so:

* casts *to* the type encode arbitrary nested / scalar values as JSON
  text (or JSON bytes for ``BJsonType``);
* casts *from* the type decode JSON into the requested target — array,
  map, struct, or any primitive — using the same vectorised helpers
  the nested types already use for ``string → list / struct / map``.

Inference (``handles_arrow_type`` / ``handles_polars_type`` /
``handles_spark_type``) deliberately returns ``False`` so a plain
``pa.string()`` column never silently lands as ``SJsonType`` — the
JSON intent has to come from the user (``DataType.from_str("sjson")``,
an explicit field annotation, etc.).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from .binary import BinaryType
from .string import StringType
from ..id import DataTypeId

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst
    from ...options import CastOptions


__all__ = ["SJsonType", "BJsonType"]


# --------------------------------------------------------------------------
# Shared helpers — nested encode / string-source decode plumbing.
#
# Imported lazily inside the cast methods so ``primitive`` stays free of
# a top-level dependency on ``nested`` (the import order at module
# import time is primitive → nested, and nested.cast helpers in turn
# reach back into ``primitive`` for StringType/BinaryType).
# --------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    import datetime as _dt
    import decimal as _decimal

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


def _dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False)


def _convert_pyobj_to_json_text(value: Any, safe: bool = False) -> str | None:
    """Coerce a Python value to JSON text.

    Strings/bytes are passed through (re-validated when ``safe`` is set);
    everything else is ``json.dumps``'d via :func:`_dumps`. The
    ``safe`` flag mirrors the rest of the primitive ``_convert_pyobj``
    surface: when ``True`` we raise on failure, when ``False`` we
    return ``None`` so the caller can null the cell out.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if safe:
            try:
                json.loads(value)
            except Exception as e:
                raise ValueError(
                    f"Value is not valid JSON text: {value!r}"
                ) from e
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            text = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            if safe:
                raise ValueError(
                    f"Cannot decode bytes as UTF-8 JSON: {value!r}"
                )
            return None
        if safe:
            try:
                json.loads(text)
            except Exception as e:
                raise ValueError(
                    f"Value is not valid JSON text: {text!r}"
                ) from e
        return text
    try:
        return _dumps(value)
    except Exception:
        if safe:
            raise ValueError(
                f"Cannot convert {type(value).__name__} to JSON text: {value!r}"
            )
        return None


# --------------------------------------------------------------------------
# Mixin — the cast/encode side is identical for SJSON and BJSON; they only
# differ in the storage type they ride on. We attach the cast hooks via a
# shared helper invoked from each subclass to keep the dataclass
# inheritance chain (``StringType`` / ``BinaryType``) clean.
# --------------------------------------------------------------------------


def _cast_arrow_to_json(
    self: "SJsonType | BJsonType",
    array: "pa.Array | pa.ChunkedArray",
    options: "CastOptions",
):
    from ..nested._cast_json import cast_arrow_json_encode_array

    options = options.check_source(array).check_target(self)
    source_id = options.source.dtype.type_id

    if source_id.is_nested:
        return cast_arrow_json_encode_array(array, options=options)

    # SJSON ↔ BJSON / SJSON ↔ STRING / BJSON ↔ BINARY: storage is
    # already a string/binary blob, so reuse StringType / BinaryType's
    # plain cast (utf-8 encode/decode + view/large reshape).
    return self._cast_storage_arrow(array, options)


def _cast_polars_to_json_series(
    self: "SJsonType | BJsonType",
    series: Any,
    options: "CastOptions",
) -> Any:
    from ..nested._cast_json import cast_polars_json_encode_series

    options = options.check_source(series).check_target(self)
    source_id = options.source.dtype.type_id

    if source_id.is_nested:
        return cast_polars_json_encode_series(series, options=options)

    return self._cast_storage_polars_series(series, options)


def _cast_polars_to_json_expr(
    self: "SJsonType | BJsonType",
    expr: Any,
    options: "CastOptions",
) -> Any:
    options = options.check_target(self)
    source_id = options.source.dtype.type_id

    if source_id.is_nested:
        # Polars has ``Expr.struct.json_encode`` for structs, but no
        # equivalent for list / map expressions — and the encode helper
        # operates per-series. Fall back to materialising via the series
        # cast at execution time by wrapping the expr through arrow.
        from ..nested._cast_json import cast_polars_json_encode_series
        from yggdrasil.lazy_imports import polars_module

        polars_module()
        series_cast = cast_polars_json_encode_series

        def _apply(series: Any) -> Any:
            return series_cast(series, options=options)

        return expr.map_batches(_apply, return_dtype=self.to_polars())

    return self._cast_storage_polars_expr(expr, options)


def _cast_spark_to_json_column(
    self: "SJsonType | BJsonType",
    column: Any,
    options: "CastOptions",
) -> Any:
    from ..nested._cast_json import cast_spark_json_encode_column

    options = options.check_source(column).check_target(self)
    source_id = options.source.dtype.type_id

    if source_id.is_nested:
        return cast_spark_json_encode_column(column, options=options)

    return self._cast_storage_spark(column, options)


# --------------------------------------------------------------------------
# SJSON — JSON encoded as UTF-8 text. Storage matches StringType.
# --------------------------------------------------------------------------


@dataclass(frozen=True, repr=False)
class SJsonType(StringType):
    """JSON encoded as a UTF-8 string (text JSON).

    ``to_arrow()`` emits ``pa.string()`` (or ``pa.large_string()`` /
    ``pa.string_view()`` when ``large`` / ``view`` are set) and Polars /
    Spark land on their plain string types — there is no ABI-stable
    native JSON dtype in any of those engines that we want to bind to.
    The distinction lives at the yggdrasil layer: a column declared
    ``sjson`` round-trips through schema metadata, gets the right cast
    behavior (encode on the way in, decode on the way out), and shows
    up as ``sjson`` in pretty-printed output.
    """

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.SJSON

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        s = "sjson"
        if self.large:
            s = "large_" + s
        if self.view:
            s = s + "_view"
        return f"{pad}{s}"

    # ------------------------------------------------------------------
    # Engine probes / constructors — inference is opt-in. A plain
    # ``pa.string()`` column should NEVER silently land as SJsonType;
    # the JSON intent has to come from the user (an explicit dtype,
    # ``DataType.from_str("sjson")``, etc.).
    # ------------------------------------------------------------------

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return False

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        return False

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return False

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.SJSON, "SJSON", "JSON_STRING")

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "SJsonType":
        try:
            return cls(
                large=bool(value.get("large", False)),
                view=bool(value.get("view", False)),
                fixed_size=bool(value.get("fixed_size", False)),
                byte_size=value.get("byte_size"),
            )
        except Exception as e:
            if default is ...:
                raise e
            return default

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    def to_spark_name(self) -> str:
        # Databricks has no native JSON DDL — store as STRING and rely
        # on ``from_json`` / ``to_json`` SQL helpers downstream.
        return "STRING"

    # ------------------------------------------------------------------
    # Cast — JSON encode on the way in, plain-string passthrough otherwise
    # ------------------------------------------------------------------

    def _cast_storage_arrow(self, array, options):
        return StringType._cast_arrow_array(self, array, options)

    def _cast_storage_polars_series(self, series, options):
        return StringType._cast_polars_series(self, series, options)

    def _cast_storage_polars_expr(self, expr, options):
        return StringType._cast_polars_expr(self, expr, options)

    def _cast_storage_spark(self, column, options):
        return StringType._cast_spark_column(self, column, options)

    def _cast_arrow_array(self, array, options):
        return _cast_arrow_to_json(self, array, options)

    def _cast_polars_series(self, series, options):
        return _cast_polars_to_json_series(self, series, options)

    def _cast_polars_expr(self, expr, options):
        return _cast_polars_to_json_expr(self, expr, options)

    def _cast_spark_column(self, column, options):
        return _cast_spark_to_json_column(self, column, options)

    # ------------------------------------------------------------------
    # Defaults / scalar conversion
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        return "null"

    def _convert_pyobj(self, value: Any, safe: bool = False):
        return _convert_pyobj_to_json_text(value, safe=safe)


# --------------------------------------------------------------------------
# BJSON — JSON encoded as bytes. Storage matches BinaryType.
# --------------------------------------------------------------------------


@dataclass(frozen=True, repr=False)
class BJsonType(BinaryType):
    """JSON encoded as bytes (binary / packed JSON).

    Storage matches :class:`BinaryType` — ``pa.binary()`` / ``pa.large_binary()``
    / ``pa.binary_view()`` / fixed-width ``pa.binary(n)`` depending on
    flags. The wire format is plain UTF-8 JSON unless a downstream
    consumer interprets it differently (BSON, MessagePack-encoded JSON,
    etc.); yggdrasil itself only writes UTF-8 JSON when encoding from
    nested / scalar values.
    """

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.BJSON

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        s = "bjson"
        if self.large:
            s = "large_" + s
        if self.view:
            s = s + "_view"
        if self.byte_size is not None:
            s = f"{s}({self.byte_size})"
        return f"{pad}{s}"

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return False

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        return False

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return False

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.BJSON, "BJSON", "JSON_BINARY", "JSONB")

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "BJsonType":
        try:
            return cls(
                byte_size=value.get("byte_size"),
                large=bool(value.get("large", False)),
                view=bool(value.get("view", False)),
            )
        except Exception as e:
            if default is ...:
                raise e
            return default

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    def to_spark_name(self) -> str:
        return "BINARY"

    # ------------------------------------------------------------------
    # Cast
    # ------------------------------------------------------------------

    def _cast_storage_arrow(self, array, options):
        return BinaryType._cast_arrow_array(self, array, options)

    def _cast_storage_polars_series(self, series, options):
        return BinaryType._cast_polars_series(self, series, options)

    def _cast_storage_polars_expr(self, expr, options):
        return BinaryType._cast_polars_expr(self, expr, options)

    def _cast_storage_spark(self, column, options):
        return BinaryType._cast_spark_column(self, column, options)

    def _cast_arrow_array(self, array, options):
        return _cast_arrow_to_json(self, array, options)

    def _cast_polars_series(self, series, options):
        return _cast_polars_to_json_series(self, series, options)

    def _cast_polars_expr(self, expr, options):
        return _cast_polars_to_json_expr(self, expr, options)

    def _cast_spark_column(self, column, options):
        return _cast_spark_to_json_column(self, column, options)

    # ------------------------------------------------------------------
    # Defaults / scalar conversion
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        return b"null"

    def _convert_pyobj(self, value: Any, safe: bool = False):
        text = _convert_pyobj_to_json_text(value, safe=safe)
        if text is None:
            return None
        out = text.encode("utf-8")
        if self.byte_size is not None:
            if len(out) < self.byte_size:
                out = out.ljust(self.byte_size, b"\x00")
            elif len(out) > self.byte_size:
                if safe:
                    raise ValueError(
                        f"JSON byte length {len(out)} exceeds fixed "
                        f"byte_size={self.byte_size}: {out!r}"
                    )
                out = out[: self.byte_size]
        return out
