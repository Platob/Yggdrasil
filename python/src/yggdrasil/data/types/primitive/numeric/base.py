"""Shared base for the numeric type family.

:class:`NumericType` lives here — the abstract :class:`PrimitiveType`
subclass that :class:`IntegerType`, :class:`FloatingPointType`, and
:class:`DecimalType` all inherit from. Two responsibilities:

* an empty-string / empty-bytes ``→ NULL`` normalization that runs
  before any numeric cast (Databricks / Spark CSV ingest semantics);
* a ``merge_with`` override that bridges any two integers (or any
  two floats) as same-kind, regardless of width / signedness — the
  specialized fixed-width subclasses carry distinct ``DataTypeId``
  values but still merge as one family.

Polars integer-signedness helpers (``_polars_flip_int_signedness``
/ ``_polars_is_integer`` / ``_polars_is_signed_int``) are kept here
too. ``IntegerType._cast_polars_*`` reaches for them when flipping
signed↔unsigned at the cast level — close enough to the type to
share a module, narrow enough to stay private.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.enums import Mode

from ..base import PrimitiveType
from ...base import DataType
from ...id import DataTypeId
from ...support import get_polars, get_spark_sql

if TYPE_CHECKING:
    import polars  # noqa: F401
    from ....cast.options import CastOptions  # noqa: F401


__all__ = [
    "NumericType",
    # Polars-int helpers exported to the integer submodule.
    "_polars_flip_int_signedness",
    "_polars_is_integer",
    "_polars_is_signed_int",
]


# View dtypes (``string_view`` / ``binary_view``) lack ``equal`` /
# ``if_else`` kernels in pyarrow — materialize to the concrete type
# before the empty-string normalization, then round-trip back.
_VIEW_TO_CONCRETE: dict[pa.DataType, pa.DataType] = {
    pa.string_view(): pa.large_string(),
    pa.binary_view(): pa.large_binary(),
}


@dataclass(frozen=True)
class NumericType(PrimitiveType, ABC):
    """Abstract base for every numeric leaf type.

    Sits between :class:`PrimitiveType` and the concrete numeric
    leaves. Handles cross-engine empty-string → null normalization
    (Databricks CSV ingest convention) and bridges sibling integer
    / float subclasses through ``merge_with`` so the specialized
    fixed-width classes still merge as one family.
    """

    @staticmethod
    def _source_type_id(options: "CastOptions") -> DataTypeId | None:
        sf = options.source
        return sf.dtype.type_id if sf is not None else None

    def merge_with(
        self,
        other: "DataType",
        mode: Mode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ):
        # Specialized fixed-width subclasses (``Int32Type``, ``UInt8Type``,
        # ``Float64Type``, ...) carry distinct ``type_id`` values, but
        # they still merge as one kind: any two integers collapse via
        # ``_merge_with_same_id`` regardless of width / signedness, and
        # likewise for floats. Decimals stay strict — they have their
        # own ``DECIMAL`` id and don't bridge to ints / floats here.
        mode = Mode.from_(mode, Mode.UPSERT)
        if mode is Mode.IGNORE:
            return self

        same_kind = (
            self.type_id == other.type_id
            or (self.type_id.is_integer and other.type_id.is_integer)
            or (self.type_id.is_floating_point and other.type_id.is_floating_point)
        )
        if same_kind:
            return self._merge_with_same_id(
                other=other, mode=mode, downcast=downcast, upcast=upcast
            )
        return self._merge_with_different_id(
            other=other, mode=mode, downcast=downcast, upcast=upcast
        )

    # ------------------------------------------------------------------
    # Arrow
    # ------------------------------------------------------------------

    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.Array:
        if options.need_cast(array, self):
            src_id = self._source_type_id(options)
            if src_id in (DataTypeId.STRING, DataTypeId.BINARY):
                # Materialize view → concrete so the ``equal`` kernel resolves.
                # pyarrow's ``equal`` is not registered for the view variants.
                original_type = array.type
                concrete_type = _VIEW_TO_CONCRETE.get(original_type, original_type)

                if concrete_type is not original_type:
                    array = array.cast(concrete_type)

                empty = b"" if src_id == DataTypeId.BINARY else ""
                array = pc.if_else(
                    pc.equal(array, pa.scalar(empty, type=concrete_type)),
                    pa.scalar(None, type=concrete_type),
                    array,
                )

                # Round-trip back to the original view type if we materialized.
                if concrete_type is not original_type:
                    array = array.cast(original_type)

        return super()._cast_arrow_array(array, options)

    # ------------------------------------------------------------------
    # Polars
    # ------------------------------------------------------------------

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ) -> "polars.Series":
        if options.need_cast(series, self):
            src_id = self._source_type_id(options)
            if src_id == DataTypeId.STRING:
                # Replace empty strings with null before the numeric cast.
                series = series.set(series.str.len_bytes() == 0, None)
            elif src_id == DataTypeId.BINARY:
                # Polars Binary: length is bytes-length of the buffer.
                series = series.set(series.bin.size() == 0, None)

        return super()._cast_polars_series(series, options)

    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options: "CastOptions",
    ) -> "polars.Expr":
        if options.need_cast(expr, self):
            pl = get_polars()
            src_id = self._source_type_id(options)

            if src_id == DataTypeId.STRING:
                expr = (
                    pl.when(expr.str.len_bytes() == 0)
                    .then(None)
                    .otherwise(expr)
                )
            elif src_id == DataTypeId.BINARY:
                expr = (
                    pl.when(expr.bin.size() == 0)
                    .then(None)
                    .otherwise(expr)
                )

        return super()._cast_polars_expr(expr, options)

    # ------------------------------------------------------------------
    # Spark
    # ------------------------------------------------------------------

    def _cast_spark_column(
        self,
        column: Any,
        options: "CastOptions",
    ) -> Any:
        if options.need_cast(column, self):
            spark = get_spark_sql()
            F = spark.functions
            src_id = self._source_type_id(options)

            # Spark's ``length()`` returns character length on StringType and
            # byte length on BinaryType — exactly the semantic we want.
            if src_id in (DataTypeId.STRING, DataTypeId.BINARY):
                column = F.when(F.length(column) == 0, F.lit(None)).otherwise(column)

        return super()._cast_spark_column(column, options)


# ---------------------------------------------------------------------------
# Polars integer signedness helpers — used by :mod:`integer` to flip
# signed↔unsigned at the cast level. Kept private to the numeric
# package; they're not useful outside it.
# ---------------------------------------------------------------------------


def _polars_flip_int_signedness(dtype: Any) -> Any:
    pl = get_polars()
    flip = {
        pl.Int8: pl.UInt8,
        pl.Int16: pl.UInt16,
        pl.Int32: pl.UInt32,
        pl.Int64: pl.UInt64,
        pl.UInt8: pl.Int8,
        pl.UInt16: pl.Int16,
        pl.UInt32: pl.Int32,
        pl.UInt64: pl.Int64,
    }
    return flip[dtype]


def _polars_is_integer(dtype: Any) -> bool:
    pl = get_polars()
    return dtype in {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    }


def _polars_is_signed_int(dtype: Any) -> bool:
    pl = get_polars()
    return dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64}
