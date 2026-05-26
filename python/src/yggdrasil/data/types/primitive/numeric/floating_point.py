"""Floating-point type family — :class:`FloatingPointType` + sized subclasses.

Layout:

* :class:`FloatingPointType` — abstract / unsized IEEE float with a
  ``__new__`` redirect that promotes generic
  ``FloatingPointType(byte_size=N)`` calls to the matching
  fixed-width subclass via :data:`_SPECIALIZED_FLOAT_TYPES`.
* :class:`Float8Type` — 1-byte FP8 (E4M3 / E5M2 storage tag); no
  native Arrow / Polars / Spark equivalent — widens at every engine
  boundary.
* :class:`Float16Type` / :class:`Float32Type` / :class:`Float64Type`
  — IEEE half / single / double precision. Spark only has 32-bit
  and 64-bit, so :meth:`FloatingPointType.as_spark` widens 1-byte
  and 2-byte forms to ``Float32Type``.

The dispatch table at the bottom is the single source of truth for
the redirect.
"""
from __future__ import annotations

import decimal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from yggdrasil.enums import Mode

from .._helpers import _coerce_str
from ...base import _default_singleton
from ...id import DataTypeId
from yggdrasil.lazy_imports import polars_module, spark_sql_module
from .base import NumericType

if TYPE_CHECKING:
    import polars  # noqa: F401
    import pyspark.sql.types as pst  # noqa: F401
    from ...base import DataType  # noqa: F401


__all__ = [
    "FloatingPointType",
    "Float8Type",
    "Float16Type",
    "Float32Type",
    "Float64Type",
]


@dataclass(frozen=True, repr=False)
class FloatingPointType(NumericType):

    def __new__(cls, byte_size: int | None = None, **kwargs):
        # See :meth:`IntegerType.__new__` — same dispatch, keyed on
        # ``byte_size`` only. ``bfloat16`` rides the same Float16Type as
        # IEEE half-precision; we don't model BF separately here.
        # Always redirect when a specialized class is registered, so
        # ``Float32Type(byte_size=8)`` lands on ``Float64Type`` rather
        # than carrying a malformed width.
        target = _SPECIALIZED_FLOAT_TYPES.get(byte_size)
        resolved = target if (target is not None and target is not cls) else cls
        # Singleton fast path — leaf ``FloatXType`` classes share one
        # instance so the lazy ``to_arrow`` / ``to_polars`` /
        # ``to_spark`` caches survive across every caller.
        if not kwargs and resolved in _SPECIALIZED_FLOAT_TYPES.values():
            return _default_singleton(resolved)
        return object.__new__(resolved)

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * indent * level
        bits = (self.byte_size or 8) * 8
        return f"{pad}float{bits}"

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.FLOAT

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: Mode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "FloatingPointType":
        if not isinstance(other, FloatingPointType):
            raise TypeError(
                f"Cannot merge FloatingPointType with {other.__class__.__name__}"
            )

        if mode is Mode.IGNORE:
            return self
        if mode is Mode.OVERWRITE:
            return other

        if mode is Mode.AUTO:
            byte_size = self.byte_size or other.byte_size
            if byte_size == self.byte_size:
                return self
            return FloatingPointType(byte_size=byte_size)

        if downcast == upcast:
            return self

        left_size = self.byte_size or 8
        right_size = other.byte_size or 8
        byte_size = min(left_size, right_size) if downcast else max(left_size, right_size)
        # Build through the abstract base so the ``__new__`` redirect
        # selects the right ``FloatXX`` specialized class.
        return FloatingPointType(byte_size=byte_size)

    # ------------------------------------------------------------------
    # Engine probes / constructors
    # ------------------------------------------------------------------

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_floating(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "FloatingPointType":
        if pa.types.is_float32(dtype):
            return cls(byte_size=4)
        if pa.types.is_float64(dtype):
            return cls(byte_size=8)
        if pa.types.is_float16(dtype):
            return cls(byte_size=2)
        raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = polars_module()
        return dtype in {pl.Float32, pl.Float64}

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "FloatingPointType":
        pl = polars_module()
        if dtype == pl.Float32:
            return cls(byte_size=4)
        if dtype == pl.Float64:
            return cls(byte_size=8)
        raise TypeError(f"Unsupported Polars data type: {dtype!r}")

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = spark_sql_module()
        return isinstance(dtype, (spark.types.FloatType, spark.types.DoubleType))

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "FloatingPointType":
        spark = spark_sql_module()
        if isinstance(dtype, spark.types.FloatType):
            return cls(byte_size=4)
        if isinstance(dtype, spark.types.DoubleType):
            return cls(byte_size=8)
        raise TypeError(f"Unsupported Spark data type: {dtype!r}")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, cls.class_type_id())

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "FloatingPointType":
        try:
            # Use the subclass's declared default so a payload that
            # only carries ``id`` (e.g. ``{"id": FLOAT16}``) lands on
            # the matching class — the abstract ``byte_size=8`` would
            # send ``Float8Type.from_dict({...})`` straight to
            # ``Float64Type`` via the ``__new__`` redirect.
            byte_size_default = cls.__dataclass_fields__["byte_size"].default
            if byte_size_default is None:
                byte_size_default = 8
            return cls(byte_size=value.get("byte_size", byte_size_default))
        except Exception as e:
            if default is ...:
                raise ValueError(
                    f"Could not parse FloatingPointType from dict: {value!r}."
                ) from e
            return default

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    @property
    def _size(self) -> int:
        return self.byte_size or 8

    def _default_pyhint(self) -> Any:
        # ``float`` is the canonical Python hint regardless of byte
        # width (Python doesn't expose float8 / float16 / float32 as
        # distinct annotations). ``np.float32`` aliases survive
        # via the ``_pyhint_cache`` stamp on first parse.
        return float

    def to_arrow(self) -> pa.DataType:
        if self._size == 2:
            return pa.float16() if hasattr(pa, "float16") else pa.float32()
        return pa.float64() if self._size == 8 else pa.float32()

    def to_polars(self) -> "polars.DataType":
        pl = polars_module()
        return pl.Float64 if self._size == 8 else pl.Float32

    def to_spark(self) -> Any:
        t = spark_sql_module().types
        return t.DoubleType() if self._size == 8 else t.FloatType()

    def as_spark(self) -> "FloatingPointType":
        # Spark has ``FloatType`` (32-bit) and ``DoubleType`` (64-bit)
        # but no native sub-32-bit floats — widen ``Float8Type`` and
        # ``Float16Type`` up to 32-bit.
        if self._size in (1, 2):
            return FloatingPointType(byte_size=4)
        return self

    def as_polars(self) -> "FloatingPointType":
        # Polars has ``Float32`` (32-bit) and ``Float64`` (64-bit)
        # only — same widening rule as :meth:`as_spark`.
        if self._size in (1, 2):
            return FloatingPointType(byte_size=4)
        return self

    def to_spark_name(self) -> str:
        return "DOUBLE" if self._size == 8 else "FLOAT"

    # ------------------------------------------------------------------
    # Defaults / conversion
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else 0.0

    def _convert_pyobj(self, value: Any, safe: bool = False) -> float | None:
        token = _coerce_str(value)
        if token is not None:
            stripped = token.strip()
            if not stripped:
                if safe:
                    raise ValueError(
                        f"Cannot parse float from empty string for {type(self).__name__}."
                    )
                return None
            try:
                return float(stripped)
            except (TypeError, ValueError):
                if safe:
                    raise ValueError(
                        f"Cannot parse float from {value!r} for {type(self).__name__}."
                    )
                return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, decimal.Decimal):
            return float(value)
        if safe:
            raise ValueError(
                f"Cannot convert {type(value).__name__} to float "
                f"for {type(self).__name__}: {value!r}."
            )
        return None


# ---------------------------------------------------------------------------
# Specialized fixed-width float subclasses. Each carries its own
# ``DataTypeId``; the dispatch table at the bottom drives the
# ``__new__`` redirect.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, repr=False)
class Float8Type(FloatingPointType):
    """1-byte FP8 — the storage tag for ML-framework Float8 variants.

    No native Arrow / Polars / Spark equivalent: ``to_arrow`` and
    ``to_polars`` widen to 32-bit, ``to_spark`` produces ``FloatType``,
    and :meth:`FloatingPointType.as_spark` collapses to
    ``Float32Type`` so downstream Spark pipelines see a width Spark
    can actually represent. Carry the tag through schemas / round-
    trips at the yggdrasil layer; convert at the boundary.
    """

    byte_size: int | None = 1

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.FLOAT8


@dataclass(frozen=True, repr=False)
class Float16Type(FloatingPointType):
    byte_size: int | None = 2

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.FLOAT16


@dataclass(frozen=True, repr=False)
class Float32Type(FloatingPointType):
    byte_size: int | None = 4

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.FLOAT32


@dataclass(frozen=True, repr=False)
class Float64Type(FloatingPointType):
    byte_size: int | None = 8

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.FLOAT64


_SPECIALIZED_FLOAT_TYPES: dict[int | None, type[FloatingPointType]] = {
    1: Float8Type,
    2: Float16Type,
    4: Float32Type,
    8: Float64Type,
}
