""":class:`IntegerCurator` / :class:`FloatCurator` — numeric shrinkers.

Both downcast a numeric column to the smallest dtype whose range still
holds every observed value losslessly. Useful when an upstream produced
``int64`` / ``float64`` defensively but the actual values fit in a
narrower (and cheaper to ship across the wire) type.

* :class:`IntegerCurator` inspects ``min`` / ``max`` and picks the
  narrowest signed integer that covers the range. With
  ``allow_unsigned=True`` (default) a non-negative column collapses to
  the matching unsigned width instead — same number of bits, double
  the positive range.

* :class:`FloatCurator` tries float32 (and optionally float16) and
  keeps the narrowed result only when round-tripping through it
  preserves every non-null cell exactly. NaN / infinity round-trip
  cleanly because float32 preserves them.

Both rules are vectorised (``pc.min_max`` / ``pc.cast`` /
``pc.equal``) — no Python row loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.types import (
    Float16Type,
    Float32Type,
    Float64Type,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    NullType,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
)

from .base import ArrayLike, CurationResult, Curator

if TYPE_CHECKING:
    from yggdrasil.data.types import DataType


__all__ = ["IntegerCurator", "FloatCurator"]


# Ordered narrow → wide. Iterated top-down so the first fit wins.
_SIGNED_INT_LADDER: tuple[tuple[pa.DataType, int, int, type], ...] = (
    (pa.int8(), -(2**7), 2**7 - 1, Int8Type),
    (pa.int16(), -(2**15), 2**15 - 1, Int16Type),
    (pa.int32(), -(2**31), 2**31 - 1, Int32Type),
    (pa.int64(), -(2**63), 2**63 - 1, Int64Type),
)
_UNSIGNED_INT_LADDER: tuple[tuple[pa.DataType, int, int, type], ...] = (
    (pa.uint8(), 0, 2**8 - 1, UInt8Type),
    (pa.uint16(), 0, 2**16 - 1, UInt16Type),
    (pa.uint32(), 0, 2**32 - 1, UInt32Type),
    (pa.uint64(), 0, 2**64 - 1, UInt64Type),
)


@dataclass(frozen=True)
class IntegerCurator(Curator):
    """Downcast integer columns to the narrowest dtype that holds them.

    Rules:

    1. **All null** → :class:`NullType`. (Same short-circuit as every
       other curator — no width has to "hold" zero values.)
    2. Pick the narrowest signed width whose ``[min, max]`` covers the
       observed range.
    3. With ``allow_unsigned=True`` (default), if ``min >= 0`` swap to
       the matching-or-narrower unsigned width when it's strictly
       smaller — ``uint8`` (1 B) beats ``int16`` (2 B) for [0..255].
    """

    allow_unsigned: bool = True
    #: Cap on the output width. ``"int64"`` keeps the upcast space
    #: fully open; set to ``"int32"`` (or ``"int16"`` / ``"int8"``) to
    #: cap downstream storage budgets that won't tolerate 64-bit.
    max_width: str = "int64"

    @classmethod
    def handles(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_integer(dtype)

    def infer(self, array: ArrayLike) -> "DataType":
        return self.curate(array).dtype

    def curate(self, array: ArrayLike) -> CurationResult:
        if array.null_count == len(array):
            return CurationResult(
                array=pa.nulls(len(array)),
                dtype=NullType(),
            )

        lo, hi = _min_max(array)
        chosen_arrow, chosen_dtype = self._pick_width(lo, hi)
        if chosen_arrow == array.type:
            return CurationResult(array=array, dtype=chosen_dtype())
        casted = pc.cast(array, chosen_arrow, safe=True)
        return CurationResult(array=casted, dtype=chosen_dtype())

    # ------------------------------------------------------------------

    def _pick_width(self, lo: int, hi: int):
        max_signed = self._max_signed_value()
        # Signed ladder — first row whose [min, max] covers the range.
        signed_pick: Optional[tuple[pa.DataType, type]] = None
        for arrow_type, lo_bound, hi_bound, ygg_type in _SIGNED_INT_LADDER:
            if lo_bound > -max_signed - 1 - 1:
                # Cap reached — the next ladder rung would breach
                # ``max_width``, so stop. (This branch is unreachable
                # because the loop break below catches it first, but
                # kept as a defensive guard for future ladder edits.)
                pass
            if lo_bound <= lo and hi <= hi_bound:
                signed_pick = (arrow_type, ygg_type)
                break
            if arrow_type.bit_width >= self._max_bits():
                break

        if not self.allow_unsigned or lo < 0:
            if signed_pick is None:
                # Range exceeds even int64 — keep the original width
                # so the caller's data doesn't silently lose
                # precision.
                return _SIGNED_INT_LADDER[-1][0], _SIGNED_INT_LADDER[-1][3]
            return signed_pick

        # Unsigned candidate — prefer it whenever the bit width is
        # the same as the signed pick (uint8 beats int8 by carrying
        # double the positive range at the same storage cost) or
        # narrower. Skip when only the signed width fits.
        for arrow_type, _lo_bound, hi_bound, ygg_type in _UNSIGNED_INT_LADDER:
            if hi <= hi_bound:
                if (
                    signed_pick is None
                    or arrow_type.bit_width <= signed_pick[0].bit_width
                ):
                    return arrow_type, ygg_type
                return signed_pick
            if arrow_type.bit_width >= self._max_bits():
                break

        return signed_pick or (_SIGNED_INT_LADDER[-1][0], _SIGNED_INT_LADDER[-1][3])

    def _max_bits(self) -> int:
        token = self.max_width.lower()
        if token.startswith("int") or token.startswith("uint"):
            return int(token.removeprefix("u").removeprefix("int"))
        raise ValueError(
            f"IntegerCurator.max_width must be one of 'int8' / 'int16' / "
            f"'int32' / 'int64' (or the uint variants); got {self.max_width!r}."
        )

    def _max_signed_value(self) -> int:
        return 2 ** (self._max_bits() - 1) - 1


# Float ladder, narrow → wide. ``float16`` is gated behind a flag
# because round-trip equality fails for many real-world values
# (only 11-bit mantissa), so it makes a poor default.
_FLOAT_LADDER: tuple[tuple[pa.DataType, type], ...] = (
    (pa.float32(), Float32Type),
    (pa.float64(), Float64Type),
)


@dataclass(frozen=True)
class FloatCurator(Curator):
    """Downcast float columns to the narrowest dtype that preserves values.

    Tries ``float32`` (and optionally ``float16``) and keeps the
    narrowed result only when a round-trip through it returns every
    non-null cell unchanged. NaN / +Inf / -Inf survive any IEEE-754
    width so they don't disqualify the cast.
    """

    allow_float16: bool = False

    @classmethod
    def handles(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_floating(dtype)

    def infer(self, array: ArrayLike) -> "DataType":
        return self.curate(array).dtype

    def curate(self, array: ArrayLike) -> CurationResult:
        if array.null_count == len(array):
            return CurationResult(array=pa.nulls(len(array)), dtype=NullType())

        candidates: list[tuple[pa.DataType, type]] = []
        if self.allow_float16:
            candidates.append((pa.float16(), Float16Type))
        candidates.extend(_FLOAT_LADDER)

        # Walk narrow → wide. The current width disqualifies anything
        # wider, so stop as soon as we hit it.
        for arrow_type, ygg_type in candidates:
            if arrow_type == array.type:
                return CurationResult(array=array, dtype=ygg_type())
            if arrow_type.bit_width >= array.type.bit_width:
                break
            if _round_trip_preserves(array, arrow_type):
                return CurationResult(
                    array=pc.cast(array, arrow_type, safe=True),
                    dtype=ygg_type(),
                )

        # Nothing narrow fit — keep the source dtype.
        return CurationResult(
            array=array,
            dtype=_ygg_for_arrow_float(array.type)(),
        )


# ===================================================================== utils


def _min_max(array: ArrayLike) -> tuple[int, int]:
    """Vectorised ``(min, max)`` over non-null cells."""
    res = pc.min_max(array)
    return res["min"].as_py(), res["max"].as_py()


def _round_trip_preserves(array: ArrayLike, narrow: pa.DataType) -> bool:
    """True iff every non-null cell survives ``→ narrow → back`` unchanged.

    NaN doesn't compare equal to itself under IEEE-754 — pyarrow's
    ``pc.equal`` returns ``false`` (not ``null``) for NaN-vs-NaN, so
    we have to mask NaN explicitly before the elementwise compare.
    The original nulls land in the ``pc.equal`` null output and pass
    through automatically.
    """
    narrowed = pc.cast(array, narrow, safe=False)
    widened = pc.cast(narrowed, array.type, safe=False)
    same = pc.equal(array, widened)
    # NaN-vs-NaN reads as ``false`` from pc.equal — patch the mask.
    nan_in_original = pc.is_nan(array)
    nan_in_widened = pc.is_nan(widened)
    nan_match = pc.and_(nan_in_original, nan_in_widened)
    # Original-nulls land as ``null`` in ``same``; treat as preserved.
    coverage = pc.or_(pc.or_(pc.is_null(same), same), nan_match)
    return pc.all(coverage).as_py()


def _ygg_for_arrow_float(dtype: pa.DataType) -> type:
    if pa.types.is_float16(dtype):
        return Float16Type
    if pa.types.is_float32(dtype):
        return Float32Type
    return Float64Type
