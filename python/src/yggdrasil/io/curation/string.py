""":class:`StringCurator` — auto-type cleaned string columns.

Reads in the boundary case: a column of strings that *might* actually
be booleans, integers, floats, dates, times, or timestamps wearing a
text disguise (CSV, JSON-as-text, dict payloads, "we sent everything
as VARCHAR" SQL drivers, …). Cleans whitespace, normalizes the
common null tokens, then probes each candidate family in order and
keeps the first that absorbs every non-null cell.

Timestamps get *uniformized*: every parsed value lands in
:attr:`StringCurator.target_tz` (``"UTC"`` by default), so a mixed
column of offsets like ``"…+02:00"`` and ``"…-05:00"`` collapses to a
single ``timestamp(us, tz="UTC")`` array pointing at the same
instants the strings originally encoded. Naive strings stay naive
unless :attr:`StringCurator.assume_naive_tz` is set, in which case
they get stamped with that zone first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.enums.timezone import Timezone
from yggdrasil.data.types import (
    BooleanType,
    DateType,
    Float64Type,
    Int64Type,
    NullType,
    StringType,
    TimestampType,
    TimeType,
)

from .base import ArrayLike, CurationResult, Curator

if TYPE_CHECKING:
    from yggdrasil.data.types import DataType


__all__ = ["StringCurator"]


# Tokens recognized as "null". Compared after ``utf8_lower`` + trim, so
# the catalogue stays lower-case and the matching kernel handles the
# casing in one vectorised pass.
_DEFAULT_NULL_TOKENS = frozenset(
    {"", "null", "none", "na", "n/a", "#n/a", "nan", "-", "--", "?"}
)

_DEFAULT_TRUE_TOKENS = frozenset({"true", "t", "yes", "y"})
_DEFAULT_FALSE_TOKENS = frozenset({"false", "f", "no", "n"})

# Regexes used to pre-validate before ``pc.cast`` so we skip the
# costly ArrowInvalid throw when the column is plainly non-numeric.
# ``%`` is the conservative ascii integer / float shape; matches
# Python's ``int()`` / ``float()`` rules for the strings real CSVs
# carry. Exponent forms (``1e3``) match the float regex.
_INT_REGEX = r"^[+-]?\d+$"
_FLOAT_REGEX = r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$"


def _RE_MATCHES_OR_NULL(array: pa.Array, pattern: str) -> bool:
    """True iff every non-null cell of *array* matches *pattern*.

    Vectorised pre-check: ``pc.match_substring_regex`` does the work
    in C++ and the boolean reduction is one more kernel pass — total
    cost is roughly one pyarrow.compute scan, far cheaper than the
    ArrowInvalid throw a failed ``pc.cast`` pays.
    """
    matches = pc.match_substring_regex(array, pattern=pattern)
    covered = pc.or_(matches, pc.is_null(array))
    return pc.all(covered).as_py()


# Format catalogues match the polars ones in
# ``yggdrasil.data.types.primitive.temporal`` so the curator emits the
# same "this column is a date / time / timestamp" decision the Polars
# coalesce-strptime path would, even though we run everything through
# pyarrow.compute here. Day-first before month-first means
# ``"01/02/2024"`` parses as 1 Feb (the EU CSV default) rather than 2
# Jan.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
)
_TIME_FORMATS: tuple[str, ...] = (
    "%H:%M:%S.%f",
    "%H:%M:%S",
    "%H:%M",
    "%I:%M:%S %p",
)
# Aware-first: if the string carries an offset we want the parser to
# notice and produce a tz-aware Arrow timestamp. The first format that
# matches a given cell wins via :func:`pc.coalesce`.
_TIMESTAMP_FORMATS_AWARE: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S%z",
)
_TIMESTAMP_FORMATS_NAIVE: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
)


@dataclass(frozen=True)
class StringCurator(Curator):
    """Curate UTF-8 string columns into the most specific DataType.

    Rule order applied to a cleaned (trimmed + null-normalized) array:

    1. **All-null** → :class:`NullType`.
    2. **Boolean** if every non-null cell matches :attr:`true_tokens`
       or :attr:`false_tokens` (case-insensitive).
    3. **Int64** if every cell parses as an integer.
    4. **Float64** if every cell parses as a float.
    5. **Date** if every cell matches one of the date formats.
    6. **Time** if every cell matches one of the time formats.
    7. **Timestamp** if every cell matches one of the timestamp
       formats. Aware patterns are tried first; if any cell carries
       an offset, the column comes back uniformized to
       :attr:`target_tz`.
    8. **String** — fallback. The cleaned array is returned as
       :class:`StringType`.

    Parsing is fully vectorised — pyarrow.compute kernels stay inside
    the C++ runtime, no Python row loops.
    """

    null_tokens: frozenset[str] = field(default_factory=lambda: _DEFAULT_NULL_TOKENS)
    true_tokens: frozenset[str] = field(default_factory=lambda: _DEFAULT_TRUE_TOKENS)
    false_tokens: frozenset[str] = field(default_factory=lambda: _DEFAULT_FALSE_TOKENS)
    trim: bool = True
    parse_bool: bool = True
    parse_int: bool = True
    parse_float: bool = True
    parse_temporal: bool = True
    #: Target timezone for any tz-aware timestamp the column produces.
    #: Lets a mixed-offset column collapse to a single Arrow type.
    target_tz: Optional[str] = "UTC"
    #: When set, naive timestamp strings ("no offset") get this zone
    #: stamped on before normalization to :attr:`target_tz`. Leave
    #: ``None`` to keep naive strings naive.
    assume_naive_tz: Optional[str] = None
    timestamp_unit: str = "us"
    #: Drop null cells from the curated output array. Defaults to
    #: ``True`` so a per-array call to ``StringCurator().curate(...)``
    #: returns a clean, null-free typed array — the common shape for
    #: "I have these strings, give me the numbers". Auto-disabled by
    #: :meth:`Curator.curate_arrow_tabular` (and the engine DataFrame
    #: wrappers that go through it) because dropping nulls per column
    #: would break row alignment across columns.
    purge_nulls: bool = True

    # =========================================================== handles

    @classmethod
    def handles(cls, dtype: pa.DataType) -> bool:
        return (
            pa.types.is_string(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_string_view(dtype)
        )

    # ============================================================= main

    def curate(self, array: ArrayLike) -> CurationResult:
        chunked = isinstance(array, pa.ChunkedArray)
        if chunked:
            # ChunkedArray with one chunk is the cheap case; >1 we
            # combine because every trial below runs at array scope.
            # The combined array is a single contiguous buffer — same
            # zero-copy guarantee Polars / pandas get when they go via
            # ``combine_chunks``.
            flat = array.combine_chunks() if array.num_chunks != 1 else array.chunk(0)
        else:
            flat = array

        cleaned = self._clean(flat)

        for trial in self._trial_order():
            outcome = trial(cleaned)
            if outcome is not None:
                result, dtype = outcome
                break
        else:
            # Nothing matched — return the cleaned strings as-is.
            # ``flat`` came in as some string variant; downgrade to plain
            # ``string`` so the inferred type stays canonical.
            result, dtype = pc.cast(cleaned, pa.string()), StringType()

        # Auto-purge nulls — only kicks in for typed results that have
        # null cells to drop. ``NullType`` outputs stay length-preserved
        # because everything was null and "drop all" would be surprising.
        if (
            self.purge_nulls
            and not isinstance(dtype, NullType)
            and result.null_count > 0
        ):
            result = result.filter(pc.invert(pc.is_null(result)))

        if chunked:
            result = pa.chunked_array([result])
        return CurationResult(array=result, dtype=dtype)

    def infer(self, array: ArrayLike) -> "DataType":
        return self.curate(array).dtype

    # ========================================================== cleaning

    def _clean(self, array: pa.Array) -> pa.Array:
        """Trim + null-normalize. One pyarrow.compute pipeline."""
        # Cast through ``string`` so the rest of the pipeline doesn't
        # have to branch on large_string / string_view variants. The
        # cast is zero-copy when the input is already plain string.
        if array.type != pa.string():
            array = pc.cast(array, pa.string())

        if self.trim:
            array = pc.utf8_trim_whitespace(array)

        if self.null_tokens:
            value_set = pa.array(sorted(self.null_tokens), type=pa.string())
            is_null_token = pc.is_in(pc.utf8_lower(array), value_set=value_set)
            # ``pc.if_else`` doesn't take a typed null scalar without a
            # matching value-side type, so build a typed null array.
            null_arr = pa.nulls(len(array), type=pa.string())
            array = pc.if_else(is_null_token, null_arr, array)

        return array

    # ============================================================ trials

    def _trial_order(self):
        order = []
        if self.parse_bool:
            order.append(self._try_bool)
        if self.parse_int:
            order.append(self._try_int)
        if self.parse_float:
            order.append(self._try_float)
        if self.parse_temporal:
            order.append(self._try_date)
            order.append(self._try_time)
            order.append(self._try_timestamp)
        # Null short-circuit goes first regardless — it's a degenerate
        # case the other trials would also accept but with a less
        # specific type, so we want to claim it before they get a shot.
        order.insert(0, self._try_null)
        return order

    def _try_null(self, array: pa.Array):
        if array.null_count == len(array):
            # ``pc.cast(string, null)`` isn't a registered Arrow cast,
            # but a plain ``pa.nulls(n)`` lands on the same shape.
            return pa.nulls(len(array)), NullType()
        return None

    def _try_bool(self, array: pa.Array):
        lower = pc.utf8_lower(array)
        true_set = pa.array(sorted(self.true_tokens), type=pa.string())
        false_set = pa.array(sorted(self.false_tokens), type=pa.string())
        is_t = pc.is_in(lower, value_set=true_set)
        is_f = pc.is_in(lower, value_set=false_set)
        is_null = pc.is_null(array)
        covered = pc.or_(pc.or_(is_t, is_f), is_null)
        if not pc.all(covered).as_py():
            return None
        truthy = pc.if_else(is_t, pa.scalar(True), pa.scalar(False))
        null_arr = pa.nulls(len(array), type=pa.bool_())
        result = pc.if_else(is_null, null_arr, truthy)
        return result, BooleanType()

    def _try_int(self, array: pa.Array):
        # Cheap vectorised pre-check: are all non-null cells
        # integer-shaped? Avoids the costly ``pc.cast`` exception path
        # when the column is actually floats / labels / timestamps —
        # ArrowInvalid throwing on 10k rows of "1.5" was the dominant
        # cost on the StringCurator fallback hot path.
        if not _RE_MATCHES_OR_NULL(array, _INT_REGEX):
            return None
        return pc.cast(array, pa.int64(), safe=True), Int64Type()

    def _try_float(self, array: pa.Array):
        # Same shape as ``_try_int``: pre-validate via regex to skip
        # the exception cost when the column is non-numeric text.
        if not _RE_MATCHES_OR_NULL(array, _FLOAT_REGEX):
            return None
        return pc.cast(array, pa.float64(), safe=True), Float64Type()

    def _try_date(self, array: pa.Array):
        parsed = self._coalesce_strptime(array, _DATE_FORMATS, unit="s")
        if parsed is None:
            return None
        try:
            casted = pc.cast(parsed, pa.date32())
        except pa.ArrowInvalid:
            return None
        return casted, DateType()

    def _try_time(self, array: pa.Array):
        parsed = self._coalesce_strptime(array, _TIME_FORMATS, unit="us")
        if parsed is None:
            return None
        # ``pc.strptime`` returns ``timestamp`` even for time-only inputs
        # (anchored at 1900-01-01). Cast straight to ``time64[us]`` to
        # drop the synthetic date component.
        try:
            casted = pc.cast(parsed, pa.time64("us"))
        except pa.ArrowInvalid:
            return None
        return casted, TimeType(unit="us")

    def _try_timestamp(self, array: pa.Array):
        # Aware first: if the input carries offsets, we want the tz-aware
        # parse to claim the column and uniformize to ``target_tz``.
        # ``%z`` only matches cells that actually have an offset, so on
        # naive inputs the aware coalesce returns all nulls and we fall
        # through to the naive catalogue.
        aware = self._coalesce_strptime(
            array, _TIMESTAMP_FORMATS_AWARE, unit=self.timestamp_unit
        )
        if aware is not None:
            tz_str = self.target_tz or "UTC"
            # ``strptime`` with ``%z`` already produces ``timestamp[unit, tz=UTC]``;
            # cast to the requested zone for a uniform result.
            try:
                if tz_str != "UTC":
                    aware = pc.cast(aware, pa.timestamp(self.timestamp_unit, tz=tz_str))
            except pa.ArrowInvalid:
                return None
            return aware, TimestampType(
                unit=self.timestamp_unit, tz=Timezone.from_(tz_str)
            )

        naive = self._coalesce_strptime(
            array, _TIMESTAMP_FORMATS_NAIVE, unit=self.timestamp_unit
        )
        if naive is None:
            return None

        if self.assume_naive_tz is None:
            return naive, TimestampType(unit=self.timestamp_unit, tz=Timezone.NAIVE)

        # Naive strings get stamped with ``assume_naive_tz`` and then
        # converted to ``target_tz`` — a uniformized aware column built
        # from naive inputs, the same way ``polars.Series.dt.replace_time_zone``
        # would handle it.
        stamped = pc.assume_timezone(naive, self.assume_naive_tz)
        target_tz = self.target_tz or self.assume_naive_tz
        if target_tz != self.assume_naive_tz:
            stamped = pc.cast(stamped, pa.timestamp(self.timestamp_unit, tz=target_tz))
        return stamped, TimestampType(
            unit=self.timestamp_unit, tz=Timezone.from_(target_tz)
        )

    # ============================================================ helpers

    @staticmethod
    def _coalesce_strptime(
        array: pa.Array, formats: tuple[str, ...], *, unit: str
    ) -> Optional[pa.Array]:
        """Try every format in order; return the coalesced result if
        every non-null cell parsed under at least one format.

        Returns ``None`` when at least one originally-non-null cell
        comes back null — meaning no format absorbed it, so this
        family is the wrong call.
        """
        original_nulls = pc.is_null(array)
        out: Optional[pa.Array] = None
        for fmt in formats:
            try:
                attempt = pc.strptime(array, format=fmt, unit=unit, error_is_null=True)
            except pa.ArrowInvalid:
                # ``%I``/``%p`` style formats occasionally raise on
                # corner-case inputs (empty array, unsupported
                # combination); treat that as "this format doesn't
                # apply" and move on.
                continue
            out = attempt if out is None else pc.coalesce(out, attempt)

        if out is None:
            return None

        # A cell that was originally non-null but came back null after
        # the coalesce means none of the formats matched it. That
        # disqualifies the whole family.
        ended_null = pc.is_null(out)
        regressions = pc.and_(ended_null, pc.invert(original_nulls))
        if pc.any(regressions).as_py():
            return None
        return out
