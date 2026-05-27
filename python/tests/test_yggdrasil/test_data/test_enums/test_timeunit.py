"""Behaviors of :class:`yggdrasil.enums.timeunit.TimeUnit`.

The enum is the canonical token table for every temporal ``DataType``
in the codebase (``DateType`` / ``TimeType`` / ``TimestampType`` /
``DurationType``). The contract:

* members carry the canonical short string (``s``, ``ms``, ``us``,
  ``ns``, ``d``, ``year_month``, ``day_time``, ``month_day_nano``);
* :meth:`TimeUnit.from_` accepts any common spelling — long names,
  plurals, ``TimeUnit`` instances — and raises on unknowns;
* :class:`TimeUnit` ``IS`` :class:`str`, so it slots into existing
  ``unit: str`` declarations without coercion at the boundary.
"""
from __future__ import annotations

import pytest

from yggdrasil.enums.timeunit import TimeUnit


class TestCanonicalMembers:

    def test_members_match_short_token(self) -> None:
        assert TimeUnit.SECOND.value == "s"
        assert TimeUnit.MILLISECOND.value == "ms"
        assert TimeUnit.MICROSECOND.value == "us"
        assert TimeUnit.NANOSECOND.value == "ns"
        assert TimeUnit.DAY.value == "d"
        assert TimeUnit.YEAR_MONTH.value == "year_month"
        assert TimeUnit.DAY_TIME.value == "day_time"
        assert TimeUnit.MONTH_DAY_NANO.value == "month_day_nano"

    def test_subclasses_str(self) -> None:
        # ``unit: str = "us"`` field declarations accept TimeUnit
        # without coercion — the enum subclasses str.
        assert isinstance(TimeUnit.MICROSECOND, str)
        assert TimeUnit.MICROSECOND == "us"
        assert "us" == TimeUnit.MICROSECOND

    def test_short_aliases_resolve_to_canonical(self) -> None:
        assert TimeUnit.S is TimeUnit.SECOND
        assert TimeUnit.MS is TimeUnit.MILLISECOND
        assert TimeUnit.US is TimeUnit.MICROSECOND
        assert TimeUnit.NS is TimeUnit.NANOSECOND
        assert TimeUnit.D is TimeUnit.DAY


class TestFrom:

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("s", TimeUnit.SECOND),
            ("ms", TimeUnit.MILLISECOND),
            ("us", TimeUnit.MICROSECOND),
            ("ns", TimeUnit.NANOSECOND),
            ("d", TimeUnit.DAY),
            # Plurals and long forms.
            ("seconds", TimeUnit.SECOND),
            ("millisecond", TimeUnit.MILLISECOND),
            ("microseconds", TimeUnit.MICROSECOND),
            ("nano", TimeUnit.NANOSECOND),
            ("days", TimeUnit.DAY),
            # Case / whitespace tolerance.
            ("  US  ", TimeUnit.MICROSECOND),
            ("Microsecond", TimeUnit.MICROSECOND),
            # Unicode μs.
            ("µs", TimeUnit.MICROSECOND),
            # Calendar interval forms.
            ("year_month", TimeUnit.YEAR_MONTH),
            ("YearMonth", TimeUnit.YEAR_MONTH),
            ("day-time", TimeUnit.DAY_TIME),
            ("month_day_nano", TimeUnit.MONTH_DAY_NANO),
        ],
    )
    def test_accepts_known_aliases(self, value: str, expected: TimeUnit) -> None:
        assert TimeUnit.from_(value) is expected

    def test_returns_timeunit_unchanged(self) -> None:
        assert TimeUnit.from_(TimeUnit.NANOSECOND) is TimeUnit.NANOSECOND

    def test_rejects_unknown_string(self) -> None:
        with pytest.raises(ValueError, match="Unknown time unit"):
            TimeUnit.from_("fortnight")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(TypeError):
            TimeUnit.from_(42)

    def test_default_returned_on_unknown(self) -> None:
        assert TimeUnit.from_("xyz", default=None) is None
        assert TimeUnit.from_(None, default=TimeUnit.SECOND) is TimeUnit.SECOND

    def test_none_without_default_raises(self) -> None:
        with pytest.raises(ValueError):
            TimeUnit.from_(None)

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            TimeUnit.from_("")
        assert TimeUnit.from_("   ", default=TimeUnit.MICROSECOND) is TimeUnit.MICROSECOND


class TestIsValid:

    @pytest.mark.parametrize("value", ["s", "ns", "microseconds", "DAY", TimeUnit.MICROSECOND])
    def test_recognises_known(self, value) -> None:
        assert TimeUnit.is_valid(value) is True

    @pytest.mark.parametrize("value", ["fortnight", "", None, 42])
    def test_rejects_unknown(self, value) -> None:
        assert TimeUnit.is_valid(value) is False


class TestProperties:

    def test_seconds_per_unit(self) -> None:
        assert TimeUnit.SECOND.seconds == 1.0
        assert TimeUnit.MILLISECOND.seconds == 1.0e-3
        assert TimeUnit.MICROSECOND.seconds == 1.0e-6
        assert TimeUnit.NANOSECOND.seconds == 1.0e-9
        assert TimeUnit.DAY.seconds == 86400.0

    def test_calendar_intervals_have_no_fixed_seconds(self) -> None:
        import math

        assert math.isnan(TimeUnit.YEAR_MONTH.seconds)
        assert math.isnan(TimeUnit.DAY_TIME.seconds)
        assert math.isnan(TimeUnit.MONTH_DAY_NANO.seconds)

    def test_order_ranks_subsecond_finest(self) -> None:
        # Used by temporal merge to pick the wider precision.
        assert TimeUnit.NANOSECOND.order > TimeUnit.MICROSECOND.order
        assert TimeUnit.MICROSECOND.order > TimeUnit.MILLISECOND.order
        assert TimeUnit.MILLISECOND.order > TimeUnit.SECOND.order

    def test_is_subsecond(self) -> None:
        assert TimeUnit.MILLISECOND.is_subsecond
        assert TimeUnit.MICROSECOND.is_subsecond
        assert TimeUnit.NANOSECOND.is_subsecond
        assert not TimeUnit.SECOND.is_subsecond
        assert not TimeUnit.DAY.is_subsecond

    def test_is_calendar(self) -> None:
        assert TimeUnit.YEAR_MONTH.is_calendar
        assert TimeUnit.DAY_TIME.is_calendar
        assert TimeUnit.MONTH_DAY_NANO.is_calendar
        assert not TimeUnit.MICROSECOND.is_calendar
        assert not TimeUnit.SECOND.is_calendar
