"""Subclass-level type-system safety on primitive ``from_arrow_type``.

Each primitive subclass owns ``handles_arrow_type`` and
``from_arrow_type``; the latter must raise on inputs the former
rejects, instead of producing nonsense via attribute access. These
tests pin that contract for the temporal subclasses (which have the
richest unit/tz handling) and lock in the integer ``__str__`` form
that callers grep for in error messages.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.types.primitive import (
    DurationType,
    IntegerType,
    TimeType,
    TimestampType,
)


class TestTimeTypeArrowSafety:

    def test_rejects_non_time_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported Arrow data type"):
            TimeType.from_arrow_type(pa.int64())

    def test_accepts_time32_with_unit(self) -> None:
        out = TimeType.from_arrow_type(pa.time32("ms"))

        assert out.byte_size == 4
        assert out.unit == "ms"

    def test_accepts_time64_with_unit(self) -> None:
        out = TimeType.from_arrow_type(pa.time64("us"))

        assert out.byte_size == 8
        assert out.unit == "us"


class TestTimestampTypeArrowSafety:

    def test_rejects_non_timestamp_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported Arrow data type"):
            TimestampType.from_arrow_type(pa.int64())

    def test_preserves_tz_and_unit(self) -> None:
        out = TimestampType.from_arrow_type(pa.timestamp("us", tz="UTC"))

        assert out.unit == "us"
        assert out.tz == "Etc/UTC"


class TestDurationTypeArrowSafety:

    def test_rejects_non_duration_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported Arrow data type"):
            DurationType.from_arrow_type(pa.int64())

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_accepts_every_arrow_unit(self, unit: str) -> None:
        out = DurationType.from_arrow_type(pa.duration(unit))

        assert out.unit == unit
        assert out.byte_size == 8


class TestIntegerTypeStr:
    """``str(IntegerType(...))`` is grepped by error messages — pin it."""

    @pytest.mark.parametrize(
        "byte_size,expected",
        [(1, "int8"), (2, "int16"), (4, "int32"), (8, "int64")],
    )
    def test_signed(self, byte_size: int, expected: str) -> None:
        assert str(IntegerType(byte_size=byte_size, signed=True)) == expected

    @pytest.mark.parametrize(
        "byte_size,expected",
        [(1, "uint8"), (2, "uint16"), (4, "uint32"), (8, "uint64")],
    )
    def test_unsigned(self, byte_size: int, expected: str) -> None:
        assert str(IntegerType(byte_size=byte_size, signed=False)) == expected
