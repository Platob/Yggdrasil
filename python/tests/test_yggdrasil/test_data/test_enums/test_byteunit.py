"""Behaviors of :class:`yggdrasil.enums.byteunit.ByteUnit`.

The enum is the canonical byte-size table for every memory /
capacity / spill threshold in the codebase. The contract:

* members carry the IEC binary value (``B`` = 1, ``KIB`` = 1024,
  ``MIB`` = 1024², …);
* :class:`ByteUnit` ``IS`` :class:`int`, so ``128 * ByteUnit.MIB``
  reads as a plain integer at the call site;
* :meth:`ByteUnit.from_` accepts any common spelling — IEC tokens
  (``KiB``), short forms (``KB``), long forms (``kibibyte``,
  ``megabytes``), case-insensitive;
* :meth:`ByteUnit.parse_size` is the entry point for "give me an
  integer byte count from arbitrary config input" — strings like
  ``"128 MB"``, ints, members all converge.
"""
from __future__ import annotations

import pytest

from yggdrasil.enums.byteunit import ByteUnit


class TestCanonicalMembers:

    def test_members_match_iec_binary_values(self) -> None:
        assert ByteUnit.B.value == 1
        assert ByteUnit.KIB.value == 1024
        assert ByteUnit.MIB.value == 1024 ** 2
        assert ByteUnit.GIB.value == 1024 ** 3
        assert ByteUnit.TIB.value == 1024 ** 4
        assert ByteUnit.PIB.value == 1024 ** 5

    def test_short_aliases_point_to_iec_members(self) -> None:
        # KB == KiB == 1024 — there is no SI 10³ in this enum.
        assert ByteUnit.KB is ByteUnit.KIB
        assert ByteUnit.MB is ByteUnit.MIB
        assert ByteUnit.GB is ByteUnit.GIB
        assert ByteUnit.TB is ByteUnit.TIB
        assert ByteUnit.PB is ByteUnit.PIB

    def test_int_subclass_arithmetic(self) -> None:
        # The whole point: arithmetic against an int field works
        # without an explicit cast.
        assert 128 * ByteUnit.MIB == 128 * 1024 * 1024
        n: int = 4 * ByteUnit.KIB + 16
        assert n == 4 * 1024 + 16


class TestFromCoercion:

    def test_passthrough_member(self) -> None:
        assert ByteUnit.from_(ByteUnit.MIB) is ByteUnit.MIB

    @pytest.mark.parametrize("token,expected", [
        ("B", ByteUnit.B),
        ("kb", ByteUnit.KIB),
        ("KB", ByteUnit.KIB),
        ("KiB", ByteUnit.KIB),
        ("kibibyte", ByteUnit.KIB),
        ("kilobytes", ByteUnit.KIB),
        ("MB", ByteUnit.MIB),
        ("MiB", ByteUnit.MIB),
        ("megabyte", ByteUnit.MIB),
        ("GB", ByteUnit.GIB),
        ("gigabyte", ByteUnit.GIB),
        ("TB", ByteUnit.TIB),
        ("PB", ByteUnit.PIB),
    ])
    def test_string_aliases(self, token: str, expected: ByteUnit) -> None:
        assert ByteUnit.from_(token) is expected

    def test_canonical_member_name(self) -> None:
        # ``"MIB"`` is not in the alias table but matches a member.
        assert ByteUnit.from_("MIB") is ByteUnit.MIB

    def test_integer_value(self) -> None:
        assert ByteUnit.from_(1024) is ByteUnit.KIB
        assert ByteUnit.from_(1024 ** 2) is ByteUnit.MIB

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown byte unit"):
            ByteUnit.from_("zigabyte")

    def test_unknown_int_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ByteUnit value"):
            ByteUnit.from_(1500)

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be derived from None"):
            ByteUnit.from_(None)

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            ByteUnit.from_(True)

    def test_default_swallows_errors(self) -> None:
        sentinel = object()
        assert ByteUnit.from_("zigabyte", default=sentinel) is sentinel
        assert ByteUnit.from_(None, default=sentinel) is sentinel
        assert ByteUnit.from_(1500, default=sentinel) is sentinel

    def test_is_valid_predicate(self) -> None:
        assert ByteUnit.is_valid("128 MB"[3:].strip())  # "MB"
        assert ByteUnit.is_valid("KiB")
        assert ByteUnit.is_valid(1024)
        assert not ByteUnit.is_valid("zigabyte")
        assert not ByteUnit.is_valid(None)


class TestParseSize:

    def test_int_passthrough(self) -> None:
        assert ByteUnit.parse_size(0) == 0
        assert ByteUnit.parse_size(42) == 42
        assert ByteUnit.parse_size(1024) == 1024

    def test_member_yields_value(self) -> None:
        assert ByteUnit.parse_size(ByteUnit.MIB) == 1024 ** 2

    @pytest.mark.parametrize("text,expected", [
        ("128", 128),
        ("0", 0),
        ("128 B", 128),
        ("128B", 128),
        ("128 KB", 128 * 1024),
        ("128KB", 128 * 1024),
        ("128 KiB", 128 * 1024),
        ("128 MB", 128 * 1024 * 1024),
        ("128MiB", 128 * 1024 * 1024),
        ("1 GB", 1024 ** 3),
        ("1.5 GiB", int(1.5 * 1024 ** 3)),
        ("1.5 KiB", 1536),
        ("KB", 1024),  # bare unit → one of that unit
        ("MiB", 1024 ** 2),
    ])
    def test_quantity_strings(self, text: str, expected: int) -> None:
        assert ByteUnit.parse_size(text) == expected

    def test_negative_int_raises(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            ByteUnit.parse_size(-1)

    def test_negative_string_raises(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            ByteUnit.parse_size("-128 MB")

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown byte unit"):
            ByteUnit.parse_size("128 zigabyte")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot parse"):
            ByteUnit.parse_size("")

    def test_garbage_string_raises(self) -> None:
        # No digits + unrecognized unit → falls through bare-unit path
        # and surfaces the unit-table error, which is the more helpful
        # message anyway.
        with pytest.raises(ValueError, match="Unknown byte unit"):
            ByteUnit.parse_size("not a size")

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="None"):
            ByteUnit.parse_size(None)

    def test_default_swallows_errors(self) -> None:
        assert ByteUnit.parse_size(None, default=0) == 0
        assert ByteUnit.parse_size("", default=-1) == -1
        assert ByteUnit.parse_size("garbage", default=-1) == -1

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            ByteUnit.parse_size(True)

    def test_float_rounds(self) -> None:
        # parse_size(1.5) → 2 (round-half-to-even: 1.5 → 2)
        assert ByteUnit.parse_size(1.5) == 2
        assert ByteUnit.parse_size(0.4) == 0


class TestProperties:

    def test_bytes_property(self) -> None:
        assert ByteUnit.B.bytes == 1
        assert ByteUnit.MIB.bytes == 1024 ** 2

    def test_short_token(self) -> None:
        assert ByteUnit.B.short == "B"
        assert ByteUnit.KIB.short == "KB"
        assert ByteUnit.MIB.short == "MB"
        assert ByteUnit.GIB.short == "GB"

    def test_iec_token(self) -> None:
        assert ByteUnit.B.iec == "B"
        assert ByteUnit.KIB.iec == "KiB"
        assert ByteUnit.MIB.iec == "MiB"
        assert ByteUnit.GIB.iec == "GiB"


class TestFormat:

    def test_format_iec_default(self) -> None:
        assert ByteUnit.format(0) == "0 B"
        assert ByteUnit.format(1023) == "1023 B"
        assert ByteUnit.format(1024) == "1.0 KiB"
        assert ByteUnit.format(1536) == "1.5 KiB"
        assert ByteUnit.format(1024 ** 2) == "1.0 MiB"
        assert ByteUnit.format(128 * 1024 ** 2) == "128.0 MiB"

    def test_format_short(self) -> None:
        assert ByteUnit.format(1024, iec=False) == "1.0 KB"
        assert ByteUnit.format(128 * 1024 ** 2, iec=False) == "128.0 MB"

    def test_format_precision(self) -> None:
        assert ByteUnit.format(1536, precision=0) == "2 KiB"
        assert ByteUnit.format(1536, precision=2) == "1.50 KiB"

    def test_format_negative(self) -> None:
        assert ByteUnit.format(-1024) == "-1.0 KiB"


class TestEnumExportedFromPackage:
    """``ByteUnit`` is reachable from ``yggdrasil.enums``."""

    def test_top_level_import(self) -> None:
        from yggdrasil.enums import ByteUnit as Top
        assert Top is ByteUnit


class TestMemoryIntegration:
    """:class:`Memory` accepts ``ByteUnit``-style input for ``spill_bytes``."""

    def test_int_threshold(self) -> None:
        from yggdrasil.io.memory import Memory
        m = Memory(spill_bytes=4 * 1024)
        m.write_bytes(b"x" * (8 * 1024), 0)
        assert m.is_spilled
        m.clear()

    def test_string_threshold(self) -> None:
        from yggdrasil.io.memory import Memory
        m = Memory(spill_bytes="4 KiB")
        m.write_bytes(b"x" * (8 * 1024), 0)
        assert m.is_spilled
        m.clear()

    def test_byteunit_member_threshold(self) -> None:
        from yggdrasil.io.memory import Memory
        m = Memory(spill_bytes=ByteUnit.KIB)
        m.write_bytes(b"x" * (4 * 1024), 0)
        assert m.is_spilled
        m.clear()

    def test_arithmetic_threshold(self) -> None:
        from yggdrasil.io.memory import Memory
        m = Memory(spill_bytes=4 * ByteUnit.KIB)
        m.write_bytes(b"x" * (16 * 1024), 0)
        assert m.is_spilled
        m.clear()

    def test_invalid_threshold_raises(self) -> None:
        from yggdrasil.io.memory import Memory
        with pytest.raises(ValueError, match="non-negative byte count"):
            Memory(spill_bytes="not a size")
        with pytest.raises(ValueError, match="non-negative byte count"):
            Memory(spill_bytes=-1)
