"""Behavior tests for :class:`yggdrasil.data.enums.mode.Mode`.

`Mode` is the cross-engine save-disposition enum: every sink (Path,
SQL writer, Spark/Delta writer) takes a ``Mode | str`` and routes
through ``Mode.from_`` for normalization. The contract under test:

* IntEnum members round-trip through ``int`` and ``cls(int)`` so
  persisted dispositions stay stable.
* ``Mode.from_`` is forgiving — POSIX ``open()`` strings, human
  aliases, the integer code, an existing ``Mode``, and ``None`` all
  resolve. Unknown / mixed / illegal shapes raise loudly.
* ``Mode.os_mode`` matches the disposition the BytesIO transactional
  layer expects (``OVERWRITE`` → ``"wb+"``, ``APPEND`` → ``"ab+"``,
  …).
"""
from __future__ import annotations

import pytest

from yggdrasil.data.enums.mode import Mode, STR_MAPPING


class TestCanonicalMembers:

    def test_members_round_trip_via_int(self) -> None:
        for m in Mode:
            assert Mode(int(m)) is m

    def test_read_only_predicates(self) -> None:
        assert Mode.READ_ONLY.is_read_only
        assert not Mode.READ_ONLY.allows_write
        for m in (Mode.OVERWRITE, Mode.APPEND, Mode.UPSERT, Mode.AUTO):
            assert not m.is_read_only
            assert m.allows_write


class TestOsMode:
    """`Mode.os_mode` is what BytesIO and LocalPath wrap into open()."""

    def test_overwrite_uses_wb_plus(self) -> None:
        assert Mode.OVERWRITE.os_mode == "wb+"
        assert Mode.TRUNCATE.os_mode == "wb+"

    def test_append_uses_ab_plus(self) -> None:
        assert Mode.APPEND.os_mode == "ab+"

    def test_error_if_exists_uses_xb_plus(self) -> None:
        assert Mode.ERROR_IF_EXISTS.os_mode == "xb+"

    def test_read_only_uses_rb(self) -> None:
        assert Mode.READ_ONLY.os_mode == "rb"

    def test_in_place_modes_use_rb_plus(self) -> None:
        for m in (Mode.AUTO, Mode.IGNORE, Mode.UPSERT, Mode.MERGE):
            assert m.os_mode == "rb+"


class TestFromIdentity:

    def test_mode_passes_through(self) -> None:
        for m in Mode:
            assert Mode.from_(m) is m

    def test_none_falls_to_auto_by_default(self) -> None:
        assert Mode.from_(None) is Mode.AUTO

    def test_none_uses_explicit_default_when_supplied(self) -> None:
        assert Mode.from_(None, default=Mode.READ_ONLY) is Mode.READ_ONLY

    def test_empty_string_resolves_to_auto(self) -> None:
        assert Mode.from_("") is Mode.AUTO
        assert Mode.from_("   ") is Mode.AUTO


class TestFromAlias:
    """Human / Spark / SQL aliases resolve via STR_MAPPING."""

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("write", Mode.OVERWRITE),
            ("overwrite", Mode.OVERWRITE),
            ("OVERWRITE", Mode.OVERWRITE),
            ("replace", Mode.OVERWRITE),
            ("clobber", Mode.OVERWRITE),

            ("append", Mode.APPEND),
            ("add", Mode.APPEND),

            ("read", Mode.READ_ONLY),
            ("readonly", Mode.READ_ONLY),
            ("read_only", Mode.READ_ONLY),
            ("read-only", Mode.READ_ONLY),
            ("ro", Mode.READ_ONLY),

            ("upsert", Mode.UPSERT),
            ("update", Mode.UPSERT),
            ("merge", Mode.MERGE),
            ("ignore", Mode.IGNORE),
            ("skip", Mode.IGNORE),
            ("truncate", Mode.TRUNCATE),

            ("error", Mode.ERROR_IF_EXISTS),
            ("fail", Mode.ERROR_IF_EXISTS),
            ("raise", Mode.ERROR_IF_EXISTS),
            ("error-if-exists", Mode.ERROR_IF_EXISTS),
            ("error_if_exists", Mode.ERROR_IF_EXISTS),

            ("auto", Mode.AUTO),
            ("default", Mode.AUTO),
        ],
    )
    def test_alias_resolves(self, alias: str, expected: Mode) -> None:
        assert Mode.from_(alias) is expected

    def test_alias_table_round_trip(self) -> None:
        for alias, mode in STR_MAPPING.items():
            assert Mode.from_(alias) is mode


class TestFromOSMode:
    """POSIX open() strings parse structurally — any flag order works."""

    @pytest.mark.parametrize(
        ("os_mode", "expected"),
        [
            ("r", Mode.READ_ONLY),
            ("rb", Mode.READ_ONLY),
            ("rt", Mode.READ_ONLY),
            ("w", Mode.OVERWRITE),
            ("wb", Mode.OVERWRITE),
            ("wt", Mode.OVERWRITE),
            ("a", Mode.APPEND),
            ("ab", Mode.APPEND),
            ("at", Mode.APPEND),
            ("x", Mode.ERROR_IF_EXISTS),
            ("xb", Mode.ERROR_IF_EXISTS),
            ("xt", Mode.ERROR_IF_EXISTS),
        ],
    )
    def test_pure_modes(self, os_mode: str, expected: Mode) -> None:
        assert Mode.from_(os_mode) is expected

    @pytest.mark.parametrize(
        ("os_mode", "expected"),
        [
            ("r+", Mode.AUTO),
            ("rb+", Mode.AUTO),
            ("r+b", Mode.AUTO),
            ("+rb", Mode.AUTO),
            ("rt+", Mode.AUTO),
            ("w+", Mode.OVERWRITE),
            ("wb+", Mode.OVERWRITE),
            ("w+b", Mode.OVERWRITE),
            ("a+", Mode.APPEND),
            ("ab+", Mode.APPEND),
            ("x+", Mode.ERROR_IF_EXISTS),
            ("xb+", Mode.ERROR_IF_EXISTS),
        ],
    )
    def test_plus_variants(self, os_mode: str, expected: Mode) -> None:
        assert Mode.from_(os_mode) is expected


class TestFromInvalid:

    def test_two_primaries_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            Mode.from_("rw")

    def test_mixed_b_and_t_raises(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            Mode.from_("wbt")

    def test_unknown_string_raises_with_helpful_message(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            Mode.from_("nope")

    def test_non_string_non_int_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="expected a string or Mode"):
            Mode.from_([1, 2, 3])  # type: ignore[arg-type]

    def test_unknown_int_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="integer codes"):
            Mode.from_(999)


class TestFromInteger:

    def test_int_round_trip(self) -> None:
        for m in Mode:
            assert Mode.from_(int(m)) is m

    def test_bool_input_is_typeerror(self) -> None:
        # ``True``/``False`` would otherwise sneak in as int(1) / int(0).
        # The contract is "string or enum"; bools are neither.
        with pytest.raises(TypeError):
            Mode.from_(True)  # type: ignore[arg-type]
