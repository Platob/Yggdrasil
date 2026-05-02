"""Tests for yggdrasil.io.enums.mode."""

from __future__ import annotations

import pytest

from yggdrasil.io.enums.mode import STR_MAPPING, Mode


class TestModeMembers:
    def test_str_subclass(self):
        assert isinstance(Mode.OVERWRITE, str)
        assert Mode.OVERWRITE == "overwrite"

    def test_all_expected_members(self):
        names = {m.name for m in Mode}
        assert names == {
            "AUTO",
            "OVERWRITE",
            "APPEND",
            "IGNORE",
            "UPSERT",
            "TRUNCATE",
            "ERROR_IF_EXISTS",
        }


class TestModeFromIdentity:
    def test_mode_returned_as_is(self):
        assert Mode.from_(Mode.APPEND) is Mode.APPEND

    def test_none_returns_default(self):
        assert Mode.from_(None) is Mode.AUTO
        assert Mode.from_(None, default=Mode.APPEND) is Mode.APPEND


class TestModeFromAliases:
    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("overwrite", Mode.OVERWRITE),
            ("REPLACE", Mode.OVERWRITE),
            ("clobber", Mode.OVERWRITE),
            ("append", Mode.APPEND),
            ("add", Mode.APPEND),
            ("ignore", Mode.IGNORE),
            ("skip", Mode.IGNORE),
            ("upsert", Mode.UPSERT),
            ("merge", Mode.UPSERT),
            ("truncate", Mode.TRUNCATE),
            ("error_if_exists", Mode.ERROR_IF_EXISTS),
            ("error-if-exists", Mode.ERROR_IF_EXISTS),
            ("auto", Mode.AUTO),
            ("default", Mode.AUTO),
            ("", Mode.AUTO),
        ],
    )
    def test_alias(self, alias, expected):
        assert Mode.from_(alias) is expected


class TestModeFromOSStrings:
    @pytest.mark.parametrize(
        ("os_mode", "expected"),
        [
            ("r", Mode.AUTO),
            ("rb", Mode.AUTO),
            ("rt", Mode.AUTO),
            ("rb+", Mode.AUTO),
            ("r+b", Mode.AUTO),
            ("w", Mode.OVERWRITE),
            ("wb", Mode.OVERWRITE),
            ("wt", Mode.OVERWRITE),
            ("wb+", Mode.OVERWRITE),
            ("a", Mode.APPEND),
            ("ab", Mode.APPEND),
            ("ab+", Mode.APPEND),
            ("x", Mode.ERROR_IF_EXISTS),
            ("xb", Mode.ERROR_IF_EXISTS),
        ],
    )
    def test_os_modes(self, os_mode, expected):
        assert Mode.from_(os_mode) is expected

    def test_invalid_os_mode_raises(self):
        with pytest.raises(ValueError):
            Mode.from_("rw")  # two primaries

    def test_b_and_t_mutually_exclusive(self):
        with pytest.raises(ValueError):
            Mode.from_("rbt")


class TestModeFromInvalid:
    def test_unknown_string_raises(self):
        with pytest.raises(ValueError):
            Mode.from_("not-a-mode")

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            Mode.from_(42)  # type: ignore[arg-type]


class TestStrMapping:
    def test_table_consistency(self):
        # Spot-check that the table aligns with from_() for each entry.
        for alias, expected in STR_MAPPING.items():
            assert Mode.from_(alias) is expected
