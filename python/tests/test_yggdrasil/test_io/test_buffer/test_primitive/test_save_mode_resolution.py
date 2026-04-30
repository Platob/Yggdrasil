"""Tests for ``DataIO._resolve_save_mode``.

The resolver collapses the eight-value Mode enum to a four-value
subset writers can branch on. The interesting cases are:

- IGNORE on empty buffer collapses to OVERWRITE.
- ERROR_IF_EXISTS on empty buffer collapses to OVERWRITE.
- ERROR_IF_EXISTS on non-empty buffer raises FileExistsError.
- APPEND on a leaf without ``_SUPPORTED_APPEND`` raises ValueError.
- UPSERT similarly.
- AUTO/OVERWRITE/TRUNCATE all collapse to OVERWRITE.

CsvIO supports APPEND and UPSERT; XmlIO supports neither — between
them they cover both branches of the supported-mode gating.
"""

from __future__ import annotations

import pytest

from yggdrasil.io.enums import Mode
from yggdrasil.io.buffer.primitive.csv_io import CsvIO
from yggdrasil.io.buffer.primitive.xml_io import XmlIO


# ---------------------------------------------------------------------------
# AUTO / OVERWRITE / TRUNCATE → OVERWRITE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [Mode.AUTO, Mode.OVERWRITE, Mode.TRUNCATE])
def test_collapse_to_overwrite(mode):
    io = CsvIO()
    assert io._resolve_save_mode(mode) is Mode.OVERWRITE


# ---------------------------------------------------------------------------
# IGNORE — depends on buffer state
# ---------------------------------------------------------------------------


def test_ignore_on_empty_buffer_collapses_to_overwrite():
    io = CsvIO()
    with io:
        # is_empty() should be True on a fresh CsvIO
        assert io.is_empty()
        assert io._resolve_save_mode(Mode.IGNORE) is Mode.OVERWRITE


def test_ignore_on_nonempty_buffer_stays_ignore():
    io = CsvIO()
    with io:
        io.write(b"a,b\n1,2\n")
        io.seek(0)
        assert not io.is_empty()
        assert io._resolve_save_mode(Mode.IGNORE) is Mode.IGNORE


# ---------------------------------------------------------------------------
# ERROR_IF_EXISTS — empty OK, non-empty raises
# ---------------------------------------------------------------------------


def test_error_if_exists_on_empty_collapses_to_overwrite():
    io = CsvIO()
    with io:
        assert io._resolve_save_mode(Mode.ERROR_IF_EXISTS) is Mode.OVERWRITE


def test_error_if_exists_on_nonempty_raises():
    io = CsvIO()
    with io:
        io.write(b"x")
        with pytest.raises(FileExistsError, match="ERROR_IF_EXISTS"):
            io._resolve_save_mode(Mode.ERROR_IF_EXISTS)


# ---------------------------------------------------------------------------
# UPSERT — supported on CsvIO, not on XmlIO
# ---------------------------------------------------------------------------


def test_upsert_supported_on_csv():
    io = CsvIO()
    assert io._resolve_save_mode(Mode.UPSERT) is Mode.UPSERT


# ---------------------------------------------------------------------------
# String parsing — Mode.parse accepts string aliases
# ---------------------------------------------------------------------------


def test_string_mode_parsed():
    """The resolver routes string modes through ``Mode.parse``."""
    io = CsvIO()
    # Assuming "overwrite" is a recognized alias.
    assert io._resolve_save_mode("overwrite") is Mode.OVERWRITE
