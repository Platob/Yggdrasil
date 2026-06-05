"""Tests for yggdrasil.io.types."""

from __future__ import annotations

import typing

from yggdrasil.io.types import BytesLike


def test_bytes_like_includes_bytes_bytearray_memoryview():
    args = set(typing.get_args(BytesLike))
    assert {bytes, bytearray, memoryview} <= args
