# yggdrasil/pyutils/dynamic_buffer/_types.py
"""Shared type aliases for the dynamic_buffer package."""

from __future__ import annotations

from typing import Union

__all__ = ["BytesLike"]

BytesLike = Union[bytes, bytearray, memoryview]