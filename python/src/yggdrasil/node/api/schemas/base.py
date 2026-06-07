"""Shared request/response model base + ID helper.

``StrictModel`` is the contract type for everything that crosses the wire:
``extra="forbid"`` so a typo in a client payload fails loudly instead of
being silently dropped.

``make_id`` produces the xxhash composite IDs the project uses everywhere:
``xxh32(semantic_key) << 32 | timestamp_ms_low32``. Stable for the same
semantic key within a millisecond, monotonic-ish across time.
"""
from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict

__all__ = ["StrictModel", "make_id", "now_ms"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def now_ms() -> int:
    return int(time.time() * 1000)


def make_id(key: str, *, ts_ms: int | None = None) -> int:
    """Composite int64 ID: ``xxh32(key) << 32 | (ts_ms & 0xFFFFFFFF)``."""
    import xxhash

    ts = now_ms() if ts_ms is None else ts_ms
    return (xxhash.xxh32(key.encode("utf-8")).intdigest() << 32) | (ts & 0xFFFFFFFF)
