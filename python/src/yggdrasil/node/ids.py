"""Numeric ID generation using xxhash for fast, non-cryptographic hashing.

IDs are int64 composed of two parts:
- High 32 bits: xxhash of a semantic key (name, concept)
- Low 32 bits: timestamp-based component (milliseconds since epoch, truncated)

This gives collision-resistant, sortable, compact identifiers.
"""
from __future__ import annotations

import time


def _xxh32(data: str) -> int:
    """Fast xxhash32. Uses xxhash if available, falls back to a simple hash."""
    try:
        import xxhash
        return xxhash.xxh32_intdigest(data.encode())
    except ImportError:
        h = hash(data) & 0xFFFFFFFF
        return h


def make_id(semantic_key: str) -> int:
    """Generate an int64 ID: xxh32(key) << 32 | timestamp_ms & 0xFFFFFFFF."""
    high = _xxh32(semantic_key)
    low = int(time.time() * 1000) & 0xFFFFFFFF
    return (high << 32) | low


def make_id_pair(key1: str, key2: str) -> int:
    """Generate an int64 from two semantic keys: xxh32(key1) << 32 | xxh32(key2)."""
    return (_xxh32(key1) << 32) | _xxh32(key2)
