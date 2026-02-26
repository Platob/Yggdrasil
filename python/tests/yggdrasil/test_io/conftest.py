# tests/io/conftest.py
"""Shared fixtures and compression helpers for yggdrasil.io tests."""

from __future__ import annotations

import bz2
import gzip
import lzma
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Skip entire suite if yggdrasil is not installed
# ---------------------------------------------------------------------------

pytest.importorskip("yggdrasil")

from yggdrasil.io.buffer.bytes_io import BytesIO   # noqa: E402
from yggdrasil.io.config import BufferConfig        # noqa: E402

# ---------------------------------------------------------------------------
# Shared payload constants
# ---------------------------------------------------------------------------

#: ~6 KB of repetitive data — compresses well, large enough to trigger spills.
PAYLOAD = b"Brent ICE front-month daily close " * 200

#: A handful of bytes — always stays in memory under any reasonable threshold.
SMALL = b"Henry Hub prompt settle"


# ---------------------------------------------------------------------------
# Compression helpers — skip gracefully when optional deps are absent
# ---------------------------------------------------------------------------

def compress_gzip(data: bytes) -> bytes:
    return gzip.compress(data)


def compress_bz2(data: bytes) -> bytes:
    return bz2.compress(data)


def compress_xz(data: bytes) -> bytes:
    return lzma.compress(data)


def compress_zstd(data: bytes) -> bytes:
    zstd = pytest.importorskip("zstandard")
    return zstd.ZstdCompressor().compress(data)


def compress_lz4(data: bytes) -> bytes:
    lz4 = pytest.importorskip("lz4.frame")
    return lz4.compress(data)


def compress_snappy(data: bytes) -> bytes:
    cramjam = pytest.importorskip("cramjam")
    return bytes(cramjam.snappy.compress(data))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_buf() -> BytesIO:
    """In-memory BytesIO seeded with SMALL."""
    return BytesIO(SMALL)


@pytest.fixture
def empty_buf() -> BytesIO:
    """Freshly created empty BytesIO."""
    return BytesIO()


@pytest.fixture
def spill_config(tmp_path: Path) -> BufferConfig:
    """BufferConfig with a 64-byte spill threshold backed by tmp_path."""
    return BufferConfig(spill_bytes=64, tmp_dir=tmp_path)


@pytest.fixture
def spilled_buf(spill_config: BufferConfig) -> BytesIO:
    """A BytesIO that has already migrated to disk."""
    return BytesIO(PAYLOAD, config=spill_config)