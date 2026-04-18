# tests/io/conftest.py
"""Shared fixtures for yggdrasil.io tests.

Factories / constants live in `_helpers.py` so tests can import them
directly (conftest modules are pytest plugins, not importable modules).
"""

from __future__ import annotations

import bz2
import gzip
import lzma
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Skip everything if yggdrasil itself isn't importable.
pytest.importorskip("yggdrasil")

from yggdrasil.io.buffer.bytes_io import BytesIO       # noqa: E402
from yggdrasil.io.config import BufferConfig           # noqa: E402
from yggdrasil.io.request import PreparedRequest       # noqa: E402
from yggdrasil.io.response import Response             # noqa: E402

from ._helpers import (                                # noqa: E402
    MockSession,
    make_request,
    make_response,
    make_table_mock,
)

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


@pytest.fixture
def req() -> PreparedRequest:
    """A fresh GET request with deterministic `sent_at`."""
    return make_request()


@pytest.fixture
def resp(req: PreparedRequest) -> Response:
    """A fresh 200 response anchored to `req`."""
    return make_response(request=req)


@pytest.fixture
def mock_session() -> MockSession:
    """A MockSession with an empty response queue."""
    return MockSession()


@pytest.fixture
def mock_table() -> MagicMock:
    """A MagicMock table whose SQL execute returns zero hits."""
    return make_table_mock()
