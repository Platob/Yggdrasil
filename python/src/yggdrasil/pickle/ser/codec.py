from __future__ import annotations

import gzip
import threading
import zlib
from typing import Final

from yggdrasil.pickle.ser.constants import (
    CODEC_GZIP,
    CODEC_NONE,
    CODEC_ZLIB,
    CODEC_ZSTD,
)
from yggdrasil.pickle.ser.errors import InvalidCodecError

try:
    import zstandard as _zstd
except Exception:  # pragma: no cover - optional dependency
    _zstd = None

__all__ = [
    "DEFAULT_CODEC",
    "codec_name",
    "default_codec",
    "compress_bytes",
    "decompress_bytes",
]

_CODEC_NAMES: Final[dict[int, str]] = {
    CODEC_NONE: "none",
    CODEC_GZIP: "gzip",
    CODEC_ZSTD: "zstd",
    CODEC_ZLIB: "zlib",
}

# Fast serializer-oriented defaults.
_GZIP_COMPRESSLEVEL: Final[int] = 1
_ZLIB_LEVEL: Final[int] = 1
_ZSTD_LEVEL: Final[int] = 1

# zstandard's ZstdCompressor / ZstdDecompressor wrap mutable native contexts
# (ZSTD_CCtx / ZSTD_DCtx) that are NOT safe to share across threads. Concurrent
# .compress() / .decompress() calls on the same instance corrupt the context
# and produce native access violations on Windows.
#
# We keep one compressor and one decompressor per thread via threading.local,
# so each thread reuses its own contexts (cheap, avoids per-call allocation)
# without ever sharing them.
_tls = threading.local()


def _zstd_compressor() -> "_zstd.ZstdCompressor":
    c = getattr(_tls, "zstd_c", None)
    if c is None:
        c = _zstd.ZstdCompressor(level=_ZSTD_LEVEL)
        _tls.zstd_c = c
    return c


def _zstd_decompressor() -> "_zstd.ZstdDecompressor":
    d = getattr(_tls, "zstd_d", None)
    if d is None:
        d = _zstd.ZstdDecompressor()
        _tls.zstd_d = d
    return d


def codec_name(codec: int) -> str:
    """Return a human-readable codec name."""
    try:
        return _CODEC_NAMES[codec]
    except KeyError as exc:
        raise InvalidCodecError(f"Unknown codec id: {codec}") from exc


def default_codec() -> int:
    """
    Return the preferred default compression codec for this runtime.

    Preference order:
    1. zstd, if installed
    2. zlib, always available in stdlib
    """
    if _zstd is not None:
        return CODEC_ZSTD
    return CODEC_ZLIB


DEFAULT_CODEC: Final[int] = default_codec()


def compress_bytes(data: bytes, codec: int) -> bytes:
    """Compress bytes using the configured wire codec."""
    if codec == CODEC_NONE:
        return data

    if codec == CODEC_GZIP:
        return gzip.compress(data, compresslevel=_GZIP_COMPRESSLEVEL)

    if codec == CODEC_ZLIB:
        return zlib.compress(data, level=_ZLIB_LEVEL)

    if codec == CODEC_ZSTD:
        if _zstd is None:
            raise InvalidCodecError(
                "CODEC_ZSTD requested but zstandard is not installed"
            )
        return _zstd_compressor().compress(data)

    raise InvalidCodecError(f"Unknown codec id: {codec}")


def decompress_bytes(data: bytes, codec: int) -> bytes:
    """Decompress bytes using the configured wire codec."""
    if codec == CODEC_NONE:
        return data

    if codec == CODEC_GZIP:
        return gzip.decompress(data)

    if codec == CODEC_ZLIB:
        return zlib.decompress(data)

    if codec == CODEC_ZSTD:
        if _zstd is None:
            raise InvalidCodecError(
                "CODEC_ZSTD encountered but zstandard is not installed"
            )
        return _zstd_decompressor().decompress(data)

    raise InvalidCodecError(f"Unknown codec id: {codec}")