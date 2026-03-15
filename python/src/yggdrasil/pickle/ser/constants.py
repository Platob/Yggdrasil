from __future__ import annotations

__all__ = [
    "CODEC_NONE",
    "CODEC_GZIP",
    "CODEC_ZSTD",
    "CODEC_ZLIB",
    "HEADER_SIZE",
    "COMPRESS_THRESHOLD",
    "MAGIC",
    "FORMAT_VERSION"
]

FORMAT_VERSION: int = 1

CODEC_NONE: int = 0
CODEC_GZIP: int = 1
CODEC_ZSTD: int = 2
CODEC_ZLIB: int = 3

HEADER_SIZE: int = 12
COMPRESS_THRESHOLD: int = 512 * 1024

if FORMAT_VERSION == 1:
    MAGIC: bytes = b"YGG1"
else:
    raise ValueError(f"Unsupported format version: {FORMAT_VERSION}")