from __future__ import annotations

__all__ = [
    "CODEC_NONE",
    "CODEC_GZIP",
    "CODEC_ZSTD",
    "CODEC_ZLIB",
    "HEADER_SIZE",
    "COMPRESS_THRESHOLD",
    "MAGIC",
    "MAGIC_LENGTH",
    "is_valid_magic",
]

CODEC_NONE: int = 0
CODEC_GZIP: int = 1
CODEC_ZSTD: int = 2
CODEC_ZLIB: int = 3

HEADER_SIZE: int = 12
COMPRESS_THRESHOLD: int = 512 * 1024
MAGIC: bytes = b"YgD1"
MAGIC_LENGTH: int = len(MAGIC)
OLD_MAGIC: bytes = b"YGG1"


def is_valid_magic(value: bytes) -> bool:
    if value == MAGIC:
        return True
    if value == OLD_MAGIC:
        from yggdrasil.environ import PyEnv

        if PyEnv.in_databricks():
            raise RuntimeError(
                "Yggdrasil<0.6.6 cannot be used in databricks, "
                "update with uv pip install ygg[data,databricks,pickle]>=0.6.6"
            )
        return True
    return False