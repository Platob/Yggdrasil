from __future__ import annotations

import gzip as _gzip
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator

from yggdrasil.io import BytesIO

from .registry import REGISTRY

try:
    import zstandard as _zstd
except ImportError:  # pragma: no cover
    _zstd = None  # type: ignore[assignment]

__all__ = [
    "Serialized",
    "PrimitiveSerialized",
    "LogicalSerialized",
    "NestedScalar",
    "ArraySerialized",
    "MapSerialized",
    "_COMPRESS_THRESHOLD"
]

CODEC_NONE: int = 0
CODEC_GZIP: int = 1
CODEC_ZSTD: int = 2

_COMPRESS_THRESHOLD: int = 512 * 1024  # 512 KiB


@dataclass(frozen=True, slots=True)
class Serialized(ABC):
    metadata: dict[bytes, bytes]
    data: BytesIO
    size: int
    start_index: int = 0
    codec: int = 0  # 0 = none, 1 = gzip, 2 = zstandard

    TAG: ClassVar[int]
    _VALID_CODECS: ClassVar[frozenset[int]] = frozenset({0, 1, 2})

    def __post_init__(self) -> None:
        if not isinstance(self.data, BytesIO):
            object.__setattr__(self, "data", BytesIO.parse(self.data))

        if self.size < 0:
            raise ValueError(f"size must be >= 0, got {self.size}")

        if self.start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {self.start_index}")

        if self.codec not in self._VALID_CODECS:
            raise ValueError(
                f"codec must be one of {sorted(self._VALID_CODECS)}, got {self.codec}"
            )

    def __init_subclass__(cls, **kwargs):
        # Use explicit super target to avoid zero-arg super() issues with
        # dataclass(slots=True) class rewriting on some Python versions.
        super(Serialized, cls).__init_subclass__(**kwargs)
        tag = getattr(cls, "TAG", None)
        if isinstance(tag, int):
            REGISTRY.register_tag(tag, cls)

    @property
    def tag(self) -> int:
        return self.TAG

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"tag={self.tag}, size={self.size}, start_index={self.start_index}, "
            f"metadata={self.metadata})"
        )

    def __str__(self) -> str:
        return repr(self)

    def pread(self, n: int, pos: int = 0) -> bytes:
        return self.data.pread(n, self.start_index + pos)

    def payload(self) -> bytes:
        raw = self.data.pread(self.size, self.start_index)

        if self.codec == CODEC_NONE:
            return raw
        return _decompress(raw, self.codec)

    def pwrite(self, mv: memoryview, pos: int = 0) -> int:
        return self.data.pwrite(mv, self.start_index + pos)

    def bwrite(self, buffer: BytesIO) -> int:
        meta = BytesIO()
        meta.write(len(self.metadata).to_bytes(2, "big"))
        for k, v in self.metadata.items():
            meta.write(len(k).to_bytes(2, "big"))
            meta.write(len(v).to_bytes(4, "big"))
            meta.write(k)
            meta.write(v)

        meta_bytes = meta.to_bytes()

        buffer.write(bytes((self.tag, self.codec)))
        buffer.write(self.size.to_bytes(4, "big"))
        buffer.write(len(meta_bytes).to_bytes(4, "big"))
        buffer.write(meta_bytes)
        buffer.write(self.data.pread(self.size, self.start_index))

        return 1 + 1 + 4 + 4 + len(meta_bytes) + self.size

    def to_bytes(self) -> bytes:
        buffer = BytesIO()
        self.bwrite(buffer)
        return buffer.to_bytes()

    @classmethod
    def from_raw(
        cls,
        raw: bytes,
        *,
        metadata: dict[bytes, bytes] | None = None,
    ) -> "Serialized":
        buf = BytesIO(raw)
        return cls(
            metadata={} if metadata is None else dict(metadata),
            data=buf,
            size=len(raw),
            start_index=0,
        )

    @property
    @abstractmethod
    def value(self) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "Serialized":
        raise NotImplementedError

    @classmethod
    def _read_metadata(cls, raw: bytes) -> dict[bytes, bytes]:
        n = int.from_bytes(raw, "big")
        metadata: dict[bytes, bytes] = {}
        for _ in range(n):
            klen = int.from_bytes(buf.read(2), "big")
            vlen = int.from_bytes(buf.read(4), "big")
            metadata[buf.read(klen)] = buf.read(vlen)
        return metadata

    @classmethod
    def pread_from(cls, buffer: BytesIO, pos: int) -> tuple["Serialized", int]:
        n = 10 if cls.COMPRESSIBLE else 9
        head = buffer.pread(n, pos)
        tag = head[0]
        codec = head[1]
        size = int.from_bytes(head[2:6], "big")
        meta_size = int.from_bytes(head[6:10], "big")

        meta_pos = pos + 10
        meta_raw = buffer.pread(meta_size, meta_pos)
        if len(meta_raw) != meta_size:
            raise EOFError(f"Expected {meta_size} metadata bytes, got {len(meta_raw)}")

        payload_pos = meta_pos + meta_size
        payload = buffer.pread(size, payload_pos)
        if len(payload) != size:
            raise EOFError(f"Expected {size} payload bytes, got {len(payload)}")

        subcls = REGISTRY.get_by_tag(tag)
        obj = subcls(
            metadata=cls._read_metadata(meta_raw),
            data=buffer,
            size=size,
            start_index=payload_pos,
            codec=codec,
        )
        next_pos = payload_pos + size
        return obj, next_pos

    @classmethod
    def pread(cls, buffer: BytesIO) -> "Serialized":
        obj, _ = cls.pread_from(buffer, buffer.tell())
        return obj

    @classmethod
    def from_python(
        cls,
        obj: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "Serialized":
        subcls = REGISTRY.get_by_python_value(obj)
        return subcls.from_value(
            obj,
            payload=payload, metadata=metadata, byte_limit=byte_limit
        )

    @classmethod
    def _maybe_compress(
        cls,
        data: BytesIO,
        size: int,
        start_index: int,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> tuple[BytesIO, int, int, int]:
        """Compress payload in-place if above *byte_limit*.

        Returns ``(data, size, start_index, codec)`` ready for the
        constructor.  When *byte_limit* is ``None`` compression is
        disabled.  When no compression is applied ``codec`` is
        ``CODEC_NONE`` and the other values are returned unchanged.
        """
        if byte_limit is None or size <= byte_limit:
            return data, size, start_index, CODEC_NONE

        raw = data.pread(size, start_index)
        compressed = _compress(raw, CODEC_ZSTD)

        # Only use compressed form if it actually shrinks the data.
        if len(compressed) >= size:
            return data, size, start_index, CODEC_NONE

        buf = BytesIO(compressed)
        return buf, len(compressed), 0, CODEC_ZSTD


def _compress(raw: bytes, codec: int) -> bytes:
    """Compress *raw* bytes with the given codec."""
    if codec == CODEC_GZIP:
        return _gzip.compress(raw)
    if codec == CODEC_ZSTD:
        global _zstd
        if _zstd is None:
            from yggdrasil.environ import runtime_import_module
            _zstd = runtime_import_module("zstandard", pip_name="zstandard", install=True)
        return _zstd.ZstdCompressor().compress(raw)
    raise ValueError(f"Unknown compression codec: {codec}")


def _decompress(raw: bytes, codec: int) -> bytes:
    """Decompress *raw* bytes with the given codec."""
    if codec == CODEC_GZIP:
        return _gzip.decompress(raw)
    if codec == CODEC_ZSTD:
        global _zstd
        if _zstd is None:
            from yggdrasil.environ import runtime_import_module
            _zstd = runtime_import_module("zstandard", pip_name="zstandard", install=True)
        return _zstd.ZstdDecompressor().decompress(raw)
    raise ValueError(f"Unknown decompression codec: {codec}")


@dataclass(frozen=True, slots=True)
class PrimitiveSerialized(Serialized, ABC):
    pass


@dataclass(frozen=True, slots=True)
class LogicalSerialized(Serialized, ABC):
    inner: Serialized = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super(LogicalSerialized, self).__post_init__()
        object.__setattr__(self, "inner", self._build_inner())

    @abstractmethod
    def _build_inner(self) -> Serialized:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class NestedScalar(LogicalSerialized, ABC):
    pass


@dataclass(frozen=True, slots=True)
class ArraySerialized(Serialized, ABC):
    def iter_(self) -> Iterator["Serialized"]:
        """Iterate child serialized items stored in this payload."""
        if self.codec != CODEC_NONE:
            # Decompress into a temporary buffer and iterate over that.
            raw = self.payload()
            buf = BytesIO(raw)
            pos = 0
            end = len(raw)
        else:
            buf = self.data
            pos = self.start_index
            end = self.start_index + self.size

        while pos < end:
            item, pos = Serialized.pread_from(buf, pos)
            yield item

        if pos != end:
            raise ValueError("Array payload parsing did not consume exact payload size")


@dataclass(frozen=True, slots=True)
class MapSerialized(ArraySerialized, ABC):
    pass

