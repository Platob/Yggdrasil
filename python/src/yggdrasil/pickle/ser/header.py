from __future__ import annotations

from typing import Mapping

from yggdrasil.io import BytesIO
from yggdrasil.io.buffer.bytes_view import BytesIOView
from yggdrasil.pickle.ser.constants import HEADER_SIZE
from yggdrasil.pickle.ser.errors import HeaderDecodeError, MetadataDecodeError

__all__ = [
    "Metadata",
    "decode_metadata",
    "encode_metadata",
    "Header",
]

Metadata = dict[bytes, bytes] | None
MAX_META_SIZE = 128 * 1024 * 1024
MAX_PAYLOAD_SIZE = 2 * 1024 * 1024 * 1024
MAX_METADATA_ENTRIES = 10_000
MAX_METADATA_KEY_SIZE = 64 * 1024
MAX_METADATA_VALUE_SIZE = 16 * 1024 * 1024

def encode_metadata(metadata: Mapping[bytes, bytes] | None) -> bytes:
    """Encode metadata as length-prefixed key/value pairs.

    Format:
        [k_len:u32][key][v_len:u32][value]...
    """
    if not metadata:
        return b""

    out = bytearray()
    for key, value in metadata.items():
        if not isinstance(key, bytes):
            raise TypeError(f"Metadata keys must be bytes, got {type(key)!r}")
        if not isinstance(value, bytes):
            raise TypeError(f"Metadata values must be bytes, got {type(value)!r}")

        out += len(key).to_bytes(4, "big")
        out += key
        out += len(value).to_bytes(4, "big")
        out += value
    return bytes(out)


def decode_metadata(blob: bytes) -> Metadata:
    if not blob:
        return None

    pos = 0
    size = len(blob)
    out: dict[bytes, bytes] = {}
    entries = 0

    while pos < size:
        entries += 1
        if entries > MAX_METADATA_ENTRIES:
            raise MetadataDecodeError("Too many metadata entries")

        if pos + 4 > size:
            raise MetadataDecodeError("Unexpected EOF while reading metadata key length")
        k_len = int.from_bytes(blob[pos : pos + 4], "big")
        pos += 4

        if k_len > MAX_METADATA_KEY_SIZE:
            raise MetadataDecodeError(f"Metadata key too large: {k_len}")

        if pos + k_len > size:
            raise MetadataDecodeError("Unexpected EOF while reading metadata key")
        key = blob[pos : pos + k_len]
        pos += k_len

        if pos + 4 > size:
            raise MetadataDecodeError("Unexpected EOF while reading metadata value length")
        v_len = int.from_bytes(blob[pos : pos + 4], "big")
        pos += 4

        if v_len > MAX_METADATA_VALUE_SIZE:
            raise MetadataDecodeError(f"Metadata value too large: {v_len}")

        if pos + v_len > size:
            raise MetadataDecodeError("Unexpected EOF while reading metadata value")
        value = blob[pos : pos + v_len]
        pos += v_len

        out[key] = value

    return out


class Header:
    """Binary header for a serialized payload.

    Layout:
        tag:u16
        codec:u16
        size:u32
        meta_size:u32
        metadata:meta_size bytes
        payload:size bytes
    """

    __slots__ = ("tag", "codec", "size", "meta_size", "start", "metadata")

    def __init__(
        self,
        tag: int,
        codec: int,
        size: int,
        meta_size: int,
        start: int,
        metadata: Metadata = None,
    ) -> None:
        self.tag = tag
        self.codec = codec
        self.size = size
        self.meta_size = meta_size
        self.start = start
        self.metadata = metadata

    @property
    def header_start(self) -> int:
        return self.start - HEADER_SIZE - self.meta_size

    @property
    def payload_end(self) -> int:
        return self.start + self.size

    @classmethod
    def build(
        cls,
        *,
        tag: int,
        codec: int,
        size: int,
        metadata: Mapping[bytes, bytes] | None = None,
        start: int = 0,
    ) -> "Header":
        """Build a header from semantic fields."""
        encoded = encode_metadata(metadata)
        return cls(
            tag=tag,
            codec=codec,
            size=size,
            meta_size=len(encoded),
            start=start + HEADER_SIZE + len(encoded),
            metadata=dict(metadata) if metadata else None,
        )

    @classmethod
    def read_from(
        cls,
        buffer: BytesIO,
        *,
        pos: int | None = None,
    ) -> "Header":
        """Parse a header from a buffer without materializing the payload."""
        if pos is None:
            pos = buffer.tell()

        fixed = buffer.pread(HEADER_SIZE, pos=pos)
        if len(fixed) != HEADER_SIZE:
            raise HeaderDecodeError(
                f"Expected {HEADER_SIZE} header bytes at pos={pos}, got {len(fixed)}"
            )

        tag = int.from_bytes(fixed[0:2], "big")
        codec = int.from_bytes(fixed[2:4], "big")
        size = int.from_bytes(fixed[4:8], "big")
        meta_size = int.from_bytes(fixed[8:12], "big")

        if meta_size > MAX_META_SIZE:
            raise HeaderDecodeError(f"Metadata too large: {meta_size}")

        if size > MAX_PAYLOAD_SIZE:
            raise HeaderDecodeError(f"Payload too large: {size}")

        meta_blob = b""
        if meta_size:
            meta_blob = buffer.pread(meta_size, pos=pos + HEADER_SIZE)
            if len(meta_blob) != meta_size:
                raise HeaderDecodeError(
                    f"Expected {meta_size} metadata bytes at pos={pos + HEADER_SIZE}, "
                    f"got {len(meta_blob)}"
                )

        metadata = decode_metadata(meta_blob)
        return cls(
            tag=tag,
            codec=codec,
            size=size,
            meta_size=meta_size,
            start=pos + HEADER_SIZE + meta_size,
            metadata=metadata,
        )

    def payload_view(self, buffer: BytesIO) -> BytesIOView:
        """Return a zero-copy view of the payload bytes."""
        return buffer.view(pos=self.start, size=self.size)

    def write_to(
        self,
        data: bytes | bytearray | memoryview,
        *,
        buffer: BytesIO | None = None,
    ) -> BytesIO:
        """Write header + payload into a buffer and return that buffer."""
        if buffer is None:
            buffer = BytesIO()

        payload = memoryview(data)
        if len(payload) != self.size:
            raise ValueError(
                f"Payload size mismatch: header.size={self.size}, actual={len(payload)}"
            )

        metadata_blob = encode_metadata(self.metadata)
        if len(metadata_blob) != self.meta_size:
            raise ValueError(
                f"Metadata size mismatch: header.meta_size={self.meta_size}, "
                f"actual={len(metadata_blob)}"
            )

        fixed = bytearray(HEADER_SIZE)
        fixed[0:2] = self.tag.to_bytes(2, "big")
        fixed[2:4] = self.codec.to_bytes(2, "big")
        fixed[4:8] = self.size.to_bytes(4, "big")
        fixed[8:12] = self.meta_size.to_bytes(4, "big")

        buffer.write_bytes(fixed)
        if metadata_blob:
            buffer.write_bytes(metadata_blob)
        buffer.write_bytes(payload)
        return buffer