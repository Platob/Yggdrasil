"""Delta deletion vectors.

A deletion vector marks rows of a parquet file as logically deleted
without rewriting the file. Reads filter the marked rows; writes
emit a new DV (and a Remove+Add action pair on the same data file)
to record additional deletes.

Two layers
----------

1. **DeletionVectorDescriptor** — the JSON-serializable bookkeeping
   (storage type, path, offset, size, cardinality) that lives
   inside ``AddFile.deletionVector`` / ``RemoveFile.deletionVector``.
2. **DeletionVectorBlob** — the binary payload itself. May be
   stored inline (Z85-encoded inside the descriptor's
   ``pathOrInlineDv`` field for very small DVs) or in a separate
   ``.bin`` file under the table root, possibly sharing the file
   with other DVs (offset+size addressing).

Wire format of the DV blob
--------------------------

::

    +----------------+
    | u8  format     |  always 1 currently. Anything else → refuse.
    +----------------+
    | u32 LE size    |  size of the roaring payload that follows.
    +----------------+
    | bytes          |  roaring bitmap, "portable" 64-bit serialization.
    +----------------+
    | u32 LE crc?    |  optional CRC32 over the [size] bytes.
    +----------------+

The CRC's presence is signaled by the ``sizeInBytes`` field of the
descriptor: descriptor.size_in_bytes == 1 (format) + 4 (size) +
size + 4 (CRC) when CRC is present, vs 1 + 4 + size when not. We
use that to decide whether to verify a trailing CRC.

Roaring bitmap
--------------

We delegate to ``pyroaring.BitMap64`` (64-bit roaring) for decode
and encode — the bitmap can be large in practice (millions of
deleted ordinals across a long-lived table) and the serialization
format has corner cases we don't want to reimplement. This module
imports pyroaring lazily so non-DV code paths don't pay the import
cost.

Path-storage type ``u``
-----------------------

When a DV lives in a separate file, the descriptor's
``storageType="u"`` and ``pathOrInlineDv`` is a Z85-encoded UUID
followed by a "/relative-path" suffix. Reference implementations
use this to compactly identify shared-bin files. We support
reading them; on write we emit ``storageType="p"`` paths (plain
relative URI), which is also spec-conformant and simpler.

Storage types we recognize:

- ``u`` — UUID-derived shared file (read).
- ``p`` — plain relative path (read + write).
- ``i`` — inline Z85-encoded blob (read + write).

What this file does NOT do
--------------------------

- It does not own the lifecycle of ``.bin`` files. The
  IO/commit code creates and deletes them.
- It does not decide *when* to inline vs externalize. Heuristic
  lives at the call site: tiny DVs go inline.
"""

from __future__ import annotations

import dataclasses
import struct
import zlib
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Mapping

if TYPE_CHECKING:
    from pyroaring import BitMap64  # type: ignore[import-untyped]


__all__ = [
    "DeletionVectorDescriptor",
    "encode_dv_blob",
    "decode_dv_blob",
    "MAX_INLINE_DV_BYTES",
]


#: Maximum inline DV blob size in bytes. Above this, externalize to a
#: ``.bin`` file. Reference value matches what delta-rs uses; it's a
#: heuristic, not a spec mandate. Keeps inline blobs from bloating
#: AddFile actions.
MAX_INLINE_DV_BYTES: int = 32 * 1024  # 32 KiB

#: Format byte we emit and accept.
DV_FORMAT_VERSION: int = 1


# ---------------------------------------------------------------------------
# Z85 — a base85-variant Delta uses for inline DVs and 'u' paths
# ---------------------------------------------------------------------------

# Z85 alphabet, from the ZeroMQ spec, which Delta's spec references.
# 85 characters; the encoding encodes 4 bytes into 5 ASCII chars.
_Z85_ALPHABET: bytes = (
    b"0123456789"
    b"abcdefghij" b"klmnopqrst" b"uvwxyz"
    b"ABCDEFGHIJ" b"KLMNOPQRST" b"UVWXYZ"
    b".-:+=^!/*?" b"&<>()[]{}@" b"%$#"
)

_Z85_DECODE_MAP: dict[int, int] = {c: i for i, c in enumerate(_Z85_ALPHABET)}


def _z85_encode(data: bytes) -> str:
    """Encode bytes as Z85.

    Input length must be a multiple of 4. The DV spec pads inline
    payloads on encode and trims on decode to satisfy this.
    """
    if len(data) % 4 != 0:
        raise ValueError(
            f"Z85 input must be a multiple of 4 bytes; got {len(data)}."
        )
    out = bytearray()
    for i in range(0, len(data), 4):
        n = (
            (data[i] << 24) | (data[i + 1] << 16)
            | (data[i + 2] << 8) | data[i + 3]
        )
        chunk = bytearray(5)
        for j in range(4, -1, -1):
            chunk[j] = _Z85_ALPHABET[n % 85]
            n //= 85
        out.extend(chunk)
    return out.decode("ascii")


def _z85_decode(text: str) -> bytes:
    """Decode Z85 text to bytes.

    Strict: rejects characters outside the Z85 alphabet rather than
    silently producing wrong bytes.
    """
    encoded = text.encode("ascii")
    if len(encoded) % 5 != 0:
        raise ValueError(
            f"Z85 input length must be a multiple of 5; got {len(encoded)}."
        )
    out = bytearray()
    for i in range(0, len(encoded), 5):
        n = 0
        for j in range(5):
            ch = encoded[i + j]
            try:
                n = n * 85 + _Z85_DECODE_MAP[ch]
            except KeyError as exc:
                raise ValueError(
                    f"Invalid Z85 character {chr(ch)!r}."
                ) from exc
        if n > 0xFFFFFFFF:
            raise ValueError("Z85 group overflows 32 bits.")
        out.extend([
            (n >> 24) & 0xFF, (n >> 16) & 0xFF,
            (n >> 8) & 0xFF, n & 0xFF,
        ])
    return bytes(out)


# ---------------------------------------------------------------------------
# Descriptor — the JSON-side bookkeeping
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class DeletionVectorDescriptor:
    """The JSON-serialized DV bookkeeping inside an AddFile/RemoveFile.

    :attr:`storage_type` is one of:

    - ``"u"`` — UUID-keyed shared bin file. ``path_or_inline``
      starts with a 20-char Z85 UUID, optionally followed by a
      ``/relative-path`` suffix.
    - ``"p"`` — plain relative path. ``path_or_inline`` is a URI
      relative to the table root.
    - ``"i"`` — inline. ``path_or_inline`` is the Z85-encoded DV
      payload (already including the 1+4 framing header).

    :attr:`offset` is the offset within the .bin file at which the
    DV's framed payload starts. ``None`` for inline DVs.
    :attr:`size_in_bytes` is the framed DV size — that is, the
    number of bytes from the format byte through the optional
    trailing CRC. We use this to detect whether a CRC is present
    on read (size > 1+4+payload_size implies trailing CRC).
    :attr:`cardinality` is the number of deleted row ordinals.
    Spec-mandated; must match the bitmap's cardinality on decode
    (we verify).
    """

    storage_type: str
    path_or_inline: str
    size_in_bytes: int
    cardinality: int
    offset: int | None = None

    VALID_STORAGE_TYPES: ClassVar[frozenset[str]] = frozenset({"u", "p", "i"})

    def __post_init__(self) -> None:
        if self.storage_type not in self.VALID_STORAGE_TYPES:
            raise ValueError(
                f"Unknown DV storageType {self.storage_type!r}; "
                f"expected one of {sorted(self.VALID_STORAGE_TYPES)}."
            )
        if self.cardinality < 0:
            raise ValueError(
                f"DV cardinality must be >= 0; got {self.cardinality}."
            )
        if self.size_in_bytes <= 0:
            raise ValueError(
                f"DV sizeInBytes must be > 0; got {self.size_in_bytes}."
            )
        if self.storage_type == "i" and self.offset is not None:
            raise ValueError(
                "Inline DV must not carry an offset."
            )

    # ------------------------------------------------------------------
    # JSON I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "DeletionVectorDescriptor":
        return cls(
            storage_type=str(raw["storageType"]),
            path_or_inline=str(raw["pathOrInlineDv"]),
            size_in_bytes=int(raw["sizeInBytes"]),
            cardinality=int(raw["cardinality"]),
            offset=int(raw["offset"]) if raw.get("offset") is not None else None,
        )

    def to_json(self) -> Mapping[str, Any]:
        out: dict[str, Any] = {
            "storageType": self.storage_type,
            "pathOrInlineDv": self.path_or_inline,
            "sizeInBytes": self.size_in_bytes,
            "cardinality": self.cardinality,
        }
        if self.offset is not None:
            out["offset"] = self.offset
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_inline(self) -> bool:
        return self.storage_type == "i"

    @property
    def is_empty(self) -> bool:
        return self.cardinality == 0

    def relative_path(self) -> str | None:
        """The relative path to the .bin file, for non-inline DVs.

        For ``storage_type="p"``, this is just :attr:`path_or_inline`
        (already a relative path). For ``"u"``, the format is
        ``<z85-uuid>[/path]``; we return the encoded form unchanged
        because resolving it requires knowing the writer's directory
        scheme. The ``replay`` module handles ``u`` resolution.
        """
        if self.storage_type == "p":
            return self.path_or_inline
        if self.storage_type == "u":
            return self.path_or_inline
        return None


# ---------------------------------------------------------------------------
# Binary blob: framing + roaring
# ---------------------------------------------------------------------------


def decode_dv_blob(
    blob: bytes,
    *,
    expected_cardinality: int,
    has_crc: bool,
) -> "BitMap64":
    """Parse a framed DV blob into a 64-bit roaring bitmap.

    :param blob: bytes from offset for ``size_in_bytes`` bytes —
        i.e. one DV's framed payload, excluding any neighbours in
        a shared .bin file.
    :param expected_cardinality: declared by the descriptor; we
        verify that the decoded bitmap's cardinality matches.
        Mismatch is corruption — refuse rather than read silently.
    :param has_crc: whether a trailing 4-byte CRC32 is present.
        Caller derives this from
        ``size_in_bytes vs 1 + 4 + payload_size``.
    """
    if len(blob) < 5:
        raise ValueError(
            f"DV blob too short: {len(blob)} bytes; need at least 5 "
            "for format byte + size."
        )

    fmt = blob[0]
    if fmt != DV_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported DV format byte {fmt}; only version "
            f"{DV_FORMAT_VERSION} is supported."
        )

    (payload_size,) = struct.unpack_from("<I", blob, 1)
    payload_start = 5
    payload_end = payload_start + payload_size

    expected_total = payload_end + (4 if has_crc else 0)
    if len(blob) != expected_total:
        raise ValueError(
            f"DV blob size mismatch: header declares {payload_size}-byte "
            f"payload (+CRC={has_crc}), expected total {expected_total}, "
            f"got {len(blob)}."
        )

    payload = blob[payload_start:payload_end]

    if has_crc:
        (declared_crc,) = struct.unpack_from("<I", blob, payload_end)
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if declared_crc != actual_crc:
            raise ValueError(
                f"DV blob CRC mismatch: declared {declared_crc:#010x}, "
                f"computed {actual_crc:#010x}."
            )

    bitmap = _decode_roaring(payload)

    actual_cardinality = len(bitmap)
    if actual_cardinality != expected_cardinality:
        raise ValueError(
            f"DV cardinality mismatch: declared {expected_cardinality}, "
            f"bitmap has {actual_cardinality}."
        )

    return bitmap


def encode_dv_blob(
    bitmap: "BitMap64",
    *,
    include_crc: bool = True,
) -> bytes:
    """Frame a roaring bitmap as a DV blob.

    Default ``include_crc=True`` — defense in depth. The cost is
    4 bytes per blob and one CRC32 pass; readers that don't expect
    a CRC won't see it (we set ``size_in_bytes`` accordingly so
    the descriptor signals the framing).
    """
    payload = _encode_roaring(bitmap)
    body = bytes([DV_FORMAT_VERSION]) + struct.pack("<I", len(payload)) + payload
    if include_crc:
        body += struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF)
    return body


def make_inline_descriptor(bitmap: "BitMap64") -> DeletionVectorDescriptor:
    """Build a ``storageType="i"`` descriptor for *bitmap*.

    Z85 requires 4-byte input multiples; we right-pad the framed
    blob to the next multiple of 4 and record the original length
    in ``size_in_bytes`` so readers know where the real bytes end.
    The pad bytes are ignored on decode.
    """
    framed = encode_dv_blob(bitmap, include_crc=True)
    pad = (-len(framed)) % 4
    padded = framed + b"\x00" * pad
    encoded = _z85_encode(padded)
    return DeletionVectorDescriptor(
        storage_type="i",
        path_or_inline=encoded,
        size_in_bytes=len(framed),
        cardinality=len(bitmap),
        offset=None,
    )


def decode_inline_descriptor(
    descriptor: DeletionVectorDescriptor,
) -> "BitMap64":
    """Decode the bitmap from an inline DV descriptor."""
    if descriptor.storage_type != "i":
        raise ValueError(
            f"decode_inline_descriptor called on non-inline descriptor "
            f"(storageType={descriptor.storage_type!r})."
        )

    raw = _z85_decode(descriptor.path_or_inline)
    # raw is the padded framed blob; trim back to size_in_bytes.
    if descriptor.size_in_bytes > len(raw):
        raise ValueError(
            f"Inline DV size_in_bytes={descriptor.size_in_bytes} exceeds "
            f"decoded length {len(raw)}."
        )
    framed = raw[:descriptor.size_in_bytes]

    # Has-CRC determination: framed length vs declared payload size.
    if len(framed) < 5:
        raise ValueError("Inline DV framed blob too short.")
    (payload_size,) = struct.unpack_from("<I", framed, 1)
    has_crc = len(framed) == 5 + payload_size + 4

    return decode_dv_blob(
        framed,
        expected_cardinality=descriptor.cardinality,
        has_crc=has_crc,
    )


# ---------------------------------------------------------------------------
# Roaring bitmap delegation
# ---------------------------------------------------------------------------


def _decode_roaring(payload: bytes) -> "BitMap64":
    """Decode the roaring portable serialization to a BitMap64.

    pyroaring's ``BitMap64.deserialize`` accepts the same byte
    format Delta uses. Wrapping it here gives us one place to
    upgrade if the underlying library API shifts.
    """
    BitMap64 = _bitmap_class()
    try:
        return BitMap64.deserialize(payload)
    except Exception as exc:
        raise ValueError(
            f"Failed to decode roaring bitmap payload "
            f"({len(payload)} bytes): {exc!r}."
        ) from exc


def _encode_roaring(bitmap: "BitMap64") -> bytes:
    """Encode a BitMap64 as bytes."""
    serialize = getattr(bitmap, "serialize", None)
    if serialize is None:
        raise TypeError(
            f"Object {bitmap!r} is not a pyroaring BitMap64; cannot serialize."
        )
    return bytes(serialize())


def _bitmap_class():
    """Lazy-import pyroaring.BitMap64.

    Keeps the import out of module load — non-DV reads / writes
    don't need pyroaring at all. Raises a clear error if the
    package isn't installed.
    """
    try:
        from pyroaring import BitMap64
    except ImportError as exc:
        raise ImportError(
            "DeltaIO deletion-vector support requires the 'pyroaring' "
            "package. Install it with `pip install pyroaring`."
        ) from exc
    return BitMap64


def empty_bitmap() -> "BitMap64":
    """Return a fresh empty BitMap64. Public for callers building DVs."""
    return _bitmap_class()()


def bitmap_from_iter(ordinals: Iterable[int]) -> "BitMap64":
    """Build a BitMap64 from an iterable of row ordinals.

    Convenience for the merge path: collect deleted ordinals,
    call this, hand the result to :func:`encode_dv_blob`.
    """
    BitMap64 = _bitmap_class()
    return BitMap64(ordinals)
