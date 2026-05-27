"""Deletion-vector encode/decode + Arrow batch masking.

A Delta deletion vector is a serialised roaring bitmap whose bits mark
the **logical row indices** of a parquet file that have been deleted.

Three storage shapes
--------------------

- ``"i"`` (inline) — Z85-encoded directly in the descriptor.
- ``"u"`` (UUID sidecar) — ``<table>/deletion_vector_<uuid>.bin``.
- ``"p"`` (absolute path) — table-relative path.

Bitmap layout
-------------

Two envelope shapes:

- Magic 0x01 prefix + 64-bit Roaring ("RoaringBitmapArray").
- Magic 0x00 prefix + simple list of 64-bit row ids.

Read path supports both + bare portable Roaring (no magic).
Write path emits Roaring for large DVs and simple-list for small ones.
"""

from __future__ import annotations

import base64
import dataclasses
import struct
import uuid as _uuid
from typing import TYPE_CHECKING, Iterable, List, Optional, Set

from yggdrasil.io.nested.delta.protocol import DeletionVectorDescriptor

if TYPE_CHECKING:
    import pyarrow as pa
    from yggdrasil.path import Path


__all__ = [
    "DeletionVector",
    "DeletionVectorDescriptor",
    "decode_deletion_vector",
    "encode_inline_deletion_vector",
    "write_uuid_deletion_vector",
    "mask_batch_with_dv",
]


# ---------------------------------------------------------------------------
# Z85 / base85 inline coding
# ---------------------------------------------------------------------------

_Z85_ALPHABET = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ".-:+=^!/*?&<>()[]{}@%$#"
)
_B85_ALPHABET = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "!#$%&()*+-;<=>?@^_`{|}~"
)
_Z85_TO_B85 = bytes.maketrans(
    _Z85_ALPHABET.encode("ascii"),
    _B85_ALPHABET.encode("ascii"),
)
_B85_TO_Z85 = bytes.maketrans(
    _B85_ALPHABET.encode("ascii"),
    _Z85_ALPHABET.encode("ascii"),
)


def _z85_decode(data: str) -> bytes:
    if not data:
        return b""
    encoded = data.encode("ascii").translate(_Z85_TO_B85)
    return base64.b85decode(encoded)


def _z85_encode(data: bytes) -> str:
    if not data:
        return ""
    if len(data) % 4 != 0:
        raise ValueError(
            f"_z85_encode requires 4-byte aligned input; got len={len(data)}."
        )
    encoded = base64.b85encode(data).translate(_B85_TO_Z85)
    return encoded.decode("ascii")


# ---------------------------------------------------------------------------
# Roaring-bitmap decode
# ---------------------------------------------------------------------------

_MAGIC_ROARING_64 = 1681511377  # 0x6439D3D1
_MAGIC_SIMPLE = 1681511376  # 0x6439D3D0


def _read_portable_roaring(buf: memoryview, pos: int) -> "tuple[Set[int], int]":
    cookie = struct.unpack_from("<I", buf, pos)[0]
    pos += 4

    # Portable Roaring cookies:
    #   SERIAL_COOKIE = 12347 (0x303B) — has run containers
    #   SERIAL_COOKIE_NO_RUNCONTAINER = 12346 (0x303A) — no runs
    # Run-flag cookie: low 16 bits = 12347, high 16 = (n_containers - 1)
    if (cookie & 0xFFFF) == 12347:  # SERIAL_COOKIE
        n_containers = ((cookie >> 16) & 0xFFFF) + 1
        bitmap_of_runs_size = (n_containers + 7) // 8
        run_flag_bytes = bytes(buf[pos : pos + bitmap_of_runs_size])
        pos += bitmap_of_runs_size
        has_run_flag = True
    else:
        n_containers = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        run_flag_bytes = b""
        has_run_flag = False

    key_card: list[tuple[int, int]] = []
    for _ in range(n_containers):
        key, card_minus_one = struct.unpack_from("<HH", buf, pos)
        pos += 4
        key_card.append((key, card_minus_one + 1))

    # Offset table: present for BOTH cookie types when n >= 4.
    # Per the Roaring spec, the offset table is always emitted when
    # the container count reaches 4, regardless of cookie kind.
    if n_containers >= 4:
        pos += 4 * n_containers

    out: Set[int] = set()
    for idx, (key, card) in enumerate(key_card):
        high = key << 16
        is_run = (
            bool(run_flag_bytes[idx >> 3] & (1 << (idx & 7))) if has_run_flag else False
        )

        if is_run:
            n_runs = struct.unpack_from("<H", buf, pos)[0]
            pos += 2
            for _ in range(n_runs):
                start, length = struct.unpack_from("<HH", buf, pos)
                pos += 4
                for v in range(start, start + length + 1):
                    out.add(high | v)
        elif card <= 4096:
            for _ in range(card):
                v = struct.unpack_from("<H", buf, pos)[0]
                pos += 2
                out.add(high | v)
        else:
            for word_idx in range(1024):
                word = struct.unpack_from("<Q", buf, pos)[0]
                pos += 8
                if word == 0:
                    continue
                base = word_idx << 6
                bit = 0
                while word:
                    if word & 1:
                        out.add(high | (base + bit))
                    word >>= 1
                    bit += 1
    return out, pos


def _decode_payload(payload: bytes) -> Set[int]:
    if len(payload) < 4:
        return set()
    mv = memoryview(payload)
    magic = struct.unpack_from("<I", mv, 0)[0]
    pos = 4

    if magic == _MAGIC_SIMPLE:
        count = struct.unpack_from("<Q", mv, pos)[0]
        pos += 8
        return {struct.unpack_from("<Q", mv, pos + i * 8)[0] for i in range(count)}

    if magic == _MAGIC_ROARING_64:
        n_chunks = struct.unpack_from("<Q", mv, pos)[0]
        pos += 8
        out: Set[int] = set()
        for _ in range(n_chunks):
            high_key = struct.unpack_from("<I", mv, pos)[0]
            pos += 4
            sub, pos = _read_portable_roaring(mv, pos)
            high = high_key << 32
            out.update(high | x for x in sub)
        return out

    # Bare portable Roaring (no 64-bit envelope): check for standard cookies
    cookie_low = magic & 0xFFFF
    if cookie_low in (12346, 12347):  # SERIAL_COOKIE_NO_RUNCONTAINER, SERIAL_COOKIE
        sub, _ = _read_portable_roaring(mv, 0)
        return sub

    return set()


# ---------------------------------------------------------------------------
# Roaring-bitmap encode
# ---------------------------------------------------------------------------


def _encode_roaring_payload(row_ids: Iterable[int]) -> bytes:
    """Encode row IDs into a 64-bit Roaring bitmap envelope.

    Produces the RoaringBitmapArray format (magic 0x64426152) that both
    Spark and Delta readers understand.
    """
    rows = sorted(set(int(r) for r in row_ids))
    if not rows:
        return struct.pack("<I", _MAGIC_SIMPLE) + struct.pack("<Q", 0)

    # Group by high 32 bits
    chunks: dict[int, list[int]] = {}
    for r in rows:
        hi = (r >> 32) & 0xFFFFFFFF
        lo = r & 0xFFFFFFFF
        chunks.setdefault(hi, []).append(lo)

    buf = bytearray()
    buf += struct.pack("<I", _MAGIC_ROARING_64)
    buf += struct.pack("<Q", len(chunks))

    for hi_key in sorted(chunks):
        buf += struct.pack("<I", hi_key)
        lo_values = sorted(chunks[hi_key])
        buf += _encode_portable_roaring(lo_values)

    return bytes(buf)


def _encode_portable_roaring(values: list[int]) -> bytes:
    """Encode a set of 32-bit values into a portable Roaring bitmap."""
    containers: dict[int, list[int]] = {}
    for v in values:
        key = (v >> 16) & 0xFFFF
        low = v & 0xFFFF
        containers.setdefault(key, []).append(low)

    n_containers = len(containers)
    sorted_keys = sorted(containers)

    # SERIAL_COOKIE_NO_RUNCONTAINER = 12346 (0x303A)
    # Standard portable Roaring cookie for bitmaps without run containers.
    # Followed by 4-byte container count. Offset table present when n >= 4.
    buf = bytearray()
    buf += struct.pack("<I", 12346)
    buf += struct.pack("<I", n_containers)

    for key in sorted_keys:
        card = len(containers[key])
        buf += struct.pack("<HH", key, card - 1)

    # Pre-build container data to know offsets
    container_chunks: list[bytes] = []
    for key in sorted_keys:
        vals = sorted(containers[key])
        card = len(vals)
        chunk = bytearray()
        if card <= 4096:
            for v in vals:
                chunk += struct.pack("<H", v)
        else:
            bitmap = [0] * 1024
            for v in vals:
                word_idx = v >> 6
                bit = v & 63
                bitmap[word_idx] |= (1 << bit)
            for word in bitmap:
                chunk += struct.pack("<Q", word)
        container_chunks.append(bytes(chunk))

    # Offset table: absolute byte offsets from stream start, per Roaring spec.
    # Header size = 4 (cookie) + 4 (count) + 4*n (key-card pairs) + 4*n (offsets)
    if n_containers >= 4:
        header_size = 4 + 4 + 4 * n_containers + 4 * n_containers
        cumulative = 0
        for chunk in container_chunks:
            buf += struct.pack("<I", header_size + cumulative)
            cumulative += len(chunk)

    for chunk in container_chunks:
        buf += chunk
    return bytes(buf)


# ---------------------------------------------------------------------------
# Encode — choose between simple and roaring based on cardinality
# ---------------------------------------------------------------------------

# Delta spec guidance: simple-list for small DVs, Roaring above one
# array container (4096 values). Matches Spark's DeletionVectorStore.
_ROARING_THRESHOLD = 4096


def _encode_simple_payload(row_ids: Iterable[int]) -> bytes:
    rows = sorted(set(int(r) for r in row_ids))
    body = struct.pack("<I", _MAGIC_SIMPLE) + struct.pack("<Q", len(rows))
    body += b"".join(struct.pack("<Q", r) for r in rows)
    return body


def _encode_dv_payload(row_ids: Iterable[int]) -> bytes:
    """Pick the best encoding for the given row IDs."""
    rows = sorted(set(int(r) for r in row_ids))
    if len(rows) <= _ROARING_THRESHOLD:
        return _encode_simple_payload(rows)
    return _encode_roaring_payload(rows)


def encode_inline_deletion_vector(
    row_ids: Iterable[int],
) -> DeletionVectorDescriptor:
    payload = _encode_dv_payload(row_ids)
    encoded = _z85_encode(payload)
    return DeletionVectorDescriptor(
        storage_type="i",
        path_or_inline_dv=encoded,
        size_in_bytes=len(payload),
        cardinality=_count_rows(row_ids),
    )


def write_uuid_deletion_vector(
    row_ids: Iterable[int],
    *,
    table_root: "Path",
) -> DeletionVectorDescriptor:
    payload = _encode_dv_payload(row_ids)
    framed = struct.pack(">I", len(payload)) + payload + struct.pack(">I", 0)

    uid = _uuid.uuid4().hex
    sidecar = table_root / f"deletion_vector_{uid}.bin"
    with sidecar.open("wb") as bio:
        bio.truncate(0)
        bio.write_bytes(framed)

    return DeletionVectorDescriptor(
        storage_type="u",
        path_or_inline_dv=uid,
        size_in_bytes=len(payload),
        cardinality=_count_rows(row_ids),
        offset=0,
    )


def _count_rows(row_ids: Iterable[int]) -> int:
    if isinstance(row_ids, (set, frozenset)):
        return len(row_ids)
    return len(set(row_ids))


def _count_from_payload(payload: bytes) -> int:
    if len(payload) < 12:
        return 0
    magic = struct.unpack_from("<I", payload, 0)[0]
    if magic != _MAGIC_SIMPLE:
        return 0
    return int(struct.unpack_from("<Q", payload, 4)[0])


# ---------------------------------------------------------------------------
# Sidecar fetch
# ---------------------------------------------------------------------------


def _read_sidecar_window(
    sidecar: "Path",
    offset: int,
    size: int,
) -> bytes:
    with sidecar.open("rb") as bio:
        bio.seek(offset)
        raw = bio.read(size + 8)
    if not raw:
        return b""

    if len(raw) >= 4:
        framed_size = struct.unpack(">I", raw[:4])[0]
        if 0 < framed_size <= size:
            return bytes(raw[4 : 4 + framed_size])
    return bytes(raw[:size])


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class DeletionVector:
    """Decoded deletion vector — the set of logical row indices to mask."""

    descriptor: DeletionVectorDescriptor
    deleted_rows: Set[int] = dataclasses.field(default_factory=set)

    def is_empty(self) -> bool:
        return not self.deleted_rows

    def __len__(self) -> int:
        return len(self.deleted_rows)

    def __contains__(self, row: int) -> bool:
        return row in self.deleted_rows

    def filter_indices(self, num_rows: int) -> List[int]:
        if not self.deleted_rows:
            return list(range(num_rows))
        deleted = self.deleted_rows
        return [i for i in range(num_rows) if i not in deleted]

    def merge(self, other: "DeletionVector") -> "DeletionVector":
        """Combine two DVs (union of deleted rows)."""
        return DeletionVector(
            descriptor=self.descriptor,
            deleted_rows=self.deleted_rows | other.deleted_rows,
        )


def decode_deletion_vector(
    descriptor: Optional[DeletionVectorDescriptor],
    *,
    table_root: "Path | None" = None,
    sidecar_cache: "dict[str, bytes] | None" = None,
) -> Optional[DeletionVector]:
    if descriptor is None:
        return None

    storage = (descriptor.storage_type or "").lower()

    if storage == "i":
        # Inline payloads are raw DV bytes (no int32 size + CRC framing).
        # The Z85-decoded bytes are the payload directly.
        try:
            raw = _z85_decode(descriptor.path_or_inline_dv)
        except Exception:
            return DeletionVector(descriptor=descriptor)
        return DeletionVector(
            descriptor=descriptor,
            deleted_rows=_decode_payload(raw),
        )

    if table_root is None:
        return DeletionVector(descriptor=descriptor)

    if storage in ("u", "p"):
        sidecar = _resolve_dv_sidecar(descriptor, table_root)
        if sidecar is None:
            return DeletionVector(descriptor=descriptor)

        offset = int(descriptor.offset or 0)
        size = int(descriptor.size_in_bytes or 0)
        cache_key = f"{sidecar.full_path()}|{offset}|{size}"
        cached: "bytes | None" = (
            sidecar_cache.get(cache_key) if sidecar_cache is not None else None
        )
        if cached is None:
            try:
                cached = _read_sidecar_window(sidecar, offset, size)
            except Exception:
                cached = b""
            if sidecar_cache is not None:
                sidecar_cache[cache_key] = cached
        return DeletionVector(
            descriptor=descriptor,
            deleted_rows=_decode_payload(cached),
        )

    return DeletionVector(descriptor=descriptor)


def _resolve_dv_sidecar(
    descriptor: DeletionVectorDescriptor,
    table_root: "Path",
) -> "Path | None":
    raw = descriptor.path_or_inline_dv or ""
    if not raw:
        return None

    storage = (descriptor.storage_type or "").lower()

    if storage == "p":
        return table_root / raw

    uuid_str = raw
    if len(uuid_str) > 0 and not (
        uuid_str[0].isalnum() and uuid_str[0] not in "ghijklmnopqrstuvwxyz"
    ):
        uuid_str = uuid_str[1:]
    name = f"deletion_vector_{uuid_str}.bin"
    return table_root / name


def mask_batch_with_dv(
    batch: "pa.RecordBatch",
    dv: Optional[DeletionVector],
    *,
    base_offset: int = 0,
) -> "pa.RecordBatch":
    """Drop deleted rows from a single Arrow record batch.

    Vectorised through numpy for O(|deleted|) cost regardless of
    batch size.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    if dv is None or dv.is_empty():
        return batch
    n = batch.num_rows
    if n == 0:
        return batch

    deleted = dv.deleted_rows
    if not deleted:
        return batch
    del_arr = np.fromiter(deleted, dtype=np.int64, count=len(deleted))
    rel = del_arr - np.int64(base_offset)
    in_range = (rel >= 0) & (rel < np.int64(n))
    if not in_range.any():
        return batch
    rel = rel[in_range]

    keep = np.ones(n, dtype=bool)
    keep[rel] = False
    if keep.all():
        return batch
    if not keep.any():
        return batch.slice(0, 0)

    mask = pa.array(keep)
    return pc.filter(batch, mask)
