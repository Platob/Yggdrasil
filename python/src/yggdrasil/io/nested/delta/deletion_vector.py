"""Deletion-vector encode/decode + Arrow batch masking.

Storage shapes: inline (Z85), UUID sidecar, absolute path.
Envelope formats: simple-list (magic 0x6439D3D0) and
64-bit Roaring (magic 0x6439D3D1).
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
    "DeletionVector", "DeletionVectorDescriptor",
    "decode_deletion_vector", "encode_inline_deletion_vector",
    "write_uuid_deletion_vector", "mask_batch_with_dv",
]

# ---------------------------------------------------------------------------
# Z85 codec
# ---------------------------------------------------------------------------

_Z85_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
_B85_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
_Z85_TO_B85 = bytes.maketrans(_Z85_ALPHABET.encode("ascii"), _B85_ALPHABET.encode("ascii"))
_B85_TO_Z85 = bytes.maketrans(_B85_ALPHABET.encode("ascii"), _Z85_ALPHABET.encode("ascii"))

def _z85_decode(data: str) -> bytes:
    if not data: return b""
    return base64.b85decode(data.encode("ascii").translate(_Z85_TO_B85))

def _z85_encode(data: bytes) -> str:
    if not data: return ""
    if len(data) % 4 != 0:
        raise ValueError(f"_z85_encode requires 4-byte aligned input; got len={len(data)}.")
    return base64.b85encode(data).translate(_B85_TO_Z85).decode("ascii")

# ---------------------------------------------------------------------------
# Roaring decode
# ---------------------------------------------------------------------------

_MAGIC_ROARING_64 = 1681511377  # 0x6439D3D1
_MAGIC_SIMPLE = 1681511376      # 0x6439D3D0
_ROARING_THRESHOLD = 4096


def _read_portable_roaring(buf: memoryview, pos: int) -> "tuple[Set[int], int]":
    cookie = struct.unpack_from("<I", buf, pos)[0]; pos += 4

    if (cookie & 0xFFFF) == 12347:  # SERIAL_COOKIE (has run containers)
        n_containers = ((cookie >> 16) & 0xFFFF) + 1
        run_size = (n_containers + 7) // 8
        run_flags = bytes(buf[pos:pos + run_size]); pos += run_size
        has_runs = True
    else:
        n_containers = struct.unpack_from("<I", buf, pos)[0]; pos += 4
        run_flags = b""; has_runs = False

    key_card = []
    for _ in range(n_containers):
        k, cm1 = struct.unpack_from("<HH", buf, pos); pos += 4
        key_card.append((k, cm1 + 1))

    if n_containers >= 4:
        pos += 4 * n_containers

    out: Set[int] = set()
    for idx, (key, card) in enumerate(key_card):
        high = key << 16
        is_run = bool(run_flags[idx >> 3] & (1 << (idx & 7))) if has_runs else False
        if is_run:
            n_runs = struct.unpack_from("<H", buf, pos)[0]; pos += 2
            for _ in range(n_runs):
                start, length = struct.unpack_from("<HH", buf, pos); pos += 4
                for v in range(start, start + length + 1): out.add(high | v)
        elif card <= 4096:
            for _ in range(card):
                out.add(high | struct.unpack_from("<H", buf, pos)[0]); pos += 2
        else:
            for wi in range(1024):
                word = struct.unpack_from("<Q", buf, pos)[0]; pos += 8
                if not word: continue
                base = wi << 6; bit = 0
                while word:
                    if word & 1: out.add(high | (base + bit))
                    word >>= 1; bit += 1
    return out, pos


def _decode_payload(payload: bytes) -> Set[int]:
    if len(payload) < 4: return set()
    mv = memoryview(payload)
    magic = struct.unpack_from("<I", mv, 0)[0]

    if magic == _MAGIC_SIMPLE:
        count = struct.unpack_from("<Q", mv, 4)[0]
        return {struct.unpack_from("<Q", mv, 12 + i * 8)[0] for i in range(count)}

    if magic == _MAGIC_ROARING_64:
        n_chunks = struct.unpack_from("<Q", mv, 4)[0]
        out: Set[int] = set(); pos = 12
        for _ in range(n_chunks):
            hi = struct.unpack_from("<I", mv, pos)[0]; pos += 4
            sub, pos = _read_portable_roaring(mv, pos)
            out.update((hi << 32) | x for x in sub)
        return out

    if (magic & 0xFFFF) in (12346, 12347):
        sub, _ = _read_portable_roaring(mv, 0)
        return sub
    return set()

# ---------------------------------------------------------------------------
# Roaring encode
# ---------------------------------------------------------------------------

def _encode_roaring_payload(row_ids: Iterable[int]) -> bytes:
    rows = sorted(set(int(r) for r in row_ids))
    if not rows:
        return struct.pack("<I", _MAGIC_SIMPLE) + struct.pack("<Q", 0)
    chunks: dict[int, list[int]] = {}
    for r in rows:
        chunks.setdefault((r >> 32) & 0xFFFFFFFF, []).append(r & 0xFFFFFFFF)
    buf = bytearray(struct.pack("<I", _MAGIC_ROARING_64) + struct.pack("<Q", len(chunks)))
    for hi_key in sorted(chunks):
        buf += struct.pack("<I", hi_key)
        buf += _encode_portable_roaring(sorted(chunks[hi_key]))
    return bytes(buf)


def _encode_portable_roaring(values: list[int]) -> bytes:
    containers: dict[int, list[int]] = {}
    for v in values:
        containers.setdefault((v >> 16) & 0xFFFF, []).append(v & 0xFFFF)
    n = len(containers)
    sorted_keys = sorted(containers)

    buf = bytearray(struct.pack("<I", 12346) + struct.pack("<I", n))
    for key in sorted_keys:
        buf += struct.pack("<HH", key, len(containers[key]) - 1)

    chunks: list[bytes] = []
    for key in sorted_keys:
        vals = sorted(containers[key])
        if len(vals) <= 4096:
            chunks.append(b"".join(struct.pack("<H", v) for v in vals))
        else:
            bitmap = [0] * 1024
            for v in vals: bitmap[v >> 6] |= (1 << (v & 63))
            chunks.append(b"".join(struct.pack("<Q", w) for w in bitmap))

    if n >= 4:
        header_size = 4 + 4 + 4 * n + 4 * n
        cum = 0
        for c in chunks:
            buf += struct.pack("<I", header_size + cum); cum += len(c)
    for c in chunks: buf += c
    return bytes(buf)


def _encode_dv_payload(row_ids: Iterable[int]) -> bytes:
    rows = sorted(set(int(r) for r in row_ids))
    if len(rows) <= _ROARING_THRESHOLD:
        body = struct.pack("<I", _MAGIC_SIMPLE) + struct.pack("<Q", len(rows))
        body += b"".join(struct.pack("<Q", r) for r in rows)
        return body
    return _encode_roaring_payload(rows)

# ---------------------------------------------------------------------------
# Public encode/write
# ---------------------------------------------------------------------------

def encode_inline_deletion_vector(row_ids: Iterable[int]) -> DeletionVectorDescriptor:
    payload = _encode_dv_payload(row_ids)
    return DeletionVectorDescriptor(
        storage_type="i", path_or_inline_dv=_z85_encode(payload),
        size_in_bytes=len(payload), cardinality=len(set(row_ids) if not isinstance(row_ids, (set, frozenset)) else row_ids),
    )

def write_uuid_deletion_vector(row_ids: Iterable[int], *, table_root: "Path") -> DeletionVectorDescriptor:
    payload = _encode_dv_payload(row_ids)
    framed = struct.pack(">I", len(payload)) + payload + struct.pack(">I", 0)
    uid = _uuid.uuid4().hex
    sidecar = table_root / f"deletion_vector_{uid}.bin"
    with sidecar.open("wb") as bio:
        bio.truncate(0); bio.write_bytes(framed)
    return DeletionVectorDescriptor(
        storage_type="u", path_or_inline_dv=uid,
        size_in_bytes=len(payload), cardinality=len(set(row_ids) if not isinstance(row_ids, (set, frozenset)) else row_ids),
        offset=0,
    )

# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

@dataclasses.dataclass(slots=True)
class DeletionVector:
    descriptor: DeletionVectorDescriptor
    deleted_rows: Set[int] = dataclasses.field(default_factory=set)

    def is_empty(self) -> bool: return not self.deleted_rows
    def __len__(self) -> int: return len(self.deleted_rows)
    def __contains__(self, row: int) -> bool: return row in self.deleted_rows


def decode_deletion_vector(
    descriptor: Optional[DeletionVectorDescriptor], *,
    table_root: "Path | None" = None,
    sidecar_cache: "dict[str, bytes] | None" = None,
) -> Optional[DeletionVector]:
    if descriptor is None: return None
    storage = (descriptor.storage_type or "").lower()

    if storage == "i":
        try: raw = _z85_decode(descriptor.path_or_inline_dv)
        except Exception: return DeletionVector(descriptor=descriptor)
        return DeletionVector(descriptor=descriptor, deleted_rows=_decode_payload(raw))

    if table_root is None:
        return DeletionVector(descriptor=descriptor)

    if storage not in ("u", "p"):
        return DeletionVector(descriptor=descriptor)

    # Resolve sidecar path
    raw_path = descriptor.path_or_inline_dv or ""
    if not raw_path: return DeletionVector(descriptor=descriptor)
    if storage == "p":
        sidecar_path = table_root / raw_path
    else:
        uid = raw_path
        if uid and not (uid[0].isalnum() and uid[0] not in "ghijklmnopqrstuvwxyz"):
            uid = uid[1:]
        sidecar_path = table_root / f"deletion_vector_{uid}.bin"

    offset = int(descriptor.offset or 0)
    size = int(descriptor.size_in_bytes or 0)
    cache_key = f"{sidecar_path.full_path()}|{offset}|{size}"
    cached = sidecar_cache.get(cache_key) if sidecar_cache is not None else None
    if cached is None:
        try:
            with sidecar_path.open("rb") as bio:
                bio.seek(offset)
                raw = bio.read(size + 8)
            if raw and len(raw) >= 4:
                framed_size = struct.unpack(">I", raw[:4])[0]
                cached = bytes(raw[4:4 + framed_size]) if 0 < framed_size <= size else bytes(raw[:size])
            else:
                cached = b""
        except Exception:
            cached = b""
        if sidecar_cache is not None:
            sidecar_cache[cache_key] = cached
    return DeletionVector(descriptor=descriptor, deleted_rows=_decode_payload(cached))


def mask_batch_with_dv(batch: "pa.RecordBatch", dv: Optional[DeletionVector], *,
                       base_offset: int = 0) -> "pa.RecordBatch":
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    if dv is None or dv.is_empty(): return batch
    n = batch.num_rows
    if n == 0: return batch
    deleted = dv.deleted_rows
    if not deleted: return batch

    del_arr = np.fromiter(deleted, dtype=np.int64, count=len(deleted))
    rel = del_arr - np.int64(base_offset)
    in_range = (rel >= 0) & (rel < np.int64(n))
    if not in_range.any(): return batch
    rel = rel[in_range]

    keep = np.ones(n, dtype=bool); keep[rel] = False
    if keep.all(): return batch
    if not keep.any(): return batch.slice(0, 0)
    return pc.filter(batch, pa.array(keep))
