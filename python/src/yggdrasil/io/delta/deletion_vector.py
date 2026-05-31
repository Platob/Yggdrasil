"""Deletion-vector encode/decode + Arrow batch masking."""

from __future__ import annotations

import base64
import dataclasses
import logging
import struct
import uuid as _uuid
from typing import TYPE_CHECKING, Iterable, Optional, Set

from yggdrasil.io.delta.protocol import DeletionVectorDescriptor

if TYPE_CHECKING:
    import pyarrow as pa
    from yggdrasil.path import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DeletionVector", "DeletionVectorDescriptor",
    "decode_deletion_vector", "encode_inline_deletion_vector",
    "write_uuid_deletion_vector", "mask_batch_with_dv",
]

_Z85_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
_B85_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
_Z85_TO_B85 = bytes.maketrans(_Z85_ALPHABET.encode("ascii"), _B85_ALPHABET.encode("ascii"))
_B85_TO_Z85 = bytes.maketrans(_B85_ALPHABET.encode("ascii"), _Z85_ALPHABET.encode("ascii"))

_MAGIC_ROARING_64 = 1681511377  # 0x6439D3D1
_MAGIC_SIMPLE = 1681511376      # 0x6439D3D0
_ROARING_THRESHOLD = 4096


def _z85_encode(data: bytes) -> str:
    if not data: return ""
    if len(data) % 4 != 0:
        raise ValueError(f"_z85_encode requires 4-byte aligned input; got len={len(data)}.")
    return base64.b85encode(data).translate(_B85_TO_Z85).decode("ascii")


# ---------------------------------------------------------------------------
# Roaring decode
# ---------------------------------------------------------------------------

def _read_portable_roaring(buf: memoryview, pos: int) -> "tuple[Set[int], int]":
    cookie = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    if (cookie & 0xFFFF) == 12347:
        n_containers = ((cookie >> 16) & 0xFFFF) + 1
        rs = (n_containers + 7) // 8
        run_flags = bytes(buf[pos:pos + rs]); pos += rs; has_runs = True
    else:
        n_containers = struct.unpack_from("<I", buf, pos)[0]; pos += 4
        run_flags = b""; has_runs = False
    key_card = []
    for _ in range(n_containers):
        k, cm1 = struct.unpack_from("<HH", buf, pos); pos += 4
        key_card.append((k, cm1 + 1))
    if n_containers >= 4: pos += 4 * n_containers
    out: Set[int] = set()
    for idx, (key, card) in enumerate(key_card):
        high = key << 16
        is_run = bool(run_flags[idx >> 3] & (1 << (idx & 7))) if has_runs else False
        if is_run:
            nr = struct.unpack_from("<H", buf, pos)[0]; pos += 2
            for _ in range(nr):
                s, l = struct.unpack_from("<HH", buf, pos); pos += 4
                for v in range(s, s + l + 1): out.add(high | v)
        elif card <= 4096:
            for _ in range(card):
                out.add(high | struct.unpack_from("<H", buf, pos)[0]); pos += 2
        else:
            for wi in range(1024):
                w = struct.unpack_from("<Q", buf, pos)[0]; pos += 8
                if not w: continue
                b = wi << 6; bit = 0
                while w:
                    if w & 1: out.add(high | (b + bit))
                    w >>= 1; bit += 1
    return out, pos


def _decode_payload(payload: bytes) -> Set[int]:
    if len(payload) < 4: return set()
    mv = memoryview(payload)
    magic = struct.unpack_from("<I", mv, 0)[0]
    if magic == _MAGIC_SIMPLE:
        count = struct.unpack_from("<Q", mv, 4)[0]
        return {struct.unpack_from("<Q", mv, 12 + i * 8)[0] for i in range(count)}
    if magic == _MAGIC_ROARING_64:
        n = struct.unpack_from("<Q", mv, 4)[0]
        out: Set[int] = set(); pos = 12
        for _ in range(n):
            hi = struct.unpack_from("<I", mv, pos)[0]; pos += 4
            sub, pos = _read_portable_roaring(mv, pos)
            out.update((hi << 32) | x for x in sub)
        return out
    if (magic & 0xFFFF) in (12346, 12347):
        sub, _ = _read_portable_roaring(mv, 0); return sub
    return set()


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def _encode_simple_payload(row_ids: Iterable[int]) -> bytes:
    """Encode *row_ids* as the simple-list deletion-vector envelope.

    Layout: ``<I magic><Q count><Q row>...``. Always emits the
    simple-list envelope regardless of cardinality — readers from
    other engines (delta-rs, Spark) only need to support this
    shape to consume our DVs at small / medium row counts.
    """
    rows = sorted(set(int(r) for r in row_ids))
    return (struct.pack("<I", _MAGIC_SIMPLE) + struct.pack("<Q", len(rows))
            + b"".join(struct.pack("<Q", r) for r in rows))


def _encode_dv_payload(row_ids: Iterable[int]) -> bytes:
    rows = sorted(set(int(r) for r in row_ids))
    if len(rows) <= _ROARING_THRESHOLD:
        return _encode_simple_payload(rows)

    # Roaring64 envelope: group by high 32 bits, encode each as portable Roaring
    chunks: dict[int, list[int]] = {}
    for r in rows: chunks.setdefault((r >> 32) & 0xFFFFFFFF, []).append(r & 0xFFFFFFFF)
    buf = bytearray(struct.pack("<I", _MAGIC_ROARING_64) + struct.pack("<Q", len(chunks)))
    for hi_key in sorted(chunks):
        buf += struct.pack("<I", hi_key)
        # Portable Roaring for the low-32 values
        containers: dict[int, list[int]] = {}
        for v in sorted(chunks[hi_key]):
            containers.setdefault((v >> 16) & 0xFFFF, []).append(v & 0xFFFF)
        nc = len(containers); sk = sorted(containers)
        pr = bytearray(struct.pack("<I", 12346) + struct.pack("<I", nc))
        for k in sk: pr += struct.pack("<HH", k, len(containers[k]) - 1)
        cdata: list[bytes] = []
        for k in sk:
            vals = sorted(containers[k])
            if len(vals) <= 4096:
                cdata.append(b"".join(struct.pack("<H", v) for v in vals))
            else:
                bm = [0] * 1024
                for v in vals: bm[v >> 6] |= (1 << (v & 63))
                cdata.append(b"".join(struct.pack("<Q", w) for w in bm))
        if nc >= 4:
            hs = 4 + 4 + 4 * nc + 4 * nc; cum = 0
            for c in cdata: pr += struct.pack("<I", hs + cum); cum += len(c)
        for c in cdata: pr += c
        buf += pr
    return bytes(buf)


def encode_inline_deletion_vector(row_ids: Iterable[int]) -> DeletionVectorDescriptor:
    payload = _encode_dv_payload(row_ids)
    return DeletionVectorDescriptor(
        storage_type="i", path_or_inline_dv=_z85_encode(payload),
        size_in_bytes=len(payload),
        cardinality=len(set(row_ids) if not isinstance(row_ids, (set, frozenset)) else row_ids),
    )

def write_uuid_deletion_vector(row_ids: Iterable[int], *, table_root: "Path") -> DeletionVectorDescriptor:
    payload = _encode_dv_payload(row_ids)
    framed = struct.pack(">I", len(payload)) + payload + struct.pack(">I", 0)
    uid = _uuid.uuid4().hex
    sidecar = table_root / f"deletion_vector_{uid}.bin"
    with sidecar.open("wb") as bio:
        bio.truncate(0); bio.write_bytes(framed)
    return DeletionVectorDescriptor(
        storage_type="u", path_or_inline_dv=uid, size_in_bytes=len(payload),
        cardinality=len(set(row_ids) if not isinstance(row_ids, (set, frozenset)) else row_ids),
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
        try:
            raw = base64.b85decode(descriptor.path_or_inline_dv.encode("ascii").translate(_Z85_TO_B85)) if descriptor.path_or_inline_dv else b""
        except Exception: return DeletionVector(descriptor=descriptor)
        return DeletionVector(descriptor=descriptor, deleted_rows=_decode_payload(raw))

    if table_root is None or storage not in ("u", "p"):
        return DeletionVector(descriptor=descriptor)

    raw_path = descriptor.path_or_inline_dv or ""
    if not raw_path: return DeletionVector(descriptor=descriptor)
    if storage == "p":
        sidecar_path = table_root / raw_path
    else:
        # storage == "u": pathOrInlineDv encodes the DV file UUID.
        # Two shapes flow through here:
        # - Yggdrasil's writer: 32-char lowercase hex UUID
        #   stored verbatim. On disk: deletion_vector_<hex>.bin.
        #   Detected by exact 32-hex shape so a Z85 string can't be
        #   mis-routed.
        # - Delta spec: <random-prefix><Z85-encoded-UUID> where the
        #   trailing 20 chars are the Z85 UUID and the leading 0-2
        #   chars are a random prefix distributing files across
        #   subdirectories. On disk:
        #   <prefix>/deletion_vector_<canonical-UUID>.bin.
        if (len(raw_path) == 32
                and all("0" <= c <= "9" or "a" <= c <= "f" for c in raw_path)):
            sidecar_path = table_root / f"deletion_vector_{raw_path}.bin"
        elif len(raw_path) >= 20:
            try:
                uuid_str = str(_uuid.UUID(bytes=base64.b85decode(
                    raw_path[-20:].encode("ascii").translate(_Z85_TO_B85),
                )))
            except Exception:
                return DeletionVector(descriptor=descriptor)
            prefix = raw_path[:-20]
            leaf = f"deletion_vector_{uuid_str}.bin"
            sidecar_path = (
                table_root / prefix / leaf if prefix else table_root / leaf
            )
        else:
            return DeletionVector(descriptor=descriptor)

    offset = int(descriptor.offset or 0)
    size = int(descriptor.size_in_bytes or 0)
    logger.debug(
        "decode_deletion_vector: storage=%s pathOrInline=%r offset=%d "
        "size=%d -> sidecar=%s",
        storage, raw_path, offset, size, sidecar_path,
    )
    cache_key = f"{sidecar_path.full_path()}|{offset}|{size}"
    cached = sidecar_cache.get(cache_key) if sidecar_cache is not None else None
    if cached is None:
        try:
            with sidecar_path.open("rb") as bio:
                bio.seek(offset); raw = bio.read(size + 8)
            # File layout per Delta protocol:
            #   <4 bytes BE size header><size bytes payload><4 bytes BE CRC>
            # The previous code took ``raw[:size]`` when the BE size
            # header didn't fit ``0 < fs <= size`` (which it never does
            # — the header value is size+4 for CRC), folding the header
            # into the payload and truncating the actual DV bytes. Slice
            # off the header and take exactly ``size`` bytes of payload.
            if size > 0 and len(raw) >= 4 + size:
                cached = bytes(raw[4:4 + size])
            else:
                cached = b""
                logger.warning(
                    "DV sidecar short read: path=%s size=%d got=%d bytes",
                    sidecar_path, size, len(raw),
                )
        except Exception as exc:
            logger.warning(
                "DV sidecar read failed: path=%s -> %r", sidecar_path, exc,
            )
            cached = b""
        if sidecar_cache is not None: sidecar_cache[cache_key] = cached
    rows = _decode_payload(cached)
    logger.debug(
        "DV decoded: %d deleted rows from %s", len(rows), sidecar_path,
    )
    return DeletionVector(descriptor=descriptor, deleted_rows=rows)


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
    keep = np.ones(n, dtype=bool); keep[rel[in_range]] = False
    if keep.all(): return batch
    if not keep.any(): return batch.slice(0, 0)
    return pc.filter(batch, pa.array(keep))
