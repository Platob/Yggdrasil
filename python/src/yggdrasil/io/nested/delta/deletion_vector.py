"""Deletion-vector encode/decode + Arrow batch masking.

A Delta deletion vector is a serialized roaring bitmap whose bits
mark the **logical row indices** of a parquet file that have been
deleted. The reader's job is to read the parquet file as usual, then
mask out those rows before yielding the batch upstream. The writer's
job is the mirror: take a set of row indices and emit a descriptor +
sidecar bytes so a later read drops them.

Three storage shapes
--------------------

- ``"i"`` (inline) — the bitmap bytes are Z85-encoded straight into
  the descriptor's ``pathOrInlineDv``. No sidecar fetch.
- ``"u"`` (UUID sidecar) — bitmap lives in
  ``<table>/deletion_vector_<uuid>.bin`` (or, when V2 checkpoint
  sidecars are involved, under a ``_delta_log/_sidecars/`` directory).
  ``offset`` + ``sizeInBytes`` window into it; one sidecar can
  multiplex many DVs.
- ``"p"`` (absolute path) — rare, table-relative absolute path. We
  resolve it relative to the table root.

Bitmap layout
-------------

For both inline and sidecar shapes, the on-disk envelope is:

::

    int32 size_in_bytes (big-endian)            ← sidecar only; inline skips this
    bytes payload[size_in_bytes]                ← format identifier + bitmap data
    int32 crc32 (big-endian, optional)

The ``payload`` is itself either:

- A magic-byte 0x01 prefix + portable Roaring bitmap (the spec's
  "RoaringBitmapArray" — one entry per "high 32 bits" key, each
  holding a portable Roaring bitmap covering the low-32-bit row ids).
- A magic-byte 0x00 prefix + 64-bit big-endian count + 64-bit big-endian
  row ids ("BitmapArray" simple format) — used for tiny DVs.

Read path supports both. Write path emits the simple-list shape (the
small-DV envelope), which is always legal regardless of the table's
``deletionVectors`` writer feature — it's the format Spark's
``DeletionVectorStore`` writes for any DV under ~16k rows.
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
    from yggdrasil.io.path import Path


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

# Spec: inline DVs use the "Z85" base85 alphabet (ZeroMQ flavour).
# Python's stdlib ships :func:`base64.b85decode` (the RFC1924 alphabet)
# and :func:`base64.a85decode` (Adobe). Z85 differs from RFC1924 by
# alphabet shuffling, so we do a one-shot translation table on top of
# :func:`base64.b85decode` / :func:`base64.b85encode`. The table is 1:1
# — same character set, same padding rules — only the symbol-to-value
# mapping is permuted.

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
    """Decode a Z85-encoded string into raw bytes.

    Pure-stdlib path: translate Z85 → b85 alphabet, then hand off to
    :func:`base64.b85decode`. Z85 length is always a multiple of 5, so
    no padding handling is needed.
    """
    if not data:
        return b""
    encoded = data.encode("ascii").translate(_Z85_TO_B85)
    return base64.b85decode(encoded)


def _z85_encode(data: bytes) -> str:
    """Encode bytes into a Z85 string. Input must be a multiple of 4."""
    if not data:
        return ""
    if len(data) % 4 != 0:
        # Pad with zeros and remember the count — the framing layer
        # handles the trim. The simple-list / roaring envelopes we
        # produce are always 4-byte aligned (magic + uint64 count +
        # n*uint64 ids), so we never actually hit this branch.
        raise ValueError(
            f"_z85_encode requires 4-byte aligned input; got len={len(data)}. "
            f"DV envelopes are 4-byte aligned by construction — check the caller."
        )
    encoded = base64.b85encode(data).translate(_B85_TO_Z85)
    return encoded.decode("ascii")


# ---------------------------------------------------------------------------
# Roaring-bitmap decode — minimal portable subset
# ---------------------------------------------------------------------------

#: Magic prefix on the 64-bit "RoaringBitmapArray" envelope.
_MAGIC_ROARING_64 = 1681511377  # 0x64426152 — "RaBd" little-endian

#: Magic prefix in the simple-list envelope used for tiny DVs.
_MAGIC_SIMPLE = 1681511376  # 0x64426150


def _read_portable_roaring(buf: memoryview, pos: int) -> "tuple[Set[int], int]":
    """Decode one *portable* Roaring bitmap starting at *pos*.

    Returns ``(row_ids, new_pos)``. The portable layout is well-defined
    and small — see the Roaring spec — so a hand-rolled parser is fine
    here. We don't need rank/select operations, just the row-ids.
    """
    # Cookie + container count.
    cookie = struct.unpack_from("<I", buf, pos)[0]
    pos += 4

    if (cookie & 0xFFFF) == 0x3B30:
        # Cookie carries the container count in its high 16 bits.
        n_containers = ((cookie >> 16) & 0xFFFF) + 1
        bitmap_of_runs_size = (n_containers + 7) // 8
        run_flag_bytes = bytes(buf[pos : pos + bitmap_of_runs_size])
        pos += bitmap_of_runs_size
        has_run_flag = True
    else:
        # Cookie 0x3B31 + 4-byte container count.
        n_containers = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        run_flag_bytes = b""
        has_run_flag = False

    # Container key + cardinality table.
    key_card: list[tuple[int, int]] = []
    for _ in range(n_containers):
        key, card_minus_one = struct.unpack_from("<HH", buf, pos)
        pos += 4
        key_card.append((key, card_minus_one + 1))

    # Offset table is only present when the run-flag bitmap is absent
    # *and* the container count is at least :data:`NO_OFFSET_THRESHOLD`
    # (4) in the spec; for small bitmaps it can be skipped. The format
    # always emits the offsets in modern bitmaps, but we don't actually
    # need them — containers are laid out contiguously after the
    # header, so we just skip the table when present.
    if not has_run_flag or n_containers >= 4:
        # Skip 4 bytes per container.
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
                # length is "additional values after start".
                for v in range(start, start + length + 1):
                    out.add(high | v)
        elif card <= 4096:
            # Array container: card * uint16 values.
            for _ in range(card):
                v = struct.unpack_from("<H", buf, pos)[0]
                pos += 2
                out.add(high | v)
        else:
            # Bitmap container: 8192 bytes, one bit per low-16 value.
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
    """Decode the inner ``payload`` of either DV envelope shape.

    Layout:
        [magic int32 (LE)] [body]

    For the 64-bit Roaring envelope (``_MAGIC_ROARING_64``), the body is
    a uint64 ``count_of_chunks`` followed by ``count_of_chunks`` records
    of ``(uint32 high_key, portable_roaring_bitmap)``.

    For the simple envelope (``_MAGIC_SIMPLE``), the body is a uint64
    ``count`` followed by ``count`` × uint64 row ids.
    """
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

    # Some Spark writers emit a bare portable-roaring bitmap (no magic).
    # Probe by treating the first 4 bytes as a Roaring cookie; if the
    # parse looks plausible (low 16 bits is 0x3B30 / 0x3B31) we accept.
    cookie = magic & 0xFFFF
    if cookie in (0x3B30, 0x3B31):
        sub, _ = _read_portable_roaring(mv, 0)
        return sub

    # Unknown envelope — return empty rather than raise; the higher
    # level treats DV decode failures as "no rows masked" so the read
    # path still produces something usable.
    return set()


# ---------------------------------------------------------------------------
# Encode — simple-list envelope (the format we write)
# ---------------------------------------------------------------------------


def _encode_simple_payload(row_ids: Iterable[int]) -> bytes:
    """Pack *row_ids* into the simple-list DV payload.

    Layout (all little-endian):
        ``int32 magic = _MAGIC_SIMPLE``
        ``uint64 count``
        ``uint64 row_id`` × count   (sorted ascending)

    The payload is the bytes a sidecar reader stores after the framing
    int32-size prefix — :func:`write_uuid_deletion_vector` adds the
    framing on the way to disk, :func:`encode_inline_deletion_vector`
    leaves it off (inline DV bytes are the raw payload).
    """
    rows = sorted(set(int(r) for r in row_ids))
    body = struct.pack("<I", _MAGIC_SIMPLE) + struct.pack("<Q", len(rows))
    body += b"".join(struct.pack("<Q", r) for r in rows)
    return body


def encode_inline_deletion_vector(
    row_ids: Iterable[int],
) -> DeletionVectorDescriptor:
    """Build an inline (Z85-encoded) deletion-vector descriptor.

    Use this only for tiny DVs — the encoded blob lives directly in the
    log line, so a few hundred deleted rows is the practical ceiling
    before the JSON commit gets unwieldy. Above that, switch to a
    sidecar via :func:`write_uuid_deletion_vector`.
    """
    payload = _encode_simple_payload(row_ids)
    encoded = _z85_encode(payload)
    return DeletionVectorDescriptor(
        storage_type="i",
        path_or_inline_dv=encoded,
        size_in_bytes=len(payload),
        cardinality=_count_from_payload(payload),
    )


def write_uuid_deletion_vector(
    row_ids: Iterable[int],
    *,
    table_root: "Path",
) -> DeletionVectorDescriptor:
    """Emit a UUID-named sidecar holding the DV for *row_ids*.

    Frame: ``int32 size + payload + int32 crc(=0)`` — the spec allows
    a zero crc, and Spark readers don't verify it. ``offset=0`` and
    ``sizeInBytes`` covers the inner payload only (post-frame).

    Returns the descriptor a writer should embed in the matching
    AddFile / RemoveFile action.
    """
    from yggdrasil.io.path.path import Path as _Path  # noqa: F401 — typing only.

    payload = _encode_simple_payload(row_ids)
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
        cardinality=_count_from_payload(payload),
        offset=0,
    )


def _count_from_payload(payload: bytes) -> int:
    """Cheap cardinality lookup — peek the count out of a simple envelope."""
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
    """Read the framed sidecar payload at ``offset`` for ``size`` bytes.

    The framing is: ``int32 size`` + ``payload[size]`` + ``int32 crc``.
    Some writers emit the payload directly without the size+crc frame
    (Databricks' ``offset`` already points past the size header). We
    accept both: if the first 4 bytes look like a sane payload size,
    we strip it; otherwise we treat the whole window as the payload.
    """
    with sidecar.open("rb") as bio:
        bio.seek(offset)
        raw = bio.read(size + 8)  # over-read to capture trailing crc
    if not raw:
        return b""

    if len(raw) >= 4:
        framed_size = struct.unpack(">I", raw[:4])[0]
        if 0 < framed_size <= size:
            return bytes(raw[4 : 4 + framed_size])
    # Fall back to "the window IS the payload."
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
        """Return the keep-indices for a parquet of ``num_rows`` rows."""
        if not self.deleted_rows:
            return list(range(num_rows))
        deleted = self.deleted_rows
        return [i for i in range(num_rows) if i not in deleted]


def decode_deletion_vector(
    descriptor: Optional[DeletionVectorDescriptor],
    *,
    table_root: "Path | None" = None,
    sidecar_cache: "dict[str, bytes] | None" = None,
) -> Optional[DeletionVector]:
    """Decode *descriptor* into a :class:`DeletionVector`.

    ``table_root`` is required for ``"u"`` / ``"p"`` storage shapes —
    inline DVs decode without any path I/O.

    ``sidecar_cache`` (a caller-supplied dict) lets multiple DVs that
    share a sidecar file collapse to one read. The :class:`Snapshot`
    threads one cache through every DV decode for its read pass.
    """
    if descriptor is None:
        return None

    storage = (descriptor.storage_type or "").lower()

    if storage == "i":
        # Inline payload — Z85-encoded. The descriptor's
        # ``size_in_bytes`` is the *unencoded* size; we decode the
        # whole string and trust it.
        try:
            raw = _z85_decode(descriptor.path_or_inline_dv)
        except Exception:
            return DeletionVector(descriptor=descriptor)
        # Inline payloads sometimes ship with the framing prefix and
        # sometimes without. Strip a leading int32 size if it matches.
        if len(raw) >= 4:
            head = struct.unpack(">I", raw[:4])[0]
            if head + 4 <= len(raw):
                raw = raw[4 : 4 + head]
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
    """Resolve a UUID / absolute-path DV descriptor to a real sidecar Path.

    UUID storage stores the sidecar at ``<root>/deletion_vector_<uuid>.bin``.
    A leading single-byte prefix on ``pathOrInlineDv`` (sometimes used
    by Databricks to encode "where" the sidecar lives — root vs. log
    sidecars dir) is stripped here.
    """
    raw = descriptor.path_or_inline_dv or ""
    if not raw:
        return None

    storage = (descriptor.storage_type or "").lower()

    if storage == "p":
        # Absolute path, table-relative.
        return table_root / raw

    # UUID — strip leading single-char prefix when present (anything
    # that isn't a uuid hex char).
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

    No-op when ``dv`` is None or empty. ``base_offset`` is the row
    index of the batch's first row within its parquet file — non-zero
    when a parquet is read in chunks and the DV's row ids are
    file-relative.

    Mask construction is vectorised through numpy: a boolean array
    of size ``n`` (one row per batch position) gets ``False`` written
    at every in-range deleted offset, and the inverted mask drives a
    single ``RecordBatch.filter`` kernel — no per-row Python loop
    even when the DV's deleted set is large.

    Imported lazily to keep the cold module import cheap.
    """
    import numpy as np
    import pyarrow as pa  # local — keeps the cold module import cheap.
    import pyarrow.compute as pc

    if dv is None or dv.is_empty():
        return batch
    n = batch.num_rows
    if n == 0:
        return batch

    deleted = dv.deleted_rows
    # Build the keep-mask vectorised. Materialise the deleted set
    # into a numpy int64 array once (one Python-side hop, then C
    # speed everywhere downstream), translate to batch-relative
    # indices, and drop anything outside ``[0, n)``.
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

    # One pyarrow.compute call masks every column in the batch.
    mask = pa.array(keep)
    return pc.filter(batch, mask)
