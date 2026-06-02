"""Tests for :mod:`yggdrasil.delta.deletion_vector`."""
from __future__ import annotations

import struct

from yggdrasil.delta.deletion_vector import (
    DeletionVector,
    DeletionVectorDescriptor,
    _decode_payload,
    _MAGIC_SIMPLE,
    decode_deletion_vector,
    mask_batch_with_dv,
)
from yggdrasil.io.delta.deletion_vector import _encode_dv_payload
from yggdrasil.delta.tests import DeltaTestCase


class TestSimpleDecoder(DeltaTestCase):
    def test_simple_envelope_round_trip(self) -> None:
        rows = [1, 5, 7, 100]
        payload = (
            struct.pack("<I", _MAGIC_SIMPLE)
            + struct.pack("<Q", len(rows))
            + b"".join(struct.pack("<Q", r) for r in rows)
        )
        out = _decode_payload(payload)
        self.assertEqual(sorted(out), rows)


class TestDatabricksRoaringDV(DeltaTestCase):
    """Decode the portable RoaringBitmap format Databricks actually writes.

    Databricks serialises even a single-row DV as a roaring bitmap with the
    *no-run* cookie (0x303a), which always carries the per-container offset
    header. The decoder used to skip that header only for >= 4 containers, so
    a one-container DV misread the 4-byte offset (16) as the deleted index.
    """

    # A real DV payload captured from a Databricks ``DELETE`` that marks local
    # row-index 1 as deleted: magic 1681511377, one high bucket, no-run cookie,
    # one array container, offset header = 16, value = 1.
    _DBX_PAYLOAD = bytes.fromhex(
        "d1d339640100000000000000000000003a3000000100000000000000100000000100"
    )

    def test_decodes_real_databricks_single_row_dv(self) -> None:
        self.assertEqual(_decode_payload(self._DBX_PAYLOAD), {1})

    def test_roaring_round_trip_across_container_counts(self) -> None:
        # >4096 rows forces the roaring envelope (≤4096 uses the simple list);
        # spread over several 16-bit containers so the offset header matters.
        rows = sorted({i * 7 for i in range(5000)})
        self.assertEqual(sorted(_decode_payload(_encode_dv_payload(rows))), rows)

    def test_roaring_single_container_round_trip(self) -> None:
        # All values in one 16-bit container — the previously-broken shape.
        rows = sorted(range(0, 9000, 2))  # 4500 evens, all < 65536
        self.assertEqual(sorted(_decode_payload(_encode_dv_payload(rows))), rows)


class TestMaskBatch(DeltaTestCase):
    def test_no_dv_passes_through(self) -> None:
        batch = self.pa.record_batch({"id": [1, 2, 3]})
        out = mask_batch_with_dv(batch, None)
        self.assertIs(out, batch)

    def test_empty_dv_passes_through(self) -> None:
        batch = self.pa.record_batch({"id": [1, 2, 3]})
        descriptor = DeletionVectorDescriptor(
            storage_type="i", path_or_inline_dv="", size_in_bytes=0,
        )
        dv = DeletionVector(descriptor=descriptor, deleted_rows=set())
        out = mask_batch_with_dv(batch, dv)
        self.assertIs(out, batch)

    def test_drops_marked_rows(self) -> None:
        batch = self.pa.record_batch({"id": [10, 20, 30, 40]})
        descriptor = DeletionVectorDescriptor(
            storage_type="i", path_or_inline_dv="", size_in_bytes=0,
        )
        dv = DeletionVector(descriptor=descriptor, deleted_rows={1, 3})
        out = mask_batch_with_dv(batch, dv)
        self.assertEqual(out.column("id").to_pylist(), [10, 30])

    def test_base_offset_translates_deleted_ids(self) -> None:
        # The DV holds *file-relative* row ids; a parquet read in
        # chunks hands us a batch whose first row is at
        # ``base_offset`` within the file. Pin the translation.
        batch = self.pa.record_batch({"id": [10, 20, 30, 40]})
        descriptor = DeletionVectorDescriptor(
            storage_type="i", path_or_inline_dv="", size_in_bytes=0,
        )
        # File rows 101 and 103 → batch rows 1 and 3 with offset=100.
        dv = DeletionVector(descriptor=descriptor, deleted_rows={101, 103})
        out = mask_batch_with_dv(batch, dv, base_offset=100)
        self.assertEqual(out.column("id").to_pylist(), [10, 30])

    def test_deleted_outside_batch_range_is_a_noop(self) -> None:
        # File rows 200..299 land in a later batch — for this batch
        # the DV shouldn't drop anything. Vectorised path uses an
        # in-range mask, so the cost stays O(|deleted|) regardless of
        # batch size.
        batch = self.pa.record_batch({"id": [10, 20, 30, 40]})
        descriptor = DeletionVectorDescriptor(
            storage_type="i", path_or_inline_dv="", size_in_bytes=0,
        )
        dv = DeletionVector(
            descriptor=descriptor,
            deleted_rows=frozenset(range(200, 300)),
        )
        out = mask_batch_with_dv(batch, dv, base_offset=0)
        self.assertEqual(out.column("id").to_pylist(), [10, 20, 30, 40])

    def test_all_rows_deleted_returns_empty_batch(self) -> None:
        batch = self.pa.record_batch({"id": [10, 20, 30]})
        descriptor = DeletionVectorDescriptor(
            storage_type="i", path_or_inline_dv="", size_in_bytes=0,
        )
        dv = DeletionVector(descriptor=descriptor, deleted_rows={0, 1, 2})
        out = mask_batch_with_dv(batch, dv)
        self.assertEqual(out.num_rows, 0)
        self.assertEqual(out.schema, batch.schema)

    def test_dense_dv_preserves_row_order(self) -> None:
        # Vectorised mask path must keep row order — pyarrow.compute
        # filter is order-preserving but the regression target here
        # is the case where the mask boolean array is mostly False.
        n = 1000
        batch = self.pa.record_batch({"id": list(range(n))})
        descriptor = DeletionVectorDescriptor(
            storage_type="i", path_or_inline_dv="", size_in_bytes=0,
        )
        # Delete every other row.
        dv = DeletionVector(
            descriptor=descriptor,
            deleted_rows=frozenset(range(0, n, 2)),
        )
        out = mask_batch_with_dv(batch, dv)
        self.assertEqual(out.num_rows, n // 2)
        # Odd-indexed rows survive — id values 1, 3, 5, ...
        self.assertEqual(
            out.column("id").to_pylist()[:5], [1, 3, 5, 7, 9],
        )
        self.assertEqual(out.column("id").to_pylist()[-1], n - 1)


class TestSidecarDecode(DeltaTestCase):
    def test_uuid_sidecar_simple_envelope(self) -> None:
        rows = [2, 4, 8]
        payload = (
            struct.pack("<I", _MAGIC_SIMPLE)
            + struct.pack("<Q", len(rows))
            + b"".join(struct.pack("<Q", r) for r in rows)
        )
        # Frame: ``int32 size + payload + int32 crc``.
        framed = struct.pack(">I", len(payload)) + payload + struct.pack(">I", 0)

        from yggdrasil.path import LocalPath

        uid = "deadbeefdeadbeefdeadbeefdeadbeef"
        sidecar = self.tmp_path / f"deletion_vector_{uid}.bin"
        sidecar.write_bytes(framed)

        descriptor = DeletionVectorDescriptor(
            storage_type="u",
            path_or_inline_dv=uid,
            size_in_bytes=len(payload),
            cardinality=len(rows),
            offset=0,
        )
        dv = decode_deletion_vector(
            descriptor, table_root=LocalPath(str(self.tmp_path)),
        )
        assert dv is not None
        self.assertEqual(sorted(dv.deleted_rows), rows)
