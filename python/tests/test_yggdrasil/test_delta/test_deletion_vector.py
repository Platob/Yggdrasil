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

        from yggdrasil.io.path import LocalPath

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
