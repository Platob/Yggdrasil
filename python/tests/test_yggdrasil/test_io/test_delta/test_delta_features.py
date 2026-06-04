"""Tests for the rewritten :mod:`yggdrasil.io.delta` package.

Focuses on the features that distinguish this implementation from a
plain folder-of-parquets:

- DV write (inline + sidecar) and read round-trip
- ``DeltaFolder.delete`` with the rewrite strategy and the DV strategy
- :class:`yggdrasil.data.Schema` ↔ Spark JSON schema bridges
- The canonical import path (``yggdrasil.io.delta``) lights up
  the same class registered against ``MimeTypes.DELTA_FOLDER`` as the
  back-compat shim at ``yggdrasil.delta``.

Modes / partitions / time-travel / V1 + V2 checkpoints are already
covered by the parity suite at
``tests/test_yggdrasil/test_delta/test_delta_io.py``; this file adds
to that coverage rather than duplicating it.
"""

from __future__ import annotations

import struct

from yggdrasil.io.delta.deletion_vector import (
    _MAGIC_SIMPLE,
    DeletionVectorDescriptor,
    decode_deletion_vector,
    encode_inline_deletion_vector,
    write_uuid_deletion_vector,
)
from yggdrasil.io.delta.tests import DeltaTestCase

# ---------------------------------------------------------------------------
# Canonical import path
# ---------------------------------------------------------------------------


class TestImportSurface(DeltaTestCase):
    def test_canonical_path_resolves_same_class_as_shim(self) -> None:
        from yggdrasil.delta import DeltaFolder as Shim
        from yggdrasil.io.delta import DeltaFolder as Canonical

        self.assertIs(Shim, Canonical)

    def test_io_nested_reexports_deltaio(self) -> None:
        from yggdrasil.io.delta import DeltaFolder, DeltaOptions
        from yggdrasil.io.delta import DeltaFolder as Canonical
        from yggdrasil.io.delta import DeltaOptions as CanonicalOpts

        self.assertIs(DeltaFolder, Canonical)
        self.assertIs(DeltaOptions, CanonicalOpts)


# ---------------------------------------------------------------------------
# Deletion-vector encode round-trips
# ---------------------------------------------------------------------------


class TestInlineDVRoundTrip(DeltaTestCase):
    def test_inline_dv_round_trip(self) -> None:
        rows = [1, 5, 7, 100, 4096, 999999]
        desc = encode_inline_deletion_vector(rows)
        self.assertEqual(desc.storage_type, "i")
        self.assertEqual(desc.cardinality, len(rows))

        decoded = decode_deletion_vector(desc)
        self.assertIsNotNone(decoded)
        self.assertEqual(sorted(decoded.deleted_rows), sorted(rows))

    def test_inline_dv_empty(self) -> None:
        desc = encode_inline_deletion_vector([])
        decoded = decode_deletion_vector(desc)
        self.assertIsNotNone(decoded)
        self.assertTrue(decoded.is_empty())


class TestSidecarDVRoundTrip(DeltaTestCase):
    def test_uuid_sidecar_round_trip(self) -> None:
        from yggdrasil.path import LocalPath

        rows = list(range(0, 200, 3))  # 67 row indices
        root = LocalPath(str(self.tmp_path))
        desc = write_uuid_deletion_vector(rows, table_root=root)
        self.assertEqual(desc.storage_type, "u")
        self.assertEqual(desc.cardinality, len(rows))
        # Sidecar exists on disk.
        sidecar = self.tmp_path / f"deletion_vector_{desc.path_or_inline_dv}.bin"
        self.assertTrue(sidecar.exists())

        decoded = decode_deletion_vector(desc, table_root=root)
        self.assertIsNotNone(decoded)
        self.assertEqual(sorted(decoded.deleted_rows), sorted(rows))


class TestDecodeUnknownEnvelope(DeltaTestCase):
    def test_zero_filled_payload_yields_empty_dv(self) -> None:
        # Magic ``0x00000000`` matches no envelope we recognize and no
        # plausible Roaring cookie — the decoder degrades to "no rows
        # masked" rather than raising. Inline DVs fall through this
        # path when a bad commit lines up some empty bytes; we don't
        # want a single corrupt action to crash a whole read.
        descriptor = DeletionVectorDescriptor(
            storage_type="i",
            path_or_inline_dv="00000",  # Z85 zeros -> 4 zero bytes
            size_in_bytes=4,
        )
        decoded = decode_deletion_vector(descriptor)
        self.assertIsNotNone(decoded)
        self.assertTrue(decoded.is_empty())


# ---------------------------------------------------------------------------
# DeltaFolder.delete — rewrite strategy
# ---------------------------------------------------------------------------


class TestDeleteByRewrite(DeltaTestCase):
    def test_rewrite_drops_matched_rows(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(
            self.pa.table({"id": [1, 2, 3, 4, 5], "v": ["a", "b", "c", "d", "e"]}),
        )
        self.assertIs(d.delete("id IN (2, 4)"), d)   # returns the tabular

        out = d.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 3, 5])
        # Snapshot moved one version forward.
        self.assertGreaterEqual(d.snapshot(fresh=True).version, 1)

    def test_delete_without_predicate_removes_every_row(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]}))
        self.assertIs(d.delete(), d)  # no predicate → delete all, returns self
        self.assertEqual(d.read_arrow_table().num_rows, 0)

    def test_rewrite_no_match_no_commit(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2, 3]}))
        before = d.snapshot(fresh=True).version

        d.delete("id > 999")
        # No matched rows → no commit.
        self.assertEqual(d.snapshot(fresh=True).version, before)


# ---------------------------------------------------------------------------
# DeltaFolder.delete — DV strategy
# ---------------------------------------------------------------------------


class TestDeleteByDV(DeltaTestCase):
    def _opts(self):
        from yggdrasil.io.delta import DeltaOptions

        return DeltaOptions(delete_via_dv=True)

    def test_dv_delete_marks_without_rewrite(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(
            self.pa.table({"id": [10, 20, 30, 40, 50]}),
        )
        snap_before = d.snapshot(fresh=True)
        # Capture the on-disk parquet path so we can verify it stays
        # exactly as written.
        original_files = [
            d.snapshot().resolve(a) for a in snap_before.active_files.values()
        ]
        original_sizes = {p.full_path(): p.size for p in original_files}

        d.delete("id IN (20, 40)", options=self._opts())

        # The snapshot's AddFiles still point at the same paths, but
        # they now carry a deletion vector descriptor.
        snap_after = d.snapshot(fresh=True)
        survivors = list(snap_after.active_files.values())
        self.assertEqual(len(survivors), 1)
        self.assertIsNotNone(survivors[0].deletion_vector)

        # Original parquet bytes haven't changed — DV-based delete is
        # explicitly metadata-only on the file side.
        for p in original_files:
            self.assertTrue(p.exists())
            self.assertEqual(p.size, original_sizes[p.full_path()])

        # Read sees the masked rows hidden.
        out = d.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), [10, 30, 50])

    def test_dv_delete_protocol_bumped(self) -> None:
        from yggdrasil.io.delta import DeltaOptions

        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2, 3]}))
        d.delete("id = 2", options=DeltaOptions(delete_via_dv=True))

        snap = d.snapshot(fresh=True)
        self.assertGreaterEqual(snap.protocol.min_reader_version, 3)
        self.assertGreaterEqual(snap.protocol.min_writer_version, 7)
        self.assertIn("deletionVectors", snap.protocol.writer_features)
        self.assertIn("deletionVectors", snap.protocol.reader_features)


# ---------------------------------------------------------------------------
# Schema codec — yggdrasil.data.Schema ↔ Spark JSON
# ---------------------------------------------------------------------------


class TestSchemaCodec(DeltaTestCase):
    def test_schema_round_trip(self) -> None:
        from yggdrasil.data.data_field import Field
        from yggdrasil.data.schema import Schema
        from yggdrasil.data.types.primitive import Int64Type, StringType
        from yggdrasil.io.delta import (
            schema_to_spark_json,
            spark_json_to_schema,
        )

        s = Schema()
        s.with_field(Field(name="id", dtype=Int64Type()))
        s.with_field(Field(name="name", dtype=StringType()))

        text = schema_to_spark_json(s)
        self.assertIn('"name":"id"', text)
        self.assertIn('"long"', text)

        rebuilt = spark_json_to_schema(text)
        names = [f.name for f in rebuilt.fields]
        self.assertEqual(names, ["id", "name"])

    def test_arrow_schema_passes_metadata(self) -> None:
        from yggdrasil.io.delta import (
            arrow_schema_to_spark_json,
            spark_json_to_arrow_schema,
        )

        original = self.pa.schema(
            [self.pa.field("id", self.pa.int64(), metadata={"k": "v"})],
        )
        text = arrow_schema_to_spark_json(original)
        rebuilt = spark_json_to_arrow_schema(text)
        self.assertEqual(rebuilt.field("id").type, self.pa.int64())
        self.assertEqual(
            rebuilt.field("id").metadata,
            {b"k": b"v"},
        )


# ---------------------------------------------------------------------------
# Engine bridges — Tabular fan-out exposes Polars / pandas reads
# ---------------------------------------------------------------------------


class TestEngineBridges(DeltaTestCase):
    def test_read_polars_round_trip(self) -> None:
        try:
            import polars as pl  # noqa: F401
        except ImportError:
            self.skipTest("polars not installed")

        d = self.delta_io()
        d.write_arrow_table(
            self.pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
        )
        df = d.read_polars_frame()
        self.assertEqual(df.shape, (3, 2))
        self.assertEqual(df.columns, ["id", "name"])
        self.assertEqual(sorted(df["id"].to_list()), [1, 2, 3])

    def test_read_pandas_round_trip(self) -> None:
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")

        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [10, 20, 30]}))
        pdf = d.read_pandas_frame()
        self.assertEqual(pdf.shape, (3, 1))
        self.assertEqual(sorted(pdf["id"].tolist()), [10, 20, 30])


# ---------------------------------------------------------------------------
# make_new_version=False — write data files without committing a version
# ---------------------------------------------------------------------------


class TestMakeNewVersion(DeltaTestCase):
    def _opts(self, **kw):
        from yggdrasil.io.delta import DeltaOptions

        return DeltaOptions(make_new_version=False, **kw)

    def _log_commits(self, d):
        return sorted(
            c.name for c in (d.path / "_delta_log").iterdir()
            if c.name.endswith(".json")
        )

    def _data_files(self, d):
        return sorted(c.name for c in d.path.iterdir() if c.name.endswith(".parquet"))

    def test_append_writes_files_but_no_commit(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2, 3]}))
        version_before = d.snapshot(fresh=True).version
        log_before = self._log_commits(d)
        data_before = self._data_files(d)

        d.write_arrow_table(self.pa.table({"id": [4, 5, 6]}), options=self._opts())

        # Version + log are untouched ...
        self.assertEqual(d.snapshot(fresh=True).version, version_before)
        self.assertEqual(self._log_commits(d), log_before)
        # ... but the data parquet files were physically written.
        self.assertGreater(len(self._data_files(d)), len(data_before))
        # The uncommitted files aren't visible to a reader.
        self.assertEqual(sorted(d.read_arrow_table().column("id").to_pylist()),
                         [1, 2, 3])

    def test_default_true_still_commits(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2, 3]}))
        before = d.snapshot(fresh=True).version
        d.write_arrow_table(self.pa.table({"id": [4, 5, 6]}))  # default make_new_version=True
        self.assertEqual(d.snapshot(fresh=True).version, before + 1)
        self.assertEqual(sorted(d.read_arrow_table().column("id").to_pylist()),
                         [1, 2, 3, 4, 5, 6])

    def test_delete_is_noop(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2, 3, 4, 5]}))
        before = d.snapshot(fresh=True).version
        log_before = self._log_commits(d)

        d.delete("id IN (2, 4)", options=self._opts())

        # No commit, no version bump, rows still all present.
        self.assertEqual(d.snapshot(fresh=True).version, before)
        self.assertEqual(self._log_commits(d), log_before)
        self.assertEqual(sorted(d.read_arrow_table().column("id").to_pylist()),
                         [1, 2, 3, 4, 5])


# ---------------------------------------------------------------------------
# Sanity — the simple-list envelope's decoder is the inverse of the encoder
# ---------------------------------------------------------------------------


class TestEncodeDecodeBytes(DeltaTestCase):
    def test_encoded_payload_starts_with_simple_magic(self) -> None:
        # The encoder always emits the simple-list envelope, regardless
        # of cardinality. Verifying the magic byte locks in that
        # contract — readers from other engines (delta-rs, Spark) only
        # need to support the simple-list shape to consume our DVs.
        from yggdrasil.io.delta.deletion_vector import (
            _encode_simple_payload,
        )

        body = _encode_simple_payload([1, 2, 3])
        magic = struct.unpack_from("<I", body, 0)[0]
        self.assertEqual(magic, _MAGIC_SIMPLE)
        count = struct.unpack_from("<Q", body, 4)[0]
        self.assertEqual(count, 3)
