"""Live :class:`VolumePath` integration — IO, navigation, call efficiency,
tabular round-trips against a real Unity Catalog volume (no mocks).

Provisions a per-run scratch directory under the shared
``trading_tgp_dev``.``ygg_integration``.``ygg_integration`` volume
(created if missing, never dropped — only the scratch dir is removed); a
permission error degrades to a skip. Volume *lifecycle* (creating
schemas / managed + external volumes) lives in
``test_volume_lifecycle_integration``; bulk / concurrency in
``test_volume_load_integration``.

Sections:

* ``TestVolumeRoundTrip`` — the shared backend contract (CRUD + remove).
* ``TestVolumeIO`` — opened IO (``wb`` / ``rb`` / ``ab``) + overwrite.
* ``TestVolumeCallEfficiency`` — asserts the *number* of backend calls
  per op (whole-file write = one PUT; cold stat = one call; warm
  metadata = zero) so chattiness regressions are caught.
* ``TestVolumeTabular`` — Parquet / CSV / Arrow / polars round-trips and
  their call counts.
"""
from __future__ import annotations

import secrets
import unittest
from typing import ClassVar

from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied

from yggdrasil.enums import MimeTypes
from yggdrasil.pandas.tests import PandasTestCase

from ._base import FsIntegrationCase, FsRoundTripMixin


__all__ = [
    "TestVolumeRoundTrip",
    "TestVolumeIO",
    "TestVolumeCallEfficiency",
    "TestVolumeTabular",
    "TestVolumePandas",
]


class VolumeFsCase(FsIntegrationCase):
    """Per-class scratch directory under the shared integration volume."""

    ext = "bin"

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        volume = cls.integration_volume()
        cls.root = volume.path(f"_ygg_{secrets.token_hex(4)}")
        try:
            cls.root.mkdir(parents=True, exist_ok=True)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"cannot write to {cls.root}: {exc}") from exc


class TestVolumeRoundTrip(VolumeFsCase, FsRoundTripMixin):
    """The shared backend round-trip / remove contract over a volume."""


class TestVolumeIO(VolumeFsCase):
    """Opened file handles — the shape pandas / openpyxl / arrow use."""

    def test_open_write_then_read_roundtrip(self) -> None:
        with self._fresh("opened.bin").open("wb") as io:
            io.write(b"opened-write-payload")
        with self._fresh("opened.bin").open("rb") as io:
            self.assertEqual(io.read(), b"opened-write-payload")

    def test_open_append(self) -> None:
        p = self._fresh("append.bin")
        p.write_bytes(b"head-")
        with p.open("ab") as io:
            io.write(b"tail")
        self.assertEqual(bytes(self._fresh("append.bin").read_bytes()), b"head-tail")

    def test_overwrite_replaces_contents(self) -> None:
        p = self._fresh("overwrite.bin")
        p.write_bytes(b"first")
        p.write_bytes(b"second-longer")
        self.assertEqual(bytes(self._fresh("overwrite.bin").read_bytes()), b"second-longer")

    def test_exists_flips_on_write_and_unlink(self) -> None:
        p = self._fresh("flip.bin")
        self.assertFalse(p.exists())
        p.write_bytes(b"x")
        self.assertTrue(p.exists())
        p.unlink()
        p.invalidate_singleton()
        self.assertFalse(p.exists())

    def test_mkdir_parents_creates_intermediate_dirs(self) -> None:
        # Three-deep so ``parents=True`` is genuinely exercised — the
        # Files API rejects create_directory when an ancestor is missing.
        deep = self.root / f"a/b/c-{secrets.token_hex(3)}"
        deep.mkdir(parents=True, exist_ok=True)
        self.assertTrue(deep.exists())
        deep.mkdir(parents=True, exist_ok=True)  # idempotent


class TestVolumeCallEfficiency(VolumeFsCase):
    """Backend-call counts per operation — regressions in chattiness fail."""

    def test_write_bytes_is_single_overwrite_put(self) -> None:
        p = self._fresh("a.bin")
        with self._count() as calls:
            p.write_bytes(b"hello world")
        # whole-file write → exactly one PUT, no read-modify-write
        self.assertEqual(calls.get("PUT"), 1, dict(calls))
        self.assertEqual(calls.get("GET", 0), 0, dict(calls))
        self.assertEqual(bytes(self._fresh("a.bin").read_bytes()), b"hello world")

    def test_metadata_after_write_is_cached(self) -> None:
        p = self._fresh("b.bin")
        p.write_bytes(b"x" * 32)
        # exists / size / is_dir read the stat the write seeded — no new calls
        with self._count() as calls:
            self.assertTrue(p.exists())
            self.assertEqual(p.size, 32)
            self.assertFalse(p.is_dir())
            self.assertTrue(p.is_file())
        self.assertEqual(sum(calls.values()), 0, dict(calls))

    def test_stat_on_cold_handle_is_one_call(self) -> None:
        self._fresh("c.bin").write_bytes(b"data!")
        p = self._fresh("c.bin")
        with self._count() as calls:
            self.assertEqual(p.size, 5)
        self.assertLessEqual(sum(calls.values()), 1, dict(calls))

    def test_ls_is_single_listing(self) -> None:
        self._fresh("d1.bin").write_bytes(b"1")
        self._fresh("d2.bin").write_bytes(b"2")
        with self._count() as calls:
            names = sorted(c.name for c in self.root.ls())
        self.assertIn("d1.bin", names)
        self.assertEqual(calls.get("GET"), 1, dict(calls))

    def test_unlink_is_one_stat_one_delete(self) -> None:
        self._fresh("u.bin").write_bytes(b"x")
        p = self._fresh("u.bin")
        with self._count() as calls:
            p.unlink()
        # One kind-probe (file vs dir) then the delete — no second re-stat.
        self.assertEqual(calls.get("DELETE"), 1, dict(calls))
        self.assertLessEqual(calls.get("HEAD", 0), 1, dict(calls))


class TestVolumeTabular(VolumeFsCase):
    """Tabular content + call counts: a whole-file table write is one PUT,
    a read is one GET — across Arrow, polars, CSV."""

    def test_parquet_write_is_one_put_read_is_one_get(self) -> None:
        with self._count() as calls:
            self._fresh("t.parquet").write_table(self._table())
        self.assertEqual(calls.get("PUT"), 1, dict(calls))
        self.assertEqual(calls.get("GET", 0), 0, dict(calls))

        with self._count() as calls:
            got = self._fresh("t.parquet").read_arrow_table()
        self.assertEqual(calls.get("GET"), 1, dict(calls))
        self.assertEqual(got.num_rows, 3)
        self.assertEqual(got.column("id").to_pylist(), [1, 2, 3])
        self.assertEqual(got.schema.names, ["id", "v", "g"])

    def test_parquet_roundtrip_to_polars(self) -> None:
        self._fresh("p.parquet").write_table(self._table())
        frame = self._fresh("p.parquet").read_polars_frame()
        self.assertEqual(frame.shape, (3, 3))
        self.assertEqual(sorted(frame["id"].to_list()), [1, 2, 3])

    def test_csv_roundtrip(self) -> None:
        self._fresh("t.csv").as_media(MimeTypes.CSV).write_arrow_table(self._table())
        got = self._fresh("t.csv").as_media(MimeTypes.CSV).read_arrow_table()
        self.assertEqual(got.num_rows, 3)
        self.assertEqual(sorted(got.column("g").to_pylist()), ["a", "a", "b"])

    def test_open_parquet_arrow_round_trip(self) -> None:
        """The ygg-native parquet surface end to end: upload + footer-aware
        read of a non-trivial table through ``open(media_type="parquet")``."""
        import pyarrow as pa
        from yggdrasil.enums import Mode

        table = pa.table({
            "id": pa.array(range(5000), pa.int64()),
            "label": pa.array([f"row-{i}" for i in range(5000)], pa.string()),
            "amount": pa.array([i * 1.5 for i in range(5000)], pa.float64()),
        })
        p = self._fresh("frame.parquet")
        with p.open("wb", media_type="parquet") as pf:
            pf.write_arrow_table(table, mode=Mode.OVERWRITE)
        with self._fresh("frame.parquet").open("rb", media_type="parquet") as pf:
            out = pf.read_arrow_table()
        self.assertEqual(out.num_rows, 5000)
        self.assertEqual(out.column("id").to_pylist(), list(range(5000)))
        self.assertEqual(out.column("label")[0].as_py(), "row-0")


class TestVolumePandas(VolumeFsCase, PandasTestCase):
    """pandas writers round-trip through ``VolumePath.open`` — the
    ``with path.open("wb") as fh: df.to_parquet(fh)`` shape caller code
    uses for parquet, csv and (buffered) xlsx."""

    def test_parquet_round_trip(self) -> None:
        df = self.df({"id": [1, 2, 3], "name": ["a", "b", "c"], "score": [0.1, 0.2, 0.3]})
        with self._fresh("frame.parquet").open("wb") as fh:
            df.to_parquet(fh, index=False)
        with self._fresh("frame.parquet").open("rb") as fh:
            loaded = self.pd.read_parquet(fh)
        self.assertFrameEqual(loaded, df)

    def test_csv_round_trip(self) -> None:
        df = self.df({"id": [10, 20, 30], "label": ["x", "y", "z"]})
        with self._fresh("frame.csv").open("wb") as fh:
            df.to_csv(fh, index=False)
        with self._fresh("frame.csv").open("rb") as fh:
            loaded = self.pd.read_csv(fh)
        self.assertFrameEqual(loaded, df)

    def test_excel_round_trip(self) -> None:
        """xlsx writers seek inside a single zip container — they can't
        stream chunks — so callers buffer a BytesIO and flush it through
        ``open("wb").write(...)``. Pin that shape."""
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            self.skipTest("openpyxl not installed — xlsx round-trip skipped.")
        import io

        df = self.df({"id": [1, 2], "label": ["foo", "bar"]})
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        with self._fresh("frame.xlsx").open("wb") as fh:
            fh.write(buffer.getvalue())
        with self._fresh("frame.xlsx").open("rb") as fh:
            loaded = self.pd.read_excel(io.BytesIO(fh.read()))
        self.assertFrameEqual(loaded, df)
