"""Live-integration tests for :class:`VolumePath` read / write.

The volume singleton + dispatcher mechanics are pinned by
``test_volume.py`` (unit) and ``test_path_dispatch_integration.py``
(integration); this file targets the next layer down — that bytes
written through a :class:`VolumePath` actually land in Unity Catalog
and come back identical via the same path.

What runs against a live workspace:

- raw bytes round-trip (``write_bytes`` / ``read_bytes``),
- ``open("wb")`` / ``open("rb")`` round-trip — the file-handle shape
  pandas / openpyxl / arrow writers use,
- pandas Parquet round-trip via ``open()``,
- pandas CSV round-trip via ``open()``,
- ``exists`` / ``iterdir`` / ``mkdir(parents=True)`` against UC.

Skip rules
----------

Skipped wholesale unless ``DATABRICKS_HOST`` is set (see
:class:`DatabricksIntegrationCase`). The catalog is read from
:envvar:`DATABRICKS_INTEGRATION_CATALOG` (default ``trading``); the
test identity must have CREATE SCHEMA / CREATE VOLUME on it.

Cleanup
-------

A unique schema (``yg_volio_<hex>``) holds a single volume for the
whole class; every test writes under a per-test sub-directory and
``unlink`` / ``rmdir`` cleans up on the way out. Class teardown drops
the schema cascade-style so any orphaned files go with it.
"""
from __future__ import annotations

import io
import os
import secrets
import unittest
from typing import ClassVar

import pytest
from databricks.sdk.errors import DatabricksError, PermissionDenied

from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.databricks.schema.schema import UCSchema
from yggdrasil.databricks.volume.volume import Volume
from yggdrasil.pandas.tests import PandasTestCase

from tests.test_yggdrasil.test_databricks import DatabricksIntegrationCase


__all__ = [
    "TestVolumeBytesRoundTrip",
    "TestVolumePandasRoundTrip",
    "TestVolumeNavigation",
]


def _resolve_catalog() -> str:
    name = os.environ.get(
        "DATABRICKS_INTEGRATION_CATALOG", "trading_tgp_dev",
    ).strip()
    if not name:
        raise unittest.SkipTest(
            "DATABRICKS_INTEGRATION_CATALOG is empty — set it to a "
            "catalog the test identity has CREATE SCHEMA on."
        )
    return name


class _VolumeIOFixture(DatabricksIntegrationCase):
    """Per-class throw-away schema + volume; per-test sub-directory."""

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    volume_name: ClassVar[str]
    schema: ClassVar[UCSchema]
    volume: ClassVar[Volume]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = _resolve_catalog()
        cls.schema_name = f"yg_volio"
        cls.volume_name = f"yg_vol"
        try:
            cls.schema = cls.client.schemas(
                catalog_name=cls.catalog_name,
            ).schema(schema_name=cls.schema_name)
            cls.schema.ensure_created(
                comment="yggdrasil volume IO integration",
            )
            cls.volume = cls.client.volumes(
                catalog_name=cls.catalog_name,
                schema_name=cls.schema_name,
            ).volume(volume_name=cls.volume_name)
            cls.volume.create()
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Cannot create "
                f"{cls.catalog_name}.{cls.schema_name}.{cls.volume_name}: "
                f"{exc}. Override DATABRICKS_INTEGRATION_CATALOG with a "
                f"catalog the test identity can CREATE SCHEMA + VOLUME on."
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            schema = getattr(cls, "schema", None)
            if schema is not None:
                schema.delete(force=True, raise_error=False)
        finally:
            super().tearDownClass()

    def _scratch(self, leaf: str) -> VolumePath:
        """Path under the class volume, namespaced per test method."""
        sub = f"scratch/{self._testMethodName}/{leaf}"
        return self.volume.path(sub)


@pytest.mark.integration
class TestVolumeBytesRoundTrip(_VolumeIOFixture):
    """Bytes go in, bytes come back — through both ``write_bytes`` and
    ``open()`` so neither code path silently drops data."""

    def test_write_bytes_round_trip(self) -> None:
        payload = b"hello-volume-" + secrets.token_bytes(32)
        path = self._scratch(f"bytes-{secrets.token_hex(4)}.bin")
        try:
            path.write_bytes(payload)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), payload)
        finally:
            path.unlink(missing_ok=True)

    def test_open_streaming_round_trip(self) -> None:
        """``open("wb")`` / ``open("rb")`` is the shape pandas /
        openpyxl / pyarrow writers consume — pin it independently of
        the ``write_bytes`` fast path."""
        payload = b"streaming-" + secrets.token_bytes(64)
        path = self._scratch(f"stream-{secrets.token_hex(4)}.bin")
        try:
            with path.open("wb") as fh:
                fh.write(payload)
            with path.open("rb") as fh:
                self.assertEqual(fh.read(), payload)
        finally:
            path.unlink(missing_ok=True)

    def test_overwrite_replaces_contents(self) -> None:
        path = self._scratch(f"overwrite-{secrets.token_hex(4)}.bin")
        try:
            path.write_bytes(b"first")
            path.write_bytes(b"second-longer")
            self.assertEqual(path.read_bytes(), b"second-longer")
        finally:
            path.unlink(missing_ok=True)


@pytest.mark.integration
class TestVolumePandasRoundTrip(_VolumeIOFixture, PandasTestCase):
    """pandas writers (parquet, csv) round-trip through ``VolumePath.open``.

    Mirrors the file-handle shape used by ``DBXFileManager.to_parquet``
    / ``read_csv`` etc. in caller code: ``with path.open("wb") as fh:
    df.to_parquet(fh)`` and the matching ``open("rb")`` read.
    """

    def test_parquet_round_trip(self) -> None:
        df = self.df({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "score": [0.1, 0.2, 0.3],
        })
        path = self._scratch(f"frame-{secrets.token_hex(4)}.parquet")
        try:
            with path.open("wb") as fh:
                df.to_parquet(fh, index=False)
            with path.open("rb") as fh:
                loaded = self.pd.read_parquet(fh)
            self.assertFrameEqual(loaded, df)
        finally:
            path.unlink(missing_ok=True)

    def test_csv_round_trip(self) -> None:
        df = self.df({
            "id": [10, 20, 30],
            "label": ["x", "y", "z"],
        })
        path = self._scratch(f"frame-{secrets.token_hex(4)}.csv")
        try:
            with path.open("wb") as fh:
                df.to_csv(fh, index=False)
            with path.open("rb") as fh:
                loaded = self.pd.read_csv(fh)
            self.assertFrameEqual(loaded, df)
        finally:
            path.unlink(missing_ok=True)

    def test_excel_round_trip(self) -> None:
        """xlsx writers can't stream chunks — they seek inside a
        single zip container — so the caller hand-rolls a BytesIO and
        flushes it through ``open("wb").write(...)``. Pin that shape."""
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            self.skipTest("openpyxl not installed — xlsx round-trip skipped.")

        df = self.df({"id": [1, 2], "label": ["foo", "bar"]})
        path = self._scratch(f"frame-{secrets.token_hex(4)}.xlsx")
        try:
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            buffer.seek(0)
            with path.open("wb") as fh:
                fh.write(buffer.read())
            with path.open("rb") as fh:
                loaded = self.pd.read_excel(io.BytesIO(fh.read()))
            self.assertFrameEqual(loaded, df)
        finally:
            path.unlink(missing_ok=True)
    
    def test_storage_path(self):
        v = self.volume / "external"
        
        assert v.client is self.volume.client
    

@pytest.mark.integration
class TestVolumeNavigation(_VolumeIOFixture):
    """``exists`` / ``iterdir`` / ``mkdir`` against a live volume."""

    def test_exists_flips_on_write_and_unlink(self) -> None:
        path = self._scratch(f"flip-{secrets.token_hex(4)}.bin")
        self.assertFalse(path.exists())
        try:
            path.write_bytes(b"x")
            self.assertTrue(path.exists())
        finally:
            path.unlink(missing_ok=True)
        self.assertFalse(path.exists())

    def test_mkdir_parents_creates_intermediate_dirs(self) -> None:
        # Three-deep so we genuinely exercise ``parents=True`` — the
        # Files API rejects ``create_directory`` when an ancestor is
        # missing, so the recovery path is what's under test.
        deep = self._scratch(f"a/b/c-{secrets.token_hex(3)}")
        try:
            deep.mkdir(parents=True, exist_ok=True)
            self.assertTrue(deep.exists())
            # Idempotent: re-running with ``exist_ok=True`` is a no-op.
            deep.mkdir(parents=True, exist_ok=True)
        finally:
            # Best-effort cleanup of the leaf; the schema teardown
            # cascade drops everything else.
            try:
                deep.remove(recursive=True, missing_ok=True)
            except DatabricksError:
                pass

    def test_iterdir_lists_written_file(self) -> None:
        directory = self._scratch(f"listing-{secrets.token_hex(3)}")
        directory.mkdir(parents=True, exist_ok=True)
        leaf_name = f"entry-{secrets.token_hex(4)}.bin"
        leaf = directory / leaf_name
        try:
            leaf.write_bytes(b"listed")
            names = {child.name for child in directory.iterdir()}
            self.assertIn(leaf_name, names)
        finally:
            leaf.unlink(missing_ok=True)
            try:
                directory.remove(recursive=True, missing_ok=True)
            except DatabricksError:
                pass


class TestExternalVolume(_VolumeIOFixture):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.external_volume_name = f"yg_external"
        cls.external_volume = cls.client.volumes(
            catalog_name=cls.catalog_name,
            schema_name=cls.schema_name,
        ).create(
            volume_name=cls.external_volume_name,
            volume_type="EXTERNAL"
        )

    def test_details(self):
        v = self.external_volume
