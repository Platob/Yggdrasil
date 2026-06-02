"""Live volume *lifecycle* integration — creating schemas + managed and
external volumes and round-tripping through them.

Distinct from ``test_volume_fs_integration`` (which writes under an
*existing* volume): this file needs CREATE SCHEMA / CREATE VOLUME on the
catalog, so it provisions a throw-away schema (``yg_volife_<hex>``) per
class and drops it cascade-style on teardown. The catalog is read from
:envvar:`DATABRICKS_INTEGRATION_CATALOG` (default ``trading``); a
permission error degrades to a skip.

The external-volume test is the seed for the future S3Path coverage —
it creates a ``volume_type="EXTERNAL"`` volume (storage location derived
from the workspace default, or :envvar:`DATABRICKS_INTEGRATION_EXTERNAL_LOCATION`
when set) and round-trips bytes through its backing object store.
"""
from __future__ import annotations

import os
import secrets
import unittest
from typing import ClassVar

from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied

from yggdrasil.databricks.schema.schema import UCSchema
from yggdrasil.databricks.volume.volume import Volume
from yggdrasil.io.io_stats import IOKind

from .. import DatabricksIntegrationCase


__all__ = ["TestManagedVolumeLifecycle", "TestExternalVolumeLifecycle"]


class _VolumeLifecycleCase(DatabricksIntegrationCase):
    """Per-class throw-away schema the volume tests create volumes under."""

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    schema: ClassVar[UCSchema]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = os.environ.get(
            "DATABRICKS_INTEGRATION_CATALOG", "trading",
        ).strip()
        cls.schema_name = f"yg_volife_{secrets.token_hex(4)}"
        try:
            cls.schema = cls.client.schemas(
                catalog_name=cls.catalog_name,
            ).schema(schema_name=cls.schema_name)
            cls.schema.ensure_created(comment="yggdrasil volume lifecycle test")
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"cannot create schema {cls.catalog_name}.{cls.schema_name}: "
                f"{exc}. Override DATABRICKS_INTEGRATION_CATALOG with a catalog "
                f"the test identity can CREATE SCHEMA + VOLUME on."
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            schema = getattr(cls, "schema", None)
            if schema is not None:
                schema.delete(force=True, raise_error=False)
        finally:
            super().tearDownClass()

    def _volumes(self):
        return self.client.volumes(
            catalog_name=self.catalog_name, schema_name=self.schema_name,
        )


class TestManagedVolumeLifecycle(_VolumeLifecycleCase):
    """Create a managed volume, round-trip bytes, then drop it."""

    def test_create_volume_then_round_trip(self) -> None:
        vol: Volume = self._volumes().volume(volume_name=f"yg_managed_{secrets.token_hex(3)}")
        vol.create()
        try:
            self.assertEqual((vol.volume_type or "MANAGED").upper(), "MANAGED")
            path = vol.path(f"scratch/{secrets.token_hex(4)}.bin")
            payload = b"managed-" + secrets.token_bytes(16)
            path.write_bytes(payload)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), payload)
            self.assertIs(path.client, vol.client)
        finally:
            vol.delete(raise_error=False)

    def test_insert_volume_path_round_trip(self) -> None:
        """:meth:`Table.insert_volume_path` is the SQL-engine staging entry
        point — the minted path must round-trip against the live Files API."""
        from yggdrasil.databricks.table.table import Table

        vol = self._volumes().volume(volume_name=f"yg_stage_{secrets.token_hex(3)}")
        vol.create()
        try:
            table = Table(
                service=self.client.tables,
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_name="integration",
            )
            staged = table.insert_volume_path(temporary=False)
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(b"staged")
            self.assertEqual(staged.read_bytes(), b"staged")
            staged.unlink(missing_ok=True)
        finally:
            vol.delete(raise_error=False)


class TestExternalVolumeLifecycle(_VolumeLifecycleCase):
    """Create an EXTERNAL volume (object-store backed) and round-trip through
    it — the seed for S3Path coverage via a Databricks external location."""

    def test_create_external_volume_then_round_trip(self) -> None:
        storage_location = os.environ.get(
            "DATABRICKS_INTEGRATION_EXTERNAL_LOCATION", "",
        ).strip() or None
        try:
            vol: Volume = self._volumes().create(
                volume_name=f"yg_external_{secrets.token_hex(3)}",
                volume_type="EXTERNAL",
                storage_location=storage_location,
            )
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"cannot create external volume (no usable external location): "
                f"{exc}. Set DATABRICKS_INTEGRATION_EXTERNAL_LOCATION to an "
                f"S3/ADLS/GCS prefix the workspace can write to."
            ) from exc
        try:
            self.assertEqual((vol.volume_type or "").upper(), "EXTERNAL")
            # The backing object store is real cloud storage (e.g. S3).
            self.assertTrue(vol.storage_location())
            path = vol.path(f"scratch/{secrets.token_hex(4)}.bin")
            payload = b"external-" + secrets.token_bytes(16)
            path.write_bytes(payload)
            self.assertIs(path.stat().kind, IOKind.FILE)
            self.assertEqual(path.read_bytes(), payload)
            path.unlink(missing_ok=True)
        finally:
            vol.delete(raise_error=False)
