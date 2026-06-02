"""Live :class:`S3Path` integration — driven through a Databricks
**external volume**.

S3Path needs a real bucket + credentials. Rather than wire a standalone
AWS account, we borrow Unity Catalog's: create an EXTERNAL volume (its
storage location is a real ``s3://`` prefix), vend short-lived AWS
credentials for it via ``temporary_volume_credentials``, and point a
plain :class:`S3Path` at that prefix. So the same pure-HTTP S3 data plane
that runs against any bucket is exercised here against the object store
backing the volume — no mocks.

Skips (not fails) when the environment can't support it:

* no ``DATABRICKS_HOST`` (base class),
* the identity can't CREATE SCHEMA / CREATE VOLUME on the catalog,
* the identity lacks ``EXTERNAL USE SCHEMA`` (and can't self-grant it),
  so temporary credentials can't be minted,
* the volume is backed by Azure / GCP (no ``aws_temp_credentials``), or
* the S3 endpoint isn't reachable from the runner's network.

Catalog defaults to :envvar:`DATABRICKS_INTEGRATION_CATALOG` (``trading``).
"""
from __future__ import annotations

import os
import secrets
import unittest
from typing import ClassVar

import pyarrow as pa
from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied

from yggdrasil.aws.fs.path import S3Path
from yggdrasil.enums import Mode
from yggdrasil.io.io_stats import IOKind

from .. import DatabricksIntegrationCase


__all__ = ["TestS3PathViaExternalVolume"]


class TestS3PathViaExternalVolume(DatabricksIntegrationCase):
    """Round-trip real S3 objects via an external volume's storage + creds."""

    schema: ClassVar
    volume: ClassVar
    s3_service: ClassVar
    base_url: ClassVar[str]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        catalog = os.environ.get("DATABRICKS_INTEGRATION_CATALOG", "trading").strip()
        schema_name = f"yg_s3_{secrets.token_hex(4)}"
        try:
            cls.schema = cls.client.schemas(catalog_name=catalog).schema(
                schema_name=schema_name,
            )
            cls.schema.ensure_created(comment="yggdrasil S3Path integration")
            cls.volume = cls.client.volumes(
                catalog_name=catalog, schema_name=schema_name,
            ).create(volume_name="yg_ext", volume_type="EXTERNAL")
            storage = cls.volume.storage_location().rstrip("/")
        except (DatabricksError, PermissionDenied) as exc:
            cls._safe_drop_schema()
            raise unittest.SkipTest(
                f"cannot provision external volume on {catalog}: {exc}"
            ) from exc

        if not storage.startswith("s3://"):
            cls._safe_drop_schema()
            raise unittest.SkipTest(
                f"external volume is not S3-backed ({storage!r}); "
                f"S3Path coverage needs an AWS workspace."
            )

        from yggdrasil.databricks.aws import AWSDatabricksVolumeCredentials

        provider = AWSDatabricksVolumeCredentials(str(cls.volume.volume_id), client=cls.client)
        try:
            # Force a credential mint now (the lazy refresher would defer it to
            # first S3 byte) so a missing EXTERNAL USE SCHEMA grant — or a
            # non-S3 backing — degrades to a skip instead of erroring mid-test.
            provider.get_credentials(mode=Mode.OVERWRITE)
            cls.s3_service = provider.aws_client(mode=Mode.OVERWRITE).s3
        except (PermissionDenied, RuntimeError) as exc:
            cls._safe_drop_schema()
            raise unittest.SkipTest(
                f"cannot mint S3 credentials for the external volume "
                f"(needs EXTERNAL USE SCHEMA, S3 backing): {exc}"
            ) from exc

        cls.base_url = f"{storage}/_ygg_s3_{secrets.token_hex(4)}"

    @classmethod
    def _safe_drop_schema(cls) -> None:
        schema = getattr(cls, "schema", None)
        if schema is not None:
            try:
                schema.delete(force=True, raise_error=False)
            except Exception:
                pass

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls._safe_drop_schema()
        finally:
            super().tearDownClass()

    def _path(self, leaf: str) -> S3Path:
        p = S3Path(f"{self.base_url}/{leaf}", service=self.s3_service)
        p.invalidate_singleton()
        return p

    # -- bytes ---------------------------------------------------------
    def test_write_read_bytes_roundtrip(self) -> None:
        payload = b"s3-via-uc-" + secrets.token_bytes(24)
        p = self._path("bytes.bin")
        try:
            p.write_bytes(payload)
            self.assertTrue(p.exists())
            self.assertEqual(p.stat().kind, IOKind.FILE)
            self.assertEqual(bytes(self._path("bytes.bin").read_bytes()), payload)
        finally:
            p.unlink(missing_ok=True)

    def test_overwrite_replaces_contents(self) -> None:
        p = self._path("overwrite.bin")
        try:
            p.write_bytes(b"first")
            p.write_bytes(b"second-longer")
            self.assertEqual(bytes(self._path("overwrite.bin").read_bytes()), b"second-longer")
        finally:
            p.unlink(missing_ok=True)

    # -- opened handles (write coalescing on the whole-blob path) ------
    def test_open_write_then_read(self) -> None:
        try:
            with self._path("opened.bin").open("wb") as io:
                io.write(b"opened-")
                io.write(b"payload")
            self.assertEqual(bytes(self._path("opened.bin").read_bytes()), b"opened-payload")
        finally:
            self._path("opened.bin").unlink(missing_ok=True)

    def test_open_append(self) -> None:
        p = self._path("append.bin")
        try:
            p.write_bytes(b"head-")
            with self._path("append.bin").open("ab") as io:
                io.write(b"tail")
            self.assertEqual(bytes(self._path("append.bin").read_bytes()), b"head-tail")
        finally:
            p.unlink(missing_ok=True)

    # -- tabular -------------------------------------------------------
    def test_parquet_roundtrip(self) -> None:
        table = pa.table({"id": pa.array(range(2000), pa.int64()),
                          "v": pa.array([f"r{i}" for i in range(2000)], pa.string())})
        p = self._path("t.parquet")
        try:
            p.write_table(table)
            got = self._path("t.parquet").read_arrow_table()
            self.assertEqual(got.num_rows, 2000)
            self.assertEqual(got.column("id").to_pylist()[:3], [0, 1, 2])
        finally:
            p.unlink(missing_ok=True)

    # -- navigation ----------------------------------------------------
    def test_iterdir_and_unlink(self) -> None:
        leaf = self._path("listing/entry.bin")
        try:
            leaf.write_bytes(b"x")
            names = {c.name for c in self._path("listing").iterdir()}
            self.assertIn("entry.bin", names)
            leaf.unlink()
            leaf.invalidate_singleton()
            self.assertIs(leaf.stat().kind, IOKind.MISSING)
        finally:
            leaf.unlink(missing_ok=True)
