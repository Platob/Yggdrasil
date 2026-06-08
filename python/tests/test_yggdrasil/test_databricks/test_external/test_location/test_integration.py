"""Live Databricks integration tests for UC **external locations** over AWS.

Two tiers, so it does something useful whatever your privileges are:

* **read-only** (just ``DATABRICKS_HOST`` + LIST/READ on a location) — list
  external locations, resolve one's storage ``.path`` to an :class:`S3Path`,
  and exercise IO (``ls`` + a bounded read) through it. Point at a specific one
  with ``YGG_TEST_EXTERNAL_LOCATION``, else the first listable S3-backed one is
  used. Skips cleanly when there's nothing readable or the runner lacks AWS
  creds for the bucket.
* **create / write** (``YGG_TEST_AWS_ROLE_ARN`` + ``YGG_TEST_S3_URL`` + the UC
  privileges) — stand up a storage credential + external location, round-trip a
  large blob and a streaming Parquet through the S3 ``_upload_stream`` path,
  update, delete. Each create/delete step ``skipTest``\\s (not fails) on
  ``PermissionDenied``.

Run:
    DATABRICKS_HOST=... DATABRICKS_TOKEN=... \
    [YGG_TEST_EXTERNAL_LOCATION=my_location] \
    [YGG_TEST_AWS_ROLE_ARN=arn:aws:iam::123:role/UCRole YGG_TEST_S3_URL=s3://bucket/ygg/] \
    pytest tests/test_yggdrasil/test_databricks/test_external_location_integration.py -m integration -v
"""
from __future__ import annotations

import os
import secrets

from databricks.sdk.errors import PermissionDenied
from databricks.sdk.service.catalog import CredentialPurpose
from tests.test_yggdrasil.test_databricks import DatabricksIntegrationCase
from yggdrasil.aws.fs.path import S3Path

_AWS_ROLE = os.environ.get("YGG_TEST_AWS_ROLE_ARN", "").strip()
_S3_URL = os.environ.get("YGG_TEST_S3_URL", "").strip()
_EL_NAME = os.environ.get("YGG_TEST_EXTERNAL_LOCATION", "").strip()
_MIB = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _bucket_of(url: str) -> str:
    return url.split("://", 1)[1].split("/", 1)[0]


class TestExternalLocationIntegration(DatabricksIntegrationCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()  # SkipTest when DATABRICKS_HOST is unset
        cls.locations = cls.client.external_locations
        cls.creds = cls.client.credentials
        cls._cleanup_loc: "list[str]" = []
        cls._cleanup_cred: "list[str]" = []

    @classmethod
    def tearDownClass(cls) -> None:
        for name in getattr(cls, "_cleanup_loc", []):
            try:
                cls.locations.delete(name, force=True)
            except Exception:
                pass
        for name in getattr(cls, "_cleanup_cred", []):
            try:
                cls.creds.delete(name, force=True)
            except Exception:
                pass
        super().tearDownClass()

    # ------------------------------------------------------------------
    # read-only: list + IO through an existing location
    # ------------------------------------------------------------------
    def test_list_locations(self) -> None:
        try:
            names = self.locations.names()
        except PermissionDenied as exc:
            self.skipTest(f"no permission to list external locations: {exc}")
        self.assertIsInstance(names, list)

    def _existing_s3_location(self):
        if _EL_NAME:
            return self.locations.get(_EL_NAME)
        try:
            for el in self.locations.list():
                if (el.url or "").startswith(("s3://", "s3a://")) and not el.name.startswith("__"):
                    return el
        except PermissionDenied:
            return None
        return None

    def test_existing_location_metadata_and_path(self) -> None:
        el = self._existing_s3_location()
        if el is None:
            self.skipTest("no readable S3 external location (set YGG_TEST_EXTERNAL_LOCATION)")
        self.assertTrue(el.url and el.credential_name)
        try:
            path = el.path
        except PermissionDenied as exc:
            # The path is vended through ``generate_temporary_path_credentials``;
            # the token may lack ``EXTERNAL USE LOCATION`` on this particular
            # location. That's an environment grant, not a code defect.
            self.skipTest(f"no EXTERNAL USE LOCATION grant for {el.name!r}: {exc}")
        self.assertIsInstance(path, S3Path)
        self.assertEqual(path.bucket, _bucket_of(el.url))

    def test_io_list_and_read_existing(self) -> None:
        el = self._existing_s3_location()
        if el is None:
            self.skipTest("no readable S3 external location available")
        try:
            path = el.path
        except PermissionDenied as exc:
            self.skipTest(f"no EXTERNAL USE LOCATION grant for {el.name!r}: {exc}")
        try:
            children = list(path.ls(limit=25))
        except Exception as exc:  # ambient AWS creds may not reach the bucket
            self.skipTest(f"cannot list external-location storage (AWS creds for the bucket?): {exc}")

        # Listing worked — try a bounded read of the first non-empty file.
        for child in children:
            try:
                if child.is_file() and int(child.size) > 0:
                    head = bytes(child.read_mv(min(4096, int(child.size)), 0))
                    self.assertGreater(len(head), 0)
                    return
            except Exception:
                continue
        # An empty location (no files) is a valid pass — the list path worked.

    # ------------------------------------------------------------------
    # create / write (needs UC privileges + bucket access)
    # ------------------------------------------------------------------
    def _writable_location(self):
        if not (_AWS_ROLE and _S3_URL):
            self.skipTest("YGG_TEST_AWS_ROLE_ARN + YGG_TEST_S3_URL not set")
        cred_name = f"ygg_test_elcred_{secrets.token_hex(4)}"
        try:
            self.creds.create_aws(
                cred_name, _AWS_ROLE, purpose=CredentialPurpose.STORAGE,
                comment="ygg external-location integration", skip_validation=True,
            )
        except PermissionDenied as exc:
            self.skipTest(f"no permission to create UC credential: {exc}")
        type(self)._cleanup_cred.append(cred_name)

        name = f"ygg_test_el_{secrets.token_hex(4)}"
        type(self)._cleanup_loc.append(name)
        try:
            return self.locations.create(
                name, _S3_URL, cred_name, comment="ygg integration test", skip_validation=True,
            )
        except PermissionDenied as exc:
            self.skipTest(f"no permission to create external location: {exc}")

    def test_create_read_update_delete(self) -> None:
        el = self._writable_location()
        self.assertEqual(el.url, _S3_URL)
        self.assertEqual(self.locations.get(el.name).url, _S3_URL)
        self.assertTrue(self.locations.exists(el.name))

        el.update(comment="updated by ygg")
        self.assertEqual(self.locations.get(el.name).comment, "updated by ygg")

        self.locations.delete(el.name, force=True)
        type(self)._cleanup_loc.remove(el.name)
        self.assertFalse(self.locations.exists(el.name))

    def test_streaming_write_read(self) -> None:
        """The S3 ``_upload_stream`` path (large blob + streaming Parquet)
        against an external location's storage."""
        import pyarrow as pa

        from yggdrasil.io.parquet_file import ParquetFile

        el = self._writable_location()
        base = el.path / f"ygg-s3-stream-{secrets.token_hex(4)}"
        blob = base / "blob.bin"
        parquet = base / "data.parquet"
        try:
            payload = secrets.token_bytes(_env_int("YGG_TEST_S3_LARGE_MB", 8) * _MIB)
            try:
                blob.write_bytes(payload)
            except Exception as exc:  # ambient AWS creds may not reach the bucket
                self.skipTest(f"cannot write to external-location storage (AWS creds?): {exc}")
            self.assertEqual(blob.read_bytes(), payload)

            rows = _env_int("YGG_TEST_S3_PARQUET_ROWS", 200_000)
            ParquetFile(holder=parquet).write_arrow_table(
                pa.table({"id": pa.array(range(rows), type=pa.int64())})
            )
            parquet.invalidate_singleton()  # cold read, off the write cache
            self.assertEqual(ParquetFile(holder=parquet).read_arrow_table().num_rows, rows)
        finally:
            for p in (blob, parquet):
                try:
                    p.unlink()
                except Exception:
                    pass
