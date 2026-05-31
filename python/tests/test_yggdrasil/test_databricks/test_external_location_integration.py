"""Live Databricks integration tests for UC **external locations** over AWS.

Creates a storage credential (from an IAM role ARN) + an external location bound
to an S3 URL, reads it back through the resource, checks the storage ``.path``
resolves to an :class:`S3Path`, then cleans up.

Run (skipped otherwise):
    DATABRICKS_HOST=... DATABRICKS_TOKEN=... \
    YGG_TEST_AWS_ROLE_ARN=arn:aws:iam::123:role/MyUCRole \
    YGG_TEST_S3_URL=s3://my-bucket/ygg-ext-test/ \
    pytest tests/test_yggdrasil/test_databricks/test_external_location_integration.py -m integration -v
"""
from __future__ import annotations

import os
import secrets
import unittest

from databricks.sdk.errors import PermissionDenied
from databricks.sdk.service.catalog import CredentialPurpose

from tests.test_yggdrasil.test_databricks import DatabricksIntegrationCase

from yggdrasil.aws.fs.path import S3Path

_AWS_ROLE = os.environ.get("YGG_TEST_AWS_ROLE_ARN", "").strip()
_S3_URL = os.environ.get("YGG_TEST_S3_URL", "").strip()  # e.g. s3://my-bucket/prefix/


@unittest.skipUnless(
    _AWS_ROLE and _S3_URL,
    "YGG_TEST_AWS_ROLE_ARN + YGG_TEST_S3_URL not set — skipping external-location integration",
)
class TestExternalLocationIntegration(DatabricksIntegrationCase):
    """Live UC external locations over AWS S3."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()  # SkipTest when DATABRICKS_HOST is unset
        cls.creds = cls.client.credentials
        cls.locations = cls.client.external_locations
        cls._locations: "list[str]" = []
        cls._cred_name = f"ygg_test_elcred_{secrets.token_hex(4)}"
        try:
            cls.creds.create_aws(
                cls._cred_name, _AWS_ROLE, purpose=CredentialPurpose.STORAGE,
                comment="ygg external-location integration", skip_validation=True,
            )
        except PermissionDenied as exc:
            raise unittest.SkipTest(f"no permission to create UC credentials: {exc}")

    @classmethod
    def tearDownClass(cls) -> None:
        for name in getattr(cls, "_locations", []):
            try:
                cls.locations.delete(name, force=True)
            except Exception:
                pass
        try:
            cls.creds.delete(cls._cred_name, force=True)
        except Exception:
            pass
        super().tearDownClass()

    def _create(self, tag: str, *, url: str = _S3_URL, read_only: bool = False):
        name = f"ygg_test_el_{tag}_{secrets.token_hex(4)}"
        type(self)._locations.append(name)
        try:
            return self.locations.create(
                name, url, self._cred_name,
                comment="ygg integration test", read_only=read_only, skip_validation=True,
            )
        except PermissionDenied as exc:
            self.skipTest(f"no permission to create external locations: {exc}")

    # ------------------------------------------------------------------
    def test_create_read_list_delete(self) -> None:
        el = self._create("crud")
        self.assertEqual(el.url, _S3_URL)
        self.assertEqual(el.credential_name, self._cred_name)

        fetched = self.locations.get(el.name)
        self.assertEqual(fetched.url, _S3_URL)
        self.assertTrue(self.locations.exists(el.name))
        self.assertIn(el.name, self.locations.names())

        self.locations.delete(el.name, force=True)
        type(self)._locations.remove(el.name)
        self.assertFalse(self.locations.exists(el.name))

    def test_update_comment(self) -> None:
        el = self._create("update")
        el.update(comment="updated by ygg")
        self.assertEqual(self.locations.get(el.name).comment, "updated by ygg")

    def test_storage_path_resolves(self) -> None:
        el = self._create("path")
        if _S3_URL.startswith("s3://") or _S3_URL.startswith("s3a://"):
            self.assertIsInstance(el.path, S3Path)
            self.assertEqual(el.path.bucket, _S3_URL.split("://", 1)[1].split("/", 1)[0])
