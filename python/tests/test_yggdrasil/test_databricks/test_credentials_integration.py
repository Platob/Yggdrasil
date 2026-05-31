"""Live Databricks integration tests for UC **credentials** over AWS.

Exercises the full create → read → (refreshable STS) → delete cycle against a
real workspace. AWS-dedicated: builds the credential from an IAM role ARN and
checks the temporary-credential / refreshable-AWSClient path.

Run (skipped otherwise):
    DATABRICKS_HOST=... DATABRICKS_TOKEN=... \
    YGG_TEST_AWS_ROLE_ARN=arn:aws:iam::123:role/MyUCRole \
    [YGG_TEST_AWS_REGION=eu-central-1] \
    pytest tests/test_yggdrasil/test_databricks/test_credentials_integration.py -m integration -v
"""
from __future__ import annotations

import os
import secrets
import unittest

from databricks.sdk.errors import DatabricksError, PermissionDenied
from databricks.sdk.service.catalog import CredentialPurpose

from tests.test_yggdrasil.test_databricks import DatabricksIntegrationCase

from yggdrasil.aws.client import AWSClient
from yggdrasil.aws.config import AwsCredentials
from yggdrasil.databricks.credentials import Credential

_AWS_ROLE = os.environ.get("YGG_TEST_AWS_ROLE_ARN", "").strip()
_REGION = os.environ.get("YGG_TEST_AWS_REGION", "us-east-1").strip() or "us-east-1"


@unittest.skipUnless(_AWS_ROLE, "YGG_TEST_AWS_ROLE_ARN not set — skipping AWS credential integration")
class TestAwsCredentialsIntegration(DatabricksIntegrationCase):
    """Live UC credentials over AWS — create from a role ARN, generate
    refreshable STS creds, clean up after."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()  # SkipTest when DATABRICKS_HOST is unset
        cls.svc = cls.client.credentials
        cls._created: "list[str]" = []

    @classmethod
    def tearDownClass(cls) -> None:
        for name in getattr(cls, "_created", []):
            try:
                cls.svc.delete(name, force=True)
            except Exception:
                pass
        super().tearDownClass()

    def _name(self, tag: str) -> str:
        name = f"ygg_test_cred_{tag}_{secrets.token_hex(4)}"
        type(self)._created.append(name)
        return name

    def _create(self, tag: str, *, purpose=CredentialPurpose.SERVICE) -> Credential:
        try:
            return self.svc.create_aws(
                self._name(tag), _AWS_ROLE, purpose=purpose,
                comment="ygg integration test", skip_validation=True,
            )
        except PermissionDenied as exc:
            self.skipTest(f"no permission to create UC credentials: {exc}")

    # ------------------------------------------------------------------
    def test_create_get_list_delete(self) -> None:
        cred = self._create("crud")
        self.assertEqual(cred.aws_role_arn, _AWS_ROLE)

        fetched = self.svc.get(cred.name)
        self.assertEqual(fetched.aws_role_arn, _AWS_ROLE)
        self.assertTrue(self.svc.exists(cred.name))
        self.assertIn(cred.name, self.svc.names())

        self.svc.delete(cred.name, force=True)
        type(self)._created.remove(cred.name)
        self.assertFalse(self.svc.exists(cred.name))

    def test_update_comment(self) -> None:
        cred = self._create("update")
        cred.update(comment="updated by ygg")
        self.assertEqual(self.svc.get(cred.name).comment, "updated by ygg")

    def test_refreshable_aws_credentials(self) -> None:
        cred = self._create("temp", purpose=CredentialPurpose.SERVICE)
        try:
            ac = cred.aws_credentials()
        except DatabricksError as exc:
            self.skipTest(f"temporary service credential not available: {exc}")
        self.assertIsInstance(ac, AwsCredentials)
        self.assertTrue(ac.is_complete())  # access key + secret present

        client = cred.aws_client(region=_REGION)
        self.assertIsInstance(client, AWSClient)
        self.assertEqual(client.region, _REGION)
        frozen = client.session.get_credentials().get_frozen_credentials()
        self.assertTrue(frozen.access_key and frozen.secret_key)
