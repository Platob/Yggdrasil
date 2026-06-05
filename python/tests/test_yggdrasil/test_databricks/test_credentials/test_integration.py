"""Live Databricks integration tests for UC **credentials** over AWS.

Two tiers, so it does something useful at whatever privilege level you have:

* **create** (``YGG_TEST_AWS_ROLE_ARN`` + UC privileges) — create a credential
  from an IAM role ARN, read it back, update, delete.
* **use** (just ``DATABRICKS_HOST`` + LIST/USE on a credential) — when creation
  isn't allowed, fall back to the **first existing** SERVICE-purpose AWS
  credential and drive the refreshable path off it.

The refreshable test runs in either tier and pins the moving parts: the vended
STS token has a TTL (expiry), the provider is a per-credential singleton, and
the refreshable :class:`AWSClient` is cached per region (so the refresh cycle is
shared, not re-seeded on every call).

Run:
    DATABRICKS_HOST=... DATABRICKS_TOKEN=... \
    [YGG_TEST_AWS_ROLE_ARN=arn:aws:iam::123:role/UCRole] [YGG_TEST_AWS_REGION=eu-central-1] \
    pytest tests/test_yggdrasil/test_databricks/test_credentials/test_integration.py -m integration -v
"""
from __future__ import annotations

import os
import secrets

from databricks.sdk.errors import DatabricksError, PermissionDenied
from databricks.sdk.service.catalog import CredentialPurpose

from tests.test_yggdrasil.test_databricks import DatabricksIntegrationCase

from yggdrasil.aws.client import AWSClient
from yggdrasil.aws.config import AwsCredentials
from yggdrasil.databricks.credentials import Credential, DatabricksCredentialAwsProvider

_AWS_ROLE = os.environ.get("YGG_TEST_AWS_ROLE_ARN", "").strip()
_REGION = os.environ.get("YGG_TEST_AWS_REGION", "us-east-1").strip() or "us-east-1"


class TestAwsCredentialsIntegration(DatabricksIntegrationCase):
    """Live UC credentials over AWS — create when allowed, otherwise reuse an
    existing one to exercise refreshable STS, singleton, and TTL."""

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

    # -- helpers --------------------------------------------------------
    def _create(self, tag: str, *, purpose=CredentialPurpose.SERVICE) -> Credential:
        if not _AWS_ROLE:
            self.skipTest("YGG_TEST_AWS_ROLE_ARN not set — cannot create a credential")
        name = f"ygg_test_cred_{tag}_{secrets.token_hex(4)}"
        type(self)._created.append(name)
        try:
            return self.svc.create_aws(
                name, _AWS_ROLE, purpose=purpose, comment="ygg integration test", skip_validation=True,
            )
        except PermissionDenied as exc:
            self.skipTest(f"no permission to create UC credentials: {exc}")

    def _service_aws_credential(self) -> Credential:
        """A SERVICE-purpose AWS credential to drive the refreshable path:
        create a fresh one if we can, else fall back to the first existing one
        we're allowed to list/use (which is left untouched)."""
        if _AWS_ROLE:
            try:
                name = f"ygg_test_cred_svc_{secrets.token_hex(4)}"
                type(self)._created.append(name)
                return self.svc.create_aws(
                    name, _AWS_ROLE, purpose=CredentialPurpose.SERVICE, skip_validation=True,
                )
            except PermissionDenied:
                type(self)._created.pop()  # nothing created → don't try to delete it
        # Fall back to the first existing SERVICE-purpose AWS credential.
        try:
            for cred in self.svc.list(purpose=CredentialPurpose.SERVICE):
                if cred.info.aws_iam_role is not None:
                    return cred
        except PermissionDenied:
            pass
        self.skipTest("cannot create and no existing SERVICE AWS credential available to use")

    # -- create tier ----------------------------------------------------
    def test_create_get_list_delete(self) -> None:
        cred = self._create("crud")
        self.assertEqual(cred.aws_role_arn, _AWS_ROLE)

        self.assertEqual(self.svc.get(cred.name).aws_role_arn, _AWS_ROLE)
        self.assertTrue(self.svc.exists(cred.name))
        self.assertIn(cred.name, self.svc.names())

        self.svc.delete(cred.name, force=True)
        type(self)._created.remove(cred.name)
        self.assertFalse(self.svc.exists(cred.name))

    def test_update_comment(self) -> None:
        cred = self._create("update")
        cred.update(comment="updated by ygg")
        self.assertEqual(self.svc.get(cred.name).comment, "updated by ygg")

    # -- refreshable / singleton / ttl (create OR first existing) -------
    def test_refreshable_singleton_and_ttl(self) -> None:
        cred = self._service_aws_credential()
        try:
            ac = cred.aws_credentials()
        except DatabricksError as exc:
            self.skipTest(f"temporary service credential not available: {exc}")

        # STS token, with a TTL (so botocore can refresh before expiry).
        self.assertIsInstance(ac, AwsCredentials)
        self.assertTrue(ac.is_complete())
        self.assertIsNotNone(ac.expiration, "vended STS token should carry an expiry (TTL)")

        # Provider is a singleton per host|credential.
        provider = cred.aws_provider()
        self.assertIsInstance(provider, DatabricksCredentialAwsProvider)
        self.assertIs(cred.aws_provider(), provider)

        # The refreshable client is cached per region — one shared refresh
        # cycle, not a fresh seed on every call.
        client = cred.aws_client(region=_REGION)
        self.assertIsInstance(client, AWSClient)
        self.assertEqual(client.region, _REGION)
        self.assertIs(cred.aws_client(region=_REGION), client)

        # botocore session is wired to a refreshable (TTL-driven) provider and
        # vends valid frozen creds.
        creds = client.session.get_credentials()
        self.assertIn("Refreshable", type(creds).__name__)
        frozen = creds.get_frozen_credentials()
        self.assertTrue(frozen.access_key and frozen.secret_key)

        # Re-vending through the provider works (the refresh hook).
        self.assertTrue(provider.get_credentials().is_complete())
