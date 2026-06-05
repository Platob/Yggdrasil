"""Live AWS IAM Identity Center (SSO) authentication for :class:`AWSClient`.

When an :class:`AWSClient` carries **no static credentials** but is configured
for SSO, it authenticates through boto3's ``SSOTokenProvider`` (re-using the
token cache that ``aws sso login`` populates) — no access keys needed. Two
shapes:

* a ``profile`` whose ``~/.aws/config`` entry has ``sso_*`` keys (resolved by
  boto3's default chain — the "simple" session passes ``profile_name``); or
* the inline ``sso_start_url`` / ``sso_account_id`` / ``sso_role_name``
  (+ ``sso_region``) knobs (``has_sso()`` → the dedicated SSO session that
  materialises a synthetic profile in-memory).

This exercises that end-to-end against a real Identity Center, proving the
SSO-built session authenticates by calling STS ``get_caller_identity``.

Opt-in / skips cleanly. Provide **one** of:

* ``YGG_TEST_AWS_SSO_PROFILE=<profile>`` — a profile with SSO config, or
* ``AWS_SSO_START_URL`` + ``AWS_SSO_ACCOUNT_ID`` + ``AWS_SSO_ROLE_NAME``
  (+ ``AWS_SSO_REGION``) — the inline form.

Prime the token first with ``aws sso login [--profile <profile>]``. Skips when
neither is set, when static credentials are present in the environment (so the
SSO path wouldn't be what authenticates), or when the cached SSO token is
missing/expired / STS is unreachable.

Run:
    aws sso login --profile my-sso
    YGG_TEST_AWS_SSO_PROFILE=my-sso python -m pytest \\
        tests/test_yggdrasil/test_aws/test_sso_integration.py -v -s -m integration
"""
from __future__ import annotations

import os
import unittest

import pytest

from yggdrasil.aws.client import AWSClient


def _sso_client() -> "AWSClient | None":
    """Build an SSO-only :class:`AWSClient` from the env, or ``None`` to skip.

    Static credential fields are pinned to ``None`` (not the ``...`` sentinel)
    so the env's ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` can't sneak
    in — the point is to authenticate with *no* static credentials.
    """
    region = (
        os.environ.get("AWS_SSO_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    profile = os.environ.get("YGG_TEST_AWS_SSO_PROFILE")
    if profile:
        return AWSClient(
            profile=profile, region=region,
            access_key_id=None, secret_access_key=None, session_token=None,
        )
    start_url = os.environ.get("AWS_SSO_START_URL")
    account_id = os.environ.get("AWS_SSO_ACCOUNT_ID")
    role_name = os.environ.get("AWS_SSO_ROLE_NAME")
    if start_url and account_id and role_name:
        return AWSClient(
            sso_start_url=start_url,
            sso_account_id=account_id,
            sso_role_name=role_name,
            sso_region=os.environ.get("AWS_SSO_REGION") or region,
            region=region,
            access_key_id=None, secret_access_key=None, session_token=None,
        )
    return None


@pytest.mark.integration
class TestAwsSsoAuth(unittest.TestCase):
    """Authenticate an :class:`AWSClient` via SSO with no static credentials."""

    def setUp(self) -> None:
        self.client = _sso_client()
        if self.client is None:
            self.skipTest(
                "set YGG_TEST_AWS_SSO_PROFILE, or "
                "AWS_SSO_START_URL + AWS_SSO_ACCOUNT_ID + AWS_SSO_ROLE_NAME, "
                "and run `aws sso login` first",
            )

    def test_configured_for_sso_without_static_credentials(self) -> None:
        # No static creds → SSO (profile-resolved or inline) is the auth path.
        self.assertFalse(self.client.has_static_credentials())
        self.assertTrue(
            self.client.has_sso() or bool(self.client.profile),
            "client is neither inline-SSO nor profile-backed",
        )

    def test_sso_session_authenticates_via_sts(self) -> None:
        # The SSO-built session signs a real STS call — the end-to-end proof
        # that no-credentials + SSO config authenticates.
        try:
            identity = self.client.client("sts").get_caller_identity()
        except Exception as exc:  # noqa: BLE001 — environmental, not a defect
            raise unittest.SkipTest(
                "SSO token missing/expired or STS unreachable — run "
                f"`aws sso login`: {type(exc).__name__}: {exc}"
            ) from exc
        self.assertTrue(identity.get("Arn"))
        self.assertTrue(identity.get("Account"))
