"""Behaviors for :meth:`DatabricksClient.files_headers`.

Files-API traffic bypasses the SDK transport (see
:meth:`DatabricksClient.files_session`), so the client must stamp the
headers the SDK's ``ApiClient`` would otherwise send — ``User-Agent``
and ``X-Databricks-Workspace-Id`` — alongside the ``Authorization``
header. Databricks' edge rate-limits unattributed traffic harder, so
missing these shows up as extra 429s.
"""
from __future__ import annotations

import os

import pytest

from yggdrasil.databricks import DatabricksClient


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Strip ambient ``DATABRICKS_*`` credentials so the explicit PAT
    kwargs below are the only auth in play (see test_client_url.py), and
    stub the SDK's host-metadata discovery — it issues a real HTTP GET
    against the (fake) host during ``Config`` init."""
    from yggdrasil.databricks.client import invalidate_env_defaults
    import databricks.sdk.config as sdk_config

    for var in list(os.environ):
        if var.startswith("DATABRICKS_"):
            monkeypatch.delenv(var, raising=False)
    invalidate_env_defaults()

    def _offline(host):
        raise OSError(f"offline test: no metadata for {host}")

    monkeypatch.setattr(sdk_config, "get_host_metadata", _offline, raising=False)
    yield
    invalidate_env_defaults()


class TestFilesHeaders:

    def test_carries_auth_and_user_agent(self) -> None:
        client = DatabricksClient(
            host="https://files-headers-a.cloud.databricks.com",
            token="tok-a", auth_type="pat",
        )
        client._workspace_id_probed = True  # no live workspace to probe
        headers = client.files_headers()
        assert headers["Authorization"] == "Bearer tok-a"
        # The SDK Config's User-Agent: product + databricks-sdk-py +
        # platform + auth type — the attribution the edge keys on.
        assert "databricks-sdk" in headers["User-Agent"]
        assert "auth/pat" in headers["User-Agent"]
        assert "X-Databricks-Workspace-Id" not in headers

    def test_workspace_id_header_when_known(self) -> None:
        client = DatabricksClient(
            host="https://files-headers-b.cloud.databricks.com",
            token="tok-b", auth_type="pat",
        )
        client.workspace_id = 12345
        headers = client.files_headers()
        assert headers["X-Databricks-Workspace-Id"] == "12345"

    def test_workspace_id_derived_from_azure_host(self) -> None:
        client = DatabricksClient(
            host="https://adb-5678901234567890.12.azuredatabricks.net",
            token="tok-az", auth_type="pat",
        )
        assert client.workspace_id == "5678901234567890"
        headers = client.files_headers()
        assert headers["X-Databricks-Workspace-Id"] == "5678901234567890"

    def test_fresh_dict_per_call(self) -> None:
        client = DatabricksClient(
            host="https://files-headers-c.cloud.databricks.com",
            token="tok-c", auth_type="pat",
        )
        client._workspace_id_probed = True
        first = client.files_headers()
        first["Range"] = "bytes=0-1"
        assert "Range" not in client.files_headers()
