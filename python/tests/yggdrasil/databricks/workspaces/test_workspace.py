import os

import pytest

from yggdrasil.databricks.workspaces.workspace import DBXWorkspace
from yggdrasil.databricks.workspaces import workspace as workspace_module


def test_find_in_env_prefers_azure_prefix_for_msal():
    env = {
        "AZURE_CLIENT_ID": "cid",
        "AZURE_CLIENT_SECRET": "secret",
        "AZURE_TENANT_ID": "tenant",
        "AZURE_SCOPES": "scope1,scope2",
        "DATABRICKS_HOST": "https://example.cloud.databricks.com",
        "DATABRICKS_AUTH_TYPE": "external-browser",
    }

    ws = DBXWorkspace.find_in_env(env=env)

    assert ws.host == env["DATABRICKS_HOST"]
    assert ws.auth_type == env["DATABRICKS_AUTH_TYPE"]
    assert ws.msal_auth is not None
    assert ws.msal_auth.client_id == env["AZURE_CLIENT_ID"]
    assert ws.msal_auth.scopes == ["scope1", "scope2"]


def test_connect_maps_u2m_oauth_to_external_browser(monkeypatch):
    captured_kwargs = {}

    def fake_workspace_client(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs

        class Dummy:
            pass

        return Dummy()

    monkeypatch.setattr(workspace_module.databricks_sdk, "WorkspaceClient", fake_workspace_client)

    workspace = DBXWorkspace(
        host="https://example.cloud.databricks.com",
        auth_type="u2m-oauth",
    )

    workspace.connect(reset=True)

    assert captured_kwargs["auth_type"] == "external-browser"
    assert captured_kwargs["host"] == "https://example.cloud.databricks.com"


def test_connect_sets_external_browser_cache(monkeypatch):
    captured_kwargs = {}

    class DummyTokenCache:
        BASE_PATH = "dummy"

    def fake_workspace_client(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs

        class Dummy:
            pass

        return Dummy()

    monkeypatch.setattr(workspace_module.databricks_sdk.oauth, "TokenCache", DummyTokenCache)
    monkeypatch.setattr(workspace_module.databricks_sdk, "WorkspaceClient", fake_workspace_client)

    workspace = DBXWorkspace(
        host="https://example.cloud.databricks.com",
        auth_type="u2m-oauth",
    )

    workspace.connect(reset=True)

    assert DummyTokenCache.BASE_PATH.endswith(
        "/.ygg/databricks/auth/cache/external-browser"
    )
    assert captured_kwargs["auth_type"] == "external-browser"


def test_connect_builds_cache_when_missing(monkeypatch, tmp_path):
    captured_kwargs = {}

    class DummyToken:
        def __init__(self, payload=None):
            self.payload = payload or {}

        def as_dict(self):
            return self.payload

        @classmethod
        def from_dict(cls, raw):
            return cls(payload=raw)

    class DummySessionCredentials:
        def __init__(self, token):
            self._token = token

        def as_dict(self):
            return {"token": self._token.as_dict()}

        @classmethod
        def from_dict(
            cls,
            raw,
            token_endpoint=None,
            client_id=None,
            client_secret=None,
            redirect_url=None,
        ):
            return cls(token=DummyToken.from_dict(raw["token"]))

        def token(self):
            return self._token

    class DummyOidc:
        def __init__(self, token_endpoint):
            self.token_endpoint = token_endpoint

    class DummySDK:
        def __init__(self):
            self.oauth = type(
                "DummyOAuth",
                (),
                {"SessionCredentials": DummySessionCredentials, "Token": DummyToken},
            )()

        def WorkspaceClient(self, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return object()

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(workspace_module, "databricks_sdk", DummySDK())

    workspace = DBXWorkspace(
        host="https://example.cloud.databricks.com",
        auth_type="external-browser",
    )

    workspace.connect(reset=True)

    cache_cls = workspace_module.databricks_sdk.oauth.TokenCache
    cache_instance = cache_cls(
        host=workspace.host,
        oidc_endpoints=DummyOidc("https://tokens"),
        client_id="cid",
    )

    credentials = DummySessionCredentials(DummyToken({"access_token": "tok"}))
    cache_instance.save(credentials)

    reloaded = cache_instance.load()

    assert cache_cls.BASE_PATH.endswith("/.ygg/databricks/auth/cache/external-browser")
    assert os.path.exists(cache_instance.filename)
    assert reloaded is not None
    assert reloaded.token().as_dict()["access_token"] == "tok"
    assert captured_kwargs["auth_type"] == "external-browser"
