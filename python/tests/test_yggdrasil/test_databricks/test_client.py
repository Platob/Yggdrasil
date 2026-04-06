import pytest

import yggdrasil.databricks.client as module_under_test
from yggdrasil.databricks.client import DatabricksClient, getenv, getenv_factory


class DummyConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class DummyWorkspaceClient:
    def __init__(self, config):
        self.config = config


class DummyAccountClient:
    def __init__(self, config):
        self.config = config


def reset_singleton():
    module_under_test.CURRENT_BASE_CLIENT = None


@pytest.fixture(autouse=True)
def _reset_singleton_fixture():
    reset_singleton()
    yield
    reset_singleton()


def test_getenv_returns_none_for_missing_or_empty(monkeypatch):
    monkeypatch.delenv("X_TEST_ENV", raising=False)
    assert getenv("X_TEST_ENV") is None

    monkeypatch.setenv("X_TEST_ENV", "")
    assert getenv("X_TEST_ENV") is None

    monkeypatch.setenv("X_TEST_ENV", "abc")
    assert getenv("X_TEST_ENV") == "abc"


def test_getenv_factory_reads_env(monkeypatch):
    monkeypatch.setenv("X_FACTORY_ENV", "hello")
    factory = getenv_factory("X_FACTORY_ENV")
    assert factory() == "hello"


def test_post_init_adds_https_when_missing_scheme():
    client = DatabricksClient(host="my-workspace.cloud.databricks.com")
    assert client.host == "https://my-workspace.cloud.databricks.com"


def test_post_init_adds_trailing_slash_when_scheme_present():
    client = DatabricksClient(host="https://my-workspace.cloud.databricks.com")
    assert client.host == "https://my-workspace.cloud.databricks.com"


def test_post_init_keeps_trailing_slash_when_already_present():
    client = DatabricksClient(host="https://my-workspace.cloud.databricks.com/")
    assert client.host == "https://my-workspace.cloud.databricks.com"


def test_to_url_includes_non_none_init_fields():
    client = DatabricksClient(
        host="https://adb-123.databricks.com/",
        token="secret",
        profile="dev",
        auth_type="pat",
        http_timeout_seconds=30,
    )

    url = client.to_url()

    assert url.scheme == "dbks"
    assert url.host == "adb-123.databricks.com"
    assert url.path == "/"
    assert url.userinfo == ":secret"

    query = dict(url.query_items())
    assert "host" not in query
    assert query["profile"] == "dev"
    assert query["auth_type"] == "pat"
    assert query["http_timeout_seconds"] == "30"


def test_from_parsed_url_round_trip():
    client = DatabricksClient(
        host="https://adb-123.databricks.com/",
        token="secret",
        profile="dev",
        auth_type="pat",
    )

    url = client.to_url()
    parsed = DatabricksClient.from_parsed_url(url)

    assert parsed.host == "https://adb-123.databricks.com"
    assert parsed.token == "secret"
    assert parsed.profile == "dev"
    assert parsed.auth_type == "pat"


def test_from_parsed_url_rejects_empty_path():
    class FakeURL:
        path = ""
        host = "adb-123.databricks.com"

        @staticmethod
        def query_items():
            return []

    with pytest.raises(ValueError, match="Invalid path"):
        DatabricksClient.from_parsed_url(FakeURL())


def test_config_maps_public_fields(monkeypatch):
    monkeypatch.setattr(module_under_test, "Config", DummyConfig)

    client = DatabricksClient(
        host="https://adb-123.databricks.com/",
        account_id="acc-1",
        token="tok",
        client_id="cid",
        client_secret="csec",
        token_audience="aud",
        azure_workspace_resource_id="res-id",
        azure_use_msi=True,
        azure_client_secret="az-sec",
        azure_client_id="az-id",
        azure_tenant_id="tenant",
        azure_environment="public",
        google_credentials="gcred",
        google_service_account="svc@x",
        profile="dev",
        config_file="/tmp/.databrickscfg",
        http_timeout_seconds=10,
        retry_timeout_seconds=20,
        debug_truncate_bytes=1000,
        debug_headers=True,
        rate_limit=99,
    )

    cfg = client.config

    assert cfg.host == "https://adb-123.databricks.com"
    assert cfg.account_id == "acc-1"
    assert cfg.token == "tok"
    assert cfg.client_id == "cid"
    assert cfg.client_secret == "csec"
    assert cfg.token_audience == "aud"
    assert cfg.azure_workspace_resource_id == "res-id"
    assert cfg.azure_use_msi is True
    assert cfg.azure_client_secret == "az-sec"
    assert cfg.azure_client_id == "az-id"
    assert cfg.azure_tenant_id == "tenant"
    assert cfg.azure_environment == "public"
    assert cfg.google_credentials == "gcred"
    assert cfg.google_service_account == "svc@x"
    assert cfg.profile == "dev"
    assert cfg.config_file == "/tmp/.databrickscfg"
    assert cfg.http_timeout_seconds == 10
    assert cfg.retry_timeout_seconds == 20
    assert cfg.debug_truncate_bytes == 1000
    assert cfg.debug_headers is True
    assert cfg.rate_limit == 99


def test_connected_false_when_no_clients():
    client = DatabricksClient()
    assert client.connected is False


def test_workspace_sdk_is_cached(monkeypatch):
    monkeypatch.setattr(module_under_test, "DWC", DummyWorkspaceClient)
    monkeypatch.setattr(module_under_test, "DAC", DummyAccountClient)
    monkeypatch.setattr(module_under_test, "Config", DummyConfig)

    client = DatabricksClient(host="https://adb-123.databricks.com/")

    w1 = client.workspace_client()
    w2 = client.workspace_client()

    assert isinstance(w1, DummyWorkspaceClient)
    assert w1 is w2


def test_current_builds_singleton_and_connects(monkeypatch):
    monkeypatch.setattr(module_under_test, "DWC", DummyWorkspaceClient)
    monkeypatch.setattr(module_under_test, "DAC", DummyAccountClient)
    monkeypatch.setattr(module_under_test, "Config", DummyConfig)

    c1 = DatabricksClient.current(host="https://adb-123.databricks.com/")
    c2 = DatabricksClient.current()

    assert c1 is c2
    assert c1.connected is False


def test_current_reset_replaces_singleton(monkeypatch):
    monkeypatch.setattr(module_under_test, "DWC", DummyWorkspaceClient)
    monkeypatch.setattr(module_under_test, "DAC", DummyAccountClient)
    monkeypatch.setattr(module_under_test, "Config", DummyConfig)

    c1 = DatabricksClient.current(host="https://adb-123.databricks.com/")
    c2 = DatabricksClient.current(reset=True, host="https://adb-456.databricks.com/")

    assert c1 is not c2
    assert c2.host == "https://adb-456.databricks.com"


def test_set_current_sets_singleton():
    client = DatabricksClient(host="https://adb-123.databricks.com/")
    DatabricksClient.set_current(client)

    assert module_under_test.CURRENT_BASE_CLIENT is client


def test_is_in_databricks_environment_true(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    assert DatabricksClient.is_in_databricks_environment() is True


def test_is_in_databricks_environment_false(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    assert DatabricksClient.is_in_databricks_environment() is False
