from __future__ import annotations

import pytest

pytest.importorskip("databricks.sdk", reason="databricks-sdk not installed")

from yggdrasil.databricks.lib import AccountClient, Config, WorkspaceClient
from yggdrasil.pickle.ser import dumps, loads, serialize
from yggdrasil.pickle.ser.databricks import (
    DatabricksAccountClientSerialized,
    DatabricksConfigSerialized,
    DatabricksWorkspaceClientSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


def _assert_config_equivalent(left: Config, right: Config) -> None:
    assert right.as_dict() == left.as_dict()
    assert getattr(right, "scopes", None) == getattr(left, "scopes", None)
    assert getattr(right, "authorization_details", None) == getattr(left, "authorization_details", None)
    assert getattr(right, "_custom_headers", None) == getattr(left, "_custom_headers", None)
    assert getattr(right, "_product_info", None) == getattr(left, "_product_info", None)


def test_databricks_config_roundtrip() -> None:
    cfg = Config(
        host="https://example.cloud.databricks.com",
        token="abc",
        account_id="123",
        auth_type="pat",
        debug_headers=True,
        custom_headers={"X-Test": "1"},
        product="ygg",
        product_version="1.2.3",
        scopes=["all-apis"],
        authorization_details=[{"type": "test"}],
    )

    ser = serialize(cfg)

    assert isinstance(ser, DatabricksConfigSerialized)
    assert ser.tag == Tags.DATABRICKS_CONFIG
    assert Tags.get_class(Tags.DATABRICKS_CONFIG) is DatabricksConfigSerialized

    out = ser.as_python()

    assert isinstance(out, Config)
    _assert_config_equivalent(cfg, out)


def test_databricks_workspace_client_roundtrip() -> None:
    cfg = Config(
        host="https://example.cloud.databricks.com",
        token="abc",
        custom_headers={"X-Trace": "workspace"},
        product="ygg",
        product_version="1.2.3",
    )
    client = WorkspaceClient(config=cfg)

    ser = Serialized.from_python_object(client)

    assert isinstance(ser, DatabricksWorkspaceClientSerialized)
    assert ser.tag == Tags.DATABRICKS_WORKSPACE_CLIENT

    out = ser.as_python()

    assert isinstance(out, WorkspaceClient)
    _assert_config_equivalent(client.config, out.config)


def test_databricks_account_client_roundtrip() -> None:
    cfg = Config(
        host="https://accounts.cloud.databricks.com",
        account_id="acc-123",
        token="abc",
        custom_headers={"X-Trace": "account"},
    )
    client = AccountClient(config=cfg)

    ser = Serialized.from_python_object(client)

    assert isinstance(ser, DatabricksAccountClientSerialized)
    assert ser.tag == Tags.DATABRICKS_ACCOUNT_CLIENT

    out = ser.as_python()

    assert isinstance(out, AccountClient)
    _assert_config_equivalent(client.config, out.config)


def test_databricks_tags_category_and_lookup() -> None:
    assert Tags.get_category(Tags.DATABRICKS_CONFIG) == Tags.CATEGORY_DATABRICKS
    assert Tags.get_category(Tags.DATABRICKS_WORKSPACE_CLIENT) == Tags.CATEGORY_DATABRICKS
    assert Tags.get_category(Tags.DATABRICKS_ACCOUNT_CLIENT) == Tags.CATEGORY_DATABRICKS

    assert Tags.is_databricks(Tags.DATABRICKS_CONFIG)
    assert Tags.is_databricks(Tags.DATABRICKS_WORKSPACE_CLIENT)
    assert Tags.is_databricks(Tags.DATABRICKS_ACCOUNT_CLIENT)

    assert Tags.get_name(Tags.DATABRICKS_CONFIG) == "DATABRICKS_CONFIG"
    assert Tags.get_name(Tags.DATABRICKS_WORKSPACE_CLIENT) == "DATABRICKS_WORKSPACE_CLIENT"
    assert Tags.get_name(Tags.DATABRICKS_ACCOUNT_CLIENT) == "DATABRICKS_ACCOUNT_CLIENT"


def test_databricks_dumps_and_loads_roundtrip_workspace_client() -> None:
    client = WorkspaceClient(
        config=Config(
            host="https://example.cloud.databricks.com",
            token="abc",
            custom_headers={"X-Test": "1"},
        )
    )

    payload = dumps(client)
    out = loads(payload)

    assert isinstance(out, WorkspaceClient)
    _assert_config_equivalent(client.config, out.config)

