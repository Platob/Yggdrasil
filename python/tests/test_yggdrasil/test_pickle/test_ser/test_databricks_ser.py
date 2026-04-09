from __future__ import annotations

import json

import pytest

pytest.importorskip("databricks.sdk", reason="databricks-sdk not installed")

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.compute.cluster import Cluster
from yggdrasil.databricks.compute.command_execution import CommandExecution
from yggdrasil.databricks.compute.execution_context import ExecutionContext
from yggdrasil.databricks.compute.service import Clusters
from yggdrasil.databricks.lib import AccountClient, Config, WorkspaceClient
from yggdrasil.pickle.ser import dumps, loads, serialize
from yggdrasil.pickle.ser.databricks import (
    DatabricksAccountClientSerialized,
    DatabricksClientSerialized,
    DatabricksCommandExecutionSerialized,
    DatabricksConfigSerialized,
    DatabricksExecutionContextSerialized,
    DatabricksWorkspaceClientSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags
from databricks.sdk.service.compute import Language


def _inc(x: int) -> int:
    return x + 1


def _assert_config_equivalent(left: Config, right: Config) -> None:
    assert right.as_dict() == left.as_dict()
    assert getattr(right, "scopes", None) == getattr(left, "scopes", None)
    assert getattr(right, "authorization_details", None) == getattr(left, "authorization_details", None)
    assert getattr(right, "_custom_headers", None) == getattr(left, "_custom_headers", None)
    assert getattr(right, "_product_info", None) == getattr(left, "_product_info", None)


def _decoded_payload(ser: Serialized[object]) -> dict[str, object]:
    payload = json.loads(ser.decode().decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def _assert_ygg_client_equivalent(left: DatabricksClient, right: DatabricksClient) -> None:
    for key in (
        "host",
        "account_id",
        "workspace_id",
        "token",
        "client_id",
        "client_secret",
        "token_audience",
        "cluster_id",
        "serverless_compute_id",
        "azure_workspace_resource_id",
        "azure_use_msi",
        "azure_client_secret",
        "azure_client_id",
        "azure_tenant_id",
        "azure_environment",
        "google_credentials",
        "google_service_account",
        "profile",
        "config_file",
        "auth_type",
        "http_timeout_seconds",
        "retry_timeout_seconds",
        "debug_truncate_bytes",
        "debug_headers",
        "rate_limit",
        "product",
        "product_version",
        "custom_tags",
    ):
        assert getattr(right, key) == getattr(left, key)

    assert right.connected is False
    assert getattr(right, "_workspace_client", None) is None
    assert getattr(right, "_account_client", None) is None


def _make_clustered_client() -> tuple[DatabricksClient, Cluster]:
    client = DatabricksClient(
        host="https://example.cloud.databricks.com",
        token="abc",
        account_id="acc-123",
    )
    cluster = Cluster(
        service=Clusters(client=client),
        cluster_id="cluster-123",
        cluster_name="cluster-name",
    )
    return client, cluster


def _assert_execution_context_equivalent(left: ExecutionContext, right: ExecutionContext) -> None:
    assert right.context_id == left.context_id
    assert right.context_key == left.context_key
    assert right.temporary == left.temporary
    assert right.close_after == left.close_after
    assert right.language == left.language
    assert right.cluster.cluster_id == left.cluster.cluster_id
    assert right.cluster.cluster_name == left.cluster.cluster_name
    _assert_ygg_client_equivalent(left.client, right.client)


def _assert_command_execution_equivalent(left: CommandExecution, right: CommandExecution) -> None:
    assert right.command_id == left.command_id
    assert right.language == left.language
    assert right.command == left.command
    assert dict(right.environ or {}) == dict(left.environ or {})
    _assert_execution_context_equivalent(left.context, right.context)


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


def test_databricks_config_payload_ignores_heavy_private_fields() -> None:
    cfg = Config(
        host="https://example.cloud.databricks.com",
        token="abc",
        custom_headers={"X-Test": "1"},
    )
    cfg._huge_blob = object()
    cfg._api_client = object()

    ser = serialize(cfg)
    payload = _decoded_payload(ser)

    # flat payload — no version or kwargs wrapper
    assert "version" not in payload
    assert "kwargs" not in payload
    assert "_huge_blob" not in payload
    assert "_api_client" not in payload
    assert payload.get("host") == "https://example.cloud.databricks.com"
    assert payload.get("token") == "abc"
    assert payload.get("custom_headers") == {"X-Test": "1"}

    out = ser.as_python()
    assert isinstance(out, Config)
    assert not hasattr(out, "_huge_blob")
    assert not hasattr(out, "_api_client")


def test_databricks_config_roundtrip_reuses_cached_instance() -> None:
    cfg = Config(host="https://example.cloud.databricks.com", token="abc")

    payload = dumps(cfg)
    out1 = loads(payload)
    out2 = loads(payload)

    assert isinstance(out1, Config)
    assert out1 is out2


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


def test_databricks_workspace_client_payload_is_lean_and_cached() -> None:
    cfg = Config(
        host="https://example.cloud.databricks.com",
        token="abc",
        custom_headers={"X-Trace": "workspace"},
    )
    cfg._huge_blob = object()
    client = WorkspaceClient(config=cfg)

    ser = Serialized.from_python_object(client)
    assert isinstance(ser, DatabricksWorkspaceClientSerialized)

    payload = _decoded_payload(ser)
    cfg_inner = payload.get("cfg")

    # new compact keys — no version wrapper
    assert "version" not in payload
    assert payload.get("k") == "workspace"
    assert "config" not in payload
    assert "kwargs" not in payload
    assert isinstance(cfg_inner, dict)
    assert "_huge_blob" not in cfg_inner

    out1 = ser.as_python()
    out2 = loads(dumps(client))
    out3 = loads(dumps(client))

    assert isinstance(out1, WorkspaceClient)
    assert out1 is out2 is out3


def test_ygg_databricks_client_roundtrip() -> None:
    client = DatabricksClient(
        host="https://example.cloud.databricks.com",
        token="abc",
        account_id="acc-123",
        workspace_id="ws-123",
        cluster_id="cluster-123",
        auth_type="pat",
        debug_headers=True,
        custom_tags={"Team": "Data"},
    )

    ser = Serialized.from_python_object(client)

    assert isinstance(ser, DatabricksClientSerialized)
    assert ser.tag == Tags.DATABRICKS_CLIENT

    out = ser.as_python()

    assert isinstance(out, DatabricksClient)
    _assert_ygg_client_equivalent(client, out)


def test_ygg_databricks_client_payload_is_lean_and_cached() -> None:
    client = DatabricksClient(
        host="https://example.cloud.databricks.com",
        token="abc",
        client_id="cid",
        client_secret="secret",
        custom_tags={"Env": "dev"},
    )
    object.__setattr__(client, "_workspace_client", object())
    object.__setattr__(client, "_account_client", object())
    object.__setattr__(client, "_workspace_config", object())
    object.__setattr__(client, "_account_config", object())
    object.__setattr__(client, "_was_connected", True)
    object.__setattr__(client, "_sql", object())

    ser = Serialized.from_python_object(client)
    assert isinstance(ser, DatabricksClientSerialized)

    payload = _decoded_payload(ser)

    # flat payload — no version or kwargs wrapper
    assert "version" not in payload
    assert "kwargs" not in payload
    assert "_workspace_client" not in payload
    assert "_account_client" not in payload
    assert "_workspace_config" not in payload
    assert "_account_config" not in payload
    assert "_was_connected" not in payload
    assert "_sql" not in payload
    assert payload.get("host") == "https://example.cloud.databricks.com"
    assert payload.get("custom_tags") == {"Env": "dev"}

    out1 = ser.as_python()
    out2 = loads(dumps(client))
    out3 = loads(dumps(client))

    assert isinstance(out1, DatabricksClient)
    assert out1 is out2 is out3
    assert out1.connected is False


def test_databricks_execution_context_roundtrip() -> None:
    _client, cluster = _make_clustered_client()
    context = ExecutionContext(
        cluster=cluster,
        context_id="ctx-123",
        context_key="ctx-key",
        language=Language.PYTHON,
        temporary=True,
        close_after=42.0,
    )

    ser = Serialized.from_python_object(context)

    assert isinstance(ser, DatabricksExecutionContextSerialized)
    assert ser.tag == Tags.DATABRICKS_EXECUTION_CONTEXT

    out = ser.as_python()

    assert isinstance(out, ExecutionContext)
    _assert_execution_context_equivalent(context, out)


def test_databricks_execution_context_payload_is_lean_and_cached() -> None:
    _client, cluster = _make_clustered_client()
    context = ExecutionContext(
        cluster=cluster,
        context_id="ctx-123",
        context_key="ctx-key",
        language=Language.SQL,
    )
    context._remote_metadata = object()
    context._created_at = 1.0
    context._last_used_at = 2.0

    ser = Serialized.from_python_object(context)
    assert isinstance(ser, DatabricksExecutionContextSerialized)

    payload = _decoded_payload(ser)
    # compact keys — no version wrapper
    assert "version" not in payload
    assert payload.get("id") == "ctx-123"
    assert payload.get("l") == "sql"
    assert "_remote_metadata" not in payload
    assert "_created_at" not in payload
    assert "_last_used_at" not in payload

    out1 = ser.as_python()
    out2 = loads(dumps(context))
    out3 = loads(dumps(context))

    assert isinstance(out1, ExecutionContext)
    assert out1 is out2 is out3


def test_databricks_command_execution_roundtrip() -> None:
    _client, cluster = _make_clustered_client()
    context = ExecutionContext(
        cluster=cluster,
        context_id="ctx-123",
        context_key="ctx-key",
        language=Language.PYTHON,
    )
    command = CommandExecution(
        context=context,
        command_id="cmd-123",
        language=Language.PYTHON,
        command="print('hi')",
        environ={"A": "B"},
        pyfunc=_inc,
    )

    ser = Serialized.from_python_object(command)

    assert isinstance(ser, DatabricksCommandExecutionSerialized)
    assert ser.tag == Tags.DATABRICKS_COMMAND_EXECUTION

    out = ser.as_python()

    assert isinstance(out, CommandExecution)
    _assert_command_execution_equivalent(command, out)
    assert out.pyfunc is not None
    assert out.pyfunc(3) == 4


def test_databricks_command_execution_payload_is_lean() -> None:
    _client, cluster = _make_clustered_client()
    context = ExecutionContext(cluster=cluster, context_id="ctx-123")
    command = CommandExecution(
        context=context,
        command_id="cmd-123",
        command="SELECT 1",
        language=Language.SQL,
        environ={"A": "B"},
    )
    command._details = object()
    command._remote_payload_path = "/tmp/payload"
    command._shutdown_hook = object()

    ser = Serialized.from_python_object(command)
    assert isinstance(ser, DatabricksCommandExecutionSerialized)

    payload = _decoded_payload(ser)
    # compact keys — no version wrapper
    assert "version" not in payload
    assert payload.get("cid") == "cmd-123"
    assert payload.get("cmd") == "SELECT 1"
    assert payload.get("l") == "sql"
    assert payload.get("env") == {"A": "B"}
    assert "pf" not in payload
    assert "_details" not in payload
    assert "_remote_payload_path" not in payload
    assert "_shutdown_hook" not in payload


def test_databricks_command_execution_payload_preserves_pyfunc() -> None:
    _client, cluster = _make_clustered_client()
    context = ExecutionContext(cluster=cluster, context_id="ctx-123")
    command = CommandExecution(
        context=context,
        command_id="cmd-456",
        command="print('pyfunc')",
        language=Language.PYTHON,
        pyfunc=_inc,
    )

    ser = Serialized.from_python_object(command)
    assert isinstance(ser, DatabricksCommandExecutionSerialized)

    payload = _decoded_payload(ser)
    # compact keys — no version wrapper
    assert "version" not in payload
    assert isinstance(payload.get("pf"), str)

    out = ser.as_python()
    assert isinstance(out, CommandExecution)
    assert out.pyfunc is not None
    assert out.pyfunc(10) == 11


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


def test_databricks_account_client_roundtrip_reuses_cached_instance() -> None:
    client = AccountClient(
        config=Config(
            host="https://accounts.cloud.databricks.com",
            account_id="acc-123",
            token="abc",
        )
    )

    payload = dumps(client)
    out1 = loads(payload)
    out2 = loads(payload)

    assert isinstance(out1, AccountClient)
    assert out1 is out2


def test_databricks_tags_category_and_lookup() -> None:
    assert Tags.get_category(Tags.DATABRICKS_CONFIG) == Tags.CATEGORY_DATABRICKS
    assert Tags.get_category(Tags.DATABRICKS_WORKSPACE_CLIENT) == Tags.CATEGORY_DATABRICKS
    assert Tags.get_category(Tags.DATABRICKS_ACCOUNT_CLIENT) == Tags.CATEGORY_DATABRICKS
    assert Tags.get_category(Tags.DATABRICKS_CLIENT) == Tags.CATEGORY_DATABRICKS
    assert Tags.get_category(Tags.DATABRICKS_EXECUTION_CONTEXT) == Tags.CATEGORY_DATABRICKS
    assert Tags.get_category(Tags.DATABRICKS_COMMAND_EXECUTION) == Tags.CATEGORY_DATABRICKS

    assert Tags.is_databricks(Tags.DATABRICKS_CONFIG)
    assert Tags.is_databricks(Tags.DATABRICKS_WORKSPACE_CLIENT)
    assert Tags.is_databricks(Tags.DATABRICKS_ACCOUNT_CLIENT)
    assert Tags.is_databricks(Tags.DATABRICKS_CLIENT)
    assert Tags.is_databricks(Tags.DATABRICKS_EXECUTION_CONTEXT)
    assert Tags.is_databricks(Tags.DATABRICKS_COMMAND_EXECUTION)

    assert Tags.get_name(Tags.DATABRICKS_CONFIG) == "DATABRICKS_CONFIG"
    assert Tags.get_name(Tags.DATABRICKS_WORKSPACE_CLIENT) == "DATABRICKS_WORKSPACE_CLIENT"
    assert Tags.get_name(Tags.DATABRICKS_ACCOUNT_CLIENT) == "DATABRICKS_ACCOUNT_CLIENT"
    assert Tags.get_name(Tags.DATABRICKS_CLIENT) == "DATABRICKS_CLIENT"
    assert Tags.get_name(Tags.DATABRICKS_EXECUTION_CONTEXT) == "DATABRICKS_EXECUTION_CONTEXT"
    assert Tags.get_name(Tags.DATABRICKS_COMMAND_EXECUTION) == "DATABRICKS_COMMAND_EXECUTION"


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

