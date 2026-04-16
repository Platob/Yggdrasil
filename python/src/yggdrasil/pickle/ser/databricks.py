from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Mapping

from yggdrasil.databricks import lib as databricks_lib
from yggdrasil.databricks.client import (
    DATABRICKS_CLIENT_INIT_NAMES as _YGG_DATABRICKS_CLIENT_INIT_NAMES,
    DatabricksClient as _YGGDatabricksClient,
)
from yggdrasil.databricks.compute.cluster import Cluster as _YGGCluster
from yggdrasil.databricks.compute.command_execution import CommandExecution as _YGGCommandExecution
from yggdrasil.databricks.compute.execution_context import ExecutionContext as _YGGExecutionContext
from yggdrasil.databricks.compute.service import Clusters as _YGGClusters
from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

if TYPE_CHECKING:
    from databricks.sdk import AccountClient, WorkspaceClient
    from databricks.sdk.config import Config
    from databricks.sdk.service.compute import Language
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.compute.command_execution import CommandExecution
    from yggdrasil.databricks.compute.execution_context import ExecutionContext

_DBXAccountClient = databricks_lib.AccountClient
_DBXWorkspaceClient = databricks_lib.WorkspaceClient
_DBXConfig = databricks_lib.Config

_TAG_DATABRICKS_CONFIG = 800
_TAG_DATABRICKS_WORKSPACE_CLIENT = 801
_TAG_DATABRICKS_ACCOUNT_CLIENT = 802
_TAG_DATABRICKS_CLIENT = 803
_TAG_DATABRICKS_EXECUTION_CONTEXT = 804
_TAG_DATABRICKS_COMMAND_EXECUTION = 805


_CONFIG_CACHE: dict[str, Config] = {}
_WORKSPACE_CLIENT_CACHE: dict[str, WorkspaceClient] = {}
_ACCOUNT_CLIENT_CACHE: dict[str, AccountClient] = {}
_YGG_DATABRICKS_CLIENT_CACHE: dict[str, DatabricksClient] = {}
_EXECUTION_CONTEXT_CACHE: dict[str, ExecutionContext] = {}

__all__ = [
    "DatabricksSerialized",
    "DatabricksConfigSerialized",
    "DatabricksWorkspaceClientSerialized",
    "DatabricksAccountClientSerialized",
    "DatabricksClientSerialized",
    "DatabricksExecutionContextSerialized",
    "DatabricksCommandExecutionSerialized",
]


def _merge_metadata(
    base: Mapping[bytes, bytes] | None,
    extra: Mapping[bytes, bytes] | None = None,
) -> dict[bytes, bytes] | None:
    if not base and not extra:
        return None

    out: dict[bytes, bytes] = {}
    if base:
        out.update(base)
    if extra:
        out.update(extra)
    return out


def _serialize_json_payload(obj: object) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _deserialize_json_payload(data: bytes) -> object:
    return json.loads(data.decode("utf-8"))


def _normalize_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        return [_normalize_json_value(v) for v in value]

    if isinstance(value, dict):
        out: dict[str, object] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                k = str(k)
            normalized = _normalize_json_value(v)
            if normalized is not None:
                out[k] = normalized
        return out

    if hasattr(value, "as_dict"):
        return _normalize_json_value(value.as_dict())

    if hasattr(value, "__dict__"):
        return _normalize_json_value(dict(vars(value)))

    return str(value)


def _is_useful_config_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict, tuple, str)):
        return len(value) > 0
    return True


def _canonicalize_config_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in sorted(kwargs):
        normalized = _normalize_json_value(kwargs[key])
        if _is_useful_config_value(normalized):
            out[key] = normalized
    return out


def _config_cache_key_from_kwargs(kwargs: Mapping[str, object]) -> str:
    return json.dumps(_canonicalize_config_kwargs(kwargs), sort_keys=True, separators=(",", ":"))


def _canonicalize_client_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in sorted(kwargs):
        normalized = _normalize_json_value(kwargs[key])
        if _is_useful_config_value(normalized):
            out[key] = normalized
    return out


def _nested_serialized_bytes(obj: object) -> bytes:
    nested = Serialized.from_python_object(obj)
    with BytesIO() as buf:
        nested.write_to(buf)
        buf.seek(0)
        return buf.read()


def _load_nested_serialized(data: bytes) -> object:
    with BytesIO(data, copy=False) as buf:
        return Serialized.read_from(buf, pos=0).as_python()


def _extract_config_kwargs(config: Config) -> dict[str, object]:
    kwargs = _canonicalize_config_kwargs(dict(config.as_dict()))

    scopes = getattr(config, "scopes", None)
    if scopes is not None:
        normalized_scopes = _normalize_json_value(scopes)
        if _is_useful_config_value(normalized_scopes):
            kwargs["scopes"] = normalized_scopes

    authorization_details = getattr(config, "authorization_details", None)
    if authorization_details is not None:
        normalized_auth = _normalize_json_value(authorization_details)
        if _is_useful_config_value(normalized_auth):
            kwargs["authorization_details"] = normalized_auth

    custom_headers = getattr(config, "_custom_headers", None)
    if custom_headers:
        normalized_headers = _normalize_json_value(dict(custom_headers))
        if _is_useful_config_value(normalized_headers):
            kwargs["custom_headers"] = normalized_headers

    product_info = getattr(config, "_product_info", None)
    if isinstance(product_info, tuple):
        if len(product_info) >= 1 and product_info[0] is not None:
            kwargs["product"] = product_info[0]
        if len(product_info) >= 2 and product_info[1] is not None:
            kwargs["product_version"] = product_info[1]

    return _canonicalize_config_kwargs(kwargs)


def _extract_ygg_client_kwargs(client: DatabricksClient) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for key in _YGG_DATABRICKS_CLIENT_INIT_NAMES:
        value = getattr(client, key, None)
        normalized = _normalize_json_value(value)
        if _is_useful_config_value(normalized):
            kwargs[key] = normalized
    return _canonicalize_client_kwargs(kwargs)


def _client_from_kwargs(kwargs_obj: Mapping[str, object]) -> DatabricksClient:
    kwargs = _canonicalize_client_kwargs(kwargs_obj)
    key = json.dumps(_canonicalize_client_kwargs(kwargs), sort_keys=True, separators=(",", ":"))

    cached = _YGG_DATABRICKS_CLIENT_CACHE.get(key)
    if cached is not None:
        return cached

    client = _YGGDatabricksClient(**kwargs)
    _YGG_DATABRICKS_CLIENT_CACHE[key] = client
    return client


def _language_to_name(language: Language | None) -> str | None:
    if language is None:
        return None
    return getattr(language, "value", None) or getattr(language, "name", None) or str(language)


def _language_from_name(name: str | None) -> Language | None:
    if name is None:
        return None
    from databricks.sdk.service.compute import Language as _Language

    try:
        return _Language(name)
    except Exception:
        return _Language[name.upper()]


def _cluster_from_payload(
    *,
    client_kwargs: Mapping[str, object],
    cluster_id: str | None,
    cluster_name: str | None,
) -> _YGGCluster:
    client = _client_from_kwargs(client_kwargs)
    return _YGGCluster(
        service=_YGGClusters(client=client),
        cluster_id=cluster_id,
        cluster_name=cluster_name,
    )


def _execution_context_cache_key(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _extract_execution_context_payload(context: ExecutionContext) -> dict[str, object]:
    payload: dict[str, object] = {
        "c": _extract_ygg_client_kwargs(context.client),
        "cid": context.cluster.cluster_id,
        "cn": context.cluster.cluster_name,
        "k": context.context_key,
    }
    if context.context_id:
        payload["id"] = context.context_id
    if context.language is not None:
        payload["l"] = _language_to_name(context.language)
    if context.temporary:
        payload["t"] = True
    if context.close_after is not None:
        payload["ca"] = context.close_after
    return payload


def _execution_context_from_payload(payload_obj: Mapping[str, object]) -> ExecutionContext:
    payload = _canonicalize_client_kwargs(payload_obj)

    key = _execution_context_cache_key(payload)
    cached = _EXECUTION_CONTEXT_CACHE.get(key)
    if cached is not None:
        return cached

    client_kwargs = payload.get("c")
    if not isinstance(client_kwargs, dict):
        raise ValueError("ExecutionContext payload missing dict 'c'")

    cluster = _cluster_from_payload(
        client_kwargs=client_kwargs,
        cluster_id=payload.get("cid") if isinstance(payload.get("cid"), str) else None,
        cluster_name=payload.get("cn") if isinstance(payload.get("cn"), str) else None,
    )
    context = _YGGExecutionContext(
        cluster=cluster,
        context_id=payload.get("id") if isinstance(payload.get("id"), str) else "",
        context_key=payload.get("k") if isinstance(payload.get("k"), str) else None,
        language=_language_from_name(payload.get("l") if isinstance(payload.get("l"), str) else None),
        temporary=bool(payload.get("t", False)),
        close_after=float(payload["ca"]) if isinstance(payload.get("ca"), (int, float)) else None,
    )

    _EXECUTION_CONTEXT_CACHE[key] = context
    return context


def _extract_command_execution_payload(command: CommandExecution) -> dict[str, object]:
    pyfunc_dump = None
    if command.pyfunc is not None:
        from yggdrasil.pickle.ser import dumps as _ygg_dumps

        pyfunc_dump = _ygg_dumps(command.pyfunc, b64=True)

    payload: dict[str, object] = {
        "ctx": _nested_serialized_bytes(command.context).hex(),
        "cid": command.command_id,
        "l": _language_to_name(command.language),
        "cmd": command.command,
    }
    
    if command.environ is not None:
        normalized_env = _normalize_json_value(dict(command.environ))
        if _is_useful_config_value(normalized_env):
            payload["env"] = normalized_env
    
    if pyfunc_dump is not None:
        payload["pf"] = pyfunc_dump

    return payload


def _command_execution_from_payload(payload_obj: Mapping[str, object]) -> CommandExecution:
    payload = payload_obj

    context_hex = payload.get("ctx")
    if not isinstance(context_hex, str):
        raise ValueError("CommandExecution payload missing hex 'ctx'")

    context = _load_nested_serialized(bytes.fromhex(context_hex))
    if not isinstance(context, _YGGExecutionContext):
        raise TypeError(f"Deserialized context must be ExecutionContext, got {type(context)!r}")

    environ_obj = payload.get("env")
    environ = environ_obj if isinstance(environ_obj, Mapping) else None

    pyfunc = None
    pyfunc_dump = payload.get("pf")
    if isinstance(pyfunc_dump, str):
        from yggdrasil.pickle.ser import loads as _ygg_loads

        pyfunc = _ygg_loads(pyfunc_dump)

    out = _YGGCommandExecution(
        context=context,
        command_id=payload.get("cid") if isinstance(payload.get("cid"), str) else None,
        language=_language_from_name(payload.get("l") if isinstance(payload.get("l"), str) else None),
        command=payload.get("cmd") if isinstance(payload.get("cmd"), str) else None,
        pyfunc=pyfunc,
        environ=environ,
    )
    return out


def _config_from_kwargs(kwargs_obj: Mapping[str, object]) -> Config:
    kwargs = _canonicalize_config_kwargs(kwargs_obj)
    key = _config_cache_key_from_kwargs(kwargs)

    cached = _CONFIG_CACHE.get(key)
    if cached is not None:
        return cached

    config = _DBXConfig(**kwargs)
    _CONFIG_CACHE[key] = config
    return config


def _dump_config_payload(config: Config) -> bytes:
    return _serialize_json_payload(_extract_config_kwargs(config))


def _load_config_payload(data: bytes) -> Config:
    payload_obj = _deserialize_json_payload(data)
    if not isinstance(payload_obj, dict):
        raise ValueError(f"Expected dict payload, got {type(payload_obj)!r}")
    return _config_from_kwargs(payload_obj)


def _dump_client_payload(
    client: WorkspaceClient | AccountClient,
    *,
    kind: str,
) -> bytes:
    return _serialize_json_payload(
        {
            "k": kind,
            "cfg": _extract_config_kwargs(client.config),
        }
    )


def _client_cache_for_kind(kind: str) -> dict[str, WorkspaceClient | AccountClient]:
    if kind == "workspace":
        return _WORKSPACE_CLIENT_CACHE
    if kind == "account":
        return _ACCOUNT_CLIENT_CACHE
    raise ValueError(f"Unsupported Databricks client kind: {kind!r}")


def _load_client_payload(data: bytes) -> tuple[str, Config]:
    payload_obj = _deserialize_json_payload(data)
    if not isinstance(payload_obj, dict):
        raise ValueError(f"Expected dict payload, got {type(payload_obj)!r}")

    kind = payload_obj.get("k")
    if kind not in ("workspace", "account"):
        raise ValueError(f"Unsupported Databricks client kind: {kind!r}")

    cfg_obj = payload_obj.get("cfg")
    if not isinstance(cfg_obj, dict):
        raise ValueError(f"Client payload cfg must be dict, got {type(cfg_obj)!r}")

    return kind, _config_from_kwargs(cfg_obj)



def _load_cached_client(data: bytes, *, kind: str) -> WorkspaceClient | AccountClient:
    payload_kind, config = _load_client_payload(data)
    if payload_kind != kind:
        raise ValueError(
            f"Databricks client payload kind mismatch: expected {kind!r}, got {payload_kind!r}"
        )

    key = _config_cache_key_from_kwargs(_extract_config_kwargs(config))
    cache = _client_cache_for_kind(kind)
    cached = cache.get(key)
    if cached is not None:
        return cached

    client = (
        _DBXWorkspaceClient(config=config)
        if kind == "workspace"
        else _DBXAccountClient(config=config)
    )
    cache[key] = client
    return client


def _load_ygg_client_payload(data: bytes) -> DatabricksClient:
    payload_obj = _deserialize_json_payload(data)
    if not isinstance(payload_obj, dict):
        raise ValueError(f"Expected dict payload, got {type(payload_obj)!r}")

    return _client_from_kwargs(payload_obj)


def _dump_ygg_client_payload(client: DatabricksClient) -> bytes:
    return _serialize_json_payload(_extract_ygg_client_kwargs(client))


@dataclass(frozen=True, slots=True)
class DatabricksSerialized(Serialized[object]):
    TAG: ClassVar[int] = -1

    def as_python(self) -> object:
        raise NotImplementedError

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        if type(obj) is _YGGDatabricksClient:
            return DatabricksClientSerialized.from_value(obj, metadata=metadata, codec=codec)

        if type(obj) is _YGGExecutionContext:
            return DatabricksExecutionContextSerialized.from_value(obj, metadata=metadata, codec=codec)

        if type(obj) is _YGGCommandExecution:
            return DatabricksCommandExecutionSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, _DBXConfig):
            return DatabricksConfigSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, _DBXWorkspaceClient):
            return DatabricksWorkspaceClientSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, _DBXAccountClient):
            return DatabricksAccountClientSerialized.from_value(obj, metadata=metadata, codec=codec)

        return None


@dataclass(frozen=True, slots=True)
class DatabricksConfigSerialized(DatabricksSerialized):
    TAG: ClassVar[int] = _TAG_DATABRICKS_CONFIG

    @property
    def value(self) -> Config:
        return _load_config_payload(self.decode())

    def as_python(self) -> Config:
        return self.value

    @classmethod
    def from_value(
        cls,
        config: Config,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "DatabricksConfigSerialized":
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=_dump_config_payload(config),
            metadata=_merge_metadata(metadata, {b"ygg_object": b"databricks_config"}),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class DatabricksWorkspaceClientSerialized(DatabricksSerialized):
    TAG: ClassVar[int] = _TAG_DATABRICKS_WORKSPACE_CLIENT

    @property
    def value(self) -> WorkspaceClient:
        return _load_cached_client(self.decode(), kind="workspace")

    def as_python(self) -> WorkspaceClient:
        return self.value

    @classmethod
    def from_value(
        cls,
        client: WorkspaceClient,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "DatabricksWorkspaceClientSerialized":
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=_dump_client_payload(client, kind="workspace"),
            metadata=_merge_metadata(metadata, {b"ygg_object": b"databricks_workspace_client"}),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class DatabricksAccountClientSerialized(DatabricksSerialized):
    TAG: ClassVar[int] = _TAG_DATABRICKS_ACCOUNT_CLIENT

    @property
    def value(self) -> AccountClient:
        return _load_cached_client(self.decode(), kind="account")

    def as_python(self) -> AccountClient:
        return self.value

    @classmethod
    def from_value(
        cls,
        client: AccountClient,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "DatabricksAccountClientSerialized":
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=_dump_client_payload(client, kind="account"),
            metadata=_merge_metadata(metadata, {b"ygg_object": b"databricks_account_client"}),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class DatabricksClientSerialized(DatabricksSerialized):
    TAG: ClassVar[int] = _TAG_DATABRICKS_CLIENT

    @property
    def value(self) -> DatabricksClient:
        return _load_ygg_client_payload(self.decode())

    def as_python(self) -> DatabricksClient:
        return self.value

    @classmethod
    def from_value(
        cls,
        client: DatabricksClient,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "DatabricksClientSerialized":
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=_dump_ygg_client_payload(client),
            metadata=_merge_metadata(metadata, {b"ygg_object": b"ygg_databricks_client"}),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class DatabricksExecutionContextSerialized(DatabricksSerialized):
    TAG: ClassVar[int] = _TAG_DATABRICKS_EXECUTION_CONTEXT

    @property
    def value(self) -> ExecutionContext:
        payload_obj = _deserialize_json_payload(self.decode())
        if not isinstance(payload_obj, dict):
            raise ValueError(f"Expected dict payload, got {type(payload_obj)!r}")
        return _execution_context_from_payload(payload_obj)

    def as_python(self) -> ExecutionContext:
        return self.value

    @classmethod
    def from_value(
        cls,
        context: ExecutionContext,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "DatabricksExecutionContextSerialized":
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=_serialize_json_payload(_extract_execution_context_payload(context)),
            metadata=_merge_metadata(metadata, {b"ygg_object": b"databricks_execution_context"}),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class DatabricksCommandExecutionSerialized(DatabricksSerialized):
    TAG: ClassVar[int] = _TAG_DATABRICKS_COMMAND_EXECUTION

    @property
    def value(self) -> CommandExecution:
        payload_obj = _deserialize_json_payload(self.decode())
        if not isinstance(payload_obj, dict):
            raise ValueError(f"Expected dict payload, got {type(payload_obj)!r}")
        return _command_execution_from_payload(payload_obj)

    def as_python(self) -> CommandExecution:
        return self.value

    @classmethod
    def from_value(
        cls,
        command: CommandExecution,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "DatabricksCommandExecutionSerialized":
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=_serialize_json_payload(_extract_command_execution_payload(command)),
            metadata=_merge_metadata(metadata, {b"ygg_object": b"databricks_command_execution"}),
            codec=codec,
        )


for _cls, _pytype in (
    (DatabricksConfigSerialized, _DBXConfig),
    (DatabricksWorkspaceClientSerialized, _DBXWorkspaceClient),
    (DatabricksAccountClientSerialized, _DBXAccountClient),
    (DatabricksClientSerialized, _YGGDatabricksClient),
    (DatabricksExecutionContextSerialized, _YGGExecutionContext),
    (DatabricksCommandExecutionSerialized, _YGGCommandExecution),
):
    Tags.register_class(_cls, tag=_cls.TAG, pytype=_pytype)

DatabricksConfigSerialized = Tags.get_class(Tags.DATABRICKS_CONFIG) or DatabricksConfigSerialized
DatabricksWorkspaceClientSerialized = (
    Tags.get_class(Tags.DATABRICKS_WORKSPACE_CLIENT) or DatabricksWorkspaceClientSerialized
)
DatabricksAccountClientSerialized = (
    Tags.get_class(Tags.DATABRICKS_ACCOUNT_CLIENT) or DatabricksAccountClientSerialized
)
DatabricksClientSerialized = Tags.get_class(Tags.DATABRICKS_CLIENT) or DatabricksClientSerialized
DatabricksExecutionContextSerialized = (
    Tags.get_class(Tags.DATABRICKS_EXECUTION_CONTEXT) or DatabricksExecutionContextSerialized
)
DatabricksCommandExecutionSerialized = (
    Tags.get_class(Tags.DATABRICKS_COMMAND_EXECUTION) or DatabricksCommandExecutionSerialized
)

Tags.TYPES[_DBXConfig] = DatabricksConfigSerialized
Tags.TYPES[_DBXWorkspaceClient] = DatabricksWorkspaceClientSerialized
Tags.TYPES[_DBXAccountClient] = DatabricksAccountClientSerialized
Tags.TYPES[_YGGDatabricksClient] = DatabricksClientSerialized
Tags.TYPES[_YGGExecutionContext] = DatabricksExecutionContextSerialized
Tags.TYPES[_YGGCommandExecution] = DatabricksCommandExecutionSerialized

del _cls, _pytype
