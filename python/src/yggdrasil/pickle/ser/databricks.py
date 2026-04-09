from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Mapping

from yggdrasil.databricks import lib as databricks_lib
from yggdrasil.pickle.ser.libs import (
    _FORMAT_VERSION,
    _deserialize_nested,
    _require_dict,
    _require_tuple_len,
    _serialize_nested,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

if TYPE_CHECKING:
    from databricks.sdk import AccountClient, WorkspaceClient
    from databricks.sdk.config import Config

_DBXAccountClient = databricks_lib.AccountClient
_DBXWorkspaceClient = databricks_lib.WorkspaceClient
_DBXConfig = databricks_lib.Config

_TAG_DATABRICKS_CONFIG = 800
_TAG_DATABRICKS_WORKSPACE_CLIENT = 801
_TAG_DATABRICKS_ACCOUNT_CLIENT = 802

__all__ = [
    "DatabricksSerialized",
    "DatabricksConfigSerialized",
    "DatabricksWorkspaceClientSerialized",
    "DatabricksAccountClientSerialized",
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


def _extract_config_kwargs(config: Config) -> dict[str, object]:
    kwargs = dict(config.as_dict())

    scopes = getattr(config, "scopes", None)
    if scopes is not None:
        kwargs["scopes"] = list(scopes)

    authorization_details = getattr(config, "authorization_details", None)
    if authorization_details is not None:
        kwargs["authorization_details"] = authorization_details

    custom_headers = getattr(config, "_custom_headers", None)
    if custom_headers:
        kwargs["custom_headers"] = dict(custom_headers)

    product_info = getattr(config, "_product_info", None)
    if isinstance(product_info, tuple):
        if len(product_info) >= 1 and product_info[0] is not None:
            kwargs["product"] = product_info[0]
        if len(product_info) >= 2 and product_info[1] is not None:
            kwargs["product_version"] = product_info[1]

    return kwargs


def _dump_config_payload(config: Config) -> bytes:
    return _serialize_nested((_FORMAT_VERSION, _extract_config_kwargs(config)))


def _load_config_payload(data: bytes) -> Config:
    payload = _require_tuple_len(
        _deserialize_nested(data),
        name="Databricks Config payload",
        expected=2,
    )
    version, kwargs_obj = payload

    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported Databricks Config payload version: {version!r}")

    kwargs = _require_dict(kwargs_obj, name="Databricks Config kwargs")
    return _DBXConfig(**kwargs)


def _dump_client_payload(client: WorkspaceClient | AccountClient) -> bytes:
    return _serialize_nested((_FORMAT_VERSION, client.config))


def _load_client_config(data: bytes) -> Config:
    payload = _require_tuple_len(
        _deserialize_nested(data),
        name="Databricks client payload",
        expected=2,
    )
    version, config = payload

    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported Databricks client payload version: {version!r}")
    if not isinstance(config, _DBXConfig):
        raise TypeError(f"Databricks client payload must contain Config, got {type(config)!r}")
    return config


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
        return _DBXWorkspaceClient(config=_load_client_config(self.decode()))

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
            data=_dump_client_payload(client),
            metadata=_merge_metadata(metadata, {b"ygg_object": b"databricks_workspace_client"}),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class DatabricksAccountClientSerialized(DatabricksSerialized):
    TAG: ClassVar[int] = _TAG_DATABRICKS_ACCOUNT_CLIENT

    @property
    def value(self) -> AccountClient:
        return _DBXAccountClient(config=_load_client_config(self.decode()))

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
            data=_dump_client_payload(client),
            metadata=_merge_metadata(metadata, {b"ygg_object": b"databricks_account_client"}),
            codec=codec,
        )


for _cls, _pytype in (
    (DatabricksConfigSerialized, _DBXConfig),
    (DatabricksWorkspaceClientSerialized, _DBXWorkspaceClient),
    (DatabricksAccountClientSerialized, _DBXAccountClient),
):
    Tags.register_class(_cls, tag=_cls.TAG, pytype=_pytype)

del _cls, _pytype
