import base64
import io
import logging

from databricks.sdk.errors import NotFound

from yggdrasil.data import any_to_datetime

import yggdrasil.pickle.json as json_module
from dataclasses import dataclass, field
from typing import Optional, Any, Mapping
import datetime as dt

from databricks.sdk.service.workspace import GetSecretResponse, AclPermission

from .service import Secrets
from ..client import DatabricksResource

__all__ = [
    "Scope",
    "Secret",
    "Permission"
]


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Permission:
    principal: str
    acl: AclPermission

    @classmethod
    def parse(cls, obj: Any):
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, str):
            return cls.parse_str(obj)
        elif isinstance(obj, Mapping):
            principal = obj.get("principal")
            acl = obj.get("acl", AclPermission.READ)
            if not principal:
                raise ValueError("Principal is required to parse Permission from mapping")
            return cls(principal=principal, acl=AclPermission(acl))
        else:
            raise ValueError(f"Cannot parse {obj!r} as Permission")

    @classmethod
    def parse_str(cls, value: str):
        if value in {"users"}:
            return cls(principal=value, acl=AclPermission.READ)
        return cls(principal=value, acl=AclPermission.MANAGE)


@dataclass
class Scope(DatabricksResource):
    service: Secrets = field(
        default_factory=Secrets.current,
        hash=False,
        compare=False,
        repr=False
    )
    key: Optional[str] = field(default=None)

    def __bool__(self):
        return bool(self.key)

    def __getitem__(self, item):
        if not self.key:
            raise ValueError("Scope must have a key to access secrets")

        return self.service.secret(item, scope=self).refresh()

    def __setitem__(self, key, value):
        if not self.key:
            raise ValueError("Scope must have a key to set secrets")

        found = self.service.secret(key, scope=self).refresh(raise_error=False)

        if not found.b64:
            return self.service.create_secret(key=key, value=value, scope=self)
        else:
            return found.update(value=value)

    def __delitem__(self, key):
        if not self.key:
            raise ValueError("Scope must have a key to delete secrets")

        return self.service.delete_secret(key=key, scope=self)

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        service: Optional[Secrets] = None,
    ):
        if isinstance(obj, cls):
            return obj

        elif obj is None:
            return cls(service=service or Secrets.current())

        elif isinstance(obj, str):
            return cls(
                service=service or Secrets.current(),
                key=obj,
            )

        elif isinstance(obj, Mapping):
            return cls.parse_mapping(obj, service=service)

        else:
            raise ValueError(f"Cannot parse {obj!r} as Scope")

    @classmethod
    def parse_mapping(
        cls,
        data: Mapping[str, Any] = None,
        *,
        service: Optional[Secrets] = None,
    ):
        if data is None:
            data = {}

        key = data.get("key")

        if not key:
            raise ValueError("Key is required to parse Scope from mapping")

        return cls(
            service=service or Secrets.current(),
            key=key,
        )

    def update(
        self,
        key: str | None = None,
        permissions: Optional[list[Permission]] = None,
    ):
        if not self.key:
            if key:
                self.key = key
            else:
                raise ValueError("Secret must have both scope and key to be updated")
        elif key and key != self.key:
            raise ValueError(f"Cannot change scope key from {self.key!r} to {key!r}")

        client = self.client.workspace_client().secrets

        if not permissions:
            return self

        permissions = [Permission.parse(_) for _ in permissions]

        for p in permissions:
            if p not in Permission:
                raise ValueError(f"Invalid permission: {p!r}")

            LOGGER.debug("Updating ACL for %s: %s", self, p)

            client.put_acl(
                scope=self.key,
                principal=p.principal,
                permission=p.acl
            )

            LOGGER.info("Updated ACL for %s: %s", self, p)

        return self


@dataclass
class Secret(DatabricksResource):
    service: Secrets = field(
        default_factory=Secrets,
        hash=False,
        compare=False,
        repr=False
    )
    scope: Scope = field(default=None)
    key: Optional[str] = field(default=None)
    b64: Optional[str] = field(default=None, repr=False, compare=False)
    update_timestamp: Optional[dt.datetime] = None

    def __post_init__(self):
        self.scope = Scope.parse(self.scope, service=self.service)

        if self.update_timestamp:
            self.update_timestamp = any_to_datetime(self.update_timestamp)

    def set_value(self, value: Any):
        if not isinstance(value, str):
            if isinstance(value, (bytes, bytearray, io.BytesIO)):
                self.b64 = base64.b64encode(value).decode("utf-8")
                return self
            else:
                self.b64 = base64.b64encode(json_module.dumps(value)).decode("utf-8")
                return self

        # Check if the value is already base64-encoded to avoid double encoding
        try:
            base64.b64decode(value, validate=True)
            self.b64 = value
        except ValueError:
            self.b64 = base64.b64encode(value.encode("utf-8")).decode("utf-8")

        return self

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        scope: Optional[Scope] = None,
        service: Optional[Secrets] = None,
    ):
        if scope:
            scope = Scope.parse(scope, service=service)

            if not service:
                service = scope.service

        if isinstance(obj, cls):
            if scope:
                obj.scope = scope
            return obj

        elif isinstance(obj, str):
            if "/" in obj:
                scope_str, key = obj.split("/", 1)
                scope = Scope.parse(scope_str, service=service)
                return cls(service=service or scope.service, scope=scope, key=key)
            elif ":" in obj:
                scope_str, key = obj.split(":", 1)
                scope = Scope.parse(scope_str, service=service)
                return cls(service=service or scope.service, scope=scope, key=key)
            else:
                return cls(service=service or Secrets.current(), scope=scope, key=obj)

        elif isinstance(obj, GetSecretResponse):
            return cls(
                service=service or Secrets.current(),
                scope=Scope.parse(obj.scope, service=service),
                key=obj.key,
            ).set_value(obj.value)

        else:
            raise ValueError(f"Cannot parse {obj!r} as Secret")

    @classmethod
    def parse_mapping(
        cls,
        data: Mapping[str, Any] = None,
        *,
        service: Optional[Secrets] = None,
        **kwargs
    ) -> "Secret":
        if data:
            kwargs.update(data)

        key = kwargs.get("key")

        if not key:
            raise ValueError("Key is required to parse Secret from mapping")

        if "/" in key:
            scope_str, key = key.split("/", 1)
            kwargs["scope"] = scope_str
            kwargs["key"] = key
        elif ":" in key:
            scope_str, key = key.split(":", 1)
            kwargs["scope"] = scope_str
            kwargs["key"] = key

        scope = kwargs.pop("scope", None)
        if not scope:
            raise ValueError("Scope is required to parse Secret from mapping")

        scope = Scope.parse(scope, service=kwargs.get("service", service))
        service = service if service else scope.service
        value = kwargs.get("value")

        built = cls(
            service=service,
            scope=scope,
            key=key,
            update_timestamp=kwargs.get("update_timestamp"),
        )

        if value is not None:
            built.set_value(value)

        return built

    def to_bytes(self) -> bytes:
        if not self.b64:
            self.refresh()

        return base64.b64decode(self.b64) if self.b64 else b""

    @property
    def object(self) -> Any:
        bvalue = self.to_bytes()

        if not bvalue:
            return None

        if (
            bvalue.startswith(b"{") and bvalue.endswith(b"}")
            or bvalue.startswith(b"[") and bvalue.endswith(b"]")
            or bvalue.startswith(b'"') and bvalue.endswith(b'"')
        ):
            return json_module.loads(bvalue)

        elif bvalue.isdigit():
            return int(bvalue)
        elif bvalue.replace(b".", b"", 1).isdigit():
            return float(bvalue)

        return bvalue

    def set_details(self, details: GetSecretResponse) -> "Secret":
        if isinstance(details, GetSecretResponse):
            self.key = details.key
            self.b64 = details.value

        else:
            raise ValueError(f"Cannot set details from {details!r}, expected GetSecretResponse")

        return self

    def refresh(self, raise_error: bool = True) -> "Secret":
        if self.scope and self.key:
            try:
                infos = self.client.workspace_client().secrets.get_secret(scope=self.scope.key, key=self.key)
            except NotFound:
                if raise_error:
                    raise
                else:
                    return self
            return self.set_details(infos)
        return self

    def update(
        self,
        value: Optional[Any] = None,
        *,
        permissions: Optional[list[Permission]] = None,
    ):
        if not self.scope or not self.key:
            raise ValueError("Secret must have both scope and key to be updated")

        if value is not None:
            previous_value = self.b64
            self.set_value(value)

            if self.b64 != previous_value:
                try:
                    client = self.client.workspace_client().secrets

                    LOGGER.debug("Updating %s with new value", self)

                    client.put_secret(
                        scope=self.scope.key,
                        key=self.key,
                        bytes_value=self.b64,
                    )
                except:
                    self.b64 = previous_value
                    raise

                LOGGER.info("Updated %s with new value", self)

        if permissions:
            self.scope.update(permissions=permissions)

        return self.refresh()