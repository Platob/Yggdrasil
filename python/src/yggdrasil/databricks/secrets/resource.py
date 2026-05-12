import base64
import datetime as dt
import io
import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Mapping

import yggdrasil.pickle.json as json_module
from databricks.sdk.errors import NotFound
from databricks.sdk.service.workspace import AclItem, AclPermission, GetSecretResponse
from yggdrasil.data.cast import any_to_datetime

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
    def parse(cls, obj: Any, *, acl: Any = None):
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, AclItem):
            return cls(principal=obj.principal, acl=AclPermission(obj.permission))
        if isinstance(obj, str):
            return cls.parse_str(obj, acl=acl)
        if isinstance(obj, Mapping):
            principal = obj.get("principal")
            if not principal:
                raise ValueError(
                    f"Cannot parse Permission from mapping {obj!r}: "
                    f"'principal' is required."
                )
            value = obj.get("acl", obj.get("permission", AclPermission.READ))
            return cls(principal=principal, acl=AclPermission(value))
        raise ValueError(
            f"Cannot parse {obj!r} as Permission. "
            f"Expected Permission, AclItem, str, or mapping with 'principal'."
        )

    @classmethod
    def parse_str(cls, value: str, *, acl: Any = None):
        if acl is None:
            acl = AclPermission.READ if value in {"users"} else AclPermission.MANAGE
        return cls(principal=value, acl=AclPermission(acl))


class Scope(DatabricksResource):
    service: Secrets = field(
        default_factory=Secrets.current,
        hash=False,
        compare=False,
        repr=False
    )
    key: Optional[str] = field(default=None)

    def __init__(
        self,
        service: Secrets | None = None,
        key: Optional[str] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.service = Secrets.current() if service is None else service
        self.key = key

    def __bool__(self):
        return bool(self.key)

    def __getitem__(self, item):
        if not self.key:
            raise ValueError("Scope must have a key to access secrets")

        return self.service.secret(item, scope=self).refresh()

    def __setitem__(self, key, value):
        if not self.key:
            raise ValueError("Scope must have a key to set secrets")

        return self.service.create_secret(key=key, value=value, scope=self)

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
                raise ValueError("Scope must have a key to be updated")
        elif key and key != self.key:
            raise ValueError(f"Cannot change scope key from {self.key!r} to {key!r}")

        if not permissions:
            return self

        for p in permissions:
            self.set_permission(p)

        return self

    def list_secrets(self) -> list["Secret"]:
        if not self.key:
            raise ValueError("Scope must have a key to list secrets")

        api = self.client.workspace_client().secrets

        try:
            metas = list(api.list_secrets(scope=self.key))
        except NotFound:
            return []

        out: list[Secret] = []
        for meta in metas:
            ts = getattr(meta, "last_updated_timestamp", None)
            out.append(
                Secret(
                    service=self.service,
                    scope=self,
                    key=meta.key,
                    update_timestamp=any_to_datetime(ts) if ts else None,
                )
            )
        return out

    def list_permissions(self) -> list[Permission]:
        if not self.key:
            raise ValueError("Scope must have a key to list permissions")

        api = self.client.workspace_client().secrets

        try:
            items = list(api.list_acls(scope=self.key))
        except NotFound:
            return []

        return [Permission.parse(item) for item in items]

    def permission(self, principal: str) -> Optional[Permission]:
        if not self.key:
            raise ValueError("Scope must have a key to read a permission")

        api = self.client.workspace_client().secrets

        try:
            item = api.get_acl(scope=self.key, principal=principal)
        except NotFound:
            return None

        return Permission.parse(item)

    def set_permission(
        self,
        principal: Any,
        acl: Any = None,
    ) -> Permission:
        if not self.key:
            raise ValueError("Scope must have a key to set a permission")

        if isinstance(principal, str):
            target = Permission.parse_str(principal, acl=acl)
        else:
            target = Permission.parse(principal, acl=acl)

        existing = self.permission(target.principal)
        if existing == target:
            LOGGER.debug("Permission %s already set on %s; skipping put_acl", target, self)
            return existing

        LOGGER.debug("Updating ACL for %s: %s", self, target)

        self.client.workspace_client().secrets.put_acl(
            scope=self.key,
            principal=target.principal,
            permission=target.acl,
        )

        LOGGER.info("Updated ACL for %s: %s", self, target)
        return target

    def delete_permission(self, principal: str) -> None:
        if not self.key:
            raise ValueError("Scope must have a key to delete a permission")

        try:
            self.client.workspace_client().secrets.delete_acl(
                scope=self.key,
                principal=principal,
            )
        except NotFound:
            LOGGER.debug(
                "ACL for principal %r on scope %s does not exist; skipping delete",
                principal, self.key,
            )

    def delete(self) -> None:
        if not self.key:
            raise ValueError("Scope must have a key to be deleted")

        LOGGER.debug("Deleting scope %s", self)

        try:
            self.client.workspace_client().secrets.delete_scope(scope=self.key)
        except NotFound:
            LOGGER.warning("Scope %s does not exist; skipping delete", self)
            return

        LOGGER.info("Deleted scope %s", self)

    def secret(self, key: str) -> "Secret":
        return self.service.secret(key, scope=self)

    def __iter__(self):
        return iter(self.list_secrets())

    def __contains__(self, key: Any) -> bool:
        if not self.key or not isinstance(key, str):
            return False
        return self.service.secret(key, scope=self).refresh(raise_error=False).b64 is not None


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

    def __init__(
        self,
        service: Secrets | None = None,
        scope: Scope | None = None,
        key: Optional[str] = None,
        b64: Optional[str] = None,
        update_timestamp: Optional[dt.datetime] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.service = Secrets.current() if service is None else service
        self.scope = scope
        self.key = key
        self.b64 = b64
        self.update_timestamp = update_timestamp

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

    def bvalue(self) -> bytes:
        if not self.b64:
            self.refresh()

        return base64.b64decode(self.b64) if self.b64 else b""

    def svalue(self) -> str:
        return self.bvalue().decode("utf-8")

    @property
    def object(self) -> Any:
        bvalue = self.bvalue()

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

    def mapvalue(self) -> Mapping[str, Any]:
        parsed = self.object
        if not isinstance(parsed, Mapping):
            raise ValueError(f"Secret {self} is not a mapping")
        return parsed

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

    def delete(self) -> None:
        if not self.scope or not self.scope.key or not self.key:
            raise ValueError("Secret must have both scope and key to be deleted")

        LOGGER.debug("Deleting secret %s", self)

        try:
            self.client.workspace_client().secrets.delete_secret(
                scope=self.scope.key,
                key=self.key,
            )
        except NotFound:
            LOGGER.warning("Secret %s does not exist; skipping delete", self)
            return

        LOGGER.info("Deleted secret %s", self)