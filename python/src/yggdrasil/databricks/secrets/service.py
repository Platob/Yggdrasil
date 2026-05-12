from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

from databricks.sdk.errors import NotFound, ResourceAlreadyExists, ResourceDoesNotExist

from ..client import DatabricksService

if TYPE_CHECKING:
    from .resource import Permission, Scope, Secret

__all__ = ["Secrets"]


LOGGER = logging.getLogger(__name__)


class Secrets(DatabricksService):

    def __getitem__(self, item):
        if "/" in item or ":" in item:
            return self.secret(item).refresh()
        return self.scope(item)

    def __setitem__(self, key, value):
        return self.create_secret(key=key, value=value)

    def __delitem__(self, key):
        return self.delete_secret(key=key)

    def __iter__(self):
        return iter(self.list_scopes())

    def __contains__(self, item: Any) -> bool:
        if not isinstance(item, str):
            return False
        if "/" in item or ":" in item:
            return self.secret(item).refresh(raise_error=False).b64 is not None
        return any(s.key == item for s in self.list_scopes())

    # ------------------------------------------------------------------ scopes

    def scope(
        self,
        key: str,
    ) -> "Scope":
        from .resource import Scope

        return Scope.from_(
            key,
            service=self,
        )

    def list_scopes(self) -> list["Scope"]:
        from .resource import Scope

        api = self.client.workspace_client().secrets
        return [Scope(service=self, key=s.name) for s in api.list_scopes()]

    def create_scope(
        self,
        key: str | None = None,
        *,
        permissions: Union[list["Permission"], None] = None,
        scope: Union["Scope", str, None] = None,
    ) -> "Scope":
        from .resource import Scope

        if not scope:
            scope = Scope.from_mapping(
                {"key": key} if key else None,
                service=self,
            )
        elif isinstance(scope, str):
            scope = Scope.from_(scope, service=self)

        if not scope.key:
            raise ValueError("Scope must have a key to be created")

        LOGGER.debug("Creating scope %s", scope)

        api = self.client.workspace_client().secrets

        try:
            api.create_scope(scope=scope.key)
            LOGGER.info("Created scope %s", scope)
        except ResourceAlreadyExists:
            LOGGER.debug("Scope %s already exists; skipping create", scope)

        return scope.update(permissions=permissions)

    def delete_scope(
        self,
        scope: Union["Scope", str],
    ) -> None:
        return self.scope(scope).delete() if isinstance(scope, str) else scope.delete()

    # ----------------------------------------------------------------- secrets

    def secret(
        self,
        key: str,
        *,
        scope: Union["Scope", str, None] = None,
    ) -> "Secret":
        from .resource import Secret

        return Secret.from_(
            key,
            service=self,
            scope=scope
        )

    def list_secrets(
        self,
        scope: Union["Scope", str],
    ) -> list["Secret"]:
        return self.scope(scope).list_secrets() if isinstance(scope, str) else scope.list_secrets()

    def create_secret(
        self,
        key: str,
        value: Any,
        *,
        scope: Union["Scope", str, None] = None,
        permissions: Union[list["Permission"], None] = None,
        secret: Union["Secret", str, None] = None,
    ) -> "Secret":
        from .resource import Secret

        if not secret:
            secret = Secret.from_mapping(
                {"key": key, "scope": scope, "value": value},
                service=self,
            )
        elif isinstance(secret, str):
            secret = Secret.from_(secret, service=self, scope=scope)
            secret.set_value(value)

        if not secret.scope or not secret.scope.key or not secret.key:
            raise ValueError("Secret must have both scope and key to be created")

        target_b64 = secret.b64
        existing = self.secret(key=secret.key, scope=secret.scope).refresh(raise_error=False)

        # Stronger dedup: skip the put_secret round-trip when the existing
        # value already matches the desired b64 payload. The Databricks
        # Secrets API doesn't return cleartext outside DBR, but GetSecretResponse
        # carries the same base64 we'd otherwise upload, so b64 equality is the
        # safest cheap fingerprint we can compute client-side.
        if target_b64 and existing.b64 == target_b64:
            LOGGER.debug("Secret %s already at desired value; skipping put_secret", secret)
            secret.update_timestamp = existing.update_timestamp
            if permissions:
                secret.scope.update(permissions=permissions)
            return secret

        LOGGER.debug("Creating secret %s", secret)

        api = self.client.workspace_client().secrets

        try:
            api.put_secret(
                scope=secret.scope.key,
                key=secret.key,
                bytes_value=target_b64,
            )
        except ResourceDoesNotExist as e:
            msg = str(e)

            if secret.scope.key in msg:
                secret.scope = self.create_scope(
                    key=secret.scope.key,
                    permissions=permissions,
                    scope=secret.scope,
                )

                api.put_secret(
                    scope=secret.scope.key,
                    key=secret.key,
                    bytes_value=target_b64,
                )
            else:
                raise

        LOGGER.info("Created secret %s", secret)

        if permissions:
            secret.scope.update(permissions=permissions)

        return secret

    def delete_secret(
        self,
        key: Union[str, "Secret"],
        *,
        scope: Union["Scope", str, None] = None,
    ) -> None:
        from .resource import Secret

        if isinstance(key, Secret):
            return key.delete()

        return self.secret(key=key, scope=scope).delete()

    # ------------------------------------------------------------- permissions

    def list_permissions(
        self,
        scope: Union["Scope", str],
    ) -> list["Permission"]:
        return self.scope(scope).list_permissions() if isinstance(scope, str) else scope.list_permissions()

    def permission(
        self,
        scope: Union["Scope", str],
        principal: str,
    ) -> Optional["Permission"]:
        target = self.scope(scope) if isinstance(scope, str) else scope
        return target.permission(principal)

    def set_permission(
        self,
        scope: Union["Scope", str],
        principal: Any,
        acl: Any = None,
    ) -> "Permission":
        target = self.scope(scope) if isinstance(scope, str) else scope
        return target.set_permission(principal, acl=acl)

    def set_permissions(
        self,
        scope: Union["Scope", str],
        permissions: Iterable[Any],
    ) -> list["Permission"]:
        target = self.scope(scope) if isinstance(scope, str) else scope
        return [target.set_permission(p) for p in permissions]

    def delete_permission(
        self,
        scope: Union["Scope", str],
        principal: str,
    ) -> None:
        target = self.scope(scope) if isinstance(scope, str) else scope
        return target.delete_permission(principal)
