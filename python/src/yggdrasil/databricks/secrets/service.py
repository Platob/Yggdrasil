from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, Any, Optional

from databricks.sdk.errors import ResourceDoesNotExist

from ..client import DatabricksService

if TYPE_CHECKING:
    from .resource import Secret, Scope, Permission

__all__ = ["Secrets"]


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Secrets(DatabricksService):

    def __getitem__(self, item):
        if "/" in item or ":" in item:
            return self.secret(item).refresh()
        return self.scope(item)

    def __setitem__(self, key, value):
        found = self.secret(key).refresh(raise_error=False)

        if not found.b64:
            return self.create_secret(key=key, value=value)
        else:
            return found.update(value=value)

    def __delitem__(self, key):
        return self.delete_secret(key=key)

    def scope(
        self,
        key: str,
    ) -> "Scope":
        from .resource import Scope

        return Scope.parse(
            key,
            service=self,
        )

    def create_scope(
        self,
        key: Optional[str] = None,
        *,
        permissions: Union[list["Permission"], None] = None,
        scope: Union["Scope", str, None] = None,
    ) -> "Scope":
        from .resource import Scope

        if not scope:
            scope = Scope.parse_mapping(
                key,
                service=self,
            )

        if not scope.key:
            raise ValueError("Scope must have a key to be created")

        LOGGER.debug("Creating scope %s", scope)

        client = self.client.workspace_client().secrets

        client.create_scope(scope=scope.key)

        LOGGER.info("Created scope %s", scope)

        return scope.update(permissions=permissions)

    def secret(
        self,
        key: str,
        *,
        scope: Union["Scope", str, None] = None,
    ) -> "Secret":
        from .resource import Secret

        return Secret.parse(
            key,
            service=self,
            scope=scope
        )

    def create_secret(
        self,
        key: str,
        value: Any,
        *,
        scope: Union["Scope", str, None] = None,
        permissions: Union[list["Permission"], None] = None,
        secret: Union["Secret", str] = None,
    ) -> "Secret":
        from .resource import Secret

        if not secret:
            secret = Secret.parse_mapping(
                key=key,
                scope=scope,
                service=self,
                value=value
            )

        if not secret.scope or not secret.key:
            raise ValueError("Secret must have both scope and key to be created")

        LOGGER.debug("Creating secret %s", secret)

        client = self.client.workspace_client().secrets

        try:
            client.put_secret(
                scope=secret.scope.key,
                key=secret.key,
                bytes_value=secret.b64,
            )
        except ResourceDoesNotExist as e:
            msg = str(e)

            if secret.scope.key in msg:
                secret.scope = self.create_scope(
                    key=secret.scope.key,
                    permissions=permissions,
                    scope=secret.scope,
                )

                client.put_secret(
                    scope=secret.scope.key,
                    key=secret.key,
                    bytes_value=secret.b64,
                )
            else:
                raise

        LOGGER.info("Created secret %s", secret)

        return secret.update(permissions=permissions)

    def delete_secret(
        self,
        key: str,
        *,
        scope: Union["Scope", str, None] = None,
    ) -> None:
        secret = self.secret(key=key, scope=scope)

        if not secret.scope or not secret.key:
            raise ValueError("Secret must have both scope and key to be deleted")

        LOGGER.debug("Deleting secret %s", secret)

        client = self.client.workspace_client().secrets

        try:
            client.delete_secret(
                scope=secret.scope.key,
                key=secret.key,
            )
        except ResourceDoesNotExist:
            LOGGER.warning("Secret %s does not exist; skipping delete", secret)