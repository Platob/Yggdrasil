"""Loki skill for the **secrets** service (``dbc.secrets``).

Databricks secret scopes hold named secrets (e.g. external credentials). This
skill lists scopes, and a scope's keys — **names only, never values**: secrets
are read at runtime by jobs, not surfaced to the agent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksSecretsSkill"]


@register
class DatabricksSecretsSkill(DatabricksServiceSkill):
    """List secret scopes (and a scope's secret keys — names only, never values)."""

    name = "databricks-secrets"
    description = "List secret scopes/keys (names only), or put/delete a secret, or create/delete a scope."
    preprompt = (
        "You manage secret scopes/keys via dbc.secrets. List NAMES ONLY — never "
        "print or request a secret value. You can put a secret (key+value into a "
        "scope), delete one, or create/delete a scope; treat these as real, "
        "stateful, security-sensitive actions."
    )

    def run(self, agent: "Loki", *, scope: Optional[str] = None, key: Optional[str] = None,
            value: Optional[str] = None, op: str = "list", **_: Any) -> dict[str, Any]:
        secrets = self._client(agent).secrets
        if op == "put":
            if not (scope and key and value is not None):
                raise ValueError("put needs scope=, key=, value=")
            secrets.create_secret(key, value, scope=scope)
            return {"scope": scope, "put": key}            # never echo the value
        if op in ("delete", "delete-secret"):
            if not (scope and key):
                raise ValueError("delete needs scope= and key=")
            secrets.delete_secret(key, scope=scope)
            return {"scope": scope, "deleted": key}
        if op == "create-scope":
            if not scope:
                raise ValueError("create-scope needs scope=")
            secrets.create_scope(scope)
            return {"created_scope": scope}
        if op == "delete-scope":
            if not scope:
                raise ValueError("delete-scope needs scope=")
            secrets.delete_scope(scope)
            return {"deleted_scope": scope}
        if scope:                                          # a scope's keys
            return {"scope": scope, "keys": names(secrets.scope(scope).list_secrets(), attrs=("key", "name"))}
        # A UC secret Scope identifies by ``.key`` (not ``.name``).
        return {"scopes": names(secrets.list_scopes(), attrs=("key", "name"))}
