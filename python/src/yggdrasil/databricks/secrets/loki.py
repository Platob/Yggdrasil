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
    description = "List Databricks secret scopes and a scope's keys (names only)."
    preprompt = (
        "You list secret scopes/keys via dbc.secrets — NAMES ONLY, never the "
        "values. A UC Scope identifies by its key. Secrets are consumed at "
        "runtime by jobs; never print or request a secret value."
    )

    def run(self, agent: "Loki", *, scope: Optional[str] = None, **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if scope:
            sc = client.secrets.scope(scope)
            return {"scope": scope, "keys": names(sc.list_secrets(), attrs=("key", "name"))}
        # A UC secret Scope identifies by ``.key`` (not ``.name``).
        return {"scopes": names(client.secrets.list_scopes(), attrs=("key", "name"))}
