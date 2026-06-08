"""Loki skill for the **IAM** service (``dbc.iam``).

Identity and access: who the agent is authenticated as, and the workspace's
users and groups. ``what="me"`` resolves the current user (the SCIM /Me call),
``"users"`` / ``"groups"`` list the directory.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksIAMSkill"]


@register
class DatabricksIAMSkill(DatabricksServiceSkill):
    """Who am I, and list workspace users/groups."""

    name = "databricks-iam"
    description = "Resolve the current user; list workspace users or groups."
    preprompt = (
        "You answer identity/access questions via dbc.iam: who am I (the /Me "
        "call), or list users/groups. Use least-privilege framing; don't expose "
        "more identity detail than asked."
    )

    def run(self, agent: "Loki", *, what: str = "me", **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if what == "users":
            return {"users": names(client.iam.users(), attrs=("user_name", "name", "id"))}
        if what == "groups":
            return {"groups": names(client.iam.groups(), attrs=("display_name", "name", "id"))}
        # "who am I" — the SCIM /Me call (same path Loki.whoami uses).
        me = client.workspace_client().current_user.me()
        return {"me": getattr(me, "user_name", str(me)), "id": getattr(me, "id", None)}
