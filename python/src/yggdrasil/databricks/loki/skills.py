"""Cross-service Databricks Loki skill: the **managed MCP** connector.

The per-service skills live next to their service code in
``databricks/<service>/loki.py`` (see :mod:`yggdrasil.databricks.loki`). What
remains here is the one skill that isn't a single ``dbc`` accessor:
``databricks-mcp`` connects *out* to the workspace's managed MCP servers
(Unity Catalog functions, Genie, vector search) using the agent's credentials.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import register

from .base import DatabricksServiceSkill

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksMCPSkill"]

#: Databricks-hosted managed MCP server URL templates, keyed by kind.
_MCP_TEMPLATES = {
    "functions": "/api/2.0/mcp/functions/{catalog}/{schema}",
    "genie": "/api/2.0/mcp/genie/{space}",
    "vector-search": "/api/2.0/mcp/vector-search/{catalog}/{schema}",
}


def _mcp_url(host: str, kind: str, **parts: Any) -> str:
    """Build a Databricks managed-MCP server URL from *host* + *kind* + params."""
    tmpl = _MCP_TEMPLATES.get(kind.replace("_", "-"))
    if tmpl is None:
        raise ValueError(f"unknown MCP kind {kind!r}; known: {', '.join(_MCP_TEMPLATES)}")
    missing = [k for k in ("catalog", "schema", "space") if "{" + k + "}" in tmpl and not parts.get(k)]
    if missing:
        raise ValueError(f"MCP kind {kind!r} needs: {', '.join(missing)}")
    return host.rstrip("/") + tmpl.format(**{k: parts.get(k, "") for k in ("catalog", "schema", "space")})


async def _mcp_tools(url: str, headers: dict) -> list[str]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.list_tools()
            return [t.name for t in res.tools]


async def _mcp_call(url: str, headers: dict, tool: str, args: dict) -> Any:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(tool, args)
            return [getattr(c, "text", str(c)) for c in getattr(res, "content", [])]


@register
class DatabricksMCPSkill(DatabricksServiceSkill):
    """Connect to a Databricks-hosted MCP server and list (or call) its tools.

    Reaches the workspace's **managed MCP servers** — Unity Catalog functions
    (``kind="functions"``, ``catalog``/``schema``), Genie (``kind="genie"``,
    ``space``), or vector search (``kind="vector-search"``) — or any explicit
    ``url`` (e.g. a custom MCP app), authenticating with the agent's Databricks
    credentials. Lists the server's tools, or calls ``tool`` with ``args``.
    Requires the ``mcp`` package.
    """

    name = "databricks-mcp"
    description = "List or call tools on a Databricks-hosted MCP server (UC functions / Genie / vector search)."
    preprompt = (
        "You bridge to Databricks managed MCP servers via dbc credentials — UC "
        "functions, Genie, or vector search. List a server's tools, then call "
        "one with explicit args; treat tool calls as real actions."
    )

    def run(
        self,
        agent: "Loki",
        *,
        url: Optional[str] = None,
        kind: str = "functions",
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        space: Optional[str] = None,
        tool: Optional[str] = None,
        args: Optional[dict] = None,
        **_: Any,
    ) -> dict[str, Any]:
        import asyncio

        w = self._client(agent).workspace_client()
        headers = dict(w.config.authenticate())  # Authorization bearer header
        target = url or _mcp_url(w.config.host, kind,
                                 catalog=catalog, schema=schema, space=space)
        if tool:
            result = asyncio.run(_mcp_call(target, headers, tool, args or {}))
            return {"server": target, "tool": tool, "result": result}
        return {"server": target, "tools": asyncio.run(_mcp_tools(target, headers))}
