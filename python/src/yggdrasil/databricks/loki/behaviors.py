"""Specialized Databricks Loki behaviors — one agent skill per dbc service.

Each behavior is a :class:`~yggdrasil.loki.behavior.LokiBehavior` that
``requires="databricks"`` and drives one ``dbc.<service>`` accessor through
the agent's token provider (``agent.databricks``). They register into the
global Loki catalog on import, so ``ygg loki behaviors`` lists them and
``ygg loki run databricks-sql --kwarg query='…'`` dispatches them — the
"databricks on databricks" scheme, one specialized skill per surface.

Read/list operations are the default (safe to run); the few that act (run a
job, query a serving endpoint, execute SQL) take explicit arguments. The
catalog mirrors the client's accessors: ``sql``, ``tables``, ``warehouses``,
``jobs``, ``clusters``, ``volumes``, ``secrets``, ``iam``, ``serving``
(``genie`` ships in the global behavior set).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.behavior import LokiBehavior, register

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = [
    "DatabricksServiceBehavior",
    "DatabricksMCPBehavior",
    "DatabricksSQLBehavior",
    "DatabricksTablesBehavior",
    "DatabricksWarehousesBehavior",
    "DatabricksJobsBehavior",
    "DatabricksClustersBehavior",
    "DatabricksVolumesBehavior",
    "DatabricksSecretsBehavior",
    "DatabricksIAMBehavior",
    "DatabricksServingBehavior",
]


def _names(items: Any, *, attrs: tuple[str, ...] = ("name", "id", "full_name")) -> list[Any]:
    """Summarize an iterable of SDK objects to their most identifying field."""
    out: list[Any] = []
    for it in list(items or [])[:200]:
        for a in attrs:
            v = getattr(it, a, None)
            if v is not None:
                out.append(v)
                break
        else:
            out.append(str(it))
    return out


def _frame(result: Any) -> Any:
    """Best-effort tabular view of a statement/Genie result."""
    for m in ("to_polars", "to_pandas", "to_pylist"):
        fn = getattr(result, m, None)
        if callable(fn):
            return fn()
    return result


class DatabricksServiceBehavior(LokiBehavior):
    """Base for the Databricks service skills — guards the session for all of them."""

    requires = "databricks"

    def _client(self, agent: "Loki") -> Any:
        client = agent.databricks
        if client is None:  # available() already guards; belt-and-suspenders
            raise RuntimeError("no Databricks session — run `ygg databricks configure`")
        return client


@register
class DatabricksSQLBehavior(DatabricksServiceBehavior):
    """Run SQL on a Databricks warehouse and return the rows."""

    name = "databricks-sql"
    description = "Execute a SQL query on a Databricks SQL warehouse → rows."

    def run(self, agent: "Loki", *, query: str, rows: bool = True, **_: Any) -> dict[str, Any]:
        result = self._client(agent).sql.execute(query)
        out: dict[str, Any] = {"query": query, "statement_id": getattr(result, "statement_id", None)}
        if rows:
            frame = _frame(result)
            out["rows"] = frame
            out["row_count"] = getattr(frame, "height", getattr(frame, "shape", [None])[0]
                                       if hasattr(frame, "shape") else None)
        return out


@register
class DatabricksTablesBehavior(DatabricksServiceBehavior):
    """List tables (or describe one) in a catalog.schema — via SHOW/DESCRIBE."""

    name = "databricks-tables"
    description = "List tables in a catalog.schema, or describe a table."

    def run(
        self,
        agent: "Loki",
        *,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        client = self._client(agent)
        if table:
            full = ".".join(p for p in (catalog, schema, table) if p)
            return {"table": full, "schema": _frame(client.sql.execute(f"DESCRIBE TABLE {full}"))}
        where = ".".join(p for p in (catalog, schema) if p)
        stmt = f"SHOW TABLES IN {where}" if where else "SHOW TABLES"
        return {"in": where or "(current)", "tables": _frame(client.sql.execute(stmt))}


@register
class DatabricksWarehousesBehavior(DatabricksServiceBehavior):
    """List the SQL warehouses reachable to the agent."""

    name = "databricks-warehouses"
    description = "List the Databricks SQL warehouses."

    def run(self, agent: "Loki", **_: Any) -> dict[str, Any]:
        whs = self._client(agent).warehouses.list_warehouses()
        return {"warehouses": _names(whs)}


@register
class DatabricksJobsBehavior(DatabricksServiceBehavior):
    """List jobs, or run one by name/id."""

    name = "databricks-jobs"
    description = "List Databricks jobs, or run one by name/id."

    def run(self, agent: "Loki", *, run: Optional[str] = None, **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if run:
            job = client.jobs.get(run)
            return {"ran": run, "run": str(getattr(job, "run", lambda: None)() or job)}
        return {"jobs": _names(client.jobs.list())}


@register
class DatabricksClustersBehavior(DatabricksServiceBehavior):
    """List the compute clusters."""

    name = "databricks-clusters"
    description = "List the Databricks compute clusters."

    def run(self, agent: "Loki", **_: Any) -> dict[str, Any]:
        return {"clusters": _names(self._client(agent).compute.clusters.list())}


@register
class DatabricksVolumesBehavior(DatabricksServiceBehavior):
    """List Unity Catalog volumes (optionally within a catalog.schema)."""

    name = "databricks-volumes"
    description = "List Unity Catalog volumes."

    def run(self, agent: "Loki", *, catalog: Optional[str] = None,
            schema: Optional[str] = None, **_: Any) -> dict[str, Any]:
        vols = self._client(agent).volumes.list(catalog, schema) if catalog else \
            self._client(agent).volumes.list()
        return {"volumes": _names(vols)}


@register
class DatabricksSecretsBehavior(DatabricksServiceBehavior):
    """List secret scopes (and a scope's secret keys — names only, never values)."""

    name = "databricks-secrets"
    description = "List Databricks secret scopes and a scope's keys (names only)."

    def run(self, agent: "Loki", *, scope: Optional[str] = None, **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if scope:
            sc = client.secrets.scope(scope)
            return {"scope": scope, "keys": _names(sc.list_secrets(), attrs=("key", "name"))}
        return {"scopes": _names(client.secrets.list_scopes(), attrs=("name",))}


@register
class DatabricksIAMBehavior(DatabricksServiceBehavior):
    """Who am I, and list workspace users/groups."""

    name = "databricks-iam"
    description = "Resolve the current user; list workspace users or groups."

    def run(self, agent: "Loki", *, what: str = "me", **_: Any) -> dict[str, Any]:
        iam = self._client(agent).iam
        if what == "users":
            return {"users": _names(iam.users(), attrs=("user_name", "name", "id"))}
        if what == "groups":
            return {"groups": _names(iam.groups(), attrs=("display_name", "name", "id"))}
        user = iam.current_user
        return {"me": getattr(user, "user_name", str(user))}


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
class DatabricksMCPBehavior(DatabricksServiceBehavior):
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


@register
class DatabricksServingBehavior(DatabricksServiceBehavior):
    """List model-serving endpoints, or query one with a prompt."""

    name = "databricks-serving"
    description = "List Databricks serving endpoints, or query one with a prompt."

    def run(self, agent: "Loki", *, endpoint: Optional[str] = None,
            prompt: Optional[str] = None, **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if prompt:
            from yggdrasil.loki.engines import DatabricksServingEngine

            eng = DatabricksServingEngine(client=client, endpoint=endpoint)
            return {"endpoint": eng.endpoint, "reply": eng.generate(prompt)}
        eps = client.workspace_client().serving_endpoints.list()
        return {"endpoints": _names(eps, attrs=("name", "id"))}
