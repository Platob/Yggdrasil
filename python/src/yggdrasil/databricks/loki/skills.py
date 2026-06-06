"""Specialized Databricks Loki behaviors — one agent skill per dbc service.

Each behavior is a :class:`~yggdrasil.loki.behavior.LokiSkill` that
``requires="databricks"`` and drives one ``dbc.<service>`` accessor through
the agent's token provider (``agent.databricks``). They register into the
global Loki catalog on import, so ``ygg loki behaviors`` lists them and
``ygg loki run databricks-sql --kwarg query='…'`` dispatches them — the
"databricks on databricks" scheme, one specialized skill per surface.

Read/list operations are the default (safe to run); the few that act (run a
job, query a serving endpoint, execute SQL) take explicit arguments. The
catalog mirrors the client's accessors: ``sql``, ``tables``, ``warehouses``,
``jobs``, ``clusters``, ``volumes``, ``secrets``, ``iam``, ``serving``, and
``genie`` (AI/BI Genie spaces).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import LokiSkill, register

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = [
    "DatabricksServiceSkill",
    "DatabricksMCPSkill",
    "DatabricksSQLSkill",
    "DatabricksCatalogsSkill",
    "DatabricksTablesSkill",
    "DatabricksWarehousesSkill",
    "DatabricksJobsSkill",
    "DatabricksClustersSkill",
    "DatabricksVolumesSkill",
    "DatabricksSecretsSkill",
    "DatabricksIAMSkill",
    "DatabricksServingSkill",
    "GenieSkill",
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


class DatabricksServiceSkill(LokiSkill):
    """Base for the Databricks service skills — guards the session for all of them."""

    requires = "databricks"
    preprompt = (
        "You are a Databricks expert operating through yggdrasil's DatabricksClient "
        "(dbc.<service> accessors). Prefer serverless compute for inner I/O, Unity "
        "Catalog three-level names, Arrow-returning SQL (results are Tabular), and "
        "the seeded ygg wheel environments — never %pip install per run. Be precise, "
        "least-privilege, and safe with destructive statements."
    )

    def _client(self, agent: "Loki") -> Any:
        client = agent.databricks
        if client is None:  # available() already guards; belt-and-suspenders
            raise RuntimeError("no Databricks session — run `ygg databricks configure`")
        return client


@register
class DatabricksSQLSkill(DatabricksServiceSkill):
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
class DatabricksCatalogsSkill(DatabricksServiceSkill):
    """Navigate Unity Catalog — list catalogs, or the schemas within one
    (``dbc.catalogs``). The entry point for "what data is here?"."""

    name = "databricks-catalogs"
    description = "List Unity Catalog catalogs, or the schemas in a catalog."

    def run(self, agent: "Loki", *, catalog: Optional[str] = None, **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if catalog:
            schemas = [
                getattr(s, "schema_name", None) or getattr(s, "name", None) or str(s)
                for s in client.catalogs.catalog(catalog).schemas()
            ]
            return {"catalog": catalog, "schemas": schemas}
        return {"catalogs": _names(client.catalogs.list_catalogs())}


@register
class DatabricksTablesSkill(DatabricksServiceSkill):
    """List tables in a catalog.schema, or describe one — through the Unity
    Catalog ``dbc.tables`` accessor (the UC API), so it needs **no SQL
    warehouse** and returns typed column metadata."""

    name = "databricks-tables"
    description = "List tables in a catalog.schema, or describe a table (Unity Catalog API)."

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
            t = client.tables.get(full)
            if t is None:
                return {"table": full, "found": False}
            return {
                "table": t.full_name(),
                "type": str(t.table_type),
                "columns": [
                    {"name": c.name, "type": str(getattr(getattr(c, "field", None), "dtype", ""))}
                    for c in t.columns
                ],
            }
        tables = client.tables.list_tables(catalog_name=catalog, schema_name=schema)
        return {"catalog": catalog, "schema": schema, "tables": _names(tables)}


@register
class DatabricksWarehousesSkill(DatabricksServiceSkill):
    """List the SQL warehouses reachable to the agent."""

    name = "databricks-warehouses"
    description = "List the Databricks SQL warehouses."

    def run(self, agent: "Loki", **_: Any) -> dict[str, Any]:
        whs = self._client(agent).warehouses.list_warehouses()
        return {"warehouses": _names(whs)}


@register
class DatabricksJobsSkill(DatabricksServiceSkill):
    """List jobs, or trigger a run of one by name/id (returns the new run)."""

    name = "databricks-jobs"
    description = "List Databricks jobs, or trigger a run of one by name/id."

    def run(
        self,
        agent: "Loki",
        *,
        run: Optional[str] = None,
        parameters: Optional[dict] = None,
        **_: Any,
    ) -> dict[str, Any]:
        client = self._client(agent)
        if run:
            job = client.jobs.get(run, default=None)
            if job is None:
                return {"job": run, "found": False}
            job_run = job.run(parameters=parameters or None)
            return {
                "job": job.name or job.job_id,
                "job_id": job.job_id,
                "run_id": getattr(job_run, "run_id", None),
                "url": str(getattr(job_run, "url", "")) or None,
            }
        return {"jobs": _names(client.jobs.list())}


@register
class DatabricksClustersSkill(DatabricksServiceSkill):
    """List the compute clusters."""

    name = "databricks-clusters"
    description = "List the Databricks compute clusters."

    def run(self, agent: "Loki", **_: Any) -> dict[str, Any]:
        return {"clusters": _names(self._client(agent).compute.clusters.list())}


@register
class DatabricksVolumesSkill(DatabricksServiceSkill):
    """List Unity Catalog volumes (optionally within a catalog.schema)."""

    name = "databricks-volumes"
    description = "List Unity Catalog volumes."

    def run(self, agent: "Loki", *, catalog: Optional[str] = None,
            schema: Optional[str] = None, **_: Any) -> dict[str, Any]:
        vols = self._client(agent).volumes.list(catalog, schema) if catalog else \
            self._client(agent).volumes.list()
        return {"volumes": _names(vols)}


@register
class DatabricksSecretsSkill(DatabricksServiceSkill):
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
class DatabricksIAMSkill(DatabricksServiceSkill):
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
class DatabricksServingSkill(DatabricksServiceSkill):
    """List model-serving endpoints, or query one with a prompt."""

    name = "databricks-serving"
    description = "List Databricks serving endpoints, or query one with a prompt."

    def run(self, agent: "Loki", *, endpoint: Optional[str] = None,
            prompt: Optional[str] = None, **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if prompt:
            from yggdrasil.loki.engines import DatabricksServingEngine

            eng = DatabricksServingEngine(client=client, endpoint=endpoint)
            # The domain preprompt steers the served model toward the best answer.
            return {"endpoint": eng.endpoint,
                    "reply": eng.generate(prompt, system=self.preprompt)}
        eps = client.workspace_client().serving_endpoints.list()
        return {"endpoints": _names(eps, attrs=("name", "id"))}


@register
class GenieSkill(DatabricksServiceSkill):
    """Ask a Databricks AI/BI Genie space a question (text + SQL + rows).

    A Databricks-native skill: it drives ``dbc.genie`` through the agent's
    token provider. When no space is named it reasons against the first space
    the current user can reach.
    """

    name = "genie"
    description = "Ask a Databricks AI/BI Genie space a question (text + SQL + rows)."

    def run(self, agent: "Loki", *, question: str, space: Optional[str] = None,
            rows: bool = False, **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if space is None:
            spaces = client.genie.spaces()
            if not spaces:
                raise RuntimeError("no Genie spaces are accessible to this user")
            target = spaces[0]
        else:
            target = client.genie.space(space)

        answer = target.ask(question)
        out: dict[str, Any] = {
            "space_id": target.space_id,
            "conversation_id": answer.conversation_id,
            "text": answer.text,
            "query": answer.query,
            "statement_id": answer.statement_id,
        }
        if rows and answer.query:
            out["rows"] = answer.to_polars()
        return out
