"""Loki as an MCP server — expose the agent to any MCP client.

``ygg loki mcp`` runs a `Model Context Protocol <https://modelcontextprotocol.io>`_
server (stdio) that surfaces Loki to editors/clients like Claude Desktop. The
agent's whole surface is available as MCP tools: ``reason`` with the best
engine, ``skills``/``run`` (dispatch any skill — ``databricks-sql``, ``aws-s3``,
``genie``, …), ``web``, ``guide`` (the optimized yggdrasil way), ``tabular``
(read any source → cached frame), ``engines``, ``usage`` (token KPIs),
``setup`` (a free local model), and ``capabilities``. Requires the optional
``mcp`` package (auto-installed on first use).

    from yggdrasil.loki.mcp import build_server
    build_server().run()        # stdio transport
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from .agent import Loki

__all__ = ["build_server", "main"]


def _sanitize(obj: Any) -> Any:
    """Make a behavior result JSON-safe for the MCP wire (frames → str, …)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return str(obj)[:8000]


def build_server(loki: "Optional[Loki]" = None) -> "FastMCP":
    """Build the Loki MCP server (does not run it). Registers the agent's tools."""
    from .runtime import load

    FastMCP = load("mcp.server.fastmcp", "mcp").FastMCP

    from . import Loki

    agent = loki or Loki.current()
    try:
        agent.load_specialists()  # surface databricks-*/aws-* if reachable
    except Exception:
        pass

    server = FastMCP("loki")

    @server.tool()
    def reason(prompt: str, engine: str = "", tier: str = "") -> str:
        """Reason about a prompt with Loki's best (or named) engine/tier."""
        return agent.reason(prompt, engine=engine or None, tier=tier or None)

    @server.tool()
    def skills() -> list[dict]:
        """List Loki's available skills (name, description, required backend)."""
        return [b.to_dict() for b in agent.skills()]

    @server.tool()
    def run(skill: str, kwargs: Optional[dict] = None) -> Any:
        """Run a Loki skill by name with keyword args.

        Examples: skill="databricks-sql" kwargs={"query": "SELECT 1"};
        skill="aws-s3"; skill="web" kwargs={"url": "https://…"}.
        """
        return _sanitize(agent.run(skill, **(kwargs or {})))

    @server.tool()
    def web(url: str, question: str = "") -> Any:
        """Fetch a URL (table / JSON / image / page), optionally answering a question."""
        return _sanitize(agent.run("web", url=url, question=question or None))

    @server.tool()
    def guide(task: str, topic: str = "", plan: bool = False) -> Any:
        """The optimized yggdrasil way to build *task* — matched recipes (the right
        abstraction, snippet, anti-pattern); ``plan=True`` adds a grounded plan."""
        kw = {"topic": topic} if topic else {"task": task, "plan": plan}
        return _sanitize(agent.run("guide", **kw))

    @server.tool()
    def tabular(url: str = "", cache: str = "", store: str = "") -> Any:
        """Read a data source (URL / local path / s3 / dbfs) into a cached frame —
        returns its shape, columns, a compact preview, and reuse next-steps."""
        kw = {k: v for k, v in (("url", url), ("cache", cache), ("store", store)) if v}
        return _sanitize(agent.run("tabular", **kw))

    @server.tool()
    def engines() -> list[dict]:
        """The reasoning engines: name, model, local/remote, and availability."""
        return [{"name": e.name, "model": e.model_label, "local": e.local,
                 "available": e.available()} for e in agent.engines()]

    @server.tool()
    def usage() -> dict:
        """Live token usage + USD KPIs — per (engine, model) and the global total."""
        from .usage import METER

        return _sanitize(METER.snapshot())

    @server.tool()
    def setup(model: str = "") -> Any:
        """Bootstrap a free local model sized to this machine (lazy-install on demand)."""
        return _sanitize(agent.bootstrap_local(model=model or None))

    @server.tool()
    def capabilities() -> dict:
        """Loki's identity, detected backends, engines, and skills (its card)."""
        return _sanitize(agent.card())

    return server


def main(argv: "list[str] | None" = None) -> int:
    """Run the Loki MCP server over stdio."""
    try:
        server = build_server()
    except ImportError:
        sys.stderr.write("MCP is not installed — `uv pip install mcp`\n")
        return 1
    server.run()  # stdio transport
    return 0
