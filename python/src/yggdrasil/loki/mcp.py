"""Loki as an MCP server — expose the agent to any MCP client.

``ygg loki mcp`` runs a `Model Context Protocol <https://modelcontextprotocol.io>`_
server (stdio) that surfaces Loki to editors/clients like Claude Desktop. The
agent's whole surface is available as MCP tools: reason with the best engine,
list/dispatch any behavior (``databricks-sql``, ``aws-s3``, ``genie``, ``web``,
…), fetch the web, and read capabilities. Requires the optional ``mcp`` package.

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
    from mcp.server.fastmcp import FastMCP

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

        Examples: behavior="databricks-sql" kwargs={"query": "SELECT 1"};
        behavior="aws-s3"; behavior="web" kwargs={"url": "https://…"}.
        """
        return _sanitize(agent.run(skill, **(kwargs or {})))

    @server.tool()
    def web(url: str, question: str = "") -> Any:
        """Fetch a URL (table / JSON / image / page), optionally answering a question."""
        return _sanitize(agent.run("web", url=url, question=question or None))

    @server.tool()
    def capabilities() -> dict:
        """Loki's identity, detected backends, engines, and behaviors (its card)."""
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
