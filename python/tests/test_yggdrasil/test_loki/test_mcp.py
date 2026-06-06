"""Tests for the Loki MCP server (tool registration + dispatch)."""
from __future__ import annotations

import asyncio
import unittest

try:
    import mcp  # noqa: F401

    _HAVE_MCP = True
except Exception:  # pragma: no cover
    _HAVE_MCP = False

from yggdrasil.loki import Loki
from yggdrasil.loki.skill import REGISTRY, LokiSkill, register
from yggdrasil.loki.capability import Backend
from yggdrasil.loki.mcp import _sanitize


class TestSanitize(unittest.TestCase):
    def test_coerces_unknown_to_str_and_keeps_basics(self):
        out = _sanitize({"n": 3, "items": ["a", object()], "ok": True, "x": None})
        self.assertEqual(out["n"], 3)
        self.assertEqual(out["items"][0], "a")
        self.assertIsInstance(out["items"][1], str)      # object → str
        self.assertEqual(out["ok"], True)
        self.assertIsNone(out["x"])


@unittest.skipUnless(_HAVE_MCP, "requires the mcp package")
class TestLokiMCPServer(unittest.TestCase):
    def setUp(self):
        self._saved = dict(REGISTRY)

    def tearDown(self):
        REGISTRY.clear()
        REGISTRY.update(self._saved)

    def _loki(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        return loki

    def test_registers_the_agent_tools(self):
        from yggdrasil.loki.mcp import build_server

        server = build_server(self._loki())
        names = {t.name for t in asyncio.run(server.list_tools())}
        self.assertEqual(names, {"reason", "skills", "run", "web", "capabilities"})

    def test_run_tool_dispatches_a_behavior(self):
        from yggdrasil.loki.mcp import build_server

        @register
        class _Echo(LokiSkill):
            name = "mcp-echo"
            description = "echo"

            def run(self, agent, **kw):
                return {"echoed": kw}

        server = build_server(self._loki())
        result = asyncio.run(server.call_tool("run", {"skill": "mcp-echo",
                                                      "kwargs": {"x": 1}}))
        # call_tool returns (content, structured) — assert the dispatch happened.
        self.assertIn("echoed", str(result))
        self.assertIn("'x': 1", str(result).replace('"', "'"))

    def test_behaviors_tool_lists_catalog(self):
        from yggdrasil.loki.mcp import build_server

        server = build_server(self._loki())
        result = asyncio.run(server.call_tool("skills", {}))
        self.assertIn("python_project", str(result))


if __name__ == "__main__":
    unittest.main()
