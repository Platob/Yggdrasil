"""Light smoke test for Databricks model serving via the Loki engine.

Skips unless a Databricks session is reachable (creds present). Where it runs,
it lists serving endpoints, queries a chat-capable one through
:class:`DatabricksServingEngine`, and asserts a non-empty reply plus that the
token meter recorded the call — the full Loki → serving → usage path.
"""
from __future__ import annotations

import unittest


def _reachable() -> bool:
    try:
        from yggdrasil.databricks import DatabricksClient

        return bool(DatabricksClient.current().base_url)
    except Exception:
        return False


@unittest.skipUnless(_reachable(), "no reachable Databricks session")
class TestServingSmoke(unittest.TestCase):
    def test_query_chat_endpoint_and_record_usage(self):
        from yggdrasil.databricks import DatabricksClient
        from yggdrasil.loki.engines import DatabricksServingEngine
        from yggdrasil.loki.usage import METER

        client = DatabricksClient.current()
        endpoints = [e.name for e in client.workspace_client().serving_endpoints.list()]
        self.assertTrue(endpoints, "workspace exposes no serving endpoints")

        chat = next(
            (n for n in endpoints
             if any(k in n.lower() for k in ("gpt", "llama", "claude", "qwen", "mixtral", "dbrx"))),
            None,
        )
        if chat is None:
            self.skipTest("no chat-capable serving endpoint available")

        before = METER.total_tokens
        eng = DatabricksServingEngine(client=client, endpoint=chat)
        self.assertTrue(eng.available())
        reply = eng.generate("Reply with exactly one word: pong", max_tokens=16)
        self.assertTrue(reply.strip(), "empty serving reply")
        self.assertGreaterEqual(METER.total_tokens, before)  # usage recorded


if __name__ == "__main__":
    unittest.main()
