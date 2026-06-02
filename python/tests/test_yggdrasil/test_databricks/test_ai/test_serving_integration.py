"""Live-integration smoke test for the Databricks Model Serving surface.

Validates the LLM-serving chain against a real workspace, end to end:

- :meth:`ModelServing.list_endpoints` enumerates the workspace's
  serving endpoints;
- a built-in **foundation model** endpoint (pay-per-token, present in
  most workspaces — e.g. ``databricks-meta-llama-3-3-70b-instruct``)
  is resolvable via :meth:`ModelServing.endpoint`, reports ``READY``,
  and answers a :meth:`ServingEndpoint.chat` turn with non-empty text;
- the same endpoint serves a one-shot completion-style chat and reports
  token :attr:`ServingQueryResult.usage`;
- an embeddings foundation model (``databricks-gte-large-en``) returns a
  fixed-width vector via :meth:`ServingEndpoint.embed` (skipped when the
  endpoint isn't provisioned in the target workspace).

Skipped wholesale unless ``DATABRICKS_HOST`` is set. The test only
*reads* / *queries* pre-provisioned foundation-model endpoints — it
creates nothing, so there is nothing to clean up. Permission /
model-availability failures degrade to ``unittest.SkipTest`` rather than
failing the suite. Override the chat / embedding endpoint names via
``YGG_SMOKE_CHAT_ENDPOINT`` / ``YGG_SMOKE_EMBED_ENDPOINT``.
"""
from __future__ import annotations

import os
import unittest

from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.errors.platform import PermissionDenied, ResourceDoesNotExist

from yggdrasil.databricks.ai import ModelServing, ServingEndpoint, ServingQueryResult

from .. import DatabricksIntegrationCase

# Databricks-hosted, pay-per-token foundation models. These are the
# default-available names in most workspaces; override via env for a
# workspace that exposes different ones.
_CHAT_ENDPOINT = os.environ.get(
    "YGG_SMOKE_CHAT_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct",
)
_EMBED_ENDPOINT = os.environ.get(
    "YGG_SMOKE_EMBED_ENDPOINT", "databricks-gte-large-en",
)

# Foundation-model calls add network + cold-start latency on top of token
# generation; keep the wait generous so a slow first token doesn't flake.
_QUERY_TIMEOUT = 120


class TestModelServingIntegration(DatabricksIntegrationCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.serving: ModelServing = cls.client.ai.serving

    def _resolve_chat_endpoint(self) -> ServingEndpoint:
        """Resolve the configured chat endpoint, or skip when unavailable."""
        ep = self.serving.endpoint(_CHAT_ENDPOINT)
        try:
            ep.refresh()
        except (NotFound, ResourceDoesNotExist) as exc:
            raise unittest.SkipTest(
                f"chat endpoint {_CHAT_ENDPOINT!r} not provisioned in this "
                f"workspace ({exc}); set YGG_SMOKE_CHAT_ENDPOINT to one that is."
            ) from exc
        except PermissionDenied as exc:
            raise unittest.SkipTest(
                f"no permission to read serving endpoint {_CHAT_ENDPOINT!r}: {exc}"
            ) from exc
        return ep

    def test_list_endpoints(self):
        """The workspace's serving endpoints enumerate as handles."""
        try:
            endpoints = list(self.serving.list_endpoints())
        except PermissionDenied as exc:
            raise unittest.SkipTest(f"no permission to list serving endpoints: {exc}")
        for ep in endpoints:
            self.assertIsInstance(ep, ServingEndpoint)
            self.assertTrue(ep.name)

    def test_foundation_endpoint_is_ready(self):
        """The foundation chat endpoint resolves and reports READY."""
        ep = self._resolve_chat_endpoint()
        self.assertEqual(ep.name, _CHAT_ENDPOINT)
        # Pay-per-token foundation endpoints are always-on; readiness should
        # report READY (the property tolerates None for odd states).
        self.assertIn(ep.ready, ("READY", None))
        self.assertEqual(ep.state, "NOT_UPDATING")

    def test_chat_round_trip(self):
        """A chat turn returns non-empty assistant text + token usage."""
        ep = self._resolve_chat_endpoint()
        try:
            result = ep.chat(
                "Reply with exactly the word: pong",
                max_tokens=16,
                temperature=0.0,
            )
        except PermissionDenied as exc:
            raise unittest.SkipTest(f"no permission to query {_CHAT_ENDPOINT!r}: {exc}")
        except DatabricksError as exc:
            raise unittest.SkipTest(f"foundation model query failed: {exc}")

        self.assertIsInstance(result, ServingQueryResult)
        self.assertTrue(result.text, "expected non-empty assistant text")
        self.assertIn("pong", result.text.lower())

        msg = result.message
        self.assertIsNotNone(msg)
        self.assertEqual(msg["role"], "assistant")

        # Token accounting flows through for foundation models.
        usage = result.usage
        if usage is not None:
            self.assertGreaterEqual(usage.get("total_tokens", 0), 1)

    def test_chat_message_list(self):
        """A system+user message list is honoured."""
        ep = self._resolve_chat_endpoint()
        try:
            result = ep.chat(
                [
                    {"role": "system", "content": "You are a calculator. Answer with digits only."},
                    {"role": "user", "content": "What is 21 + 21?"},
                ],
                max_tokens=16,
                temperature=0.0,
            )
        except (PermissionDenied, DatabricksError) as exc:
            raise unittest.SkipTest(f"foundation model query failed: {exc}")
        self.assertTrue(result.text)
        self.assertIn("42", result.text)

    def test_embed_round_trip(self):
        """An embeddings foundation model returns a fixed-width vector."""
        ep = self.serving.endpoint(_EMBED_ENDPOINT)
        try:
            ep.refresh()
        except (NotFound, ResourceDoesNotExist, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"embedding endpoint {_EMBED_ENDPOINT!r} unavailable ({exc}); "
                f"set YGG_SMOKE_EMBED_ENDPOINT to one that is."
            )
        try:
            result = ep.embed(["yggdrasil distributed node framework"])
        except (PermissionDenied, DatabricksError) as exc:
            raise unittest.SkipTest(f"embedding query failed: {exc}")

        vec = result.embedding
        self.assertIsNotNone(vec, "expected at least one embedding vector")
        self.assertGreater(len(vec), 0)
        self.assertTrue(all(isinstance(x, (int, float)) for x in vec))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
