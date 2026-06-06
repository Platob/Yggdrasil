"""Tests for Loki reasoning engines and engine selection."""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.loki import Loki
from yggdrasil.loki.capability import Backend
from yggdrasil.loki.engine import Completion, TokenEngine
from yggdrasil.loki.engines import ClaudeEngine, DatabricksServingEngine, OpenAIEngine


class TestEngineContract(unittest.TestCase):
    def test_generate_wraps_complete(self):
        class Fixed(TokenEngine):
            name = "fixed"

            def available(self):
                return True

            def complete(self, messages, *, system=None, max_tokens=16000, **o):
                return Completion(text="hi " + messages[-1]["content"])

        self.assertEqual(Fixed().generate("there"), "hi there")


class TestOpenAIEngine(unittest.TestCase):
    def test_available_requires_key(self):
        self.assertFalse(OpenAIEngine(api_key=None).available())
        self.assertTrue(OpenAIEngine(api_key="sk-x").available())

    def test_complete_calls_chat_completions(self):
        fake_openai = types.ModuleType("openai")
        client = MagicMock()
        msg = MagicMock(); msg.content = "answer"
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)], model="gpt-4o-mini", usage=None,
        )
        fake_openai.OpenAI = MagicMock(return_value=client)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            out = OpenAIEngine(api_key="sk-x").generate("q", system="sys")
        self.assertEqual(out, "answer")
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(sent[0], {"role": "system", "content": "sys"})


def _fake_anthropic(text: str = "claude says hi"):
    """A stand-in ``anthropic`` module whose client returns *text*."""
    fake = types.ModuleType("anthropic")
    client = MagicMock()
    block = MagicMock(); block.type = "text"; block.text = text
    client.messages.create.return_value = MagicMock(
        content=[block], model="claude-opus-4-8", usage=None,
    )
    fake.Anthropic = MagicMock(return_value=client)
    return fake, client


class TestClaudeEngine(unittest.TestCase):
    _CRED_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")

    def setUp(self):
        import os

        # Isolate credential resolution from the host environment / creds file.
        self._saved = {k: os.environ.pop(k, None) for k in self._CRED_VARS}
        self._nofile = patch(
            "yggdrasil.loki.engines.claude_engine._oauth_token_from_file",
            return_value=None,
        )
        self._nofile.start()

    def tearDown(self):
        import os

        self._nofile.stop()
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_default_model_is_current_opus(self):
        self.assertEqual(ClaudeEngine().default_model, "claude-opus-4-8")

    def test_complete_passes_system_separately(self):
        fake, client = _fake_anthropic()
        with patch.dict(sys.modules, {"anthropic": fake}):
            out = ClaudeEngine(api_key="k").generate("q", system="be terse")
        self.assertEqual(out, "claude says hi")
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["system"], "be terse")
        self.assertNotIn("system", [m["role"] for m in kwargs["messages"]])
        # API-key path uses x-api-key, not bearer/OAuth.
        self.assertEqual(fake.Anthropic.call_args.kwargs.get("api_key"), "k")
        self.assertNotIn("auth_token", fake.Anthropic.call_args.kwargs)

    # -- keyless OAuth / subscription auth ---------------------------------

    def test_unavailable_without_any_credential(self):
        self.assertFalse(ClaudeEngine().available())

    def test_available_with_only_oauth_token(self):
        eng = ClaudeEngine(auth_token="sk-ant-oat01-x")
        self.assertTrue(eng.available())
        self.assertTrue(eng.uses_oauth)

    def test_api_key_takes_precedence_over_oauth(self):
        eng = ClaudeEngine(api_key="k", auth_token="oat")
        self.assertFalse(eng.uses_oauth)

    def test_oauth_token_resolved_from_env(self):
        import os

        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = "sk-ant-oat01-env"
        try:
            self.assertEqual(ClaudeEngine().auth_token, "sk-ant-oat01-env")
        finally:
            os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)

    def test_oauth_token_resolved_from_credentials_file(self):
        with patch(
            "yggdrasil.loki.engines.claude_engine._oauth_token_from_file",
            return_value="sk-ant-oat01-file",
        ):
            self.assertEqual(ClaudeEngine().auth_token, "sk-ant-oat01-file")

    def test_oauth_complete_sends_bearer_and_identity(self):
        fake, client = _fake_anthropic("via subscription")
        with patch.dict(sys.modules, {"anthropic": fake}):
            out = ClaudeEngine(auth_token="sk-ant-oat01-x").generate(
                "q", system="be terse"
            )
        self.assertEqual(out, "via subscription")
        # Bearer token (no api_key) + the OAuth beta header.
        ctor = fake.Anthropic.call_args.kwargs
        self.assertEqual(ctor.get("auth_token"), "sk-ant-oat01-x")
        self.assertNotIn("api_key", ctor)
        self.assertEqual(ctor["default_headers"]["anthropic-beta"], "oauth-2025-04-20")
        # System prompt leads with the Claude Code identity, caller's follows.
        system = client.messages.create.call_args.kwargs["system"]
        self.assertEqual(system[0]["text"], "You are Claude Code, Anthropic's official CLI for Claude.")
        self.assertEqual(system[1]["text"], "be terse")


class TestDatabricksServingEngine(unittest.TestCase):
    def test_defaults_to_the_lowest_endpoint(self):
        # The smallest / cheapest broadly-available Foundation Model endpoint —
        # cheap by default unless a caller opts up via endpoint=.
        self.assertEqual(
            DatabricksServingEngine().endpoint, "databricks-meta-llama-3-1-8b-instruct"
        )

    def test_complete_uses_openai_compatible_client(self):
        client = MagicMock()
        oai = MagicMock()
        msg = MagicMock(); msg.content = "served"
        oai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)], model="ep",
        )
        client.workspace_client.return_value.serving_endpoints.get_open_ai_client.return_value = oai
        eng = DatabricksServingEngine(client=client, endpoint="ep")
        self.assertEqual(eng.generate("hi"), "served")
        self.assertEqual(oai.chat.completions.create.call_args.kwargs["model"], "ep")


class TestEngineSelection(unittest.TestCase):
    def test_prefers_claude_then_openai(self):
        loki = Loki()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k2"}):
            self.assertEqual(loki.engine().name, "claude")

    def test_reason_raises_without_engine(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        with patch.object(Loki, "_engine_instances", return_value={
            "claude": MagicMock(available=lambda: False),
        }):
            with self.assertRaises(RuntimeError):
                loki.reason("hello")

    def test_named_engine_lookup(self):
        loki = Loki()
        self.assertEqual(loki.engine("openai").name, "openai")
        with self.assertRaises(KeyError):
            loki.engine("nope")


if __name__ == "__main__":
    unittest.main()
