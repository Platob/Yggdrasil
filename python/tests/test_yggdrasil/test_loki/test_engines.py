"""Tests for Loki reasoning engines and engine selection."""
from __future__ import annotations

import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.loki import Loki
from yggdrasil.loki.capability import Backend
from yggdrasil.loki.engine import Completion, TokenEngine
from yggdrasil.loki.engines import (
    ClaudeEngine,
    DatabricksServingEngine,
    OllamaEngine,
    OpenAIEngine,
    TransformersEngine,
)


class TestEngineContract(unittest.TestCase):
    def test_generate_wraps_complete(self):
        class Fixed(TokenEngine):
            name = "fixed"

            def available(self):
                return True

            def complete(self, messages, *, system=None, max_tokens=16000, **o):
                return Completion(text="hi " + messages[-1]["content"])

        self.assertEqual(Fixed().generate("there"), "hi there")


class TestStreaming(unittest.TestCase):
    def test_base_stream_falls_back_to_complete(self):
        class Fixed(TokenEngine):
            name = "fixed"

            def available(self):
                return True

            def complete(self, messages, *, system=None, max_tokens=16000, tier=None, **o):
                return Completion(text="whole reply")

        self.assertEqual(list(Fixed().generate_stream("x")), ["whole reply"])

    def test_claude_streams_text_and_records_usage(self):
        from yggdrasil.loki.usage import METER

        class _Stream:
            text_stream = ["Hel", "lo!"]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get_final_message(self):
                m = MagicMock()
                m.usage = MagicMock(input_tokens=5, output_tokens=2)
                return m

        fake = types.ModuleType("anthropic")
        client = MagicMock()
        client.messages.stream.return_value = _Stream()
        fake.Anthropic = MagicMock(return_value=client)
        METER.reset()
        with patch.dict(sys.modules, {"anthropic": fake}):
            chunks = list(ClaudeEngine(api_key="k").generate_stream("hi"))
        self.assertEqual(chunks, ["Hel", "lo!"])
        row = METER.rows_for("claude")[0]
        self.assertEqual((row.input_tokens, row.output_tokens), (5, 2))


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


class TestAdaptiveModelSelection(unittest.TestCase):
    """The default model adapts to the request; pins/tiers override it."""

    def test_choose_tier_short_is_fast_long_is_deep(self):
        eng = ClaudeEngine(api_key="k")
        self.assertEqual(eng.choose_tier([{"role": "user", "content": "hi"}]), "fast")
        big = [{"role": "user", "content": "x" * 5000}]
        self.assertEqual(eng.choose_tier(big), "deep")

    def test_choose_tier_reasoning_signal_forces_deep(self):
        eng = ClaudeEngine(api_key="k")
        msgs = [{"role": "user", "content": "refactor the parser"}]
        self.assertEqual(eng.choose_tier(msgs), "deep")

    def test_resolve_model_pin_wins(self):
        eng = ClaudeEngine(api_key="k", model="claude-sonnet-4-6")
        self.assertEqual(
            eng.resolve_model(messages=[{"role": "user", "content": "hi"}]),
            "claude-sonnet-4-6",
        )

    def test_resolve_model_forced_tier(self):
        eng = ClaudeEngine(api_key="k")
        self.assertEqual(eng.resolve_model(tier="deep"), "claude-opus-4-8")
        self.assertEqual(eng.resolve_model(tier="fast"), "claude-haiku-4-5")

    def test_resolve_model_adaptive_default(self):
        eng = ClaudeEngine(api_key="k")
        self.assertEqual(
            eng.resolve_model(messages=[{"role": "user", "content": "hello"}]),
            "claude-haiku-4-5",  # short → fast
        )
        self.assertEqual(
            eng.resolve_model(messages=[{"role": "user", "content": "x" * 4000}]),
            "claude-opus-4-8",   # long → deep
        )

    def test_model_label_reports_adaptive_ceiling(self):
        self.assertEqual(ClaudeEngine(api_key="k").model_label, "claude-opus-4-8 (adaptive)")
        self.assertEqual(ClaudeEngine(api_key="k", model="m").model_label, "m")

    def test_claude_complete_uses_adaptive_model(self):
        fake, client = _fake_anthropic()
        with patch.dict(sys.modules, {"anthropic": fake}):
            ClaudeEngine(api_key="k").generate("hi")              # short → fast
        self.assertEqual(client.messages.create.call_args.kwargs["model"], "claude-haiku-4-5")

    def test_claude_complete_forced_tier_overrides(self):
        fake, client = _fake_anthropic()
        with patch.dict(sys.modules, {"anthropic": fake}):
            ClaudeEngine(api_key="k").generate("hi", tier="deep")  # short but forced
        self.assertEqual(client.messages.create.call_args.kwargs["model"], "claude-opus-4-8")

    def test_openai_complete_uses_adaptive_model(self):
        fake_openai = types.ModuleType("openai")
        client = MagicMock()
        msg = MagicMock(); msg.content = "ok"
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)], model="x", usage=None,
        )
        fake_openai.OpenAI = MagicMock(return_value=client)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            OpenAIEngine(api_key="sk-x").generate("design a distributed scheduler")
        # "design" is a reasoning signal → deep tier.
        self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "gpt-4o")


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


class TestTransformersEngine(unittest.TestCase):
    """The local HuggingFace engine — open models on this workstation, free."""

    def setUp(self):
        TransformersEngine._PIPES.clear()  # don't leak a pipeline across tests

    def test_is_local_and_free(self):
        self.assertTrue(TransformersEngine.local)
        from yggdrasil.loki.usage import price_for

        p = price_for("transformers", "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual((p.input_usd_per_mtok, p.output_usd_per_mtok), (0.0, 0.0))

    def test_available_requires_transformers_and_torch(self):
        with patch("importlib.util.find_spec", return_value=object()):
            self.assertTrue(TransformersEngine().available())
        with patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(TransformersEngine().available())

    def test_adaptive_tier_picks_small_then_large(self):
        eng = TransformersEngine()
        self.assertEqual(eng.resolve_model(tier="fast"), "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(eng.resolve_model(tier="deep"), "Qwen/Qwen2.5-1.5B-Instruct")

    def test_complete_runs_pipeline_and_records_usage(self):
        from yggdrasil.loki.usage import METER

        calls = {}

        def pipe(chat, **kw):
            calls["chat"] = chat
            calls["kw"] = kw
            return [{"generated_text": [{"role": "assistant", "content": "local reply"}]}]

        fake = types.ModuleType("transformers")
        fake.pipeline = MagicMock(return_value=pipe)
        METER.reset()
        with patch.dict(sys.modules, {"transformers": fake}):
            out = TransformersEngine(model="m").generate("hi", system="sys")
        self.assertEqual(out, "local reply")
        # System prompt leads the chat; new-token cap kept small for CPU.
        self.assertEqual(calls["chat"][0], {"role": "system", "content": "sys"})
        self.assertEqual(calls["kw"]["max_new_tokens"], 512)
        self.assertEqual(METER.rows_for("transformers")[0].calls, 1)
        self.assertEqual(METER.total_cost, 0.0)  # local is free

    @unittest.skipUnless(
        os.getenv("YGG_LOKI_TEST_LOCAL") == "1",
        "set YGG_LOKI_TEST_LOCAL=1 to run the real (downloads a small model) test",
    )
    def test_real_small_model_replies(self):
        eng = TransformersEngine(tier="fast")
        if not eng.available():
            self.skipTest("transformers/torch not installed")
        out = eng.generate("Reply with the single word: ping", system="Be terse.")
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip())


class TestOllamaEngine(unittest.TestCase):
    """The local Ollama engine — a model served on this machine, free."""

    def test_is_local_and_free(self):
        self.assertTrue(OllamaEngine.local)
        from yggdrasil.loki.usage import price_for

        p = price_for("ollama", "llama3.2")
        self.assertEqual((p.input_usd_per_mtok, p.output_usd_per_mtok), (0.0, 0.0))

    def test_host_from_env(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://box:1234/"}):
            self.assertEqual(OllamaEngine().host, "http://box:1234")

    def test_available_pings_tags(self):
        eng = OllamaEngine()
        ok = MagicMock(); ok.status = 200
        ok.__enter__ = lambda s: s; ok.__exit__ = lambda s, *a: False
        with patch("urllib.request.urlopen", return_value=ok):
            self.assertTrue(eng.available())
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            self.assertFalse(eng.available())

    def test_bootstrap_model_is_lightweight(self):
        self.assertEqual(OllamaEngine.bootstrap_model, "qwen2.5:3b")

    def test_installed_models_and_has_model(self):
        tags = json.dumps({"models": [{"name": "qwen2.5:3b"}, {"name": "llama3.2:latest"}]}).encode()
        resp = MagicMock()
        resp.read.return_value = tags
        resp.__enter__ = lambda s: s; resp.__exit__ = lambda s, *a: False
        with patch("urllib.request.urlopen", return_value=resp):
            eng = OllamaEngine()
            self.assertEqual(eng.installed_models(), ["qwen2.5:3b", "llama3.2:latest"])
            self.assertTrue(eng.has_model("qwen2.5:3b"))
            self.assertTrue(eng.has_model("llama3.2"))      # matches :latest by base
            self.assertFalse(eng.has_model("mistral"))

    def test_ensure_skips_pull_when_present(self):
        eng = OllamaEngine()
        with patch.object(OllamaEngine, "has_model", return_value=True), \
             patch.object(OllamaEngine, "pull") as pull:
            receipt = eng.ensure("qwen2.5:3b")
        pull.assert_not_called()
        self.assertTrue(receipt["was_present"])

    def test_ensure_pulls_when_missing(self):
        eng = OllamaEngine()
        with patch.object(OllamaEngine, "has_model", return_value=False), \
             patch.object(OllamaEngine, "pull", return_value="success") as pull:
            receipt = eng.ensure()
        pull.assert_called_once_with("qwen2.5:3b")
        self.assertEqual((receipt["was_present"], receipt["status"]), (False, "success"))

    def test_pull_streams_progress_and_returns_final_status(self):
        lines = [json.dumps({"status": "pulling manifest"}).encode(),
                 b"", json.dumps({"status": "success"}).encode()]
        resp = MagicMock()
        resp.__iter__ = lambda s: iter(lines)
        resp.__enter__ = lambda s: s; resp.__exit__ = lambda s, *a: False
        with patch("urllib.request.urlopen", return_value=resp) as urlopen:
            status = OllamaEngine().pull("qwen2.5:3b")
        self.assertEqual(status, "success")
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["name"], "qwen2.5:3b")

    def test_complete_posts_chat_and_records_provider_tokens(self):
        from yggdrasil.loki.usage import METER

        payload = json.dumps({
            "message": {"role": "assistant", "content": "ollama reply"},
            "prompt_eval_count": 11, "eval_count": 4,
        }).encode()
        resp = MagicMock()
        resp.read.return_value = payload
        resp.__enter__ = lambda s: s; resp.__exit__ = lambda s, *a: False
        METER.reset()
        with patch("urllib.request.urlopen", return_value=resp) as urlopen:
            out = OllamaEngine(model="llama3.2").generate("hi", system="sys")
        self.assertEqual(out, "ollama reply")
        sent = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "sys"})
        self.assertFalse(sent["stream"])
        row = METER.rows_for("ollama")[0]
        self.assertEqual((row.input_tokens, row.output_tokens), (11, 4))
        self.assertEqual(METER.total_cost, 0.0)  # local is free

    @unittest.skipUnless(
        os.getenv("YGG_LOKI_TEST_LOCAL") == "1",
        "set YGG_LOKI_TEST_LOCAL=1 (and run a local ollama) for the real test",
    )
    def test_real_local_server(self):
        eng = OllamaEngine()
        if not eng.available():
            self.skipTest("no local ollama server reachable")
        self.assertTrue(eng.generate("Reply with one word: ping").strip())


class TestResourceAwareSelection(unittest.TestCase):
    """``Loki.select`` weighs task complexity against workstation resources."""

    @staticmethod
    def _eng(name, local, available=True):
        e = MagicMock(spec=["name", "local", "available", "resolve_model"])
        e.name = name
        e.local = local
        e.available.return_value = available
        e.resolve_model.return_value = f"{name}-deep"
        return e

    def _select(self, engines, *, cpu, ram_gb, gpu=False, **kw):
        loki = Loki()
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: gpu)
        sysconf = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": int(ram_gb * 1e9 / 4096)}
        with patch.object(Loki, "_engine_instances", return_value=engines), \
             patch.dict(sys.modules, {"torch": fake_torch}), \
             patch("os.cpu_count", return_value=cpu), \
             patch("os.sysconf", side_effect=lambda n: sysconf[n]):
            return loki.select(**kw)

    def test_simple_on_capable_box_goes_local(self):
        engines = {"claude": self._eng("claude", False),
                   "ollama": self._eng("ollama", True)}
        chosen = self._select(engines, cpu=8, ram_gb=16, text="hi there")
        self.assertEqual(chosen.name, "ollama")

    def test_complex_goes_remote_even_on_capable_box(self):
        engines = {"claude": self._eng("claude", False),
                   "ollama": self._eng("ollama", True)}
        chosen = self._select(engines, cpu=8, ram_gb=16, text="refactor the planner")
        self.assertEqual(chosen.name, "claude")

    def test_forced_deep_tier_goes_remote(self):
        engines = {"claude": self._eng("claude", False),
                   "ollama": self._eng("ollama", True)}
        chosen = self._select(engines, cpu=8, ram_gb=16, text="hi", tier="deep")
        self.assertEqual(chosen.name, "claude")

    def test_thin_box_goes_remote_for_simple_work(self):
        engines = {"claude": self._eng("claude", False),
                   "ollama": self._eng("ollama", True)}
        chosen = self._select(engines, cpu=2, ram_gb=2, text="hi")
        self.assertEqual(chosen.name, "claude")

    def test_no_remote_falls_back_to_local(self):
        engines = {"claude": self._eng("claude", False, available=False),
                   "ollama": self._eng("ollama", True)}
        chosen = self._select(engines, cpu=2, ram_gb=2, text="refactor everything")
        self.assertEqual(chosen.name, "ollama")

    def test_nothing_available_returns_none(self):
        engines = {"claude": self._eng("claude", False, available=False)}
        self.assertIsNone(self._select(engines, cpu=8, ram_gb=16, text="hi"))

    def test_heavy_escalation_to_remote_asks_confirm(self):
        engines = {"claude": self._eng("claude", False),
                   "ollama": self._eng("ollama", True)}
        asked = {}

        def confirm(engine, model):
            asked["engine"] = engine.name
            asked["model"] = model
            return True

        chosen = self._select(engines, cpu=8, ram_gb=16,
                              text="refactor the planner", base="claude", confirm=confirm)
        self.assertEqual(chosen.name, "claude")          # escalated, with consent
        self.assertEqual(asked["engine"], "claude")

    def test_declined_escalation_stays_local(self):
        engines = {"claude": self._eng("claude", False),
                   "ollama": self._eng("ollama", True)}
        chosen = self._select(engines, cpu=8, ram_gb=16,
                              text="refactor the planner", base="claude",
                              confirm=lambda e, m: False)
        self.assertEqual(chosen.name, "ollama")          # declined → kept free/local

    def test_no_confirm_prompt_when_base_is_already_remote_only(self):
        # No local engine at all → no "switch to remote" to confirm.
        engines = {"claude": self._eng("claude", False)}
        confirm = MagicMock()
        chosen = self._select(engines, cpu=2, ram_gb=2,
                              text="refactor everything", base="claude", confirm=confirm)
        self.assertEqual(chosen.name, "claude")
        confirm.assert_not_called()

    def test_base_sticks_for_light_work_without_local(self):
        engines = {"claude": self._eng("claude", False),
                   "openai": self._eng("openai", False)}
        chosen = self._select(engines, cpu=2, ram_gb=2, text="hi", base="openai")
        self.assertEqual(chosen.name, "openai")          # honored the session base


class TestBootstrapLocal(unittest.TestCase):
    """`Loki.bootstrap_local` readies a free local engine, lazily installing."""

    def test_uses_ollama_and_ensures_bootstrap_model(self):
        loki = Loki()
        fake = MagicMock()
        fake.available.return_value = True
        fake.bootstrap_model = "qwen2.5:3b"
        fake.ensure.return_value = {"model": "qwen2.5:3b", "was_present": False,
                                    "status": "success"}
        with patch("yggdrasil.loki.engines.OllamaEngine", return_value=fake):
            res = loki.bootstrap_local()
        self.assertEqual((res["engine"], res["ready"], res["model"]),
                         ("ollama", True, "qwen2.5:3b"))
        fake.ensure.assert_called_once_with("qwen2.5:3b")

    def test_falls_back_to_transformers(self):
        loki = Loki()
        oll = MagicMock(); oll.available.return_value = False
        tf = MagicMock(); tf.available.return_value = True; tf.bootstrap_model = "Qwen/Qwen2.5-1.5B-Instruct"
        with patch("yggdrasil.loki.engines.OllamaEngine", return_value=oll), \
             patch("yggdrasil.loki.engines.TransformersEngine", return_value=tf):
            res = loki.bootstrap_local()
        self.assertEqual((res["engine"], res["ready"]), ("transformers", True))

    def test_reports_install_when_no_local_engine(self):
        loki = Loki()
        oll = MagicMock(); oll.available.return_value = False
        tf = MagicMock(); tf.available.return_value = False
        with patch("yggdrasil.loki.engines.OllamaEngine", return_value=oll), \
             patch("yggdrasil.loki.engines.TransformersEngine", return_value=tf):
            res = loki.bootstrap_local()
        self.assertFalse(res["ready"])
        self.assertTrue(res["install"])


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
