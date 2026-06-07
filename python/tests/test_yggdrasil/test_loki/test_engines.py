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
    OpenVINOEngine,
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
        oai = MagicMock()
        msg = MagicMock(); msg.content = "served"
        oai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)], model="ep",
        )
        eng = DatabricksServingEngine(client=MagicMock(), endpoint="ep")
        # _oai_client() auto-installs the openai dep then returns the SDK's
        # OpenAI-compatible client — patch it so the test needs neither.
        with patch.object(DatabricksServingEngine, "_oai_client", return_value=oai):
            self.assertEqual(eng.generate("hi"), "served")
        self.assertEqual(oai.chat.completions.create.call_args.kwargs["model"], "ep")

    def test_oai_client_auto_installs_openai(self):
        client = MagicMock()
        eng = DatabricksServingEngine(client=client)
        with patch("yggdrasil.loki.runtime.load") as load:
            eng._oai_client()
        # The openai dep is auto-installed via the databricks-sdk extra.
        load.assert_called_once_with("openai", "databricks-sdk[openai]")
        client.workspace_client.return_value.serving_endpoints.get_open_ai_client.assert_called_once()


class TestTransformersEngine(unittest.TestCase):
    """The local HuggingFace engine — open models on this workstation, free."""

    def setUp(self):
        TransformersEngine._PIPES.clear()   # don't leak a pipeline across tests
        TransformersEngine._FAILED.clear()  # nor a remembered load failure

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

    def test_resolve_device_pin_wins_over_autodetect(self):
        # An explicit ctor pin or YGG_LOKI_HF_DEVICE short-circuits detection.
        self.assertEqual(TransformersEngine(device="cuda:1").resolve_device(), "cuda:1")
        with patch.dict(os.environ, {"YGG_LOKI_HF_DEVICE": "xpu"}):
            self.assertEqual(TransformersEngine().resolve_device(), "xpu")

    def test_resolve_device_auto_detects_intel_gpu(self):
        from yggdrasil.loki import resources

        # No pin → the engine lands the model on the detected accelerator
        # (Intel GPU here), or CPU (None) when there's none.
        with patch.object(resources, "accelerator", return_value="xpu"):
            self.assertEqual(TransformersEngine().resolve_device(), "xpu")
        with patch.object(resources, "accelerator", return_value=None):
            self.assertIsNone(TransformersEngine().resolve_device())

    def test_model_sizes_to_resources_not_prompt_tier(self):
        from yggdrasil.loki import resources

        eng = TransformersEngine()
        # Local models are bounded by the box → sized to resources, not the
        # remote fast/deep cost tier (which is ignored here).
        with patch.object(resources, "snapshot", return_value={"cpu": 8, "ram_gb": 12, "gpu": False}):
            self.assertEqual(eng.resolve_model(tier="deep"), "Qwen/Qwen2.5-1.5B-Instruct")
        with patch.object(resources, "snapshot", return_value={"cpu": 16, "ram_gb": 64, "gpu": False}):
            self.assertEqual(eng.resolve_model(tier="fast"), "Qwen/Qwen2.5-7B-Instruct")
        with patch.object(resources, "snapshot", return_value={"cpu": 16, "ram_gb": 64, "gpu": True}):
            self.assertEqual(eng.bootstrap_model, "Qwen/Qwen2.5-14B-Instruct")

    def test_explicit_model_pin_overrides_resource_sizing(self):
        eng = TransformersEngine(model="some/custom-model")
        self.assertEqual(eng.resolve_model(), "some/custom-model")

    def test_pipeline_loads_on_detected_intel_gpu(self):
        # No device pin → the build targets the auto-detected Intel GPU.
        from yggdrasil.loki import resources

        pipe = MagicMock(return_value=[{"generated_text": [
            {"role": "assistant", "content": "ok"}]}])
        fake = types.ModuleType("transformers")
        fake.pipeline = MagicMock(return_value=pipe)
        fake_torch = types.ModuleType("torch")
        TransformersEngine._PIPES.clear()
        TransformersEngine._FAILED.clear()
        eng = TransformersEngine(model="m")
        with patch.object(resources, "accelerator", return_value="xpu"), \
                patch.dict(sys.modules, {"transformers": fake, "torch": fake_torch}):
            eng.generate("hi")
        self.assertEqual(fake.pipeline.call_args.kwargs["device"], "xpu")

    def test_pipeline_disables_xet_and_quiets_transformers(self):
        # The corporate-proxy 403 came from HuggingFace's xet transfer; the build
        # forces the classic LFS download and pins transformers to errors so a
        # failure is one line, not nested framework tracebacks.
        pipe = MagicMock(return_value=[{"generated_text": [
            {"role": "assistant", "content": "ok"}]}])
        fake = types.ModuleType("transformers")
        fake.pipeline = MagicMock(return_value=pipe)
        fake.logging = types.SimpleNamespace(set_verbosity_error=MagicMock())
        fake_torch = types.ModuleType("torch")
        TransformersEngine._PIPES.clear()
        TransformersEngine._FAILED.clear()
        with patch.dict(os.environ, {}, clear=False), \
                patch.dict(sys.modules, {"transformers": fake, "torch": fake_torch}):
            os.environ.pop("HF_HUB_DISABLE_XET", None)
            TransformersEngine(model="m").generate("hi")
            self.assertEqual(os.environ.get("HF_HUB_DISABLE_XET"), "1")
        fake.logging.set_verbosity_error.assert_called_once()

    def test_local_engine_never_charges_cost(self):
        # Local models run on this box → free. The meter records tokens but the
        # USD cost (and budget) must not move.
        from yggdrasil.loki.usage import METER

        pipe = MagicMock(return_value=[{"generated_text": [
            {"role": "assistant", "content": "a local reply"}]}])
        fake = types.ModuleType("transformers")
        fake.pipeline = MagicMock(return_value=pipe)
        fake_torch = types.ModuleType("torch")
        TransformersEngine._PIPES.clear()
        TransformersEngine._FAILED.clear()
        METER.reset()
        with patch.dict(sys.modules, {"transformers": fake, "torch": fake_torch}):
            TransformersEngine(model="m").generate("hi there")
        self.assertGreater(METER.total_tokens, 0)   # tokens are still counted
        self.assertEqual(METER.total_cost, 0.0)      # but cost stays zero

    def test_brief_collapses_nested_traceback(self):
        from yggdrasil.loki.engines.transformers_engine import _brief

        huge = "Could not load model X\n" + ("Traceback line\n" * 500)
        out = _brief(huge)
        self.assertLessEqual(len(out), 200)
        self.assertNotIn("\n", out)
        self.assertTrue(out.startswith("Could not load model X"))

    def test_complete_runs_pipeline_and_records_usage(self):
        from yggdrasil.loki.usage import METER

        calls = {}

        def pipe(chat, **kw):
            calls["chat"] = chat
            calls["kw"] = kw
            return [{"generated_text": [{"role": "assistant", "content": "local reply"}]}]

        fake = types.ModuleType("transformers")
        fake.pipeline = MagicMock(return_value=pipe)
        fake_torch = types.ModuleType("torch")   # the pipeline backend (loaded first)
        METER.reset()
        with patch.dict(sys.modules, {"transformers": fake, "torch": fake_torch}):
            out = TransformersEngine(model="m").generate("hi", system="sys")
        self.assertEqual(out, "local reply")
        # System prompt leads the chat; new-token cap kept small for CPU.
        self.assertEqual(calls["chat"][0], {"role": "system", "content": "sys"})
        self.assertEqual(calls["kw"]["max_new_tokens"], 512)
        self.assertEqual(METER.rows_for("transformers")[0].calls, 1)
        self.assertEqual(METER.total_cost, 0.0)  # local is free

    @staticmethod
    def _fake_hub():
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = MagicMock(return_value="/cache/m")
        return hub

    def test_failed_load_is_cached_not_retried_and_surfaces_cause(self):
        # A local load is slow and can fail late (corrupt download, torch
        # mismatch). It tries once, repairs the cache and retries once, then
        # gives up — and a remembered failure fast-fails the *next* turn instead
        # of re-downloading weights on every chat.
        fake = types.ModuleType("transformers")
        fake.pipeline = MagicMock(side_effect=ValueError("Could not load model X"))
        fake_torch = types.ModuleType("torch")
        hub = self._fake_hub()
        eng = TransformersEngine(model="m")
        with patch.dict(sys.modules, {"transformers": fake, "torch": fake_torch,
                                      "huggingface_hub": hub}):
            with self.assertRaises(RuntimeError) as first:
                eng.generate("hi")
            with self.assertRaises(RuntimeError) as second:
                eng.generate("hi again")
        # First turn: initial build + one repair retry. Second turn: neither —
        # the remembered failure fast-fails without touching the network.
        self.assertEqual(fake.pipeline.call_count, 2)
        hub.snapshot_download.assert_called_once_with("m", force_download=True)
        # The surfaced error names the model and the underlying cause.
        self.assertIn("m", str(first.exception))
        self.assertIn("Could not load model X", str(first.exception))
        self.assertIs(first.exception, second.exception)

    def test_gpu_failure_falls_back_to_cpu_without_redownload(self):
        # The reported bug: a model on a GPU that torch can't actually drive
        # failed, and the old code force-re-downloaded the whole model *every
        # run*. Now a device error falls back to CPU and keeps the cached weights.
        from yggdrasil.loki import resources

        pipe = MagicMock(return_value=[{"generated_text": [
            {"role": "assistant", "content": "ok"}]}])
        fake = types.ModuleType("transformers")
        fake.pipeline = MagicMock(side_effect=[RuntimeError("XPU backend not available"), pipe])
        fake_torch = types.ModuleType("torch")
        hub = self._fake_hub()
        TransformersEngine._PIPES.clear()
        TransformersEngine._FAILED.clear()
        eng = TransformersEngine(model="m")
        with patch.object(resources, "accelerator", return_value="xpu"), \
                patch.dict(sys.modules, {"transformers": fake, "torch": fake_torch,
                                         "huggingface_hub": hub}):
            out = eng.generate("hi")
        self.assertEqual(out, "ok")
        # First call targets the GPU, second falls back to CPU (device=None) …
        self.assertEqual(fake.pipeline.call_args_list[0].kwargs["device"], "xpu")
        self.assertIsNone(fake.pipeline.call_args_list[1].kwargs["device"])
        # … and crucially the weights are NOT re-downloaded.
        hub.snapshot_download.assert_not_called()

    def test_device_error_is_not_mistaken_for_corruption(self):
        from yggdrasil.loki.engines.transformers_engine import _looks_corrupt

        self.assertFalse(_looks_corrupt(RuntimeError("XPU backend not available")))
        self.assertFalse(_looks_corrupt(RuntimeError("CUDA out of memory")))
        self.assertTrue(_looks_corrupt(OSError("truncated safetensors")))
        self.assertTrue(_looks_corrupt(ValueError("Could not load model X")))

    def test_partial_download_is_repaired_and_retried(self):
        # The Windows failure mode: a truncated first download, then a clean
        # re-fetch loads. The engine repairs and recovers without the user
        # clearing the cache by hand.
        pipe = MagicMock(return_value=[{"generated_text": [
            {"role": "assistant", "content": "ok"}]}])
        fake = types.ModuleType("transformers")
        fake.pipeline = MagicMock(side_effect=[OSError("truncated safetensors"), pipe])
        fake_torch = types.ModuleType("torch")
        hub = self._fake_hub()
        eng = TransformersEngine(model="m")
        with patch.dict(sys.modules, {"transformers": fake, "torch": fake_torch,
                                      "huggingface_hub": hub}):
            out = eng.generate("hi")
        self.assertEqual(out, "ok")
        hub.snapshot_download.assert_called_once_with("m", force_download=True)
        self.assertTrue(eng.ready("m"))   # repaired pipeline is now cached

    def test_warm_preloads_pipeline_and_swallows_failure(self):
        pipe = MagicMock()
        fake = types.ModuleType("transformers")
        fake.pipeline = MagicMock(return_value=pipe)
        fake_torch = types.ModuleType("torch")
        eng = TransformersEngine(model="m")
        with patch.dict(sys.modules, {"transformers": fake, "torch": fake_torch}):
            eng.warm()
            self.assertTrue(eng.ready("m"))   # built ahead of any turn

        # A failing warm is best-effort — it must not raise (the first real
        # turn surfaces the remembered failure instead).
        TransformersEngine._PIPES.clear()
        TransformersEngine._FAILED.clear()
        fake.pipeline = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict(sys.modules, {"transformers": fake, "torch": fake_torch}):
            eng.warm()                        # does not raise
        self.assertIn("m", TransformersEngine._FAILED)

    def test_ready_tracks_loaded_pipeline(self):
        eng = TransformersEngine(model="m")
        self.assertFalse(eng.ready())          # nothing loaded yet
        TransformersEngine._PIPES["m"] = object()
        self.assertTrue(eng.ready())
        self.assertTrue(eng.ready("m"))
        self.assertFalse(eng.ready("other"))

    def test_stream_yields_tokens_live_and_records_usage(self):
        from yggdrasil.loki.usage import METER

        class _FakeStreamer:
            def __init__(self, tokenizer, **kw):
                self.kw = kw

            def __iter__(self):
                return iter(["Hel", "lo!"])

        pipe = MagicMock()
        pipe.tokenizer = object()
        fake = types.ModuleType("transformers")
        fake.pipeline = MagicMock(return_value=pipe)
        fake.TextIteratorStreamer = _FakeStreamer
        fake_torch = types.ModuleType("torch")
        METER.reset()
        with patch.dict(sys.modules, {"transformers": fake, "torch": fake_torch}):
            chunks = list(TransformersEngine(model="m").generate_stream("hi"))
        # Tokens arrive incrementally (live), not as one final blob.
        self.assertEqual(chunks, ["Hel", "lo!"])
        # The generation ran on a worker thread the streamer drained.
        pipe.assert_called_once()
        self.assertEqual(pipe.call_args.kwargs["streamer"].__class__.__name__, "_FakeStreamer")
        self.assertEqual(METER.rows_for("transformers")[0].calls, 1)

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

    @staticmethod
    def _fake_session(*, get_json=None, get_status=200, post_json=None):
        """A stand-in HTTPSession whose get/post return canned HTTPResponses."""
        sess = MagicMock()
        sess.get.return_value = MagicMock(status_code=get_status,
                                          json=MagicMock(return_value=get_json or {}))
        sess.post.return_value = MagicMock(json=MagicMock(return_value=post_json or {}))
        return sess

    def test_available_pings_tags_via_httpsession(self):
        # Fresh instances: a 200 from /api/tags is up, a refused connection down.
        with patch.object(OllamaEngine, "_session", return_value=self._fake_session(get_status=200)):
            self.assertTrue(OllamaEngine().available())
        sess = MagicMock(); sess.get.side_effect = OSError("refused")
        with patch.object(OllamaEngine, "_session", return_value=sess):
            self.assertFalse(OllamaEngine().available())

    def test_available_probe_is_memoized(self):
        # The liveness probe is a network round-trip asked several times per
        # `ygg loki` command (engine() + engines() + select()), so it's cached
        # for a short TTL — repeated checks hit the cache, not the network.
        eng = OllamaEngine()
        sess = self._fake_session(get_status=200)
        with patch.object(OllamaEngine, "_session", return_value=sess):
            self.assertTrue(eng.available())
            self.assertTrue(eng.available())
            self.assertTrue(eng.available())
        self.assertEqual(sess.get.call_count, 1)  # one probe, then cached

    def test_bootstrap_model_scales_with_resources(self):
        from yggdrasil.loki import resources

        eng = OllamaEngine()
        with patch.object(resources, "snapshot", return_value={"cpu": 8, "ram_gb": 12, "gpu": False}):
            self.assertEqual(eng.bootstrap_model, "qwen2.5:3b")    # modest box → small
        with patch.object(resources, "snapshot", return_value={"cpu": 16, "ram_gb": 64, "gpu": False}):
            self.assertEqual(eng.bootstrap_model, "qwen2.5:14b")   # big box → larger
        with patch.object(resources, "snapshot", return_value={"cpu": 16, "ram_gb": 64, "gpu": True}):
            self.assertEqual(eng.bootstrap_model, "qwen2.5:32b")   # GPU → xlarge

    def test_installed_models_and_has_model(self):
        sess = self._fake_session(get_json={"models": [{"name": "qwen2.5:3b"},
                                                        {"name": "llama3.2:latest"}]})
        with patch.object(OllamaEngine, "_session", return_value=sess):
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
        from yggdrasil.loki import resources

        eng = OllamaEngine()
        with patch.object(resources, "snapshot", return_value={"cpu": 8, "ram_gb": 12, "gpu": False}), \
             patch.object(OllamaEngine, "has_model", return_value=False), \
             patch.object(OllamaEngine, "pull", return_value="success") as pull:
            receipt = eng.ensure()
        pull.assert_called_once_with("qwen2.5:3b", on_progress=None)   # default = resource-sized model
        self.assertEqual((receipt["was_present"], receipt["status"]), (False, "success"))

    def test_pull_posts_and_returns_final_status(self):
        sess = self._fake_session(post_json={"status": "success"})
        with patch.object(OllamaEngine, "_session", return_value=sess):
            status = OllamaEngine().pull("qwen2.5:3b")
        self.assertEqual(status, "success")
        # The pull rides HTTPSession.post with the non-streaming body.
        kwargs = sess.post.call_args.kwargs
        self.assertEqual(kwargs["json"], {"name": "qwen2.5:3b", "stream": False})

    def test_streaming_pull_reports_progress_and_returns_status(self):
        # With an on_progress callback the pull streams Ollama's NDJSON events;
        # each tick is reported and the final status is returned.
        events = [
            b'{"status":"pulling manifest"}\n',
            b'{"status":"downloading","total":100,"completed":40}\n{"status":"down',
            b'loading","total":100,"completed":100}\n',
            b'{"status":"success"}\n',
        ]
        resp = MagicMock()
        resp.stream = MagicMock(return_value=iter(events))
        sess = MagicMock()
        sess.post.return_value = resp
        ticks = []
        with patch.object(OllamaEngine, "_session", return_value=sess):
            status = OllamaEngine().pull("qwen2.5:3b", on_progress=ticks.append)
        self.assertEqual(status, "success")
        # The request opted into streaming (body un-preloaded) …
        kwargs = sess.post.call_args.kwargs
        self.assertEqual(kwargs["json"], {"name": "qwen2.5:3b", "stream": True})
        self.assertEqual(kwargs["send_config"], {"stream": True})
        # … and every event surfaced, including the one split across two chunks.
        self.assertEqual([e.get("completed") for e in ticks if e.get("completed")], [40, 100])
        self.assertEqual(ticks[-1]["status"], "success")

    def test_streaming_pull_raises_on_error_event(self):
        resp = MagicMock()
        resp.stream = MagicMock(return_value=iter([b'{"error":"manifest not found"}\n']))
        sess = MagicMock()
        sess.post.return_value = resp
        with patch.object(OllamaEngine, "_session", return_value=sess):
            with self.assertRaises(RuntimeError) as ctx:
                OllamaEngine().pull("nope", on_progress=lambda e: None)
        self.assertIn("manifest not found", str(ctx.exception))

    def test_iter_ndjson_buffers_across_chunks_and_skips_junk(self):
        from yggdrasil.loki.engines.ollama_engine import _iter_ndjson

        resp = MagicMock()
        resp.stream = MagicMock(return_value=iter([
            b'{"a":1}\n{"b":', b'2}\n\n', b'not json\n', b'{"c":3}',
        ]))
        out = list(_iter_ndjson(resp))
        self.assertEqual(out, [{"a": 1}, {"b": 2}, {"c": 3}])

    def test_complete_posts_chat_and_records_provider_tokens(self):
        from yggdrasil.loki.usage import METER

        sess = self._fake_session(post_json={
            "message": {"role": "assistant", "content": "ollama reply"},
            "prompt_eval_count": 11, "eval_count": 4,
        })
        METER.reset()
        with patch.object(OllamaEngine, "_session", return_value=sess):
            out = OllamaEngine(model="llama3.2").generate("hi", system="sys")
        self.assertEqual(out, "ollama reply")
        sent = sess.post.call_args.kwargs["json"]
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

    def test_remote_base_demotes_to_local_for_light_work(self):
        # The session is anchored on a remote (claude), but a light task on a
        # capable box drops down to the free local model (remote → local).
        engines = {"claude": self._eng("claude", False),
                   "ollama": self._eng("ollama", True)}
        chosen = self._select(engines, cpu=8, ram_gb=16, text="hi", base="claude")
        self.assertEqual(chosen.name, "ollama")

    def test_local_base_escalates_to_remote_for_heavy_work(self):
        # Anchored on a local model, a heavy task climbs to the remote (local →
        # remote), asked via confirm.
        engines = {"claude": self._eng("claude", False),
                   "ollama": self._eng("ollama", True)}
        chosen = self._select(engines, cpu=8, ram_gb=16,
                              text="refactor and debug the planner",
                              base="ollama", confirm=lambda e, m: True)
        self.assertEqual(chosen.name, "claude")

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
        fake.ensure.assert_called_once_with("qwen2.5:3b", on_progress=None)

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

    def test_available_engines_probes_in_parallel_and_filters(self):
        import threading

        loki = Loki()
        seen: list[str] = []
        barrier = threading.Barrier(2, timeout=5)

        def mk(name, ok, *, blocks=False):
            e = MagicMock(); e.name = name

            def available():
                seen.append(name)
                if blocks:            # both blockers must be in-flight at once
                    barrier.wait()    # → only passes if probed concurrently
                return ok

            e.available.side_effect = available
            return e

        engines = {"claude": mk("claude", True, blocks=True),
                   "ollama": mk("ollama", True, blocks=True),
                   "openai": mk("openai", False)}
        with patch.object(Loki, "_engine_instances", return_value=engines):
            avail = loki.available_engines()
        # Only the reachable engines come back, keyed by name.
        self.assertEqual(set(avail), {"claude", "ollama"})
        self.assertEqual(set(seen), {"claude", "ollama", "openai"})

    def test_available_engines_guards_a_raising_probe(self):
        # A probe that raises must not sink the whole parallel sweep.
        loki = Loki()
        boom = MagicMock(); boom.name = "boom"; boom.available.side_effect = OSError("x")
        good = MagicMock(); good.name = "good"; good.available.return_value = True
        with patch.object(Loki, "_engine_instances",
                          return_value={"boom": boom, "good": good}):
            self.assertEqual(set(loki.available_engines()), {"good"})


class TestOpenVINOEngine(unittest.TestCase):
    """The local OpenVINO engine — a model on the Intel NPU (else GPU/CPU)."""

    def setUp(self):
        OpenVINOEngine._PIPES.clear()
        OpenVINOEngine._FAILED.clear()

    @staticmethod
    def _stub_pipe():
        return MagicMock(return_value=[{"generated_text": [
            {"role": "assistant", "content": "ov reply"}]}])

    def _fakes(self, *, devices=("NPU", "CPU"), from_pretrained=None, pipe=None):
        ov = types.ModuleType("openvino")
        ov.Core = lambda: types.SimpleNamespace(available_devices=list(devices))
        optimum_parent = types.ModuleType("optimum")
        optimum = types.ModuleType("optimum.intel")
        optimum.OVModelForCausalLM = types.SimpleNamespace(
            from_pretrained=from_pretrained or MagicMock(return_value="ovmodel"))
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=MagicMock(return_value="tok"))
        transformers.logging = types.SimpleNamespace(set_verbosity_error=MagicMock())
        transformers.pipeline = MagicMock(return_value=pipe or self._stub_pipe())
        return {"openvino": ov, "optimum": optimum_parent, "optimum.intel": optimum,
                "transformers": transformers}

    def test_is_local(self):
        self.assertTrue(OpenVINOEngine.local)
        self.assertEqual(OpenVINOEngine.name, "openvino")

    def test_available_needs_packages_and_an_accelerator(self):
        eng = OpenVINOEngine()
        with patch("importlib.util.find_spec", return_value=object()), \
                patch.object(eng, "_devices", return_value=["NPU", "CPU"]):
            self.assertTrue(eng.available())                 # NPU present → offer it
        with patch("importlib.util.find_spec", return_value=object()), \
                patch.object(eng, "_devices", return_value=["GPU.0", "CPU"]):
            self.assertTrue(eng.available())                 # an Intel GPU counts too
        with patch("importlib.util.find_spec", return_value=object()), \
                patch.object(eng, "_devices", return_value=["CPU"]):
            self.assertFalse(eng.available())                # CPU-only → leave to others
        with patch("importlib.util.find_spec", return_value=None), \
                patch.object(eng, "_devices", return_value=["NPU"]):
            self.assertFalse(eng.available())                # packages missing

    def test_devices_lists_openvino_core_and_memoizes(self):
        ov = types.ModuleType("openvino")
        calls = []
        ov.Core = lambda: (calls.append(1), types.SimpleNamespace(
            available_devices=["NPU", "GPU", "CPU"]))[1]
        eng = OpenVINOEngine()
        with patch.dict(sys.modules, {"openvino": ov}):
            self.assertEqual(eng._devices(), ["NPU", "GPU", "CPU"])
            eng._devices()                                   # memoized — no second Core()
        self.assertEqual(len(calls), 1)

    def test_device_chain_prefers_npu_then_gpu_then_cpu(self):
        eng = OpenVINOEngine()
        with patch.object(eng, "_devices", return_value=["GPU.0", "CPU"]):
            self.assertEqual(eng._device_chain(), ["GPU", "CPU"])
        with patch.object(eng, "_devices", return_value=["NPU", "GPU.0", "CPU"]):
            self.assertEqual(eng._device_chain(), ["NPU", "GPU", "CPU"])
            self.assertEqual(eng.resolve_device(), "NPU")     # NPU is the whole point
        self.assertEqual(OpenVINOEngine(device="CPU")._device_chain(), ["CPU"])  # pin wins

    def test_pipeline_loads_on_npu_without_converting_an_ov_model(self):
        fakes = self._fakes(devices=("NPU", "CPU"))
        eng = OpenVINOEngine(model="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov")
        with patch.dict(sys.modules, fakes):
            eng.generate("hi")
        load = fakes["optimum.intel"].OVModelForCausalLM.from_pretrained
        self.assertEqual(load.call_args.kwargs["device"], "NPU")
        self.assertFalse(load.call_args.kwargs["export"])    # already OpenVINO IR

    def test_pipeline_converts_a_plain_hf_model(self):
        fakes = self._fakes(devices=("NPU", "CPU"))
        eng = OpenVINOEngine(model="Qwen/Qwen2.5-1.5B-Instruct")
        with patch.dict(sys.modules, fakes):
            eng.generate("hi")
        load = fakes["optimum.intel"].OVModelForCausalLM.from_pretrained
        self.assertTrue(load.call_args.kwargs["export"])     # convert to OV IR

    def test_pipeline_falls_back_npu_to_gpu_to_cpu(self):
        from_pretrained = MagicMock(side_effect=[
            RuntimeError("NPU compile failed"),
            RuntimeError("GPU out of memory"),
            "ovmodel",                                       # CPU succeeds
        ])
        fakes = self._fakes(devices=("NPU", "GPU", "CPU"), from_pretrained=from_pretrained)
        eng = OpenVINOEngine(model="OpenVINO/m-int4-ov")
        with patch.dict(sys.modules, fakes):
            out = eng.generate("hi")
        self.assertEqual(out, "ov reply")
        tried = [c.kwargs["device"] for c in from_pretrained.call_args_list]
        self.assertEqual(tried, ["NPU", "GPU", "CPU"])       # walked the chain
        fakes["transformers"].pipeline.assert_called_once()  # built once, on CPU

    def test_failed_load_is_remembered_not_retried(self):
        from_pretrained = MagicMock(side_effect=RuntimeError("boom"))
        fakes = self._fakes(devices=("NPU", "CPU"), from_pretrained=from_pretrained)
        eng = OpenVINOEngine(model="OpenVINO/m-int4-ov")
        with patch.dict(sys.modules, fakes):
            with self.assertRaises(RuntimeError):
                eng.generate("hi")
            calls_after_first = from_pretrained.call_count
            with self.assertRaises(RuntimeError):
                eng.generate("again")
        # The second turn fast-fails on the remembered error — no new load attempts.
        self.assertEqual(from_pretrained.call_count, calls_after_first)

    def test_complete_is_free(self):
        from yggdrasil.loki.usage import METER

        METER.reset()
        fakes = self._fakes(devices=("NPU", "CPU"))
        with patch.dict(sys.modules, fakes):
            out = OpenVINOEngine(model="OpenVINO/m-int4-ov").generate("hi", system="s")
        self.assertEqual(out, "ov reply")
        self.assertGreater(METER.total_tokens, 0)
        self.assertEqual(METER.total_cost, 0.0)              # local → free

    def test_env_overrides_model_and_device(self):
        with patch.dict(os.environ, {"YGG_LOKI_OV_MODEL": "my/ov-model",
                                     "YGG_LOKI_OV_DEVICE": "GPU"}):
            eng = OpenVINOEngine()
        self.assertEqual(eng.resolve_model(), "my/ov-model")
        self.assertEqual(eng.resolve_device(), "GPU")        # pin wins over autodetect


class TestResourceAccelerator(unittest.TestCase):
    """The accelerator probe spans CUDA, Intel GPU (xpu), and Apple mps; the
    Intel NPU is detected and reported separately."""

    @staticmethod
    def _torch(*, cuda=False, xpu=False, mps=False):
        t = types.ModuleType("torch")
        t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        if xpu:
            t.xpu = types.SimpleNamespace(is_available=lambda: True)
        t.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps))
        return t

    def test_detects_cuda_first(self):
        from yggdrasil.loki import resources

        with patch.dict(sys.modules, {"torch": self._torch(cuda=True, xpu=True)}):
            self.assertEqual(resources.accelerator(), "cuda")

    def test_detects_intel_gpu_xpu(self):
        from yggdrasil.loki import resources

        with patch.dict(sys.modules, {"torch": self._torch(xpu=True)}):
            self.assertEqual(resources.accelerator(), "xpu")

    def test_detects_apple_mps(self):
        from yggdrasil.loki import resources

        with patch.dict(sys.modules, {"torch": self._torch(mps=True)}):
            self.assertEqual(resources.accelerator(), "mps")

    def test_cpu_only_has_no_accelerator(self):
        from yggdrasil.loki import resources

        with patch.dict(sys.modules, {"torch": self._torch()}):
            self.assertIsNone(resources.accelerator())

    def test_snapshot_reports_accelerator_and_npu(self):
        from yggdrasil.loki import resources

        with patch.object(resources, "accelerator", return_value="xpu"), \
                patch.object(resources, "has_npu", return_value=True):
            snap = resources.snapshot()
        self.assertEqual(snap["accelerator"], "xpu")
        self.assertTrue(snap["npu"])
        self.assertFalse(snap["gpu"])  # gpu stays the CUDA-only xlarge driver

    def test_intel_gpu_enables_local_without_big_ram(self):
        from yggdrasil.loki import resources

        # An Intel GPU box can host a local model even on modest CPU/RAM.
        snap = {"cpu": 2, "ram_gb": 4, "gpu": False,
                "accelerator": "xpu", "npu": False}
        self.assertTrue(resources.can_run_local(snap))

    def test_has_npu_detects_intel_npu_via_openvino(self):
        from yggdrasil.loki import resources

        ov = types.ModuleType("openvino")
        ov.Core = lambda: types.SimpleNamespace(available_devices=["CPU", "GPU", "NPU"])
        with patch.dict(sys.modules, {"openvino": ov}):
            self.assertTrue(resources.has_npu())
        # No NPU listed → falls through to the OS probe (stubbed off here so the
        # result is deterministic on a host that actually has an NPU).
        ov.Core = lambda: types.SimpleNamespace(available_devices=["CPU", "GPU"])
        with patch.dict(sys.modules, {"openvino": ov}), \
                patch.object(resources, "_os_has_npu", return_value=False):
            self.assertFalse(resources.has_npu())

    def test_has_npu_falls_back_to_os_probe_without_openvino(self):
        from yggdrasil.loki import resources

        # No OpenVINO, but the OS reports the intel_vpu accel device → True.
        with patch.dict(sys.modules, {"openvino": None}), \
                patch.object(resources, "_os_has_npu", return_value=True):
            self.assertTrue(resources.has_npu())
        with patch.dict(sys.modules, {"openvino": None}), \
                patch.object(resources, "_os_has_npu", return_value=False):
            self.assertFalse(resources.has_npu())

    def test_os_has_npu_matches_intel_vpu_accel_device(self):
        from yggdrasil.loki import resources

        # A /sys/class/accel/accel0 whose driver is intel_vpu → NPU present.
        with patch.object(resources.glob, "glob",
                          side_effect=lambda p: ["/sys/class/accel/accel0"] if "accel" in p else []), \
                patch.object(resources.os, "readlink", return_value="/.../bus/pci/drivers/intel_vpu"):
            self.assertTrue(resources._os_has_npu())
        # Nothing present anywhere → False.
        with patch.object(resources.glob, "glob", return_value=[]):
            self.assertFalse(resources._os_has_npu())

    def test_intel_gpu_present_reads_drm_vendor(self):
        from unittest.mock import mock_open

        from yggdrasil.loki import resources

        with patch.object(resources.glob, "glob",
                          return_value=["/sys/class/drm/card0/device/vendor"]), \
                patch("builtins.open", mock_open(read_data="0x8086\n")):
            self.assertTrue(resources.intel_gpu_present())
        # A non-Intel vendor id (NVIDIA 0x10de) → not an Intel GPU.
        with patch.object(resources.glob, "glob",
                          return_value=["/sys/class/drm/card0/device/vendor"]), \
                patch("builtins.open", mock_open(read_data="0x10de\n")):
            self.assertFalse(resources.intel_gpu_present())

    def test_snapshot_reports_present_intel_gpu_even_without_torch(self):
        from yggdrasil.loki import resources

        # torch can't drive it (accelerator None) but the OS sees the iGPU.
        with patch.object(resources, "accelerator", return_value=None), \
                patch.object(resources, "intel_gpu_present", return_value=True), \
                patch.object(resources, "has_npu", return_value=False):
            snap = resources.snapshot()
        self.assertTrue(snap["intel_gpu"])
        self.assertIsNone(snap["accelerator"])
        self.assertFalse(snap["gpu"])


if __name__ == "__main__":
    unittest.main()
