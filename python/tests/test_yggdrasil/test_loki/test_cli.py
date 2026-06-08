"""Tests for ``ygg loki`` CLI helpers."""
from __future__ import annotations

import json
import unittest

from yggdrasil.loki import cli


class TestJsonable(unittest.TestCase):
    """``--json`` must survive whatever a skill returns (frames included)."""

    def test_passthrough_basics(self):
        self.assertEqual(cli._jsonable({"a": 1, "b": [True, None, "x"]}),
                         {"a": 1, "b": [True, None, "x"]})

    def test_polars_frame_becomes_records(self):
        try:
            import polars as pl
        except Exception:
            self.skipTest("polars not installed")
        df = pl.DataFrame({"n": [1, 2], "city": ["Paris", "Tokyo"]})
        out = cli._jsonable({"rows": df, "row_count": 2})
        self.assertEqual(out["rows"], [{"n": 1, "city": "Paris"}, {"n": 2, "city": "Tokyo"}])
        # And the whole thing is now orjson-encodable.
        self.assertIn("Paris", cli._json({"rows": df}))

    def test_unknown_object_falls_back_to_str(self):
        class Weird:
            def __str__(self):
                return "weird!"

        self.assertEqual(cli._jsonable({"x": Weird()}), {"x": "weird!"})

    def test_model_like_to_dict_is_used(self):
        class Model:
            def to_dict(self):
                return {"k": "v"}

        self.assertEqual(cli._jsonable(Model()), {"k": "v"})
        # End-to-end: valid JSON out.
        self.assertEqual(json.loads(cli._json(Model())), {"k": "v"})


class TestUsageTitle(unittest.TestCase):
    """The live KPIs that ride the terminal title bar instead of a per-turn line."""

    def setUp(self):
        from yggdrasil.loki.usage import METER
        self.METER = METER
        METER.reset()
        METER.set_limit(None)

    def tearDown(self):
        self.METER.reset()
        self.METER.set_limit(None)

    def test_title_summarizes_tokens_and_cost(self):
        self.METER.record("transformers", "qwen", 100, 50)
        title = cli._usage_title()
        self.assertIn("150 tok", title)
        self.assertIn("$0.0000", title)          # local → free
        self.assertNotIn("left", title)          # no budget set → no "left"
        # Plain text only — it's destined for the window chrome.
        self.assertNotIn("\033[", title)

    def test_title_shows_budget_left_when_capped(self):
        self.METER.set_limit(1.0)
        self.METER.record("claude", "claude-opus-4-8", 1_000_000, 0)  # $5/Mtok in
        title = cli._usage_title()
        self.assertIn("left", title)


class TestHardwareLine(unittest.TestCase):
    """``ygg loki status`` surfaces the detected accelerator (Intel GPU / NPU)."""

    def _render(self, snap):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch

        from yggdrasil.cli import style
        from yggdrasil.loki import resources

        buf = io.StringIO()
        snap = resources.Resources(**snap)          # typed snapshot, not a loose dict
        with patch.object(resources, "snapshot", return_value=snap), redirect_stdout(buf):
            cli._print_hardware(style)
        return style.strip(buf.getvalue())

    def test_shows_intel_gpu_and_npu(self):
        line = self._render({"cpu": 16, "ram_gb": 63.5, "gpu": False,
                             "accelerator": "xpu", "npu": True})
        self.assertIn("Intel GPU (xpu)", line)
        self.assertIn("Intel NPU", line)
        self.assertIn("16 cores", line)

    def test_cpu_only_box(self):
        line = self._render({"cpu": 4, "ram_gb": 8.0, "gpu": False,
                             "accelerator": None, "intel_gpu": False, "npu": False})
        self.assertIn("CPU only", line)
        self.assertNotIn("NPU", line)

    def test_present_intel_gpu_without_torch_support(self):
        # torch can't drive it (accelerator None) but the OS detected the iGPU:
        # show the GPU + the install hint instead of a flat "CPU only".
        line = self._render({"cpu": 8, "ram_gb": 16.0, "gpu": False,
                             "accelerator": None, "intel_gpu": True, "npu": True})
        self.assertIn("Intel GPU", line)
        self.assertNotIn("CPU only", line)
        self.assertIn("intel-extension-for-pytorch", line)
        self.assertIn("Intel NPU", line)


class TestIntelGpuEnable(unittest.TestCase):
    """``ygg loki setup`` turns a detected Intel GPU into a usable one."""

    def _run(self, *, gpu=False, accel=None, npu=False, answer="n", rc=0):
        import io
        import subprocess
        from contextlib import redirect_stdout
        from unittest.mock import patch

        from yggdrasil.cli import style
        from yggdrasil.loki import resources

        style.force_color(False)
        buf = io.StringIO()
        proc = subprocess.CompletedProcess([], rc, stdout="", stderr="boom\nthe real error")
        with patch.object(resources, "intel_gpu_present", return_value=gpu), \
                patch.object(resources, "accelerator", return_value=accel), \
                patch.object(resources, "has_npu", return_value=npu), \
                patch("builtins.input", return_value=answer) as inp, \
                patch("subprocess.run", return_value=proc) as run, \
                redirect_stdout(buf):
            cli._enable_intel_gpu(style)
        return style.strip(buf.getvalue()), inp, run

    def test_silent_without_intel_gpu(self):
        out, inp, run = self._run(gpu=False)
        self.assertEqual(out, "")
        inp.assert_not_called()
        run.assert_not_called()

    def test_offers_install_when_gpu_present_but_unusable(self):
        out, inp, run = self._run(gpu=True, accel=None, answer="n")
        self.assertIn("Intel GPU", out)
        self.assertIn("download.pytorch.org/whl/xpu", out)   # the exact command shown
        inp.assert_called_once()                # prompted…
        run.assert_not_called()                 # …but declined → no install

    def test_runs_xpu_install_on_yes(self):
        out, inp, run = self._run(gpu=True, accel=None, answer="y", rc=0)
        run.assert_called_once()
        argv = run.call_args.args[0]
        self.assertIn("--index-url", argv)
        self.assertIn("https://download.pytorch.org/whl/xpu", argv)
        self.assertIn("torch", argv)

    def test_install_failure_surfaces_error_tail(self):
        out, inp, run = self._run(gpu=True, accel=None, answer="y", rc=1)
        self.assertIn("failed", out)
        self.assertIn("the real error", out)    # last stderr line, not the whole dump

    def test_skips_when_gpu_already_usable(self):
        out, inp, run = self._run(gpu=True, accel="xpu")
        self.assertEqual(out, "")               # already xpu → nothing to do


class TestIntelNpuEnable(unittest.TestCase):
    """``ygg loki setup`` enables the NPU (openvino) engine when the NPU is present."""

    def _run(self, *, npu=True, packages=False, answer="n", rc=0):
        import io
        import subprocess
        from contextlib import redirect_stdout
        from unittest.mock import patch

        from yggdrasil.cli import style
        from yggdrasil.loki import resources

        style.force_color(False)
        buf = io.StringIO()
        proc = subprocess.CompletedProcess([], rc, stdout="", stderr="pip\nthe real error")
        # packages present → find_spec returns an object; missing → None.
        spec = object() if packages else None
        with patch.object(resources, "has_npu", return_value=npu), \
                patch("importlib.util.find_spec", return_value=spec), \
                patch("builtins.input", return_value=answer) as inp, \
                patch("subprocess.run", return_value=proc) as run, \
                redirect_stdout(buf):
            cli._enable_intel_npu(style)
        return style.strip(buf.getvalue()), inp, run

    def test_silent_without_npu(self):
        out, inp, run = self._run(npu=False)
        self.assertEqual(out, "")
        inp.assert_not_called()
        run.assert_not_called()

    def test_points_at_engine_when_already_installed(self):
        out, inp, run = self._run(npu=True, packages=True)
        self.assertIn("Intel NPU", out)
        self.assertIn("/engine openvino", out)
        inp.assert_not_called()                 # nothing to install
        run.assert_not_called()

    def test_offers_install_when_packages_missing(self):
        out, inp, run = self._run(npu=True, packages=False, answer="n")
        self.assertIn("optimum[openvino]", out)
        inp.assert_called_once()                # prompted…
        run.assert_not_called()                 # …declined → no install

    def test_runs_optimum_install_on_yes(self):
        out, inp, run = self._run(npu=True, packages=False, answer="y", rc=0)
        run.assert_called_once()
        argv = run.call_args.args[0]
        self.assertIn("optimum[openvino]", argv)
        self.assertIn("/engine openvino", out)

    def test_install_failure_surfaces_error_tail(self):
        out, inp, run = self._run(npu=True, packages=False, answer="y", rc=1)
        self.assertIn("failed", out)
        self.assertIn("the real error", out)


class TestPrompt(unittest.TestCase):
    """The REPL prompt tag — engine·tier, plus a pinned model when set."""

    def test_shows_engine_and_tier(self):
        from yggdrasil.cli import style
        style.force_color(False)
        self.assertIn("claude·auto", style.strip(cli._prompt(style, {"engine": "claude", "tier": None})))

    def test_appends_pinned_model_compactly(self):
        from yggdrasil.cli import style
        style.force_color(False)
        out = style.strip(cli._prompt(style, {
            "engine": "openvino", "tier": "fast",
            "model": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"}))
        self.assertIn("openvino·fast·", out)
        self.assertIn("…", out)                  # long id truncated to a compact tail
        self.assertNotIn("OpenVINO/", out)       # shown as the last path segment only


class TestModelChooser(unittest.TestCase):
    """``/model`` pins the engine's model for the session (interactive picker)."""

    def _engine(self, *, local=True, resource=None, models=None, installed=None, current="cur"):
        from unittest.mock import MagicMock
        eng = MagicMock()
        eng.local = local
        eng.model = None
        eng.RESOURCE_MODELS = resource or {}
        eng.MODELS = models or {}
        eng.bootstrap_model = "rec"
        eng.resolve_model.return_value = current
        if installed is not None:
            eng.installed_models.return_value = installed
        else:
            del eng.installed_models   # remote engine: no such method
        return eng

    def _run(self, eng, *, arg="", answer=""):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import MagicMock, patch

        from yggdrasil.cli import style
        style.force_color(False)
        loki = MagicMock()
        loki.engine.return_value = eng
        state = {"engine": "ollama"}
        buf = io.StringIO()
        with patch("builtins.input", return_value=answer), redirect_stdout(buf):
            cli._choose_model(loki, style, state, arg)
        return state, style.strip(buf.getvalue())

    def test_explicit_arg_pins_directly(self):
        eng = self._engine()
        state, out = self._run(eng, arg="my/model")
        self.assertEqual(eng.model, "my/model")
        self.assertEqual(state["model"], "my/model")

    def test_lists_presets_and_picks_by_number(self):
        eng = self._engine(local=False, models={"fast": "f-model", "deep": "d-model"},
                           current="f-model")
        # The remote engine has no installed_models; pick #2 → the deep model.
        state, out = self._run(eng, answer="2")
        self.assertIn("f-model", out)
        self.assertIn("d-model", out)
        self.assertIn("recommended", out)          # deep is the remote recommendation
        self.assertEqual(eng.model, "d-model")

    def test_ollama_lists_installed_and_keeps_on_blank(self):
        eng = self._engine(local=True, resource={"small": "qwen:3b"},
                           installed=["qwen:3b", "mistral:7b"], current="qwen:3b")
        state, out = self._run(eng, answer="")     # Enter → keep current
        self.assertIn("mistral:7b", out)           # an already-pulled model is offered
        self.assertIn("installed", out)
        self.assertIsNone(eng.model)               # nothing pinned on blank
        self.assertNotIn("model", state)

    def test_no_engine_warns(self):
        from unittest.mock import MagicMock, patch
        import io
        from contextlib import redirect_stdout
        from yggdrasil.cli import style
        style.force_color(False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._choose_model(MagicMock(), style, {"engine": None}, "")
        self.assertIn("no engine", style.strip(buf.getvalue()))


class TestPullBar(unittest.TestCase):
    """The Ollama download progress callback renders a byte bar."""

    def test_renders_bar_on_byte_events(self):
        from unittest.mock import patch
        from yggdrasil.cli import style

        calls = []
        with patch.object(style, "progress", side_effect=lambda c, t, label="": calls.append((c, t, label))):
            cb = cli._pull_bar(style)
            cb({"status": "downloading abc", "completed": 50, "total": 200})
        self.assertEqual(calls[0][0], 50)
        self.assertEqual(calls[0][1], 200)
        self.assertIn("MB", calls[0][2])

    def test_status_only_event_does_not_call_bar(self):
        from unittest.mock import patch
        from yggdrasil.cli import style
        with patch.object(style, "progress") as prog, \
                patch.object(style, "clear_line"), patch.object(style, "out"):
            cli._pull_bar(style)({"status": "verifying sha256 digest"})
        prog.assert_not_called()


class TestActMonitor(unittest.TestCase):
    """The live cumulative step view for the autonomous ``act`` loop."""

    def _drive(self, records, *, verbose=False, max_steps=12):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch

        from yggdrasil.cli import style

        style.force_color(False)
        buf = io.StringIO()
        # Off-TTY: the spinner degrades to a printed line (no animation thread),
        # which keeps the captured output deterministic.
        with patch.object(style, "_IS_TTY", False), redirect_stdout(buf):
            mon = cli._ActMonitor(style, max_steps, verbose=verbose)
            for rec in records:
                mon.think(rec["n"])
                mon.step(rec)
            mon.close()
        return mon, style.strip(buf.getvalue())

    def test_commits_each_tool_step_with_its_thought(self):
        records = [
            {"n": 1, "tool": "list_dir", "args": {"path": "."},
             "thought": "scout the tree first", "observation": "a.py\nb.py"},
            {"n": 2, "tool": "read_file", "args": {"path": "a.py"},
             "thought": "inspect before editing", "observation": "print(1)"},
            {"n": 3, "done": True, "answer": "done"},   # final turn commits nothing
        ]
        mon, out = self._drive(records)
        self.assertEqual(mon.count, 2)                  # only the two tool steps
        self.assertIn("list_dir(path=.)", out)
        self.assertIn("read_file(path=a.py)", out)
        self.assertIn("scout the tree first", out)      # the thinking is surfaced
        self.assertIn("inspect before editing", out)

    def test_verbose_adds_first_observation_line_only(self):
        records = [{"n": 1, "tool": "read_file", "args": {"path": "a.py"},
                    "thought": "look", "observation": "line one\nline two"}]
        _, out = self._drive(records, verbose=True)
        self.assertIn("line one", out)
        self.assertNotIn("line two", out)               # only the first line

    def test_confirm_pauses_spinner_and_delegates(self):
        from unittest.mock import patch

        from yggdrasil.cli import style

        style.force_color(False)
        with patch.object(style, "_IS_TTY", False):
            mon = cli._ActMonitor(style, 12)
            mon.think(1)                                 # spinner up
            with patch("builtins.input", return_value="y"):
                self.assertTrue(mon.confirm("delete it"))
            self.assertIsNone(mon._spin)                 # spinner dropped for the prompt


if __name__ == "__main__":
    unittest.main()
