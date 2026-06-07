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
