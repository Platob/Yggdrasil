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
                             "accelerator": None, "npu": False})
        self.assertIn("CPU only", line)
        self.assertNotIn("NPU", line)


if __name__ == "__main__":
    unittest.main()
