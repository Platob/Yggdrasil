"""Tests for token accounting — pricing, the meter, and the budget."""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.exceptions import TokenBudgetExceeded
from yggdrasil.loki.engines import ClaudeEngine
from yggdrasil.loki.usage import (
    METER,
    ModelPricing,
    TokenMeter,
    estimate_tokens,
    price_for,
)


class TestPricing(unittest.TestCase):
    def test_cost_is_per_million_tokens(self):
        p = ModelPricing(5.0, 25.0)  # $5 in / $25 out per 1M
        self.assertAlmostEqual(p.cost(1_000_000, 0), 5.0)
        self.assertAlmostEqual(p.cost(0, 1_000_000), 25.0)
        self.assertAlmostEqual(p.cost(1000, 1000), (5.0 + 25.0) / 1000)

    def test_price_for_falls_back_engine_then_default(self):
        self.assertEqual(price_for("claude", "claude-opus-4-8").input_usd_per_mtok, 5.0)
        # Unknown claude model → per-engine "*" row.
        self.assertEqual(price_for("claude", "mystery").input_usd_per_mtok, 5.0)
        # Unknown engine → global default.
        self.assertEqual(price_for("nope", "x").input_usd_per_mtok, 1.0)

    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertGreater(estimate_tokens("x" * 400), 90)


class TestTokenMeter(unittest.TestCase):
    def setUp(self):
        self.m = TokenMeter()

    def test_record_accumulates_per_model_and_global(self):
        self.m.record("claude", "claude-haiku-4-5", 1000, 200)
        self.m.record("claude", "claude-haiku-4-5", 500, 100)
        self.m.record("openai", "gpt-4o", 300, 50)
        rows = self.m.rows()
        self.assertEqual(len(rows), 2)
        haiku = next(r for r in rows if r.model == "claude-haiku-4-5")
        self.assertEqual(haiku.calls, 2)
        self.assertEqual(haiku.input_tokens, 1500)
        self.assertEqual(haiku.output_tokens, 300)
        self.assertEqual(self.m.total().total_tokens, 1500 + 300 + 300 + 50)

    def test_cost_prices_each_row_at_its_own_rate(self):
        self.m.record("claude", "claude-haiku-4-5", 1_000_000, 0)   # $1
        self.m.record("openai", "gpt-4o", 0, 1_000_000)             # $10
        self.assertAlmostEqual(self.m.total_cost, 11.0)

    def test_rows_for_filters_by_engine(self):
        self.m.record("claude", "claude-opus-4-8", 10, 10)
        self.m.record("openai", "gpt-4o", 10, 10)
        self.assertEqual([r.engine for r in self.m.rows_for("openai")], ["openai"])

    def test_cost_budget_set_raise_remaining_and_check(self):
        # haiku is $1 / 1M input tokens → 500k input = $0.50.
        self.m.record("claude", "claude-haiku-4-5", 500_000, 0)
        self.assertIsNone(self.m.remaining())          # no cap → unlimited
        self.m.set_limit(1.0)                          # $1 cap
        self.assertAlmostEqual(self.m.remaining(), 0.50)
        self.m.check_budget()                          # under → no raise
        self.m.record("claude", "claude-haiku-4-5", 600_000, 0)  # +$0.60 → $1.10
        self.assertTrue(self.m.over_budget())
        with self.assertRaises(TokenBudgetExceeded) as ctx:
            self.m.check_budget()
        self.assertAlmostEqual(ctx.exception.limit, 1.0)
        self.assertAlmostEqual(ctx.exception.used, 1.10)
        self.m.raise_limit()                           # +$1 step → $2
        self.assertAlmostEqual(self.m.cost_limit, 2.0)
        self.m.check_budget()                          # back under

    def test_raise_limit_by_amount(self):
        self.m.set_limit(1.0)
        self.assertAlmostEqual(self.m.raise_limit(0.5), 1.5)

    def test_reset_clears_rows(self):
        self.m.record("claude", "claude-opus-4-8", 10, 10)
        self.m.reset()
        self.assertEqual(self.m.rows(), [])
        self.assertEqual(self.m.total_tokens, 0)


class TestEngineRecordsUsage(unittest.TestCase):
    """A completion records into the global meter, keyed by engine + model."""

    def setUp(self):
        self._saved = dict(METER._rows)
        self._limit = METER.cost_limit
        METER.reset()
        METER.set_limit(None)

    def tearDown(self):
        METER.reset()
        METER._rows.update(self._saved)
        METER.cost_limit = self._limit

    def test_claude_records_provider_usage(self):
        fake = types.ModuleType("anthropic")
        client = MagicMock()
        block = MagicMock(); block.type = "text"; block.text = "hi"
        usage = MagicMock(); usage.input_tokens = 123; usage.output_tokens = 45
        client.messages.create.return_value = MagicMock(
            content=[block], model="claude-haiku-4-5", usage=usage,
        )
        fake.Anthropic = MagicMock(return_value=client)
        with patch.dict(sys.modules, {"anthropic": fake}):
            ClaudeEngine(api_key="k").generate("hi")   # short → haiku tier
        rows = METER.rows_for("claude")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].input_tokens, 123)
        self.assertEqual(rows[0].output_tokens, 45)
        self.assertEqual(rows[0].model, "claude-haiku-4-5")

    def test_records_estimate_when_provider_gives_no_usage(self):
        fake = types.ModuleType("anthropic")
        client = MagicMock()
        block = MagicMock(); block.type = "text"; block.text = "x" * 400
        client.messages.create.return_value = MagicMock(
            content=[block], model="claude-haiku-4-5", usage=None,
        )
        fake.Anthropic = MagicMock(return_value=client)
        with patch.dict(sys.modules, {"anthropic": fake}):
            ClaudeEngine(api_key="k").generate("a short prompt")
        row = METER.rows_for("claude")[0]
        self.assertGreater(row.output_tokens, 90)      # estimated from 400 chars
        self.assertGreater(row.input_tokens, 0)


if __name__ == "__main__":
    unittest.main()
