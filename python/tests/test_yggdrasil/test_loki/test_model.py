"""Tests for the TokenModel catalog and complexity-based selection."""
from __future__ import annotations

import unittest

from yggdrasil.loki.model import (
    Complexity,
    ModelSpec,
    Provider,
    TokenModel,
    models,
    register_model,
    select_model,
    unregister_model,
)


class TestTokenModel(unittest.TestCase):
    def test_member_exposes_spec(self):
        m = TokenModel.CLAUDE_OPUS_4_8
        self.assertEqual(m.id, "claude-opus-4-8")
        self.assertEqual(m.provider, Provider.ANTHROPIC)
        self.assertEqual(m.complexity, Complexity.HIGH)

    def test_models_filter_by_provider_and_complexity(self):
        low_oai = models(Provider.OPENAI, Complexity.LOW)
        self.assertTrue(all(s.provider == Provider.OPENAI and s.complexity == Complexity.LOW
                            for s in low_oai))
        self.assertIn("gpt-4o-mini", [s.id for s in low_oai])


class TestSelection(unittest.TestCase):
    def test_exact_tier(self):
        self.assertEqual(select_model(Provider.ANTHROPIC, Complexity.LOW).id, "claude-haiku-4-5")
        self.assertEqual(select_model(Provider.ANTHROPIC, "high").id, "claude-opus-4-8")
        self.assertEqual(select_model(Provider.ANTHROPIC, 2).id, "claude-sonnet-4-6")

    def test_falls_up_then_down(self):
        # Drop the HIGH anthropic model; HIGH request falls to the next best.
        try:
            unregister_model("claude-opus-4-8")
            chosen = select_model(Provider.ANTHROPIC, Complexity.HIGH)
            self.assertEqual(chosen.id, "claude-sonnet-4-6")  # nearest below
        finally:
            register_model(TokenModel.CLAUDE_OPUS_4_8.value)


class TestRegistry(unittest.TestCase):
    def test_register_and_unregister(self):
        register_model("my-model", provider="databricks", complexity="low")
        try:
            self.assertIn("my-model", [s.id for s in models(Provider.DATABRICKS)])
            # LOW Databricks had only the built-in oss-20b; selection still
            # returns a LOW-tier model, and the new one is now a candidate.
            self.assertEqual(select_model("databricks", "low").complexity, Complexity.LOW)
        finally:
            unregister_model("my-model")
        self.assertNotIn("my-model", [s.id for s in models(Provider.OPENAI)])

    def test_register_bare_id_needs_provider_and_complexity(self):
        with self.assertRaises(ValueError):
            register_model("x")


if __name__ == "__main__":
    unittest.main()
