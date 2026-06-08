"""Tests for the `guide` skill — optimized yggdrasil implementation advice."""
from __future__ import annotations

import unittest

from yggdrasil.loki import Loki
from yggdrasil.loki.capability import Backend
from yggdrasil.loki.guides import GUIDES, match


def _loki():
    loki = Loki()
    loki._backends = [Backend("local", True)]
    return loki


class TestGuideMatching(unittest.TestCase):
    def test_every_guide_has_signals_and_content(self):
        for g in GUIDES:
            self.assertTrue(g.signals and g.use and g.example and g.avoid, g.id)

    def test_match_picks_relevant_recipes(self):
        ids = [g.id for g in match("read a parquet file and write arrow")]
        self.assertIn("tabular-io", ids)
        ids = [g.id for g in match("fetch JSON from an API endpoint")]
        self.assertIn("http-fetch", ids)
        ids = [g.id for g in match("share a dataframe with an llm in a prompt")]
        self.assertIn("llm-data", ids)
        ids = [g.id for g in match("cast a column to a typed date with timezone")]
        self.assertIn("schema-cast", ids)

    def test_match_returns_nothing_for_unrelated(self):
        self.assertEqual(match("xyzzy plugh nothing relevant"), [])


class TestGuideSkill(unittest.TestCase):
    def test_registered_and_available_anywhere(self):
        names = [s.name for s in _loki().skills()]
        self.assertIn("guide", names)

    def test_run_with_task_returns_guides_and_topics(self):
        res = _loki().run("guide", task="fetch a CSV and cache it as parquet, typed")
        self.assertTrue(res["guides"])
        self.assertIn("tabular-io", [g["id"] for g in res["guides"]])
        self.assertIn("http-fetch", res["topics"])
        # No engine asked for → no synthesized plan.
        self.assertNotIn("plan", res)

    def test_run_with_topic(self):
        res = _loki().run("guide", topic="databricks-compute")
        self.assertEqual(res["guides"][0]["id"], "databricks-compute")

    def test_unknown_topic_raises(self):
        with self.assertRaises(KeyError):
            _loki().run("guide", topic="not-a-topic")

    def test_requires_task_or_topic(self):
        with self.assertRaises(ValueError):
            _loki().run("guide")

    def test_plan_grounds_the_engine_on_matched_guides(self):
        loki = _loki()
        captured = {}

        def fake_reason(prompt, *, system=None, **_):
            captured["prompt"] = prompt
            captured["system"] = system
            return "1. Use IO.from_(...)\n2. ..."

        loki.reason = fake_reason
        loki.engine = lambda name=None: type("E", (), {"available": lambda self: True})()
        res = loki.run("guide", task="read a parquet and type it", plan=True)
        self.assertIn("plan", res)
        # The prompt is grounded in the matched recipes (their snippets).
        self.assertIn("IO.from_", captured["prompt"])
        self.assertIn("yggdrasil expert", captured["system"])


class TestGuideRouting(unittest.TestCase):
    def test_how_to_in_yggdrasil_routes_to_guide(self):
        plan = _loki().plan("what's the best way to fetch a CSV in yggdrasil?")
        self.assertEqual(plan["category"], "guide")
        self.assertEqual(plan["action"], "guide")

    def test_plain_how_to_without_yggdrasil_does_not_route_to_guide(self):
        plan = _loki().plan("how do I fix this null pointer bug")
        self.assertNotEqual(plan["action"], "guide")


if __name__ == "__main__":
    unittest.main()
