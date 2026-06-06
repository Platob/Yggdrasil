"""Specialized skills carry optimized preprompts and use them when reasoning."""
from __future__ import annotations

import unittest

from yggdrasil.loki import Loki
from yggdrasil.loki.capability import Backend


def _loki():
    loki = Loki()
    loki._backends = [Backend("local", True)]
    return loki


class TestPreprompts(unittest.TestCase):
    def test_reasoning_skills_have_preprompts(self):
        loki = _loki()
        for name in ("web", "guide", "python_project"):
            skill = loki.skill(name)
            self.assertTrue(skill.preprompt, name)
            self.assertIn("preprompt", skill.to_dict())

    def test_specialized_fleets_have_domain_preprompts(self):
        from yggdrasil.aws.loki.skills import AWSServiceSkill
        from yggdrasil.databricks.loki.skills import DatabricksServiceSkill

        self.assertIn("Databricks", DatabricksServiceSkill.preprompt)
        self.assertIn("serverless", DatabricksServiceSkill.preprompt)
        self.assertIn("AWS", AWSServiceSkill.preprompt)

    def test_python_project_grounds_prompt_and_uses_preprompt(self):
        loki = _loki()
        captured = {}

        def fake_reason(prompt, *, system=None, **_):
            captured["prompt"] = prompt
            captured["system"] = system
            return "print('ok')"

        loki.reason = fake_reason
        loki.run("python_project", project="g", task="read a parquet file and sum a column",
                 run=False)
        # The reasoning is steered by the skill's preprompt …
        self.assertEqual(captured["system"], loki.skill("python_project").preprompt)
        # … and grounded in the matched yggdrasil recipe (io handlers).
        self.assertIn("IO.from_", captured["prompt"])

    def test_web_question_uses_preprompt(self):
        loki = _loki()
        captured = {}
        loki.reason = lambda prompt, *, system=None, **_: captured.update(system=system) or "ans"
        loki.engine = lambda name=None: type("E", (), {"available": lambda self: True})()

        from unittest.mock import patch

        from yggdrasil.loki import web

        with patch.object(web, "read_text", return_value={"url": "u", "status": 200,
                                                          "content_type": "text/html",
                                                          "text": "hello", "links": [],
                                                          "truncated": False}):
            loki.run("web", url="https://x", question="what does it say?")
        self.assertEqual(captured["system"], loki.skill("web").preprompt)


if __name__ == "__main__":
    unittest.main()
