"""Unit tests for the global yggdrasil agent (Loki)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.loki import Loki
from yggdrasil.loki.behavior import REGISTRY, LokiBehavior, register
from yggdrasil.loki.capability import Backend


class _Dummy(LokiBehavior):
    name = "dummy-test"
    description = "echo kwargs"

    def run(self, agent, **kwargs):
        return {"agent": agent.name, **kwargs}


class _NeedsDatabricks(LokiBehavior):
    name = "needs-dbx-test"
    requires = "databricks"

    def run(self, agent, **kwargs):
        return "ran"


class TestBehaviorRegistry(unittest.TestCase):
    def setUp(self):
        self._saved = dict(REGISTRY)

    def tearDown(self):
        REGISTRY.clear()
        REGISTRY.update(self._saved)

    def test_register_and_dispatch(self):
        register(_Dummy)
        loki = Loki()
        self.assertIn("dummy-test", [b.name for b in loki.behaviors()])
        self.assertEqual(loki.run("dummy-test", x=1), {"agent": "loki", "x": 1})

    def test_unknown_behavior_raises(self):
        with self.assertRaises(KeyError):
            Loki().run("nope-not-real")

    def test_unavailable_behavior_raises(self):
        register(_NeedsDatabricks)
        loki = Loki()
        loki._backends = [Backend("databricks", available=False)]
        with self.assertRaises(RuntimeError):
            loki.run("needs-dbx-test")

    def test_available_behavior_runs(self):
        register(_NeedsDatabricks)
        loki = Loki()
        loki._backends = [Backend("databricks", available=True)]
        self.assertEqual(loki.run("needs-dbx-test"), "ran")


class TestAgent(unittest.TestCase):
    def test_identity_is_stable_int64(self):
        loki = Loki()
        self.assertIsInstance(loki.agent_id, int)
        self.assertEqual(loki.agent_id, Loki().agent_id)
        self.assertLess(loki.agent_id, 2**63)

    def test_backend_lookup_and_has(self):
        loki = Loki()
        loki._backends = [Backend("databricks", True, {"host": "h"}), Backend("local", True)]
        self.assertTrue(loki.has("databricks"))
        self.assertFalse(loki.has("node"))
        self.assertEqual(loki.backend("databricks").detail["host"], "h")

    def test_databricks_token_provider(self):
        loki = Loki()
        loki._backends = [Backend("databricks", True, {"host": "https://w", "auth_type": "pat"})]
        info = loki.token_info()
        self.assertTrue(info["available"])
        self.assertEqual(info["host"], "https://w")

        fake_client = MagicMock()
        with patch("yggdrasil.databricks.DatabricksClient") as DC:
            DC.current.return_value = fake_client
            self.assertIs(loki.databricks, fake_client)

    def test_no_databricks_means_no_token(self):
        loki = Loki()
        loki._backends = [Backend("databricks", False), Backend("local", True)]
        self.assertIsNone(loki.databricks)
        self.assertFalse(loki.token_info()["available"])

    def test_card_shape(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        card = loki.card()
        self.assertEqual(card["agent"], "loki")
        self.assertIn("backends", card)
        self.assertIn("behaviors", card)
        self.assertIn("token", card)


class TestGenieBehavior(unittest.TestCase):
    def test_genie_requires_databricks_and_asks_first_space(self):
        from yggdrasil.loki.behaviors import GenieBehavior

        beh = GenieBehavior()
        self.assertEqual(beh.requires, "databricks")

        loki = Loki()
        loki._backends = [Backend("databricks", True)]
        client = MagicMock()
        space = MagicMock(); space.space_id = "s1"
        answer = MagicMock()
        answer.conversation_id = "c1"; answer.text = "hi"; answer.query = None
        answer.statement_id = None
        space.ask.return_value = answer
        client.genie.spaces.return_value = [space]
        with patch("yggdrasil.databricks.DatabricksClient") as DC:
            DC.current.return_value = client
            out = loki.run("genie", question="how many orders?")
        self.assertEqual(out["space_id"], "s1")
        self.assertEqual(out["text"], "hi")
        client.genie.spaces.assert_called_once()


class TestPythonProjectBehavior(unittest.TestCase):
    def test_runs_anywhere_and_executes_supplied_code(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        out = loki.run(
            "python_project",
            project="My Demo",
            code="def main():\n    print('hi from', 7 * 6)\n\nif __name__ == '__main__':\n    main()\n",
        )
        self.assertEqual(out["package"], "my_demo")           # slugged
        self.assertEqual(out["returncode"], 0)
        self.assertIn("hi from 42", out["stdout"])
        self.assertIn("pyproject.toml", out["files"])
        self.assertIn("my_demo/main.py", out["files"])

    def test_reasons_code_from_task_and_strips_fences(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        loki.reason = lambda prompt, *, system=None, **_: (
            "```python\nprint('reasoned ok')\n```"
        )
        out = loki.run("python_project", project="r", task="say hello")
        self.assertEqual(out["returncode"], 0)
        self.assertIn("reasoned ok", out["stdout"])

    def test_scaffold_only_without_run(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        out = loki.run("python_project", project="noexec", code="print('x')\n", run=False)
        self.assertNotIn("returncode", out)
        self.assertTrue(out["project_dir"].endswith("noexec"))

    def test_requires_code_or_task(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        with self.assertRaises(ValueError):
            loki.run("python_project", project="empty")


if __name__ == "__main__":
    unittest.main()
