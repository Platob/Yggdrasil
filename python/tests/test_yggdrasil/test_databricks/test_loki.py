"""Tests for the specialized DatabricksLoki agent."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.loki import DatabricksLoki


def _session(**over):
    base = {"host": "https://w.cloud.databricks.com", "profile": "prod",
            "user": "me@co.com", "auth_type": "pat"}
    base.update(over)
    return base


class TestDetection(unittest.TestCase):
    def test_detects_only_from_configure_session(self):
        with patch("yggdrasil.databricks.loki.agent.read_session", return_value=_session()):
            loki = DatabricksLoki()
            b = loki.backend("databricks")
        self.assertTrue(b.available)
        self.assertEqual(b.detail["profile"], "prod")
        self.assertEqual(b.detail["source"], "ygg databricks configure")

    def test_no_session_means_no_databricks(self):
        with patch("yggdrasil.databricks.loki.agent.read_session", return_value=None):
            loki = DatabricksLoki()
            self.assertFalse(loki.has("databricks"))
            self.assertIsNone(loki.databricks)

    def test_client_built_from_session_profile(self):
        with patch("yggdrasil.databricks.loki.agent.read_session", return_value=_session(profile="prod")):
            loki = DatabricksLoki()
            with patch("yggdrasil.databricks.DatabricksClient") as DC:
                DC.return_value = MagicMock()
                client = loki.databricks
            DC.assert_called_once_with(profile="prod")
            self.assertIs(client, loki.databricks)  # cached


class TestEngines(unittest.TestCase):
    def test_prefers_databricks_serving(self):
        self.assertEqual(DatabricksLoki.ENGINE_PREFERENCE[0], "databricks")

    def test_databricks_engine_only_when_session_client(self):
        with patch("yggdrasil.databricks.loki.agent.read_session", return_value=None):
            loki = DatabricksLoki()
            # No configure session → no databricks engine bound (no env fallback).
            self.assertNotIn("databricks", loki._engine_instances())


class TestDeploy(unittest.TestCase):
    def test_deploy_requires_session(self):
        with patch("yggdrasil.databricks.loki.agent.read_session", return_value=None):
            with self.assertRaises(RuntimeError):
                DatabricksLoki().deploy()

    def test_deploy_creates_serverless_job(self):
        client = MagicMock()
        with patch("yggdrasil.databricks.loki.agent.read_session", return_value=_session()), \
             patch("yggdrasil.databricks.DatabricksClient", return_value=client), \
             patch("yggdrasil.databricks.job.wheel.ygg_environment", return_value="ENV"):
            loki = DatabricksLoki()
            loki.deploy(name="loki-test", behavior="reason", prompt="hi")
        call = client.jobs.create_or_update.call_args
        self.assertEqual(call.kwargs["name"], "loki-test")
        self.assertEqual(call.kwargs["environments"], ["ENV"])
        task = call.kwargs["tasks"][0]
        # Single ``ygg`` entry point; reason routes through `ygg loki reason`.
        self.assertEqual(task.python_wheel_task.entry_point, "ygg")
        self.assertEqual(task.python_wheel_task.package_name, "ygg")
        self.assertEqual(task.python_wheel_task.parameters, ["loki", "reason", "hi"])

    def test_deploy_behavior_routes_through_loki_run(self):
        client = MagicMock()
        with patch("yggdrasil.databricks.loki.agent.read_session", return_value=_session()), \
             patch("yggdrasil.databricks.DatabricksClient", return_value=client), \
             patch("yggdrasil.databricks.job.wheel.ygg_environment", return_value="ENV"):
            loki = DatabricksLoki()
            loki.deploy(name="loki-genie", behavior="genie", question="revenue?")
        task = client.jobs.create_or_update.call_args.kwargs["tasks"][0]
        self.assertEqual(task.python_wheel_task.entry_point, "ygg")
        self.assertEqual(
            task.python_wheel_task.parameters,
            ["loki", "run", "genie", "--kwarg", 'question="revenue?"'],
        )


if __name__ == "__main__":
    unittest.main()
