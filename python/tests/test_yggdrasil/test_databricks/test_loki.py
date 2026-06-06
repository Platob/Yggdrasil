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


class TestReplication(unittest.TestCase):
    def test_inherits_local_spawn(self):
        # Replication is local process spawning inherited from Loki — no jobs.
        loki = DatabricksLoki()
        self.assertTrue(hasattr(loki, "spawn"))
        self.assertTrue(hasattr(loki, "gather"))
        self.assertFalse(hasattr(loki, "deploy"))


if __name__ == "__main__":
    unittest.main()
