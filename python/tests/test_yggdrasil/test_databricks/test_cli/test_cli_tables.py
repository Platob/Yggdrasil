"""Dispatch tests for ``ygg databricks tables`` (mocked client)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


class TestTablesHelp(unittest.TestCase):
    def test_tables_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["tables", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_autoload_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["tables", "autoload", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestAutoloadDispatch(unittest.TestCase):
    def _client(self):
        client = MagicMock()
        table = MagicMock()
        table.full_name.return_value = "cat.sch.events"
        job = MagicMock()
        job.job_id = 99
        job.name = "[YGG][AUTOLOADER] cat.sch.events"
        table.auto_loader.return_value = job
        client.tables.table.return_value = table
        return client, table, job

    def test_autoload_resolves_table_and_calls_auto_loader(self):
        client, table, job = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main(["tables", "autoload", "cat.sch.events", "--file-arrival"])
        self.assertEqual(rc, 0)
        client.tables.table.assert_called_once_with("cat.sch.events")
        kwargs = table.auto_loader.call_args.kwargs
        self.assertIs(kwargs["file_arrival"], True)
        self.assertEqual(kwargs["available_now"], True)        # default sweep
        self.assertEqual(kwargs["environment"], "yellow")      # default named env
        self.assertEqual(kwargs["bundle_dependencies"], True)  # default 0-pip-install
        self.assertIs(kwargs["deploy"], True)

    def test_no_environment_and_no_bundle_flags(self):
        client, table, _ = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main([
                "tables", "autoload", "cat.sch.events",
                "--no-environment", "--no-bundle", "--continuous", "--format", "json",
            ])
        self.assertEqual(rc, 0)
        kwargs = table.auto_loader.call_args.kwargs
        self.assertIsNone(kwargs["environment"])
        self.assertEqual(kwargs["bundle_dependencies"], False)
        self.assertEqual(kwargs["available_now"], False)       # --continuous
        self.assertEqual(kwargs["file_format"], "json")

    def test_no_deploy_returns_configured_flow(self):
        client, table, _ = self._client()
        table.auto_loader.return_value = MagicMock(name="etl_flow")
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main(["tables", "autoload", "cat.sch.events", "--no-deploy"])
        self.assertEqual(rc, 0)
        self.assertIs(table.auto_loader.call_args.kwargs["deploy"], False)

    def test_run_triggers_a_run(self):
        client, table, job = self._client()
        run = job.run.return_value
        run.run_id = 5
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main(["tables", "autoload", "cat.sch.events", "--run"])
        self.assertEqual(rc, 0)
        job.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
