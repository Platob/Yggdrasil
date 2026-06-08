"""Dispatch tests for ``ygg databricks job`` (mocked client)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


class TestJobsHelp(unittest.TestCase):
    def test_job_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["job", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestJobsDispatch(unittest.TestCase):
    def _client(self) -> MagicMock:
        client = MagicMock()
        job = MagicMock()
        job.job_id = 42
        job.name = "etl"
        job.dag.return_value = SimpleNamespace(keys=["a", "b"], edges=lambda: [("a", "b")])
        client.jobs.list.return_value = [job]
        client.jobs.get.return_value = job
        return client, job

    def test_list_dispatches_to_jobs_service(self):
        client, _ = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main(["job", "list", "--name", "etl"])
        self.assertEqual(rc, 0)
        client.jobs.list.assert_called_once_with(name="etl", limit=None)

    def test_get_resolves_by_name_and_prints_dag(self):
        client, job = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main(["job", "get", "etl"])
        self.assertEqual(rc, 0)
        client.jobs.get.assert_called_once_with(name="etl")
        job.dag.assert_called_once()

    def test_get_resolves_numeric_id(self):
        client, job = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main(["job", "get", "42"])
        self.assertEqual(rc, 0)
        client.jobs.get.assert_called_once_with(job_id=42)

    def test_run_passes_parameters_and_waits(self):
        client, job = self._client()
        run = job.run.return_value
        run.run_id = 7
        run.is_succeeded = True
        run.duration_seconds = 1.5
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main(["job", "run", "etl", "--param", "day=2024-01-01", "--wait", "--timeout", "10"])
        self.assertEqual(rc, 0)
        kwargs = job.run.call_args.kwargs
        self.assertEqual(kwargs["parameters"], {"day": "2024-01-01"})
        self.assertEqual(kwargs["wait"], 10.0)

    def test_delete_by_id(self):
        client, _ = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main(["job", "delete", "42"])
        self.assertEqual(rc, 0)
        client.jobs.delete.assert_called_once_with(job_id=42)


if __name__ == "__main__":
    unittest.main()
