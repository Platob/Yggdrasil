"""Tests for ``ygg databricks table`` — help + async_insert dispatch."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


class TestTableHelp(unittest.TestCase):

    def test_table_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["table", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_table_async_insert_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["table", "async_insert", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_async_insert_requires_table_name_and_data(self):
        # argparse errors (exit 2) when the required flags are missing.
        with self.assertRaises(SystemExit):
            main(["table", "async_insert"])


class TestTableAsyncInsertDispatch(unittest.TestCase):
    """The CLI is a thin shell — stage via ``Table.async_insert`` (which reads
    the source), and load via the log-path ``Tables.async_insert`` loader."""

    def _client(self):
        client = MagicMock()
        table = MagicMock()
        client.tables.__getitem__.return_value = table
        log_file = MagicMock()
        log_file.full_path.return_value = "/Volumes/c/s/t/.sql/async/logs/op.json"
        table.async_insert.return_value = log_file
        return client, table, log_file

    def test_stage_only(self):
        client, table, _ = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main([
                "table", "async_insert",
                "--table-name", "cat.sch.tbl",
                "--data", "s3://bucket/data.parquet",
                "--mode", "overwrite",
            ])
        self.assertEqual(rc, 0)
        client.tables.__getitem__.assert_called_once_with("cat.sch.tbl")
        table.async_insert.assert_called_once_with(
            "s3://bucket/data.parquet", mode="overwrite",
        )
        client.tables.async_insert.assert_not_called()   # no load without --execute
        table.async_job.assert_not_called()

    def test_execute_runs_loader_on_log_path(self):
        client, table, log_file = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main([
                "table", "async_insert",
                "--table-name", "cat.sch.tbl",
                "--data", "data.parquet",
                "--execute",
            ])
        self.assertEqual(rc, 0)
        # default mode append; loader called with the staged log path
        table.async_insert.assert_called_once_with("data.parquet", mode="append")
        client.tables.async_insert.assert_called_once_with(
            "/Volumes/c/s/t/.sql/async/logs/op.json", wait=True,
        )

    def test_ensure_job_deploys_loader(self):
        client, table, _ = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main([
                "table", "async_insert",
                "--table-name", "cat.sch.tbl",
                "--data", "data.parquet",
                "--ensure-job",
            ])
        self.assertEqual(rc, 0)
        table.async_job.return_value.ensure.assert_called_once_with()
        client.tables.async_insert.assert_not_called()


class TestTableExecuteAsyncInsert(unittest.TestCase):
    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["table", "execute_async_insert", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_runs_loader_over_logs_dir(self):
        client = MagicMock()
        client.tables.async_insert.return_value = 5
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main([
                "table", "execute_async_insert",
                "--logs", "/Volumes/c/s/t/.sql/async/logs",
            ])
        self.assertEqual(rc, 0)
        client.tables.async_insert.assert_called_once_with(
            logs="/Volumes/c/s/t/.sql/async/logs", log_files=None, wait=True,
        )

    def test_runs_loader_over_explicit_log_files(self):
        client = MagicMock()
        client.tables.async_insert.return_value = 2
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main([
                "table", "execute_async_insert",
                "--log-file", "/logs/a.json",
                "--log-file", "/logs/b.json",
            ])
        self.assertEqual(rc, 0)
        client.tables.async_insert.assert_called_once_with(
            logs=None, log_files=["/logs/a.json", "/logs/b.json"], wait=True,
        )
