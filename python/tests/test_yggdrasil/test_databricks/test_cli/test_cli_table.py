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
    """The CLI is a thin shell over the centralized ``insert`` module — stage
    via ``stage_async_insert`` (which reads the source), and load via the
    log-path ``load_async`` loader."""

    def _client(self):
        client = MagicMock()
        table = MagicMock()
        client.tables.__getitem__.return_value = table
        log_file = MagicMock()
        log_file.full_path.return_value = "/Volumes/c/s/t/.sql/async/logs/op.json"
        return client, table, log_file

    def test_stage_only(self):
        client, table, log_file = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.databricks.table.insert.stage_async_insert",
                   return_value=log_file) as stage, \
             patch("yggdrasil.databricks.table.insert.load_async") as load, \
             patch("yggdrasil.databricks.table.insert.ensure_async_job") as ensure:
            rc = main([
                "table", "async_insert",
                "--table-name", "cat.sch.tbl",
                "--data", "s3://bucket/data.parquet",
                "--mode", "overwrite",
            ])
        self.assertEqual(rc, 0)
        client.tables.__getitem__.assert_called_once_with("cat.sch.tbl")
        stage.assert_called_once_with(table, "s3://bucket/data.parquet", mode="overwrite")
        load.assert_not_called()        # no load without --execute
        ensure.assert_not_called()

    def test_execute_runs_loader_on_log_path(self):
        client, table, log_file = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.databricks.table.insert.stage_async_insert",
                   return_value=log_file) as stage, \
             patch("yggdrasil.databricks.table.insert.load_async") as load:
            rc = main([
                "table", "async_insert",
                "--table-name", "cat.sch.tbl",
                "--data", "data.parquet",
                "--execute",
            ])
        self.assertEqual(rc, 0)
        # default mode append; loader called with the staged log path
        stage.assert_called_once_with(table, "data.parquet", mode="append")
        load.assert_called_once_with(
            client.tables, "/Volumes/c/s/t/.sql/async/logs/op.json", wait=True,
        )

    def test_ensure_job_deploys_loader(self):
        client, table, log_file = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.databricks.table.insert.stage_async_insert",
                   return_value=log_file), \
             patch("yggdrasil.databricks.table.insert.load_async") as load, \
             patch("yggdrasil.databricks.table.insert.ensure_async_job") as ensure:
            rc = main([
                "table", "async_insert",
                "--table-name", "cat.sch.tbl",
                "--data", "data.parquet",
                "--ensure-job",
            ])
        self.assertEqual(rc, 0)
        ensure.assert_called_once_with(table)      # get-or-create the loader job
        load.assert_not_called()


class TestTableExecuteAsyncInsert(unittest.TestCase):
    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["table", "execute_async_insert", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_runs_loader_over_logs_dir(self):
        client = MagicMock()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.databricks.table.insert.load_async", return_value=5) as load:
            rc = main([
                "table", "execute_async_insert",
                "--logs", "/Volumes/c/s/t/.sql/async/logs",
            ])
        self.assertEqual(rc, 0)
        load.assert_called_once_with(
            client.tables, logs="/Volumes/c/s/t/.sql/async/logs", log_files=None,
            wait=True, debug=False, prune_partitions=False, use_spark=False,
        )

    def test_runs_loader_over_explicit_log_files(self):
        client = MagicMock()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.databricks.table.insert.load_async", return_value=2) as load:
            rc = main([
                "table", "execute_async_insert",
                "--log-file", "/logs/a.json",
                "--log-file", "/logs/b.json",
            ])
        self.assertEqual(rc, 0)
        load.assert_called_once_with(
            client.tables, logs=None, log_files=["/logs/a.json", "/logs/b.json"],
            wait=True, debug=False, prune_partitions=False, use_spark=False,
        )
