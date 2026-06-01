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
    """The CLI is a thin shell — it delegates to ``client.tables.async_insert``."""

    def test_async_insert_delegates_to_service(self):
        client = MagicMock()
        log_file = MagicMock()
        log_file.full_path.return_value = "/Volumes/c/s/t/.sql/async/logs/op.json"
        client.tables.async_insert.return_value = log_file

        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main([
                "table", "async_insert",
                "--table-name", "cat.sch.tbl",
                "--data", "s3://bucket/data.parquet",
                "--mode", "overwrite",
            ])

        self.assertEqual(rc, 0)
        client.tables.async_insert.assert_called_once_with(
            "cat.sch.tbl", "s3://bucket/data.parquet",
            mode="overwrite", ensure_job=False,
        )

    def test_async_insert_passes_ensure_job_and_default_mode(self):
        client = MagicMock()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main([
                "table", "async_insert",
                "--table-name", "cat.sch.tbl",
                "--data", "data.parquet",
                "--ensure-job",
            ])

        self.assertEqual(rc, 0)
        client.tables.async_insert.assert_called_once_with(
            "cat.sch.tbl", "data.parquet", mode="append", ensure_job=True,
        )
