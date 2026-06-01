"""Tests for ``ygg databricks table`` — help + async_insert dispatch."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main
from yggdrasil.enums.mode import Mode


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

    def test_async_insert_routes_to_table_insert(self):
        client = MagicMock()
        table = MagicMock()
        client.tables.__getitem__.return_value = table
        arrow = object()
        source = MagicMock()
        source.read_arrow_table.return_value = arrow

        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.io.holder.IO.from_", return_value=source) as io_from:
            rc = main([
                "table", "async_insert",
                "--table-name", "cat.sch.tbl",
                "--data", "s3://bucket/data.parquet",
                "--mode", "overwrite",
            ])

        self.assertEqual(rc, 0)
        client.tables.__getitem__.assert_called_once_with("cat.sch.tbl")
        io_from.assert_called_once_with("s3://bucket/data.parquet")
        source.read_arrow_table.assert_called_once_with()
        table.insert.assert_called_once_with(arrow, wait=False, mode=Mode.OVERWRITE)
        # without --ensure-job the loader job is not deployed
        table.async_job.assert_not_called()

    def test_async_insert_ensure_job_deploys_loader(self):
        client = MagicMock()
        table = MagicMock()
        client.tables.__getitem__.return_value = table
        source = MagicMock()
        source.read_arrow_table.return_value = object()

        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.io.holder.IO.from_", return_value=source):
            rc = main([
                "table", "async_insert",
                "--table-name", "cat.sch.tbl",
                "--data", "data.parquet",
                "--ensure-job",
            ])

        self.assertEqual(rc, 0)
        # default mode is append
        table.insert.assert_called_once_with(
            source.read_arrow_table.return_value, wait=False, mode=Mode.APPEND,
        )
        table.async_job.return_value.ensure.assert_called_once_with()
