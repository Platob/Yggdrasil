"""Dispatch tests for ``ygg databricks sql`` (mocked client + real file export)."""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pyarrow as pa

from yggdrasil.databricks.cli import main

_TABLE = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


def _client_returning(result):
    client = MagicMock()
    client.sql.execute.return_value = result
    client.sql.statement_result.return_value = result
    return client


def _result(statement_id="01ef-abc"):
    r = MagicMock()
    r.statement_id = statement_id
    r.to_arrow_table.return_value = _TABLE
    return r


class TestSQLHelp(unittest.TestCase):
    def test_sql_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["sql", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_export_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["sql", "export", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestSQLExport(unittest.TestCase):
    def test_export_by_statement_id_writes_parquet(self):
        client = _client_returning(_result())
        target = os.path.join(tempfile.mkdtemp(), "out.parquet")
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["sql", "export", "--statement-id", "01ef-abc", "--target", target])
        self.assertEqual(rc, 0)
        client.sql.statement_result.assert_called_once_with(
            "01ef-abc", warehouse_id=None, warehouse_name=None,
        )
        client.sql.execute.assert_not_called()
        self.assertEqual(pa.parquet.read_table(target).num_rows, 3)

    def test_export_by_query_runs_then_writes_csv(self):
        client = _client_returning(_result())
        target = os.path.join(tempfile.mkdtemp(), "out.csv")
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["sql", "export", "--query", "SELECT 1", "--target", target])
        self.assertEqual(rc, 0)
        client.sql.execute.assert_called_once()
        client.sql.statement_result.assert_not_called()
        self.assertTrue(os.path.getsize(target) > 0)

    def test_export_format_override(self):
        client = _client_returning(_result())
        # No usable extension → must rely on --format.
        target = os.path.join(tempfile.mkdtemp(), "result")
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["sql", "export", "--statement-id", "x", "--target", target, "--format", "parquet"])
        self.assertEqual(rc, 0)
        self.assertEqual(pa.parquet.read_table(target).num_rows, 3)

    def test_export_requires_a_source(self):
        client = _client_returning(_result())
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            rc = main(["sql", "export", "--target", "out.parquet"])
        self.assertEqual(rc, 1)

    def test_export_workspace_target_uses_client_path(self):
        client = _client_returning(_result())
        sink = MagicMock()
        buf = io.BytesIO()
        sink.open.return_value.__enter__.return_value = MagicMock(
            write_arrow_table=lambda t: buf.write(b"x" * t.num_rows))
        sink.full_path.return_value = "/Volumes/main/default/stg/out.csv"
        client.path.return_value = sink
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["sql", "export", "--statement-id", "x",
                       "--target", "/Volumes/main/default/stg/out.csv"])
        self.assertEqual(rc, 0)
        client.path.assert_called_once_with("/Volumes/main/default/stg/out.csv")


class TestSQLQuery(unittest.TestCase):
    def test_query_preview_defaults_to_50_row_limit(self):
        client = _client_returning(_result())
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["sql", "query", "SELECT * FROM t"])
        self.assertEqual(rc, 0)
        self.assertEqual(client.sql.execute.call_args.kwargs["row_limit"], 50)

    def test_query_with_target_exports_full_result(self):
        client = _client_returning(_result())
        target = os.path.join(tempfile.mkdtemp(), "q.parquet")
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["sql", "query", "SELECT * FROM t", "--target", target])
        self.assertEqual(rc, 0)
        # Exports pull the whole result — no implicit preview limit.
        self.assertIsNone(client.sql.execute.call_args.kwargs["row_limit"])
        self.assertEqual(pa.parquet.read_table(target).num_rows, 3)


if __name__ == "__main__":
    unittest.main()
