"""Dispatch tests for ``ygg databricks table autoload`` — the on-cluster data
plane that the deployed Auto Loader wheel-task runs (calls :func:`auto_load`)."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from yggdrasil.databricks.cli import main


class TestTableHelp(unittest.TestCase):
    def test_table_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["table", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_autoload_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["table", "autoload", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_table_and_source_are_required(self):
        # argparse exits non-zero when a required option is missing.
        with self.assertRaises(SystemExit) as ctx:
            main(["table", "autoload"])
        self.assertNotEqual(ctx.exception.code, 0)


class TestAutoloadRun(unittest.TestCase):
    def test_calls_auto_load_with_coerced_kwargs(self):
        with patch("yggdrasil.databricks.table.auto_loader.auto_load",
                   return_value={"table": "c.s.t", "rows": 7}) as al:
            rc = main([
                "table", "autoload",
                "--table", "c.s.t",
                "--source", "s3://bkt/landing",
                "--format", "json",
                "--checkpoint", "s3://bkt/ckpt",
                "--clean-source",
                "--clean-source-retention", "30 days",
            ])
        self.assertEqual(rc, 0)
        al.assert_called_once_with(
            table="c.s.t",
            source="s3://bkt/landing",
            file_format="json",
            checkpoint="s3://bkt/ckpt",
            available_now=True,
            clean_source=True,
            clean_source_retention="30 days",
        )

    def test_short_flags_and_defaults(self):
        with patch("yggdrasil.databricks.table.auto_loader.auto_load",
                   return_value={}) as al:
            rc = main(["table", "autoload", "-t", "c.s.t", "-s", "/Volumes/x/y/z"])
        self.assertEqual(rc, 0)
        kwargs = al.call_args.kwargs
        self.assertEqual(kwargs["table"], "c.s.t")
        self.assertEqual(kwargs["source"], "/Volumes/x/y/z")
        self.assertEqual(kwargs["file_format"], "parquet")   # default
        self.assertEqual(kwargs["checkpoint"], "")           # default
        self.assertIs(kwargs["available_now"], True)         # default
        self.assertIs(kwargs["clean_source"], False)         # default
        self.assertEqual(kwargs["clean_source_retention"], "8 days")

    def test_no_available_now_runs_continuous(self):
        with patch("yggdrasil.databricks.table.auto_loader.auto_load",
                   return_value={}) as al:
            rc = main(["table", "autoload", "-t", "c.s.t", "-s", "s3://x",
                       "--no-available-now"])
        self.assertEqual(rc, 0)
        self.assertIs(al.call_args.kwargs["available_now"], False)


if __name__ == "__main__":
    unittest.main()
