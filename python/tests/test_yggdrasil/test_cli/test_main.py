"""``ygg`` is the single console-script entry point — every feature is a
subcommand (databricks / loki), and deployed jobs invoke ``ygg`` with a leading
subcommand (e.g. ``ygg databricks table autoload``) rather than a per-feature
script."""
from __future__ import annotations

from unittest.mock import patch

from yggdrasil.cli.main import main


def test_databricks_table_autoload_delegates(monkeypatch):
    # The deployed Auto Loader wheel-task invokes this on the cluster.
    monkeypatch.setattr(
        "sys.argv",
        ["ygg", "databricks", "table", "autoload", "--table", "c.s.t", "--source", "s3://x"],
    )
    with patch("yggdrasil.databricks.cli.main", return_value=0) as dbks:
        rc = main()
    assert rc == 0
    dbks.assert_called_once_with(
        ["table", "autoload", "--table", "c.s.t", "--source", "s3://x"]
    )


def test_loki_subcommand_delegates_to_loki_cli(monkeypatch):
    monkeypatch.setattr("sys.argv", ["ygg", "loki", "status"])
    with patch("yggdrasil.loki.cli.main", return_value=0) as loki:
        rc = main()
    assert rc == 0
    loki.assert_called_once_with(["status"])


def test_databricks_subcommand_delegates(monkeypatch):
    monkeypatch.setattr("sys.argv", ["ygg", "databricks", "seed"])
    with patch("yggdrasil.databricks.cli.main", return_value=0) as dbks:
        rc = main()
    assert rc == 0
    dbks.assert_called_once_with(["seed"])


def test_no_command_prints_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["ygg"])
    rc = main([])
    assert rc == 0
