"""Dispatch tests for ``ygg databricks environment`` — CRUD via ``dbc.environments``."""
from __future__ import annotations

import contextlib
import io
import types
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


def _env(name="ygg-1.0-py311"):
    env_dir = f"/Workspace/Shared/environment/{name.split('-')[0]}"
    return types.SimpleNamespace(
        name=name, project="ygg",
        serverless=f"{env_dir}/{name}.yml",
        cluster=f"{env_dir}/{name}.requirements.txt",
        dependencies=["a", "b", "c"],
    )


def _run(argv, client):
    buf = io.StringIO()
    with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
         patch("yggdrasil.cli.style.print_logo"), \
         contextlib.redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


class TestEnvironmentHelp(unittest.TestCase):
    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["environment", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestEnvironmentCreate(unittest.TestCase):
    def test_create_builds_and_writes(self):
        client = MagicMock()
        client.environments.create.return_value = _env()
        rc, out = _run(["environment", "create", "ygg", "--python", "3.11", "--extra", "databricks"], client)
        self.assertEqual(rc, 0)
        client.environments.create.assert_called_once()
        self.assertEqual(client.environments.create.call_args.args[0], "ygg")
        self.assertEqual(client.environments.create.call_args.kwargs["python"], "3.11")
        self.assertIn("ygg-1.0-py311.yml", out)

    def test_update_overwrites(self):
        client = MagicMock()
        client.environments.update.return_value = _env()
        rc, _ = _run(["environment", "update", "ygg"], client)
        self.assertEqual(rc, 0)
        client.environments.update.assert_called_once()


class TestEnvironmentFindGet(unittest.TestCase):
    def test_find_builds_on_miss(self):
        client = MagicMock()
        client.environments.find.return_value = _env()
        rc, out = _run(["environment", "find", "ygg"], client)
        self.assertEqual(rc, 0)
        self.assertTrue(client.environments.find.call_args.kwargs["install"])
        self.assertIn("ygg-1.0-py311.requirements.txt", out)

    def test_get_missing_returns_one(self):
        client = MagicMock()
        client.environments.get.return_value = None
        rc, _ = _run(["environment", "get", "ygg"], client)
        self.assertEqual(rc, 1)


class TestEnvironmentDelete(unittest.TestCase):
    def test_delete_removes(self):
        client = MagicMock()
        client.environments.delete.return_value = [_env()]
        rc, out = _run(["environment", "delete", "ygg"], client)
        self.assertEqual(rc, 0)
        self.assertIn("deleted", out)


class TestEnvironmentList(unittest.TestCase):
    def test_list_prints_files(self):
        client = MagicMock()
        client.environments.list.return_value = [_env("ygg-1.0-py312")]
        rc, out = _run(["environment", "list"], client)
        self.assertEqual(rc, 0)
        self.assertIn("ygg-1.0-py312.yml", out)

    def test_list_empty_returns_one(self):
        client = MagicMock()
        client.environments.list.return_value = []
        rc, _ = _run(["env", "list"], client)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
