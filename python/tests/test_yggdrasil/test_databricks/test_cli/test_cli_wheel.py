"""Dispatch tests for ``ygg databricks wheel`` — uniform CRUD via ``dbc.wheels``."""
from __future__ import annotations

import contextlib
import io
import types
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


def _wheel(path, dist="ygg", version="1.0"):
    return types.SimpleNamespace(path=path, dist=dist, version=version)


def _run(argv, client):
    buf = io.StringIO()
    with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
         patch("yggdrasil.cli.style.print_logo"), \
         contextlib.redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


class TestWheelHelp(unittest.TestCase):
    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["wheel", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestWheelCreate(unittest.TestCase):
    def test_create_fetches_and_uploads(self):
        client = MagicMock()
        client.wheels.create.return_value = [_wheel("/Workspace/Shared/pypi/mypkg/mypkg-1.0-py3-none-any.whl")]
        rc, out = _run(["wheel", "create", "mypkg", "1.0", "--deps", "--extra", "databricks"], client)
        self.assertEqual(rc, 0)
        client.wheels.create.assert_called_once()
        self.assertEqual(client.wheels.create.call_args.args[0], "mypkg")
        self.assertEqual(client.wheels.create.call_args.args[1], "1.0")
        self.assertTrue(client.wheels.create.call_args.kwargs["deps"])
        self.assertEqual(client.wheels.create.call_args.kwargs["extras"], ("databricks",))
        self.assertIn("mypkg-1.0-py3-none-any.whl", out)

    def test_update_overwrites(self):
        client = MagicMock()
        client.wheels.update.return_value = [_wheel("/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl")]
        rc, out = _run(["wheel", "update", "ygg"], client)
        self.assertEqual(rc, 0)
        client.wheels.update.assert_called_once()
        self.assertIn("updated", out)


class TestWheelFindGet(unittest.TestCase):
    def test_find_builds_on_miss(self):
        client = MagicMock()
        client.wheels.find.return_value = _wheel("/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl")
        rc, out = _run(["wheel", "find", "ygg"], client)
        self.assertEqual(rc, 0)
        self.assertTrue(client.wheels.find.call_args.kwargs["install"])
        self.assertIn("ygg-1.0-py3-none-any.whl", out)

    def test_find_no_install(self):
        client = MagicMock()
        client.wheels.find.return_value = None
        rc, _ = _run(["wheel", "find", "ygg", "--no-install"], client)
        self.assertEqual(rc, 1)
        self.assertFalse(client.wheels.find.call_args.kwargs["install"])

    def test_get_never_builds(self):
        client = MagicMock()
        client.wheels.get.return_value = _wheel("/ws/ygg/ygg-1.0.whl")
        rc, out = _run(["wheel", "get", "ygg"], client)
        self.assertEqual(rc, 0)
        client.wheels.get.assert_called_once()
        self.assertIn("ygg-1.0.whl", out)


class TestWheelDelete(unittest.TestCase):
    def test_delete_removes(self):
        client = MagicMock()
        client.wheels.delete.return_value = [_wheel("/ws/ygg/ygg-1.0.whl")]
        rc, out = _run(["wheel", "delete", "ygg", "1.0"], client)
        self.assertEqual(rc, 0)
        self.assertEqual(client.wheels.delete.call_args.args[0], "ygg")
        self.assertIn("deleted", out)

    def test_delete_nothing_returns_one(self):
        client = MagicMock()
        client.wheels.delete.return_value = []
        rc, _ = _run(["wheel", "rm", "nope"], client)
        self.assertEqual(rc, 1)


class TestWheelList(unittest.TestCase):
    def test_list_distributions(self):
        client = MagicMock()
        client.wheels.list.return_value = ["ygg", "pkg"]
        rc, out = _run(["wheel", "list"], client)
        self.assertEqual(rc, 0)
        self.assertIsNone(client.wheels.list.call_args.args[0])
        self.assertIn("ygg/", out)
        self.assertIn("pkg/", out)

    def test_list_wheels_for_project(self):
        client = MagicMock()
        client.wheels.list.return_value = [_wheel("/ws/ygg/ygg-1.0-py3-none-any.whl")]
        rc, out = _run(["wheel", "list", "ygg"], client)
        self.assertEqual(rc, 0)
        self.assertEqual(client.wheels.list.call_args.args[0], "ygg")
        self.assertIn("/ws/ygg/ygg-1.0-py3-none-any.whl", out)


if __name__ == "__main__":
    unittest.main()
