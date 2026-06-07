"""Dispatch tests for ``ygg databricks wheel`` (mocked ``dbc.wheels`` service)."""
from __future__ import annotations

import contextlib
import io
import types
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


def _wheel(path):
    """A ``Wheel``-shaped handle (the CLI reads ``.path``)."""
    return types.SimpleNamespace(path=path)


class TestWheelHelp(unittest.TestCase):
    def test_wheel_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["wheel", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_wheel_build_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["wheel", "build", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestWheelBuild(unittest.TestCase):
    def test_build_prints_paths(self):
        client = MagicMock()
        client.wheels.build.return_value = ["/tmp/dist/mypkg-1.0-py3-none-any.whl"]
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(buf):
            rc = main(["wheel", "build", "mypkg", "--no-deps", "--out-dir", "/tmp/dist"])
        self.assertEqual(rc, 0)
        client.wheels.build.assert_called_once()
        self.assertEqual(client.wheels.build.call_args.args[0], "mypkg")
        self.assertTrue(client.wheels.build.call_args.kwargs["no_deps"])
        self.assertEqual(client.wheels.build.call_args.kwargs["dest_dir"], "/tmp/dist")
        self.assertIn("mypkg-1.0-py3-none-any.whl", buf.getvalue())

    def test_build_all_versions(self):
        client = MagicMock()
        client.wheels.build.return_value = ["/tmp/dist/ygg-1.0-py3-none-any.whl"]
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["wheel", "build", "ygg", "--all-versions", "--extra", "databricks"])
        self.assertEqual(rc, 0)
        self.assertEqual(client.wheels.build.call_args.args[0], "ygg")
        self.assertTrue(client.wheels.build.call_args.kwargs["all_versions"])
        self.assertEqual(client.wheels.build.call_args.kwargs["extras"], ("databricks",))


class TestWheelUpload(unittest.TestCase):
    def test_upload_pushes_each_wheel(self):
        client = MagicMock()
        client.wheels.upload.side_effect = lambda w, **k: _wheel(f"/Workspace/Shared/pypi/{w.split('/')[-1]}")
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(buf):
            rc = main(["wheel", "upload", "a.whl", "dir/b.whl"])
        self.assertEqual(rc, 0)
        self.assertEqual(client.wheels.upload.call_count, 2)
        out = buf.getvalue()
        self.assertIn("/Workspace/Shared/pypi/a.whl", out)
        self.assertIn("/Workspace/Shared/pypi/b.whl", out)


class TestWheelDefault(unittest.TestCase):
    def test_bare_wheel_builds_and_uploads_ygg(self):
        client = MagicMock()
        client.wheels.deploy_ygg.return_value = [_wheel("/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl")]
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["wheel"])
        self.assertEqual(rc, 0)
        client.wheels.deploy_ygg.assert_called_once()
        self.assertFalse(client.wheels.deploy_ygg.call_args.kwargs["all_versions"])

    def test_bare_wheel_all_versions(self):
        client = MagicMock()
        client.wheels.deploy_ygg.return_value = [_wheel("/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl")]
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["wheel", "--all-versions"])
        self.assertEqual(rc, 0)
        self.assertTrue(client.wheels.deploy_ygg.call_args.kwargs["all_versions"])

    def test_deploy_defaults_package_to_ygg(self):
        client = MagicMock()
        client.wheels.deploy.return_value = [_wheel("/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl")]
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["wheel", "deploy"])
        self.assertEqual(rc, 0)
        self.assertEqual(client.wheels.deploy.call_args.args[0], "ygg")


class TestWheelDeploy(unittest.TestCase):
    def test_deploy_builds_and_uploads_package(self):
        client = MagicMock()
        client.wheels.deploy.return_value = [_wheel("/Workspace/Shared/pypi/pkg/pkg-1.0-py3-none-any.whl")]
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["wheel", "deploy", "mypkg", "--no-deps", "--extra", "databricks"])
        self.assertEqual(rc, 0)
        client.wheels.deploy.assert_called_once()
        self.assertEqual(client.wheels.deploy.call_args.args[0], "mypkg")
        self.assertTrue(client.wheels.deploy.call_args.kwargs["no_deps"])
        self.assertEqual(client.wheels.deploy.call_args.kwargs["extras"], ("databricks",))

    def test_deploy_all_versions(self):
        client = MagicMock()
        client.wheels.deploy.return_value = [_wheel("/Workspace/Shared/pypi/pkg/pkg-1.0-py3-none-any.whl")]
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["wheel", "deploy", "mypkg", "--all-versions"])
        self.assertEqual(rc, 0)
        self.assertEqual(client.wheels.deploy.call_args.args[0], "mypkg")
        self.assertTrue(client.wheels.deploy.call_args.kwargs["all_versions"])


class TestWheelList(unittest.TestCase):
    def test_list_distributions_when_no_package(self):
        client = MagicMock()
        client.wheels.list.return_value = ["ygg", "pkg"]   # distribution folders
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(buf):
            rc = main(["wheel", "list"])
        self.assertEqual(rc, 0)
        self.assertIsNone(client.wheels.list.call_args.args[0])  # browse mode
        self.assertIn("ygg/", buf.getvalue())
        self.assertIn("pkg/", buf.getvalue())

    def test_list_wheels_for_a_package(self):
        client = MagicMock()
        client.wheels.list.return_value = [_wheel("/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl")]
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(buf):
            rc = main(["wheel", "list", "yggdrasil"])
        self.assertEqual(rc, 0)
        self.assertEqual(client.wheels.list.call_args.args[0], "yggdrasil")
        self.assertIn("/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
