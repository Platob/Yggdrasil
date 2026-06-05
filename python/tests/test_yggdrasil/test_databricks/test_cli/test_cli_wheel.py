"""Dispatch tests for ``ygg databricks wheel`` (mocked wheel machinery)."""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


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
    def test_build_is_offline_and_prints_paths(self):
        buf = io.StringIO()
        # build needs no client — patch the client to assert it is never built.
        with patch("yggdrasil.databricks.client.DatabricksClient") as client, \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.build_wheel") as build, \
             contextlib.redirect_stdout(buf):
            build.return_value = ["/tmp/dist/mypkg-1.0-py3-none-any.whl"]
            rc = main(["wheel", "build", "mypkg", "--no-deps", "--out-dir", "/tmp/dist"])
        self.assertEqual(rc, 0)
        client.assert_not_called()
        build.assert_called_once()
        self.assertEqual(build.call_args.args[0], "mypkg")
        self.assertTrue(build.call_args.kwargs["no_deps"])
        self.assertEqual(build.call_args.kwargs["dest_dir"], "/tmp/dist")
        self.assertIn("mypkg-1.0-py3-none-any.whl", buf.getvalue())

    def test_build_all_versions_uses_matrix_builder(self):
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.build_wheels_for_versions") as build, \
             contextlib.redirect_stdout(io.StringIO()):
            build.return_value = ["/tmp/dist/ygg-1.0-py3-none-any.whl"]
            rc = main(["wheel", "build", "ygg", "--all-versions", "--extra", "databricks"])
        self.assertEqual(rc, 0)
        build.assert_called_once()
        self.assertEqual(build.call_args.args[0], "ygg")
        self.assertEqual(build.call_args.kwargs["extras"], ("databricks",))


class TestWheelUpload(unittest.TestCase):
    def test_upload_pushes_each_wheel(self):
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.upload_wheel") as upload, \
             contextlib.redirect_stdout(buf):
            upload.side_effect = lambda c, w, **k: f"/Workspace/Shared/pypi/{w.split('/')[-1]}"
            rc = main(["wheel", "upload", "a.whl", "dir/b.whl"])
        self.assertEqual(rc, 0)
        self.assertEqual(upload.call_count, 2)
        out = buf.getvalue()
        self.assertIn("/Workspace/Shared/pypi/a.whl", out)
        self.assertIn("/Workspace/Shared/pypi/b.whl", out)


class TestWheelDeploy(unittest.TestCase):
    def test_deploy_builds_and_uploads_package(self):
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_wheel") as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            ensure.return_value = ["/Workspace/Shared/pypi/pkg/pkg-1.0-py3-none-any.whl"]
            rc = main(["wheel", "deploy", "mypkg", "--no-deps", "--extra", "databricks"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        self.assertEqual(ensure.call_args.args[1], "mypkg")
        self.assertTrue(ensure.call_args.kwargs["no_deps"])
        self.assertEqual(ensure.call_args.kwargs["extras"], ("databricks",))

    def test_deploy_all_versions_uses_matrix_builder(self):
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_wheels") as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            ensure.return_value = ["/Workspace/Shared/pypi/pkg/pkg-1.0-py3-none-any.whl"]
            rc = main(["wheel", "deploy", "mypkg", "--all-versions"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        self.assertEqual(ensure.call_args.args[1], "mypkg")


class TestWheelList(unittest.TestCase):
    def test_list_distributions_when_no_package(self):
        root = MagicMock()
        root.exists.return_value = True
        dist_a = MagicMock(name="ygg"); dist_a.is_dir.return_value = True; dist_a.name = "ygg"
        dist_b = MagicMock(name="pkg"); dist_b.is_dir.return_value = True; dist_b.name = "pkg"
        root.iterdir.return_value = [dist_a, dist_b]
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.path.DatabricksPath.from_", return_value=root), \
             contextlib.redirect_stdout(buf):
            rc = main(["wheel", "list"])
        self.assertEqual(rc, 0)
        self.assertIn("ygg/", buf.getvalue())
        self.assertIn("pkg/", buf.getvalue())

    def test_list_wheels_for_a_package(self):
        folder = MagicMock()
        folder.exists.return_value = True
        whl = MagicMock(); whl.name = "ygg-1.0-py3-none-any.whl"
        whl.full_path.return_value = "/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"
        other = MagicMock(); other.name = "README.md"
        folder.iterdir.return_value = [whl, other]
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.distribution_for", return_value="ygg"), \
             patch("yggdrasil.databricks.path.DatabricksPath.from_", return_value=folder), \
             contextlib.redirect_stdout(buf):
            rc = main(["wheel", "list", "yggdrasil"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl", out)
        self.assertNotIn("README.md", out)


if __name__ == "__main__":
    unittest.main()
