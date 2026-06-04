"""Dispatch tests for ``ygg databricks deploy`` (mocked wheel/env machinery)."""
from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


class TestDeployHelp(unittest.TestCase):
    def test_deploy_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["deploy", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestDeployYgg(unittest.TestCase):
    def test_ygg_builds_and_uploads(self):
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel") as ensure:
            ensure.return_value = ["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]
            rc = main(["deploy", "ygg"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        self.assertFalse(ensure.call_args.kwargs["rebuild"])

    def test_ygg_all_versions_uses_matrix_builder(self):
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheels") as ensure:
            ensure.return_value = ["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]
            rc = main(["deploy", "ygg", "--all-versions", "--rebuild"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        self.assertTrue(ensure.call_args.kwargs["rebuild"])


class TestDeployWheel(unittest.TestCase):
    def test_wheel_builds_and_uploads_package(self):
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.databricks.job.wheel.ensure_wheel") as ensure:
            ensure.return_value = ["/Workspace/Shared/pypi/pkg/pkg-1.0-py3-none-any.whl"]
            rc = main(["deploy", "wheel", "mypkg", "--no-deps", "--extra", "databricks"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        args, kwargs = ensure.call_args
        self.assertEqual(args[1], "mypkg")
        self.assertTrue(kwargs["no_deps"])
        self.assertEqual(kwargs["extras"], ("databricks",))

    def test_wheel_all_versions_uses_matrix_builder(self):
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.databricks.job.wheel.ensure_wheels") as ensure:
            ensure.return_value = ["/Workspace/Shared/pypi/pkg/pkg-1.0-py3-none-any.whl"]
            rc = main(["deploy", "wheel", "mypkg", "--all-versions"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        self.assertEqual(ensure.call_args.args[1], "mypkg")


class TestDeployEnvironment(unittest.TestCase):
    def test_environment_prints_job_environment_json(self):
        env = MagicMock()
        env.as_dict.return_value = {"environment_key": "default", "spec": {"environment_version": "5"}}
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ygg_environment", return_value=env) as build, \
             contextlib.redirect_stdout(buf):
            rc = main(["deploy", "environment", "--key", "default"])
        self.assertEqual(rc, 0)
        build.assert_called_once()
        self.assertEqual(build.call_args.kwargs["environment_key"], "default")
        self.assertEqual(json.loads(buf.getvalue())["environment_key"], "default")

    def test_environment_all_versions_lists_configs(self):
        e1, e2 = MagicMock(), MagicMock()
        e1.as_dict.return_value = {"environment_key": "default"}
        e2.as_dict.return_value = {"environment_key": "py311"}
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ygg_environments", return_value=[e1, e2]) as build, \
             contextlib.redirect_stdout(buf):
            rc = main(["deploy", "environment", "--all-versions"])
        self.assertEqual(rc, 0)
        build.assert_called_once()
        self.assertEqual([e["environment_key"] for e in json.loads(buf.getvalue())], ["default", "py311"])


class TestDeployDefault(unittest.TestCase):
    def test_bare_deploy_ships_wheel_then_environment(self):
        env = MagicMock()
        env.as_dict.return_value = {"environment_key": "default"}
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel") as ensure, \
             patch("yggdrasil.databricks.job.wheel.ygg_environment", return_value=env) as build, \
             contextlib.redirect_stdout(io.StringIO()):
            ensure.return_value = ["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]
            rc = main(["deploy"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        build.assert_called_once()
        # The wheel was just built; the environment step must not rebuild it again.
        self.assertFalse(build.call_args.kwargs["rebuild"])


if __name__ == "__main__":
    unittest.main()
