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
    def test_ygg_default_uses_matrix_builder(self):
        # Default builds every supported Python (matrix builder).
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheels") as ensure:
            ensure.return_value = ["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]
            rc = main(["deploy", "ygg", "--rebuild"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        self.assertTrue(ensure.call_args.kwargs["rebuild"])

    def test_ygg_current_uses_single_builder(self):
        # ``--current`` narrows to the local interpreter's Python.
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel") as ensure:
            ensure.return_value = ["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]
            rc = main(["deploy", "ygg", "--current"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        self.assertFalse(ensure.call_args.kwargs["rebuild"])


class TestDeployWheel(unittest.TestCase):
    def test_wheel_current_uses_single_builder(self):
        # ``--current`` builds the local Python only and honors --no-deps/--extra.
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.databricks.job.wheel.ensure_wheel") as ensure:
            ensure.return_value = ["/Workspace/Shared/pypi/pkg/pkg-1.0-py3-none-any.whl"]
            rc = main(["deploy", "wheel", "mypkg", "--current", "--no-deps", "--extra", "databricks"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        args, kwargs = ensure.call_args
        self.assertEqual(args[1], "mypkg")
        self.assertTrue(kwargs["no_deps"])
        self.assertEqual(kwargs["extras"], ("databricks",))

    def test_wheel_default_uses_matrix_builder(self):
        # Default builds every supported Python (matrix builder).
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.databricks.job.wheel.ensure_wheels") as ensure:
            ensure.return_value = ["/Workspace/Shared/pypi/pkg/pkg-1.0-py3-none-any.whl"]
            rc = main(["deploy", "wheel", "mypkg"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        self.assertEqual(ensure.call_args.args[1], "mypkg")


class TestDeployEnvironment(unittest.TestCase):
    def test_environment_current_prints_job_environment_json(self):
        env = MagicMock()
        env.as_dict.return_value = {"environment_key": "default", "spec": {"environment_version": "5"}}
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ygg_environment", return_value=env) as build, \
             contextlib.redirect_stdout(buf):
            rc = main(["deploy", "environment", "--current", "--key", "default"])
        self.assertEqual(rc, 0)
        build.assert_called_once()
        self.assertEqual(build.call_args.kwargs["environment_key"], "default")
        self.assertEqual(json.loads(buf.getvalue())["environment_key"], "default")

    def test_environment_default_lists_configs(self):
        e1, e2 = MagicMock(), MagicMock()
        e1.as_dict.return_value = {"environment_key": "default"}
        e2.as_dict.return_value = {"environment_key": "py311"}
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ygg_environments", return_value=[e1, e2]) as build, \
             contextlib.redirect_stdout(buf):
            rc = main(["deploy", "environment"])
        self.assertEqual(rc, 0)
        build.assert_called_once()
        self.assertEqual([e["environment_key"] for e in json.loads(buf.getvalue())], ["default", "py311"])


class TestDeployProject(unittest.TestCase):
    def _info(self):
        return {
            "name": "myproj", "version": "0.1.0",
            "env_name": "myproj-0.1.0-py311", "env_dir": "/Workspace/Shared/environment/myproj",
            "dependencies": ["/Workspace/Shared/pypi/myproj/myproj-0.1.0-py3-none-any.whl", "polars"],
            "n_wheels": 2,
            "serverless": "/Workspace/Shared/environment/myproj/myproj-0.1.0-py311.yml",
            "cluster": "/Workspace/Shared/environment/myproj/myproj-0.1.0-py311.requirements.txt",
            "requires_python": ">=3.10",
        }

    def test_project_builds_env_and_creates_cluster(self):
        client = MagicMock()
        client.workspace_client.return_value.current_user.me.return_value.user_name = "me@co.com"
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_project_environment",
                   return_value=self._info()) as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "project", "/tmp/myproj", "--extra", "databricks"])
        self.assertEqual(rc, 0)
        # discovered + built from the given path, with the requested extra
        self.assertEqual(ensure.call_args.args[1], "/tmp/myproj")
        self.assertEqual(ensure.call_args.kwargs["extras"], ("databricks",))
        # default cluster created, named for the project, installing its requirements
        create = client.compute.clusters.all_purpose_cluster
        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs["name"], "myproj")
        self.assertEqual(create.call_args.kwargs["single_user_name"], "me@co.com")
        self.assertEqual(
            create.call_args.kwargs["environment"],
            "/Workspace/Shared/environment/myproj/myproj-0.1.0-py311.requirements.txt",
        )

    def test_project_no_cluster_skips_cluster_creation(self):
        client = MagicMock()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_project_environment",
                   return_value=self._info()) as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "project", "--no-cluster", "--bundle"])
        self.assertEqual(rc, 0)
        self.assertTrue(ensure.call_args.kwargs["bundle"])
        client.compute.clusters.all_purpose_cluster.assert_not_called()

    def test_project_mode_threaded_into_deploy(self):
        from yggdrasil.enums.mode import Mode

        client = MagicMock()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_project_environment",
                   return_value=self._info()) as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "project", "--mode", "append"])
        self.assertEqual(rc, 0)
        self.assertEqual(ensure.call_args.kwargs["mode"], Mode.APPEND)
        # append → get-or-create the cluster (never the OVERWRITE update path)
        client.compute.clusters.all_purpose_cluster.assert_called_once()
        client.compute.clusters.find_cluster.assert_not_called()

    def test_project_overwrite_updates_existing_cluster(self):
        client = MagicMock()
        client.workspace_client.return_value.current_user.me.return_value.user_name = "me@co.com"
        existing = MagicMock()
        client.compute.clusters.find_cluster.return_value = existing
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_project_environment",
                   return_value=self._info()), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "project", "--mode", "overwrite"])
        self.assertEqual(rc, 0)
        # overwrite → update the existing cluster's libraries (no fresh create)
        client.compute.clusters.find_cluster.assert_called_once()
        existing.update.assert_called_once()
        libs = existing.update.call_args.kwargs["libraries"]
        self.assertIn(self._info()["cluster"], libs)
        client.compute.clusters.all_purpose_cluster.assert_not_called()


class TestDeployDefault(unittest.TestCase):
    def test_bare_deploy_ships_wheels_then_environments(self):
        # Default bare deploy ships every supported Python: matrix wheel builder
        # then matrix JobEnvironments.
        e1 = MagicMock(); e1.as_dict.return_value = {"environment_key": "default"}
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheels") as ensure, \
             patch("yggdrasil.databricks.job.wheel.ygg_environments", return_value=[e1]) as build, \
             contextlib.redirect_stdout(io.StringIO()):
            ensure.return_value = ["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]
            rc = main(["deploy"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        build.assert_called_once()
        # The wheel was just built; the environment step must not rebuild it again.
        self.assertFalse(build.call_args.kwargs["rebuild"])

    def test_current_deploy_ships_single_wheel_then_environment(self):
        env = MagicMock(); env.as_dict.return_value = {"environment_key": "default"}
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel") as ensure, \
             patch("yggdrasil.databricks.job.wheel.ygg_environment", return_value=env) as build, \
             contextlib.redirect_stdout(io.StringIO()):
            ensure.return_value = ["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]
            rc = main(["deploy", "--current"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        build.assert_called_once()
        self.assertFalse(build.call_args.kwargs["rebuild"])


if __name__ == "__main__":
    unittest.main()
