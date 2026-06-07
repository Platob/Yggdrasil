"""Dispatch tests for ``ygg databricks deploy`` (mocked wheel/env machinery).

``deploy`` deploys the current project: build wheel + environment, then
provision the default warehouse and cluster. (The ygg image and arbitrary
package wheels/environments live under ``ygg databricks wheel`` /
``ygg databricks environment``.)
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


class TestDeployHelp(unittest.TestCase):
    def test_deploy_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["deploy", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestDeployProject(unittest.TestCase):
    """Bare ``ygg databricks deploy [path]`` deploys the current project:
    wheel + environment, then the default warehouse and cluster."""

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

    def test_bare_deploy_builds_project_warehouse_and_cluster(self):
        client = MagicMock()
        client.workspace_client.return_value.current_user.me.return_value.user_name = "me@co.com"
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_project_environment",
                   return_value=self._info()) as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "/tmp/myproj", "--extra", "databricks"])
        self.assertEqual(rc, 0)
        # discovered + built from the given path, with the requested extra
        self.assertEqual(ensure.call_args.args[1], "/tmp/myproj")
        self.assertEqual(ensure.call_args.kwargs["extras"], ("databricks",))
        # default serverless warehouse named for the project
        wh = client.warehouses.create_or_update
        wh.assert_called_once()
        self.assertEqual(wh.call_args.kwargs["name"], "myproj")
        self.assertTrue(wh.call_args.kwargs["enable_serverless_compute"])
        # default cluster created, named for the project, installing its env config
        create = client.compute.clusters.all_purpose_cluster
        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs["name"], "myproj")
        self.assertEqual(create.call_args.kwargs["single_user_name"], "me@co.com")
        self.assertEqual(
            create.call_args.kwargs["environment"],
            "/Workspace/Shared/environment/myproj/myproj-0.1.0-py311.requirements.txt",
        )

    def test_no_cluster_skips_cluster_creation(self):
        client = MagicMock()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_project_environment",
                   return_value=self._info()) as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "--no-cluster", "--bundle"])
        self.assertEqual(rc, 0)
        self.assertTrue(ensure.call_args.kwargs["bundle"])
        client.compute.clusters.all_purpose_cluster.assert_not_called()
        client.warehouses.create_or_update.assert_called_once()  # warehouse still built

    def test_no_warehouse_skips_warehouse_creation(self):
        client = MagicMock()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_project_environment",
                   return_value=self._info()), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "--no-warehouse"])
        self.assertEqual(rc, 0)
        client.warehouses.create_or_update.assert_not_called()
        client.compute.clusters.all_purpose_cluster.assert_called_once()  # cluster still built

    def test_mode_threaded_into_deploy(self):
        from yggdrasil.enums.mode import Mode

        client = MagicMock()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_project_environment",
                   return_value=self._info()) as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "--mode", "append"])
        self.assertEqual(rc, 0)
        self.assertEqual(ensure.call_args.kwargs["mode"], Mode.APPEND)
        # append → get-or-create the cluster (never the OVERWRITE update path)
        client.compute.clusters.all_purpose_cluster.assert_called_once()
        client.compute.clusters.find_cluster.assert_not_called()

    def test_overwrite_updates_existing_cluster(self):
        client = MagicMock()
        client.workspace_client.return_value.current_user.me.return_value.user_name = "me@co.com"
        existing = MagicMock()
        client.compute.clusters.find_cluster.return_value = existing
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_project_environment",
                   return_value=self._info()), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "--mode", "overwrite"])
        self.assertEqual(rc, 0)
        # overwrite → update the existing cluster's libraries (no fresh create)
        client.compute.clusters.find_cluster.assert_called_once()
        existing.update.assert_called_once()
        libs = existing.update.call_args.kwargs["libraries"]
        self.assertIn(self._info()["cluster"], libs)
        client.compute.clusters.all_purpose_cluster.assert_not_called()


if __name__ == "__main__":
    unittest.main()
