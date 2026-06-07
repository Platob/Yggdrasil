"""Dispatch tests for ``ygg databricks deploy`` (mocked services).

``deploy`` deploys the current project via ``dbc.environments.deploy_project``,
then provisions the default warehouse and cluster. (The ygg image and arbitrary
package wheels/environments live under ``ygg databricks wheel`` /
``ygg databricks environment``.)
"""
from __future__ import annotations

import contextlib
import io
import types
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
    wheel + environment (``deploy_project``), then warehouse + cluster."""

    CLUSTER_CFG = "/Workspace/Shared/environment/myproj/myproj-0.1.0-py311.requirements.txt"

    def _client(self):
        """A mock client whose ``environments.deploy_project`` returns a project
        :class:`Environment`-shaped handle."""
        client = MagicMock()
        client.workspace_client.return_value.current_user.me.return_value.user_name = "me@co.com"
        env = types.SimpleNamespace(
            project="myproj", version="0.1.0", name="myproj-0.1.0-py311",
            env_dir="/Workspace/Shared/environment/myproj",
            serverless="/Workspace/Shared/environment/myproj/myproj-0.1.0-py311.yml",
            cluster=self.CLUSTER_CFG,
            dependencies=["/Workspace/Shared/pypi/myproj/myproj-0.1.0-py3-none-any.whl", "polars"],
        )
        client.environments.deploy_project.return_value = env
        return client

    def test_bare_deploy_builds_project_warehouse_and_cluster(self):
        client = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "/tmp/myproj", "--extra", "databricks"])
        self.assertEqual(rc, 0)
        # the project was discovered + built from the given path, with the extra
        dp = client.environments.deploy_project
        dp.assert_called_once()
        self.assertEqual(dp.call_args.args[0], "/tmp/myproj")
        self.assertEqual(dp.call_args.kwargs["extras"], ("databricks",))
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
        self.assertEqual(create.call_args.kwargs["environment"], self.CLUSTER_CFG)

    def test_no_cluster_skips_cluster_creation(self):
        client = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "--no-cluster", "--bundle"])
        self.assertEqual(rc, 0)
        self.assertTrue(client.environments.deploy_project.call_args.kwargs["bundle"])
        client.compute.clusters.all_purpose_cluster.assert_not_called()
        client.warehouses.create_or_update.assert_called_once()  # warehouse still built

    def test_no_warehouse_skips_warehouse_creation(self):
        client = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "--no-warehouse"])
        self.assertEqual(rc, 0)
        client.warehouses.create_or_update.assert_not_called()
        client.compute.clusters.all_purpose_cluster.assert_called_once()  # cluster still built

    def test_mode_threaded_into_deploy(self):
        from yggdrasil.enums.mode import Mode

        client = self._client()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "--mode", "append"])
        self.assertEqual(rc, 0)
        self.assertEqual(client.environments.deploy_project.call_args.kwargs["mode"], Mode.APPEND)
        # append → get-or-create the cluster (never the OVERWRITE update path)
        client.compute.clusters.all_purpose_cluster.assert_called_once()
        client.compute.clusters.find_cluster.assert_not_called()

    def test_overwrite_updates_existing_cluster(self):
        client = self._client()
        existing = MagicMock()
        client.compute.clusters.find_cluster.return_value = existing
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["deploy", "--mode", "overwrite"])
        self.assertEqual(rc, 0)
        # overwrite → update the existing cluster's libraries (no fresh create)
        client.compute.clusters.find_cluster.assert_called_once()
        existing.update.assert_called_once()
        self.assertIn(self.CLUSTER_CFG, existing.update.call_args.kwargs["libraries"])
        client.compute.clusters.all_purpose_cluster.assert_not_called()


if __name__ == "__main__":
    unittest.main()
