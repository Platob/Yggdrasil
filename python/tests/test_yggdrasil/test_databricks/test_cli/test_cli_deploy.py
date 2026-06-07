"""Dispatch tests for ``ygg databricks deploy`` — project deploy via ``dbc.environments``."""
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
    CLUSTER_CFG = "/Workspace/Shared/environment/myproj/myproj-0.1.0-py311.requirements.txt"

    def _client(self):
        client = MagicMock()
        client.workspace_client.return_value.current_user.me.return_value.user_name = "me@co.com"
        env = types.SimpleNamespace(
            project="myproj", version="0.1.0", name="myproj-0.1.0-py311",
            serverless="/Workspace/Shared/environment/myproj/myproj-0.1.0-py311.yml",
            cluster=self.CLUSTER_CFG,
            dependencies=["/ws/pypi/myproj/myproj-0.1.0-py3-none-any.whl", "polars"],
        )
        client.environments.create.return_value = env
        return client

    def _run(self, argv, client):
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            return main(argv)

    def test_bare_deploy_builds_project_warehouse_and_cluster(self):
        client = self._client()
        rc = self._run(["deploy", "/tmp/myproj", "--extra", "databricks"], client)
        self.assertEqual(rc, 0)
        create = client.environments.create
        create.assert_called_once()
        self.assertEqual(create.call_args.args[0], "/tmp/myproj")
        self.assertEqual(create.call_args.kwargs["extras"], ("databricks",))
        # default serverless warehouse named for the project
        wh = client.warehouses.create_or_update
        wh.assert_called_once()
        self.assertEqual(wh.call_args.kwargs["name"], "Myproj")  # capitalized project
        self.assertTrue(wh.call_args.kwargs["enable_serverless_compute"])
        # default cluster, named for the project, installing its env config
        cl = client.compute.clusters.all_purpose_cluster
        cl.assert_called_once()
        self.assertEqual(cl.call_args.kwargs["name"], "myproj")
        self.assertEqual(cl.call_args.kwargs["single_user_name"], "me@co.com")
        self.assertEqual(cl.call_args.kwargs["environment"], self.CLUSTER_CFG)

    def test_cwd_default(self):
        client = self._client()
        rc = self._run(["deploy"], client)
        self.assertEqual(rc, 0)
        self.assertEqual(client.environments.create.call_args.args[0], ".")

    def test_no_cluster_skips_cluster(self):
        client = self._client()
        rc = self._run(["deploy", "--no-cluster"], client)
        self.assertEqual(rc, 0)
        client.compute.clusters.all_purpose_cluster.assert_not_called()
        client.warehouses.create_or_update.assert_called_once()

    def test_no_warehouse_skips_warehouse(self):
        client = self._client()
        rc = self._run(["deploy", "--no-warehouse"], client)
        self.assertEqual(rc, 0)
        client.warehouses.create_or_update.assert_not_called()
        client.compute.clusters.all_purpose_cluster.assert_called_once()

    def test_rebuild_updates_existing_cluster(self):
        client = self._client()
        existing = MagicMock()
        client.compute.clusters.find_cluster.return_value = existing
        rc = self._run(["deploy", "--rebuild"], client)
        self.assertEqual(rc, 0)
        self.assertTrue(client.environments.create.call_args.kwargs["rebuild"])
        client.compute.clusters.find_cluster.assert_called_once()
        existing.update.assert_called_once()
        self.assertIn(self.CLUSTER_CFG, existing.update.call_args.kwargs["libraries"])
        client.compute.clusters.all_purpose_cluster.assert_not_called()


if __name__ == "__main__":
    unittest.main()
