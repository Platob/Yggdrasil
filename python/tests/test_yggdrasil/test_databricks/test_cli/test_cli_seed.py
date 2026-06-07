"""Dispatch tests for ``ygg databricks seed`` — wheels/environments via the services."""
from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main

REQ = "/Workspace/Shared/environment/ygg/ygg-1.0-py3X.requirements.txt"


def _wheel():
    return types.SimpleNamespace(path="/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl", version="1.0")


def _env():
    return types.SimpleNamespace(
        name="ygg-1.0-py3X", dependencies=["a", "b"],
        serverless="/Workspace/Shared/environment/ygg/ygg-1.0-py3X.yml", cluster=REQ,
    )


def _client_with(*, user="me@co.com", warehouses=None, default_wh=...):
    client = MagicMock()
    client.workspace_client.return_value.current_user.me.return_value.user_name = user
    client.base_url = "https://ws.example.com"
    client.catalog_name = "main"
    client.schema_name = "default"
    client.get_workspace_id.return_value = 123
    client.warehouses.list_warehouses.return_value = iter(warehouses or [])
    if default_wh is not ...:
        client.warehouses.find_default.return_value = default_wh
    # wheels / environments services — present by default (override per test).
    client.wheels.get.return_value = _wheel()
    client.wheels.create.return_value = [_wheel()]
    client.environments.get.return_value = _env()
    client.environments.create.return_value = _env()
    pool = MagicMock()
    pool.instance_pool_name = "Yggdrasil Light"
    pool.node_type_id = "r5d.xlarge"
    pool.instance_pool_id = "pool-light"
    client.compute.instance_pools.seed_default_pools.return_value = [pool]
    client.compute.instance_pools.find.return_value = pool
    return client


def _seed(argv, client):
    with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
         patch("yggdrasil.cli.style.print_logo"), \
         patch("yggdrasil.databricks.assistant.deploy",
               return_value={"uploaded": [], "missing": [], "api": None}), \
         contextlib.redirect_stdout(io.StringIO()):
        return main(argv)


class TestSeedHelp(unittest.TestCase):
    def test_seed_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["seed", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestSeedCheck(unittest.TestCase):
    def test_check_all_present_returns_zero(self):
        wh = MagicMock(); wh.warehouse_name = "Starter"; wh.warehouse_id = "abc"
        client = _client_with(warehouses=[wh])
        self.assertEqual(_seed(["seed", "--check"], client), 0)

    def test_check_missing_environment_returns_one(self):
        wh = MagicMock(); wh.warehouse_name = "Starter"; wh.warehouse_id = "abc"
        client = _client_with(warehouses=[wh])
        client.environments.get.return_value = None
        self.assertEqual(_seed(["seed", "--check"], client), 1)

    def test_check_missing_wheel_and_no_warehouse_returns_one(self):
        client = _client_with(warehouses=[])
        client.wheels.get.return_value = None
        client.environments.get.return_value = None
        self.assertEqual(_seed(["seed", "--check"], client), 1)

    def test_check_missing_cluster_returns_one(self):
        wh = MagicMock(); wh.warehouse_name = "Starter"; wh.warehouse_id = "abc"
        client = _client_with(warehouses=[wh])
        client.compute.clusters.find_cluster.return_value = None
        rc = _seed(["seed", "--check"], client)
        self.assertEqual(rc, 1)
        client.compute.clusters.all_purpose_cluster.assert_not_called()

    def test_check_missing_pools_returns_one(self):
        wh = MagicMock(); wh.warehouse_name = "Starter"; wh.warehouse_id = "abc"
        client = _client_with(warehouses=[wh])
        client.compute.instance_pools.find.return_value = None
        self.assertEqual(_seed(["seed", "--check"], client), 1)

    def test_check_never_provisions(self):
        client = _client_with(warehouses=[])
        client.wheels.get.return_value = None
        client.environments.get.return_value = None
        _seed(["seed", "--check"], client)
        client.wheels.create.assert_not_called()
        client.environments.create.assert_not_called()
        client.warehouses.find_default.assert_not_called()
        client.compute.instance_pools.seed_default_pools.assert_not_called()


class TestSeedProvision(unittest.TestCase):
    def test_seed_builds_wheel_env_and_default_warehouse(self):
        wh = MagicMock(); wh.warehouse_name = "Starter Serverless"; wh.warehouse_id = "wid"
        wh.state = MagicMock(value="RUNNING")
        client = _client_with(default_wh=wh)
        self.assertEqual(_seed(["seed"], client), 0)
        client.wheels.create.assert_called_once()
        client.environments.create.assert_called_once()       # one Python ([None])
        client.warehouses.find_default.assert_called_once()
        client.compute.instance_pools.seed_default_pools.assert_called_once()

    def test_seed_provisions_default_single_user_cluster(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(user="alice@co.com", default_wh=wh)
        self.assertEqual(_seed(["seed"], client), 0)
        client.compute.instance_pools.find.assert_any_call(name="Yggdrasil Light")
        client.compute.clusters.all_purpose_cluster.assert_called_once_with(
            single_user_name="alice@co.com", instance_pool_id="pool-light",
            python_version=f"3.{sys.version_info.minor}", environment=REQ, wait=False,
        )

    def test_seed_cluster_without_light_pool_uses_standalone(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(user="bob@co.com", default_wh=wh)
        client.compute.instance_pools.find.return_value = None
        self.assertEqual(_seed(["seed"], client), 0)
        client.compute.clusters.all_purpose_cluster.assert_called_once_with(
            single_user_name="bob@co.com", instance_pool_id=None,
            python_version=f"3.{sys.version_info.minor}", environment=REQ, wait=False,
        )

    def test_seed_no_cluster_skips_cluster_step(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        self.assertEqual(_seed(["seed", "--no-cluster"], client), 0)
        client.compute.clusters.all_purpose_cluster.assert_not_called()

    def test_seed_overwrite_rebuilds_and_ends_before_cluster(self):
        from yggdrasil.databricks.wheels.service import SUPPORTED_PYTHONS

        client = _client_with(default_wh=MagicMock())
        self.assertEqual(_seed(["seed", "--mode", "overwrite"], client), 0)
        # overwrite forces a fresh fetch of the wheel + one env per Python, then ends.
        self.assertTrue(client.wheels.create.call_args.kwargs["rebuild"])
        self.assertEqual(client.environments.create.call_count, len(SUPPORTED_PYTHONS))
        self.assertTrue(client.environments.create.call_args.kwargs["rebuild"])
        client.compute.clusters.all_purpose_cluster.assert_not_called()
        client.warehouses.find_default.assert_not_called()

    def test_seed_no_pools_skips_pool_step(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        self.assertEqual(_seed(["seed", "--no-pools"], client), 0)
        client.compute.instance_pools.seed_default_pools.assert_not_called()

    def test_seed_all_versions_builds_every_python_environment(self):
        from yggdrasil.databricks.wheels.service import SUPPORTED_PYTHONS

        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        self.assertEqual(_seed(["seed", "--all-versions"], client), 0)
        self.assertEqual(client.environments.create.call_count, len(SUPPORTED_PYTHONS))

    def test_seed_deploys_assistant_bundle(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.assistant.deploy",
                   return_value={"uploaded": ["x"], "missing": [], "api": None}) as dep, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed"])
        self.assertEqual(rc, 0)
        dep.assert_called_once()
        self.assertFalse(dep.call_args.kwargs.get("check", False))

    def test_seed_no_assistant_skips_assistant_step(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.assistant.deploy") as dep, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--no-assistant"])
        self.assertEqual(rc, 0)
        dep.assert_not_called()

    def test_unreachable_workspace_returns_one_early(self):
        client = MagicMock()
        client.workspace_client.return_value.current_user.me.side_effect = RuntimeError("401")
        self.assertEqual(_seed(["seed"], client), 1)
        client.wheels.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
