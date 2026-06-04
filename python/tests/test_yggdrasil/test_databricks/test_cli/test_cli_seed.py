"""Dispatch tests for ``ygg databricks seed`` (mocked client + wheel machinery)."""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


def _client_with(*, user="me@co.com", warehouses=None, default_wh=...):
    """A MagicMock DatabricksClient wired for the seed config/warehouse probes."""
    client = MagicMock()
    client.workspace_client.return_value.current_user.me.return_value.user_name = user
    client.base_url = "https://ws.example.com"
    client.catalog_name = "main"
    client.schema_name = "default"
    client.get_workspace_id.return_value = 123
    client.warehouses.list_warehouses.return_value = iter(warehouses or [])
    if default_wh is not ...:
        client.warehouses.find_default.return_value = default_wh
    return client


class TestSeedHelp(unittest.TestCase):
    def test_seed_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["seed", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestSeedCheck(unittest.TestCase):
    def test_check_all_present_returns_zero(self):
        wh = MagicMock(); wh.warehouse_name = "Starter"; wh.warehouse_id = "abc"
        client = _client_with(warehouses=[wh])
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels",
                   return_value=["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.ygg_runtime_dependencies", return_value=["pyarrow==1"]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--check"])
        self.assertEqual(rc, 0)

    def test_check_missing_wheel_and_no_warehouse_returns_one(self):
        client = _client_with(warehouses=[])   # no warehouses
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels", return_value=[]), \
             patch("yggdrasil.databricks.job.wheel.ygg_runtime_dependencies", return_value=["pyarrow==1"]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--check"])
        self.assertEqual(rc, 1)

    def test_check_never_provisions(self):
        client = _client_with(warehouses=[])
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels", return_value=[]), \
             patch("yggdrasil.databricks.job.wheel.ygg_runtime_dependencies", return_value=[]), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel") as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            main(["seed", "--check"])
        ensure.assert_not_called()
        client.warehouses.find_default.assert_not_called()


class TestSeedProvision(unittest.TestCase):
    def test_seed_builds_wheel_env_and_default_warehouse(self):
        wh = MagicMock(); wh.warehouse_name = "Starter Serverless"; wh.warehouse_id = "wid"
        wh.state = MagicMock(value="RUNNING")
        client = _client_with(default_wh=wh)
        env = MagicMock()
        env.environment_key = "default"
        env.spec.environment_version = "5"
        env.spec.dependencies = ["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl", "pyarrow==1"]
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel",
                   return_value=["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]) as ensure, \
             patch("yggdrasil.databricks.job.wheel.ygg_environment", return_value=env) as build_env, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        build_env.assert_called_once()
        client.warehouses.find_default.assert_called_once()

    def test_seed_all_versions_uses_matrix_builders(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        e = MagicMock(); e.environment_key = "py311"; e.spec.environment_version = "2"; e.spec.dependencies = []
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheels",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]) as ensure, \
             patch("yggdrasil.databricks.job.wheel.ygg_environments", return_value=[e, e]) as build_envs, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--all-versions"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        build_envs.assert_called_once()

    def test_unreachable_workspace_returns_one_early(self):
        client = MagicMock()
        client.workspace_client.return_value.current_user.me.side_effect = RuntimeError("401")
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel") as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed"])
        self.assertEqual(rc, 1)
        ensure.assert_not_called()   # bailed before touching the wheel step


if __name__ == "__main__":
    unittest.main()
