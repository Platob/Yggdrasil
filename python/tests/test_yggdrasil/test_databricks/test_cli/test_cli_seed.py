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
             patch("yggdrasil.databricks.job.wheel.deployed_environments",
                   return_value=["/Workspace/Shared/environments/yellow.env.yaml",
                                 "/Workspace/Shared/environments/yellow.requirements.txt"]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--check"])
        self.assertEqual(rc, 0)

    def test_check_missing_environment_files_returns_one(self):
        """A deployed wheel but no persisted env/requirements files still fails."""
        wh = MagicMock(); wh.warehouse_name = "Starter"; wh.warehouse_id = "abc"
        client = _client_with(warehouses=[wh])
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels",
                   return_value=["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.deployed_environments", return_value=[]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--check"])
        self.assertEqual(rc, 1)

    def test_check_missing_wheel_and_no_warehouse_returns_one(self):
        client = _client_with(warehouses=[])   # no warehouses
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels", return_value=[]), \
             patch("yggdrasil.databricks.job.wheel.deployed_environments", return_value=[]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--check"])
        self.assertEqual(rc, 1)

    def test_check_never_provisions(self):
        client = _client_with(warehouses=[])
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels", return_value=[]), \
             patch("yggdrasil.databricks.job.wheel.deployed_environments", return_value=[]), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel") as ensure, \
             patch("yggdrasil.databricks.job.wheel.ensure_bundle") as bundle, \
             patch("yggdrasil.databricks.job.wheel.ensure_named_environment") as ene, \
             patch("yggdrasil.databricks.job.wheel.ensure_cluster_requirements") as ecr, \
             contextlib.redirect_stdout(io.StringIO()):
            main(["seed", "--check"])
        ensure.assert_not_called()
        bundle.assert_not_called()
        ene.assert_not_called()
        ecr.assert_not_called()
        client.warehouses.find_default.assert_not_called()


class TestSeedProvision(unittest.TestCase):
    def test_seed_builds_wheel_env_and_default_warehouse(self):
        wh = MagicMock(); wh.warehouse_name = "Starter Serverless"; wh.warehouse_id = "wid"
        wh.state = MagicMock(value="RUNNING")
        client = _client_with(default_wh=wh)
        bundle = [
            "/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl",
            "/Workspace/Shared/pypi/pyarrow/pyarrow-1-cp312-cp312-linux.whl",
        ]
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel",
                   return_value=["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]) as ensure, \
             patch("yggdrasil.databricks.job.wheel.ensure_bundle", return_value=bundle) as build_bundle, \
             patch("yggdrasil.databricks.job.wheel.ensure_named_environment",
                   return_value="/Workspace/Shared/environments/ygg-1.0.yml") as ene, \
             patch("yggdrasil.databricks.job.wheel.ensure_cluster_requirements",
                   return_value="/Workspace/Shared/environments/ygg-1.0.requirements.txt") as ecr, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        build_bundle.assert_called_once()
        # The version-pinned base environment is persisted for both serverless
        # (ygg-<version>.yml) and classic clusters (requirements.txt) from the
        # built wheel bundle — only built wheels in pypi, zero PyPI at runtime.
        ene.assert_called_once()
        ecr.assert_called_once()
        self.assertTrue(ene.call_args.args[1].startswith("ygg-"))
        self.assertTrue(ene.call_args.kwargs["filename"].startswith("ygg-"))
        self.assertTrue(ene.call_args.kwargs["filename"].endswith(".yml"))
        self.assertEqual(ene.call_args.kwargs["dependencies"], bundle)
        self.assertTrue(ecr.call_args.args[1].startswith("ygg-"))
        self.assertEqual(ecr.call_args.kwargs["dependencies"], bundle)
        client.warehouses.find_default.assert_called_once()

    def test_seed_all_versions_uses_matrix_builders(self):
        from yggdrasil.databricks.job.wheel import SUPPORTED_PYTHONS
        n = len(SUPPORTED_PYTHONS)
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheels",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]) as ensure, \
             patch("yggdrasil.databricks.job.wheel.ensure_bundle",
                   return_value=["/w/pypi/ygg/ygg-1.0-py3-none-any.whl"]) as build_bundle, \
             patch("yggdrasil.databricks.job.wheel.ensure_named_environment",
                   return_value="/w/env/ygg-1.0-py312.yml") as ene, \
             patch("yggdrasil.databricks.job.wheel.ensure_cluster_requirements",
                   return_value="/w/env/ygg-1.0-py312.requirements.txt") as ecr, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--all-versions"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()              # per-Python wheel matrix
        # One zero-PyPI bundle + serverless/cluster env pair per supported Python.
        self.assertEqual(build_bundle.call_count, n)
        self.assertEqual(ene.call_count, n)
        self.assertEqual(ecr.call_count, n)
        # Each bundle is pinned to a distinct Python.
        pythons = [c.kwargs["python"] for c in build_bundle.call_args_list]
        self.assertEqual(sorted(pythons), sorted(SUPPORTED_PYTHONS))

    def test_seed_overwrite_rebuilds_all_wheels_and_ends(self):
        """--overwrite forces a from-scratch rebuild (all Pythons + bundle),
        rewrites the env, and ends before the warehouse step."""
        from yggdrasil.databricks.job.wheel import SUPPORTED_PYTHONS
        n = len(SUPPORTED_PYTHONS)
        client = _client_with(default_wh=MagicMock())
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheels",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]) as ensure_all, \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel") as ensure_one, \
             patch("yggdrasil.databricks.job.wheel.ensure_bundle",
                   return_value=["/w/pypi/ygg/ygg-1.0-py3-none-any.whl"]) as build_bundle, \
             patch("yggdrasil.databricks.job.wheel.ensure_named_environment",
                   return_value="/w/env/ygg-1.0-py312.yml") as ene, \
             patch("yggdrasil.databricks.job.wheel.ensure_cluster_requirements",
                   return_value="/w/env/ygg-1.0-py312.requirements.txt") as ecr, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--overwrite"])
        self.assertEqual(rc, 0)
        # All wheels rebuilt: the per-Python matrix (not the single-wheel builder)
        # and a zero-PyPI bundle + env pair per supported Python, all forced fresh.
        ensure_all.assert_called_once()
        self.assertTrue(ensure_all.call_args.kwargs["rebuild"])
        ensure_one.assert_not_called()
        self.assertEqual(build_bundle.call_count, n)
        self.assertTrue(all(c.kwargs["rebuild"] for c in build_bundle.call_args_list))
        self.assertEqual(ene.call_count, n)
        self.assertEqual(ecr.call_count, n)
        # Ends before the warehouse step.
        client.warehouses.find_default.assert_not_called()

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
