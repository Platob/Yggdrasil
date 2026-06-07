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

    # Default instance-pools wiring: seed_default_pools returns a concrete list
    # the seed step can iterate; find() returns a pool so --check sees them
    # present (tests that assert absence override this).
    pool = MagicMock()
    pool.instance_pool_name = "Yggdrasil Light"
    pool.node_type_id = "r5d.xlarge"
    pool.instance_pool_id = "pool-light"
    client.compute.instance_pools.seed_default_pools.return_value = [pool]
    client.compute.instance_pools.find.return_value = pool
    return client


def _env_requirements():
    """The seeded generic-environment requirements path for the local Python."""
    from yggdrasil.databricks.job.wheel import (
        WORKSPACE_ENV_DIR,
        environment_folder,
        ygg_base_environment_name,
    )

    name = ygg_base_environment_name()
    return f"{WORKSPACE_ENV_DIR}/{environment_folder('ygg')}/{name}.requirements.txt"


def _env(python):
    """A fake ``ensure_environment`` descriptor for one Python version."""
    key = "py" + (python or "3X").replace(".", "")
    name = f"ygg-1.0-{key}"
    env_dir = "/Workspace/Shared/environment/ygg"
    return {
        "python": python, "key": key, "env_name": name, "env_dir": env_dir,
        "n_wheels": 2,
        "serverless": f"{env_dir}/{name}.yml",
        "cluster": f"{env_dir}/{name}.requirements.txt",
    }


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

    def test_check_missing_cluster_returns_one(self):
        """Everything else present, but the default single-user cluster is absent."""
        wh = MagicMock(); wh.warehouse_name = "Starter"; wh.warehouse_id = "abc"
        client = _client_with(warehouses=[wh])
        client.compute.clusters.find_cluster.return_value = None   # no default cluster
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.deployed_environments",
                   return_value=["/w/env/yellow.yml", "/w/env/yellow.requirements.txt"]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--check"])
        self.assertEqual(rc, 1)
        client.compute.clusters.all_purpose_cluster.assert_not_called()  # check provisions nothing

    def test_check_missing_pools_returns_one(self):
        """Everything else present, but the default pools are absent → fail."""
        wh = MagicMock(); wh.warehouse_name = "Starter"; wh.warehouse_id = "abc"
        client = _client_with(warehouses=[wh])
        client.compute.instance_pools.find.return_value = None   # no pools
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels",
                   return_value=["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.deployed_environments",
                   return_value=["/Workspace/Shared/environments/ygg-1.0.yml",
                                 "/Workspace/Shared/environments/ygg-1.0.requirements.txt"]), \
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
             patch("yggdrasil.databricks.job.wheel.ensure_environments") as envs, \
             contextlib.redirect_stdout(io.StringIO()):
            main(["seed", "--check"])
        ensure.assert_not_called()
        envs.assert_not_called()
        client.warehouses.find_default.assert_not_called()
        client.compute.instance_pools.seed_default_pools.assert_not_called()


class TestSeedProvision(unittest.TestCase):
    def test_seed_builds_wheel_env_and_default_warehouse(self):
        wh = MagicMock(); wh.warehouse_name = "Starter Serverless"; wh.warehouse_id = "wid"
        wh.state = MagicMock(value="RUNNING")
        client = _client_with(default_wh=wh)
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel",
                   return_value=["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]) as ensure, \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(None)]) as envs, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        # One self-contained base environment built for the local Python
        # (serverless + cluster, wheel binaries under the env's own folder).
        envs.assert_called_once()
        self.assertEqual(envs.call_args.kwargs["versions"], [None])
        client.warehouses.find_default.assert_called_once()
        # The default Light/Medium/Heavy instance pools are provisioned too.
        client.compute.instance_pools.seed_default_pools.assert_called_once()

    def test_seed_provisions_default_single_user_cluster(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(user="alice@co.com", default_wh=wh)
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(None)]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed"])
        self.assertEqual(rc, 0)
        # A default all-purpose cluster is provisioned in single-user (dedicated)
        # mode for the current user, attached to the Light pool, running the
        # seeded generic environment (zero-PyPI requirements), without blocking.
        client.compute.instance_pools.find.assert_any_call(name="Yggdrasil Light")
        client.compute.clusters.all_purpose_cluster.assert_called_once_with(
            single_user_name="alice@co.com", instance_pool_id="pool-light",
            environment=_env_requirements(), wait=False,
        )

    def test_seed_cluster_without_light_pool_uses_standalone(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(user="bob@co.com", default_wh=wh)
        client.compute.instance_pools.find.return_value = None   # Light pool absent
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(None)]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed"])
        self.assertEqual(rc, 0)
        # No pool to attach to → cluster created standalone (instance_pool_id=None),
        # still running the seeded generic environment.
        client.compute.clusters.all_purpose_cluster.assert_called_once_with(
            single_user_name="bob@co.com", instance_pool_id=None,
            environment=_env_requirements(), wait=False,
        )

    def test_seed_no_cluster_skips_cluster_step(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(None)]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--no-cluster"])
        self.assertEqual(rc, 0)
        client.compute.clusters.all_purpose_cluster.assert_not_called()

    def test_seed_overwrite_skips_cluster_step(self):
        from yggdrasil.databricks.job.wheel import SUPPORTED_PYTHONS
        client = _client_with(default_wh=MagicMock())
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheels",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(v) for v in SUPPORTED_PYTHONS]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--mode", "overwrite"])
        self.assertEqual(rc, 0)
        # overwrite mode ends after the env rewrite, before the cluster step.
        client.compute.clusters.all_purpose_cluster.assert_not_called()

    def test_seed_no_pools_skips_pool_step(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(None)]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--no-pools"])
        self.assertEqual(rc, 0)
        client.compute.instance_pools.seed_default_pools.assert_not_called()

    def test_seed_overwrite_skips_pool_step(self):
        from yggdrasil.databricks.job.wheel import SUPPORTED_PYTHONS
        client = _client_with(default_wh=MagicMock())
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheels",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(v) for v in SUPPORTED_PYTHONS]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--mode", "overwrite"])
        self.assertEqual(rc, 0)
        # overwrite mode ends after the env rewrite, before warehouses + pools.
        client.compute.instance_pools.seed_default_pools.assert_not_called()

    def test_seed_all_versions_builds_every_python_environment(self):
        from yggdrasil.databricks.job.wheel import SUPPORTED_PYTHONS
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheels",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]) as ensure, \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(v) for v in SUPPORTED_PYTHONS]) as envs, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--all-versions"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()              # per-Python wheel matrix
        # A self-contained environment per supported Python, built in parallel.
        envs.assert_called_once()
        self.assertEqual(list(envs.call_args.kwargs["versions"]), list(SUPPORTED_PYTHONS))

    def test_seed_overwrite_rebuilds_all_wheels_and_ends(self):
        """overwrite mode forces a from-scratch rebuild (all Pythons + envs),
        rewrites the envs, and ends before the warehouse step."""
        from yggdrasil.databricks.job.wheel import SUPPORTED_PYTHONS
        client = _client_with(default_wh=MagicMock())
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheels",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]) as ensure_all, \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel") as ensure_one, \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(v) for v in SUPPORTED_PYTHONS]) as envs, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--mode", "overwrite"])
        self.assertEqual(rc, 0)
        # All wheels rebuilt: the per-Python matrix (not the single-wheel builder)
        # and a self-contained environment per supported Python, all forced fresh.
        ensure_all.assert_called_once()
        self.assertTrue(ensure_all.call_args.kwargs["rebuild"])
        ensure_one.assert_not_called()
        envs.assert_called_once()
        self.assertTrue(envs.call_args.kwargs["rebuild"])
        self.assertEqual(list(envs.call_args.kwargs["versions"]), list(SUPPORTED_PYTHONS))
        # Ends before the warehouse step.
        client.warehouses.find_default.assert_not_called()

    def test_seed_deploys_assistant_bundle(self):
        """Provision mode deploys the Assistant skills + guidance bundle."""
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        deployed = {
            "uploaded": ["/Workspace/Shared/.ygg/assistant/workspace_instructions.md"],
            "missing": [], "api": "skipped (no public Assistant-settings API in this SDK)",
        }
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(None)]), \
             patch("yggdrasil.databricks.assistant.deploy", return_value=deployed) as dep, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed"])
        self.assertEqual(rc, 0)
        dep.assert_called_once()
        self.assertFalse(dep.call_args.kwargs.get("check", False))

    def test_seed_auto_mode_forwards_mode_and_overwrites_assistant(self):
        from yggdrasil.enums.mode import Mode

        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        deployed = {"uploaded": [], "missing": [], "skipped": [], "api": None}
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]) as one, \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(None)]) as envs, \
             patch("yggdrasil.databricks.assistant.deploy", return_value=deployed) as dep, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed"])                              # auto is the default
        self.assertEqual(rc, 0)
        one.assert_called_once()                             # get-or-create (no rebuild)
        self.assertFalse(one.call_args.kwargs["rebuild"])
        self.assertEqual(envs.call_args.kwargs["mode"], Mode.AUTO)
        # auto create-or-updates the assistant bundle (overwrite=True)
        self.assertTrue(dep.call_args.kwargs["overwrite"])

    def test_seed_append_mode_adds_only_missing(self):
        from yggdrasil.enums.mode import Mode

        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        deployed = {"uploaded": [], "missing": [], "skipped": [], "api": None}
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]) as one, \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(None)]) as envs, \
             patch("yggdrasil.databricks.assistant.deploy", return_value=deployed) as dep, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--mode", "append"])
        self.assertEqual(rc, 0)
        one.assert_called_once()                             # get-or-create, no rebuild
        self.assertFalse(one.call_args.kwargs["rebuild"])
        self.assertEqual(envs.call_args.kwargs["mode"], Mode.APPEND)
        # append only writes assistant files that don't exist yet (overwrite=False)
        self.assertFalse(dep.call_args.kwargs["overwrite"])
        # still reaches the cluster step (append is not the early-exit overwrite)
        client.compute.clusters.all_purpose_cluster.assert_called_once()

    def test_seed_no_assistant_skips_assistant_step(self):
        wh = MagicMock(); wh.warehouse_name = "wh"; wh.warehouse_id = "id"; wh.state = "RUNNING"
        client = _client_with(default_wh=wh)
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel",
                   return_value=["/w/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(None)]), \
             patch("yggdrasil.databricks.assistant.deploy") as dep, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--no-assistant"])
        self.assertEqual(rc, 0)
        dep.assert_not_called()

    def test_check_missing_assistant_returns_one(self):
        """Everything else present, but the Assistant bundle isn't deployed → fail."""
        wh = MagicMock(); wh.warehouse_name = "Starter"; wh.warehouse_id = "abc"
        client = _client_with(warehouses=[wh])
        missing = {
            "uploaded": [],
            "missing": ["/Workspace/Shared/.ygg/assistant/workspace_instructions.md"],
            "api": None,
        }
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels",
                   return_value=["/Workspace/Shared/pypi/ygg/ygg-1.0-py3-none-any.whl"]), \
             patch("yggdrasil.databricks.job.wheel.deployed_environments",
                   return_value=["/w/env/yellow.yml", "/w/env/yellow.requirements.txt"]), \
             patch("yggdrasil.databricks.assistant.deploy", return_value=missing) as dep, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["seed", "--check"])
        self.assertEqual(rc, 1)
        self.assertTrue(dep.call_args.kwargs.get("check"))

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
