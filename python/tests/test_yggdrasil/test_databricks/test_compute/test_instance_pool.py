"""Unit tests for the Databricks instance-pool service and resource.

Uses :class:`yggdrasil.databricks.tests.DatabricksTestCase` to mock the SDK
boundary while exercising the real :class:`InstancePools` /
:class:`InstancePool` wiring (default injection, caching, lifecycle).
"""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.compute import (
    CreateInstancePoolResponse,
    InstancePoolAndStats,
    InstancePoolState,
)

from yggdrasil.databricks.compute.instance_pool import (
    DEFAULT_POOL_NAME,
    InstancePool,
    InstancePoolDefaults,
)
from yggdrasil.databricks.tests import DatabricksTestCase


class TestInstancePoolDefaults(DatabricksTestCase):
    """Service-level auto-configuration via :class:`InstancePoolDefaults`."""

    def test_defaults_attached_to_service(self):
        self.assertIsInstance(self.instance_pools.defaults, InstancePoolDefaults)
        self.assertEqual(self.instance_pools.defaults.pool_name, DEFAULT_POOL_NAME)
        self.assertEqual(self.instance_pools.defaults.max_capacity, 10)

    def test_defaults_override_per_call(self):
        self.instance_pools.defaults = replace(
            self.instance_pools.defaults,
            max_capacity=42,
            min_idle_instances=3,
        )
        self.assertEqual(self.instance_pools.defaults.max_capacity, 42)
        self.assertEqual(self.instance_pools.defaults.min_idle_instances, 3)


class TestInstancePoolsCreate(DatabricksTestCase):
    """End-to-end create path: default injection, SDK arguments, return shape."""

    def setUp(self):
        super().setUp()
        # Disable preload to keep the create assertion simple — preloading
        # would otherwise call the (mocked) clusters API for spark versions.
        self.instance_pools.defaults = replace(
            self.instance_pools.defaults,
            preload_local_python_runtime=False,
        )
        self.pools_api.create.return_value = CreateInstancePoolResponse(
            instance_pool_id="pool-1",
        )
        self.pools_api.get.return_value = self.make_instance_pool_details(
            instance_pool_id="pool-1",
            instance_pool_name="ml-pool",
        )

    def test_create_injects_defaults(self):
        pool = self.instance_pools.create(instance_pool_name="ml-pool")

        self.assertIsInstance(pool, InstancePool)
        self.assertEqual(pool.instance_pool_id, "pool-1")
        self.assertEqual(pool.instance_pool_name, "ml-pool")

        self.pools_api.create.assert_called_once()
        _, kwargs = self.pools_api.create.call_args
        self.assertEqual(kwargs["instance_pool_name"], "ml-pool")
        self.assertEqual(kwargs["node_type_id"], "rd-fleet.xlarge")
        self.assertEqual(kwargs["idle_instance_autotermination_minutes"], 30)
        self.assertEqual(kwargs["max_capacity"], 10)
        self.assertEqual(kwargs["min_idle_instances"], 0)
        self.assertTrue(kwargs["enable_elastic_disk"])
        # Default tags merged from DatabricksService.default_tags()
        self.assertIn("ServiceName", kwargs.get("custom_tags", {}))

    def test_create_explicit_args_win_over_defaults(self):
        self.instance_pools.create(
            instance_pool_name="big-pool",
            node_type_id="m5.4xlarge",
            max_capacity=100,
            min_idle_instances=4,
            idle_instance_autotermination_minutes=15,
            enable_elastic_disk=False,
            custom_tags={"Team": "ML"},
        )

        _, kwargs = self.pools_api.create.call_args
        self.assertEqual(kwargs["node_type_id"], "m5.4xlarge")
        self.assertEqual(kwargs["max_capacity"], 100)
        self.assertEqual(kwargs["min_idle_instances"], 4)
        self.assertEqual(kwargs["idle_instance_autotermination_minutes"], 15)
        self.assertFalse(kwargs["enable_elastic_disk"])
        self.assertEqual(kwargs["custom_tags"]["Team"], "ML")


class TestInstancePoolsPool(DatabricksTestCase):
    """``pool()`` find-or-create singleton helper."""

    def setUp(self):
        super().setUp()
        self.instance_pools.defaults = replace(
            self.instance_pools.defaults,
            preload_local_python_runtime=False,
        )

    def test_pool_creates_when_missing(self):
        # No pool exists initially — list returns empty
        self.pools_api.list.return_value = iter([])
        self.pools_api.create.return_value = CreateInstancePoolResponse(
            instance_pool_id="pool-new",
        )
        self.pools_api.get.return_value = self.make_instance_pool_details(
            instance_pool_id="pool-new",
            instance_pool_name="ad-hoc",
        )

        pool = self.instance_pools.pool("ad-hoc")

        self.assertEqual(pool.instance_pool_id, "pool-new")
        self.pools_api.create.assert_called_once()

    def test_pool_returns_existing(self):
        existing = InstancePoolAndStats(
            instance_pool_id="pool-existing",
            instance_pool_name="already-there",
            node_type_id="rd-fleet.xlarge",
        )
        self.pools_api.list.return_value = iter([existing])

        pool = self.instance_pools.pool("already-there")

        self.assertEqual(pool.instance_pool_id, "pool-existing")
        self.pools_api.create.assert_not_called()

    def test_pool_caches_named_singleton(self):
        existing = InstancePoolAndStats(
            instance_pool_id="pool-cached",
            instance_pool_name="cached-pool",
            node_type_id="rd-fleet.xlarge",
        )
        self.pools_api.list.return_value = iter([existing])

        first = self.instance_pools.pool("cached-pool")
        second = self.instance_pools.pool("cached-pool")

        self.assertIs(first, second)
        # list() called only on the first lookup
        self.assertEqual(self.pools_api.list.call_count, 1)

    def test_default_pool_uses_defaults_pool_name(self):
        self.instance_pools.defaults = replace(
            self.instance_pools.defaults,
            pool_name="MyDefaultPool",
            preload_local_python_runtime=False,
        )
        self.pools_api.list.return_value = iter([])
        self.pools_api.create.return_value = CreateInstancePoolResponse(
            instance_pool_id="pool-default",
        )
        self.pools_api.get.return_value = self.make_instance_pool_details(
            instance_pool_id="pool-default",
            instance_pool_name="MyDefaultPool",
        )

        pool = self.instance_pools.default_pool()

        self.assertEqual(pool.instance_pool_name, "MyDefaultPool")
        _, kwargs = self.pools_api.create.call_args
        self.assertEqual(kwargs["instance_pool_name"], "MyDefaultPool")


class TestInstancePoolsFind(DatabricksTestCase):
    """``find()`` lookup by id / name / cache."""

    def test_find_by_id(self):
        details = self.make_instance_pool_details(instance_pool_id="pool-x")
        self.pools_api.get.return_value = details

        pool = self.instance_pools.find(pool_id="pool-x")

        self.assertIsNotNone(pool)
        self.assertEqual(pool.instance_pool_id, "pool-x")
        self.pools_api.get.assert_called_once_with(instance_pool_id="pool-x")

    def test_find_missing_returns_none(self):
        self.pools_api.get.side_effect = ResourceDoesNotExist("not found")

        result = self.instance_pools.find(pool_id="nope")

        self.assertIsNone(result)

    def test_find_missing_raises_when_requested(self):
        self.pools_api.get.side_effect = ResourceDoesNotExist("not found")

        with self.assertRaises(ValueError):
            self.instance_pools.find(pool_id="nope", raise_error=True)

    def test_find_requires_pool_id_or_name(self):
        with self.assertRaises(ValueError):
            self.instance_pools.find()


class TestInstancePoolResourceLifecycle(DatabricksTestCase):
    """``InstancePool`` lifecycle: refresh, update, delete."""

    def _make_pool(self, **overrides):
        details = self.make_instance_pool_details(**overrides)
        return InstancePool(
            service=self.instance_pools,
            instance_pool_id=details.instance_pool_id,
            instance_pool_name=details.instance_pool_name,
            details=details,
        )

    def test_state_reads_from_details(self):
        pool = self._make_pool(state=InstancePoolState.ACTIVE)
        self.assertEqual(pool.state, InstancePoolState.ACTIVE)
        self.assertTrue(pool.is_active)

    def test_refresh_fetches_fresh_details(self):
        pool = self._make_pool(instance_pool_id="pool-r")
        fresh = self.make_instance_pool_details(
            instance_pool_id="pool-r",
            instance_pool_name="renamed",
        )
        self.pools_api.get.return_value = fresh

        pool.refresh()

        self.assertEqual(pool.instance_pool_name, "renamed")
        self.pools_api.get.assert_called_with(instance_pool_id="pool-r")

    def test_delete_calls_sdk_and_drops_named_cache(self):
        from yggdrasil.databricks.compute import instance_pool as _ip

        pool = self._make_pool(instance_pool_name="to-drop")
        _ip._NAMED_POOLS["to-drop"] = pool

        pool.delete()

        self.pools_api.delete.assert_called_once_with(
            instance_pool_id=pool.instance_pool_id,
        )
        self.assertNotIn("to-drop", _ip._NAMED_POOLS)


class TestInstancePoolRun(DatabricksTestCase):
    """``InstancePool.run`` simplest-python-execution entry point."""

    def _make_pool(self):
        details = self.make_instance_pool_details()
        return InstancePool(
            service=self.instance_pools,
            instance_pool_id=details.instance_pool_id,
            instance_pool_name=details.instance_pool_name,
            details=details,
        )

    def test_run_force_local_executes_locally(self):
        pool = self._make_pool()
        result = pool.run(lambda a, b: a + b, 2, 3, force_local=True)
        self.assertEqual(result, 5)

    def test_run_in_databricks_environment_executes_locally(self):
        pool = self._make_pool()
        with self.in_databricks_environment():
            result = pool.run(lambda x: x * x, 7)
        self.assertEqual(result, 49)

    def test_decorate_in_databricks_environment_collapses_to_identity(self):
        pool = self._make_pool()
        with self.in_databricks_environment():
            @pool.decorate
            def double(x):
                return x * 2

            self.assertEqual(double(3), 6)
