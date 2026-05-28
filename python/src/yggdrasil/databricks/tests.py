"""Test infrastructure for Databricks services and resources.

Provides :class:`DatabricksTestCase`, a :class:`unittest.TestCase` base that
hands every test a fully-wired :class:`DatabricksClient` whose SDK handles are
replaced with :class:`unittest.mock.MagicMock` autospec'd against the real
Databricks SDK classes. The base also short-circuits authentication, snaps
process-wide caches that bleed between tests, and exposes convenient
shortcuts for the most-tested services.

Quick start
-----------
::

    from yggdrasil.databricks.tests import DatabricksTestCase

    class TestPoolCRUD(DatabricksTestCase):
        def test_create_calls_sdk(self):
            self.pools_api.create.return_value.instance_pool_id = "pool-1"
            pool = self.instance_pools.create(instance_pool_name="ml")
            self.pools_api.create.assert_called_once()
            self.assertEqual(pool.instance_pool_id, "pool-1")

Why a real ``DatabricksClient``?
--------------------------------
Tests exercise the production wiring end-to-end: the same lazy-property
plumbing, the same ``DatabricksService`` subclass instantiations, the same
caches. Only the *external* SDK boundary is mocked — that is the layer where
network calls would otherwise happen. Anything above it (normalization,
default injection, caching, error handling) executes for real.

Resetting state
---------------
:meth:`setUp` clears the module-level caches that the services use
(``_NAMED_POOLS``, ``NAMED_CLUSTERS``, ``_NAME_ID_CACHE``, etc.) and
:meth:`tearDown` restores ``DatabricksClient.set_current(None)`` so the next
test gets a fresh singleton.

Inheriting
----------
Override :attr:`HOST` to change the workspace URL, or :meth:`make_client`
when a test class needs an extra-customised client (cluster_id, profile, …).
Override :meth:`extra_caches_to_clear` to register additional module-level
caches your service uses.
"""
from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from typing import Any, Iterator, Optional
from unittest.mock import MagicMock, patch

__all__ = ["DatabricksTestCase"]


def _import_sdk():
    """Import the Databricks SDK lazily so this module is importable without it."""
    try:
        from databricks.sdk import AccountClient, WorkspaceClient
    except ImportError as exc:  # pragma: no cover - install-time guard
        raise unittest.SkipTest(
            "databricks-sdk is required for DatabricksTestCase; "
            "install with `pip install ygg[databricks]`."
        ) from exc
    return WorkspaceClient, AccountClient


class DatabricksTestCase(unittest.TestCase):
    """Base class for Databricks service / resource unit tests."""

    #: Workspace host used for the synthetic client.
    HOST: str = os.getenv("DATABRICKS_HOST", "https://test.databricks.net")

    #: Service-account / PAT placeholder used to satisfy ``DatabricksClient``
    #: at construction time. No real authentication is ever performed.
    TOKEN: str = os.getenv("DATABRICKS_TOKEN", "fake-pat-not-a-secret")

    #: Default Unity catalog name handed to tests that need a three-part
    #: identifier without caring which catalog they target. Override on the
    #: subclass when a test suite exercises catalog-specific behavior.
    CATALOG_NAME: str = "trading"

    #: Default Unity schema name. Pairs with :attr:`CATALOG_NAME` for the
    #: ``<catalog>.<schema>.<object>`` shape most Databricks tests build.
    SCHEMA_NAME: str = "unittest"

    # ------------------------------------------------------------------ #
    # setUp / tearDown
    # ------------------------------------------------------------------ #
    def setUp(self) -> None:
        super().setUp()

        from yggdrasil.databricks.client import DatabricksClient

        WorkspaceClient, AccountClient = _import_sdk()

        env = os.getenv("DATABRICKS_HOST")

        if env:
            self.client = DatabricksClient(host=env)
            self.workspace_client = self.client.workspace_client()
            self.account_client = MagicMock(spec=AccountClient)
            self.workspace_config = self.client.workspace_config
            self.account_config = MagicMock()
        else:
            self.client = DatabricksClient(host=self.HOST, token=self.TOKEN, auth_type="pat")
            self.workspace_client = MagicMock(spec=WorkspaceClient)
            self.account_client = MagicMock(spec=AccountClient)
            self.workspace_config = MagicMock()
            self.account_config = MagicMock()

        self._clear_databricks_caches()

        # Inject mocks into the lazy slots so workspace_client() / config
        # short-circuit instead of going through the auth path.
        object.__setattr__(self.client, "_workspace_client", self.workspace_client)
        object.__setattr__(self.client, "_account_client", self.account_client)
        object.__setattr__(self.client, "_workspace_config", self.workspace_config)
        object.__setattr__(self.client, "_account_config", self.account_config)

        # Most service tests assume we're NOT running on a Databricks driver
        # node. Patch the environment probe so callers don't have to.
        self._env_patch = patch(
            "yggdrasil.databricks.client.DatabricksClient.is_in_databricks_environment",
            return_value=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        from yggdrasil.databricks.client import DatabricksClient
        DatabricksClient.set_current(self.client)

    def tearDown(self) -> None:
        from yggdrasil.databricks.client import DatabricksClient

        DatabricksClient.set_current(None)
        self._clear_databricks_caches()
        super().tearDown()

    # ------------------------------------------------------------------ #
    # Hooks for subclasses
    # ------------------------------------------------------------------ #
    def extra_caches_to_clear(self) -> tuple:
        """Return additional ``(module, attribute)`` cache locations to reset.

        Useful when a subclass test exercises a service that keeps its own
        module-level cache. Each entry must point to a dict-like the
        framework can call ``.clear()`` on.
        """
        return ()

    # ------------------------------------------------------------------ #
    # Cache reset
    # ------------------------------------------------------------------ #
    def _clear_databricks_caches(self) -> None:
        from yggdrasil.databricks.catalog.catalog import UCCatalog
        from yggdrasil.databricks.client import DatabricksClient
        from yggdrasil.databricks.cluster import service as _cs
        from yggdrasil.databricks.compute import instance_pool as _ip
        from yggdrasil.databricks.schema.schema import UCSchema
        from yggdrasil.databricks.table.table import Table
        from yggdrasil.databricks.volume.volume import Volume

        # Drop singleton caches so each test gets a fresh client +
        # fresh per-resource state; otherwise mutations on
        # ``client.<service>.defaults`` or cached ``_infos`` /
        # columns bleed across tests.
        DatabricksClient._INSTANCES.clear()
        UCCatalog._INSTANCES.clear()
        UCSchema._INSTANCES.clear()
        Table._INSTANCES.clear()
        Volume._INSTANCES.clear()

        _ip._NAMED_POOLS.clear()
        _ip._NAME_ID_CACHE.clear()
        _cs.NAMED_CLUSTERS.clear()
        _cs.NAME_ID_CACHE.clear()
        _cs._SPARK_VERSIONS_CACHE.clear()

        for cache in self.extra_caches_to_clear():
            try:
                cache.clear()
            except AttributeError:
                pass

    # ------------------------------------------------------------------ #
    # Service shortcuts (real services bound to the mock client)
    # ------------------------------------------------------------------ #
    @property
    def compute(self):
        return self.client.compute

    @property
    def clusters(self):
        return self.client.compute.clusters

    @property
    def instance_pools(self):
        return self.client.compute.instance_pools

    @property
    def secrets(self):
        return self.client.secrets

    @property
    def iam(self):
        return self.client.iam

    @property
    def sql(self):
        return self.client.sql

    @property
    def warehouses(self):
        return self.client.warehouses

    # ------------------------------------------------------------------ #
    # SDK-side mock shortcuts (where the network would normally go)
    # ------------------------------------------------------------------ #
    @property
    def clusters_api(self):
        """Autospec'd mock of ``workspace_client.clusters``."""
        return self.workspace_client.clusters

    @property
    def libraries_api(self):
        """Autospec'd mock of ``workspace_client.libraries``."""
        return self.workspace_client.libraries

    @property
    def pools_api(self):
        """Autospec'd mock of ``workspace_client.instance_pools``."""
        return self.workspace_client.instance_pools

    @property
    def command_execution_api(self):
        """Autospec'd mock of ``workspace_client.command_execution``."""
        return self.workspace_client.command_execution

    # ------------------------------------------------------------------ #
    # Test helpers
    # ------------------------------------------------------------------ #
    @contextmanager
    def in_databricks_environment(self, value: bool = True) -> Iterator[None]:
        """Temporarily flip the result of ``is_in_databricks_environment``.

        Default ``False`` is set by :meth:`setUp`; use this context manager to
        exercise the in-runtime branch of any service or decorator::

            with self.in_databricks_environment():
                self.assertEqual(pool.run(func, 1, 2), 3)  # ran locally
        """
        self._env_patch.stop()
        new_patch = patch(
            "yggdrasil.databricks.client.DatabricksClient.is_in_databricks_environment",
            return_value=value,
        )
        new_patch.start()
        try:
            yield
        finally:
            new_patch.stop()
            self._env_patch = patch(
                "yggdrasil.databricks.client.DatabricksClient.is_in_databricks_environment",
                return_value=False,
            )
            self._env_patch.start()

    def make_instance_pool_details(
        self,
        *,
        instance_pool_id: str = "pool-test-1",
        instance_pool_name: str = "test-pool",
        node_type_id: str = "rd-fleet.xlarge",
        state: Optional[Any] = None,
        **overrides: Any,
    ):
        """Build a populated :class:`GetInstancePool` for SDK mock returns."""
        from databricks.sdk.service.compute import GetInstancePool, InstancePoolState

        return GetInstancePool(
            instance_pool_id=instance_pool_id,
            instance_pool_name=instance_pool_name,
            node_type_id=node_type_id,
            state=state if state is not None else InstancePoolState.ACTIVE,
            **overrides,
        )

    def make_cluster_details(
        self,
        *,
        cluster_id: str = "cluster-test-1",
        cluster_name: str = "test-cluster",
        spark_version: str = "15.4.x-scala2.12",
        node_type_id: str = "rd-fleet.xlarge",
        state: Optional[Any] = None,
        **overrides: Any,
    ):
        """Build a populated :class:`ClusterDetails` for SDK mock returns."""
        from databricks.sdk.service.compute import ClusterDetails, State

        return ClusterDetails(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            spark_version=spark_version,
            node_type_id=node_type_id,
            state=state if state is not None else State.RUNNING,
            **overrides,
        )
