"""Tests for the :class:`ServerlessCluster` lifecycle overrides.

Serverless compute does not expose the caller-driven start / restart
hooks classic clusters do (it's always-on from the caller's
perspective) and forces dependency pinning through the workspace's
serverless environment spec instead of the per-cluster libraries
API. :class:`ServerlessCluster` encodes that divergence in three
places — these tests pin each one:

- :meth:`start` / :meth:`restart` no-op silently and return ``self``;
- :meth:`install_libraries` raises :class:`NotImplementedError` for
  any non-empty library list (so a misrouted dependency surfaces
  loudly instead of being silently dropped);
- empty / ``None`` library lists short-circuit without raising so
  the helper composes cleanly with ``Clusters.create``'s default
  ``libraries=None`` path.
"""
from __future__ import annotations

import pytest

from yggdrasil.databricks.cluster import Cluster, ServerlessCluster
from yggdrasil.databricks.tests import DatabricksTestCase


class TestServerlessLifecycleOverrides(DatabricksTestCase):

    def _sl(self) -> ServerlessCluster:
        return ServerlessCluster(service=self.clusters, cluster_id="sl-1")

    def test_serverless_subclasses_cluster(self):
        # The executor and other callers branch on ``isinstance(...,
        # ServerlessCluster)``, so this guard pins the inheritance.
        self.assertTrue(issubclass(ServerlessCluster, Cluster))

    def test_start_is_no_op_and_returns_self(self):
        sl = self._sl()
        # The classic cluster path would call the SDK's start endpoint
        # here; the serverless override skips entirely. Hitting the
        # mock SDK would fail the autospec since the cluster has no
        # ``_details`` yet — the no-op is the point.
        self.assertIs(sl.start(wait=False), sl)
        # No SDK call at all on the start path.
        self.clusters_api.start.assert_not_called()

    def test_restart_is_no_op_and_returns_self(self):
        sl = self._sl()
        self.assertIs(sl.restart(wait=False), sl)
        self.clusters_api.restart.assert_not_called()

    def test_install_libraries_with_empty_list_is_no_op(self):
        sl = self._sl()
        # The empty short-circuit is intentional so callers (e.g.
        # ``Clusters.create``) can pass ``libraries=None`` without
        # special-casing serverless.
        self.assertIs(sl.install_libraries(None), sl)
        self.assertIs(sl.install_libraries([]), sl)
        self.libraries_api.install.assert_not_called()

    def test_install_libraries_with_dependencies_raises(self):
        sl = self._sl()
        with pytest.raises(NotImplementedError) as exc:
            sl.install_libraries(["pandas"])
        # Error message must point the caller at the right channel —
        # see CLAUDE.md "Error messages must answer: what you passed,
        # what was expected, valid values, what to try next".
        msg = str(exc.value)
        self.assertIn("serverless", msg.lower())
        self.assertIn("environment spec", msg.lower())
        self.libraries_api.install.assert_not_called()


class TestServerlessSingletonCache(DatabricksTestCase):
    """ServerlessCluster shares the :class:`Singleton` cache with the
    rest of the framework — two handles to the same cluster id under
    the same service must collapse onto one instance.
    """

    def test_same_cluster_id_returns_same_instance(self):
        a = ServerlessCluster(service=self.clusters, cluster_id="sl-singleton")
        b = ServerlessCluster(service=self.clusters, cluster_id="sl-singleton")
        self.assertIs(a, b)
