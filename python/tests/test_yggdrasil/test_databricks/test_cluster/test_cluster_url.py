"""Tests for the :class:`URLBased` round-trip on :class:`Cluster`.

The Cluster module registers under ``Scheme.DATABRICKS_CLUSTER``
(``dbks+cluster://``). The class must:

- declare its ``scheme`` ClassVar;
- produce a canonical ``dbks+cluster://<host>/<cluster_id>`` URL via
  :meth:`to_url`;
- accept the same URL shape back via :meth:`from_url`;
- be reachable via :meth:`URLBased.dispatch`.
"""
from __future__ import annotations

from yggdrasil.enums import Scheme
from yggdrasil.databricks.cluster import Cluster
from yggdrasil.databricks.tests import DatabricksTestCase
from yggdrasil.url import URL, URLBased


class TestClusterURLBased(DatabricksTestCase):
    """:class:`Cluster` registers under ``dbks+cluster://`` and
    round-trips through ``to_url`` / ``from_url``."""

    def test_scheme_classvar_is_databricks_cluster(self):
        self.assertIs(Cluster.scheme, Scheme.DATABRICKS_CLUSTER)

    def test_registry_lookup_resolves_to_cluster_class(self):
        klass = URLBased.for_scheme(Scheme.DATABRICKS_CLUSTER)
        self.assertIs(klass, Cluster)

    def test_to_url_emits_dbks_cluster_with_host_and_id(self):
        cluster = Cluster(service=self.clusters, cluster_id="c-test-1")
        url = cluster.to_url()
        self.assertEqual(url.scheme, Scheme.DATABRICKS_CLUSTER.value)
        self.assertEqual(url.host, "test.databricks.net")
        self.assertEqual(url.path.lstrip("/"), "c-test-1")

    def test_from_url_rebuilds_cluster_with_same_id(self):
        url = URL.from_(
            "dbks+cluster://test.databricks.net/c-test-1",
        )
        cluster = Cluster.from_url(url, service=self.clusters)
        self.assertEqual(cluster.cluster_id, "c-test-1")
        self.assertIs(cluster.client, self.client)

    def test_url_round_trip_preserves_cluster_id(self):
        original = Cluster(service=self.clusters, cluster_id="c-rt-1")
        rebuilt = Cluster.from_url(original.to_url(), service=self.clusters)
        self.assertEqual(rebuilt.cluster_id, original.cluster_id)

    def test_dispatch_routes_to_cluster_subclass(self):
        rebuilt = URLBased.dispatch(
            "dbks+cluster://test.databricks.net/c-dispatch-1",
            service=self.clusters,
        )
        self.assertIsInstance(rebuilt, Cluster)
        self.assertEqual(rebuilt.cluster_id, "c-dispatch-1")

    def test_from_url_without_cluster_id_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Cluster.from_url("dbks+cluster://test.databricks.net/")
        self.assertIn("cluster_id", str(ctx.exception))
