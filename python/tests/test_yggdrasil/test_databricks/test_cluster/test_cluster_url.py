"""Tests for the :class:`URLBased` round-trip on :class:`Cluster` and
:class:`ServerlessCluster`.

The Cluster module now registers under
``Scheme.DATABRICKS_CLUSTER`` (``dbks+cluster://``) and the serverless
variant under ``Scheme.DATABRICKS_SERVERLESS_CLUSTER``
(``dbks+serverless-cluster://``). Each subclass must:

- declare its ``scheme`` ClassVar (read by
  :meth:`URLBased.__init_subclass__`);
- produce a canonical ``dbks+<kind>://<host>/<cluster_id>`` URL via
  :meth:`to_url`;
- accept the same URL shape back via :meth:`from_url`, resolving the
  underlying :class:`DatabricksClient` from the host;
- be reachable via the cross-cutting :meth:`URLBased.dispatch`
  entry-point so a caller with just a URL can rebuild the handle
  without knowing which subclass owns it.
"""
from __future__ import annotations

from yggdrasil.data.enums import Scheme
from yggdrasil.databricks.cluster import Cluster, ServerlessCluster
from yggdrasil.databricks.tests import DatabricksTestCase
from yggdrasil.io.url import URL, URLBased


class TestClusterURLBased(DatabricksTestCase):
    """:class:`Cluster` registers under ``dbks+cluster://`` and
    round-trips through ``to_url`` / ``from_url``."""

    def test_scheme_classvar_is_databricks_cluster(self):
        # __init_subclass__ coerces the class-body ``scheme`` to a
        # :class:`Scheme` member at registration time.
        self.assertIs(Cluster.scheme, Scheme.DATABRICKS_CLUSTER)

    def test_registry_lookup_resolves_to_cluster_class(self):
        # The cluster module's import already fired
        # ``__init_subclass__`` and wired the class into the registry.
        klass = URLBased.for_scheme(Scheme.DATABRICKS_CLUSTER)
        self.assertIs(klass, Cluster)

    def test_to_url_emits_dbks_cluster_with_host_and_id(self):
        cluster = Cluster(service=self.clusters, cluster_id="c-test-1")
        url = cluster.to_url()
        # ``test.databricks.net`` comes from ``DatabricksTestCase.HOST``.
        self.assertEqual(url.scheme, Scheme.DATABRICKS_CLUSTER.value)
        self.assertEqual(url.host, "test.databricks.net")
        # Single path segment = cluster id.
        self.assertEqual(url.path.lstrip("/"), "c-test-1")

    def test_from_url_rebuilds_cluster_with_same_id(self):
        # Build under the test client's host so ``from_url`` resolves
        # back through the singleton client cache.
        url = URL.from_(
            "dbks+cluster://test.databricks.net/c-test-1",
        )
        cluster = Cluster.from_url(url, client=self.client)
        self.assertEqual(cluster.cluster_id, "c-test-1")
        self.assertIs(cluster.client, self.client)

    def test_url_round_trip_preserves_cluster_id(self):
        original = Cluster(service=self.clusters, cluster_id="c-rt-1")
        rebuilt = Cluster.from_url(original.to_url(), client=self.client)
        self.assertEqual(rebuilt.cluster_id, original.cluster_id)

    def test_dispatch_routes_to_cluster_subclass(self):
        # The cross-cutting URLBased dispatcher picks the right
        # subclass off the scheme.
        rebuilt = URLBased.dispatch(
            "dbks+cluster://test.databricks.net/c-dispatch-1",
            client=self.client,
        )
        self.assertIsInstance(rebuilt, Cluster)
        self.assertEqual(rebuilt.cluster_id, "c-dispatch-1")

    def test_from_url_without_cluster_id_path_raises(self):
        # The grammar requires a single path segment; bare hosts
        # don't carry a cluster id.
        with self.assertRaises(ValueError) as ctx:
            Cluster.from_url("dbks+cluster://test.databricks.net/")
        self.assertIn("cluster_id", str(ctx.exception))


class TestServerlessClusterURLBased(DatabricksTestCase):
    """:class:`ServerlessCluster` carries its own scheme so a URL
    round-trip preserves the serverless flavor — important because
    the lifecycle overrides depend on the concrete subclass."""

    def test_scheme_classvar_is_serverless(self):
        self.assertIs(
            ServerlessCluster.scheme, Scheme.DATABRICKS_SERVERLESS_CLUSTER,
        )

    def test_registry_lookup_resolves_to_serverless_class(self):
        klass = URLBased.for_scheme(Scheme.DATABRICKS_SERVERLESS_CLUSTER)
        self.assertIs(klass, ServerlessCluster)

    def test_to_url_emits_serverless_scheme(self):
        sl = ServerlessCluster(service=self.clusters, cluster_id="sl-test-1")
        url = sl.to_url()
        self.assertEqual(url.scheme, Scheme.DATABRICKS_SERVERLESS_CLUSTER.value)
        self.assertEqual(url.path.lstrip("/"), "sl-test-1")

    def test_from_url_dispatches_to_serverless_subclass(self):
        # Going through URLBased.dispatch picks ServerlessCluster off
        # the scheme even though Cluster.from_url would also accept
        # the URL — that's the whole point of the separate scheme.
        rebuilt = URLBased.dispatch(
            "dbks+serverless-cluster://test.databricks.net/sl-dispatch-1",
            client=self.client,
        )
        self.assertIsInstance(rebuilt, ServerlessCluster)
        self.assertEqual(rebuilt.cluster_id, "sl-dispatch-1")

    def test_inherits_cluster_identity_helpers(self):
        # ServerlessCluster is a thin subclass — the explore_url /
        # str-repr helpers come from the parent class unchanged.
        sl = ServerlessCluster(service=self.clusters, cluster_id="sl-explore-1")
        self.assertIn("sl-explore-1", str(sl.explore_url))
