"""Live-integration tests for the cluster module.

Skipped unless ``DATABRICKS_HOST`` is exported (the standard SDK env
var). Each suite is opt-in via additional env vars so a workspace
that exposes a SQL warehouse but no all-purpose cluster can still
run the rest of the integration suite without these failing.

Required env vars
-----------------
- ``DATABRICKS_HOST`` plus the matching auth (token / OAuth / profile)
  to reach the workspace. Required by :class:`DatabricksIntegrationCase`.
- ``DATABRICKS_INTEGRATION_CATALOG`` / ``DATABRICKS_INTEGRATION_SCHEMA``
  (defaults: ``trading.unittest``) for the catalog / schema the tests
  write into.
- ``DATABRICKS_INTEGRATION_CLUSTER_ID`` (or
  ``DATABRICKS_INTEGRATION_CLUSTER_NAME``) — the all-purpose cluster
  to drive REPL commands against. The cluster must be running (or
  startable) and the test identity must have ``CAN ATTACH TO`` /
  ``CAN MANAGE`` rights. Both unset → the ``TestCluster*`` suites
  skip; the URL-round-trip suite still runs since it doesn't touch
  the cluster.

These tests exercise:

- :class:`Cluster` URL round-trip against a live workspace handle;
- :class:`ClusterStatementExecutor` non-SELECT path (a ``CREATE
  TABLE`` runs straight through ``cluster.command()``);
- :class:`ClusterStatementExecutor` SELECT path (rewrite to
  ``INSERT OVERWRITE DIRECTORY ... USING parquet``, read the
  staged Parquet folder back via :class:`pyarrow.dataset`);
- staging cleanup via the prepared statement's
  ``clear_temporary_resources`` hook fires after a successful wait.
"""
from __future__ import annotations

import os
import secrets
import unittest
from typing import ClassVar

from databricks.sdk.errors import DatabricksError

from yggdrasil.databricks.cluster import (
    Cluster,
    ClusterPreparedStatement,
    ClusterStatementExecutor,
)
from yggdrasil.databricks.volume.volume import Volume
from yggdrasil.io.url import URLBased

from .. import DatabricksIntegrationCase


__all__ = [
    "TestClusterURLIntegration",
    "TestClusterStatementExecutorIntegration",
]


class _ClusterIntegrationBase(DatabricksIntegrationCase):
    """Shared fixture for the cluster integration suites."""

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    cluster: ClassVar[Cluster]
    volume: ClassVar[Volume]
    volume_name: ClassVar[str]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = (
            os.environ.get("DATABRICKS_INTEGRATION_CATALOG", "trading").strip()
            or "trading"
        )
        cls.schema_name = (
            os.environ.get("DATABRICKS_INTEGRATION_SCHEMA", "unittest").strip()
            or "unittest"
        )

        cluster_id = os.environ.get("DATABRICKS_INTEGRATION_CLUSTER_ID", "").strip()
        cluster_name = os.environ.get(
            "DATABRICKS_INTEGRATION_CLUSTER_NAME", "",
        ).strip()
        if not cluster_id and not cluster_name:
            raise unittest.SkipTest(
                "DATABRICKS_INTEGRATION_CLUSTER_ID / "
                "DATABRICKS_INTEGRATION_CLUSTER_NAME unset — set one of "
                "them to an all-purpose cluster the test identity can "
                "attach to."
            )

        # ``find_cluster`` does the SDK round-trip; if the cluster is
        # gone or the test identity can't reach it, skip rather than
        # fail (the suite-level signal is "no live cluster", not
        # "cluster is broken").
        try:
            cls.cluster = cls.client.compute.clusters.find_cluster(
                cluster_id=cluster_id or None,
                cluster_name=cluster_name or None,
                raise_error=True,
            )
        except (DatabricksError, ValueError) as exc:
            raise unittest.SkipTest(
                f"Cannot reach integration cluster "
                f"(id={cluster_id!r}, name={cluster_name!r}): {exc}",
            ) from exc

        # Bring the cluster up if needed — REPL command-execution
        # requires a RUNNING state.
        cls.cluster.start(wait=True)

        # Each test class gets its own short-lived volume so a partial
        # failure leaves at most one orphan to clean up.
        cls.volume_name = f"yg_cluster_it_{secrets.token_hex(4)}"
        cls.volume = Volume(
            service=cls.client.volumes,
            catalog_name=cls.catalog_name,
            schema_name=cls.schema_name,
            volume_name=cls.volume_name,
        )
        try:
            cls.volume.ensure_created(
                comment="yggdrasil cluster-executor integration volume",
            )
        except DatabricksError as exc:
            raise unittest.SkipTest(
                f"Cannot create staging volume "
                f"{cls.catalog_name}.{cls.schema_name}.{cls.volume_name}: "
                f"{exc}. Set DATABRICKS_INTEGRATION_CATALOG / "
                "DATABRICKS_INTEGRATION_SCHEMA to a location the test "
                "identity can write to."
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            try:
                cls.volume.delete(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass
        finally:
            super().tearDownClass()


class TestClusterURLIntegration(DatabricksIntegrationCase):
    """:class:`Cluster` URL round-trip against the live client.

    Doesn't require ``DATABRICKS_INTEGRATION_CLUSTER_ID`` — uses a
    synthetic cluster id so we exercise the URL plumbing without
    touching a live cluster.
    """

    def test_to_url_round_trips_through_dispatch(self):
        # Build a Cluster handle bound to the live client and confirm
        # the URL round-trips through the cross-cutting URLBased
        # dispatcher (which is what a downstream consumer with just
        # the URL string would do).
        cluster = Cluster(
            service=self.client.compute.clusters,
            cluster_id="c-not-resolved-on-construction",
        )
        url = cluster.to_url()
        rebuilt = URLBased.dispatch(url, client=self.client)
        self.assertIsInstance(rebuilt, Cluster)
        self.assertEqual(rebuilt.cluster_id, cluster.cluster_id)
        # Both handles share the live client.
        self.assertIs(rebuilt.client, self.client)


class TestClusterStatementExecutorIntegration(_ClusterIntegrationBase):
    """End-to-end SQL flow through :class:`ClusterStatementExecutor`."""

    def test_non_select_runs_through_cluster_command(self):
        # Smallest possible non-SELECT — a no-op DDL that doesn't
        # require an existing table. Verifies the cluster.command()
        # routing path lands without the executor doing any rewrite.
        executor = ClusterStatementExecutor(self.cluster, self.volume)
        stmt = ClusterPreparedStatement(
            "SELECT 1 -- non-select-style: side-effecting through SET",
        )
        # Use an actual non-SELECT instead.
        stmt = ClusterPreparedStatement("SET spark.sql.ansi.enabled = true")

        result = executor.execute(stmt, wait=True, raise_error=True)
        self.assertTrue(result.done)
        self.assertFalse(result.failed)
        # No staging path — the executor only mints one for SELECTs.
        self.assertIsNone(result.statement.output_path)

    def test_select_round_trips_via_insert_overwrite_directory(self):
        executor = ClusterStatementExecutor(self.cluster, self.volume)

        # ``SELECT * FROM range(...)`` is the canonical Spark SQL
        # source that doesn't depend on any user-managed table —
        # gives us a predictable 5-row result we can verify on the
        # Parquet side.
        stmt = ClusterPreparedStatement(
            "SELECT id, CAST(id * 10 AS BIGINT) AS ten_id FROM range(5)",
        )

        result = executor.execute(stmt, wait=True, raise_error=True)
        self.assertTrue(result.done)
        self.assertFalse(result.failed)
        # The executor rewrote the SELECT and bound a staging path.
        self.assertIsNotNone(result.statement.output_path)

        # Reading the result drains the staged Parquet folder.
        table = result.read_arrow_table()
        self.assertEqual(table.num_rows, 5)
        # Column names + values come back intact through the
        # Parquet → Arrow bridge.
        self.assertEqual(
            sorted(table.column_names), ["id", "ten_id"],
        )
        ids = sorted(table.column("id").to_pylist())
        ten_ids = sorted(table.column("ten_id").to_pylist())
        self.assertEqual(ids, [0, 1, 2, 3, 4])
        self.assertEqual(ten_ids, [0, 10, 20, 30, 40])

        # Cleanup runs when the prepared statement's
        # clear_temporary_resources fires — drive it explicitly so the
        # test doesn't depend on the batch wait-hook firing order.
        output_path = result.statement.output_path
        result.statement.clear_temporary_resources()
        self.assertIsNone(result.statement.output_path)
        # Best-effort: the staged folder must be gone after cleanup.
        # ``exists`` may not be supported on every VolumePath shape;
        # fall back to ``ls`` which raises FileNotFoundError on a
        # removed prefix.
        try:
            still_there = output_path.exists  # type: ignore[attr-defined]
        except AttributeError:
            still_there = True
            try:
                list(output_path.ls(recursive=False))
            except FileNotFoundError:
                still_there = False
        self.assertFalse(
            still_there,
            f"Staged folder {output_path!r} survived cleanup",
        )

    def test_default_context_key_is_reused_across_statements(self):
        # The executor mints one context_key per (cluster, volume)
        # pair; back-to-back statements must reuse the live REPL
        # context instead of creating a fresh one each time (the
        # cluster caps at 145 contexts).
        executor = ClusterStatementExecutor(self.cluster, self.volume)

        first = executor.execute(
            ClusterPreparedStatement("SELECT 1 AS a FROM range(1)"),
            wait=True, raise_error=True,
        )
        second = executor.execute(
            ClusterPreparedStatement("SELECT 2 AS b FROM range(1)"),
            wait=True, raise_error=True,
        )
        self.assertTrue(first.done and second.done)
        self.assertFalse(first.failed or second.failed)

        # Both commands share the same context id (= REPL session) —
        # checked via the underlying CommandExecution handle.
        ctx_a = first.command.context.context_id
        ctx_b = second.command.context.context_id
        self.assertEqual(ctx_a, ctx_b)
