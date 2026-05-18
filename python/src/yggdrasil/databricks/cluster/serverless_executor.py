"""Run SQL on a :class:`ServerlessCluster` via Spark Connect.

Serverless compute does not expose the REPL ``CommandExecution`` API
that backs :class:`ClusterStatementExecutor` (per the Databricks
serverless limitations doc:
https://docs.databricks.com/aws/en/compute/serverless/limitations).
The supported path is Spark Connect — i.e. the
:class:`pyspark.sql.SparkSession` that
:meth:`DatabricksClient.spark` already builds, with the workspace's
serverless compute id auto-wired through ``serverless_compute_id``.

This executor wraps a :class:`ServerlessCluster` and lazily resolves
its bound client's Spark Connect session on first use, then delegates
SQL submission to :class:`SparkStatementExecutor`. The constructor
shape mirrors :class:`ClusterStatementExecutor` (cluster + optional
volume) so callers that branch on serverless-vs-classic can hand the
right type to either one without reshaping the call.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from yggdrasil.spark.executor import SparkStatementExecutor

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from yggdrasil.databricks.volume.volume import Volume

    from .serverless import ServerlessCluster


__all__ = ["ServerlessClusterStatementExecutor"]


LOGGER = logging.getLogger(__name__)


class ServerlessClusterStatementExecutor(SparkStatementExecutor):
    """Serverless counterpart of :class:`ClusterStatementExecutor`.

    Resolves its :class:`pyspark.sql.SparkSession` from
    ``cluster.client.spark()`` on first submission so construction
    stays cheap (no Spark Connect handshake until something actually
    runs). The bound *volume* is accepted for API parity with the
    classic executor — serverless SQL returns its rows through Spark
    Connect, so no staging directory is needed.
    """

    cluster: "ServerlessCluster"
    volume: Optional["Volume"]

    def __init__(
        self,
        cluster: "ServerlessCluster",
        volume: Optional["Volume"] = None,
        *,
        spark_session: Optional["SparkSession"] = None,
    ):
        # Late import to avoid the cluster module pulling the
        # serverless submodule at import time (mirrors the same
        # guard ClusterStatementExecutor uses).
        from .serverless import ServerlessCluster

        if not isinstance(cluster, ServerlessCluster):
            raise TypeError(
                f"{type(self).__name__} requires a ServerlessCluster, "
                f"got {type(cluster).__name__}. Use ClusterStatementExecutor "
                "for classic all-purpose clusters."
            )

        super().__init__(spark_session=spark_session)
        self.cluster = cluster
        self.volume = volume

    # ------------------------------------------------------------------ #
    # Session resolution — defer to the client's Spark Connect builder.
    # ------------------------------------------------------------------ #
    def resolve_session(
        self,
        statement=None,
        *,
        create: bool = True,
    ) -> Optional["SparkSession"]:
        if statement is not None and statement.spark_session is not None:
            return statement.spark_session
        if self.spark_session is not None:
            return self.spark_session
        if not create:
            return None
        # ``client.spark()`` honours the serverless_compute_id field
        # already baked onto the client, so the session lands on the
        # right serverless pool with no extra wiring.
        LOGGER.debug(
            "Resolving Spark Connect session for serverless executor %r "
            "(cluster=%r)",
            self, self.cluster,
        )
        session = self.cluster.client.spark()
        self.spark_session = session
        LOGGER.info(
            "Resolved Spark Connect session for serverless executor %r",
            self,
        )
        return session

    def has_session(self) -> bool:
        return self.spark_session is not None
