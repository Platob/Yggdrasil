""":class:`DatabricksSparkStatementExecutor` — Databricks Connect variant.

The base :class:`SparkStatementExecutor` falls back to
:meth:`PyEnv.spark_session` when no session is pinned, which spins up a
*local* PySpark on the driver. That's the wrong default for the
Databricks engine: when no caller-provided session is reachable, we want
:meth:`DatabricksClient.spark` (Databricks Connect) so the SQL lands on
the workspace's serverless / classic compute with the right auth, deps,
and warehouse routing.

This subclass plugs that in on the ``create=True`` branch and is the
default ``SQLEngine.spark`` executor. The cluster-bound counterparts
(:class:`ServerlessClusterStatementExecutor`,
:class:`ClusterStatementExecutor`) keep their own ``cluster.client.spark()``
override since the cluster id participates in the session builder.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from yggdrasil.spark.executor import SparkStatementExecutor

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.spark.statement import SparkPreparedStatement


__all__ = ["DatabricksSparkStatementExecutor"]


logger = logging.getLogger(__name__)


class DatabricksSparkStatementExecutor(SparkStatementExecutor):
    """:class:`SparkStatementExecutor` that builds its session via
    :meth:`DatabricksClient.spark` (Databricks Connect).

    Singleton-cached per ``(cls, client)`` — same workspace ⇒ same
    executor ⇒ same Spark Connect session, in line with the
    process-wide JVM-singleton nature of Spark.
    """

    client: "DatabricksClient"

    @classmethod
    def _singleton_key(cls, client=None, *args: Any, **kwargs: Any) -> Any:
        return (cls, client)

    def __init__(
        self,
        client: "DatabricksClient",
        *,
        spark_session: Optional["SparkSession"] = None,
    ):
        if getattr(self, "_initialized", False):
            if spark_session is not None:
                self.spark_session = spark_session
            return
        super().__init__(spark_session=spark_session)
        self.client = client

    def resolve_session(
        self,
        statement: Optional["SparkPreparedStatement"] = None,
        *,
        create: bool = True,
    ) -> Optional["SparkSession"]:
        if statement is not None and statement.spark_session is not None:
            return statement.spark_session
        if self.spark_session is not None:
            return self.spark_session

        # Active in-process session (notebook driver, an outer
        # ``client.spark()`` call, a Databricks Job task) wins over a
        # fresh Databricks Connect handshake — ``client.spark()`` would
        # return the same active session anyway, but going through
        # ``getActiveSession`` skips the dependency-classification cost.
        try:
            from pyspark.sql import SparkSession
            active = SparkSession.getActiveSession()
        except Exception:
            active = None
        if active is not None:
            logger.debug(
                "Reusing active Spark Connect session for executor %r", self,
            )
            self.spark_session = active
            return active

        if not create:
            return None

        logger.debug(
            "Resolving Spark Connect session for executor %r (client=%r)",
            self, self.client,
        )
        session = self.client.spark()
        self.spark_session = session
        logger.info(
            "Resolved Spark Connect session for executor %r", self,
        )
        return session
