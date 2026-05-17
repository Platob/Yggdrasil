"""Tests for :class:`DatabricksSparkStatementExecutor` — the
Databricks Connect-backed default executor pinned on
:class:`SQLEngine.spark`."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from yggdrasil.databricks.sql.spark_executor import (
    DatabricksSparkStatementExecutor,
)


class TestDatabricksSparkStatementExecutor:

    def setup_method(self) -> None:
        # Process-wide singleton — clear so cases don't see each other's
        # pinned sessions / clients.
        DatabricksSparkStatementExecutor._INSTANCES.clear()

    def test_resolve_session_falls_back_to_client_spark(self) -> None:
        """``resolve_session(create=True)`` with no other source builds
        a session via ``client.spark()`` rather than PyEnv's local
        bootstrap."""
        sentinel = MagicMock(name="spark_session")
        client = MagicMock(name="client")
        client.spark.return_value = sentinel

        executor = DatabricksSparkStatementExecutor(client=client)
        # No PySpark in this environment ⇒ ``getActiveSession()`` import
        # raises ⇒ the executor falls through to ``client.spark()``.
        session = executor.resolve_session(create=True)

        assert session is sentinel
        client.spark.assert_called_once_with()
        # Subsequent calls reuse the cached session — no second handshake.
        assert executor.resolve_session(create=True) is sentinel
        client.spark.assert_called_once_with()

    def test_resolve_session_create_false_returns_none(self) -> None:
        """``create=False`` never builds a session."""
        client = MagicMock(name="client")
        executor = DatabricksSparkStatementExecutor(client=client)

        assert executor.resolve_session(create=False) is None
        client.spark.assert_not_called()

    def test_pinned_session_wins_over_client_spark(self) -> None:
        pinned = MagicMock(name="pinned")
        client = MagicMock(name="client")

        executor = DatabricksSparkStatementExecutor(
            client=client, spark_session=pinned,
        )
        assert executor.resolve_session(create=True) is pinned
        client.spark.assert_not_called()

    def test_statement_session_wins_over_pinned(self) -> None:
        pinned = MagicMock(name="pinned")
        per_stmt = MagicMock(name="per_stmt")
        client = MagicMock(name="client")

        executor = DatabricksSparkStatementExecutor(
            client=client, spark_session=pinned,
        )
        statement = MagicMock(spark_session=per_stmt)
        assert executor.resolve_session(statement, create=True) is per_stmt
        client.spark.assert_not_called()

    def test_singleton_per_client(self) -> None:
        c1 = MagicMock(name="c1")
        c2 = MagicMock(name="c2")
        a = DatabricksSparkStatementExecutor(client=c1)
        b = DatabricksSparkStatementExecutor(client=c1)
        c = DatabricksSparkStatementExecutor(client=c2)
        assert a is b
        assert a is not c

    def test_active_session_short_circuits(self) -> None:
        """An active in-process SparkSession is preferred over building a
        fresh Databricks Connect session — keeps notebook drivers /
        Databricks Job tasks on the JVM-singleton session they were
        launched with."""
        active = MagicMock(name="active")
        client = MagicMock(name="client")

        fake_module = MagicMock()
        fake_module.SparkSession.getActiveSession.return_value = active
        with patch.dict(
            "sys.modules", {"pyspark.sql": fake_module, "pyspark": MagicMock()},
        ):
            executor = DatabricksSparkStatementExecutor(client=client)
            assert executor.resolve_session(create=True) is active
        client.spark.assert_not_called()
