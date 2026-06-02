"""Async / awaitable statement waiting + state polling.

Skipped unless ``DATABRICKS_HOST`` (and credentials) are set — see
:class:`SQLIntegrationCase`.
"""
from __future__ import annotations

from ._sql_integration import SQLIntegrationCase


class TestSQLWaiting(SQLIntegrationCase):

    def ignore_test_waiting(self) -> None:
        r = self.client.sql.execute(
            """WITH heavy AS (
  SELECT a.id * b.id AS x, SHA2(CAST(a.id * b.id AS STRING), 512) AS h
  FROM range(10000000) a
  CROSS JOIN range(100) b
)
SELECT COUNT(DISTINCT h) FROM heavy;""",
            wait=False
        )

        assert r.state.is_running

        r.wait()

        assert r.done
