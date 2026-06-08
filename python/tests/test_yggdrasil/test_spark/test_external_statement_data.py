"""End-to-end tests for the Spark side of :class:`ExternalStatementData`.

Drives a real local SparkSession (via :class:`SparkTestCase` — skipped
cleanly when PySpark is not installed) so we can confirm:

* Tabulars supplied as ``external_data`` are registered as temp views
  before the statement runs.
* ``{text_key}`` placeholders in the SQL are substituted with the
  generated view name.
* :meth:`SparkStatementResult.clear_temporary_resources` drops every
  view it registered, leaving the session catalog clean.
* A failed first attempt cleans up its temp views before the retry
  loop re-registers them.
"""

from __future__ import annotations

import os
import signal
import unittest

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.statement import ExternalStatementData
from yggdrasil.io.tabular import SparkTabular
from yggdrasil.spark.statement import (
    SparkPreparedStatement,
    SparkStatementResult,
)
from yggdrasil.spark.tests import SparkTestCase


# Per-test wall-clock budget. Local Spark queries finish in under a
# second; if we hit the wall something is hung (JVM GC pause, missing
# Hadoop binary, ...) and skipping is friendlier than locking the
# whole suite.  Override via the env var when you need a bigger
# window for slow CI runners.
_TEST_TIMEOUT_SECONDS = int(os.environ.get("YGG_SPARK_TEST_TIMEOUT", "60"))


def _has_alarm() -> bool:
    """``signal.SIGALRM`` is POSIX-only; Windows has no equivalent."""
    return hasattr(signal, "SIGALRM")


class _AlarmTimeout(Exception):
    """Raised by the SIGALRM handler installed in ``setUp``."""


class TestSparkExternalStatementData(SparkTestCase, ArrowTestCase):

    # --- per-test timeout (POSIX only — no-op on Windows) ---------------

    def setUp(self) -> None:
        super().setUp()
        if _has_alarm():
            self._prev_handler = signal.signal(
                signal.SIGALRM,
                lambda *_: (_ for _ in ()).throw(
                    _AlarmTimeout(
                        f"Spark test exceeded {_TEST_TIMEOUT_SECONDS}s budget"
                    )
                ),
            )
            signal.alarm(_TEST_TIMEOUT_SECONDS)

    def tearDown(self) -> None:
        if _has_alarm():
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._prev_handler)
        super().tearDown()

    # --- helpers --------------------------------------------------------

    def _spark_tabular(self, table: pa.Table) -> SparkTabular:
        """Wrap a ``pa.Table`` as a :class:`SparkTabular` driven by the
        shared session.

        Goes via pandas because PySpark 3.5 + Java 21 can't run the
        Arrow optimization path for ``createDataFrame``; pandas → Spark
        is fast enough for the row counts these tests use.
        """
        return SparkTabular(self.spark.createDataFrame(table.to_pandas()))

    def _list_temp_views(self) -> set[str]:
        return {t.name for t in self.spark.catalog.listTables() if t.isTemporary}

    # --- tests ----------------------------------------------------------

    def test_tabular_registered_as_temp_view(self) -> None:
        tab = self._spark_tabular(pa.table({"id": [1, 2, 3]}))
        before = self._list_temp_views()

        stmt = SparkPreparedStatement(
            "SELECT id FROM {src} ORDER BY id",
            spark_session=self.spark,
            external_data={"src": tab},
        )
        result = SparkStatementResult(statement=stmt)
        result.start(wait=False, raise_error=True)

        # Substitution materialized text_value to the minted view name,
        # and that view exists in the session catalog while the result
        # is alive.
        view_name = stmt.external_data["src"].text_value
        self.assertIsNotNone(view_name)
        self.assertIn(view_name, self._list_temp_views())

        # Round-trip the result frame to confirm the SQL ran.
        rows = [r.id for r in result.spark_dataframe.collect()]
        self.assertEqual(rows, [1, 2, 3])

        result.clear_temporary_resources()
        self.assertEqual(self._list_temp_views(), before)
        # Re-clearing is a no-op (idempotent).
        result.clear_temporary_resources()

    def test_two_tabulars_can_be_joined(self) -> None:
        left = self._spark_tabular(
            pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]}),
        )
        right = self._spark_tabular(
            pa.table({"id": [2, 3, 4], "w": ["x", "y", "z"]}),
        )
        stmt = SparkPreparedStatement(
            "SELECT l.id, l.v, r.w FROM {l} l JOIN {r} r ON l.id = r.id "
            "ORDER BY l.id",
            spark_session=self.spark,
            external_data={"l": left, "r": right},
        )
        result = SparkStatementResult(statement=stmt)
        try:
            result.start(wait=False, raise_error=True)
            rows = [tuple(r) for r in result.spark_dataframe.collect()]
            self.assertEqual(rows, [(2, "b", "x"), (3, "c", "y")])
        finally:
            result.clear_temporary_resources()

    def test_clear_temporary_resources_drops_views_on_failure(self) -> None:
        """Sanity: when ``session.sql`` raises, ``start`` still cleans up
        the temp views it registered before the failure surfaced.  This
        keeps the retry path (which re-runs ``start`` on the same result)
        from accumulating leaked views in the session catalog.
        """
        tab = self._spark_tabular(pa.table({"id": [1]}))
        before = self._list_temp_views()

        stmt = SparkPreparedStatement(
            # Reference a column that doesn't exist so Spark fails after
            # the temp view registration step.
            "SELECT nonexistent FROM {src}",
            spark_session=self.spark,
            external_data={"src": tab},
        )
        result = SparkStatementResult(statement=stmt)
        # ``Awaitable.start`` only re-raises the captured failure when it
        # waits — ``wait=False`` is fire-and-forget regardless of
        # ``raise_error``. ``_start`` runs synchronously (it registers the
        # views, hits the AnalysisException on the bad column, and cleans
        # up in its ``except``), so wait for the result to surface it.
        with self.assertRaises(Exception):
            result.start(wait=True, raise_error=True)

        # The view was registered then dropped on the failure cleanup
        # path; ``text_value`` is reset so the next start() attempt mints
        # a fresh name.
        self.assertEqual(self._list_temp_views(), before)
        self.assertIsNone(stmt.external_data["src"].text_value)

    def test_pre_set_text_value_uses_existing_relation(self) -> None:
        """When ``text_value`` is supplied up front (and ``tabular`` is
        ``None``), the engine substitutes verbatim — handy for already-
        baked SQL fragments like ``VALUES`` clauses or existing tables.
        """
        stmt = SparkPreparedStatement(
            "SELECT * FROM {src}",
            spark_session=self.spark,
            external_data={
                "src": ExternalStatementData(
                    "src", text_value="VALUES (1), (2), (3) AS t(id)",
                ),
            },
        )
        result = SparkStatementResult(statement=stmt)
        try:
            result.start(wait=False, raise_error=True)
            rows = [r.id for r in result.spark_dataframe.collect()]
            self.assertEqual(sorted(rows), [1, 2, 3])
        finally:
            result.clear_temporary_resources()

        # No temp view minted — text_value was preserved as-is.
        self.assertEqual(
            stmt.external_data["src"].text_value,
            "VALUES (1), (2), (3) AS t(id)",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
