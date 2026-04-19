"""
Integration tests for :class:`yggdrasil.databricks.sql.statements.Statements`.

These run against a live Databricks workspace via :class:`DatabricksCase` and
are skipped automatically when ``DATABRICKS_HOST`` is not set.
"""

from __future__ import annotations

import time
import uuid

import pytest
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.sql import StatementState

from yggdrasil.databricks.sql.statement import PreparedStatement, StatementResult
from yggdrasil.databricks.sql.statements import Statements

from ..conftest import DatabricksCase, requires_databricks

pytestmark = [requires_databricks, pytest.mark.integration]


@requires_databricks
class TestStatementsIntegration(DatabricksCase):
    """Live-workspace coverage for the ``Statements`` collection service."""

    _CATALOG = "trading"
    _SCHEMA = "unittest"

    service: Statements

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.service = cls.workspace.statements
        cls.engine = cls.workspace.sql(
            catalog_name=cls._CATALOG,
            schema_name=cls._SCHEMA,
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _tag(self) -> str:
        """Return a unique per-test SQL comment marker.

        Embedding the marker in the SQL text lets ``list_statements``
        filter to exactly the statement this test just executed.
        """
        return f"ygg_integration_{uuid.uuid4().hex}"

    def _run_marker_query(self) -> tuple[str, StatementResult]:
        """Execute a tiny SELECT with a unique marker; return (tag, result)."""
        tag = self._tag()
        result = self.engine.execute(
            f"/* {tag} */ SELECT 1 AS n",
            engine="api",
        )
        result.wait()
        self.assertTrue(result.done)
        self.assertFalse(result.failed)
        return tag, result

    # -------------------------------------------------------------------
    # Service wiring
    # -------------------------------------------------------------------

    def test_service_is_bound_to_workspace(self):
        self.assertIsInstance(self.service, Statements)
        self.assertIs(self.service.client, self.workspace)

    # -------------------------------------------------------------------
    # statement() factory
    # -------------------------------------------------------------------

    def test_statement_factory_builds_unstarted_result(self):
        stmt = self.service.statement(
            "SELECT :x AS x",
            parameters={"x": "42"},
        )
        self.assertIsInstance(stmt, StatementResult)
        self.assertIs(stmt.service, self.service)
        self.assertEqual(stmt.text, "SELECT :x AS x")
        self.assertEqual(stmt.parameters, {"x": "42"})
        self.assertFalse(stmt.started)

    def test_statement_factory_accepts_prepared_statement(self):
        cfg = PreparedStatement(text="SELECT 1 AS n")
        stmt = self.service.statement(cfg)
        self.assertIs(stmt.statement, cfg)
        self.assertIs(stmt.service, self.service)

    # -------------------------------------------------------------------
    # find_statement
    # -------------------------------------------------------------------

    def test_find_statement_resolves_live_execution(self):
        _, submitted = self._run_marker_query()

        fetched = self.service.find_statement(submitted.statement_id)
        self.assertIsInstance(fetched, StatementResult)
        self.assertEqual(fetched.statement_id, submitted.statement_id)
        self.assertIs(fetched.service, self.service)
        self.assertTrue(fetched.done)
        self.assertEqual(fetched.state, StatementState.SUCCEEDED)

    def test_find_statement_missing_returns_none_when_safe(self):
        # Statement IDs are UUIDs; this one is valid-shaped but will never exist.
        bogus = str(uuid.uuid4())
        self.assertIsNone(self.service.find_statement(bogus, raise_error=False))

    def test_find_statement_missing_raises_by_default(self):
        bogus = str(uuid.uuid4())
        with self.assertRaises(ResourceDoesNotExist):
            self.service.find_statement(bogus)

    # -------------------------------------------------------------------
    # Dict-like access
    # -------------------------------------------------------------------

    def test_getitem_delegates_to_find_statement(self):
        _, submitted = self._run_marker_query()
        stmt = self.service[submitted.statement_id]
        self.assertEqual(stmt.statement_id, submitted.statement_id)

    def test_contains_checks_existence(self):
        _, submitted = self._run_marker_query()
        self.assertIn(submitted.statement_id, self.service)
        self.assertNotIn(str(uuid.uuid4()), self.service)
        # Non-string keys are reported as not-contained without an SDK call.
        self.assertNotIn(123, self.service)

    # -------------------------------------------------------------------
    # list_statements
    # -------------------------------------------------------------------

    def test_list_statements_filters_by_text_substring(self):
        tag, submitted = self._run_marker_query()

        # ``system.query.history`` is eventually consistent; poll briefly.
        found: StatementResult | None = None
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            matches = list(
                self.service.list_statements(
                    text_contains=tag,
                    limit=5,
                )
            )
            for row in matches:
                if row.statement_id == submitted.statement_id:
                    found = row
                    break
            if found is not None:
                break
            time.sleep(2.0)

        self.assertIsNotNone(
            found,
            f"Marker {tag!r} never surfaced in system.query.history "
            f"for statement {submitted.statement_id!r}",
        )
        self.assertIs(found.service, self.service)
        # The history snapshot populates the statement text and _history map.
        self.assertIn(tag, found.text)
        self.assertIsNotNone(found._history)
        self.assertEqual(found._history.get("statement_id"), submitted.statement_id)

    def test_list_statements_with_fetch_response_hydrates_state(self):
        tag, submitted = self._run_marker_query()

        # Poll until the history row appears.
        fetched: StatementResult | None = None
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            for row in self.service.list_statements(
                text_contains=tag,
                limit=5,
                fetch_response=True,
            ):
                if row.statement_id == submitted.statement_id:
                    fetched = row
                    break
            if fetched is not None:
                break
            time.sleep(2.0)

        self.assertIsNotNone(fetched, f"Marker {tag!r} not found with fetch_response=True")
        # fetch_response=True pulls the live response so state is authoritative.
        self.assertIsNotNone(fetched._response)
        self.assertEqual(fetched.state, StatementState.SUCCEEDED)

    def test_list_statements_respects_limit(self):
        rows = list(self.service.list_statements(limit=1))
        self.assertLessEqual(len(rows), 1)

    def test_list_statements_explicit_none_time_filters_are_accepted(self):
        # Dropping both time bounds must not crash — the SQL just omits the
        # clauses.  We cap with a small limit to keep the call cheap.
        rows = list(
            self.service.list_statements(
                start_time_from=None,
                start_time_to=None,
                limit=1,
            )
        )
        self.assertLessEqual(len(rows), 1)
