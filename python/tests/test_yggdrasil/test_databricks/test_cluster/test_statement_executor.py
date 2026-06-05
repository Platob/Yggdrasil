"""Tests for :class:`ClusterStatementExecutor` and the cluster-backed
:class:`PreparedStatement` / :class:`StatementResult` pair.

The executor is a thin wrapper: it rewrites SELECT-like statements
into ``INSERT OVERWRITE DIRECTORY ... USING parquet ...`` so result
rows land on the bound volume as Parquet (the REPL stdout path
would otherwise truncate large results), and it routes every
non-SELECT through the cluster's :meth:`Cluster.command` unchanged.
The tests below cover:

- SELECT detection and rewrite (with ``output_path`` bound to the
  result's prepared statement);
- non-SELECT pass-through;
- default ``context_key`` keyed off the volume so all statements on
  the same volume share one REPL context (clusters cap at 145);
- :class:`ClusterPreparedStatement.clear_temporary_resources`
  unlinking the staged output folder.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.cluster import (
    Cluster,
    ClusterPreparedStatement,
    ClusterStatementExecutor,
    ClusterStatementResult,
)
from yggdrasil.databricks.tests import DatabricksTestCase


def _mock_volume(name: str = "cat.sch.vol") -> MagicMock:
    """Volume stand-in with the minimum surface the executor reads.

    ``executor._rewrite_for_select`` calls ``volume.path(rel)`` to
    mint the staging :class:`VolumePath`, and reads
    ``volume.full_name()`` to seed the default context key. Both
    return MagicMocks so we can assert on the call args / returned
    paths without touching a real volume.
    """
    vol = MagicMock(name="Volume")
    vol.full_name.return_value = name

    def _path(rel: str) -> MagicMock:
        path = MagicMock(name=f"VolumePath({rel})")
        path.full_path.return_value = f"/Volumes/{name.replace('.', '/')}/{rel}"
        return path

    vol.path.side_effect = _path
    return vol


class _ExecutorTestBase(DatabricksTestCase):
    """Test base that exposes a real :class:`Cluster` bound to the
    mock client plus a fresh mock volume. The cluster is built by id
    only (the SDK ``get`` call would normally back-fill details — we
    bypass it because the executor never reads ``cluster.details``).
    """

    cluster: Cluster
    volume: MagicMock
    executor: ClusterStatementExecutor

    def setUp(self) -> None:
        super().setUp()
        self.cluster = Cluster(service=self.clusters, cluster_id="c-1")
        self.volume = _mock_volume()
        self.executor = ClusterStatementExecutor(self.cluster, self.volume)


class TestConstruction(_ExecutorTestBase):

    def test_owns_cluster_and_volume(self):
        # The wrapper owns both — the executor's whole point is to
        # adapt the StatementExecutor contract on top of them.
        self.assertIs(self.executor.cluster, self.cluster)
        self.assertIs(self.executor.volume, self.volume)

    def test_default_context_key_is_volume_scoped(self):
        # One REPL context per (cluster, volume) keeps us well under
        # the 145-context-per-cluster cap. The key must mention the
        # volume's full name so the executor's default for the same
        # volume is stable across reconstructions.
        self.assertIn("cat.sch.vol", self.executor.default_context_key)

    def test_explore_url_inherits_from_cluster(self):
        # The repr / explore-url comes from the cluster handle so a
        # log line referring to the executor still points at the
        # right compute page.
        self.assertEqual(self.executor.explore_url, self.cluster.explore_url)



class TestSelectRewrite(_ExecutorTestBase):
    """SELECT-like statements must be wrapped in ``INSERT OVERWRITE
    DIRECTORY ... USING parquet ...`` so result rows are read back
    from a Parquet folder instead of the REPL stdout."""

    def _coerce(self, text: str) -> ClusterPreparedStatement:
        return self.executor.prepare(text)

    def test_select_is_wrapped_in_insert_overwrite_directory(self):
        stmt = self._coerce("SELECT 1")
        self.assertTrue(stmt.text.startswith("INSERT OVERWRITE DIRECTORY "))
        self.assertIn("USING parquet ", stmt.text)
        # The original query trails the directive untouched.
        self.assertTrue(stmt.text.rstrip().endswith("SELECT 1"))
        # The staged path is bound so result reads + cleanup know
        # where to look.
        self.assertIsNotNone(stmt.output_path)
        # ``volume.path(...)`` was called with a relative path under
        # ``.sql/cluster/select/``.
        rel = self.volume.path.call_args.args[0]
        self.assertTrue(rel.startswith(".sql/cluster/select/"))

    def test_with_query_is_treated_as_select(self):
        # ``WITH ... SELECT`` is the second most common SELECT shape;
        # ``looks_like_query`` accepts it so the executor must too.
        stmt = self._coerce("WITH t AS (SELECT 1) SELECT * FROM t")
        self.assertTrue(stmt.text.startswith("INSERT OVERWRITE DIRECTORY "))
        self.assertIsNotNone(stmt.output_path)

    def test_non_select_passes_through_unchanged(self):
        # DDL / DML / CALL / CREATE all go straight to the REPL.
        for text in (
            "CREATE TABLE t (a INT)",
            "INSERT INTO t VALUES (1)",
            "DROP TABLE IF EXISTS t",
        ):
            with self.subTest(text=text):
                stmt = self._coerce(text)
                self.assertEqual(stmt.text, text)
                self.assertIsNone(stmt.output_path)

    def test_already_rewritten_statement_is_not_double_wrapped(self):
        # ``StatementResult.retry`` will resubmit the same prepared
        # statement; double-wrapping the SELECT would mint a fresh
        # staging folder on every retry and orphan the first.
        stmt = self._coerce("SELECT 1")
        original_text = stmt.text
        original_path = stmt.output_path
        # Re-coercing an already-wrapped statement is a no-op.
        again = self.executor.prepare(stmt)
        self.assertEqual(again.text, original_text)
        self.assertIs(again.output_path, original_path)

    def test_full_path_quoted_in_directive(self):
        # The Volumes path lands inside single quotes — empty quote
        # bodies would yield a syntax error downstream and an
        # unquoted path would parse as multiple tokens.
        stmt = self._coerce("SELECT 1")
        self.assertIn(
            f"'{stmt.output_path.full_path()}'",
            stmt.text,
        )


class TestSubmission(_ExecutorTestBase):
    """:meth:`_submit_statement` builds a :class:`ClusterStatementResult`
    and (when ``start=True``) kicks off the underlying
    :class:`CommandExecution` via the cluster."""

    def test_submit_builds_typed_result(self):
        result = self.executor._submit_statement(
            self.executor.prepare("DROP TABLE IF EXISTS t"),
            start=False,
        )
        self.assertIsInstance(result, ClusterStatementResult)
        self.assertIs(result.executor, self.executor)
        # With start=False, no command has been minted yet.
        self.assertIsNone(result.command)

    def test_submit_command_routes_through_cluster_command(self):
        # ``submit_command`` exposes the inner builder used by the
        # result; calling it must reach ``cluster.command(...)`` with
        # the right language / context key.
        self.cluster.command = MagicMock(return_value=MagicMock(name="CommandExecution"))  # type: ignore[assignment]

        stmt = self.executor.prepare("DROP TABLE t")
        self.executor.submit_command(stmt)
        self.cluster.command.assert_called_once()
        kwargs = self.cluster.command.call_args.kwargs
        # The default context key is the volume-scoped one we built
        # at construction time.
        self.assertEqual(kwargs["context"], self.executor.default_context_key)
        # ``command_str`` carries the rewritten / passthrough text.
        self.assertEqual(kwargs["command_str"], stmt.text)


class TestPreparedStatementCleanup:
    """The staged output folder must be unlinked when the prepared
    statement's :meth:`clear_temporary_resources` runs (the batch's
    wait-hook fires it on success)."""

    def test_clear_unlinks_output_path(self):
        path = MagicMock(name="VolumePath")
        stmt = ClusterPreparedStatement("SELECT 1", output_path=path)
        stmt.clear_temporary_resources()
        path.remove.assert_called_once()
        # Recursive=True so the whole Parquet folder is dropped, not
        # just an empty leaf; missing_ok so already-deleted folders
        # don't blow up the wait-hook.
        kwargs = path.remove.call_args.kwargs
        assert kwargs.get("recursive") is True
        assert kwargs.get("missing_ok") is True
        # Idempotent: a second call is a no-op (output_path cleared).
        assert stmt.output_path is None
        stmt.clear_temporary_resources()
        path.remove.assert_called_once()  # still once

    def test_clear_without_output_path_is_noop(self):
        stmt = ClusterPreparedStatement("CREATE TABLE t (a INT)")
        # No path bound → nothing to remove, no errors.
        stmt.clear_temporary_resources()

    def test_clear_swallows_remove_failures(self):
        path = MagicMock(name="VolumePath")
        path.remove.side_effect = RuntimeError("boom")
        stmt = ClusterPreparedStatement("SELECT 1", output_path=path)
        # Cleanup is best-effort — a remove failure must not surface
        # past the wait-hook (it would mask the underlying SQL
        # result). The output_path is still cleared so subsequent
        # calls don't loop on the same broken path.
        stmt.clear_temporary_resources()
        assert stmt.output_path is None


class TestModuleExports:
    """The cluster package exposes the new classes at package level so
    callers can ``from yggdrasil.databricks.cluster import ...`` without
    knowing which submodule each one lives in."""

    def test_executor_and_statement_classes_exported(self):
        from yggdrasil.databricks.cluster import (
            ClusterPreparedStatement as _PS,
            ClusterStatementBatch as _SB,
            ClusterStatementExecutor as _SE,
            ClusterStatementResult as _SR,
        )
        # Class identity beats string assertions — guards against the
        # __init__ re-exporting a different module's class by mistake.
        assert _PS is ClusterPreparedStatement
        assert _SR is ClusterStatementResult
        assert _SE is ClusterStatementExecutor
        from yggdrasil.databricks.cluster.statement import ClusterStatementBatch
        assert _SB is ClusterStatementBatch
