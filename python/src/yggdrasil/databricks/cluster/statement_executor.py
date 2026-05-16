"""Run SQL through a Databricks cluster as a :class:`StatementExecutor`.

The executor is a thin wrapper: it owns a :class:`Cluster` (which it
delegates the actual REPL command execution to) and a :class:`Volume`
(used to stage SELECT result data — see "SELECT handling" below). All
heavy lifting lives on the cluster's :class:`ExecutionContext` /
:class:`CommandExecution` plumbing; this class just adapts the
``yggdrasil.data.executor.StatementExecutor`` contract on top.

SELECT handling — INSERT OVERWRITE DIRECTORY
----------------------------------------------
A SQL REPL on a cluster ships results back as text — fine for DDL /
DML but unsafe for SELECTs (the REPL truncates above ~25 MiB and
serialises Python objects rather than streaming Arrow). The
warehouse path solves this with EXTERNAL_LINKS-disposition Arrow
chunks; the cluster path does not have that surface.

Instead, when the executor sees a query that
:meth:`PreparedStatement.looks_like_query` accepts, it rewrites it to

    INSERT OVERWRITE DIRECTORY '<volume>/<op_id>' USING parquet
    <original-query>

(the documented Databricks DML grammar — see the SQL reference at
https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-dml-insert-overwrite-directory).
The SELECT's rows are landed as Parquet under the bound volume; the
result reads them back through ``pyarrow.dataset`` from
:meth:`ClusterStatementResult._read_arrow_batches`. The staged
folder is bound to the statement so the batch's
``clear_temporary_resources`` walk unlinks it on success.

Safety / reliability guardrails baked in
----------------------------------------
- **No serverless command-execution path.** Databricks serverless
  compute exposes Spark Connect, not the REPL ``CommandExecution``
  endpoint (per the serverless limitations doc:
  https://docs.databricks.com/aws/en/compute/serverless/limitations).
  Constructing this executor against a :class:`ServerlessCluster`
  raises — the warehouse path is the right route for serverless SQL.
- **Execution-context reuse.** Each cluster caps at 145 user REPL
  contexts before new notebooks fail to attach
  (https://kb.databricks.com/clusters/too-many-execution-contexts-are-open-right-now).
  The executor keys the context off the bound volume's full path so
  every statement on the same volume shares one context. Callers
  that need isolation can pass an explicit ``context_key`` on the
  prepared statement.
- **Unity-Catalog-only staging.** Serverless restrictions push every
  user toward Unity Catalog volumes (no DBFS); the executor enforces
  this by typing the volume parameter as :class:`Volume` and using
  ``/Volumes/...`` paths exclusively.
- **No silent result-size truncation.** Because SELECTs go to
  Parquet on a volume, there is no REPL stdout cap that could
  truncate without raising — the Parquet read either returns every
  row or raises if the directory is malformed.
"""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from databricks.sdk.service.compute import Language

from yggdrasil.data.executor import StatementExecutor
from yggdrasil.data.statement import PreparedStatement

from ..resource import DatabricksResource
from .statement import (
    ClusterPreparedStatement,
    ClusterStatementBatch,
    ClusterStatementResult,
)

if TYPE_CHECKING:
    from yggdrasil.databricks.volume.volume import Volume

    from ..compute.command_execution import CommandExecution
    from .cluster import Cluster


__all__ = ["ClusterStatementExecutor"]


LOGGER = logging.getLogger(__name__)


# Staging-folder layout under the bound volume — Parquet output
# folders go under ``.sql/cluster/select/`` so a caller can drain
# stale entries with a single ``ls`` of that prefix without listing
# unrelated volume contents.
_OUTPUT_ROOT = ".sql/cluster/select"


def _new_op_id() -> str:
    """Unique-ish id for one cluster statement (monotonic + random tail)."""
    return f"cluster-{int(time.time() * 1000)}-{os.urandom(4).hex()}"


def _quote_dir(path: str) -> str:
    """Single-quote a Volumes path for use in an INSERT OVERWRITE DIRECTORY."""
    return "'" + path.replace("'", "''") + "'"


class ClusterStatementExecutor(DatabricksResource, StatementExecutor[
    ClusterPreparedStatement,
    ClusterStatementResult,
    ClusterStatementBatch,
]):
    """Cluster-backed :class:`StatementExecutor`.

    Wraps a :class:`Cluster` (the actual execution backend) and a
    :class:`Volume` (used to stage SELECT result Parquet folders).
    Inherits :meth:`execute` / :meth:`execute_many` / :meth:`batch`
    from the base — only :meth:`_submit_statement` is implemented
    locally, plus the SELECT-rewrite helper.

    Singleton-cached on ``(cluster, volume)`` so every statement
    against the same staging surface lands on one executor — same
    REPL context, same cluster session, same per-instance cache. The
    cluster cap of 145 user execution contexts is one of the main
    reasons for this: an unintended duplicate executor is an
    unintended second context.
    """

    _PREPARED_STATEMENT_CLASS: ClassVar[type[ClusterPreparedStatement]] = ClusterPreparedStatement
    _STATEMENT_RESULT_CLASS: ClassVar[type[ClusterStatementResult]] = ClusterStatementResult
    _STATEMENT_BATCH_CLASS: ClassVar[type[ClusterStatementBatch]] = ClusterStatementBatch

    _SINGLETON_TTL: ClassVar[Any] = None

    cluster: "Cluster"
    volume: "Volume"

    @classmethod
    def _singleton_key(
        cls,
        cluster: "Cluster | None" = None,
        volume: "Volume | None" = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Both :class:`Cluster` and :class:`Volume` are singleton-cached
        # ``DatabricksResource``s, so they hash by their UC identity
        # already. Pin both — same cluster + same volume ⇒ same
        # executor, same REPL context.
        return (cls, cluster, volume)

    def __init__(
        self,
        cluster: "Cluster",
        volume: "Volume",
        *,
        default_language: Language = Language.SQL,
        default_context_key: Optional[str] = None,
    ):
        if getattr(self, "_initialized", False):
            return

        # Late import to avoid the cluster module pulling the
        # serverless submodule at import time.
        from .serverless import ServerlessCluster

        if isinstance(cluster, ServerlessCluster):
            raise TypeError(
                f"{type(self).__name__} cannot wrap a ServerlessCluster: "
                "serverless compute does not expose the REPL CommandExecution "
                "API used by this executor (per Databricks serverless "
                "limitations). Route serverless SQL through SQLWarehouse "
                "instead."
            )

        # DatabricksResource expects a ``service``; route through the
        # cluster's own service so callers downstream that walk
        # ``executor.service.client`` land on the right client.
        super().__init__(service=cluster.service)
        self.cluster = cluster
        self.volume = volume
        self.default_language = default_language
        # Default context key: bind one REPL context per (cluster,
        # volume) pair so every statement against the same staging
        # surface re-uses the same context. Keeps us well under the
        # 145-context-per-cluster cap.
        self.default_context_key = (
            default_context_key
            or f"ygg-cluster-sql:{volume.full_name() if hasattr(volume, 'full_name') else id(volume)}"
        )
        self._initialized = True

    # ------------------------------------------------------------------ #
    # Identity helpers (drive __repr__ via DatabricksResource)
    # ------------------------------------------------------------------ #
    @property
    def explore_url(self):
        return self.cluster.explore_url

    # ------------------------------------------------------------------ #
    # Statement preparation
    # ------------------------------------------------------------------ #
    def _coerce_statement(
        self,
        statement: "ClusterPreparedStatement | PreparedStatement | str",
    ) -> ClusterPreparedStatement:
        coerced = super()._coerce_statement(statement)
        return self._rewrite_for_select(coerced)

    def _rewrite_for_select(
        self,
        statement: ClusterPreparedStatement,
    ) -> ClusterPreparedStatement:
        """Wrap a SELECT-like statement in ``INSERT OVERWRITE DIRECTORY``.

        Non-queries pass through untouched. Already-rewritten
        statements (``output_path`` is set) also pass through so
        :meth:`StatementResult.retry` doesn't double-wrap on a
        resubmission.
        """
        if statement.output_path is not None:
            return statement
        if not PreparedStatement.looks_like_query(statement.text):
            return statement

        op_id = _new_op_id()
        output_path = self.volume.path(f"{_OUTPUT_ROOT}/{op_id}")
        wrapped = (
            f"INSERT OVERWRITE DIRECTORY {_quote_dir(output_path.full_path())} "
            f"USING parquet "
            f"{statement.text}"
        )

        statement.text = wrapped
        statement.output_path = output_path
        return statement

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #
    def submit_command(self, statement: ClusterPreparedStatement) -> "CommandExecution":
        """Build a :class:`CommandExecution` from *statement* (no start).

        Exposed so :class:`ClusterStatementResult.start` can mint the
        command lazily — the result owns the lifecycle, the executor
        only owns the construction.
        """
        context_key = statement.context_key or self.default_context_key
        language = statement.language or self.default_language
        return self.cluster.command(
            context=context_key,
            command_str=statement.text,
            language=language,
        )

    def _submit_statement(
        self,
        statement: ClusterPreparedStatement,
        start: bool = True,
    ) -> ClusterStatementResult:
        result = ClusterStatementResult(
            executor=self,
            statement=statement,
        )
        if start:
            result.start(reset=False, wait=False, raise_error=False)
        return result
