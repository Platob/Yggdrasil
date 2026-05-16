"""Cluster-backed statement primitives.

Three concrete types layered on the abstractions in
:mod:`yggdrasil.data.statement`, mirroring the shape
:mod:`yggdrasil.databricks.warehouse.statement` exposes for the SQL
warehouse path:

- :class:`ClusterPreparedStatement` — adds language / context-key
  routing so a SQL string runs as ``Language.SQL`` on a reusable
  execution context by default.
- :class:`ClusterStatementResult` — tracks one
  :class:`CommandExecution` against a Databricks cluster: state,
  cancellation, error surfacing. For SELECT statements that were
  rewritten to ``INSERT OVERWRITE DIRECTORY`` by the executor, the
  result reads its Arrow stream back from the staged Parquet
  directory on the bound volume.
- :class:`ClusterStatementBatch` — re-uses the base batch contract;
  per-statement scratch cleanup runs when the batch finishes.

The result deliberately does *not* try to materialize SELECT output
inline (REPL stdout has a result-size cap that breaks above tens of
MB). Instead, the executor rewrites SELECTs to write Parquet to the
volume, and the result reads the Parquet folder back via
:class:`pyarrow.dataset` — the same shape the warehouse path uses
for external-link result streams.
"""
from __future__ import annotations

import logging
import time
from typing import Any, ClassVar, Iterable, Iterator, Optional, TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as pds

from databricks.sdk.service.compute import CommandStatus, Language

from yggdrasil.data.enums.state import State
from yggdrasil.data.statement import (
    PreparedStatement,
    StatementBatch,
    StatementResult,
)
from yggdrasil.dataclasses.waiting import WaitingConfigArg

if TYPE_CHECKING:
    from yggdrasil.databricks.fs.volume_path import VolumePath
    from yggdrasil.data.options import CastOptions

    from ..compute.command_execution import CommandExecution
    from .statement_executor import ClusterStatementExecutor

__all__ = [
    "ClusterPreparedStatement",
    "ClusterStatementResult",
    "ClusterStatementBatch",
]


LOGGER = logging.getLogger(__name__)


# Map Databricks command-status states onto the unified State enum.
# QUEUED behaves like PENDING; FINISHED is SUCCEEDED.
_CMD_TO_STATE: dict[CommandStatus, State] = {
    CommandStatus.QUEUED:    State.PENDING,
    CommandStatus.RUNNING:   State.RUNNING,
    CommandStatus.FINISHED:  State.SUCCEEDED,
    CommandStatus.CANCELLED: State.CANCELED,
    CommandStatus.ERROR:     State.FAILED,
}


class ClusterPreparedStatement(PreparedStatement):
    """SQL statement routed through a Databricks cluster.

    Adds two cluster-specific knobs:

    - ``language`` — defaults to ``Language.SQL``; the executor uses
      it to pick the REPL the command runs in.
    - ``context_key`` — keyed reuse of an :class:`ExecutionContext`
      on the cluster. ``None`` opts into the cluster-side default
      (the executor picks one keyed off the bound volume).
    - ``output_path`` — when non-``None``, the executor wrote
      ``INSERT OVERWRITE DIRECTORY <output_path>`` so the result can
      read its rows back from this Parquet folder. Set by the
      executor at prepare time; callers shouldn't pass it directly.
    """

    language: Optional[Language] = None
    context_key: Optional[str] = None
    output_path: Optional["VolumePath"] = None

    def __init__(
        self,
        text: str = "",
        *,
        key: Optional[str] = None,
        retry: Optional[WaitingConfigArg] = None,
        language: Optional[Language] = None,
        context_key: Optional[str] = None,
        output_path: Optional["VolumePath"] = None,
        **kwargs: Any,
    ):
        super().__init__(text, key=key, retry=retry, **kwargs)
        self.language = language or Language.SQL
        self.context_key = context_key
        self.output_path = output_path

    def clear_temporary_resources(self) -> None:
        """Unlink the staged output folder, if any."""
        path = self.output_path
        if path is None:
            return
        try:
            path.remove(missing_ok=True, recursive=True, wait=False)
        except Exception:
            LOGGER.exception(
                "Failed to clean up staged cluster output %r; continuing.",
                path,
            )
        self.output_path = None


# ---------------------------------------------------------------------------
# ClusterStatementResult
# ---------------------------------------------------------------------------


class ClusterStatementResult(StatementResult):
    """Single cluster command tracked as a :class:`StatementResult`.

    Wraps a :class:`CommandExecution` plus the SQL bound to it. The
    :meth:`start` / :meth:`cancel` / :meth:`refresh_status` hooks
    forward to the command; :meth:`_read_arrow_batches` reads the
    staged Parquet directory back when the statement was a SELECT
    rewritten through ``INSERT OVERWRITE DIRECTORY``.
    """

    _PREPARED_STATEMENT_CLASS: ClassVar[type[PreparedStatement]] = ClusterPreparedStatement

    executor: "ClusterStatementExecutor"
    statement: ClusterPreparedStatement
    command: Optional["CommandExecution"] = None

    def __init__(
        self,
        executor: "ClusterStatementExecutor",
        statement: Optional[ClusterPreparedStatement] = None,
        *,
        command: Optional["CommandExecution"] = None,
        **kwargs: Any,
    ):
        self.command = command
        super().__init__(statement=statement, executor=executor, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def started(self) -> bool:
        return self.command is not None and self.command.command_id is not None

    def start(
        self,
        reset: bool = False,
        *,
        wait: WaitingConfigArg = False,
        raise_error: bool = False,
        **kwargs: Any,
    ) -> "ClusterStatementResult":
        if self.started and not reset:
            return self

        if reset and self.started:
            self.cancel(wait=False)
            self.command = None

        self.command = self.executor.submit_command(self.statement)
        self.command.start()
        self.start_timestamp = self.start_timestamp or int(time.time())
        self.iteration += 1
        return self

    def cancel(
        self,
        wait: WaitingConfigArg = False,
        raise_error: bool = False,
        **kwargs: Any,
    ) -> "ClusterStatementResult":
        if self.command is None or not self.command.command_id:
            return self
        try:
            self.command.cancel(wait=wait, raise_error=raise_error)
        except Exception:
            LOGGER.exception("Failed to cancel cluster command %r", self.command)
        return self

    def refresh_status(self) -> "ClusterStatementResult":
        cmd = self.command
        if cmd is None or not cmd.command_id:
            return self
        try:
            cmd.refresh()  # type: ignore[attr-defined]
        except AttributeError:
            # CommandExecution exposes refresh implicitly via .state /
            # .details — calling .wait(wait=False, raise_error=False)
            # forces the underlying poll without blocking.
            try:
                cmd.wait(wait=False, raise_error=False)
            except Exception:
                LOGGER.debug("Status refresh swallowed for %r", cmd, exc_info=True)
        return self

    def _compute_state(self) -> State:
        cmd = self.command
        if cmd is None or not cmd.command_id:
            return State.PENDING
        status = getattr(cmd, "state", None)
        if status is None:
            details = getattr(cmd, "_details", None)
            status = getattr(getattr(details, "status", None), "value", None)
        try:
            return _CMD_TO_STATE.get(status, State.PENDING)
        except TypeError:
            return State.PENDING

    def _raise_for_status(self) -> None:
        cmd = self.command
        if cmd is None:
            return
        # Delegate to CommandExecution.raise_for_status — it builds
        # the engine-specific exception from the last details payload.
        cmd.raise_for_status()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Output / Arrow streaming
    # ------------------------------------------------------------------
    def _output_dataset(self) -> Optional[pds.Dataset]:
        path = self.statement.output_path
        if path is None:
            return None
        return pds.dataset(path.full_path(), format="parquet")

    def _read_arrow_batches(self, options: "CastOptions") -> Iterator[pa.RecordBatch]:
        ds = self._output_dataset()
        if ds is None:
            return iter(())
        return ds.to_batches()

    def _write_arrow_batches(
        self, batches: Iterable[pa.RecordBatch], options: "CastOptions"
    ) -> None:
        raise NotImplementedError(
            "ClusterStatementResult is read-only — writes go through "
            "ClusterStatementExecutor.execute()."
        )


# ---------------------------------------------------------------------------
# ClusterStatementBatch
# ---------------------------------------------------------------------------


class ClusterStatementBatch(StatementBatch):
    """Batch of cluster-backed statements.

    Inherits the base batch contract unchanged — submission goes
    through :meth:`ClusterStatementExecutor.submit_statement`; the
    batch only adds the typed result-class pin.
    """
