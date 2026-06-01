"""Databricks JobRun resource — individual run lifecycle.

:class:`JobRun` implements :class:`~yggdrasil.dataclasses.awaitable.Awaitable`
so callers can ``run.wait()`` / ``run.cancel()`` with the same backoff and
timeout contract used by every other async surface in yggdrasil.

:class:`JobTask` is a thin read-only wrapper around
:class:`~databricks.sdk.service.jobs.RunTask` for inspecting per-task
state within a run.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, TYPE_CHECKING

from databricks.sdk.service.jobs import (
    Run as SDKRun,
    RunLifeCycleState,
    RunResultState,
    RunState,
    RunTask as SDKRunTask,
)
from yggdrasil.dataclasses.awaitable import Awaitable
from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.enums.state import State
from yggdrasil.url import URL

from .service import JobRuns, _is_numeric
from ..resource import DatabricksResource

if TYPE_CHECKING:
    from .dag import JobDag

__all__ = ["JobRun", "JobTask"]

LOGGER = logging.getLogger(__name__)

_LIFECYCLE_TO_STATE: dict[RunLifeCycleState, State] = {
    RunLifeCycleState.PENDING: State.PENDING,
    RunLifeCycleState.QUEUED: State.QUEUED,
    RunLifeCycleState.RUNNING: State.RUNNING,
    RunLifeCycleState.TERMINATING: State.RUNNING,
    RunLifeCycleState.TERMINATED: State.SUCCEEDED,
    RunLifeCycleState.SKIPPED: State.CANCELED,
    RunLifeCycleState.INTERNAL_ERROR: State.FAILED,
    RunLifeCycleState.BLOCKED: State.PENDING,
    RunLifeCycleState.WAITING_FOR_RETRY: State.PENDING,
}

_RESULT_TO_STATE: dict[RunResultState, State] = {
    RunResultState.SUCCESS: State.SUCCEEDED,
    RunResultState.FAILED: State.FAILED,
    RunResultState.CANCELED: State.CANCELED,
    RunResultState.TIMEDOUT: State.EXPIRED,
    RunResultState.SUCCESS_WITH_FAILURES: State.SUCCEEDED,
    RunResultState.EXCLUDED: State.CANCELED,
    RunResultState.DISABLED: State.CANCELED,
    RunResultState.UPSTREAM_CANCELED: State.CANCELED,
    RunResultState.UPSTREAM_FAILED: State.FAILED,
    RunResultState.MAXIMUM_CONCURRENT_RUNS_REACHED: State.REJECTED,
}

_TERMINAL_LIFECYCLE: frozenset[RunLifeCycleState] = frozenset({
    RunLifeCycleState.TERMINATED,
    RunLifeCycleState.SKIPPED,
    RunLifeCycleState.INTERNAL_ERROR,
})


def _resolve_state(run_state: RunState | None) -> State:
    if run_state is None:
        return State.IDLE

    lcs = run_state.life_cycle_state
    rs = run_state.result_state

    if lcs in _TERMINAL_LIFECYCLE and rs is not None:
        return _RESULT_TO_STATE.get(rs, State.FAILED)

    if lcs is not None:
        return _LIFECYCLE_TO_STATE.get(lcs, State.RUNNING)

    return State.IDLE


# ---------------------------------------------------------------------------
# JobRun
# ---------------------------------------------------------------------------


class JobRun(Singleton, DatabricksResource, Awaitable):
    """Individual Databricks job run — awaitable lifecycle handle.

    Parameters
    ----------
    service:
        Parent :class:`JobRuns` service.
    run_id:
        Databricks run id.  Accepts ``int`` or numeric ``str``.
    job_id:
        Owning job id.
    details:
        Pre-fetched SDK :class:`~databricks.sdk.service.jobs.Run`.

    Positional construction::

        JobRun(service, 98765)       # by run id
        JobRun(service, "98765")     # numeric string → by run id
    """

    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        service: JobRuns | None = None,
        run_id: "int | str | None" = None,
        job_id: "int | str | None" = None,
        **_kwargs: Any,
    ) -> Any:
        resolved = run_id
        if isinstance(resolved, str) and _is_numeric(resolved):
            resolved = int(resolved)
        return (cls, service, resolved)

    def __init__(
        self,
        service: JobRuns | None = None,
        run_id: "int | str | None" = None,
        job_id: "int | str | None" = None,
        *,
        details: SDKRun | None = None,
        singleton_ttl: Any = ...,
    ):
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return

        if service is None:
            service = JobRuns.current()

        # Coerce numeric strings
        if isinstance(run_id, str) and _is_numeric(run_id):
            run_id = int(run_id)
        if isinstance(job_id, str) and _is_numeric(job_id):
            job_id = int(job_id)

        super().__init__(service=service)
        self.service: JobRuns = service
        self.run_id: int | None = run_id
        self.job_id: int | None = job_id
        self._details: SDKRun | None = details

        if self._details is not None:
            self.run_id = self._details.run_id
            self.job_id = self._details.job_id
            self._state = _resolve_state(self._details.state)

        self._initialized = True

    def __str__(self) -> str:
        return f"JobRun({self.run_id}, state={self._state})"

    def __hash__(self):
        return hash((type(self), self.run_id))

    def __eq__(self, other):
        return isinstance(other, JobRun) and self.run_id == other.run_id

    @property
    def explore_url(self) -> URL:
        """Workspace UI URL for this run (``/jobs/<job_id>/runs/<run_id>``).

        Prefers the SDK-provided ``run_page_url`` when run details are
        already loaded — it carries the canonical workspace host and the
        owning ``job_id`` even for one-time submitted runs that have no
        job id of their own.  Otherwise the URL is built from the client
        host, falling back to the jobs list page when ``job_id`` isn't
        known yet (e.g. a handle built from a run id before its first
        :meth:`refresh`).
        """
        if self._details is not None and self._details.run_page_url:
            return URL.from_str(self._details.run_page_url)
        if self.job_id and self.run_id:
            return self.client.base_url.with_path(f"/jobs/{self.job_id}/runs/{self.run_id}")
        return self.client.base_url.with_path("/jobs")

    # ------------------------------------------------------------------ #
    # Details
    # ------------------------------------------------------------------ #

    @property
    def details(self) -> SDKRun | None:
        if self._details is None and self.run_id is not None:
            self.refresh()
        return self._details

    @property
    def name(self) -> str | None:
        d = self.details
        return d.run_name if d else None

    def refresh(self) -> "JobRun":
        sdk = self.client.workspace_client().jobs
        raw = sdk.get_run(run_id=self.run_id)
        self._details = raw
        self.job_id = raw.job_id
        self._state = _resolve_state(raw.state)
        return self

    @property
    def run_state(self) -> RunState | None:
        d = self.details
        return d.state if d else None

    @property
    def state_message(self) -> str | None:
        rs = self.run_state
        return rs.state_message if rs else None

    @property
    def result_state(self) -> RunResultState | None:
        rs = self.run_state
        return rs.result_state if rs else None

    @property
    def life_cycle_state(self) -> RunLifeCycleState | None:
        rs = self.run_state
        return rs.life_cycle_state if rs else None

    # ------------------------------------------------------------------ #
    # Duration / timing
    # ------------------------------------------------------------------ #

    @property
    def start_time_ms(self) -> int | None:
        d = self.details
        return d.start_time if d else None

    @property
    def end_time_ms(self) -> int | None:
        d = self.details
        return d.end_time if d else None

    @property
    def duration_ms(self) -> int | None:
        d = self.details
        return d.execution_duration if d else None

    @property
    def duration_seconds(self) -> float | None:
        ms = self.duration_ms
        return ms / 1000.0 if ms is not None else None

    # ------------------------------------------------------------------ #
    # Tasks
    # ------------------------------------------------------------------ #

    @property
    def tasks(self) -> list["JobTask"]:
        d = self.details
        raw_tasks = d.tasks if d else None
        if not raw_tasks:
            return []
        return [JobTask(t, run=self) for t in raw_tasks]

    def task(self, key: str) -> "JobTask | None":
        """The awaitable :class:`JobTask` for *key*, or ``None`` if absent.

        ``run.task("ingest").wait()`` blocks until that single task reaches a
        terminal state (by polling this run)."""
        for t in self.tasks:
            if t.task_key == key:
                return t
        return None

    def dag(self) -> "JobDag":
        """The run's task graph with live per-task state — see
        :class:`~yggdrasil.databricks.job.dag.JobDag`."""
        from .dag import JobDag

        d = self.details
        return JobDag.from_tasks(
            (d.tasks if d else None) or (),
            state_of=lambda t: _resolve_state(t.state),
        )

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def output(self) -> Any:
        sdk = self.client.workspace_client().jobs
        return sdk.get_run_output(run_id=self.run_id)

    def results(self) -> dict[str, Any]:
        """Best-effort per-task output: ``task_key`` → SDK ``RunOutput``.

        One ``get_run_output`` call per task run id; tasks whose output can't be
        fetched (no id yet, or an output-less task type) are simply omitted. The
        run must be waited on first for outputs to be meaningful."""
        sdk = self.client.workspace_client().jobs
        out: dict[str, Any] = {}
        for t in self.tasks:
            task_run_id = t.run_id
            if task_run_id is None:
                continue
            try:
                out[t.task_key] = sdk.get_run_output(run_id=task_run_id)
            except Exception:  # noqa: BLE001 - best-effort per-task output
                continue
        return out

    # ------------------------------------------------------------------ #
    # Awaitable contract
    # ------------------------------------------------------------------ #

    def _poll(self) -> None:
        self.refresh()

    def _start(self) -> None:
        pass

    def _error_for_status(self) -> BaseException | None:
        msg = self.state_message or f"Run {self.run_id} failed"
        rs = self.result_state
        if rs == RunResultState.CANCELED:
            return RuntimeError(f"Run {self.run_id} was cancelled: {msg}")
        if rs == RunResultState.TIMEDOUT:
            return TimeoutError(f"Run {self.run_id} timed out: {msg}")
        return RuntimeError(f"Run {self.run_id} failed ({rs}): {msg}")

    def _cancel(self) -> None:
        sdk = self.client.workspace_client().jobs
        LOGGER.debug("Cancelling run %s", self.run_id)
        sdk.cancel_run(run_id=self.run_id)
        self._state = State.CANCELED

    # ------------------------------------------------------------------ #
    # Repair
    # ------------------------------------------------------------------ #

    def repair(
        self,
        *,
        rerun_tasks: list[str] | None = None,
        wait: WaitingConfigArg = False,
        raise_error: bool = True,
    ) -> "JobRun":
        """Repair (rerun) failed tasks in this run.

        Parameters
        ----------
        rerun_tasks:
            Task keys to rerun.  ``None`` reruns all failed tasks.
        wait:
            Block until the repair finishes.
        raise_error:
            Raise on failure when waiting.
        """
        sdk = self.client.workspace_client().jobs

        LOGGER.debug("Repairing run %s (tasks=%s)", self.run_id, rerun_tasks)
        sdk.repair_run(
            run_id=self.run_id,
            rerun_tasks=rerun_tasks,
        )
        self._state = State.PENDING
        self._details = None

        LOGGER.info("Repair triggered for run %s", self.run_id)

        if wait is not False:
            self.wait(wait=wait, raise_error=raise_error)

        return self


# ---------------------------------------------------------------------------
# JobTask
# ---------------------------------------------------------------------------


class JobTask(Awaitable):
    """A single task within a job run — an awaitable handle.

    Built from a run's task list with a back-reference to the parent
    :class:`JobRun`, so :meth:`wait` (the :class:`Awaitable` contract) polls the
    run until *this* task reaches a terminal state. Constructed without a parent
    (``run=None``) it's a static read-only snapshot of the task at build time.

    The ``is_done`` / ``is_succeeded`` / ``is_failed`` / ``wait`` / ``cancel``
    surface comes from :class:`Awaitable`; :attr:`state` tracks the last polled
    task state. A task can't be cancelled in isolation — :meth:`_cancel` cancels
    the whole parent run.
    """

    def __init__(self, raw: SDKRunTask, run: "JobRun | None" = None):
        self._raw = raw
        self._run = run
        self._state = _resolve_state(raw.state)

    def __repr__(self) -> str:
        return f"JobTask({self.task_key!r}, state={self._state})"

    @property
    def raw(self) -> SDKRunTask:
        return self._raw

    @property
    def task_key(self) -> str:
        return self._raw.task_key

    @property
    def run(self) -> "JobRun | None":
        """The parent run this task belongs to (``None`` for a snapshot)."""
        return self._run

    @property
    def run_id(self) -> int | None:
        """This task's own run id (used to fetch its output)."""
        return self._raw.run_id

    @property
    def attempt_number(self) -> int | None:
        return self._raw.attempt_number

    @property
    def run_state(self) -> RunState | None:
        return self._raw.state

    @property
    def state_message(self) -> str | None:
        rs = self._raw.state
        return rs.state_message if rs else None

    @property
    def result_state(self) -> RunResultState | None:
        rs = self._raw.state
        return rs.result_state if rs else None

    @property
    def start_time_ms(self) -> int | None:
        return self._raw.start_time

    @property
    def end_time_ms(self) -> int | None:
        return self._raw.end_time

    @property
    def duration_ms(self) -> int | None:
        return self._raw.execution_duration

    @property
    def duration_seconds(self) -> float | None:
        ms = self.duration_ms
        return ms / 1000.0 if ms is not None else None

    @property
    def description(self) -> str | None:
        return self._raw.description

    @property
    def cluster_instance(self):
        return self._raw.cluster_instance

    # ------------------------------------------------------------------ #
    # Awaitable contract — poll the parent run for this task's state
    # ------------------------------------------------------------------ #

    def _start(self) -> None:
        """Tasks start when their run starts — nothing to trigger here."""

    def _poll(self) -> None:
        if self._run is None:
            self._state = _resolve_state(self._raw.state)
            return
        self._run.refresh()
        for t in (self._run.details.tasks or ()):
            if t.task_key == self.task_key:
                self._raw = t
                break
        self._state = _resolve_state(self._raw.state)

    def _error_for_status(self) -> BaseException | None:
        msg = self.state_message or f"Task {self.task_key!r} failed"
        return RuntimeError(
            f"Task {self.task_key!r} failed ({self.result_state}): {msg}"
        )

    def _cancel(self) -> None:
        if self._run is not None:
            self._run.cancel(wait=False, raise_error=False)
        self._state = State.CANCELED
        self._sleeper.set()
