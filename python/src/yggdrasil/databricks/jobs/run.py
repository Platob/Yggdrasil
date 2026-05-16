"""
Databricks JobRun resource — lifecycle for one job run.

Wraps a single Databricks run (either triggered by :meth:`Job.run` /
:meth:`Job.run_now` or submitted one-off via :meth:`Jobs.submit`).
Caches the :class:`Run` details and exposes:

- :meth:`refresh` — fetch fresh state from the SDK
- :meth:`wait_for_status` — poll until the run reaches a terminal state
- :meth:`cancel` / :meth:`delete` / :meth:`repair`
- :meth:`output` — pull the run output (notebook task output, error trace, …)
"""
from __future__ import annotations

import logging
import time
from typing import Any, ClassVar, List, Optional, TYPE_CHECKING

from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.jobs import (
    Run,
    RunLifeCycleState,
    RunOutput,
    RunResultState,
    RunState,
)

from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.io.url import URL

from ..client import DatabricksResource

if TYPE_CHECKING:
    from .service import Jobs


__all__ = ["JobRun"]

LOGGER = logging.getLogger(__name__)


_TERMINAL_LIFE_CYCLE_STATES: frozenset[RunLifeCycleState] = frozenset({
    RunLifeCycleState.TERMINATED,
    RunLifeCycleState.SKIPPED,
    RunLifeCycleState.INTERNAL_ERROR,
})

_PENDING_LIFE_CYCLE_STATES: frozenset[RunLifeCycleState] = frozenset({
    RunLifeCycleState.PENDING,
    RunLifeCycleState.QUEUED,
    RunLifeCycleState.RUNNING,
    RunLifeCycleState.TERMINATING,
    RunLifeCycleState.WAITING_FOR_RETRY,
    RunLifeCycleState.BLOCKED,
})


class JobRun(Singleton, DatabricksResource):
    """High-level wrapper around a single Databricks run.

    Parameters
    ----------
    service
        Parent :class:`~yggdrasil.databricks.jobs.service.Jobs` service.
    run_id
        Databricks run id.
    details
        Optional cached :class:`Run` snapshot. When omitted, details are
        fetched lazily on first access.

    Notes
    -----
    Inherits :class:`Singleton` (``_SINGLETON_TTL = None``) so two
    callers looking at the same run id share one instance — same cached
    snapshot, same polling state.
    """

    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        service: "Jobs | None" = None,
        run_id: int | None = None,
        **_kwargs: Any,
    ) -> Any:
        return (cls, service, run_id)

    def __init__(
        self,
        service: "Jobs | None" = None,
        run_id: int | None = None,
        *,
        details: Optional[Run] = None,
        singleton_ttl: Any = ...,
    ):
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return

        if service is None:
            from .service import Jobs
            service = Jobs.current()

        super().__init__(service=service)
        self.service = service
        self.run_id = run_id
        self._details = details
        self._initialized = True

    # ------------------------------------------------------------------ #
    # Identity / display
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.url().to_string()!r})"

    def __str__(self) -> str:
        return self.url().to_string()

    def url(self) -> URL:
        """Return the workspace UI URL for this run."""
        details = self._details
        page_url = getattr(details, "run_page_url", None) if details else None
        if page_url:
            return URL.from_str(page_url)
        return URL.from_str(
            f"{self.client.base_url.to_string().rstrip('/')}/jobs/runs/{self.run_id or 'unknown'}"
        )

    # ------------------------------------------------------------------ #
    # Details
    # ------------------------------------------------------------------ #
    @property
    def details(self) -> Optional[Run]:
        """Return cached run details, fetching lazily when not yet loaded."""
        if self._details is None and self.run_id is not None:
            self._details = self.service._jobs_api().get_run(run_id=self.run_id)
        return self._details

    def refresh(self) -> "JobRun":
        """Force-refresh cached run details and return self."""
        if self.run_id is None:
            return self
        self._details = self.service._jobs_api().get_run(run_id=self.run_id)
        return self

    @property
    def job_id(self) -> Optional[int]:
        details = self.details
        return details.job_id if details is not None else None

    @property
    def run_name(self) -> Optional[str]:
        details = self.details
        return details.run_name if details is not None else None

    @property
    def state(self) -> Optional[RunState]:
        """Return the cached :class:`RunState`. Refreshes once if missing."""
        details = self.details
        return details.state if details is not None else None

    @property
    def life_cycle_state(self) -> Optional[RunLifeCycleState]:
        state = self.state
        return state.life_cycle_state if state is not None else None

    @property
    def result_state(self) -> Optional[RunResultState]:
        state = self.state
        return state.result_state if state is not None else None

    @property
    def is_pending(self) -> bool:
        """True while the run is in a non-terminal lifecycle state."""
        return self.life_cycle_state in _PENDING_LIFE_CYCLE_STATES

    @property
    def is_terminal(self) -> bool:
        """True once the run has finished (succeeded, failed, skipped, …)."""
        return self.life_cycle_state in _TERMINAL_LIFE_CYCLE_STATES

    @property
    def is_successful(self) -> bool:
        """True when the run terminated with a SUCCESS result."""
        return (
            self.is_terminal
            and self.result_state == RunResultState.SUCCESS
        )

    # ------------------------------------------------------------------ #
    # Wait / cancel / delete
    # ------------------------------------------------------------------ #
    def wait_for_status(
        self,
        wait: WaitingConfigArg = True,
        *,
        raise_error: bool = False,
    ) -> "JobRun":
        """Poll until the run reaches a terminal state.

        When ``raise_error`` is set and the run terminates without
        ``SUCCESS``, a :class:`RuntimeError` carrying the state message
        is raised. The default mirrors the rest of the codebase: leave
        result interpretation to the caller.
        """
        wait_cfg = WaitingConfig.from_(wait)
        if not wait_cfg or self.run_id is None:
            return self

        start = time.time()
        iteration = 0

        while True:
            self.refresh()
            if self.is_terminal:
                break
            wait_cfg.sleep(iteration=iteration, start=start)
            iteration += 1

        if raise_error and not self.is_successful:
            state = self.state
            message = getattr(state, "state_message", None) or "<no message>"
            raise RuntimeError(
                f"Databricks run {self.run_id} terminated with "
                f"life_cycle_state={self.life_cycle_state!r} "
                f"result_state={self.result_state!r}: {message}"
            )

        return self

    def cancel(self, wait: WaitingConfigArg = True) -> "JobRun":
        """Cancel the run. When ``wait`` is truthy, blocks until terminal."""
        if self.run_id is None:
            return self

        LOGGER.debug("Cancelling job run %r", self)
        waiter = self.service._jobs_api().cancel_run(run_id=self.run_id)
        if wait:
            wait_cfg = WaitingConfig.from_(wait)
            waiter.result(timeout=wait_cfg.timeout_timedelta)
        self.refresh()
        LOGGER.info("Cancelled job run %r", self)
        return self

    def delete(self) -> None:
        """Delete the run record from the workspace."""
        if self.run_id is None:
            return
        LOGGER.debug("Deleting job run %r", self)
        try:
            self.service._jobs_api().delete_run(run_id=self.run_id)
        except ResourceDoesNotExist:
            LOGGER.debug("Job run %r already deleted", self)
        LOGGER.info("Deleted job run %r", self)

    # ------------------------------------------------------------------ #
    # Repair / output / export
    # ------------------------------------------------------------------ #
    def repair(
        self,
        *,
        rerun_all_failed_tasks: bool | None = None,
        rerun_dependent_tasks: bool | None = None,
        rerun_tasks: Optional[List[str]] = None,
        job_parameters: Optional[dict[str, str]] = None,
        notebook_params: Optional[dict[str, str]] = None,
        python_params: Optional[List[str]] = None,
        python_named_params: Optional[dict[str, str]] = None,
        jar_params: Optional[List[str]] = None,
        spark_submit_params: Optional[List[str]] = None,
        sql_params: Optional[dict[str, str]] = None,
        dbt_commands: Optional[List[str]] = None,
        latest_repair_id: int | None = None,
        wait: WaitingConfigArg = False,
        **repair_kwargs: Any,
    ) -> "JobRun":
        """Trigger a repair run via :meth:`JobsAPI.repair_run`.

        The same :class:`JobRun` is returned — repair runs share the
        parent run id.
        """
        if self.run_id is None:
            raise ValueError(f"Cannot repair {self}: run_id is not set")

        kwargs = {
            k: v for k, v in {
                "rerun_all_failed_tasks": rerun_all_failed_tasks,
                "rerun_dependent_tasks": rerun_dependent_tasks,
                "rerun_tasks": rerun_tasks,
                "job_parameters": job_parameters,
                "notebook_params": notebook_params,
                "python_params": python_params,
                "python_named_params": python_named_params,
                "jar_params": jar_params,
                "spark_submit_params": spark_submit_params,
                "sql_params": sql_params,
                "dbt_commands": dbt_commands,
                "latest_repair_id": latest_repair_id,
                **repair_kwargs,
            }.items()
            if v is not None
        }

        LOGGER.debug("Repairing job run %r with %s", self, kwargs)
        self.service._jobs_api().repair_run(run_id=self.run_id, **kwargs)

        if wait:
            self.wait_for_status(wait=wait)
        else:
            self.refresh()
        LOGGER.info("Repaired job run %r", self)
        return self

    def output(self) -> RunOutput:
        """Fetch the run output (notebook task result, error trace, logs, …)."""
        if self.run_id is None:
            raise ValueError(f"Cannot fetch output of {self}: run_id is not set")
        return self.service._jobs_api().get_run_output(run_id=self.run_id)

    def export(self, views_to_export: Any | None = None):
        """Export run views for sharing / archival via :meth:`JobsAPI.export_run`."""
        if self.run_id is None:
            raise ValueError(f"Cannot export {self}: run_id is not set")
        return self.service._jobs_api().export_run(
            run_id=self.run_id,
            views_to_export=views_to_export,
        )
