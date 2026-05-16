"""
JobTask — single-task lifecycle within a parent :class:`Job`.

Databricks doesn't expose tasks as a first-class API; tasks live inside
the parent job's settings. :class:`JobTask` round-trips every CRUD
operation through :meth:`Job.update` so the parent's task list stays
the source of truth.

For Python callables, :meth:`JobTask.from_callable` pickles the bound
``(func, args, kwargs)`` triple via :mod:`yggdrasil.pickle.dill`, drops
the pickle + a tiny runner script under the user's personal workspace
(``/Workspace/Users/<me>/.yggdrasil/jobs/``), and wraps the pair in a
:class:`SparkPythonTask`. :meth:`Job.task` is the decorator form.
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import replace as _dc_replace
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from databricks.sdk.service.jobs import SparkPythonTask, Task

if TYPE_CHECKING:
    from .job import Job


__all__ = ["JobTask", "DEFAULT_STAGING_ROOT"]

LOGGER = logging.getLogger(__name__)

#: Default staging area for :meth:`JobTask.from_callable`. The ``<me>``
#: segment is resolved by :class:`WorkspacePath` to the bound user's
#: workspace home so the same constant works across environments.
DEFAULT_STAGING_ROOT = "/Workspace/Users/<me>/.yggdrasil/jobs"

# Runner script staged alongside the pickle. Reads the pickle path from
# argv[1] (Databricks passes ``SparkPythonTask.parameters`` as argv tail)
# and calls the captured callable. Kept self-contained so the only
# yggdrasil dependency at run time is ``yggdrasil.pickle.dill``.
_RUNNER_SCRIPT = (
    b'"""yggdrasil JobTask runner. Loads a pickled (func, args, kwargs) '
    b'triple from sys.argv[1] and invokes it."""\n'
    b"import sys\n"
    b"from yggdrasil.pickle import dill\n"
    b"\n"
    b"def main() -> None:\n"
    b"    if len(sys.argv) < 2:\n"
    b'        raise SystemExit("usage: runner.py <pickle_path>")\n'
    b'    with open(sys.argv[1], "rb") as f:\n'
    b"        func, args, kwargs = dill.loads(f.read())\n"
    b"    result = func(*args, **kwargs)\n"
    b"    if result is not None:\n"
    b"        print(result)\n"
    b"\n"
    b'if __name__ == "__main__":\n'
    b"    main()\n"
)


class JobTask:
    """A single :class:`Task` bound to a parent :class:`Job`.

    Construction is cheap and never hits the API; pass ``details=None``
    when you only have a ``task_key`` and intend to :meth:`refresh`
    against an existing job-side task. All mutating operations
    (:meth:`create` / :meth:`update` / :meth:`delete`) push the parent
    job's full task list back through :meth:`Job.update`.
    """

    def __init__(
        self,
        job: "Job",
        task_key: str,
        details: Optional[Task] = None,
    ) -> None:
        self.job = job
        self.task_key = task_key
        self._details = details

    def __repr__(self) -> str:
        return (
            f"JobTask(job_id={self.job.job_id!r}, "
            f"task_key={self.task_key!r})"
        )

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    @property
    def details(self) -> Optional[Task]:
        """Return the cached :class:`Task`, fetching from the job on miss."""
        if self._details is None:
            self.refresh()
        return self._details

    def refresh(self) -> "JobTask":
        """Reload this task from the parent job's latest settings."""
        self.job.refresh()
        for t in self._existing_tasks():
            if t.task_key == self.task_key:
                self._details = t
                return self
        self._details = None
        return self

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def create(self) -> "JobTask":
        """Append this task to the parent job (raises on key collision)."""
        if self._details is None:
            raise ValueError(
                f"Cannot create {self!r}: details is None. Construct with a "
                "Task or build through :meth:`from_callable`."
            )
        existing = self._existing_tasks()
        if any(t.task_key == self.task_key for t in existing):
            raise ValueError(
                f"Task {self.task_key!r} already exists on {self.job!r}; "
                "call :meth:`update` instead."
            )
        LOGGER.debug("Creating job task %r on %r", self, self.job)
        self.job.update(tasks=[*existing, self._details])
        LOGGER.info("Created job task %r", self)
        return self

    def update(self, **fields: Any) -> "JobTask":
        """Replace fields on this task and push the new task list back."""
        if self._details is None:
            self.refresh()
        if self._details is None:
            raise ValueError(
                f"Cannot update {self!r}: task not found on {self.job!r}."
            )
        new_details = _dc_replace(self._details, **fields)
        existing = self._existing_tasks()
        updated: List[Task] = [
            new_details if t.task_key == self.task_key else t
            for t in existing
        ]
        LOGGER.debug(
            "Updating job task %r (fields=%r)", self, list(fields),
        )
        self.job.update(tasks=updated)
        self._details = new_details
        LOGGER.info("Updated job task %r", self)
        return self

    def delete(self) -> None:
        """Remove this task from the parent job (no-op if already absent)."""
        existing = self._existing_tasks()
        remaining = [t for t in existing if t.task_key != self.task_key]
        if len(remaining) == len(existing):
            LOGGER.debug("Job task %r already absent from %r", self, self.job)
            return
        LOGGER.debug("Deleting job task %r", self)
        self.job.update(tasks=remaining)
        LOGGER.info("Deleted job task %r", self)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _existing_tasks(self) -> List[Task]:
        settings = self.job.settings
        return list((settings.tasks if settings is not None else None) or [])

    # ------------------------------------------------------------------ #
    # Factory: from a Python callable
    # ------------------------------------------------------------------ #
    @classmethod
    def from_callable(
        cls,
        job: "Job",
        func: Callable[..., Any],
        *args: Any,
        task_key: Optional[str] = None,
        staging_root: str = DEFAULT_STAGING_ROOT,
        **kwargs: Any,
    ) -> "JobTask":
        """Pickle *func* + bound *args*/*kwargs* and wrap as a Task.

        Pickling goes through :mod:`yggdrasil.pickle.dill` so closures,
        lambdas, and bound methods round-trip. The pickle and a small
        runner script land under *staging_root* (default:
        ``/Workspace/Users/<me>/.yggdrasil/jobs/<task_key>-<rand>``)
        on the bound user's workspace.

        The returned :class:`JobTask` is **not** persisted on the job
        yet — call :meth:`create` (or decorate with :meth:`Job.task`,
        which does it for you). Compute is also caller-owned: layer
        ``new_cluster=`` / ``existing_cluster_id=`` / ``job_cluster_key=``
        through :meth:`update` once the task is registered.
        """
        from yggdrasil.pickle import dill
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath

        key = task_key or func.__name__
        suffix = secrets.token_hex(4)

        base = WorkspacePath(
            f"{staging_root.rstrip('/')}/{key}-{suffix}",
            client=job.client,
        )
        runner_path = base.joinpath("runner.py")
        pickle_path = base.joinpath("payload.pkl")

        LOGGER.debug(
            "Staging callable %r (runner=%r, pickle=%r)",
            func.__qualname__, runner_path, pickle_path,
        )
        runner_path.write_bytes(_RUNNER_SCRIPT)
        pickle_path.write_bytes(dill.dumps((func, args, kwargs)))

        doc = (func.__doc__ or "").strip()
        description = doc.splitlines()[0][:140] if doc else None

        details = Task(
            task_key=key,
            description=description,
            spark_python_task=SparkPythonTask(
                python_file=runner_path.full_path(),
                parameters=[pickle_path.full_path()],
            ),
        )
        return cls(job=job, task_key=key, details=details)
