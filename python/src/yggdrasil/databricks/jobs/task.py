"""
JobTask — single-task lifecycle within a parent :class:`Job`.

Databricks doesn't expose tasks as a first-class API; tasks live inside
the parent job's settings. :class:`JobTask` round-trips every CRUD
operation through :meth:`Job.update` so the parent's task list stays
the source of truth.

For Python callables, :meth:`JobTask.from_callable` extracts the raw
source via :func:`inspect.getsource`, drops a self-contained ``.py``
script under the user's personal workspace
(``/Workspace/Users/me/.yggdrasil/jobs/``), and wraps it in a
:class:`SparkPythonTask`. No pickling — the source is what runs.
:meth:`JobTask.decorate` (chained off :meth:`Job.task`) is the
decorator form.
"""
from __future__ import annotations

import inspect
import logging
import secrets
import textwrap
from dataclasses import replace as _dc_replace
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from databricks.sdk.service.jobs import SparkPythonTask, Task

if TYPE_CHECKING:
    from .job import Job


__all__ = ["JobTask", "DEFAULT_STAGING_ROOT"]

LOGGER = logging.getLogger(__name__)

#: Default staging area for :meth:`JobTask.from_callable`. Lands under
#: the bound user's workspace home.
DEFAULT_STAGING_ROOT = "/Workspace/Users/me/.yggdrasil/jobs"


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
                "call :meth:`update` or :meth:`create_or_update` instead."
            )
        LOGGER.debug("Creating job task %r on %r", self, self.job)
        self.job.update(tasks=[*existing, self._details])
        LOGGER.info("Created job task %r", self)
        return self

    def create_or_update(self) -> "JobTask":
        """Append this task — or replace the existing one with the same key.

        Used by :meth:`JobTask.decorate` so re-decorating the same
        function during development doesn't raise; the staged source
        on the second pass overwrites the first task entry in place.
        """
        if self._details is None:
            raise ValueError(
                f"Cannot create_or_update {self!r}: details is None. "
                "Construct with a Task or build through :meth:`from_callable`."
            )
        existing = self._existing_tasks()
        new_details = self._details
        replaced = False
        new_tasks: List[Task] = []
        for t in existing:
            if t.task_key == self.task_key:
                new_tasks.append(new_details)
                replaced = True
            else:
                new_tasks.append(t)
        if not replaced:
            new_tasks.append(new_details)

        LOGGER.debug(
            "%s job task %r on %r",
            "Updating" if replaced else "Creating", self, self.job,
        )
        self.job.update(tasks=new_tasks)
        LOGGER.info(
            "%s job task %r",
            "Updated" if replaced else "Created", self,
        )
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
    # Decorator: stage a Python callable onto this task
    # ------------------------------------------------------------------ #
    def decorate(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Stage *func*'s source onto this task and persist it on the job.

        Designed to be chained off :meth:`Job.task` as a decorator::

            @job.task("step_one", description="…").decorate
            def step_one(): ...

        Stages *func*'s raw source under the user's workspace via
        :meth:`from_callable`, then back-fills its derived defaults
        (``spark_python_task``, ``description`` from the docstring)
        onto :attr:`_details` *only where the caller didn't already
        set that field* through :meth:`Job.task`. Anything pre-set on
        the handle — ``spark_python_task=…``, ``description=…``,
        compute, dependencies, retries, environment_key — wins.
        Pushes the result through :meth:`create_or_update` so a
        re-decoration replaces the previous entry in place.

        Returns the original callable so the function stays usable
        in-process; the :class:`JobTask` handle is attached as
        ``func._job_task`` for downstream access.
        """
        staged = type(self).from_callable(
            self.job, func, task_key=self.task_key,
        )
        staged_details = staged._details
        assert staged_details is not None, (
            "JobTask.from_callable should always populate _details"
        )
        if self._details is None:
            self._details = staged_details
        else:
            # Caller-supplied fields win; decorate only fills in slots
            # the caller left as None on the pre-built Task.
            defaults = {
                k: v for k, v in vars(staged_details).items()
                if v is not None and getattr(self._details, k, None) is None
            }
            if defaults:
                self._details = _dc_replace(self._details, **defaults)
        self.create_or_update()
        func._job_task = self  # type: ignore[attr-defined]
        return func

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
        """Stage *func*'s source + bound *args*/*kwargs* as a Python script.

        Extracts the source via :func:`inspect.getsource`, strips any
        decorator lines (the runner side has no ``@job.task(...).decorate``
        in scope), appends an invocation that passes *args* / *kwargs*
        as Python literals, and writes the result to a single ``.py``
        file under *staging_root* (default:
        ``/Workspace/Users/me/.yggdrasil/jobs/<task_key>-<rand>.py``).
        No pickling involved — the script Databricks runs is the exact
        source of the function.

        *args* / *kwargs* are rendered via :func:`repr`, so they must be
        types whose ``repr`` round-trips through ``eval`` (built-in
        scalars, strings, tuples / lists / dicts of the same). Pass
        nothing at decoration time and let the function read its inputs
        from job parameters at run time when that's not enough.

        Limitations: ``inspect.getsource`` needs the function to live in
        an importable source file (no REPL-defined lambdas) and the body
        must be self-contained — closures, module-level globals, and
        decorators other than ``@job.task(...).decorate`` are NOT
        carried over.

        The returned :class:`JobTask` is not persisted on the job yet
        — call :meth:`create` / :meth:`create_or_update` (or use
        :meth:`JobTask.decorate`, which does it for you). Compute
        stays caller-owned: layer ``new_cluster`` / ``existing_cluster_id``
        / ``job_cluster_key`` via :meth:`update` once the task is
        registered.
        """
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath

        key = task_key or func.__name__
        suffix = secrets.token_hex(4)

        script = _render_callable_script(func, args, kwargs)

        path = WorkspacePath(
            f"{staging_root.rstrip('/')}/{key}-{suffix}.py",
            client=job.client,
        )

        LOGGER.debug(
            "Staging callable %r as raw source at %r",
            func.__qualname__, path,
        )
        path.write_bytes(script.encode())

        doc = (func.__doc__ or "").strip()
        description = doc.splitlines()[0][:140] if doc else None

        details = Task(
            task_key=key,
            description=description,
            spark_python_task=SparkPythonTask(
                python_file=path.full_path(),
            ),
        )
        return cls(job=job, task_key=key, details=details)


def _render_callable_script(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
) -> str:
    """Render *func* + bound *args* / *kwargs* as a runnable ``.py`` script.

    Returns a UTF-8 ``str``; caller encodes for the workspace write.
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as exc:  # built-ins, REPL-defined lambdas
        raise ValueError(
            f"Cannot stage {func!r} as a JobTask: inspect.getsource failed "
            f"({exc!s}). from_callable needs a function defined in an "
            "importable source file."
        ) from exc

    # Drop decorator lines preceding ``def`` — the runner has no
    # ``@job.task(...).decorate`` (or any other decorator from this
    # scope) available.
    lines = source.splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    body = "\n".join(lines).rstrip() + "\n"

    call_parts: list[str] = [repr(a) for a in args]
    call_parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
    invocation = f"{func.__name__}({', '.join(call_parts)})"

    return (
        "# Auto-generated by yggdrasil.databricks.jobs.JobTask.from_callable.\n"
        "# The function body below is the verbatim source of the decorated\n"
        f"# callable {func.__qualname__!r}; invocation is appended.\n"
        "\n"
        f"{body}"
        "\n"
        'if __name__ == "__main__":\n'
        f"    {invocation}\n"
    )
