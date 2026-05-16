"""
JobTask — single-task lifecycle within a parent :class:`Job`.

Databricks doesn't expose tasks as a first-class API; tasks live inside
the parent job's settings. :class:`JobTask` round-trips every CRUD
operation through :meth:`Job.update` so the parent's task list stays
the source of truth.

For Python callables, :meth:`JobTask.from_callable` extracts the raw
source via :func:`inspect.getsource`, drops a self-contained ``.py``
script under the user's personal workspace
(``/Workspace/Users/<me>/.ygg/jobs/``), and wraps it in a
:class:`SparkPythonTask`. No pickling — the source is what runs.
:meth:`JobTask.decorate` (chained off :meth:`Job.task`) is the
decorator form.
"""
from __future__ import annotations

import datetime as _dt
import inspect
import json
import logging
import secrets
import textwrap
from dataclasses import replace as _dc_replace
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from databricks.sdk.service.compute import Environment
from databricks.sdk.service.jobs import JobEnvironment, SparkPythonTask, Task

from yggdrasil.dataclasses.safe_function import (
    describe_signature,
    format_signature,
)

if TYPE_CHECKING:
    from .job import Job


__all__ = [
    "JobTask",
    "DEFAULT_STAGING_ROOT",
    "DEFAULT_ENVIRONMENT_KEY",
    "DEFAULT_ENVIRONMENT_CLIENT",
]

LOGGER = logging.getLogger(__name__)

#: Default staging area for :meth:`JobTask.from_callable`. Lands under
#: the bound user's workspace home; ``<me>`` is resolved to the
#: workspace client's current user (see
#: :meth:`WorkspacePath._resolve_me`).
DEFAULT_STAGING_ROOT = "/Workspace/Users/<me>/.ygg/jobs"

#: ``environment_key`` auto-attached to staged Python tasks so they
#: run on serverless workspaces without the caller pre-declaring a
#: :class:`JobEnvironment`. The parent job's ``environments`` list is
#: extended with a matching :class:`JobEnvironment` on
#: :meth:`JobTask.create` when the key isn't already defined.
DEFAULT_ENVIRONMENT_KEY = "ygg-default"

#: Minimum serverless client version paired with
#: :data:`DEFAULT_ENVIRONMENT_KEY`. Databricks requires a ``client``
#: pin on every serverless environment spec; ``"1"`` is the broadest
#: option and matches Databricks' own bundle examples.
DEFAULT_ENVIRONMENT_CLIENT = "1"


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
        *,
        order: Optional[int] = None,
    ) -> None:
        self.job = job
        self.task_key = task_key
        self._details = details
        #: Optional position to place this task at on :meth:`create` /
        #: :meth:`create`. ``None`` keeps the existing position
        #: (or appends when new). Honors Python list-slice indexing, so
        #: ``0`` lands first and ``-1`` lands second-to-last (insert
        #: semantics: ``lst[:order] + [t] + lst[order:]``).
        self.order = order

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
        """Append this task to the parent job — or update the existing entry.

        Idempotent: if a task with the same ``task_key`` already lives
        on the job, its entry is replaced in place (or moved when
        :attr:`order` is set); otherwise the task is inserted. Used by
        :meth:`JobTask.decorate` so re-decorating the same function
        during development doesn't raise — the staged source on the
        second pass overwrites the first task entry.
        """
        if self._details is None:
            raise ValueError(
                f"Cannot create {self!r}: details is None. Construct with a "
                "Task or build through :meth:`from_callable`."
            )
        existing = self._existing_tasks()
        replaced = any(t.task_key == self.task_key for t in existing)
        new_tasks = self._place(existing, self._details)

        LOGGER.debug(
            "%s job task %r on %r",
            "Updating" if replaced else "Creating", self, self.job,
        )
        update_kwargs: dict[str, Any] = {"tasks": new_tasks}
        merged_envs = self._merged_environments(self._details)
        if merged_envs is not None:
            update_kwargs["environments"] = merged_envs
        self.job.update(**update_kwargs)
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
        update_kwargs: dict[str, Any] = {"tasks": updated}
        merged_envs = self._merged_environments(new_details)
        if merged_envs is not None:
            update_kwargs["environments"] = merged_envs
        self.job.update(**update_kwargs)
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

    def _place(self, existing: List[Task], new_details: Task) -> List[Task]:
        """Build the new task list with *new_details* placed honoring ``self.order``.

        ``order is None`` keeps the existing task's position (replace
        in place) or appends when the key is new. An integer ``order``
        first strips any prior entry for ``self.task_key`` and inserts
        *new_details* at that slice index (``lst[:order] + [t] +
        lst[order:]``), so the same call both creates and reorders.
        """
        if self.order is None:
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
            return new_tasks
        others = [t for t in existing if t.task_key != self.task_key]
        return [*others[:self.order], new_details, *others[self.order:]]

    def _merged_environments(
        self, task: Task,
    ) -> Optional[List[JobEnvironment]]:
        """Return the parent job's ``environments`` extended with *task*'s key.

        Returns ``None`` when *task* doesn't reference an
        ``environment_key`` or when the key is already declared on the
        job — in both cases :meth:`Job.update` shouldn't touch the
        ``environments`` setting. Otherwise returns a new list that
        preserves the existing entries and appends a default
        :class:`JobEnvironment` so the serverless backend accepts the
        task on submit.
        """
        env_key = getattr(task, "environment_key", None)
        if not env_key:
            return None
        settings = self.job.settings
        existing: List[JobEnvironment] = list(
            (settings.environments if settings is not None else None) or []
        )
        if any(getattr(e, "environment_key", None) == env_key for e in existing):
            return None
        existing.append(_default_job_environment(env_key))
        return existing

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
        Pushes the result through :meth:`create` so a re-decoration
        replaces the previous entry in place.

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
        self.create()
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
        ``/Workspace/Users/<me>/.ygg/jobs/<task_key>-<rand>.py``).
        No pickling involved — the script Databricks runs is the exact
        source of the function.

        The returned task carries
        ``environment_key=DEFAULT_ENVIRONMENT_KEY`` so it submits cleanly
        on serverless-default workspaces;
        :meth:`create` lazily adds a matching
        :class:`JobEnvironment` to the parent job's ``environments``
        list when the key isn't already declared. Override with
        ``environment_key=...`` on the :meth:`Job.task` /
        :meth:`Job.pytask` call to pin a different serverless env, or
        replace with cluster-bound compute (``new_cluster=`` /
        ``existing_cluster_id=`` / ``job_cluster_key=``) via
        :meth:`Job.task` when running on classic clusters.

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
        — call :meth:`create` (or use :meth:`JobTask.decorate`, which
        does it for you). :meth:`create` is idempotent — same key
        replaces in place.
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

        # Description carries the formatted signature so the Databricks
        # UI surfaces "qualname(x: int = 5) -> str" without cracking the
        # script open; the docstring's first line is prepended when set.
        signature_str = format_signature(describe_signature(func))
        doc_line = (func.__doc__ or "").strip().splitlines()[0:1]
        description = (
            f"{doc_line[0]} — {signature_str}" if doc_line else signature_str
        )[:1000]

        details = Task(
            task_key=key,
            description=description,
            spark_python_task=SparkPythonTask(
                python_file=path.full_path(),
            ),
            environment_key=DEFAULT_ENVIRONMENT_KEY,
        )
        return cls(job=job, task_key=key, details=details)


def _default_job_environment(environment_key: str) -> JobEnvironment:
    """Build a minimal serverless :class:`JobEnvironment` for *environment_key*.

    Databricks' serverless backend rejects Python tasks unless the
    parent job declares a matching ``environments`` entry with a
    ``client`` pin (``Environment.spec``). The default carries no
    dependencies — callers that need extra packages should declare
    the environment themselves on the job.
    """
    return JobEnvironment(
        environment_key=environment_key,
        spec=Environment(
            client=DEFAULT_ENVIRONMENT_CLIENT,
            dependencies=[],
        ),
    )


def _render_callable_script(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
) -> str:
    """Render *func* + bound *args* / *kwargs* as a runnable ``.py`` script.

    Embeds a ``__yggdrasil_task__`` metadata block (signature, module,
    yggdrasil version, staging timestamp) and wraps the function with
    :func:`yggdrasil.dataclasses.safe_function.checkargs` so every
    call site — the staged invocation below and any future widget /
    argv re-entry — type-checks inputs against the function's
    annotations via :func:`yggdrasil.data.cast.convert`. Returns a
    UTF-8 ``str``; caller encodes for the workspace write.
    """
    from yggdrasil.version import __version__ as ygg_version

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
    # scope) available. We re-apply ``@checkargs`` ourselves below.
    lines = source.splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    body = "@checkargs\n" + "\n".join(lines).rstrip() + "\n"

    sig_meta = describe_signature(func)
    meta_payload = {
        **sig_meta,
        "yggdrasil_version": str(ygg_version),
        "staged_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    # JSON-encode for stable shape, then load via ``json.loads`` at
    # module import — keeps the metadata block readable as JSON
    # without Python tripping on ``null`` / ``true`` / ``false``.
    meta_json = json.dumps(meta_payload, indent=2, sort_keys=True)

    call_parts: list[str] = [repr(a) for a in args]
    call_parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
    invocation = f"{func.__name__}({', '.join(call_parts)})"

    return (
        "# Auto-generated by yggdrasil.databricks.jobs.JobTask.from_callable.\n"
        f"# Function: {func.__qualname__}\n"
        f"# Signature: {format_signature(sig_meta)}\n"
        "# The function body below is the verbatim source of the decorated\n"
        "# callable, re-wrapped with @checkargs so every call site coerces\n"
        "# its inputs to the function's annotated types via\n"
        "# yggdrasil.data.cast.convert. Signature metadata is embedded\n"
        "# under __yggdrasil_task__.\n"
        "\n"
        "import json as _yggdrasil_json\n"
        "from yggdrasil.dataclasses.safe_function import checkargs\n"
        f"__yggdrasil_task__ = _yggdrasil_json.loads(r\"\"\"{meta_json}\"\"\")\n"
        "\n"
        f"{body}"
        "\n"
        'if __name__ == "__main__":\n'
        f"    {invocation}\n"
    )
