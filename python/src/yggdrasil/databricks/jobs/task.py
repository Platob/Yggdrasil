"""
JobTask — single-task lifecycle within a parent :class:`Job`.

Databricks doesn't expose tasks as a first-class API; tasks live inside
the parent job's settings. :class:`JobTask` round-trips every CRUD
operation through :meth:`Job.update` so the parent's task list stays
the source of truth.

For Python callables, :meth:`JobTask.from_callable` extracts the raw
source via :func:`inspect.getsource`, drops a self-contained
``main-<digest>.py`` script under
``/Workspace/Shared/.ygg/jobs/<task_key>/``, and wraps it in a
:class:`SparkPythonTask`. No pickling — the source is what runs.
:meth:`JobTask.decorate` (chained off :meth:`Job.task`) is the
decorator form.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import inspect
import json
import logging
import textwrap
from dataclasses import replace as _dc_replace
from typing import Any, Callable, List, Optional, Sequence, TYPE_CHECKING, Tuple, Union

from databricks.sdk.service.compute import Environment
from databricks.sdk.service.jobs import (
    JobEnvironment,
    NotebookTask,
    SparkPythonTask,
    Task,
)

from yggdrasil.version import __version__ as yggdrasil_version
from yggdrasil.dataclasses.safe_function import (
    describe_signature,
    format_signature,
)

from .introspect import (
    dependencies_to_pip_specs,
    sniff_script,
)

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient

    from .job import Job
    from .workspace_pypi import WorkspacePyPI


__all__ = [
    "JobTask",
    "DEFAULT_STAGING_ROOT",
    "DEFAULT_ENVIRONMENT_KEY",
    "DEFAULT_ENVIRONMENT_CLIENT",
    "DEFAULT_ENVIRONMENT_DEPENDENCIES",
    "stage_python_callable",
    "stage_python_notebook_callable",
]

LOGGER = logging.getLogger(__name__)

#: Default staging area for :meth:`JobTask.from_callable`. Lands
#: under the shared yggdrasil tree so identical sources staged by
#: different jobs (or different callers) collapse to the same
#: per-task-key folder. Final layout:
#: ``/Workspace/Shared/.ygg/jobs/<task_key>/main-<digest>.py``.
#: ``<me>`` is still resolved via :meth:`WorkspacePath._resolve_me`
#: when callers override with a user-scoped root.
DEFAULT_STAGING_ROOT = "/Workspace/Shared/.ygg/jobs"

#: ``environment_key`` auto-attached to staged Python tasks so they
#: run on serverless workspaces without the caller pre-declaring a
#: :class:`JobEnvironment`. The parent job's ``environments`` list is
#: extended with a matching :class:`JobEnvironment` on
#: :meth:`JobTask.create` when the key isn't already defined.
DEFAULT_ENVIRONMENT_KEY = "ygg-default"

#: Serverless client version paired with :data:`DEFAULT_ENVIRONMENT_KEY`.
#: Databricks requires a ``client`` pin on every serverless environment
#: spec; ``"5"`` is the current latest workspace default (Python 3.12
#: + the modern pip resolver), and matches what the workspace UI's
#: "Configure environment" dialog shows under "Environment version".
DEFAULT_ENVIRONMENT_CLIENT = "5"

#: Default pip dependencies for the auto-attached serverless
#: environment. The staged script imports
#: ``yggdrasil.dataclasses.safe_function.checkargs``, so the runner
#: needs ``ygg`` (the PyPI distribution name for the ``yggdrasil``
#: package) on its path — unpinned, so the workspace pulls the
#: latest release on each run. The ``[data,databricks]`` extras
#: bundle the dataframe engines (pandas/polars/numpy) and the
#: Databricks SDK so staged tasks can move frames and call the
#: workspace without a follow-up install step.
DEFAULT_ENVIRONMENT_DEPENDENCIES: List[str] = [f"ygg[data,databricks]=={yggdrasil_version}"]


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
        extra_dependencies: Optional[Sequence[str]] = None,
        sniffed_env_vars: Optional[Sequence[str]] = None,
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
        #: Pip requirement strings derived from sniffing the staged
        #: script's imports — merged into the matching
        #: :class:`JobEnvironment` on :meth:`create` so the cluster /
        #: serverless env can resolve them. Populated by
        #: :meth:`from_callable` when ``auto_dependencies=True``; the
        #: caller can also pass it directly to pin extra wheels.
        self.extra_dependencies: List[str] = (
            list(extra_dependencies) if extra_dependencies else []
        )
        #: Env var names the staged script reads via
        #: ``os.getenv("X")`` / ``os.environ[...]``. Surfaced for
        #: diagnostics so a caller can spot a missing ``spark_env_vars``
        #: entry before the run actually fails.
        self.sniffed_env_vars: List[str] = (
            list(sniffed_env_vars) if sniffed_env_vars else []
        )

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
        """Remove this task from the parent job (no-op if already absent).

        Routed through ``fields_to_remove=["tasks/<task_key>"]`` rather
        than a partial ``tasks=[…]`` update: the Jobs API documents
        that ``new_settings`` arrays are merged by their unique key
        (``task_key`` for tasks, ``job_cluster_key`` for clusters),
        so submitting a shortened task list would silently leave the
        deleted entry intact. ``fields_to_remove`` with the nested
        ``tasks/<key>`` path is the only supported deletion mechanism.
        """
        existing = self._existing_tasks()
        if not any(t.task_key == self.task_key for t in existing):
            LOGGER.debug("Job task %r already absent from %r", self, self.job)
            return
        LOGGER.debug("Deleting job task %r", self)
        self.job.update(fields_to_remove=[f"tasks/{self.task_key}"])
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
        ``environment_key`` and the task carries no
        :attr:`extra_dependencies`. Otherwise returns a new list that
        either appends a default :class:`JobEnvironment` for the key
        (when missing) or extends the existing entry's pip
        ``dependencies`` with the task's auto-sniffed requirements
        (when present and not already declared).
        """
        env_key = getattr(task, "environment_key", None)
        if not env_key:
            return None
        settings = self.job.settings
        existing: List[JobEnvironment] = list(
            (settings.environments if settings is not None else None) or []
        )
        extra = list(self.extra_dependencies or ())

        for idx, env in enumerate(existing):
            if getattr(env, "environment_key", None) != env_key:
                continue
            if not extra:
                return None
            merged_spec = _extend_env_dependencies(env, extra)
            if merged_spec is env:
                # Nothing new to add — leave the env list untouched.
                return None
            existing[idx] = merged_spec
            return existing

        existing.append(
            _default_job_environment(
                env_key,
                dependencies=[*DEFAULT_ENVIRONMENT_DEPENDENCIES, *extra],
            )
        )
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
        # Carry the sniffed-from-source state across so :meth:`create`
        # sees the same dependency union it would've gotten if the
        # caller had built the staged JobTask directly.
        self.extra_dependencies = _dedupe_preserve(
            [*self.extra_dependencies, *staged.extra_dependencies],
        )
        if not self.sniffed_env_vars:
            self.sniffed_env_vars = list(staged.sniffed_env_vars)
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
        auto_dependencies: bool = True,
        extra_dependencies: Optional[Sequence[str]] = None,
        exclude_modules: Sequence[str] = (),
        workspace_pypi: Union[bool, "WorkspacePyPI", None] = False,
        **kwargs: Any,
    ) -> "JobTask":
        """Stage *func*'s source + bound *args*/*kwargs* as a Python script.

        Extracts the source via :func:`inspect.getsource`, strips any
        decorator lines (the runner side has no ``@job.task(...).decorate``
        in scope), appends an invocation that passes *args* / *kwargs*
        as Python literals, and writes the result to a single ``.py``
        file under *staging_root* (default:
        ``/Workspace/Shared/.ygg/jobs/<task_key>/main-<digest>.py``).
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

        When ``auto_dependencies`` is true (default), :func:`sniff_imports`
        walks the rendered script's AST and the result is fed through
        :func:`dependencies_to_pip_specs` to derive a pinned pip
        requirement for every non-stdlib top-level import. Those specs
        land on :attr:`extra_dependencies` and get merged into the
        target ``JobEnvironment`` on :meth:`create`, so the staged
        runner finds every transitive package without the caller
        spelling them out by hand. Pass ``exclude_modules=(...)`` to
        drop a noisy import; ``extra_dependencies=(...)`` is unioned
        in unconditionally — useful for wheels that aren't imported
        from the function body itself (e.g. a CLI plugin loaded
        through entry points).

        ``workspace_pypi`` controls how local / editable distributions
        are handled. ``False`` (default) — emit the bare requirement
        and let the cluster's pip resolution surface the failure.
        ``True`` — instantiate
        :class:`~yggdrasil.databricks.jobs.workspace_pypi.WorkspacePyPI`
        with workspace defaults, build wheels for every editable /
        local import and upload them to the workspace-side simple
        index; the rendered spec becomes a PEP 440 direct reference
        (``project @ /Workspace/.../wheel.whl``). Pass a pre-built
        :class:`WorkspacePyPI` to pin a custom root or share the same
        index across many jobs.

        :func:`sniff_env_vars` runs alongside the import sniff; the
        result lands on :attr:`sniffed_env_vars` for diagnostics. Wire
        the names through ``Job.task(spark_env_vars=…)`` or via a
        ``spark_python_task.parameters`` payload — the auto-dep path
        deliberately doesn't mutate them since env-var provenance
        belongs in the job spec, not derived from the body.

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
        details, merged_specs, sniffed_env_var_names = stage_python_callable(
            job.client,
            func,
            *args,
            task_key=task_key,
            staging_root=staging_root,
            auto_dependencies=auto_dependencies,
            extra_dependencies=extra_dependencies,
            exclude_modules=exclude_modules,
            workspace_pypi=_resolve_workspace_pypi(workspace_pypi, job),
            **kwargs,
        )
        return cls(
            job=job,
            task_key=details.task_key,
            details=details,
            extra_dependencies=merged_specs,
            sniffed_env_vars=sniffed_env_var_names,
        )

def stage_python_callable(
    client: "DatabricksClient",
    func: Callable[..., Any],
    *args: Any,
    task_key: Optional[str] = None,
    staging_root: str = DEFAULT_STAGING_ROOT,
    auto_dependencies: bool = True,
    extra_dependencies: Optional[Sequence[str]] = None,
    exclude_modules: Sequence[str] = (),
    workspace_pypi: Optional["WorkspacePyPI"] = None,
    **kwargs: Any,
) -> Tuple[Task, List[str], List[str]]:
    """Render *func* to a workspace ``.py`` file and return its :class:`Task` spec.

    The job-independent half of :meth:`JobTask.from_callable` — only
    needs *client* (to write under :data:`DEFAULT_STAGING_ROOT`) and
    *func*. Returns ``(task, pip_specs, env_var_names)``: the built
    :class:`Task` (with ``spark_python_task`` pointed at the staged
    file and ``environment_key=DEFAULT_ENVIRONMENT_KEY``), the
    auto-derived pip dependencies, and the env-var names the body
    reads. Used by both :meth:`JobTask.from_callable` (job-bound
    staging path) and :meth:`AsyncInsertJob.settings` (job-spec
    auto-staging path).
    """
    from yggdrasil.databricks.fs.workspace_path import WorkspacePath

    key = task_key or func.__name__
    script = _render_callable_script(func, args, kwargs)
    # Content hash, not a random token: re-staging the same body
    # with the same bound args lands on the same path so the
    # workspace doesn't accumulate near-duplicate
    # ``<key>-<random>.py`` files across iterations. Hashes only
    # the bits the caller controls — the function source plus
    # invocation args — so volatile slots like the embedded
    # ``staged_at`` / ``yggdrasil_version`` metadata don't shift
    # the digest between otherwise-identical re-stagings.
    digest = _content_digest(func, args, kwargs)

    # Final layout: ``<staging_root>/<task_key>/main-<digest>.py``.
    # Grouping by ``task_key`` (not job name) keeps identical
    # sources staged by different jobs landing on the same file —
    # the digest disambiguates between revisions.
    path = WorkspacePath(
        f"{staging_root.rstrip('/')}/{key}/main-{digest}.py",
        service=client.workspaces,
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

    # SparkPythonTask has no widget surface — the only channel for
    # passing per-run values into the script is the ordered
    # ``parameters`` list (Databricks delivers it as ``sys.argv[1:]``).
    # Wire one placeholder per parameter the caller left unbound,
    # using ``{{job.parameters.<name>}}`` substitution so the Job's
    # own :class:`JobParameterDefinition`s feed in at run time. The
    # rendered script's ``sys.argv``-reading invocation positions
    # match this list 1:1.
    _, unbound_param_names = _classify_invocation_params(func, args, kwargs)
    spark_parameters = (
        [f"{{{{job.parameters.{name}}}}}" for name in unbound_param_names]
        if unbound_param_names
        else None
    )

    details = Task(
        task_key=key,
        description=description,
        spark_python_task=SparkPythonTask(
            python_file=path.full_path(),
            parameters=spark_parameters,
        ),
        environment_key=DEFAULT_ENVIRONMENT_KEY,
    )

    # One AST walk feeds both the auto-dep resolver and the
    # diagnostics env-var list.
    sniffed_modules, sniffed_env_vars = sniff_script(script)
    sniffed_env_var_names = sorted(sniffed_env_vars)
    derived_specs: list[str] = []
    if auto_dependencies:
        derived_specs = dependencies_to_pip_specs(
            sniffed_modules,
            exclude=exclude_modules,
            workspace_pypi=workspace_pypi,
        )
    merged_specs = _dedupe_preserve(
        [*(extra_dependencies or ()), *derived_specs],
    )

    LOGGER.debug(
        "Sniffed staged task %r imports=%r env_vars=%r — derived "
        "dependencies %r",
        key, sorted(sniffed_modules), sniffed_env_var_names, merged_specs,
    )

    return details, merged_specs, sniffed_env_var_names


def stage_python_notebook_callable(
    client: "DatabricksClient",
    func: Callable[..., Any],
    *args: Any,
    task_key: Optional[str] = None,
    staging_root: str = DEFAULT_STAGING_ROOT,
    auto_dependencies: bool = True,
    extra_dependencies: Optional[Sequence[str]] = None,
    exclude_modules: Sequence[str] = (),
    workspace_pypi: Optional["WorkspacePyPI"] = None,
    **kwargs: Any,
) -> Tuple[Task, List[str], List[str]]:
    """Render *func* as a Databricks ``.py`` notebook and return its :class:`Task` spec.

    Notebook-flavoured sibling of :func:`stage_python_callable`. The
    same signature metadata, captured-local inlining,
    ``@checkargs`` wrap, and pip-spec sniffing apply — but the staged
    object is a Databricks-format ``.py`` notebook source (cells
    separated by ``# COMMAND ----------`` after a ``# Databricks
    notebook source`` magic header) and the returned :class:`Task`
    points at it via :class:`NotebookTask` instead of
    :class:`SparkPythonTask`. The imports / metadata block, captured
    locals, decorated function definition, and runtime invocation
    each land in their own cell so the Databricks UI surfaces
    stdout, exceptions, and ``LOGGER`` lines under the cell that
    produced them — much cleaner for diagnosing an applier run than
    the single-stream output of a Python task.

    The invocation cell renders bound *args* / *kwargs* as ``repr``'d
    literals exactly like the script path; unbound parameters fall
    back to ``dbutils.widgets.get(name)`` at runtime so Databricks
    job parameters flow through automatically (a tiny
    ``_yggdrasil_widget`` helper is emitted inline to swallow
    ``NameError`` for local re-runs outside a notebook host).
    """
    from yggdrasil.databricks.fs.workspace_path import WorkspacePath

    key = task_key or func.__name__
    source = _render_callable_notebook(func, args, kwargs)
    digest = _content_digest(func, args, kwargs)

    # ``.py`` extension paired with the ``# Databricks notebook
    # source`` magic header on line 1 — the workspace ``import``
    # path's ``format=AUTO`` detection routes that combination to
    # the notebook importer (per the SDK's content sniff), so the
    # uploaded object lands as a notebook rather than a workspace
    # file. Databricks strips the ``.py`` extension when storing
    # the notebook; :class:`NotebookTask` points at the bare path
    # below.
    upload_path = WorkspacePath(
        f"{staging_root.rstrip('/')}/{key}/main-{digest}.py",
        service=client.workspaces,
    )
    LOGGER.debug(
        "Staging callable %r as Databricks notebook at %r",
        func.__qualname__, upload_path,
    )
    upload_path.write_bytes(source.encode())

    signature_str = format_signature(describe_signature(func))
    doc_line = (func.__doc__ or "").strip().splitlines()[0:1]
    description = (
        f"{doc_line[0]} — {signature_str}" if doc_line else signature_str
    )[:1000]

    notebook_path = upload_path.full_path()
    if notebook_path.endswith(".py"):
        # Notebooks are referenced without the source extension —
        # Databricks stores ``main-<digest>.py`` as a notebook at
        # ``main-<digest>`` and the Jobs API resolves the bare
        # workspace path.
        notebook_path = notebook_path[:-3]

    details = Task(
        task_key=key,
        description=description,
        notebook_task=NotebookTask(notebook_path=notebook_path),
        environment_key=DEFAULT_ENVIRONMENT_KEY,
    )

    sniffed_modules, sniffed_env_vars = sniff_script(source)
    sniffed_env_var_names = sorted(sniffed_env_vars)
    derived_specs: list[str] = []
    if auto_dependencies:
        derived_specs = dependencies_to_pip_specs(
            sniffed_modules,
            exclude=exclude_modules,
            workspace_pypi=workspace_pypi,
        )
    merged_specs = _dedupe_preserve(
        [*(extra_dependencies or ()), *derived_specs],
    )

    LOGGER.debug(
        "Sniffed staged notebook %r imports=%r env_vars=%r — derived "
        "dependencies %r",
        key, sorted(sniffed_modules), sniffed_env_var_names, merged_specs,
    )

    return details, merged_specs, sniffed_env_var_names


def _content_digest(
    func: Callable[..., Any], args: tuple, kwargs: dict,
) -> str:
    """Return a short blake2b digest of *func*'s source + bound call args.

    Stable across re-stagings of the same body and identical
    invocation — keyword order is canonicalised — so the workspace
    path stays put when the only churn is the embedded
    ``staged_at`` / ``yggdrasil_version`` slots inside the rendered
    script.
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        # Falls back to the qualified name + module — collision-prone
        # but matches the failure surface of ``inspect.getsource`` in
        # :func:`_render_callable_script`, which will raise on the
        # subsequent render.
        source = f"{func.__module__}.{func.__qualname__}"
    h = hashlib.blake2b(digest_size=4)
    h.update(source.encode("utf-8"))
    h.update(b"\0args\0")
    h.update(repr(args).encode("utf-8"))
    h.update(b"\0kwargs\0")
    h.update(repr(sorted(kwargs.items())).encode("utf-8"))
    return h.hexdigest()


def _default_job_environment(
    environment_key: str,
    *,
    dependencies: Optional[Sequence[str]] = None,
) -> JobEnvironment:
    """Build a minimal serverless :class:`JobEnvironment` for *environment_key*.

    Databricks' serverless backend rejects Python tasks unless the
    parent job declares a matching ``environments`` entry with a
    ``client`` pin (``Environment.spec``). The default pulls in
    :data:`DEFAULT_ENVIRONMENT_DEPENDENCIES` (``ygg[data,databricks]``
    from PyPI) so the staged script's ``from yggdrasil...`` imports
    resolve at runtime. Pass *dependencies* to override the list —
    used by :meth:`JobTask.from_callable`'s auto-dep path to fold in
    the sniffed pip requirements at environment-creation time.
    """
    deps = list(dependencies) if dependencies is not None else list(
        DEFAULT_ENVIRONMENT_DEPENDENCIES,
    )
    return JobEnvironment(
        environment_key=environment_key,
        spec=Environment(
            client=DEFAULT_ENVIRONMENT_CLIENT,
            dependencies=_dedupe_preserve(deps),
        ),
    )


def _extend_env_dependencies(
    env: JobEnvironment, extra: Sequence[str],
) -> JobEnvironment:
    """Return *env* with *extra* unioned into its ``spec.dependencies``.

    Returns the same instance when every entry of *extra* is already
    declared — lets :meth:`JobTask._merged_environments` skip the
    job-side update when there's nothing to push.
    """
    spec = env.spec
    current = list((spec.dependencies if spec is not None else None) or [])
    merged = _dedupe_preserve([*current, *extra])
    if merged == current:
        return env
    return JobEnvironment(
        environment_key=env.environment_key,
        spec=Environment(
            client=(spec.client if spec is not None else DEFAULT_ENVIRONMENT_CLIENT),
            dependencies=merged,
        ),
    )


def _dedupe_preserve(items: Sequence[str]) -> List[str]:
    """Stable de-duplication used for pip-spec lists.

    ``dict.fromkeys`` preserves insertion order on 3.7+ and runs in C,
    so the common ``[x, y, x]`` case avoids the explicit ``set`` /
    ``list`` ping-pong.
    """
    return list(dict.fromkeys(item for item in items if item))


def _resolve_workspace_pypi(
    value: Union[bool, "WorkspacePyPI", None],
    job: "Job",
) -> Optional["WorkspacePyPI"]:
    """Coerce the ``workspace_pypi`` kwarg into a publisher or ``None``."""
    if value is None or value is False:
        return None
    if value is True:
        from .workspace_pypi import WorkspacePyPI
        return WorkspacePyPI(job.client)
    return value


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
    annotations via :func:`yggdrasil.data.cast.convert`.

    Walks *func*'s globals and closure cells to inline every locally
    defined helper function + literal constant the body references —
    so a script staged from a notebook / module that defines its own
    helper ``def`` blocks doesn't ``NameError`` on the runner side.
    Imported names are left alone (they ride through the auto-dep
    path); only same-module callables and literal values are inlined.
    Returns a UTF-8 ``str``; caller encodes for the workspace write.
    """
    from yggdrasil.version import __version__ as ygg_version

    captured_block = _capture_local_references(func)
    body = "@checkargs\n" + _function_source(func) + "\n"

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

    invocation_block = _render_invocation_script(func, args, kwargs)

    return (
        # ``from __future__ import annotations`` must lead every staged
        # file: the captured-locals block inlines class bodies verbatim
        # (e.g. ``AsyncInsertJob.JOB_NAME_PREFIX: ClassVar[str] = …``)
        # whose annotations the original source file evaluates lazily
        # via PEP 563. Without the future import in scope here the
        # annotation expressions run at class-body time and raise
        # ``NameError`` for typing names like ``ClassVar`` / ``Optional``
        # that the staged file doesn't otherwise import.
        "from __future__ import annotations\n"
        "\n"
        "# Auto-generated by yggdrasil.databricks.jobs.JobTask.from_callable.\n"
        f"# Function: {func.__qualname__}\n"
        f"# Signature: {format_signature(sig_meta)}\n"
        "# The function body below is the verbatim source of the decorated\n"
        "# callable, re-wrapped with @checkargs so every call site coerces\n"
        "# its inputs to the function's annotated types via\n"
        "# yggdrasil.data.cast.convert. Local helpers and constants the body\n"
        "# references are inlined above. Signature metadata is embedded\n"
        "# under __yggdrasil_task__.\n"
        "\n"
        "import json as _yggdrasil_json\n"
        "from yggdrasil.dataclasses.safe_function import checkargs\n"
        # ``ygg`` carries the runtime helpers the workflow layer leans
        # on — ygg.secret() resolves a SecretRef against the current
        # workspace client, ygg.task_value() reads a value an upstream
        # task published via dbutils.jobs.taskValues, ygg.publish_return()
        # wraps the staged invocation so this task's return surfaces on
        # the task-values map. Imported unconditionally so every staged
        # script can read SecretRef / TaskNode reprs as plain Python
        # without a special-case rendering hook. ``secret`` is also
        # re-imported here so a SecretRef expressed as a default
        # (``api_key: str = secret('vendor', 'key')``) parses at module
        # load — the staged invocation always passes an explicit value
        # so the default itself never reaches the function body.
        "from yggdrasil.databricks.workflow import ygg\n"
        "from yggdrasil.databricks.workflow.resources import secret  # noqa: F401\n"
        f"__yggdrasil_task__ = _yggdrasil_json.loads(r\"\"\"{meta_json}\"\"\")\n"
        "\n"
        f"{captured_block}"
        f"{body}"
        "\n"
        f"{invocation_block}"
    )


#: Databricks notebook cell separator. The workspace import sniff
#: keys off ``# Databricks notebook source`` on line 1 and splits the
#: file into cells on each ``# COMMAND ----------`` line — same
#: convention the workspace UI emits when exporting a notebook to
#: ``.py`` source format.
_NOTEBOOK_HEADER = "# Databricks notebook source"
_NOTEBOOK_CELL_SEPARATOR = "\n# COMMAND ----------\n\n"


def _render_callable_notebook(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
) -> str:
    """Render *func* + bound *args* / *kwargs* as a Databricks ``.py`` notebook.

    Same body content as :func:`_render_callable_script` (signature
    metadata, captured locals, ``@checkargs``-wrapped function,
    runtime invocation) — just split across notebook cells. Cell
    layout:

    1. Markdown header (``# MAGIC %md``) describing what the
       notebook does and the captured signature.
    2. Imports + the ``__yggdrasil_task__`` metadata block.
    3. Captured local references — only emitted when *func* actually
       has same-module helpers / literal constants to carry across.
    4. The ``@checkargs``-wrapped function definition.
    5. Invocation. Bound *args* / *kwargs* render as literals;
       unbound parameters fall back to
       ``dbutils.widgets.get(name)`` so Databricks job parameters
       flow through automatically. A tiny ``_yggdrasil_widget``
       helper swallows ``NameError`` for local re-runs outside a
       notebook host.
    """
    from yggdrasil.version import __version__ as ygg_version

    captured_block = _capture_local_references(func)
    body = "@checkargs\n" + _function_source(func) + "\n"

    sig_meta = describe_signature(func)
    meta_payload = {
        **sig_meta,
        "yggdrasil_version": str(ygg_version),
        "staged_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    meta_json = json.dumps(meta_payload, indent=2, sort_keys=True)

    sig_str = format_signature(sig_meta)
    header_cell = (
        "# MAGIC %md\n"
        f"# MAGIC # {func.__qualname__}\n"
        "# MAGIC\n"
        "# MAGIC Auto-generated by "
        "``yggdrasil.databricks.jobs.stage_python_notebook_callable``.\n"
        f"# MAGIC Signature: ``{sig_str}``\n"
        "# MAGIC\n"
        "# MAGIC The body below is the verbatim source of the decorated\n"
        "# MAGIC callable, re-wrapped with ``@checkargs`` so every call\n"
        "# MAGIC site coerces its inputs to the function's annotated types\n"
        "# MAGIC via ``yggdrasil.data.cast.convert``. Signature metadata\n"
        "# MAGIC is embedded under ``__yggdrasil_task__``.\n"
    )

    # ``from __future__ import annotations`` leads the first executable
    # cell: the captured-locals block below inlines class bodies
    # verbatim (e.g. ``AsyncInsertJob.JOB_NAME_PREFIX: ClassVar[str] = …``)
    # whose annotations the original module evaluates lazily via
    # PEP 563. Databricks compiles the whole ``.py`` notebook source as
    # one ``exec(compile(f.read(), ...))`` block, so a future import in
    # the first code cell is in scope for every cell that follows —
    # without it the annotation expressions run at class-body time and
    # raise ``NameError`` for typing names (``ClassVar`` / ``Optional`` /
    # …) that the staged file doesn't otherwise import.
    # The ``logging.basicConfig`` line ensures the notebook's ``LOGGER``
    # output actually surfaces under each cell in the Databricks UI —
    # the workspace's default Python root logger is configured at
    # ``WARNING`` so a bare ``LOGGER.info(...)`` from the staged body
    # would otherwise vanish. The ``force=True`` kwarg replaces any
    # pre-existing handler (which Databricks may have installed) so
    # the formatter we want wins.
    metadata_cell = (
        "from __future__ import annotations\n"
        "\n"
        "import json as _yggdrasil_json\n"
        "import logging as _yggdrasil_logging\n"
        "from yggdrasil.dataclasses.safe_function import checkargs\n"
        # ``ygg`` carries the workflow runtime helpers — see the matching
        # script renderer for the rationale; imported in every staged
        # notebook so SecretRef / TaskNode reprs evaluate cleanly.
        "from yggdrasil.databricks.workflow import ygg\n"
        "from yggdrasil.databricks.workflow.resources import secret  # noqa: F401\n"
        f"__yggdrasil_task__ = _yggdrasil_json.loads(r\"\"\"{meta_json}\"\"\")\n"
        "_yggdrasil_logging.basicConfig(\n"
        "    level=_yggdrasil_logging.INFO,\n"
        "    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',\n"
        "    force=True,\n"
        ")\n"
        "_YGG_LOGGER = _yggdrasil_logging.getLogger('yggdrasil.task')\n"
        "_YGG_LOGGER.info(\n"
        "    'Loaded staged task %r (yggdrasil=%s, staged_at=%s)',\n"
        "    __yggdrasil_task__.get('qualname'),\n"
        "    __yggdrasil_task__.get('yggdrasil_version'),\n"
        "    __yggdrasil_task__.get('staged_at'),\n"
        ")\n"
    )

    invocation_cell = _render_invocation_cell(func, args, kwargs)

    cells: list[str] = [header_cell, metadata_cell]
    if captured_block:
        cells.append(captured_block.rstrip() + "\n")
    cells.append(body.rstrip() + "\n")
    cells.append(invocation_cell)

    return (
        _NOTEBOOK_HEADER + "\n"
        + _NOTEBOOK_CELL_SEPARATOR.join(c.rstrip() + "\n" for c in cells)
    )


def _classify_invocation_params(
    func: Callable[..., Any], args: tuple, kwargs: dict,
) -> Tuple[List[Tuple[str, Optional[str]]], List[str]]:
    """Split *func*'s parameters into ``(bound_parts, unbound_names)``.

    Returns:

    * ``bound_parts`` — list of ``(name, literal_repr)`` for parameters
      the caller supplied at stage time (positional or keyword);
      ``literal_repr`` is the ``repr``'d Python literal the rendered
      script should emit. ``name`` is ``None`` for positional-only
      slots so the invocation can keep them positional.
    * ``unbound_names`` — list (in signature order) of parameter names
      the caller left for runtime resolution. ``VAR_POSITIONAL`` /
      ``VAR_KEYWORD`` slots are skipped; trailing keyword overrides
      not matching a parameter name fall through to ``bound_parts``
      as ``(k, repr(v))``.

    The two renderers share this so the SparkPythonTask path
    (``stage_python_callable``) and the NotebookTask path
    (``stage_python_notebook_callable``) plumb job parameters
    consistently — the script wires ``unbound_names`` into
    ``SparkPythonTask.parameters`` placeholders + ``sys.argv`` reads,
    the notebook wires the same names into ``dbutils.widgets.get``.
    """
    sig = inspect.signature(func)
    remaining_kwargs = dict(kwargs)
    bound_parts: List[Tuple[str, Optional[str]]] = []
    unbound_names: List[str] = []
    pos_consumed = 0
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if pos_consumed < len(args) and p.kind in (
            p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
        ):
            bound_parts.append((name, repr(args[pos_consumed])))
            pos_consumed += 1
        elif name in remaining_kwargs:
            bound_parts.append((name, repr(remaining_kwargs.pop(name))))
        else:
            unbound_names.append(name)
    for k, v in remaining_kwargs.items():
        bound_parts.append((k, repr(v)))
    return bound_parts, unbound_names


def _render_invocation_cell(
    func: Callable[..., Any], args: tuple, kwargs: dict,
) -> str:
    """Render the notebook invocation cell — literals for bound args, widgets for the rest.

    Positional / keyword parameters supplied via *args* / *kwargs* at
    stage time render as ``repr``'d literals (same shape as the
    script path's single-line ``func(a=…, b=…)`` invocation).
    Parameters the caller left unbound — the common case for the
    async-insert applier, whose ``catalog_name`` / ``schema_name`` /
    ``table_name`` come from Databricks job parameters at run time —
    fall back to ``_yggdrasil_widget(name)`` which reads
    ``dbutils.widgets.get(name)`` and tolerates a missing ``dbutils``
    binding (returns ``None``) so a local ``python -m`` re-run still
    parses.
    """
    bound, unbound = _classify_invocation_params(func, args, kwargs)
    parts: list[str] = [
        f"{name}={literal}" if name is not None else literal
        for name, literal in bound
    ]
    parts.extend(f"{name}=_yggdrasil_widget({name!r})" for name in unbound)

    if not parts:
        invocation = f"{func.__name__}()"
    elif unbound:
        joined = ",\n        ".join(parts)
        invocation = f"{func.__name__}(\n        {joined},\n    )"
    else:
        invocation = f"{func.__name__}({', '.join(parts)})"

    # Wrap the invocation in a Starting / Finished log pair so the
    # Databricks UI's per-cell output makes it obvious *where* in the
    # task run a failure happened — a bare ``apply_records(...)``
    # cell that throws halfway shows only the traceback; the log
    # frame makes "what was about to happen" explicit.
    widget_helper = (
        "def _yggdrasil_widget(name):\n"
        "    # ``dbutils`` is injected by the Databricks notebook host;\n"
        "    # the fallback keeps a local ``python`` re-run from blowing\n"
        "    # up before the function body sees the value.\n"
        "    try:\n"
        "        return dbutils.widgets.get(name)  # type: ignore[name-defined]  # noqa: F821\n"
        "    except Exception:\n"
        "        return None\n"
        "\n"
    ) if unbound else ""

    return (
        f"{widget_helper}"
        f"_YGG_LOGGER.info('Starting %s', {func.__qualname__!r})\n"
        f"try:\n"
        # ``ygg.publish_return`` forwards the return value (transparent
        # passthrough) and, when ``dbutils`` is available, mirrors it
        # onto the run's task-values map so downstream tasks in the
        # same Databricks Job can read it back via ``ygg.task_value``
        # / ``dbutils.jobs.taskValues.get``. Silent no-op outside a
        # Databricks runtime so local re-runs of the staged source
        # still terminate normally.
        f"    _ygg_result = ygg.publish_return({invocation})\n"
        f"except Exception:\n"
        f"    _YGG_LOGGER.exception('Failed %s', {func.__qualname__!r})\n"
        f"    raise\n"
        f"_YGG_LOGGER.info('Finished %s', {func.__qualname__!r})\n"
    )


def _render_invocation_script(
    func: Callable[..., Any], args: tuple, kwargs: dict,
) -> str:
    """Render the script invocation block — sys.argv reads for unbound params.

    SparkPythonTask has no widget surface, so unbound parameters
    resolve at runtime by reading the matching positional slot from
    ``sys.argv`` — the staging path sets
    ``SparkPythonTask.parameters`` to a list of
    ``{{job.parameters.<name>}}`` placeholders in the same order this
    helper emits ``sys.argv[i]`` reads, so the Job's
    :class:`JobParameterDefinition`s feed straight through to the
    function call. Bound *args* / *kwargs* render as ``repr``'d
    literals.

    The returned block is indented for placement under
    ``if __name__ == "__main__":`` — the caller is responsible for
    the outer guard.
    """
    bound, unbound = _classify_invocation_params(func, args, kwargs)
    parts: list[str] = [
        f"{name}={literal}" if name is not None else literal
        for name, literal in bound
    ]
    parts.extend(
        f"{name}=_yggdrasil_argv({idx + 1})"
        for idx, name in enumerate(unbound)
    )

    if not parts:
        invocation = f"{func.__name__}()"
    elif unbound:
        joined = ",\n        ".join(parts)
        invocation = f"{func.__name__}(\n        {joined},\n    )"
    else:
        invocation = f"{func.__name__}({', '.join(parts)})"

    # ``ygg.publish_return`` wraps the invocation so the staged function's
    # return value lands on ``dbutils.jobs.taskValues`` for downstream
    # tasks to read via ``ygg.task_value``. Silent no-op outside Databricks
    # so local ``python <script>`` re-runs still terminate normally.
    if not unbound:
        return (
            'if __name__ == "__main__":\n'
            f"    ygg.publish_return({invocation})\n"
        )
    return (
        "def _yggdrasil_argv(idx):\n"
        "    # SparkPythonTask passes the task's ``parameters`` list as\n"
        "    # ``sys.argv[1:]`` — index lookups stay tolerant of a short\n"
        "    # argv so a local ``python <script>`` re-run still parses\n"
        "    # (the function body sees ``None`` for the missing slot).\n"
        "    import sys\n"
        "    return sys.argv[idx] if 0 <= idx < len(sys.argv) else None\n"
        "\n\n"
        'if __name__ == "__main__":\n'
        f"    ygg.publish_return({invocation})\n"
    )


def _function_source(func: Callable[..., Any]) -> str:
    """Return *func*'s source with decorator lines stripped.

    The runner side has no ``@job.task(...).decorate`` (or any other
    decorator from the caller's scope) available, and the metadata
    block / ``@checkargs`` wrap is re-applied at render time, so
    every leading ``@…`` line is dropped.
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as exc:  # built-ins, REPL-defined lambdas
        raise ValueError(
            f"Cannot stage {func!r} as a JobTask: inspect.getsource failed "
            f"({exc!s}). from_callable needs a function defined in an "
            "importable source file."
        ) from exc
    lines = source.splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    return "\n".join(lines).rstrip()


#: Types we know we can render via ``repr`` and get an ``eval``-safe
#: round-trip. Anything else (live handles, classes, generators) is
#: left out of the captured-constants block — the runner will hit a
#: ``NameError`` which is more honest than an import-time crash from
#: a non-roundtrippable value.
_INLINEABLE_LITERAL_TYPES = (
    int, float, complex, bool, str, bytes, type(None),
    list, tuple, dict, set, frozenset,
)


def _capture_local_references(func: Callable[..., Any]) -> str:
    """Inline every locally-defined helper / literal *func* depends on.

    Walks ``func``'s globals + closure cells, classifies each
    referenced name, and renders the ones we can carry across the
    process boundary:

    - **Same-module callables** (functions or classes whose
      ``__module__`` matches ``func.__module__``) are inlined via
      :func:`inspect.getsource`, then themselves walked transitively
      so the staged script is self-contained.
    - **Literal-like values** (numbers, strings, bytes, bool, ``None``,
      list / tuple / dict / set / frozenset of the same) emit
      ``NAME = <repr>`` assignments.
    - **Modules** and **imported callables** are skipped — the
      auto-dependency path already adds the matching pip requirement,
      and the staged script's existing ``import`` lines bring them
      back at runtime.

    Returns the rendered block (trailing newline included) ready to
    drop above the staged function body. Empty string when no local
    helpers / constants are referenced.
    """
    func_module = getattr(func, "__module__", None)
    rendered: list[str] = []
    rendered_names: set[str] = set()
    queue: list[Callable[..., Any]] = [func]
    visited: set[int] = {id(func)}

    while queue:
        current = queue.pop(0)
        current_module = getattr(current, "__module__", None)
        try:
            closure_vars = inspect.getclosurevars(current)
        except (TypeError, ValueError):
            continue
        # ``inspect.getclosurevars`` walks ``co_names`` (the names the
        # function loads via ``LOAD_GLOBAL`` *and* the attribute /
        # IMPORT_FROM targets it touches) and classifies every hit as
        # "global" — including names that the function body actually
        # imports locally (``from X import Y`` emits ``IMPORT_FROM`` on
        # ``co_names[Y]`` followed by ``STORE_FAST`` into
        # ``co_varnames[Y]``). Inlining those names re-emits a stale
        # copy of the class / helper *and* leaves its dependencies
        # (other module-level imports — ``contextlib``, ``LOGGER``,
        # ``time``, …) unbrought, so the staged file blows up at
        # class-body time. Drop anything whose name lives in the
        # function's locals: the staged callable's own
        # ``from … import …`` line handles it at runtime.
        local_names = set(current.__code__.co_varnames)
        candidates: dict[str, Any] = {
            name: value
            for name, value in {
                **closure_vars.globals, **closure_vars.nonlocals,
            }.items()
            if name not in local_names
        }
        for name, value in candidates.items():
            if name in rendered_names:
                continue
            if name == current.__name__ or name == func.__name__:
                # Recursive self-reference — the function's own ``def``
                # already covers it; skip to avoid re-emitting.
                continue

            kind = _classify_reference(value, owning_module=func_module)
            if kind == "skip":
                continue
            if kind == "literal":
                rendered.append(f"{name} = {value!r}")
                rendered_names.add(name)
                continue
            if kind == "inline":
                if id(value) in visited:
                    continue
                visited.add(id(value))
                try:
                    body = _function_source(value)
                except ValueError as exc:
                    LOGGER.debug(
                        "Skipping inline of %r (referenced by %r): %s",
                        name, current.__qualname__, exc,
                    )
                    continue
                rendered.append(body)
                rendered_names.add(name)
                queue.append(value)
                continue
        # Closure cells (free vars) — same classification, different
        # lookup path. ``getclosurevars`` already merges them under
        # ``nonlocals`` for the common case; this guard catches the
        # vars Python doesn't expose there (e.g. cells holding
        # unhashable values).
        if current.__closure__ and current.__code__.co_freevars:
            for fname, cell in zip(
                current.__code__.co_freevars, current.__closure__,
            ):
                if fname in rendered_names:
                    continue
                try:
                    value = cell.cell_contents
                except ValueError:
                    continue
                kind = _classify_reference(value, owning_module=func_module)
                if kind == "literal":
                    rendered.append(f"{fname} = {value!r}")
                    rendered_names.add(fname)

    if not rendered:
        return ""
    return (
        "# --- captured local references --- #\n"
        + "\n\n".join(rendered).rstrip() + "\n\n"
    )


def _classify_reference(value: Any, *, owning_module: Optional[str]) -> str:
    """Return ``"inline"`` / ``"literal"`` / ``"skip"`` for a referenced value.

    - Modules and builtins: ``skip`` — the staged script's own
      ``import`` lines cover them.
    - Functions / classes defined in the same module as the entry
      callable: ``inline`` — same source tree, safe to splice.
    - Functions / classes from other modules: ``skip`` — they ride
      through the auto-dep path; inlining would either duplicate or
      bring along incompatible dependencies.
    - Plain literal-shaped values: ``literal`` — emit via ``repr``.
    """
    if inspect.ismodule(value):
        return "skip"
    if inspect.isbuiltin(value):
        return "skip"
    if inspect.isfunction(value) or inspect.isclass(value):
        target_module = getattr(value, "__module__", None)
        if (
            target_module is not None
            and owning_module is not None
            and target_module == owning_module
        ):
            return "inline"
        return "skip"
    if isinstance(value, _INLINEABLE_LITERAL_TYPES):
        return "literal"
    return "skip"
