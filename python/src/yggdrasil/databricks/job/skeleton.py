"""Prefect-style tasks & flows that deploy as Databricks **serverless** jobs.

The :func:`task` / :func:`flow` decorators wrap a function into a **callable**
:class:`Task` / :class:`Flow`:

- ``.local(x)`` runs the body **in-process** (tests, debugging, the on-cluster
  body of a deployed job);
- ``.submit(...)`` runs in the background and returns a :class:`Future`
  (``.result()`` blocks) — a flow fans tasks out with ``.submit`` / ``.map``;
- ``.deploy(client)`` registers it as a Databricks **serverless** job for a
  schedule / file-arrival trigger, shipping the **live** code as a wheel.

    @flow(name="autoload", command=["databricks", "table", "autoload",
                                    "--table", "c.s.t", "--source", "s3://…"])
    def autoloader(): ...

    autoloader.deploy(client)    # register as a serverless job

The deploy ships the **live** code as a wheel (built from the package on disk —
dev checkout or installed), placed in the shared workspace pypi registry, or in
a per-user folder + rebuilt when the package is an editable install. The single
serverless python-wheel task invokes the ``ygg`` entry point with an explicit
:attr:`~Flow.command` — the CLI subcommand the deployed job runs on the cluster,
shipped **verbatim** as the wheel-task parameters (e.g. ``databricks table
autoload --table … --source …`` for an Auto Loader job).

Class-based flows subclass :class:`Flow` and override :meth:`~Flow.run` (the
body), :attr:`~_Runnable.name`, :meth:`~_Runnable.command`, :meth:`~Flow.trigger`.
"""
from __future__ import annotations

import contextvars
import functools
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.concurrent.threading import ThreadJob
    from yggdrasil.databricks.job.job import Job

logger = logging.getLogger(__name__)

#: Active while a body runs **in-process** (``.local`` / ``.submit``). Nested
#: task/flow calls consult it so they run locally too instead of each
#: dispatching its own background run — the flow, not every task, is the unit.
_LOCAL_MODE: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "ygg_local_mode", default=False
)


def ensure_console_logging(name: str = "yggdrasil", level: int = logging.INFO) -> None:
    """Attach an INFO stdout handler to the *name* logger if it has none, so
    interactive deploys / job runs surface ygg logs (the default root config is
    WARNING-only). Idempotent and scoped — never touches the root logger."""
    lg = logging.getLogger(name)
    lg.setLevel(min(lg.level or level, level) if lg.level else level)
    if not any(isinstance(h, logging.StreamHandler) for h in lg.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        lg.addHandler(handler)


__all__ = [
    "Task",
    "Flow",
    "Future",
    "task",
    "flow",
    # legacy aliases
    "JobSkeleton",
    "TaskSkeleton",
    "CallableSkeleton",
    "job",
]

T = TypeVar("T")


def _render(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


class Future(Generic[T]):
    """Handle to a :meth:`Task.submit` / :meth:`Flow.submit` background run."""

    __slots__ = ("_job",)

    def __init__(self, job: "ThreadJob") -> None:
        self._job = job

    def result(self, timeout: Any = None, *, raise_error: bool = True) -> T:
        """Block until the run finishes and return its result."""
        return self._job.wait(timeout, raise_error=raise_error)

    wait = result

    @property
    def done(self) -> bool:
        return self._job.is_done


class _Runnable:
    """Shared run + retry + deploy surface for tasks/flows.

    A call runs **in-process** (:meth:`local` / :meth:`_call`, honouring
    :attr:`retries`). Deploying registers it as a serverless job whose single
    python-wheel task invokes the ``ygg`` entry point with an explicit
    :meth:`command` shipped verbatim as the task parameters."""

    fn: Optional[Callable]
    name: str
    retries: int
    retry_delay_seconds: float

    # -- serverless / wheel defaults (shared by Task and Flow) -----------
    package_name: str = "ygg"          # wheel that ships the single ``ygg`` entry point
    entry_point: str = "ygg"           # the only console script (e.g. ``ygg databricks …``)
    task_key: str = "ygg"
    #: The ``ygg`` CLI subcommand the deployed wheel-task runs on the cluster,
    #: shipped **verbatim** as the python-wheel task ``parameters`` (no prefix,
    #: no target ref). e.g. ``["databricks", "table", "autoload", "--table",
    #: "c.s.t", "--source", "s3://…"]``. Required to deploy — set it on the
    #: instance (function flows take ``command=`` / override :meth:`command`).
    _command: list[str] | None = None
    #: Job-level tags (key → value) carried onto :meth:`Jobs.create_or_update`,
    #: merged on top of the client's owner/product defaults. ``None`` adds none.
    job_tags: dict[str, str] | None = None
    serverless: bool = True
    environment_key: str = "default"
    #: Serverless environment version. ``None`` resolves at deploy time to match
    #: the local Python (see :func:`serverless_environment_version`); pin a string.
    environment_version: "str | None" = None
    #: Build + ship the live package as a wheel on deploy (so the cluster runs
    #: exactly this code). Set ``False`` to instead install published ``ygg``.
    build_wheel: bool = True
    #: Project extras pulled into the built wheel's metadata (``[databricks]`` so
    #: the bundled image carries its databricks runtime deps).
    wheel_extras: "tuple[str, ...]" = ("databricks",)
    #: Fallback when not shipping a wheel — published ``ygg`` (``[databricks]``).
    dependencies: "tuple[str, ...]" = ("ygg[databricks]",)
    #: Always-installed extras on top of the wheel / :attr:`dependencies`.
    extra_dependencies: "tuple[str, ...]" = ("databricks-sdk",)
    #: Attach a serverless environment for **every** supported Python (keyed
    #: ``py3XX``) on deploy, not just the local-matched ``default``. The build
    #: already produces a wheel per Python; this exposes them as job environments.
    all_environments: bool = False
    #: Bundle the **whole transitive dependency closure** as wheels and ship
    #: them all, so the serverless environment installs with **zero PyPI
    #: access** ("0 pip install") — instead of the project wheel + index
    #: requirements. Trades a larger one-time upload for an offline, fully
    #: reproducible env build. Mutually exclusive with :attr:`all_environments`
    #: (bundles target the deploy host's single Python).
    bundle_dependencies: bool = False
    #: Reference a reusable, named serverless **base environment** (a
    #: ``<name>.yml`` in the workspace, the same convention ``ygg databricks
    #: seed`` writes) instead of inlining the dependency list — the ygg image is
    #: written there once (create-or-update) and the job points at it by file
    #: path, so jobs share one cached env. ``None`` keeps the classic inline
    #: environment. Any user package layers on top as extra dependencies. Ignored
    #: when :attr:`all_environments` is set (the per-Python matrix stays inline).
    #: Falls back to inline if the env can't be written.
    base_environment_name: "str | None" = None

    _wheel_paths: "tuple[str, ...]" = ()
    #: The project base environment(s) built on :meth:`deploy` (default + the
    #: per-Python matrix when :attr:`all_environments`), referenced by path.
    _environment: "Any" = None
    _environment_matrix: "dict[str, Any] | None" = None
    _client: Any = None

    # -- execution -------------------------------------------------------
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """The body. Override it, or wrap a function with the decorator."""
        if self.fn is None:
            raise NotImplementedError(f"{type(self).__name__} has no run() body")
        return self.fn(*args, **kwargs)

    def _call(self, *args: Any, **kwargs: Any) -> Any:
        # Mark this (and everything it calls) as in-process, so a flow body's
        # nested task calls run here rather than each dispatching its own job.
        token = _LOCAL_MODE.set(True)
        try:
            attempt = 0
            while True:
                try:
                    return self.run(*args, **kwargs)
                except Exception:
                    if attempt >= self.retries:
                        raise
                    attempt += 1
                    if self.retry_delay_seconds:
                        time.sleep(self.retry_delay_seconds)
        finally:
            _LOCAL_MODE.reset(token)

    def local(self, *args: Any, **kwargs: Any) -> Any:
        """Run **in-process** (honouring :attr:`retries`), skipping the remote
        routing — the escape hatch for tests and the on-cluster runner."""
        return self._call(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the body **in-process** (honouring :attr:`retries`). A flow's
        nested task calls run here too. To run on Databricks, :meth:`deploy` it
        as a serverless job and trigger a run (the wheel-task invokes the
        ``ygg`` :meth:`command` on the cluster)."""
        return self._call(*args, **kwargs)

    def submit(self, *args: Any, **kwargs: Any) -> "Future":
        """Run in the background; return a :class:`Future`."""
        from yggdrasil.concurrent.threading import Job as ThreadCall

        return Future(ThreadCall.make(self._call, *args, **kwargs).fire_and_forget())

    def map(self, iterable: Any, **kwargs: Any) -> "list[Future]":
        """:meth:`submit` once per item — Prefect-style fan-out."""
        return [self.submit(item, **kwargs) for item in iterable]

    def with_options(self, **overrides: Any) -> Any:
        """Return a copy with attributes overridden."""
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone.__dict__.update(overrides)
        return clone

    def command(self) -> list[str] | None:
        """The ``ygg`` CLI subcommand the deployed wheel-task runs on the cluster
        (e.g. ``["databricks", "table", "autoload", "--table", …]``), shipped
        verbatim as the python-wheel task ``parameters``. ``None`` (the default)
        means no on-cluster command was configured — set ``command=`` on the
        flow / override this in a class-based flow."""
        return self._command

    # -- wheel / environment --------------------------------------------
    def wheel_package(self) -> str:
        """The top-level import package to wheel — where this task/flow is
        defined, so the deploy adapts to any project (the wheel is built from
        this package's *live* files on disk)."""
        target = self.fn if self.fn is not None else type(self)
        return target.__module__.split(".")[0]

    def _wheel_parameters(self) -> list[str]:
        """The wheel-task parameters passed to the ``ygg`` entry point — the
        configured :meth:`command` shipped verbatim. Raises if none was set
        (nothing to run on the cluster)."""
        cmd = self.command()
        if not cmd:
            raise ValueError(
                f"{type(self).__name__} {self.name!r} has no command() to deploy — "
                f"set `command=[...]` (the `ygg` CLI subcommand the wheel-task "
                f"runs, e.g. ['databricks', 'table', 'autoload', '--table', …]) "
                f"or override command() in a class-based flow."
            )
        return [_render(p) for p in cmd]

    def _project_spec(self) -> str:
        """The deploy target for this task's package: its on-disk project dir when
        discoverable (so it's built from source), else its distribution name (so
        it's fetched from PyPI) — the same local-path-or-PyPI rule the wheel /
        environment services use everywhere."""
        import importlib

        from yggdrasil.databricks.wheels.service import distribution_for, find_pyproject

        pkg = self.wheel_package()
        try:
            module = importlib.import_module(pkg)
            source = Path(module.__file__).resolve().parent
            pyproject = find_pyproject(source)
            if pyproject is not None:
                return str(pyproject.parent)
        except (ImportError, AttributeError, OSError) as exc:
            # Not importable / no ``__file__`` / no source tree on disk → ship
            # the published distribution from PyPI instead. (Narrow on purpose:
            # a bug like a missing import must surface, not silently degrade the
            # deploy to PyPI — which is how a `Path` NameError once shipped stale
            # published wheels instead of the live checkout.)
            logger.debug("no local source for %r (%s); using PyPI", pkg, exc)
        return distribution_for(pkg)

    def _serverless_dependencies(self, client: Any) -> list[str]:
        """Build this task's project base environment(s) via ``dbc.environments``
        and reference them. The environment bundles the project wheel + its whole
        dependency closure as wheels (zero-PyPI); :attr:`all_environments` builds
        one per supported Python. Returns the closure (the inline fallback)."""
        from yggdrasil.databricks.wheels.service import SUPPORTED_PYTHONS

        spec = self._project_spec()
        self._environment = client.environments.create(spec, extras=self.wheel_extras)
        self._environment_matrix = (
            {v: client.environments.create(spec, extras=self.wheel_extras, python=v)
             for v in SUPPORTED_PYTHONS}
            if self.all_environments else {}
        )
        return list(self._environment.dependencies)

    def effective_dependencies(self) -> list[str]:
        """Shipped wheels once :meth:`deploy` has composed them, else the
        published :attr:`dependencies` (``ygg`` pinned to the running version) +
        :attr:`extra_dependencies`."""
        if getattr(self, "_wheel_paths", None):
            return list(self._wheel_paths)
        return [self._pin(d) for d in self.dependencies] + list(self.extra_dependencies)

    @staticmethod
    def _pin(dependency: str) -> str:
        """Pin a bare ``ygg`` / ``ygg[...]`` requirement to the running version
        so the deployed job installs the same code."""
        if dependency == "ygg" or dependency.startswith("ygg["):
            from yggdrasil.version import __version__

            return f"{dependency}=={__version__}"
        return dependency

    def environments(self) -> Optional[list]:
        """Serverless environment list, or ``None`` when not serverless.

        After :meth:`deploy` built the project's base environment(s), the default
        env references the version-pinned ``.yml`` by path (``base_environment``);
        with :attr:`all_environments` one env per supported Python (keyed
        ``py3XX``) is appended so a task can run under any Python by key. Before a
        deploy (or for a non-built job) it falls back to an inline
        :meth:`effective_dependencies` list."""
        if not self.serverless:
            return None
        from databricks.sdk.service.compute import Environment
        from databricks.sdk.service.jobs import JobEnvironment
        from yggdrasil.databricks.wheels.service import (
            environment_key_for, serverless_environment_version,
        )

        env = getattr(self, "_environment", None)
        if env is not None:
            envs = [JobEnvironment(environment_key=self.environment_key,
                                   spec=Environment(base_environment=env.serverless))]
            for python, built in (getattr(self, "_environment_matrix", None) or {}).items():
                envs.append(JobEnvironment(environment_key=environment_key_for(python),
                                           spec=Environment(base_environment=built.serverless)))
            return envs

        return [JobEnvironment(
            environment_key=self.environment_key,
            spec=Environment(
                environment_version=self.environment_version or serverless_environment_version(),
                dependencies=self.effective_dependencies(),
            ),
        )]

    def tasks(self) -> list:
        """The single serverless python-wheel task that runs the ``ygg``
        :meth:`command` on the cluster (shipped verbatim as parameters)."""
        from databricks.sdk.service.jobs import PythonWheelTask, Task as DBTask

        return [
            DBTask(
                task_key=self.task_key,
                environment_key=(self.environment_key if self.serverless else None),
                python_wheel_task=PythonWheelTask(
                    package_name=self.package_name,
                    entry_point=self.entry_point,
                    parameters=self._wheel_parameters(),
                ),
            )
        ]

    def definition(self) -> dict:
        """Render the :meth:`Jobs.create_or_update` kwargs for this task/flow."""
        spec: dict[str, Any] = {"name": self.name, "tasks": self.tasks()}
        environments = self.environments()
        if environments is not None:
            spec["environments"] = environments
        if self.job_tags:
            spec["tags"] = dict(self.job_tags)
        return spec

    def deploy(self, client: Any) -> "Job":
        """Get-or-create the live :class:`Job` from :meth:`definition` (without
        running it). When :attr:`build_wheel` is set, ships the live package as
        wheels (:meth:`_serverless_dependencies`) so the cluster runs this code."""
        ensure_console_logging()  # so the deploy CRUD is visible interactively
        logger.info("deploying %s %r", type(self).__name__.lower(), self.name)
        if self.build_wheel and self.serverless:
            self._wheel_paths = tuple(self._serverless_dependencies(client))
        spec = self.definition()
        logger.info("create-or-update job %r", self.name)
        job = client.jobs.create_or_update(name=spec.pop("name"), **spec)
        logger.info("deployed job %r (id=%s)", self.name, getattr(job, "job_id", None))
        return job


class Task(_Runnable, Generic[T]):
    """A callable unit of work; also deployable as one databricks ``Task``."""

    def __init__(
        self,
        fn: Callable[..., T],
        *,
        name: Optional[str] = None,
        retries: int = 0,
        retry_delay_seconds: float = 0.0,
        tags: "tuple[str, ...]" = (),
        key: Optional[str] = None,
        depends_on: "tuple[str, ...] | list[str]" = (),
        entry_point: Optional[str] = None,
        package_name: Optional[str] = None,
        command: list[str] | tuple[str, ...] | None = None,
        **task_options: Any,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self.tags = tuple(tags)
        self.task_key = key or self.name
        self.depends_on = tuple(depends_on)
        self.task_options = task_options
        if entry_point is not None:
            self.entry_point = entry_point
        if package_name is not None:
            self.package_name = package_name
        if command is not None:
            self._command = list(command)
        functools.update_wrapper(self, fn)

    def to_task(self, parameters: list[str] | None = None) -> Any:
        """Render a databricks ``Task`` (python-wheel) with explicit *parameters*
        and dependency edges — for hand-built multi-task job DAGs (the single-task
        deploy uses :meth:`tasks`). *parameters* are the ``ygg`` CLI subcommand
        the wheel-task runs on the cluster, shipped verbatim (e.g. ``["databricks",
        "table", "autoload", …]``)."""
        from databricks.sdk.service.jobs import (
            PythonWheelTask,
            Task as DBTask,
            TaskDependency,
        )

        return DBTask(
            task_key=self.task_key,
            depends_on=([TaskDependency(task_key=d) for d in self.depends_on] or None),
            environment_key=(self.environment_key if self.serverless else None),
            python_wheel_task=PythonWheelTask(
                package_name=self.package_name,
                entry_point=self.entry_point,
                parameters=list(parameters or self.command() or []),
            ),
            **self.task_options,
        )


class Flow(_Runnable):
    """A callable flow; deploys as a Databricks **serverless** job."""

    def __init__(
        self,
        fn: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        trigger: Any = None,
        retries: int = 0,
        retry_delay_seconds: float = 0.0,
        command: tuple[str, ...] | list[str] | None = None,
        entry_point: Optional[str] = None,
        package_name: Optional[str] = None,
    ) -> None:
        self.fn = fn
        self.name = name or (fn.__name__ if fn is not None else type(self).__name__)
        self._trigger = trigger
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self._command = list(command) if command is not None else None
        self._wheel_paths = ()
        if entry_point is not None:
            self.entry_point = entry_point
        if package_name is not None:
            self.package_name = package_name
        if fn is not None:
            functools.update_wrapper(self, fn)

    # -- deploy surface (override in class-based flows) -----------------
    def trigger(self) -> Any:
        """The databricks ``TriggerSettings`` (file-arrival / schedule), or
        ``None``. Function-built flows carry the ``@flow(trigger=...)`` value."""
        return self._trigger

    def definition(self) -> dict:
        """:class:`_Runnable.definition` plus the schedule/file-arrival trigger."""
        spec = super().definition()
        trigger = self.trigger()
        if trigger is not None:
            spec["trigger"] = trigger
        return spec


# ---------------------------------------------------------------------------
# Decorators — wrap a function into a callable Task / Flow
# ---------------------------------------------------------------------------


def task(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    retries: int = 0,
    retry_delay_seconds: float = 0.0,
    key: Optional[str] = None,
    depends_on: "tuple[str, ...] | list[str]" = (),
    entry_point: Optional[str] = None,
    package_name: Optional[str] = None,
    **task_options: Any,
) -> Any:
    """Turn a function into a callable :class:`Task` (Prefect-style)."""

    def deco(f: Callable) -> Task:
        return Task(
            f,
            name=name,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            key=key,
            depends_on=depends_on,
            entry_point=entry_point,
            package_name=package_name,
            **task_options,
        )

    return deco(func) if callable(func) else deco


def flow(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    trigger: Any = None,
    retries: int = 0,
    retry_delay_seconds: float = 0.0,
    command: tuple[str, ...] | list[str] | None = None,
    entry_point: Optional[str] = None,
    package_name: Optional[str] = None,
) -> Any:
    """Turn a function into a callable :class:`Flow` (Prefect-style)."""

    def deco(f: Callable) -> Flow:
        return Flow(
            f,
            name=name,
            trigger=trigger,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            command=command,
            entry_point=entry_point,
            package_name=package_name,
        )

    return deco(func) if callable(func) else deco


# Legacy names kept so existing imports keep working.
JobSkeleton = Flow
TaskSkeleton = Task
CallableSkeleton = _Runnable
job = flow
