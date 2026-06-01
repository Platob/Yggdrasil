"""Prefect-style tasks & flows that deploy to Databricks serverless jobs.

The :func:`task` / :func:`flow` decorators wrap a function into a **callable**
:class:`Task` / :class:`Flow` — exactly like Prefect:

- call it to run locally (``my_flow(x)`` / ``my_task(x)``), with ``retries`` +
  ``retry_delay_seconds`` and ``.with_options(...)`` to tweak a copy;
- ``.submit(...)`` runs it in the background and returns a :class:`Future`
  (``.result()`` blocks) — a flow fans tasks out with ``.submit`` / ``.map``;
- a :class:`Flow` also ``.deploy()``s itself as a Databricks **serverless** job
  (v5 + ``ygg[databricks]``) whose single task runs the flow on the cluster.

    @task(retries=2)
    def fetch(url: str): ...

    @flow(name="etl")
    def etl(urls: list[str]):
        futures = fetch.map(urls)         # concurrent task runs
        return [f.result() for f in futures]

    etl(urls)                              # run locally
    etl.deploy(client.jobs)                # serverless Databricks job

Class-based flows subclass :class:`Flow` and override :meth:`~Flow.run` (the
body), :attr:`~Flow.name`, :meth:`~Flow.parameters`, :meth:`~Flow.trigger`.
"""
from __future__ import annotations

import functools
import logging
import re
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.concurrent.threading import ThreadJob
    from yggdrasil.databricks.job.job import Job
    from yggdrasil.databricks.job.service import Jobs

logger = logging.getLogger(__name__)


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
    """Shared retry + submit + with_options surface for tasks and flows."""

    fn: Optional[Callable]
    name: str
    retries: int
    retry_delay_seconds: float

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """The body. Override it, or wrap a function with the decorator."""
        if self.fn is None:
            raise NotImplementedError(f"{type(self).__name__} has no run() body")
        return self.fn(*args, **kwargs)

    def _call(self, *args: Any, **kwargs: Any) -> Any:
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run locally, like a plain function (honouring :attr:`retries`)."""
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


class Task(_Runnable, Generic[T]):
    """A callable unit of work; deploys as one databricks ``Task``."""

    #: wheel / serverless defaults (shared with :class:`Flow`).
    package_name: str = "ygg"
    entry_point: str = "ygg"
    serverless: bool = True
    environment_key: str = "default"

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
        functools.update_wrapper(self, fn)

    def to_task(self, parameters: "list[str] | None" = None) -> Any:
        """Render the databricks ``Task`` (python-wheel) for this task."""
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
                parameters=parameters or [],
            ),
            **self.task_options,
        )


class Flow(_Runnable):
    """A callable flow; deploys as a Databricks **serverless** job."""

    package_name: str = "ygg"
    entry_point: str = "ygg"
    task_key: str = "run"

    #: Serverless environment version (default **v5**).
    serverless: bool = True
    environment_key: str = "default"
    environment_version: str = "5"

    #: Fallback dependency when not shipping a built wheel — ``ygg`` (the
    #: published package, pulling its ``[databricks]`` extra) from an index.
    dependencies: "tuple[str, ...]" = ("ygg[databricks]",)

    #: Always-installed extras, on top of the wheel / :attr:`dependencies` —
    #: ``databricks-sdk`` (latest) so the runtime SDK is current. When building a
    #: wheel these are also bundled (so they ship as wheels, no index install).
    extra_dependencies: "tuple[str, ...]" = ("databricks-sdk",)

    #: Project extras to include when building the wheel with its dependencies
    #: (e.g. ``("databricks",)`` to pull the project's ``[databricks]`` extra).
    #: Default pulls ``[databricks]`` so the bundled ``ygg`` wheel carries the
    #: databricks deps it runs against (a no-op for projects without that extra).
    wheel_extras: "tuple[str, ...]" = ("databricks",)

    #: Build the project wheel **with its dependencies** on :meth:`deploy` and
    #: ship them all (``ygg`` + ``databricks-sdk`` latest + transitive deps) as
    #: workspace wheels — installed by path, no index access. Default ``True`` so
    #: the cluster runs exactly this code; set ``False`` to instead pip-install
    #: the published ``ygg`` (pinned to the running version) from an index.
    build_wheel: bool = True

    def __init__(
        self,
        fn: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        trigger: Any = None,
        retries: int = 0,
        retry_delay_seconds: float = 0.0,
        parameters: "tuple[str, ...] | list[str]" = (),
        entry_point: Optional[str] = None,
        package_name: Optional[str] = None,
    ) -> None:
        self.fn = fn
        self.name = name or (fn.__name__ if fn is not None else type(self).__name__)
        self._trigger = trigger
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self._parameters = tuple(parameters)
        self._wheel_paths: list[str] = []
        if entry_point is not None:
            self.entry_point = entry_point
        if package_name is not None:
            self.package_name = package_name
        if fn is not None:
            functools.update_wrapper(self, fn)

    # -- deploy surface (override in class-based flows) -----------------
    def parameters(self) -> list[str]:
        """Positional wheel parameters the deployed flow task receives."""
        return [_render(p) for p in self._parameters]

    def trigger(self) -> Any:
        """The databricks ``TriggerSettings`` (file-arrival / schedule), or
        ``None``. Function-built flows carry the ``@flow(trigger=...)`` value."""
        return self._trigger

    def wheel_dir(self) -> str:
        """Workspace folder the built wheel is uploaded to — named by the job
        (``/Workspace/Shared/.ygg/whl/<job-name>``) so each job owns its
        wheel."""
        from yggdrasil.databricks.job.wheel import WORKSPACE_WHL_DIR

        slug = re.sub(r"[^0-9A-Za-z._-]+", "_", self.name).strip("_") or "flow"
        return f"{WORKSPACE_WHL_DIR}/{slug}"

    def wheel_package(self) -> str:
        """The top-level import package to build when :attr:`build_wheel` is set
        — derived from where this flow is defined, so it adapts to any project.
        The wheel is synthesized from this package's *live* files on disk (no
        source checkout / published release needed). Override to build another."""
        target = self.fn if self.fn is not None else type(self)
        return target.__module__.split(".")[0]

    def effective_dependencies(self) -> list[str]:
        """The serverless dependencies. Once :meth:`deploy` has shipped wheels,
        it's the project wheel + every bundled dependency wheel (no index
        install); otherwise the published :attr:`dependencies` (``ygg`` **pinned
        to the running version** so the cluster gets exactly this code) +
        :attr:`extra_dependencies` names."""
        if getattr(self, "_wheel_paths", None):
            return list(self._wheel_paths)
        return [self._pin(d) for d in self.dependencies] + list(self.extra_dependencies)

    @staticmethod
    def _pin(dependency: str) -> str:
        """Pin a bare ``ygg`` / ``ygg[...]`` requirement to the running version
        so the deployed job installs the same code (avoids a stale cached one)."""
        if dependency == "ygg" or dependency.startswith("ygg["):
            from yggdrasil.version import __version__

            return f"{dependency}=={__version__}"
        return dependency

    def environments(self) -> Optional[list]:
        """Serverless environment list (v5 + :meth:`effective_dependencies`),
        or ``None``."""
        if not self.serverless:
            return None
        from databricks.sdk.service.compute import Environment
        from databricks.sdk.service.jobs import JobEnvironment

        return [
            JobEnvironment(
                environment_key=self.environment_key,
                spec=Environment(
                    environment_version=self.environment_version,
                    dependencies=self.effective_dependencies(),
                ),
            )
        ]

    def tasks(self) -> list:
        """The flow's single serverless python-wheel task."""
        from databricks.sdk.service.jobs import PythonWheelTask, Task as DBTask

        return [
            DBTask(
                task_key=self.task_key,
                environment_key=(self.environment_key if self.serverless else None),
                python_wheel_task=PythonWheelTask(
                    package_name=self.package_name,
                    entry_point=self.entry_point,
                    parameters=self.parameters(),
                ),
            )
        ]

    def definition(self) -> dict:
        """Render the :meth:`Jobs.create_or_update` kwargs for this flow."""
        spec: dict[str, Any] = {"name": self.name, "tasks": self.tasks()}
        environments = self.environments()
        if environments is not None:
            spec["environments"] = environments
        trigger = self.trigger()
        if trigger is not None:
            spec["trigger"] = trigger
        return spec

    def deploy(self, client: Any) -> "Job":
        """Get-or-create the live :class:`Job` from :meth:`definition`.

        Takes a :class:`DatabricksClient` and resolves its jobs service
        (``client.jobs``). When :attr:`build_wheel` is set, synthesizes a project
        from this flow's *live* package and builds it (with deps) into wheels,
        uploaded to ``/Workspace/Shared/.ygg/whl/`` and shipped as the serverless
        dependencies — instead of installing the published ``ygg`` from an index."""
        ensure_console_logging()  # so the deploy CRUD is visible interactively
        logger.info("deploying flow %r", self.name)
        if self.build_wheel and self.serverless:
            from yggdrasil.databricks.job.wheel import ensure_wheel

            # Synthesize a buildable project from this flow's *live* package on
            # disk (no checkout / published release) and build it WITH its deps —
            # all wheels are uploaded under the job folder, so the cluster
            # installs everything by path (no index access).
            self._wheel_paths = ensure_wheel(
                client,
                self.wheel_package(),
                workspace_dir=self.wheel_dir(),
                extras=self.wheel_extras,
                requirements=self.extra_dependencies,
            )
        spec = self.definition()
        logger.info("create-or-update job %r", self.name)
        job = client.jobs.create_or_update(name=spec.pop("name"), **spec)
        logger.info("deployed job %r (id=%s)", self.name, getattr(job, "job_id", None))
        return job


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
    parameters: "tuple[str, ...] | list[str]" = (),
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
            parameters=parameters,
            entry_point=entry_point,
            package_name=package_name,
        )

    return deco(func) if callable(func) else deco


# Legacy names kept so existing imports keep working.
JobSkeleton = Flow
TaskSkeleton = Task
CallableSkeleton = _Runnable
job = flow
