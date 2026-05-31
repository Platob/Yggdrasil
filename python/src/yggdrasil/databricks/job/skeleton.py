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
import time
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.concurrent.threading import ThreadJob
    from yggdrasil.databricks.job.job import Job
    from yggdrasil.databricks.job.service import Jobs

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
    package_name: str = "yggdrasil"
    entry_point: str = "ygg-job"
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

    package_name: str = "yggdrasil"
    entry_point: str = "ygg-job"
    task_key: str = "run"

    #: Serverless environment (default: **v5** + ``ygg[databricks]``). Set
    #: :attr:`serverless` ``False`` to deploy without one.
    serverless: bool = True
    environment_key: str = "default"
    environment_version: str = "5"
    dependencies: "tuple[str, ...]" = ("ygg[databricks]",)

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

    def environments(self) -> Optional[list]:
        """Serverless environment list (v5 + :attr:`dependencies`), or ``None``."""
        if not self.serverless:
            return None
        from databricks.sdk.service.compute import Environment
        from databricks.sdk.service.jobs import JobEnvironment

        return [
            JobEnvironment(
                environment_key=self.environment_key,
                spec=Environment(
                    environment_version=self.environment_version,
                    dependencies=list(self.dependencies),
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

    def deploy(self, service: "Jobs") -> "Job":
        """Get-or-create the live :class:`Job` from :meth:`definition`."""
        spec = self.definition()
        return service.create_or_update(name=spec.pop("name"), **spec)


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
