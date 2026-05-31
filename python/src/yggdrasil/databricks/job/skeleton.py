"""``JobSkeleton`` — define a Python-backed job and render its definition.

A :class:`~yggdrasil.databricks.job.Job` is the *runtime handle* to a job that
already exists in a workspace. A :class:`JobSkeleton` is the other half: a small
declarative class that describes a Python job in code — its name, trigger, and
the Python work to run — and renders that into a **job definition** (the kwargs
:meth:`Jobs.create_or_update` consumes), then deploys it into a live
:class:`Job`.

Declare the work with the :meth:`task` decorator; the skeleton is **callable**
like a plain function — calling it runs the tasks locally (in dependency
order)::

    class Etl(JobSkeleton):
        @property
        def name(self): return "ygg-etl"

        @JobSkeleton.task
        def extract(self): ...

        @JobSkeleton.task(depends_on=["extract"])
        def load(self): ...

    Etl()()                       # run locally: extract → load
    job = Etl().deploy(client.jobs)   # get-or-create the live Job (two tasks)

A skeleton with no ``@task`` methods falls back to a single Python-wheel task
invoking :attr:`entry_point`, and calling it runs :meth:`run`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.job.job import Job
    from yggdrasil.databricks.job.service import Jobs

__all__ = ["JobSkeleton"]

#: Attribute the :meth:`JobSkeleton.task` decorator stamps onto a method.
_TASK_ATTR = "__job_task__"


@dataclass(frozen=True)
class _TaskSpec:
    method: str
    key: str
    depends_on: tuple[str, ...]
    options: dict


class JobSkeleton(ABC):
    """Declarative, callable definition of a Python-backed Databricks job."""

    #: Wheel package + console entry point the deployed task invokes (the
    #: installed ``ygg`` script that re-enters this skeleton on the cluster).
    package_name: ClassVar[str] = "yggdrasil"
    entry_point: ClassVar[str] = "ygg-job"

    #: Task key for the default single-task job (no ``@task`` methods).
    task_key: ClassVar[str] = "run"

    # -- task decorator -------------------------------------------------
    @staticmethod
    def task(
        func: Optional[Callable] = None,
        *,
        key: Optional[str] = None,
        depends_on: "tuple[str, ...] | list[str]" = (),
        **options: Any,
    ) -> Any:
        """Mark a method as a job task.

        Use bare (``@JobSkeleton.task``) or parameterised
        (``@JobSkeleton.task(key="load", depends_on=["extract"], ...)``).
        ``key`` defaults to the method name; ``depends_on`` lists upstream task
        keys; extra ``**options`` ride onto the databricks ``Task`` (timeout,
        retries, compute, …)."""

        def deco(f: Callable) -> Callable:
            setattr(f, _TASK_ATTR, (key, tuple(depends_on), options))
            return f

        return deco(func) if callable(func) else deco

    # -- declarative surface (override in subclasses) -------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Stable job display name — the upsert key for create-or-update."""

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Body for a skeleton with no ``@task`` methods. Override it, or
        declare ``@task`` methods and let :meth:`__call__` drive them."""
        raise NotImplementedError(
            f"{type(self).__name__} declares no @task methods and no run() override"
        )

    def parameters(self) -> list[str]:
        """Positional arguments passed to :attr:`entry_point` (and the task)."""
        return []

    def trigger(self) -> Any:
        """A databricks ``TriggerSettings`` (file-arrival / schedule / …), or
        ``None`` for a manually-run job. Default ``None``."""
        return None

    # -- task discovery + local execution -------------------------------
    @classmethod
    def _task_specs(cls) -> list[_TaskSpec]:
        """All ``@task``-decorated methods across the MRO (base-first, deduped
        by method name so an override replaces its parent)."""
        specs: dict[str, _TaskSpec] = {}
        for klass in reversed(cls.__mro__):
            for attr, value in vars(klass).items():
                meta = getattr(value, _TASK_ATTR, None)
                if meta is None:
                    continue
                key, depends_on, options = meta
                specs[attr] = _TaskSpec(attr, key or attr, depends_on, options)
        return list(specs.values())

    @staticmethod
    def _ordered(specs: list[_TaskSpec]) -> list[_TaskSpec]:
        """Topologically order *specs* by ``depends_on`` (stable)."""
        by_key = {s.key: s for s in specs}
        seen: set[str] = set()
        order: list[_TaskSpec] = []

        def visit(s: _TaskSpec) -> None:
            if s.key in seen:
                return
            seen.add(s.key)
            for dep in s.depends_on:
                upstream = by_key.get(dep)
                if upstream is not None:
                    visit(upstream)
            order.append(s)

        for s in specs:
            visit(s)
        return order

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the job locally, like a plain function.

        With ``@task`` methods, invokes them in dependency order — a single
        task receives ``*args/**kwargs``, several return a ``{task_key: result}``
        dict. With no ``@task`` methods, calls :meth:`run`.
        """
        specs = self._task_specs()
        if not specs:
            return self.run(*args, **kwargs)
        ordered = self._ordered(specs)
        if len(ordered) == 1:
            return getattr(self, ordered[0].method)(*args, **kwargs)
        return {s.key: getattr(self, s.method)() for s in ordered}

    # -- databricks tasks + rendering -----------------------------------
    def tasks(self) -> list[Any]:
        """Build the databricks ``Task`` list.

        One Python-wheel task per ``@task`` method (invoking :attr:`entry_point`
        with the task key + :meth:`parameters` when there's more than one task,
        else just :meth:`parameters`), or a single default task when none are
        declared."""
        from databricks.sdk.service.jobs import (
            PythonWheelTask,
            Task,
            TaskDependency,
        )

        specs = self._ordered(self._task_specs())
        if not specs:
            return [
                Task(
                    task_key=self.task_key,
                    python_wheel_task=PythonWheelTask(
                        package_name=self.package_name,
                        entry_point=self.entry_point,
                        parameters=self.parameters(),
                    ),
                )
            ]
        multi = len(specs) > 1
        out: list[Any] = []
        for s in specs:
            out.append(
                Task(
                    task_key=s.key,
                    depends_on=(
                        [TaskDependency(task_key=d) for d in s.depends_on] or None
                    ),
                    python_wheel_task=PythonWheelTask(
                        package_name=self.package_name,
                        entry_point=self.entry_point,
                        parameters=([s.key, *self.parameters()] if multi else self.parameters()),
                    ),
                    **s.options,
                )
            )
        return out

    def definition(self) -> dict:
        """Render the :meth:`Jobs.create_or_update` kwargs for this skeleton."""
        spec: dict[str, Any] = {"name": self.name, "tasks": self.tasks()}
        trigger = self.trigger()
        if trigger is not None:
            spec["trigger"] = trigger
        return spec

    def deploy(self, service: "Jobs") -> "Job":
        """Get-or-create the live :class:`Job` from :meth:`definition`."""
        spec = self.definition()
        return service.create_or_update(name=spec.pop("name"), **spec)
