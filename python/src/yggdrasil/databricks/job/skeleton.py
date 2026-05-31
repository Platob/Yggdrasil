"""Skeletons — dataclass-defined, callable job / task definitions.

Three layers:

- :class:`CallableSkeleton` — a dataclass that is **callable** like a function.
  Its dataclass *fields are its parameters*; calling it runs :meth:`run` (or,
  when built from a function, that function with the field values). The field
  values also render into the deployed task's wheel parameters.
- :class:`TaskSkeleton` — a single job task: a :class:`CallableSkeleton` plus
  task metadata (key / depends_on / databricks ``Task`` options) and
  :meth:`~TaskSkeleton.to_task`.
- :class:`JobSkeleton` — a whole job: a name, a trigger, and either a body
  (:meth:`run`) for a single-task job or a tuple of :class:`TaskSkeleton`
  ``steps``. Renders the :meth:`Jobs.create_or_update` kwargs via
  :meth:`~JobSkeleton.definition` and get-or-creates a live :class:`Job` via
  :meth:`~JobSkeleton.deploy`.

The :func:`task` / :func:`job` decorators turn a plain function into a
``TaskSkeleton`` / ``JobSkeleton`` subclass, **grabbing the fields from the
function's signature**::

    @task(depends_on=["extract"])
    def load(table: str, mode: str = "append"):
        ...

    load(table="c.s.t").parameters()   # ["c.s.t", "append"]  (from the fields)
    load(table="c.s.t")()              # runs the function with those fields
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, make_dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.job.job import Job
    from yggdrasil.databricks.job.service import Jobs

__all__ = ["CallableSkeleton", "TaskSkeleton", "JobSkeleton", "task", "job"]


def _render(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


@dataclass
class CallableSkeleton:
    """A callable whose dataclass fields are its parameters."""

    #: Set when the skeleton is built from a function (:func:`task` / :func:`job`)
    #: — the body :meth:`run` invokes with the field values.
    _func: ClassVar[Optional[Callable]] = None

    #: Run the deployed task on **serverless** compute (default). When ``True``
    #: each task references :attr:`environment_key` and the job carries a
    #: matching serverless environment (see :class:`JobSkeleton`).
    serverless: ClassVar[bool] = True

    #: Serverless environment key shared by the job's tasks + its environment.
    environment_key: ClassVar[str] = "default"

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """The body. Override it, or build the skeleton from a function."""
        func = type(self)._func
        if func is None:
            raise NotImplementedError(f"{type(self).__name__} has no run() body")
        merged = {f.name: getattr(self, f.name) for f in fields(self)}
        merged.update(kwargs)
        return func(*args, **merged)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the skeleton locally, like a plain function."""
        return self.run(*args, **kwargs)

    def parameters(self) -> list[str]:
        """Positional parameters for the deployed task — the field values."""
        return [_render(getattr(self, f.name)) for f in fields(self)]

    @classmethod
    def from_function(cls, func: Callable, **namespace: Any) -> type:
        """Build a *cls* subclass (a dataclass) whose fields mirror *func*'s
        signature, running *func* with those fields on call."""
        flds: list = []
        for p in inspect.signature(func).parameters.values():
            if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            ann = p.annotation if p.annotation is not inspect.Parameter.empty else Any
            if p.default is inspect.Parameter.empty:
                flds.append((p.name, ann))
            else:
                flds.append((p.name, ann, field(default=p.default)))
        new_cls = make_dataclass(
            func.__name__,
            flds,
            bases=(cls,),
            namespace={"_func": staticmethod(func), **namespace},
        )
        new_cls.__doc__ = func.__doc__
        new_cls.__module__ = getattr(func, "__module__", new_cls.__module__)
        return new_cls


@dataclass
class TaskSkeleton(CallableSkeleton):
    """A single job task — a :class:`CallableSkeleton` + task metadata."""

    package_name: ClassVar[str] = "yggdrasil"
    entry_point: ClassVar[str] = "ygg-job"
    task_key: ClassVar[str] = "run"
    depends_on: ClassVar[tuple[str, ...]] = ()
    task_options: ClassVar[dict] = {}

    def to_task(self) -> Any:
        """Render the databricks ``Task`` (python-wheel) for this task."""
        from databricks.sdk.service.jobs import (
            PythonWheelTask,
            Task,
            TaskDependency,
        )

        cls = type(self)
        return Task(
            task_key=cls.task_key,
            depends_on=([TaskDependency(task_key=d) for d in cls.depends_on] or None),
            environment_key=(cls.environment_key if cls.serverless else None),
            python_wheel_task=PythonWheelTask(
                package_name=cls.package_name,
                entry_point=cls.entry_point,
                parameters=self.parameters(),
            ),
            **cls.task_options,
        )


@dataclass
class JobSkeleton(CallableSkeleton, ABC):
    """A whole job — name + trigger + body (:meth:`run`) or ``steps``."""

    package_name: ClassVar[str] = "yggdrasil"
    entry_point: ClassVar[str] = "ygg-job"
    task_key: ClassVar[str] = "run"

    #: Ordered :class:`TaskSkeleton` subclasses composing this job. Empty → a
    #: single default task that runs :meth:`run`.
    steps: ClassVar[tuple[type, ...]] = ()

    #: Trigger for function-built jobs (the :func:`job` decorator stores it
    #: here); class-based jobs override :meth:`trigger` instead.
    trigger_settings: ClassVar[Any] = None

    #: Serverless environment for the job's tasks (default: serverless **v5**
    #: with the ``ygg[databricks]`` dependency installed). Set
    #: :attr:`~CallableSkeleton.serverless` to ``False`` to drop it (e.g. to
    #: attach your own cluster via per-task ``options``).
    environment_version: ClassVar[str] = "5"
    dependencies: ClassVar[tuple[str, ...]] = ("ygg[databricks]",)

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable job display name — the upsert key for create-or-update."""

    def trigger(self) -> Any:
        """A databricks ``TriggerSettings`` (file-arrival / schedule / …), or
        ``None`` for a manually-run job."""
        return type(self).trigger_settings

    # -- steps -> task skeletons ----------------------------------------
    def _step_instances(self) -> list["TaskSkeleton"]:
        """Instantiate each ``steps`` :class:`TaskSkeleton`, binding the job's
        fields to the step's matching fields by name."""
        out: list[TaskSkeleton] = []
        for step_cls in type(self).steps:
            kwargs = {
                f.name: getattr(self, f.name)
                for f in fields(step_cls)
                if hasattr(self, f.name)
            }
            out.append(step_cls(**kwargs))
        return out

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the job locally. A single-task job (no ``steps``) runs
        :meth:`run`; composed jobs run each step in order — one step receives
        the call args, several return a ``{task_key: result}`` dict."""
        steps = self._step_instances()
        if not steps:
            return self.run(*args, **kwargs)
        if len(steps) == 1:
            return steps[0](*args, **kwargs)
        return {type(s).task_key: s() for s in steps}

    # -- databricks tasks + rendering -----------------------------------
    def tasks(self) -> list[Any]:
        """One ``Task`` per ``step``, or a single default python-wheel task
        running :meth:`run` when there are no steps."""
        from databricks.sdk.service.jobs import PythonWheelTask, Task

        steps = self._step_instances()
        if not steps:
            return [
                Task(
                    task_key=self.task_key,
                    environment_key=(self.environment_key if self.serverless else None),
                    python_wheel_task=PythonWheelTask(
                        package_name=self.package_name,
                        entry_point=self.entry_point,
                        parameters=self.parameters(),
                    ),
                )
            ]
        return [s.to_task() for s in steps]

    def environments(self) -> Optional[list]:
        """The serverless environment list, or ``None`` when not serverless.

        Defaults to one serverless **v5** environment (keyed
        :attr:`environment_key`) with :attr:`dependencies` (``ygg[databricks]``)
        installed, so the wheel task has ygg on the cluster."""
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

    def definition(self) -> dict:
        """Render the :meth:`Jobs.create_or_update` kwargs for this job."""
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
# Decorators — build a skeleton from a function's signature
# ---------------------------------------------------------------------------


def task(
    func: Optional[Callable] = None,
    *,
    key: Optional[str] = None,
    depends_on: "tuple[str, ...] | list[str]" = (),
    entry_point: Optional[str] = None,
    package_name: Optional[str] = None,
    **options: Any,
) -> Any:
    """Turn a function into a :class:`TaskSkeleton` subclass.

    The fields are grabbed from the function signature; ``key`` defaults to the
    function name, ``depends_on`` lists upstream task keys, and extra
    ``**options`` ride onto the databricks ``Task``.
    """

    def deco(f: Callable) -> type:
        ns: dict[str, Any] = {
            "task_key": key or f.__name__,
            "depends_on": tuple(depends_on),
            "task_options": options,
        }
        if entry_point is not None:
            ns["entry_point"] = entry_point
        if package_name is not None:
            ns["package_name"] = package_name
        return TaskSkeleton.from_function(f, **ns)

    return deco(func) if callable(func) else deco


def job(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    trigger: Any = None,
    steps: "tuple[type, ...] | list[type]" = (),
    entry_point: Optional[str] = None,
    package_name: Optional[str] = None,
) -> Any:
    """Turn a function into a :class:`JobSkeleton` subclass.

    The fields are grabbed from the function signature; ``name`` defaults to the
    function name, ``trigger`` / ``steps`` configure the job.
    """

    def deco(f: Callable) -> type:
        display = name or f.__name__
        ns: dict[str, Any] = {
            "name": property(lambda self, _n=display: _n),
            "steps": tuple(steps),
            "trigger_settings": trigger,
        }
        if entry_point is not None:
            ns["entry_point"] = entry_point
        if package_name is not None:
            ns["package_name"] = package_name
        return JobSkeleton.from_function(f, **ns)

    return deco(func) if callable(func) else deco
