"""``JobSkeleton`` — define a Python-backed job and render its definition.

A :class:`~yggdrasil.databricks.job.Job` is the *runtime handle* to a job that
already exists in a workspace. A :class:`JobSkeleton` is the other half: a small
declarative class that describes a Python job in code — its name, trigger, and
the Python entry point to run — and renders that into a **job definition** (the
kwargs :meth:`Jobs.create_or_update` consumes), then deploys it into a live
:class:`Job`.

    class NightlyVacuum(JobSkeleton):
        entry_point = "ygg-vacuum"
        @property
        def name(self): return "ygg-nightly-vacuum"
        def run(self): ...                       # the Python body the task runs

    job = NightlyVacuum().deploy(client.jobs)    # get-or-create the Job

Subclasses implement :attr:`name` and :meth:`run`, and override
:meth:`trigger` / :meth:`parameters` / :meth:`tasks` as needed. The default
:meth:`tasks` is a single Python-wheel task that invokes :attr:`entry_point`
(an installed ``ygg`` console script) with :meth:`parameters`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.job.job import Job
    from yggdrasil.databricks.job.service import Jobs

__all__ = ["JobSkeleton"]


class JobSkeleton(ABC):
    """Declarative definition of a Python-backed Databricks job."""

    #: Wheel package + console entry point the deployed task invokes. The
    #: entry point is the installed ``ygg`` script that re-enters this
    #: skeleton's :meth:`run` on the job cluster.
    package_name: ClassVar[str] = "yggdrasil"
    entry_point: ClassVar[str] = "ygg-job"

    #: Task key for the default single-task job.
    task_key: ClassVar[str] = "run"

    # -- declarative surface (override in subclasses) -------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Stable job display name — the upsert key for create-or-update."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """The Python body the deployed task executes on the job cluster."""

    def parameters(self) -> list[str]:
        """Positional arguments passed to :attr:`entry_point` (and :meth:`run`)."""
        return []

    def trigger(self) -> Any:
        """A databricks ``TriggerSettings`` (file-arrival / schedule / …), or
        ``None`` for a manually-run job. Default ``None``."""
        return None

    def tasks(self) -> list[Any]:
        """Job tasks. Default: one Python-wheel task invoking :attr:`entry_point`
        with :meth:`parameters`."""
        from databricks.sdk.service.jobs import PythonWheelTask, Task

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

    # -- rendering + deployment -----------------------------------------
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
