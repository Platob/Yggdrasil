"""Live-integration tests for the Jobs/JobTask + ``@job.task(...).decorate`` flow.

Skipped unless ``DATABRICKS_HOST`` is exported (see
:class:`DatabricksIntegrationCase`).

Exercises against a real workspace:

- :meth:`Jobs.get_or_create` — first call creates, second call returns
  the same handle without pushing settings back.
- :meth:`JobTask.from_callable` — stages the function's raw source as
  a ``.py`` script under ``/Workspace/Users/me/.yggdrasil/jobs/``;
  verifies the file actually lands and the script content compiles +
  calls the function.
- :meth:`JobTask.create_or_update` — re-staging the same ``task_key``
  replaces the existing entry on the job rather than raising.
- :meth:`Job.task` + :meth:`JobTask.decorate` — Prefect-style sugar
  wires through :meth:`JobTask.from_callable` +
  :meth:`JobTask.create_or_update`.
- :meth:`JobTask.update` / :meth:`JobTask.delete` — round-trip
  through :meth:`Job.update` and the parent job's settings reflect
  the change.
"""
from __future__ import annotations

import os
import secrets
from typing import ClassVar, List

from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.jobs import (
    ConditionTask,
    ConditionTaskOp,
    Task,
)

from yggdrasil.databricks.jobs import Job, JobTask
from yggdrasil.databricks.jobs.task import DEFAULT_STAGING_ROOT

from .. import DatabricksIntegrationCase


__all__ = ["TestJobsGetOrCreate", "TestJobTaskIntegration"]


def _noop_condition_task(task_key: str = "noop") -> Task:
    """Return a trivial ``1 == 1`` condition task — no compute required."""
    return Task(
        task_key=task_key,
        condition_task=ConditionTask(
            op=ConditionTaskOp.EQUAL_TO,
            left="1",
            right="1",
        ),
    )


class _JobsIntegrationBase(DatabricksIntegrationCase):
    """Shared fixture: tracks created jobs for ``tearDownClass`` cleanup."""

    created_jobs: ClassVar[List[int]]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.created_jobs = []

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            for job_id in cls.created_jobs:
                try:
                    cls.client.jobs.find(job_id=job_id).delete()
                except DatabricksError:
                    pass
                except Exception:
                    pass
        finally:
            super().tearDownClass()

    @staticmethod
    def _unique_job_name(prefix: str) -> str:
        return f"yg_int_{prefix}_{secrets.token_hex(4)}"

    @classmethod
    def _track(cls, job: Job) -> Job:
        if job.job_id is not None and job.job_id not in cls.created_jobs:
            cls.created_jobs.append(job.job_id)
        return job


class TestJobsGetOrCreate(_JobsIntegrationBase):
    """``client.jobs.get_or_create`` round-trip against a live workspace."""

    def test_get_or_create_creates_then_returns_same_handle(self):
        name = self._unique_job_name("getorcreate")

        first = self.client.jobs.get_or_create(
            name=name, tasks=[_noop_condition_task()],
        )
        self._track(first)
        assert isinstance(first, Job)
        assert first.job_id is not None

        # Second call with the same name returns the existing job —
        # no new id, no settings push.
        second = self.client.jobs.get_or_create(name=name)
        assert second.job_id == first.job_id

        # And resolves by id too.
        by_id = self.client.jobs.get_or_create(job_id=first.job_id)
        assert by_id.job_id == first.job_id

    def test_get_or_create_without_name_or_id_raises(self):
        with self.assertRaises(ValueError):
            self.client.jobs.get_or_create()


class TestJobTaskIntegration(_JobsIntegrationBase):
    """``JobTask`` CRUD + ``@job.task(...).decorate`` against a real workspace."""

    def _fresh_job(self, prefix: str) -> Job:
        """Make-or-reuse a job seeded with a no-op task so ``update``
        always has something to merge into."""
        name = self._unique_job_name(prefix)
        job = self.client.jobs.create(
            name=name, tasks=[_noop_condition_task("seed")],
        )
        return self._track(job)

    @staticmethod
    def _staged_workspace_path(task: Task) -> str:
        """Pull the ``python_file`` out of a ``spark_python_task``."""
        body = task.spark_python_task
        assert body is not None, "expected spark_python_task on staged task"
        return body.python_file

    def _workspace_read(self, path: str) -> bytes:
        """Read a workspace file via the SDK download API."""
        ws = self.client.workspace_client().workspace
        with ws.download(path) as stream:
            return stream.read()

    # ---- from_callable / staging ------------------------------------ #

    def test_from_callable_stages_raw_python_source(self):
        job = self._fresh_job("from_callable")

        def do_something(x: int = 7) -> int:
            """One-line doc that becomes the task description."""
            print("hello from do_something:", x)
            return x

        jt = JobTask.from_callable(job, do_something, 42)
        # Not yet persisted on the job.
        assert jt.task_key == "do_something"
        assert jt._details is not None
        assert jt._details.description.startswith("One-line doc")

        path = self._staged_workspace_path(jt._details)
        assert path.startswith(DEFAULT_STAGING_ROOT)
        # The staged file is a real workspace object.
        content = self._workspace_read(path).decode()
        assert "def do_something" in content
        assert "do_something(42)" in content
        # Sanity-compile.
        compile(content, path, "exec")

    # ---- decorator + create_or_update ------------------------------- #

    def test_job_task_decorator_registers_then_re_decorates_in_place(self):
        job = self._fresh_job("decorator")

        @job.task("step_one").decorate
        def step_one(a: str = "hi"):
            """First decorator pass."""
            print(a)

        # Decorator returns the original callable; the JobTask handle
        # rides on ``func._job_task``.
        registered: JobTask = step_one._job_task  # type: ignore[attr-defined]
        assert isinstance(registered, JobTask)
        assert registered.task_key == "step_one"

        job.refresh()
        task_keys = {t.task_key for t in (job.settings.tasks or [])}
        assert "step_one" in task_keys

        # Re-decorating the same task_key with a different body
        # replaces the entry in place — no duplicate task_key.
        @job.task("step_one").decorate
        def step_one(a: str = "hi"):  # noqa: F811 — intentional shadow
            """Second decorator pass — different body, same key."""
            print("rewritten", a)

        job.refresh()
        matching = [
            t for t in (job.settings.tasks or []) if t.task_key == "step_one"
        ]
        assert len(matching) == 1, (
            "create_or_update should replace, not duplicate"
        )
        new_path = self._staged_workspace_path(matching[0])
        new_content = self._workspace_read(new_path).decode()
        assert "rewritten" in new_content

    def test_job_task_decorator_with_task_fields(self):
        """``@job.task(key, description=...).decorate`` layers Task fields on."""
        job = self._fresh_job("decorator_fields")

        @job.task("custom_key", description="overridden desc").decorate
        def make_it():
            """Original docstring — should be overridden by description=."""
            print("ok")

        jt: JobTask = make_it._job_task  # type: ignore[attr-defined]
        assert jt.task_key == "custom_key"
        assert jt._details.description == "overridden desc"

        job.refresh()
        registered = next(
            (t for t in (job.settings.tasks or []) if t.task_key == "custom_key"),
            None,
        )
        assert registered is not None

    # ---- CRUD --------------------------------------------------------- #

    def test_job_task_refresh_update_delete(self):
        job = self._fresh_job("crud")

        @job.task("crud_target").decorate
        def crud_target():
            """First version."""
            print("v1")

        jt: JobTask = crud_target._job_task  # type: ignore[attr-defined]

        # Refresh reloads details from the parent job's settings.
        jt._details = None
        jt.refresh()
        assert jt.details is not None
        assert jt.details.task_key == "crud_target"

        # Update — replace description via dataclass.replace.
        jt.update(description="updated by integration test")
        job.refresh()
        again = next(
            t for t in (job.settings.tasks or []) if t.task_key == "crud_target"
        )
        assert again.description == "updated by integration test"

        # Delete — task vanishes from the job's task list.
        jt.delete()
        job.refresh()
        remaining = {t.task_key for t in (job.settings.tasks or [])}
        assert "crud_target" not in remaining
        # The seed task survives the targeted delete.
        assert "seed" in remaining

    def test_create_or_update_on_handcrafted_task(self):
        """Non-decorator path: build a Task manually + create_or_update."""
        job = self._fresh_job("handcrafted")

        condition = _noop_condition_task("noop_extra")
        jt = JobTask(job=job, task_key="noop_extra", details=condition)
        jt.create_or_update()

        job.refresh()
        keys = {t.task_key for t in (job.settings.tasks or [])}
        assert "noop_extra" in keys

        # Same key, new details — replaces in place.
        replacement = _noop_condition_task("noop_extra")
        replacement.description = "replaced via create_or_update"
        jt2 = JobTask(job=job, task_key="noop_extra", details=replacement)
        jt2.create_or_update()

        job.refresh()
        matching = [
            t for t in (job.settings.tasks or []) if t.task_key == "noop_extra"
        ]
        assert len(matching) == 1
        assert matching[0].description == "replaced via create_or_update"
