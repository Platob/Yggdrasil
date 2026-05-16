"""Unit tests for the Databricks Jobs service and Job / JobRun resources.

Uses :class:`yggdrasil.databricks.tests.DatabricksTestCase` to mock the SDK
boundary while exercising the real :class:`Jobs` / :class:`Job` / :class:`JobRun`
wiring (default injection, caching, singleton identity, lifecycle).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.jobs import (
    BaseJob,
    Job as JobInfo,
    JobPermissionLevel,
    JobSettings,
    Run,
    RunLifeCycleState,
    RunResultState,
    RunState,
)

from databricks.sdk.service.jobs import SparkPythonTask, Task

from yggdrasil.databricks.jobs import Job, JobRun
from yggdrasil.databricks.jobs.task import (
    JobTask,
    _content_digest,
    _render_callable_script,
)
from yggdrasil.databricks.tests import DatabricksTestCase


def _signature_fixture(name: str = "alice", count: int = 3) -> str:
    """Greet someone N times."""
    return f"hi {name}" * count


# --- Fixtures referenced by TestStagedScriptCapturesLocals ----------- #

_CAPTURE_CONSTANT = "value-from-module"
_CAPTURE_MAP = {"k": 1, "v": [1, 2, 3]}


def _capture_helper(prefix: str) -> str:
    """Helper that the entry callable below references at module scope."""
    return f"{prefix}::ok"


def _capture_entry(suffix: str) -> str:
    """Entry callable — uses both ``_capture_helper`` and ``_CAPTURE_CONSTANT``."""
    return _capture_helper(_CAPTURE_CONSTANT) + suffix + str(_CAPTURE_MAP["k"])


def _job_info(
    *,
    job_id: int = 1,
    name: str = "test-job",
    tasks: Any = None,
    **overrides: Any,
) -> JobInfo:
    settings = JobSettings(name=name, tasks=list(tasks) if tasks else None)
    return JobInfo(job_id=job_id, settings=settings, **overrides)


def _base_job(
    *,
    job_id: int = 1,
    name: str = "test-job",
    **overrides: Any,
) -> BaseJob:
    settings = JobSettings(name=name)
    return BaseJob(job_id=job_id, settings=settings, **overrides)


def _run(
    *,
    run_id: int = 100,
    job_id: int = 1,
    life_cycle_state: RunLifeCycleState = RunLifeCycleState.TERMINATED,
    result_state: RunResultState = RunResultState.SUCCESS,
    state_message: str = "",
    run_page_url: str | None = None,
    **overrides: Any,
) -> Run:
    state = RunState(
        life_cycle_state=life_cycle_state,
        result_state=result_state,
        state_message=state_message,
    )
    return Run(
        run_id=run_id,
        job_id=job_id,
        state=state,
        run_page_url=run_page_url,
        **overrides,
    )


class TestJobsService(DatabricksTestCase):
    """Collection-level :class:`Jobs` create / find / list path."""

    def test_create_passes_through_to_sdk(self):
        self.jobs_api.create.return_value = MagicMock(job_id=42)
        self.jobs_api.get.return_value = _job_info(job_id=42, name="etl")

        job = self.jobs.create(name="etl", tasks=[])

        self.assertIsInstance(job, Job)
        self.assertEqual(job.job_id, 42)
        self.assertEqual(job.job_name, "etl")
        self.jobs_api.create.assert_called_once()
        _, kwargs = self.jobs_api.create.call_args
        self.assertEqual(kwargs["name"], "etl")
        # Default tags from DatabricksService.default_tags() are merged in
        self.assertIn("ServiceName", kwargs.get("tags", {}))

    def test_create_with_permissions(self):
        self.jobs_api.create.return_value = MagicMock(job_id=42)
        self.jobs_api.get.return_value = _job_info(job_id=42, name="etl")

        self.jobs.create(
            name="etl",
            tasks=[],
            permissions=["alice@example.com", "data-eng"],
        )

        _, kwargs = self.jobs_api.create.call_args
        acl = kwargs.get("access_control_list")
        self.assertEqual(len(acl), 2)
        self.assertEqual(acl[0].user_name, "alice@example.com")
        self.assertEqual(acl[0].permission_level, JobPermissionLevel.CAN_MANAGE)
        self.assertEqual(acl[1].group_name, "data-eng")

    def test_find_by_id_returns_job(self):
        self.jobs_api.get.return_value = _job_info(job_id=7, name="reports")

        job = self.jobs.find(job_id=7)

        self.assertIsNotNone(job)
        self.assertEqual(job.job_id, 7)
        self.assertEqual(job.job_name, "reports")
        self.jobs_api.get.assert_called_once_with(job_id=7)

    def test_find_missing_returns_none(self):
        self.jobs_api.get.side_effect = ResourceDoesNotExist("not found")

        self.assertIsNone(self.jobs.find(job_id=9999))

    def test_find_missing_raises_when_requested(self):
        self.jobs_api.get.side_effect = ResourceDoesNotExist("not found")

        with self.assertRaises(ValueError):
            self.jobs.find(job_id=9999, raise_error=True)

    def test_find_requires_id_or_name(self):
        with self.assertRaises(ValueError):
            self.jobs.find()

    def test_find_by_name_uses_server_filter(self):
        self.jobs_api.list.return_value = iter([
            _base_job(job_id=5, name="hourly"),
        ])

        job = self.jobs.find(name="hourly")

        self.assertIsNotNone(job)
        self.assertEqual(job.job_id, 5)
        # Server-side name filter passed through to the SDK
        _, kwargs = self.jobs_api.list.call_args
        self.assertEqual(kwargs.get("name"), "hourly")

    def test_list_yields_jobs(self):
        self.jobs_api.list.return_value = iter([
            _base_job(job_id=1, name="a"),
            _base_job(job_id=2, name="b"),
        ])

        jobs = list(self.jobs.list())

        self.assertEqual(len(jobs), 2)
        self.assertEqual([j.job_id for j in jobs], [1, 2])

    def test_create_or_update_creates_when_missing(self):
        # find returns None
        self.jobs_api.get.side_effect = ResourceDoesNotExist("nope")
        self.jobs_api.create.return_value = MagicMock(job_id=99)
        # After create, refresh() does another get — give it a real JobInfo
        self.jobs_api.get.side_effect = [ResourceDoesNotExist("nope"), _job_info(job_id=99, name="new")]

        job = self.jobs.create_or_update(job_id=12345, name="new", tasks=[])

        self.assertEqual(job.job_id, 99)
        self.jobs_api.create.assert_called_once()


class TestJobSingleton(DatabricksTestCase):
    """:class:`Job` is a hash-keyed singleton per ``(service, job_id, job_name)``."""

    def test_same_id_returns_same_instance(self):
        a = Job(service=self.jobs, job_id=10, job_name="x")
        b = Job(service=self.jobs, job_id=10, job_name="x")
        self.assertIs(a, b)

    def test_different_id_returns_different_instance(self):
        a = Job(service=self.jobs, job_id=10)
        b = Job(service=self.jobs, job_id=11)
        self.assertIsNot(a, b)

    def test_init_is_idempotent(self):
        # Mutate after construction and verify the second constructor call
        # does NOT clobber the live state.
        a = Job(service=self.jobs, job_id=10, job_name="x")
        a._details = _job_info(job_id=10, name="x")
        b = Job(service=self.jobs, job_id=10, job_name="x")
        self.assertIs(a._details, b._details)


class TestJobResourceLifecycle(DatabricksTestCase):
    """Single-job lifecycle: refresh, update, reset, delete."""

    def _job(self, **overrides: Any) -> Job:
        details = _job_info(**overrides)
        return Job(
            service=self.jobs,
            job_id=details.job_id,
            job_name=overrides.get("name", "test-job"),
            details=details,
        )

    def test_refresh_fetches_fresh_details(self):
        job = self._job(job_id=3, name="reports")
        self.jobs_api.get.return_value = _job_info(job_id=3, name="renamed")

        job.refresh()

        self.assertEqual(job.job_name, "renamed")
        self.jobs_api.get.assert_called_with(job_id=3)

    def test_update_calls_sdk_with_new_settings(self):
        job = self._job(job_id=4)
        self.jobs_api.get.return_value = _job_info(job_id=4, name="post-update")

        job.update(name="post-update")

        self.jobs_api.update.assert_called_once()
        _, kwargs = self.jobs_api.update.call_args
        self.assertEqual(kwargs["job_id"], 4)
        self.assertEqual(kwargs["new_settings"].name, "post-update")

    def test_update_with_no_changes_is_noop(self):
        job = self._job(job_id=4)
        # No settings, no fields_to_remove, no permissions
        job.update()
        self.jobs_api.update.assert_not_called()

    def test_reset_overrides_settings(self):
        job = self._job(job_id=5)
        self.jobs_api.get.return_value = _job_info(job_id=5, name="reset-me")
        new_settings = JobSettings(name="reset-me")

        job.reset(new_settings)

        self.jobs_api.reset.assert_called_once_with(
            job_id=5, new_settings=new_settings,
        )

    def test_delete_calls_sdk_and_drops_name_cache(self):
        from yggdrasil.databricks.jobs import service as _js

        job = self._job(job_id=6, name="to-drop")
        # Seed the host name cache
        _js._set_cached_job_id(self.client, "to-drop", 6)

        job.delete()

        self.jobs_api.delete.assert_called_once_with(job_id=6)
        host_cache = _js._NAME_ID_CACHE.get(self.client.base_url.to_string())
        self.assertNotIn("to-drop", host_cache or {})

    def test_delete_swallows_resource_does_not_exist(self):
        job = self._job(job_id=7)
        self.jobs_api.delete.side_effect = ResourceDoesNotExist("gone")

        # Must not raise.
        job.delete()

    def test_update_permissions(self):
        job = self._job(job_id=8)

        job.update_permissions(["alice@example.com"])

        self.jobs_api.update_permissions.assert_called_once()
        _, kwargs = self.jobs_api.update_permissions.call_args
        self.assertEqual(kwargs["job_id"], "8")
        self.assertEqual(kwargs["access_control_list"][0].user_name, "alice@example.com")


class TestJobRun(DatabricksTestCase):
    """Triggering a run and inspecting state."""

    def _job(self) -> Job:
        details = _job_info(job_id=1, name="hourly")
        return Job(service=self.jobs, job_id=1, job_name="hourly", details=details)

    def test_run_calls_run_now_and_returns_job_run(self):
        job = self._job()
        waiter = MagicMock()
        waiter.run_id = 555
        self.jobs_api.run_now.return_value = waiter

        run = job.run(job_parameters={"date": "2026-05-15"})

        self.assertIsInstance(run, JobRun)
        self.assertEqual(run.run_id, 555)
        self.jobs_api.run_now.assert_called_once()
        _, kwargs = self.jobs_api.run_now.call_args
        self.assertEqual(kwargs["job_id"], 1)
        self.assertEqual(kwargs["job_parameters"], {"date": "2026-05-15"})

    def test_run_drops_unset_params(self):
        job = self._job()
        waiter = MagicMock(run_id=1)
        self.jobs_api.run_now.return_value = waiter

        job.run()  # everything unset

        _, kwargs = self.jobs_api.run_now.call_args
        # Only job_id should be present
        self.assertEqual(set(kwargs.keys()), {"job_id"})

    def test_cancel_all_runs(self):
        job = self._job()
        job.cancel_all_runs()
        self.jobs_api.cancel_all_runs.assert_called_once_with(
            job_id=1, all_queued_runs=None,
        )


class TestJobRunResource(DatabricksTestCase):
    """Lifecycle and state inspection on :class:`JobRun`."""

    def test_singleton_per_run_id(self):
        a = JobRun(service=self.jobs, run_id=42)
        b = JobRun(service=self.jobs, run_id=42)
        self.assertIs(a, b)

    def test_state_predicates_terminal_success(self):
        run = JobRun(
            service=self.jobs,
            run_id=1,
            details=_run(run_id=1, life_cycle_state=RunLifeCycleState.TERMINATED,
                         result_state=RunResultState.SUCCESS),
        )
        self.assertTrue(run.is_terminal)
        self.assertTrue(run.is_successful)
        self.assertFalse(run.is_pending)

    def test_state_predicates_running(self):
        run = JobRun(
            service=self.jobs,
            run_id=2,
            details=_run(run_id=2, life_cycle_state=RunLifeCycleState.RUNNING,
                         result_state=None),
        )
        self.assertFalse(run.is_terminal)
        self.assertTrue(run.is_pending)

    def test_state_predicates_failed(self):
        run = JobRun(
            service=self.jobs,
            run_id=3,
            details=_run(run_id=3, life_cycle_state=RunLifeCycleState.TERMINATED,
                         result_state=RunResultState.FAILED),
        )
        self.assertTrue(run.is_terminal)
        self.assertFalse(run.is_successful)

    def test_refresh_fetches_fresh_details(self):
        run = JobRun(service=self.jobs, run_id=4)
        self.jobs_api.get_run.return_value = _run(run_id=4)

        run.refresh()

        self.jobs_api.get_run.assert_called_with(run_id=4)
        self.assertTrue(run.is_terminal)

    def test_wait_for_status_polls_until_terminal(self):
        run = JobRun(service=self.jobs, run_id=5)
        # First poll: running. Second poll: terminated.
        self.jobs_api.get_run.side_effect = [
            _run(run_id=5, life_cycle_state=RunLifeCycleState.RUNNING,
                 result_state=None),
            _run(run_id=5, life_cycle_state=RunLifeCycleState.TERMINATED,
                 result_state=RunResultState.SUCCESS),
        ]

        run.wait_for_status(wait={"timeout": 5.0, "interval": 0.0})

        self.assertEqual(self.jobs_api.get_run.call_count, 2)
        self.assertTrue(run.is_successful)

    def test_wait_for_status_raises_when_requested(self):
        run = JobRun(service=self.jobs, run_id=6)
        self.jobs_api.get_run.return_value = _run(
            run_id=6,
            life_cycle_state=RunLifeCycleState.TERMINATED,
            result_state=RunResultState.FAILED,
            state_message="boom",
        )

        with self.assertRaises(RuntimeError) as ctx:
            run.wait_for_status(wait=True, raise_error=True)

        self.assertIn("boom", str(ctx.exception))

    def test_cancel_calls_sdk(self):
        run = JobRun(service=self.jobs, run_id=7)
        waiter = MagicMock()
        waiter.result.return_value = _run(
            run_id=7, life_cycle_state=RunLifeCycleState.TERMINATED,
            result_state=RunResultState.CANCELED,
        )
        self.jobs_api.cancel_run.return_value = waiter
        self.jobs_api.get_run.return_value = _run(
            run_id=7, life_cycle_state=RunLifeCycleState.TERMINATED,
            result_state=RunResultState.CANCELED,
        )

        run.cancel()

        self.jobs_api.cancel_run.assert_called_once_with(run_id=7)

    def test_url_uses_run_page_url_when_available(self):
        run = JobRun(
            service=self.jobs, run_id=8,
            details=_run(run_id=8, run_page_url="https://test.databricks.net/#job/1/run/8"),
        )
        self.assertEqual(
            run.url().to_string(),
            "https://test.databricks.net/#job/1/run/8",
        )


class TestJobTaskFactoryAndDecorate(DatabricksTestCase):
    """:meth:`Job.task` builds a :class:`JobTask`; :meth:`JobTask.decorate` back-fills."""

    def _job(self, *, job_id: int = 42) -> Job:
        return Job(
            service=self.jobs,
            job_id=job_id,
            job_name="t",
            details=_job_info(job_id=job_id, name="t"),
        )

    def test_task_factory_returns_jobtask_with_prebuilt_details(self):
        job = self._job()
        jt = job.task("step", description="hi", timeout_seconds=900)

        self.assertIsInstance(jt, JobTask)
        self.assertIs(jt.job, job)
        self.assertEqual(jt.task_key, "step")
        self.assertEqual(jt._details.task_key, "step")
        self.assertEqual(jt._details.description, "hi")
        self.assertEqual(jt._details.timeout_seconds, 900)

    def test_decorate_back_fills_only_unset_fields(self):
        job = self._job()
        preset_body = SparkPythonTask(python_file="/explicit.py")
        jt = job.task(
            "step",
            description="caller wins",
            spark_python_task=preset_body,
            timeout_seconds=600,
        )

        # Stub from_callable so the test doesn't hit the workspace.
        staged_body = SparkPythonTask(python_file="/from_callable.py")
        staged = JobTask(
            job=job,
            task_key="step",
            details=Task(
                task_key="step",
                description="from docstring",
                spark_python_task=staged_body,
            ),
        )
        with self._patch_jobtask_method("from_callable", return_value=staged), \
                self._patch_jobtask_method("create"):
            def step():
                """from docstring"""

            returned = jt.decorate(step)

        self.assertIs(returned, step)
        self.assertIs(step._job_task, jt)  # type: ignore[attr-defined]
        # Pre-set fields untouched.
        self.assertIs(jt._details.spark_python_task, preset_body)
        self.assertEqual(jt._details.description, "caller wins")
        self.assertEqual(jt._details.timeout_seconds, 600)

    def test_decorate_fills_in_missing_defaults(self):
        job = self._job()
        # No body / description on the handle — decorate should fill both.
        jt = job.task("step", timeout_seconds=600)

        staged_body = SparkPythonTask(python_file="/from_callable.py")
        staged = JobTask(
            job=job,
            task_key="step",
            details=Task(
                task_key="step",
                description="from docstring",
                spark_python_task=staged_body,
            ),
        )
        with self._patch_jobtask_method("from_callable", return_value=staged), \
                self._patch_jobtask_method("create"):
            def step():
                """from docstring"""

            jt.decorate(step)

        self.assertIs(jt._details.spark_python_task, staged_body)
        self.assertEqual(jt._details.description, "from docstring")
        self.assertEqual(jt._details.timeout_seconds, 600)

    def _patch_jobtask_method(self, name: str, *, return_value: Any = None):
        from unittest.mock import patch
        return patch.object(JobTask, name, return_value=return_value)

    def test_pytask_bare_defaults_task_key_to_func_name(self):
        job = self._job()
        staged_body = SparkPythonTask(python_file="/from_callable.py")

        def _fake_from_callable(cls, j, f, **_kw):
            return JobTask(
                job=j,
                task_key=f.__name__,
                details=Task(
                    task_key=f.__name__,
                    description="from docstring",
                    spark_python_task=staged_body,
                ),
            )

        from unittest.mock import patch
        with patch.object(JobTask, "from_callable", classmethod(_fake_from_callable)), \
                patch.object(JobTask, "create"):

            @job.pytask
            def step():
                """from docstring"""

        jt: JobTask = step._job_task  # type: ignore[attr-defined]
        self.assertEqual(jt.task_key, "step")
        self.assertIs(jt._details.spark_python_task, staged_body)

    def test_task_order_inserts_at_position_on_create(self):
        """``order=N`` places the task at slice index N in the job's task list."""
        job = self._job()
        # Seed three existing tasks so positions are observable.
        job.settings.tasks = [
            Task(task_key="a"),
            Task(task_key="b"),
            Task(task_key="c"),
        ]

        jt = job.task("new", order=1)
        jt.create()

        _, kwargs = self.jobs_api.update.call_args
        new_keys = [t.task_key for t in kwargs["new_settings"].tasks]
        self.assertEqual(new_keys, ["a", "new", "b", "c"])

    def test_task_order_moves_existing_task_to_new_position(self):
        """``order`` on an idempotent re-create first strips, then reinserts."""
        job = self._job()
        job.settings.tasks = [
            Task(task_key="a"),
            Task(task_key="b"),
            Task(task_key="c"),
        ]

        # Move "a" to the end via order=-1 + a fresh details payload.
        jt = job.task("a", order=-1, description="moved")
        jt.create()

        _, kwargs = self.jobs_api.update.call_args
        tasks = kwargs["new_settings"].tasks
        keys = [t.task_key for t in tasks]
        self.assertEqual(keys, ["b", "a", "c"])
        moved = next(t for t in tasks if t.task_key == "a")
        self.assertEqual(moved.description, "moved")

    def test_create_attaches_default_environment_when_task_uses_key(self):
        """A task carrying ``environment_key`` adds a matching :class:`JobEnvironment`
        to the parent job when none is declared yet."""
        from yggdrasil.databricks.jobs.task import (
            DEFAULT_ENVIRONMENT_CLIENT,
            DEFAULT_ENVIRONMENT_KEY,
        )

        job = self._job()
        job.settings.tasks = [Task(task_key="seed")]

        jt = job.task(
            "step", environment_key=DEFAULT_ENVIRONMENT_KEY,
        )
        jt.create()

        _, kwargs = self.jobs_api.update.call_args
        envs = kwargs["new_settings"].environments
        self.assertIsNotNone(envs)
        self.assertEqual(len(envs), 1)
        self.assertEqual(envs[0].environment_key, DEFAULT_ENVIRONMENT_KEY)
        self.assertEqual(envs[0].spec.client, DEFAULT_ENVIRONMENT_CLIENT)
        # The default spec pulls in ``ygg`` so staged
        # ``from yggdrasil...`` imports resolve at runtime.
        self.assertIn("ygg", envs[0].spec.dependencies)

    def test_create_skips_environment_merge_when_already_declared(self):
        """``environments`` isn't touched when the key already lives on the job."""
        from databricks.sdk.service.compute import Environment
        from databricks.sdk.service.jobs import JobEnvironment

        job = self._job()
        job.settings.tasks = [Task(task_key="seed")]
        job.settings.environments = [
            JobEnvironment(
                environment_key="custom",
                spec=Environment(client="1", dependencies=["pandas"]),
            ),
        ]

        jt = job.task("step", environment_key="custom")
        jt.create()

        _, kwargs = self.jobs_api.update.call_args
        # No environments update → the field is left off new_settings.
        self.assertIsNone(kwargs["new_settings"].environments)

    def test_task_without_order_keeps_existing_position(self):
        """``order=None`` (default) replaces in place — no shuffle."""
        job = self._job()
        job.settings.tasks = [
            Task(task_key="a"),
            Task(task_key="b"),
            Task(task_key="c"),
        ]

        jt = job.task("b", description="updated")
        jt.create()

        _, kwargs = self.jobs_api.update.call_args
        keys = [t.task_key for t in kwargs["new_settings"].tasks]
        self.assertEqual(keys, ["a", "b", "c"])

    def test_pytask_order_forwards_to_jobtask(self):
        """``@job.pytask(order=…)`` carries through to the JobTask handle."""
        job = self._job()
        staged_body = SparkPythonTask(python_file="/from_callable.py")

        def _fake_from_callable(cls, j, f, **_kw):
            return JobTask(
                job=j,
                task_key=_kw.get("task_key") or f.__name__,
                details=Task(
                    task_key=_kw.get("task_key") or f.__name__,
                    spark_python_task=staged_body,
                ),
            )

        from unittest.mock import patch
        with patch.object(JobTask, "from_callable", classmethod(_fake_from_callable)), \
                patch.object(JobTask, "create"):

            @job.pytask(order=2)
            def step():
                """from docstring"""

        jt: JobTask = step._job_task  # type: ignore[attr-defined]
        self.assertEqual(jt.order, 2)

    def test_delete_uses_fields_to_remove(self):
        """``JobTask.delete`` routes through ``fields_to_remove`` because the
        Jobs API merges ``new_settings.tasks`` by ``task_key`` rather than
        replacing it — a shortened task list would leave the target task
        intact on the server."""
        job = self._job()
        job.settings.tasks = [Task(task_key="a"), Task(task_key="b")]

        jt = job.task("a")
        jt.delete()

        _, kwargs = self.jobs_api.update.call_args
        self.assertEqual(kwargs["fields_to_remove"], ["tasks/a"])
        self.assertIsNone(kwargs["new_settings"])

    def test_delete_noop_when_task_already_absent(self):
        """``delete`` skips the SDK round trip when the key isn't on the job."""
        job = self._job()
        job.settings.tasks = [Task(task_key="a")]

        jt = job.task("missing")
        jt.delete()

        self.jobs_api.update.assert_not_called()

    def test_pytask_parametrized_forwards_fields(self):
        job = self._job()
        staged_body = SparkPythonTask(python_file="/from_callable.py")

        def _fake_from_callable(cls, j, f, **_kw):
            return JobTask(
                job=j,
                task_key=_kw.get("task_key") or f.__name__,
                details=Task(
                    task_key=_kw.get("task_key") or f.__name__,
                    description="from docstring",
                    spark_python_task=staged_body,
                ),
            )

        from unittest.mock import patch
        with patch.object(JobTask, "from_callable", classmethod(_fake_from_callable)), \
                patch.object(JobTask, "create"):

            @job.pytask(task_key="custom", description="caller wins", timeout_seconds=600)
            def step():
                """from docstring"""

        jt: JobTask = step._job_task  # type: ignore[attr-defined]
        self.assertEqual(jt.task_key, "custom")
        self.assertEqual(jt._details.description, "caller wins")
        self.assertEqual(jt._details.timeout_seconds, 600)
        # spark_python_task wasn't pre-set, decorate filled it in.
        self.assertIs(jt._details.spark_python_task, staged_body)


class TestStagedScriptMetadata(DatabricksTestCase):
    """Staged JobTask script: signature metadata + @checkargs wrapping."""

    def test_script_wraps_function_with_checkargs_and_coerces_inputs(self):
        import ast
        import json

        script = _render_callable_script(
            _signature_fixture, (), {"name": "bob", "count": "9"},
        )
        ast.parse(script)

        self.assertIn("__yggdrasil_task__", script)
        self.assertIn(
            "from yggdrasil.dataclasses.safe_function import checkargs",
            script,
        )
        self.assertIn("Signature: _signature_fixture(name: str", script)
        # The staged function is re-wrapped with @checkargs so every
        # call site coerces inputs to annotated types.
        self.assertIn("@checkargs\ndef _signature_fixture(", script)
        # Invocation is now a direct call; the decorator handles coercion.
        self.assertIn("_signature_fixture(name='bob', count='9')", script)

        # Exec end-to-end — @checkargs must coerce "9" → int 9.
        env: dict = {"__name__": "__main__", "_received": []}
        patched = script.replace(
            'return f"hi {name}" * count',
            '_received.append((name, count, type(count).__name__))\n'
            '    return f"hi {name}" * count',
        )
        exec(compile(patched, "<staged>", "exec"), env)
        self.assertEqual(env["_received"], [("bob", 9, "int")])

        meta = env["__yggdrasil_task__"]
        self.assertEqual(meta["qualname"], "_signature_fixture")
        self.assertEqual(meta["return"], "str")
        self.assertIn("yggdrasil_version", meta)
        self.assertIn("staged_at", meta)
        json.dumps(meta)  # stable JSON shape.

    def test_content_digest_is_stable_for_same_source_and_args(self):
        """Same callable + same bound args → identical digest across calls."""
        a = _content_digest(_signature_fixture, (), {"name": "x", "count": 1})
        b = _content_digest(_signature_fixture, (), {"name": "x", "count": 1})
        self.assertEqual(a, b)
        # Kwarg ordering doesn't perturb the digest.
        c = _content_digest(_signature_fixture, (), {"count": 1, "name": "x"})
        self.assertEqual(a, c)

    def test_content_digest_changes_with_args(self):
        """Different bound args → different digest, even for the same callable."""
        a = _content_digest(_signature_fixture, (), {"name": "x"})
        b = _content_digest(_signature_fixture, (), {"name": "y"})
        self.assertNotEqual(a, b)

    def test_script_wraps_no_arg_function_with_checkargs(self):
        import ast

        def _noop() -> None:
            return None

        script = _render_callable_script(_noop, (), {})
        ast.parse(script)
        # @checkargs is always applied — a no-arg function still gets
        # the decorator so any future widget / argv re-entry is safe.
        self.assertIn("@checkargs\ndef _noop(", script)
        self.assertIn("_noop()", script)


class TestStagedScriptCapturesLocals(DatabricksTestCase):
    """``_render_callable_script`` inlines locally-referenced helpers + literals."""

    def test_inlines_same_module_helper_and_literal_constants(self):
        import ast

        script = _render_callable_script(_capture_entry, (), {"suffix": "z"})

        # AST-valid output and the captured block is present.
        ast.parse(script)
        self.assertIn("# --- captured local references ---", script)

        # Helper function source carried verbatim.
        self.assertIn("def _capture_helper(prefix: str)", script)

        # Literal constants surfaced as ``NAME = repr(value)`` assignments.
        self.assertIn("_CAPTURE_CONSTANT = 'value-from-module'", script)
        self.assertIn("_CAPTURE_MAP = {'k': 1, 'v': [1, 2, 3]}", script)

        # End-to-end exec — the staged script runs without NameError
        # against the helper / constant references.
        env: dict = {"__name__": "__main__", "_results": []}
        patched = script.replace(
            'return _capture_helper(_CAPTURE_CONSTANT) + suffix + str(_CAPTURE_MAP["k"])',
            '_results.append(_capture_helper(_CAPTURE_CONSTANT) + suffix + str(_CAPTURE_MAP["k"]))\n'
            '    return _results[-1]',
        )
        exec(compile(patched, "<staged>", "exec"), env)
        self.assertEqual(env["_results"], ["value-from-module::okz1"])

    def test_skips_imported_callables(self):
        """A reference to a stdlib symbol stays as an import, not inlined."""
        import ast

        def uses_stdlib():
            import math
            return math.sqrt(16)

        script = _render_callable_script(uses_stdlib, (), {})
        ast.parse(script)
        # ``math`` reaches the staged script via its in-body import, not
        # as a captured definition — no top-level ``def sqrt`` or
        # ``math = ...`` lines.
        self.assertNotIn("def sqrt(", script)
        self.assertNotIn("math = ", script)


class TestJobsSubmit(DatabricksTestCase):
    """One-off :meth:`Jobs.submit` path."""

    def test_submit_returns_job_run(self):
        waiter = MagicMock()
        waiter.run_id = 99
        self.jobs_api.submit.return_value = waiter

        run = self.jobs.submit(run_name="one-off", tasks=[])

        self.assertIsInstance(run, JobRun)
        self.assertEqual(run.run_id, 99)
        self.jobs_api.submit.assert_called_once()


class TestJobSkeleton(DatabricksTestCase):
    """:class:`Job` doubles as a class-level skeleton.

    Subclasses override the ``default_*`` hooks to derive each
    :class:`JobSettings` field from their own structural inputs, then
    drive the lifecycle through :meth:`deploy` / :meth:`find_for` /
    :meth:`get_for` / :meth:`delete_for`. Caller-supplied kwargs flow
    through ``**kwargs`` so any hook can read them.
    """

    def _make_skeleton(self):
        """Return a Job subclass that derives ``name`` from a ``key`` kwarg."""
        from databricks.sdk.service.jobs import (
            JobParameterDefinition,
            NotebookTask,
            Task,
            TriggerSettings,
            FileArrivalTriggerConfiguration,
        )

        class KeyedJob(Job):
            JOB_PREFIX = "skel"

            @classmethod
            def default_name(cls, *, key: str | None = None, **_kw):
                return f"{cls.JOB_PREFIX}-{key}" if key else None

            @classmethod
            def default_tasks(cls, *, key: str | None = None, **_kw):
                return [Task(
                    task_key="apply",
                    notebook_task=NotebookTask(notebook_path=f"/p/{key}"),
                )] if key else []

            @classmethod
            def default_parameters(cls, *, key: str | None = None, **_kw):
                return [JobParameterDefinition(name="key", default=key or "")]

            @classmethod
            def default_description(cls, *, key: str | None = None, **_kw):
                return f"Apply for {key}" if key else None

            @classmethod
            def default_trigger(
                cls, *, key: str | None = None, trigger_url: str | None = None, **_kw,
            ):
                if not trigger_url:
                    return None
                return TriggerSettings(
                    file_arrival=FileArrivalTriggerConfiguration(url=trigger_url),
                )

        return KeyedJob

    def test_deploy_resolves_every_setting_from_hooks(self):
        from databricks.sdk.service.jobs import Job as JobInfo

        cls = self._make_skeleton()
        self.jobs_api.create.return_value = MagicMock(job_id=42)
        self.jobs_api.list.return_value = iter([])  # find by name → not present
        self.jobs_api.get.return_value = JobInfo(
            job_id=42, settings=JobSettings(name="skel-orders"),
        )

        instance = cls.deploy(service=self.jobs, key="orders")

        self.assertIsInstance(instance, cls)
        self.assertIsInstance(instance, Job)
        self.assertEqual(instance.job_id, 42)
        # The SDK boundary saw the resolved name + tasks + description.
        _, create_kwargs = self.jobs_api.create.call_args
        self.assertEqual(create_kwargs["name"], "skel-orders")
        self.assertEqual(create_kwargs["tasks"][0].task_key, "apply")
        self.assertEqual(
            create_kwargs["tasks"][0].notebook_task.notebook_path,
            "/p/orders",
        )
        self.assertEqual(create_kwargs["description"], "Apply for orders")

    def test_deploy_passes_trigger_only_when_hook_returns_one(self):
        cls = self._make_skeleton()
        self.jobs_api.create.return_value = MagicMock(job_id=1)
        self.jobs_api.list.return_value = iter([])

        # Hook returns None when no trigger_url → no ``trigger`` kwarg.
        cls.deploy(service=self.jobs, key="a")
        _, kwargs = self.jobs_api.create.call_args
        self.assertNotIn("trigger", kwargs)

        # Hook returns TriggerSettings when trigger_url is set.
        self.jobs_api.create.reset_mock()
        self.jobs_api.create.return_value = MagicMock(job_id=2)
        cls.deploy(
            service=self.jobs, key="b", trigger_url="dbfs:/Volumes/x/y/z/",
        )
        _, kwargs = self.jobs_api.create.call_args
        self.assertIsNotNone(kwargs.get("trigger"))

    def test_deploy_without_name_resolution_raises(self):
        """Plain :class:`Job` (default ``default_name`` = passthrough)
        with no caller-supplied ``name=`` can't resolve a job name."""
        with self.assertRaises(ValueError):
            Job.deploy(service=self.jobs)

    def test_find_for_returns_none_when_absent(self):
        cls = self._make_skeleton()
        self.jobs_api.list.return_value = iter([])

        result = cls.find_for(service=self.jobs, key="missing")
        self.assertIsNone(result)

    def test_find_for_wraps_existing_into_subclass(self):
        cls = self._make_skeleton()
        existing = _base_job(job_id=77, name="skel-orders")
        self.jobs_api.list.return_value = iter([existing])

        found = cls.find_for(service=self.jobs, key="orders")
        self.assertIsNotNone(found)
        self.assertIsInstance(found, cls)
        self.assertEqual(found.job_id, 77)

    def test_get_or_create_returns_existing_when_present(self):
        cls = self._make_skeleton()
        existing = _base_job(job_id=88, name="skel-orders")
        self.jobs_api.list.return_value = iter([existing])

        instance = cls.get_or_create(service=self.jobs, key="orders")
        self.assertEqual(instance.job_id, 88)
        # Existing → no create call.
        self.jobs_api.create.assert_not_called()

    def test_get_or_create_creates_when_missing(self):
        cls = self._make_skeleton()
        self.jobs_api.list.return_value = iter([])
        self.jobs_api.create.return_value = MagicMock(job_id=99)

        instance = cls.get_or_create(service=self.jobs, key="new")
        self.assertEqual(instance.job_id, 99)
        self.jobs_api.create.assert_called_once()

    def test_delete_for_routes_through_jobs_delete(self):
        cls = self._make_skeleton()
        self.jobs_api.list.return_value = iter([
            _base_job(job_id=44, name="skel-orders"),
        ])

        cls.delete_for(service=self.jobs, key="orders")
        self.jobs_api.delete.assert_called_once_with(job_id=44)

    def test_caller_override_wins_in_default_name(self):
        """Subclass can re-implement ``default_name`` to honor caller ``name=``."""

        class OverridableJob(Job):
            @classmethod
            def default_name(cls, *, name=None, key=None, **_kw):
                return name or (f"derived-{key}" if key else None)

        self.jobs_api.create.return_value = MagicMock(job_id=1)
        self.jobs_api.list.return_value = iter([])
        OverridableJob.deploy(service=self.jobs, name="explicit", key="ignored")
        _, kwargs = self.jobs_api.create.call_args
        self.assertEqual(kwargs["name"], "explicit")

    def test_default_schedule_coerces_cron_string(self):
        """Built-in :meth:`Job.default_schedule` accepts cron-string input."""
        from databricks.sdk.service.jobs import CronSchedule

        cls = self._make_skeleton()
        self.jobs_api.create.return_value = MagicMock(job_id=1)
        self.jobs_api.list.return_value = iter([])

        cls.deploy(
            service=self.jobs, key="a",
            schedule="0 0 */6 * * ?", schedule_timezone="UTC",
        )
        _, kwargs = self.jobs_api.create.call_args
        self.assertIsInstance(kwargs["schedule"], CronSchedule)
        self.assertEqual(
            kwargs["schedule"].quartz_cron_expression, "0 0 */6 * * ?",
        )


class TestJobTaskSkeleton(DatabricksTestCase):
    """:class:`JobTask` mirrors the same skeleton pattern.

    Subclasses set ``DEFAULT_TASK_KEY`` / override :meth:`default_details`
    to declare a task shape once, then call :meth:`deploy` /
    :meth:`find_for` / :meth:`delete_for` against a parent :class:`Job`.
    Subclasses chain recursively.
    """

    def _make_job(self, *, tasks: list[Task] | None = None) -> Job:
        details = _job_info(job_id=10, name="parent", tasks=tasks)
        return Job(
            service=self.jobs, job_id=details.job_id,
            job_name=details.settings.name, details=details,
        )

    def _make_task_subclass(self):
        class NotebookApplyTask(JobTask):
            DEFAULT_TASK_KEY = "apply"

            @classmethod
            def default_details(cls, *, notebook_path: str = "/p", **_kw):
                from databricks.sdk.service.jobs import NotebookTask, Task

                return Task(
                    task_key=cls.DEFAULT_TASK_KEY,
                    notebook_task=NotebookTask(notebook_path=notebook_path),
                )

        return NotebookApplyTask

    def test_deploy_creates_task_on_job_from_skeleton(self):
        cls = self._make_task_subclass()
        job = self._make_job(tasks=[])
        self.jobs_api.get.return_value = _job_info(
            job_id=10, name="parent",
            tasks=[cls.default_details(notebook_path="/p/apply")],
        )

        instance = cls.deploy(job, notebook_path="/p/apply")

        self.assertIsInstance(instance, cls)
        self.assertEqual(instance.task_key, "apply")
        # Pushed through Job.update via Jobs API.
        self.jobs_api.update.assert_called_once()
        _, kwargs = self.jobs_api.update.call_args
        tasks = kwargs["new_settings"].tasks
        self.assertEqual(tasks[0].task_key, "apply")
        self.assertEqual(tasks[0].notebook_task.notebook_path, "/p/apply")

    def test_deploy_requires_details_resolution(self):
        """Subclass with no ``default_details`` and no caller ``details=`` errors."""

        class BareTask(JobTask):
            DEFAULT_TASK_KEY = "bare"

        job = self._make_job(tasks=[])
        with self.assertRaises(ValueError):
            BareTask.deploy(job)

    def test_find_for_returns_existing_task(self):
        from databricks.sdk.service.jobs import NotebookTask, Task

        cls = self._make_task_subclass()
        existing_task = Task(
            task_key="apply",
            notebook_task=NotebookTask(notebook_path="/p/existing"),
        )
        job = self._make_job(tasks=[existing_task])

        found = cls.find_for(job)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, cls)
        self.assertEqual(found.task_key, "apply")

    def test_find_for_returns_none_when_missing(self):
        cls = self._make_task_subclass()
        job = self._make_job(tasks=[])
        self.assertIsNone(cls.find_for(job))

    def test_delete_for_drops_task_via_fields_to_remove(self):
        from databricks.sdk.service.jobs import NotebookTask, Task

        cls = self._make_task_subclass()
        existing = Task(
            task_key="apply",
            notebook_task=NotebookTask(notebook_path="/p"),
        )
        job = self._make_job(tasks=[existing])

        cls.delete_for(job)
        self.jobs_api.update.assert_called_once()
        _, kwargs = self.jobs_api.update.call_args
        self.assertEqual(kwargs["fields_to_remove"], ["tasks/apply"])

    def test_subclass_skeleton_chains_recursively(self):
        """A subclass of a :class:`JobTask` skeleton inherits its overrides."""
        from databricks.sdk.service.jobs import NotebookTask, Task

        cls = self._make_task_subclass()

        class FastApplyTask(cls):
            """Refines parent skeleton with a different notebook path default."""

            @classmethod
            def default_details(cls, *, notebook_path: str = "/p/fast", **_kw):
                return Task(
                    task_key=cls.DEFAULT_TASK_KEY,
                    notebook_task=NotebookTask(notebook_path=notebook_path),
                )

        job = self._make_job(tasks=[])
        self.jobs_api.get.return_value = _job_info(
            job_id=10, name="parent",
            tasks=[FastApplyTask.default_details()],
        )

        FastApplyTask.deploy(job)
        _, kwargs = self.jobs_api.update.call_args
        # Default notebook path comes from the subclass override.
        self.assertEqual(
            kwargs["new_settings"].tasks[0].notebook_task.notebook_path,
            "/p/fast",
        )

    def test_default_task_key_classvar_drives_resolution(self):
        from databricks.sdk.service.jobs import NotebookTask, Task

        class KeyedTask(JobTask):
            DEFAULT_TASK_KEY = "named-step"

            @classmethod
            def default_details(cls, **_kw):
                return Task(
                    task_key=cls.DEFAULT_TASK_KEY,
                    notebook_task=NotebookTask(notebook_path="/p"),
                )

        self.assertEqual(KeyedTask.default_task_key(), "named-step")

        job = self._make_job(tasks=[])
        self.jobs_api.get.return_value = _job_info(
            job_id=10, name="parent",
            tasks=[KeyedTask.default_details()],
        )
        KeyedTask.deploy(job)
        _, kwargs = self.jobs_api.update.call_args
        self.assertEqual(kwargs["new_settings"].tasks[0].task_key, "named-step")
