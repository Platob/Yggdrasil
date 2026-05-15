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

from yggdrasil.databricks.jobs import Job, JobRun
from yggdrasil.databricks.tests import DatabricksTestCase


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
