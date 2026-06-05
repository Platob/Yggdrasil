"""Unit tests for Job.update's no-op skip + settings comparison.

When a job's desired settings already match what the API returns, ``update``
must skip the ``reset`` call entirely — there's no diff to apply. ``settings_diff``
/ ``settings_match`` expose that comparison so a caller can verify whether the
config it built reproduces the API's exactly.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from databricks.sdk.service.jobs import (
    Job as SDKJob,
    JobSettings,
    NotebookTask,
    Task,
)

from yggdrasil.databricks.job.job import Job


def _task(key="a", path="/n"):
    return Task(task_key=key, notebook_task=NotebookTask(notebook_path=path))


def _job(settings: JobSettings) -> Job:
    job = object.__new__(Job)
    job.__dict__["_initialized"] = True
    job.service = MagicMock()
    job.service.default_tags.return_value = {}
    job.job_id = 42
    job.name = "j"
    job._details = SDKJob(job_id=42, settings=settings)
    return job


def _sdk(job: Job):
    return job.service.client.workspace_client.return_value.jobs


class TestJobUpdateSkip:
    def _settings(self, *, timeout=3600, tasks=None):
        return JobSettings(
            name="j",
            tasks=tasks or [_task()],
            timeout_seconds=timeout,
            max_concurrent_runs=1,
        )

    def test_update_skips_reset_when_settings_unchanged(self):
        job = _job(self._settings())
        job.update(timeout_seconds=3600)               # same as current
        _sdk(job).reset.assert_not_called()

    def test_update_resets_when_a_field_changes(self):
        job = _job(self._settings(timeout=3600))
        job.update(timeout_seconds=7200)               # differs → real update
        _sdk(job).reset.assert_called_once()

    def test_update_skips_when_same_tasks_replayed(self):
        # Redeploying the identical task set is a no-op — no reset.
        job = _job(self._settings(tasks=[_task("a", "/n")]))
        job.update(tasks=[_task("a", "/n")])
        _sdk(job).reset.assert_not_called()

    def test_update_resets_when_tasks_differ(self):
        job = _job(self._settings(tasks=[_task("a", "/n")]))
        job.update(tasks=[_task("a", "/other")])
        _sdk(job).reset.assert_called_once()


class TestSettingsDiff:
    def test_diff_empty_when_equivalent(self):
        job = _job(JobSettings(name="j", timeout_seconds=3600, max_concurrent_runs=1))
        desired = JobSettings(name="j", timeout_seconds=3600, max_concurrent_runs=1)
        assert job.settings_diff(desired) == {}
        assert job.settings_match(desired) is True

    def test_diff_reports_changed_fields(self):
        job = _job(JobSettings(name="j", timeout_seconds=3600, max_concurrent_runs=1))
        desired = JobSettings(name="j", timeout_seconds=7200, max_concurrent_runs=2)
        diff = job.settings_diff(desired)
        assert diff["timeout_seconds"] == {"current": 3600, "desired": 7200}
        assert diff["max_concurrent_runs"] == {"current": 1, "desired": 2}
        assert job.settings_match(desired) is False

    def test_match_false_when_no_current_settings(self):
        job = _job(None) if False else None
        j = object.__new__(Job)
        j.__dict__["_initialized"] = True
        j.service = MagicMock()
        j.job_id = 1
        j.name = "j"
        j._details = SDKJob(job_id=1, settings=None)
        assert j.settings_match(JobSettings(name="j")) is False
