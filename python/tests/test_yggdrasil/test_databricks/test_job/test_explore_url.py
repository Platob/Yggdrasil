"""explore_url deep-links for Job and JobRun.

Builds handles against a stub service whose client exposes only a
``base_url`` — no network — and asserts the workspace UI paths, including
the fallbacks for partially-known ids.
"""
from __future__ import annotations

import pytest

from yggdrasil.databricks.job.job import Job
from yggdrasil.databricks.job.run import JobRun
from yggdrasil.databricks.job.service import Jobs, JobRuns
from yggdrasil.url import URL

_HOST = "https://dbc-test.cloud.databricks.com"


class _StubClient:
    def __init__(self, host: str = _HOST):
        self.base_url = URL.from_str(host)


class _StubJobs(Jobs):
    def __init__(self, client):
        self.client = client


class _StubJobRuns(JobRuns):
    def __init__(self, client):
        self.client = client


def _make_job(job_id: int | None = 123) -> Job:
    Job._INSTANCES.clear()
    job = object.__new__(Job)
    object.__setattr__(job, "_singleton_key_", (Job, None, job_id))
    object.__setattr__(job, "_initialized", False)
    job.service = _StubJobs(_StubClient())
    job.job_id = job_id
    job.name = "etl"
    job._details = None
    job._initialized = True
    return job


def _make_run(run_id: int | None = 456, job_id: int | None = 123) -> JobRun:
    JobRun._INSTANCES.clear()
    run = object.__new__(JobRun)
    object.__setattr__(run, "_singleton_key_", (JobRun, None, run_id))
    object.__setattr__(run, "_initialized", False)
    run.service = _StubJobRuns(_StubClient())
    run.run_id = run_id
    run.job_id = job_id
    run._details = None
    run._initialized = True
    return run


@pytest.fixture(autouse=True)
def _clear_caches():
    Job._INSTANCES.clear()
    JobRun._INSTANCES.clear()
    yield
    Job._INSTANCES.clear()
    JobRun._INSTANCES.clear()


class TestJobExploreUrl:
    def test_points_at_job_page(self):
        assert _make_job(123).explore_url.to_string() == f"{_HOST}/jobs/123"

    def test_unknown_id_falls_back(self):
        assert _make_job(None).explore_url.to_string() == f"{_HOST}/jobs/unknown"

    def test_returns_url_instance(self):
        assert isinstance(_make_job().explore_url, URL)


class TestJobRunExploreUrl:
    def test_points_at_run_page(self):
        url = _make_run(run_id=456, job_id=123).explore_url
        assert url.to_string() == f"{_HOST}/jobs/123/runs/456"

    def test_missing_job_id_falls_back_to_jobs_list(self):
        assert _make_run(run_id=456, job_id=None).explore_url.to_string() == f"{_HOST}/jobs"

    def test_missing_run_id_falls_back_to_jobs_list(self):
        assert _make_run(run_id=None, job_id=123).explore_url.to_string() == f"{_HOST}/jobs"

    def test_returns_url_instance(self):
        assert isinstance(_make_run().explore_url, URL)
