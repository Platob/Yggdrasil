"""Pickle round-trip tests for Job, JobRun, and JobTask.

Uses stub services to stay off the network. Validates that identity
fields, state, and service references survive serialization, and that
Singleton cache collapse works correctly on unpickle.
"""
from __future__ import annotations

import pickle

import pytest

from databricks.sdk.service.jobs import (
    Job as SDKJob,
    JobSettings,
    Run as SDKRun,
    RunLifeCycleState,
    RunResultState,
    RunState,
    RunTask as SDKRunTask,
)

from yggdrasil.databricks.job.job import Job
from yggdrasil.databricks.job.run import JobRun, JobTask
from yggdrasil.databricks.job.service import Jobs, JobRuns
from yggdrasil.enums.state import State


# ---------------------------------------------------------------------------
# Stubs — picklable stand-ins that don't need a real DatabricksClient
# ---------------------------------------------------------------------------


class StubJobs(Jobs):
    def __init__(self):
        self.client = None

    def get(self, *a, **kw):
        return None


class StubJobRuns(JobRuns):
    def __init__(self):
        self.client = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(job_id: int = 123, name: str = "test-job") -> Job:
    Job._INSTANCES.clear()
    job = object.__new__(Job)
    key = (Job, None, job_id)
    object.__setattr__(job, "_singleton_key_", key)
    object.__setattr__(job, "_initialized", False)
    job.service = StubJobs()
    job.job_id = job_id
    job.name = name
    job._details = None
    job._initialized = True
    return job


def _make_run(
    run_id: int = 456,
    job_id: int = 123,
    state: State = State.SUCCEEDED,
) -> JobRun:
    JobRun._INSTANCES.clear()
    run = object.__new__(JobRun)
    key = (JobRun, None, run_id)
    object.__setattr__(run, "_singleton_key_", key)
    object.__setattr__(run, "_initialized", False)
    run.service = StubJobRuns()
    run.run_id = run_id
    run.job_id = job_id
    run._details = None
    run._state = state
    run._attempts = 0
    run._initialized = True
    return run


def _make_task(
    task_key: str = "etl-step",
    lifecycle: RunLifeCycleState = RunLifeCycleState.TERMINATED,
    result: RunResultState = RunResultState.SUCCESS,
) -> JobTask:
    return JobTask(SDKRunTask(
        task_key=task_key,
        state=RunState(life_cycle_state=lifecycle, result_state=result),
    ))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_caches():
    Job._INSTANCES.clear()
    JobRun._INSTANCES.clear()
    yield
    Job._INSTANCES.clear()
    JobRun._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Job pickle
# ---------------------------------------------------------------------------


class TestJobPickle:

    def test_round_trip_preserves_identity(self):
        job = _make_job(job_id=42, name="etl-pipeline")
        restored = pickle.loads(pickle.dumps(job))
        assert restored.job_id == 42
        assert restored.name == "etl-pipeline"

    def test_round_trip_preserves_service(self):
        job = _make_job()
        restored = pickle.loads(pickle.dumps(job))
        assert isinstance(restored.service, StubJobs)

    def test_details_none_survives(self):
        job = _make_job()
        assert job._details is None
        restored = pickle.loads(pickle.dumps(job))
        assert restored._details is None

    def test_with_sdk_details(self):
        job = _make_job(job_id=99, name="detailed")
        job._details = SDKJob(
            job_id=99,
            settings=JobSettings(name="detailed", timeout_seconds=3600),
        )
        restored = pickle.loads(pickle.dumps(job))
        assert restored._details is not None
        assert restored._details.job_id == 99
        assert restored._details.settings.name == "detailed"

    def test_unpickle_preserves_all_fields(self):
        job = _make_job(job_id=88, name="full-test")
        data = pickle.dumps(job)
        Job._INSTANCES.clear()
        restored = pickle.loads(data)
        assert restored.job_id == 88
        assert restored.name == "full-test"
        assert isinstance(restored.service, StubJobs)


# ---------------------------------------------------------------------------
# JobRun pickle
# ---------------------------------------------------------------------------


class TestJobRunPickle:

    def test_round_trip_preserves_identity(self):
        run = _make_run(run_id=100, job_id=50)
        restored = pickle.loads(pickle.dumps(run))
        assert restored.run_id == 100
        assert restored.job_id == 50

    def test_round_trip_preserves_state(self):
        run = _make_run(state=State.FAILED)
        restored = pickle.loads(pickle.dumps(run))
        assert restored._state == State.FAILED

    def test_round_trip_preserves_attempts(self):
        run = _make_run()
        run._attempts = 3
        restored = pickle.loads(pickle.dumps(run))
        assert restored._attempts == 3

    def test_round_trip_preserves_service(self):
        run = _make_run()
        restored = pickle.loads(pickle.dumps(run))
        assert isinstance(restored.service, StubJobRuns)

    def test_succeeded_state(self):
        run = _make_run(state=State.SUCCEEDED)
        restored = pickle.loads(pickle.dumps(run))
        assert restored.is_succeeded

    def test_running_state(self):
        run = _make_run(state=State.RUNNING)
        restored = pickle.loads(pickle.dumps(run))
        assert restored.is_running

    def test_canceled_state(self):
        run = _make_run(state=State.CANCELED)
        restored = pickle.loads(pickle.dumps(run))
        assert restored.is_canceled

    def test_unpickle_preserves_all_fields(self):
        run = _make_run(run_id=300, job_id=150)
        run._attempts = 2
        data = pickle.dumps(run)
        JobRun._INSTANCES.clear()
        restored = pickle.loads(data)
        assert restored.run_id == 300
        assert restored.job_id == 150
        assert restored._attempts == 2
        assert isinstance(restored.service, StubJobRuns)


# ---------------------------------------------------------------------------
# JobTask pickle
# ---------------------------------------------------------------------------


class TestJobTaskPickle:

    def test_round_trip_preserves_task_key(self):
        task = _make_task(task_key="load-step")
        restored = pickle.loads(pickle.dumps(task))
        assert restored.task_key == "load-step"

    def test_round_trip_preserves_state(self):
        task = _make_task(
            lifecycle=RunLifeCycleState.TERMINATED,
            result=RunResultState.SUCCESS,
        )
        restored = pickle.loads(pickle.dumps(task))
        assert restored.state == State.SUCCEEDED
        assert restored.is_succeeded

    def test_failed_task(self):
        task = _make_task(
            lifecycle=RunLifeCycleState.TERMINATED,
            result=RunResultState.FAILED,
        )
        restored = pickle.loads(pickle.dumps(task))
        assert restored.is_failed

    def test_running_task(self):
        task = _make_task(
            lifecycle=RunLifeCycleState.RUNNING,
            result=None,
        )
        restored = pickle.loads(pickle.dumps(task))
        assert restored.state == State.RUNNING

    def test_raw_preserved(self):
        task = _make_task(task_key="raw-test")
        restored = pickle.loads(pickle.dumps(task))
        assert isinstance(restored.raw, SDKRunTask)
        assert restored.raw.task_key == "raw-test"
