"""Unit tests for the generic JobSkeleton (declarative Python-job definition)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.job import JobSkeleton


class _Demo(JobSkeleton):
    entry_point = "ygg-demo"
    task_key = "demo"

    @property
    def name(self) -> str:
        return "ygg-demo-job"

    def parameters(self):
        return ["a", "b"]

    def run(self, *args, **kwargs):
        return "ran"


def test_is_abstract():
    with pytest.raises(TypeError):
        JobSkeleton()  # name + run are abstract


def test_default_definition_builds_python_wheel_task():
    spec = _Demo().definition()
    assert spec["name"] == "ygg-demo-job"
    assert "trigger" not in spec               # default trigger is None → omitted
    task = spec["tasks"][0]
    assert task.task_key == "demo"
    assert task.python_wheel_task.package_name == "yggdrasil"
    assert task.python_wheel_task.entry_point == "ygg-demo"
    assert task.python_wheel_task.parameters == ["a", "b"]


def test_trigger_included_when_set():
    class Triggered(_Demo):
        def trigger(self):
            return {"file_arrival": {"url": "/Volumes/x"}}

    spec = Triggered().definition()
    assert spec["trigger"] == {"file_arrival": {"url": "/Volumes/x"}}


def test_deploy_get_or_creates_via_service():
    jobs = MagicMock()
    job = _Demo().deploy(jobs)
    jobs.create_or_update.assert_called_once()
    kwargs = jobs.create_or_update.call_args.kwargs
    assert kwargs["name"] == "ygg-demo-job"
    assert "tasks" in kwargs and "name" not in kwargs.get("tasks", [])
    assert job is jobs.create_or_update.return_value


def test_run_is_the_python_body():
    assert _Demo().run() == "ran"


def test_callable_runs_the_body_when_no_tasks():
    # No @task methods → calling the skeleton calls run().
    assert _Demo()() == "ran"


class _Etl(JobSkeleton):
    @property
    def name(self) -> str:
        return "ygg-etl"

    @JobSkeleton.task
    def extract(self):
        self.calls.append("extract")
        return "x"

    @JobSkeleton.task(key="load", depends_on=["extract"])
    def load(self):
        self.calls.append("load")
        return "l"

    def __init__(self):
        self.calls = []


class TestTaskDecorator:
    def test_discovers_tasks(self):
        specs = {s.key: s for s in _Etl._task_specs()}
        assert set(specs) == {"extract", "load"}
        assert specs["load"].depends_on == ("extract",)

    def test_definition_builds_one_task_per_method_with_deps(self):
        spec = _Etl().definition()
        tasks = {t.task_key: t for t in spec["tasks"]}
        assert set(tasks) == {"extract", "load"}
        assert tasks["extract"].depends_on is None
        assert tasks["load"].depends_on[0].task_key == "extract"
        # multi-task → the task key is prepended to the wheel parameters
        assert tasks["load"].python_wheel_task.parameters == ["load"]

    def test_call_runs_tasks_in_dependency_order(self):
        etl = _Etl()
        results = etl()
        assert etl.calls == ["extract", "load"]     # topo order honoured
        assert results == {"extract": "x", "load": "l"}

    def test_single_task_forwards_call_args(self):
        seen = {}

        class One(JobSkeleton):
            @property
            def name(self):
                return "one"

            @JobSkeleton.task
            def go(self, *, n=0):
                seen["n"] = n
                return n

        assert One()(n=5) == 5                       # *args/**kwargs forwarded
        assert seen["n"] == 5

