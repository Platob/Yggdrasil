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
