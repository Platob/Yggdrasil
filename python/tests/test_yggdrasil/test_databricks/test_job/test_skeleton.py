"""Unit tests for the dataclass skeletons + @job / @task decorators."""
from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.job import (
    CallableSkeleton,
    JobSkeleton,
    TaskSkeleton,
    job,
    task,
)


# --------------------------------------------------------------------------- #
# CallableSkeleton — fields are the parameters; callable
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class _Greeter(CallableSkeleton):
    who: str
    times: int = 1

    def run(self):
        return "hi " * self.times + self.who


class TestCallableSkeleton:
    def test_fields_are_parameters(self):
        assert _Greeter("ada", 2).parameters() == ["ada", "2"]

    def test_is_callable(self):
        assert _Greeter("ada")() == "hi ada"


# --------------------------------------------------------------------------- #
# @task — build a TaskSkeleton from a function signature
# --------------------------------------------------------------------------- #
class TestTaskDecorator:
    def test_builds_task_skeleton_from_signature(self):
        @task(depends_on=["extract"], timeout_seconds=600)
        def load(table: str, mode: str = "append"):
            return (table, mode)

        assert issubclass(load, TaskSkeleton)
        assert [f.name for f in dataclasses.fields(load)] == ["table", "mode"]
        inst = load(table="c.s.t")
        assert inst.parameters() == ["c.s.t", "append"]   # from the fields
        assert inst() == ("c.s.t", "append")              # runs the function
        assert load.task_key == "load"
        assert load.depends_on == ("extract",)
        assert load.task_options == {"timeout_seconds": 600}

    def test_to_task_renders_python_wheel(self):
        @task(key="ld")
        def load(table: str):
            return table

        t = load(table="c.s.t").to_task()
        assert t.task_key == "ld"
        assert t.python_wheel_task.parameters == ["c.s.t"]


# --------------------------------------------------------------------------- #
# @job — build a JobSkeleton from a function signature
# --------------------------------------------------------------------------- #
class TestJobDecorator:
    def test_builds_job_skeleton_from_signature(self):
        @job(name="ygg-etl")
        def etl(src: str, dst: str = "out"):
            return f"{src}->{dst}"

        assert issubclass(etl, JobSkeleton)
        inst = etl(src="a")
        assert inst.name == "ygg-etl"
        assert inst.parameters() == ["a", "out"]
        assert inst() == "a->out"                          # callable

    def test_default_single_task_definition(self):
        @job
        def etl(src: str):
            return src

        spec = etl(src="a").definition()
        assert spec["name"] == "etl"
        assert "trigger" not in spec
        task_obj = spec["tasks"][0]
        assert task_obj.python_wheel_task.parameters == ["a"]

    def test_defaults_to_serverless_v5_with_ygg_databricks(self):
        @job
        def etl(src: str):
            return src

        spec = etl(src="a").definition()
        env = spec["environments"][0]
        assert env.environment_key == "default"
        assert env.spec.environment_version == "5"
        assert env.spec.dependencies == ["ygg[databricks]"]
        # the task runs in that serverless environment
        assert spec["tasks"][0].environment_key == "default"

    def test_serverless_false_drops_environments(self):
        @job
        def etl(src: str):
            return src

        etl.serverless = False
        spec = etl(src="a").definition()
        assert "environments" not in spec
        assert spec["tasks"][0].environment_key is None

    def test_trigger_included_when_set(self):
        @job(trigger={"file_arrival": {"url": "/Volumes/x"}})
        def etl(src: str):
            return src

        assert etl(src="a").definition()["trigger"] == {"file_arrival": {"url": "/Volumes/x"}}


# --------------------------------------------------------------------------- #
# JobSkeleton with composed task steps
# --------------------------------------------------------------------------- #
class TestComposedJob:
    def _steps(self):
        @task
        def extract(src: str):
            return f"x:{src}"

        @task(depends_on=["extract"])
        def load(src: str):
            return f"l:{src}"

        return extract, load

    def test_definition_one_task_per_step_with_deps(self):
        extract, load = self._steps()

        @job(name="ygg-pipe", steps=[extract, load])
        def pipe(src: str):
            ...

        spec = pipe(src="a").definition()
        tasks = {t.task_key: t for t in spec["tasks"]}
        assert set(tasks) == {"extract", "load"}
        assert tasks["extract"].depends_on is None
        assert tasks["load"].depends_on[0].task_key == "extract"
        # the job's field binds into each step by name
        assert tasks["extract"].python_wheel_task.parameters == ["a"]

    def test_call_runs_steps_in_dependency_order(self):
        extract, load = self._steps()

        @job(steps=[load, extract])     # declared out of order
        def pipe(src: str):
            ...

        result = pipe(src="a")()
        assert result == {"extract": "x:a", "load": "l:a"}


def test_deploy_get_or_creates_via_service():
    @job(name="ygg-demo")
    def demo(src: str):
        return src

    jobs = MagicMock()
    deployed = demo(src="a").deploy(jobs)
    jobs.create_or_update.assert_called_once()
    assert jobs.create_or_update.call_args.kwargs["name"] == "ygg-demo"
    assert deployed is jobs.create_or_update.return_value


def test_jobskeleton_requires_name():
    # abstract `name` → can't instantiate the bare base
    with pytest.raises(TypeError):
        JobSkeleton()
