"""Unit tests for the Prefect-style @task / @flow + serverless deploy."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.job import Flow, Future, Task, flow, task
from yggdrasil.version import __version__


# --------------------------------------------------------------------------- #
# @task — callable, retries, submit, with_options
# --------------------------------------------------------------------------- #
class TestTask:
    def test_decorator_yields_callable_task(self):
        @task
        def add(a, b=1):
            return a + b

        assert isinstance(add, Task)
        assert add(2) == 3                      # callable like a function
        assert add.fn.__name__ == "add"
        assert add.name == "add"

    def test_retries_until_success(self):
        calls = {"n": 0}

        @task(retries=2)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("boom")
            return "ok"

        assert flaky() == "ok"
        assert calls["n"] == 3

    def test_retries_exhausted_reraises(self):
        @task(retries=1)
        def always():
            raise ValueError("nope")

        with pytest.raises(ValueError):
            always()

    def test_submit_returns_future(self):
        @task
        def slow(x):
            time.sleep(0.01)
            return x * 2

        fut = slow.submit(21)
        assert isinstance(fut, Future)
        assert fut.result() == 42

    def test_map_fans_out(self):
        @task
        def square(x):
            return x * x

        results = [f.result() for f in square.map([1, 2, 3])]
        assert sorted(results) == [1, 4, 9]

    def test_with_options_copies(self):
        @task
        def f():
            return 1

        g = f.with_options(retries=5, name="renamed")
        assert g.retries == 5 and g.name == "renamed"
        assert f.retries == 0 and f.name == "f"   # original untouched

    def test_to_task_renders_serverless_wheel(self):
        @task(key="ld")
        def load():
            ...

        t = load.to_task(["c.s.t"])
        assert t.task_key == "ld"
        assert t.environment_key == "default"      # serverless
        assert t.python_wheel_task.parameters == ["c.s.t"]


# --------------------------------------------------------------------------- #
# @flow — callable, orchestrates tasks, deploys serverless
# --------------------------------------------------------------------------- #
class TestFlow:
    def test_flow_runs_tasks(self):
        @task
        def double(x):
            return x * 2

        @flow(name="etl")
        def etl(x):
            return double(x) + 1

        assert isinstance(etl, Flow)
        assert etl(10) == 21
        assert etl.name == "etl"

    def test_flow_fans_out_with_submit(self):
        @task
        def fetch(i):
            return i

        @flow
        def gather(items):
            return sorted(f.result() for f in fetch.map(items))

        assert gather([3, 1, 2]) == [1, 2, 3]

    def test_definition_is_serverless_v5_with_ygg_databricks(self):
        @flow(parameters=["a", "b"])
        def etl(x, y):
            ...

        spec = etl.definition()
        assert spec["name"] == "etl"
        assert "trigger" not in spec
        task_obj = spec["tasks"][0]
        assert task_obj.python_wheel_task.parameters == ["a", "b"]
        assert task_obj.environment_key == "default"
        env = spec["environments"][0]
        assert env.spec.environment_version == "5"
        assert env.spec.dependencies == [f"ygg[databricks]=={__version__}", "databricks-sdk"]

    def test_serverless_false_drops_environment(self):
        @flow
        def etl():
            ...

        etl.serverless = False
        spec = etl.definition()
        assert "environments" not in spec
        assert spec["tasks"][0].environment_key is None

    def test_trigger_included_when_set(self):
        @flow(trigger={"file_arrival": {"url": "/Volumes/x"}})
        def etl():
            ...

        assert etl.definition()["trigger"] == {"file_arrival": {"url": "/Volumes/x"}}

    def test_deploy_uses_published_ygg_by_default(self):
        @flow(name="ygg-demo", parameters=["a"])
        def demo(x):
            ...

        client = MagicMock()
        deployed = demo.deploy(client)
        client.jobs.create_or_update.assert_called_once()
        kwargs = client.jobs.create_or_update.call_args.kwargs
        assert kwargs["name"] == "ygg-demo"
        assert kwargs["tasks"][0].python_wheel_task.parameters == ["a"]
        # published ygg from the index + latest databricks-sdk — no built wheel
        assert kwargs["environments"][0].spec.dependencies == [f"ygg[databricks]=={__version__}", "databricks-sdk"]
        assert deployed is client.jobs.create_or_update.return_value

    def test_deploy_can_build_and_ship_wheel(self):
        from unittest.mock import patch

        @flow(name="ygg-demo", parameters=["a"])
        def demo(x):
            ...

        demo.build_wheel = True                       # opt in (air-gapped)
        client = MagicMock()
        wheels = [
            "/Workspace/Shared/.ygg/whl/ygg-demo/ygg-9.9-py3-none-any.whl",
            "/Workspace/Shared/.ygg/whl/ygg-demo/databricks_sdk-1.2.3-py3-none-any.whl",
        ]
        with patch("yggdrasil.databricks.job.wheel.ensure_wheel", return_value=wheels) as ew:
            demo.deploy(client)

        # isolated build of the flow's own project + deps, uploaded under the job folder
        assert ew.call_count == 1
        assert ew.call_args.kwargs["workspace_dir"] == "/Workspace/Shared/.ygg/whl/ygg-demo"
        assert ew.call_args.kwargs["requirements"] == ("databricks-sdk",)
        kwargs = client.jobs.create_or_update.call_args.kwargs
        assert kwargs["environments"][0].spec.dependencies == wheels   # all shipped wheels


def test_class_based_flow_overrides_run():
    class MyFlow(Flow):
        def __init__(self):
            super().__init__(name="mine")

        def run(self, x):
            return x + 100

    f = MyFlow()
    assert f(5) == 105                              # callable
    assert f.name == "mine"
