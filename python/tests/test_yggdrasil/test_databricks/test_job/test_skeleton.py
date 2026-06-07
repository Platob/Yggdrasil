"""Unit tests for the Prefect-style @task / @flow + transparent serverless deploy."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.job import Flow, Future, Task, flow, task
from yggdrasil.version import __version__


# --------------------------------------------------------------------------- #
# @task — local execution, retries, submit, with_options
# --------------------------------------------------------------------------- #
class TestTask:
    def test_decorator_yields_callable_task(self):
        @task
        def add(a, b=1):
            return a + b

        assert isinstance(add, Task)
        assert add.local(2) == 3                 # .local() runs in-process
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

        assert flaky.local() == "ok"
        assert calls["n"] == 3

    def test_retries_exhausted_reraises(self):
        @task(retries=1)
        def always():
            raise ValueError("nope")

        with pytest.raises(ValueError):
            always.local()

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
        assert t.python_wheel_task.entry_point == "ygg"
        # the single ``ygg`` entry point gets a leading ``run`` subcommand
        assert t.python_wheel_task.parameters == ["run", "c.s.t"]


# --------------------------------------------------------------------------- #
# transparent dispatch — in-Databricks runs local, elsewhere routes remote
# --------------------------------------------------------------------------- #
class TestTransparentDispatch:
    def test_call_inside_databricks_runs_in_process(self):
        @task
        def add(a, b):
            return a + b

        with patch(
            "yggdrasil.databricks.client.DatabricksClient.is_in_databricks_environment",
            return_value=True,
        ):
            assert add(2, 3) == 5                   # ran locally, no deploy

    def test_call_outside_databricks_dispatches_remote(self):
        @flow
        def etl(x):
            return x

        with patch(
            "yggdrasil.databricks.client.DatabricksClient.is_in_databricks_environment",
            return_value=False,
        ), patch.object(Flow, "_dispatch_remote", return_value="REMOTE") as disp:
            assert etl(7, k=1) == "REMOTE"
        disp.assert_called_once_with((7,), {"k": 1})

    def test_target_ref_points_at_the_decorated_object(self):
        # A module-level flow has an importable target the runner can resolve.
        ref = module_level_flow._target_ref()
        assert ref == f"{__name__}:module_level_flow"


# --------------------------------------------------------------------------- #
# @flow — local orchestration + serverless rendering
# --------------------------------------------------------------------------- #
class TestFlow:
    def test_flow_runs_tasks_locally(self):
        @task
        def double(x):
            return x * 2

        @flow(name="etl")
        def etl(x):
            return double(x) + 1                    # task call → in-process here

        assert isinstance(etl, Flow)
        assert etl.local(10) == 21
        assert etl.name == "etl"

    def test_flow_fans_out_with_submit(self):
        @task
        def fetch(i):
            return i

        @flow
        def gather(items):
            return sorted(f.result() for f in fetch.map(items))

        assert gather.local([3, 1, 2]) == [1, 2, 3]

    def test_definition_runs_target_via_ygg_run(self):
        from yggdrasil.databricks.wheels.service import serverless_environment_version

        @flow(parameters=["a", "b"])
        def etl(x, y):
            ...

        spec = etl.definition()
        assert spec["name"] == "etl"
        assert "trigger" not in spec
        task_obj = spec["tasks"][0]
        # the cluster runs the single ``ygg`` entry point's ``run`` subcommand
        # against the target + scheduled params
        assert task_obj.python_wheel_task.package_name == "ygg"
        assert task_obj.python_wheel_task.entry_point == "ygg"
        assert task_obj.python_wheel_task.parameters == ["run", etl._target_ref(), "a", "b"]
        assert task_obj.environment_key == "default"
        env = spec["environments"][0]
        assert env.spec.environment_version == serverless_environment_version()
        # no wheels shipped yet (definition without deploy) → published fallback
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

    def test_job_tags_flow_into_definition(self):
        @flow
        def etl():
            ...

        assert "tags" not in etl.definition()           # none by default
        etl.job_tags = {"ygg": "demo", "team": "data"}
        assert etl.definition()["tags"] == {"ygg": "demo", "team": "data"}

    def test_deploy_ships_composed_wheels_by_default(self):
        @flow(name="ygg-demo", parameters=["a"])
        def demo(x):
            ...

        client = MagicMock()
        wheels = [
            "/Workspace/Shared/pypi/ygg/ygg-9.9-py3-none-any.whl",
            "xxhash==1.2.3",
        ]
        with patch.object(Flow, "_serverless_dependencies", return_value=wheels) as sd:
            deployed = demo.deploy(client)

        sd.assert_called_once_with(client)
        kwargs = client.jobs.create_or_update.call_args.kwargs
        assert kwargs["name"] == "ygg-demo"
        assert kwargs["tasks"][0].python_wheel_task.parameters == ["run", demo._target_ref(), "a"]
        assert kwargs["environments"][0].spec.dependencies == wheels  # shipped by path
        assert deployed is client.jobs.create_or_update.return_value

    def test_deploy_can_use_published_ygg(self):
        @flow(name="ygg-demo", parameters=["a"])
        def demo(x):
            ...

        demo.build_wheel = False                      # opt out → pip-install from index
        client = MagicMock()
        deployed = demo.deploy(client)
        client.jobs.create_or_update.assert_called_once()
        kwargs = client.jobs.create_or_update.call_args.kwargs
        assert kwargs["environments"][0].spec.dependencies == [
            f"ygg[databricks]=={__version__}", "databricks-sdk",
        ]
        assert deployed is client.jobs.create_or_update.return_value


def test_class_based_flow_overrides_run():
    class MyFlow(Flow):
        def __init__(self):
            super().__init__(name="mine")

        def run(self, x):
            return x + 100

    f = MyFlow()
    assert f.local(5) == 105                          # in-process
    assert f.name == "mine"
    assert "MyFlow" in f._target_ref()                # class-based target


# A module-level flow target (importable ``module:qualname`` for the runner).
@flow(name="module-level")
def module_level_flow(x):
    return x


def test_all_environments_attaches_one_env_per_python():
    import types
    from unittest.mock import MagicMock, patch

    @flow(name="multi")
    def f(x):
        ...

    f.all_environments = True
    client = MagicMock()
    client.environments.create.side_effect = lambda spec, *, extras=(), python=None, **k: \
        types.SimpleNamespace(serverless=f"/ws/env/ygg-{python or 'default'}.yml", dependencies=[])
    with patch.object(type(f), "_project_spec", return_value="ygg"):
        f._serverless_dependencies(client)
        envs = f.environments()
    keys = [e.environment_key for e in envs]
    assert keys == ["default", "py310", "py311", "py312", "py313"]
    by = {e.environment_key: e for e in envs}
    assert by["py311"].spec.base_environment == "/ws/env/ygg-3.11.yml"


def test_default_only_single_environment_without_flag():
    @flow(name="single")
    def f(x):
        ...

    f._wheel_paths = ("/ws/ygg-1.0-py3-none-any.whl",)
    envs = f.environments()                          # no deploy yet → inline fallback
    assert [e.environment_key for e in envs] == ["default"]


def test_built_environment_referenced_by_path():
    import types

    @flow(name="job")
    def f(x):
        ...

    # Simulate a deploy that built the project's base environment.
    f._environment = types.SimpleNamespace(
        serverless="/Workspace/Shared/environment/ygg/ygg-1.0-py311.yml", dependencies=[])
    envs = f.environments()
    assert len(envs) == 1
    spec = envs[0].spec
    assert spec.base_environment == "/Workspace/Shared/environment/ygg/ygg-1.0-py311.yml"


def test_serverless_dependencies_builds_env_via_service():
    import types
    from unittest.mock import MagicMock, patch

    @flow(name="splitjob")
    def f(x):
        ...

    client = MagicMock()
    env = types.SimpleNamespace(
        serverless="/ws/env/ygg-1.0-py311.yml",
        dependencies=["/ws/pypi/ygg/ygg-1.0-py3-none-any.whl", "pyarrow>=20"])
    client.environments.create.return_value = env
    with patch.object(type(f), "_project_spec", return_value="ygg"):
        flat = f._serverless_dependencies(client)

    # The project's zero-PyPI base environment was built once through the service…
    client.environments.create.assert_called_once()
    assert f._environment is env
    # …and its wheel closure is the flat (inline) fallback dependency list.
    assert flat == ["/ws/pypi/ygg/ygg-1.0-py3-none-any.whl", "pyarrow>=20"]
