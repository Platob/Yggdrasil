"""Tests for the @function decorator framework."""
from __future__ import annotations

import pytest
from yggdrasil.node.fn import function, dag, FunctionHandle, FunctionRun, DagHandle


@function
def hello(name: str) -> str:
    return f"Hello {name}"


@function(name="custom", description="A custom function")
def my_func(x: int, y: float = 1.0) -> float:
    import numpy as np
    return np.mean([x, y])


@function
def step_a(x: int) -> int:
    return x + 1


@function
def step_b(x: int) -> int:
    return x * 2


class TestFunctionDecorator:
    def test_basic_decorator(self):
        assert isinstance(hello, FunctionHandle)
        assert hello.name == "hello"
        assert "def hello" in hello.code
        assert hello.python_version

    def test_decorator_with_args(self):
        assert my_func.name == "custom"
        assert my_func.description == "A custom function"
        assert "numpy" in my_func.dependencies

    def test_code_extraction(self):
        assert "return f" in hello.code or "return" in hello.code

    def test_with_env(self):
        clone = hello.with_env("test-env")
        assert clone is not hello
        assert clone._env_name == "test-env"
        assert clone.name == "hello"

    def test_on_remote(self):
        remote = hello.on("http://other:8100")
        assert remote is not hello
        assert remote._node_url == "http://other:8100"

    def test_chain_operator(self):
        chain = step_a >> step_b
        assert hasattr(chain, "_handles")
        assert len(chain._handles) == 2

    def test_triple_chain(self):
        @function
        def step_c(x: int) -> int:
            return x - 1

        chain = step_a >> step_b >> step_c
        assert len(chain._handles) == 3

    def test_dag_creation(self):
        pipeline = dag("test-pipeline", step_a >> step_b, description="A test DAG")
        assert isinstance(pipeline, DagHandle)
        assert pipeline.name == "test-pipeline"
        assert len(pipeline.handles) == 2

    def test_function_run_class(self):
        run = FunctionRun(run_id=12345, node_url="http://localhost:8100", function_id=67890)
        assert run.run_id == 12345
        assert not run.is_done
        assert "FunctionRun" in repr(run)
