"""Tests for the abstract execution framework and PyEnvironment."""
from __future__ import annotations

import platform
import sys
from unittest.mock import patch

import pytest

from yggdrasil.node.services.execution import (
    Environment,
    Executable,
    Execution,
    PyEnvironment,
    PyFunction,
    PyFunctionExecution,
    ShellCommand,
    ShellCommandExecution,
    venv_python,
)


# ---------------------------------------------------------------------------
# venv_python cross-platform path
# ---------------------------------------------------------------------------

class TestVenvPython:
    def test_linux_path(self):
        with patch("yggdrasil.node.services.execution._IS_WINDOWS", False):
            result = venv_python("/opt/envs/myenv")
        assert result.endswith("bin/python")

    def test_windows_path(self):
        with patch("yggdrasil.node.services.execution._IS_WINDOWS", True):
            result = venv_python("/opt/envs/myenv")
        assert "Scripts" in result
        assert result.endswith("python.exe")

    def test_pathlib_input(self):
        from pathlib import Path
        result = venv_python(Path("/some/venv"))
        assert "python" in result


# ---------------------------------------------------------------------------
# Executable / Execution type system
# ---------------------------------------------------------------------------

class TestTypeHierarchy:
    def test_pyfunction_is_executable(self):
        exe = PyFunction(code="pass")
        assert isinstance(exe, Executable)

    def test_shell_command_is_executable(self):
        exe = ShellCommand(command=["echo", "hi"])
        assert isinstance(exe, Executable)

    def test_pyfunction_execution_is_execution(self):
        ex = PyFunctionExecution(status="completed")
        assert isinstance(ex, Execution)

    def test_shell_command_execution_is_execution(self):
        ex = ShellCommandExecution(status="completed")
        assert isinstance(ex, Execution)

    def test_pyfunction_defaults(self):
        exe = PyFunction()
        assert exe.code == ""
        assert exe.args == {}
        assert exe.timeout == 30.0
        assert exe.max_memory_mb is None

    def test_shell_command_defaults(self):
        exe = ShellCommand()
        assert exe.command == []
        assert exe.cwd is None
        assert exe.env == {}
        assert exe.stdin is None
        assert exe.timeout == 30.0


# ---------------------------------------------------------------------------
# Environment dispatch
# ---------------------------------------------------------------------------

class TestEnvironmentDispatch:
    def test_dispatch_pyfunction(self):
        env = PyEnvironment()
        exe = PyFunction(code="print('hello')", timeout=10.0)
        result = env.execute(exe)
        assert isinstance(result, PyFunctionExecution)
        assert result.status == "completed"
        assert "hello" in (result.stdout or "")

    def test_dispatch_shell_command(self):
        env = PyEnvironment()
        exe = ShellCommand(command=["echo", "world"], timeout=10.0)
        result = env.execute(exe)
        assert isinstance(result, ShellCommandExecution)
        assert result.status == "completed"
        assert "world" in (result.stdout or "")

    def test_dispatch_unsupported_type(self):
        class CustomExe(Executable):
            pass

        env = PyEnvironment()
        with pytest.raises(TypeError, match="Unsupported executable"):
            env.execute(CustomExe())


# ---------------------------------------------------------------------------
# PyEnvironment.execute_pyfunction
# ---------------------------------------------------------------------------

class TestPyFunctionExecution:
    def test_simple_print(self):
        env = PyEnvironment()
        result = env.execute_pyfunction(PyFunction(code="print('ok')", timeout=10.0))
        assert result.status == "completed"
        assert result.returncode == 0
        assert "ok" in (result.stdout or "")

    def test_args_injection(self):
        code = "import json, os\ndata = json.loads(os.environ['__ygg_inputs__'])\nprint(data['x'])"
        env = PyEnvironment()
        result = env.execute_pyfunction(PyFunction(code=code, args={"x": 42}, timeout=10.0))
        assert result.status == "completed"
        assert "42" in (result.stdout or "")

    def test_error_returns_failed(self):
        env = PyEnvironment()
        result = env.execute_pyfunction(PyFunction(code="raise ValueError('boom')", timeout=10.0))
        assert result.status == "failed"
        assert result.returncode != 0
        assert "boom" in (result.stderr or "")

    def test_syntax_error(self):
        env = PyEnvironment()
        result = env.execute_pyfunction(PyFunction(code="def foo(\n", timeout=10.0))
        assert result.status == "failed"

    def test_timeout(self):
        env = PyEnvironment()
        result = env.execute_pyfunction(PyFunction(
            code="import time; time.sleep(60)",
            timeout=0.5,
        ))
        assert result.status == "failed"
        assert "Timed out" in (result.stderr or "")

    def test_structured_outputs(self):
        code = (
            "import json, os\n"
            "with open(os.environ['__ygg_outputs_file__'], 'w') as f:\n"
            "    json.dump({'answer': 42}, f)\n"
        )
        env = PyEnvironment()
        result = env.execute_pyfunction(PyFunction(code=code, timeout=10.0))
        assert result.status == "completed"
        assert result.result == {"answer": 42}

    def test_node_env_injected(self):
        code = "import os; print(os.environ.get('MY_NODE_VAR', 'MISSING'))"
        env = PyEnvironment(node_env={"MY_NODE_VAR": "injected_val"})
        result = env.execute_pyfunction(PyFunction(code=code, timeout=10.0))
        assert result.status == "completed"
        assert "injected_val" in (result.stdout or "")

    def test_custom_python_bin(self):
        env = PyEnvironment(python_bin=sys.executable)
        result = env.execute_pyfunction(PyFunction(code="print('custom')", timeout=10.0))
        assert result.status == "completed"

    def test_duration_tracked(self):
        env = PyEnvironment()
        result = env.execute_pyfunction(PyFunction(code="pass", timeout=10.0))
        assert result.duration is not None
        assert result.duration >= 0

    def test_unicode_code(self):
        env = PyEnvironment()
        result = env.execute_pyfunction(PyFunction(code="print('日本語')", timeout=10.0))
        assert result.status == "completed"

    def test_empty_code(self):
        env = PyEnvironment()
        result = env.execute_pyfunction(PyFunction(code="", timeout=10.0))
        assert result.status == "completed"
        assert result.returncode == 0

    def test_from_venv_constructor(self):
        env = PyEnvironment.from_venv("/nonexistent/venv")
        assert "python" in env.python_bin


# ---------------------------------------------------------------------------
# PyEnvironment.execute_shell_command
# ---------------------------------------------------------------------------

class TestShellCommandExecution:
    def test_simple_echo(self):
        env = PyEnvironment()
        result = env.execute_shell_command(ShellCommand(command=["echo", "hi"], timeout=10.0))
        assert result.status == "completed"
        assert result.returncode == 0
        assert "hi" in (result.stdout or "")

    def test_failing_command(self):
        env = PyEnvironment()
        result = env.execute_shell_command(ShellCommand(command=["false"], timeout=10.0))
        assert result.status == "failed"
        assert result.returncode != 0

    def test_command_with_env(self):
        env = PyEnvironment()
        result = env.execute_shell_command(ShellCommand(
            command=["printenv", "TEST_EXEC_VAR"],
            env={"TEST_EXEC_VAR": "hello_exec"},
            timeout=10.0,
        ))
        assert result.status == "completed"
        assert "hello_exec" in (result.stdout or "")

    def test_command_with_stdin(self):
        env = PyEnvironment()
        result = env.execute_shell_command(ShellCommand(
            command=["cat"],
            stdin="piped input",
            timeout=10.0,
        ))
        assert result.status == "completed"
        assert "piped input" in (result.stdout or "")

    def test_command_timeout(self):
        env = PyEnvironment()
        result = env.execute_shell_command(ShellCommand(
            command=["sleep", "60"],
            timeout=0.5,
        ))
        assert result.status == "failed"
        assert "Timed out" in (result.stderr or "")

    def test_duration_tracked(self):
        env = PyEnvironment()
        result = env.execute_shell_command(ShellCommand(command=["true"], timeout=10.0))
        assert result.duration is not None
        assert result.duration >= 0

    def test_node_env_available(self):
        env = PyEnvironment(node_env={"NODE_EXEC_VAR": "from_node"})
        result = env.execute_shell_command(ShellCommand(
            command=["printenv", "NODE_EXEC_VAR"],
            timeout=10.0,
        ))
        assert result.status == "completed"
        assert "from_node" in (result.stdout or "")

    def test_command_env_overrides_node_env(self):
        env = PyEnvironment(node_env={"OVERRIDE_VAR": "node"})
        result = env.execute_shell_command(ShellCommand(
            command=["printenv", "OVERRIDE_VAR"],
            env={"OVERRIDE_VAR": "command"},
            timeout=10.0,
        ))
        assert result.status == "completed"
        assert "command" in (result.stdout or "")


# ---------------------------------------------------------------------------
# Abstract base classes cannot be instantiated directly
# ---------------------------------------------------------------------------

class TestAbstractEnforcement:
    def test_cannot_instantiate_environment(self):
        with pytest.raises(TypeError):
            Environment()

    def test_subclass_must_implement_methods(self):
        class IncompleteEnv(Environment):
            pass

        with pytest.raises(TypeError):
            IncompleteEnv()

    def test_custom_environment_works(self):
        class MockEnvironment(Environment):
            def execute_pyfunction(self, exe):
                return PyFunctionExecution(status="mocked")

            def execute_shell_command(self, exe):
                return ShellCommandExecution(status="mocked")

        env = MockEnvironment()
        result = env.execute(PyFunction(code="irrelevant"))
        assert result.status == "mocked"
