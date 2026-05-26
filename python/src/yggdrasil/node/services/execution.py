from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Executable — something that can be run in an Environment
# ---------------------------------------------------------------------------

@dataclass
class Executable(ABC):
    timeout: float = 30.0


@dataclass
class PyFunction(Executable):
    code: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    max_memory_mb: int | None = None


@dataclass
class ShellCommand(Executable):
    command: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    stdin: str | None = None


# ---------------------------------------------------------------------------
# Execution — the result of running an Executable
# ---------------------------------------------------------------------------

@dataclass
class Execution(ABC):
    status: str = "pending"
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    duration: float | None = None


@dataclass
class PyFunctionExecution(Execution):
    result: Any = None


@dataclass
class ShellCommandExecution(Execution):
    pass


# ---------------------------------------------------------------------------
# Environment — abstract executor that dispatches by Executable type
# ---------------------------------------------------------------------------

class Environment(ABC):
    """Abstract execution environment.

    Subclasses implement the concrete strategy for each :class:`Executable`
    type.  The generic :meth:`execute` dispatches to the right handler.
    """

    def execute(self, exe: Executable) -> Execution:
        if isinstance(exe, PyFunction):
            return self.execute_pyfunction(exe)
        if isinstance(exe, ShellCommand):
            return self.execute_shell_command(exe)
        raise TypeError(f"Unsupported executable: {type(exe).__name__}")

    @abstractmethod
    def execute_pyfunction(self, exe: PyFunction) -> PyFunctionExecution: ...

    @abstractmethod
    def execute_shell_command(self, exe: ShellCommand) -> ShellCommandExecution: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IS_WINDOWS = platform.system() == "Windows"


def venv_python(venv_path: str | Path) -> str:
    """Return the python binary inside a virtualenv (cross-platform)."""
    venv_path = Path(venv_path)
    if _IS_WINDOWS:
        return str(venv_path / "Scripts" / "python.exe")
    return str(venv_path / "bin" / "python")


def _make_preexec_fn(max_memory_mb: int | None):
    if max_memory_mb is None or platform.system() != "Linux":
        return None

    import resource as resource_mod

    memory_bytes = max_memory_mb * 1024 * 1024

    def _set_limits():
        resource_mod.setrlimit(resource_mod.RLIMIT_AS, (memory_bytes, memory_bytes))

    return _set_limits


# ---------------------------------------------------------------------------
# PyEnvironment — concrete implementation using subprocess
# ---------------------------------------------------------------------------

class PyEnvironment(Environment):
    """Executes code via subprocess with an optional virtualenv python."""

    def __init__(
        self,
        python_bin: str | None = None,
        node_env: dict[str, str] | None = None,
    ) -> None:
        self.python_bin = python_bin or sys.executable
        self.node_env = node_env or {}

    @classmethod
    def from_venv(
        cls,
        venv_path: str | Path,
        node_env: dict[str, str] | None = None,
    ) -> PyEnvironment:
        return cls(python_bin=venv_python(venv_path), node_env=node_env)

    # -- PyFunction ---------------------------------------------------------

    def execute_pyfunction(self, exe: PyFunction) -> PyFunctionExecution:
        t0 = time.monotonic()
        tmp = None
        outputs_file = None

        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False,
            )
            preamble = f"import json\nargs = json.loads({json.dumps(json.dumps(exe.args))!r})\n"
            tmp.write(preamble + exe.code)
            tmp.flush()
            tmp.close()

            outputs_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            )
            outputs_file.close()

            env = os.environ.copy()
            env.update(self.node_env)
            env["__ygg_inputs__"] = json.dumps(exe.args)
            env["__ygg_outputs_file__"] = outputs_file.name

            preexec_fn = _make_preexec_fn(exe.max_memory_mb)

            kwargs: dict[str, Any] = dict(
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=exe.timeout,
                env=env,
            )
            if preexec_fn is not None:
                kwargs["preexec_fn"] = preexec_fn

            proc = subprocess.run(
                [self.python_bin, tmp.name],
                **kwargs,
            )

            duration = round(time.monotonic() - t0, 3)
            status = "completed" if proc.returncode == 0 else "failed"

            result: Any = None
            outputs_path = Path(outputs_file.name)
            if outputs_path.exists() and outputs_path.stat().st_size > 0:
                try:
                    with open(outputs_path) as f:
                        result = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

            return PyFunctionExecution(
                status=status,
                returncode=proc.returncode,
                stdout=proc.stdout or None,
                stderr=proc.stderr or None,
                duration=duration,
                result=result,
            )

        except subprocess.TimeoutExpired:
            duration = round(time.monotonic() - t0, 3)
            return PyFunctionExecution(
                status="failed",
                duration=duration,
                stderr=f"Timed out after {exe.timeout:.0f}s",
            )

        except Exception as exc:
            duration = round(time.monotonic() - t0, 3)
            return PyFunctionExecution(
                status="failed",
                duration=duration,
                stderr=str(exc),
            )

        finally:
            if tmp is not None:
                Path(tmp.name).unlink(missing_ok=True)
            if outputs_file is not None:
                Path(outputs_file.name).unlink(missing_ok=True)

    # -- ShellCommand -------------------------------------------------------

    def execute_shell_command(self, exe: ShellCommand) -> ShellCommandExecution:
        env = os.environ.copy()
        env.update(self.node_env)
        if exe.env:
            env.update(exe.env)

        t0 = time.monotonic()

        try:
            proc = subprocess.run(
                exe.command,
                cwd=exe.cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                input=exe.stdin,
                timeout=exe.timeout,
            )
        except subprocess.TimeoutExpired:
            duration = round(time.monotonic() - t0, 3)
            return ShellCommandExecution(
                status="failed",
                duration=duration,
                stderr=f"Timed out after {exe.timeout:.0f}s",
            )

        duration = round(time.monotonic() - t0, 3)
        status = "completed" if proc.returncode == 0 else "failed"

        return ShellCommandExecution(
            status=status,
            returncode=proc.returncode,
            stdout=proc.stdout or None,
            stderr=proc.stderr or None,
            duration=duration,
        )
