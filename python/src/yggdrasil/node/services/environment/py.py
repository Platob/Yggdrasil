from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ..execution.pyfunction import PyFunction, PyFunctionExecution
from ..execution.shell import ShellCommand, ShellCommandExecution


def _make_preexec_fn(max_memory_mb: int | None):
    if max_memory_mb is None or platform.system() != "Linux":
        return None

    import resource as resource_mod

    memory_bytes = max_memory_mb * 1024 * 1024

    def _set_limits():
        resource_mod.setrlimit(resource_mod.RLIMIT_AS, (memory_bytes, memory_bytes))

    return _set_limits


class PyEnvironment:
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
        from . import venv_python
        return cls(python_bin=venv_python(venv_path), node_env=node_env)

    # -- Executable dispatch ------------------------------------------------

    def execute(self, exe):
        if isinstance(exe, PyFunction):
            return self.execute_pyfunction(exe)
        if isinstance(exe, ShellCommand):
            return self.execute_shell_command(exe)
        raise TypeError(f"Unsupported executable: {type(exe).__name__}")

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
