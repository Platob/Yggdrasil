from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from pathlib import Path

from ..execution import Executable, Execution
from ..execution.pyfunction import PyFunction, PyFunctionExecution
from ..execution.shell import ShellCommand, ShellCommandExecution
from .py import PyEnvironment
from ._service import EnvironmentService

_IS_WINDOWS = platform.system() == "Windows"


def venv_python(venv_path: str | Path) -> str:
    """Return the python binary inside a virtualenv (cross-platform)."""
    venv_path = Path(venv_path)
    if _IS_WINDOWS:
        return str(venv_path / "Scripts" / "python.exe")
    return str(venv_path / "bin" / "python")


class Environment(ABC):
    """Abstract execution environment.

    Subclasses implement the concrete strategy for each Executable type.
    The generic ``execute`` dispatches to the right handler.
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


__all__ = [
    "Environment",
    "EnvironmentService",
    "PyEnvironment",
    "venv_python",
]
