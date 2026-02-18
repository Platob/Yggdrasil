# yggdrasil.environ.system_command.py
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union, TYPE_CHECKING, Optional

from yggdrasil.pyutils.waiting_config import WaitingConfig, WaitingConfigArg

if TYPE_CHECKING:
    from .environment import PyEnv

__all__ = ["SystemCommandError", "SystemCommand"]


def _is_windows() -> bool:
    return os.name == "nt"

def _format_cmd(args: Sequence[str]) -> str:
    return " ".join(map(str, args))


@dataclass(slots=True)
class SystemCommand:
    """
    Lazy command result:
    - created with a running Popen
    - call wait() to collect stdout/stderr and returncode
    - call raise_on_status() to error if non-zero
    """
    args: tuple[str, ...]
    cwd: Path | None
    env: dict[str, str] | None
    popen: subprocess.Popen[str]
    python: Optional["PyEnv"] = None
    completed: subprocess.CompletedProcess[str] | None = field(default=None, init=False, repr=False)

    @staticmethod
    def run_sync(
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if check and proc.returncode != 0:
            raise SystemCommandError(
                f"Command failed ({proc.returncode}): {_format_cmd(args)}\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}\n"
            )
        return proc

    @staticmethod
    def run_lazy(
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        python: Optional["PyEnv"] = None,
    ) -> SystemCommand:
        popen = subprocess.Popen(
            list(args),
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return SystemCommand(
            args=tuple(map(str, args)),
            cwd=cwd,
            env=env,
            popen=popen,
            python=python
        )

    def poll(self) -> int | None:
        return self.popen.poll()

    @property
    def returncode(self) -> int | None:
        return self.popen.returncode if self.completed is None else self.completed.returncode

    @property
    def stdout(self) -> str | None:
        return None if self.completed is None else self.completed.stdout

    @property
    def stderr(self) -> str | None:
        return None if self.completed is None else self.completed.stderr

    def wait(
        self,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True
    ) -> Union["SystemCommand", "SystemCommandError"]:
        if self.completed is not None:
            return self.completed  # type: ignore[return-value]

        wait = WaitingConfig.check_arg(wait)

        if wait.timeout:
            out, err = self.popen.communicate(timeout=wait.timeout_total_seconds)

            self.completed = subprocess.CompletedProcess(
                args=list(self.args),
                returncode=self.popen.returncode or 0,
                stdout=out,
                stderr=err,
            )

            return self.raise_for_status(
                wait=wait,
                raise_error=raise_error
            )
        return self

    def find_module_not_found_error(self) -> Optional[ModuleNotFoundError]:
        """
        Best-effort extraction of a ModuleNotFoundError from captured stderr.

        Supports common stderr shapes:
          - Standard Python traceback line:
              ModuleNotFoundError: No module named 'foo'
          - With module path:
              ModuleNotFoundError: No module named 'foo.bar'
          - Alternative phrasing (rare but seen):
              No module named foo

        Returns:
          - ModuleNotFoundError(name=<module>) when detected
          - None otherwise
        """
        err = self.stderr
        if not err:
            return None

        # Most reliable: the actual exception line in a Python traceback
        # Example: "ModuleNotFoundError: No module named 'requests'"
        m = re.search(
            r"ModuleNotFoundError:\s+No module named\s+['\"](?P<name>[^'\"]+)['\"]",
            err,
        )
        if m:
            name = m.group("name")
            return ModuleNotFoundError(f"No module named '{name}'", name=name)

        # Slightly weaker: unquoted variant sometimes appears in logs
        m = re.search(
            r"\bNo module named\s+(?P<name>[A-Za-z_][\w\.]*)\b",
            err,
        )
        if m:
            name = m.group("name")
            return ModuleNotFoundError(f"No module named '{name}'", name=name)

        return None

    def retry(
        self,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True,
    ) -> Union["SystemCommand", "SystemCommandError"]:
        """Re-launch the same command, replacing internal popen/completed state."""
        new_popen = subprocess.Popen(
            list(self.args),
            cwd=str(self.cwd) if self.cwd else None,
            env=self.env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # dataclass(slots=True) but not frozen — normal attribute assignment is fine.
        self.popen = new_popen
        self.completed = None
        return self.wait(wait=wait, raise_error=raise_error)

    def raise_for_status(
        self,
        *,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True,
        install_python_modules: bool = True,
    ) -> Union["SystemCommand", "SystemCommandError"]:
        if self.returncode != 0:
            module_err = self.find_module_not_found_error()

            if install_python_modules and self.python and isinstance(module_err, ModuleNotFoundError):
                # Ask the bound PyEnv to pip-install the missing package, then retry once.
                self.python.install(module_err.name)
                return self.retry(wait=wait, raise_error=raise_error)

            e = SystemCommandError(command=self)

            if raise_error:
                raise e
            return e

        return self


@dataclass(frozen=True, slots=True)
class SystemCommandError(RuntimeError):
    """Raised when a subprocess command fails."""
    command: SystemCommand

    def __str__(self):
        cp = self.command

        return cp.stderr

    def __repr__(self):
        cp = self.command

        return cp.stderr