from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg

if TYPE_CHECKING:
    from .environment import PyEnv

__all__ = ["SystemCommandError", "SystemCommand"]


def _is_windows() -> bool:
    return os.name == "nt"


def _format_cmd(args: Sequence[str]) -> str:
    """
    Format a command for human-readable display.

    Uses shell-like quoting on POSIX. On Windows falls back to a simple join
    because shlex.join() does not reflect cmd.exe quoting rules perfectly.
    """
    parts = [str(a) for a in args]
    if _is_windows():
        return " ".join(parts)
    return shlex.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_EXCEPTION_LINE_RE = re.compile(
    r"^(?P<etype>(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception|Warning|Interrupt|Exit))"
    r":\s*(?P<msg>.*)$",
    re.MULTILINE,
)

_TRACEBACK_HEADER_RE = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)

_INLINE_FRAME_RE = re.compile(
    r'^(?P<indent>\s*)File "<string>", line (?P<lineno>\d+)(?P<rest>.*)$',
    re.MULTILINE,
)

_MOD_NOT_FOUND_QUOTED_RE = re.compile(
    r"ModuleNotFoundError:\s+No module named\s+['\"](?P<name>[^'\"]+)['\"]"
)
_MOD_NOT_FOUND_BARE_RE = re.compile(
    r"\bNo module named\s+(?P<name>[A-Za-z_][\w.]*)\b"
)


@dataclass(slots=True)
class SystemCommand:
    """
    Wrap a subprocess command and its eventual result.

    The command is created around a live ``subprocess.Popen`` object and becomes
    complete after :meth:`wait` populates :attr:`completed`.

    Features
    --------
    - lazy process launching via :meth:`run_lazy`
    - synchronous helper via :meth:`run_sync`
    - Python-aware stderr inspection
    - optional one-shot auto-install retry for missing Python modules
    - richer error formatting through :class:`SystemCommandError`
    """

    args: tuple[str, ...]
    cwd: Path | None
    env: dict[str, str] | None
    popen: subprocess.Popen[str]
    python: Optional["PyEnv"] = None
    installed_python_modules: set[str] | None = field(default=None, init=False, repr=False)
    completed: subprocess.CompletedProcess[str] | None = field(default=None, init=False, repr=False)

    def __getstate__(self) -> dict:
        """
        Serialize observable state only.

        ``Popen`` itself is not picklable, so we preserve only the command
        metadata, completed result, and the latest known return code.
        """
        popen = self.popen
        returncode = getattr(popen, "returncode", None)
        if returncode is None:
            try:
                returncode = popen.poll()
            except Exception:
                pass

        return {
            "args": self.args,
            "cwd": self.cwd,
            "env": self.env,
            "python": self.python,
            "installed_python_modules": self.installed_python_modules,
            "completed": self.completed,
            "_popen_returncode": returncode,
        }

    def __setstate__(self, state: dict) -> None:
        """
        Reconstruct a dead Popen-like stub so read-only state remains usable.
        """
        import types

        stub = types.SimpleNamespace(returncode=state.pop("_popen_returncode", None))
        stub.poll = lambda: stub.returncode  # type: ignore[attr-defined]

        object.__setattr__(self, "args", state["args"])
        object.__setattr__(self, "cwd", state["cwd"])
        object.__setattr__(self, "env", state["env"])
        object.__setattr__(self, "python", state["python"])
        object.__setattr__(self, "installed_python_modules", state["installed_python_modules"])
        object.__setattr__(self, "completed", state["completed"])
        object.__setattr__(self, "popen", stub)

    # ── constructors ──────────────────────────────────────────────────────────

    @staticmethod
    def run_sync(
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run a command synchronously and optionally raise a formatted error.
        """
        proc = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if check and proc.returncode != 0:
            import types

            stub = types.SimpleNamespace(returncode=proc.returncode)
            cmd = SystemCommand(
                args=tuple(map(str, args)),
                cwd=cwd,
                env=env,
                popen=stub,  # type: ignore[arg-type]
            )
            cmd.completed = proc
            raise SystemCommandError(command=cmd)
        return proc

    @staticmethod
    def run_lazy(
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        python: Optional["PyEnv"] = None,
    ) -> "SystemCommand":
        """
        Launch a command and return a lazy wrapper around the live process.
        """
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
            python=python,
        )

    # ── process state ─────────────────────────────────────────────────────────

    def poll(self) -> int | None:
        """Return the current process return code, or ``None`` if still running."""
        return self.popen.poll()

    @property
    def returncode(self) -> int | None:
        """Return the exit status once known."""
        return self.popen.returncode if self.completed is None else self.completed.returncode

    @property
    def stdout(self) -> str | None:
        """Captured stdout, available only after completion."""
        return None if self.completed is None else self.completed.stdout

    @property
    def stderr(self) -> str | None:
        """Captured stderr, available only after completion."""
        return None if self.completed is None else self.completed.stderr

    @property
    def command_str(self) -> str:
        """Human-readable command string."""
        return _format_cmd(self.args)

    # ── stderr analysis ───────────────────────────────────────────────────────

    def find_module_not_found_error(self) -> Optional[ModuleNotFoundError]:
        """
        Best-effort extraction of a ``ModuleNotFoundError`` from stderr.
        """
        err = self.stderr
        if not err:
            return None

        match = _MOD_NOT_FOUND_QUOTED_RE.search(err)
        if match:
            name = match.group("name")
            return ModuleNotFoundError(f"No module named '{name}'", name=name)

        match = _MOD_NOT_FOUND_BARE_RE.search(err)
        if match:
            name = match.group("name")
            return ModuleNotFoundError(f"No module named '{name}'", name=name)

        return None

    def parse_python_exception(self) -> Optional[tuple[str, str]]:
        """
        Extract the last Python exception type and message from stderr.

        Returns
        -------
        tuple[str, str] | None
            ``(exception_type, message)`` for the last detected exception line,
            or ``None`` when stderr does not appear to contain a Python exception.
        """
        err = self.stderr
        if not err:
            return None

        matches = _EXCEPTION_LINE_RE.findall(err)
        if matches:
            etype, msg = matches[-1]
            return etype, msg.strip()

        return None

    def extract_traceback(self) -> Optional[str]:
        """
        Extract the last Python traceback block from stderr.

        If the command used ``python -c <code>``, inline ``<string>`` frames are
        annotated with the actual source line from the ``-c`` payload.
        """
        err = self.stderr
        if not err:
            return None

        starts = [m.start() for m in _TRACEBACK_HEADER_RE.finditer(err)]
        if not starts:
            return None

        tb = err[starts[-1]:].rstrip()

        if _INLINE_FRAME_RE.search(tb):
            tb = self._annotate_inline_frames(tb)

        return tb

    def _annotate_inline_frames(self, tb: str) -> str:
        """
        Expand ``File "<string>", line N`` frames with the matching ``-c`` source line.
        """
        args = list(self.args)
        try:
            c_index = args.index("-c")
            source = args[c_index + 1]
        except (ValueError, IndexError):
            return tb

        source_lines = source.splitlines()

        def _replace_frame(match: re.Match) -> str:
            indent = match.group("indent")
            lineno = int(match.group("lineno"))
            rest = match.group("rest")
            frame = f'{indent}File "<string>", line {lineno}{rest}'
            if 1 <= lineno <= len(source_lines):
                code_line = source_lines[lineno - 1]
                frame += f"\n{indent}  {code_line}"
            return frame

        return _INLINE_FRAME_RE.sub(_replace_frame, tb)

    # ── error presentation ────────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Build a short user-friendly summary of the failure.

        Examples
        --------
        - ``ModuleNotFoundError: No module named 'pyarrow'``
        - ``ValueError: invalid literal for int() with base 10: 'abc'``
        - ``Process exited with status 127``
        """
        py_exc = self.parse_python_exception()
        if py_exc:
            etype, msg = py_exc
            return f"{etype}: {msg}" if msg else etype

        if self.returncode is None:
            return "Process has not completed yet."

        return f"Process exited with status {self.returncode}"

    def render_error_details(self) -> str:
        """
        Render the most useful detailed error payload.

        Preference order:
        1. extracted Python traceback
        2. raw stderr
        3. raw stdout
        """
        tb = self.extract_traceback()
        if tb:
            return tb

        stderr = (self.stderr or "").rstrip()
        if stderr:
            return stderr

        stdout = (self.stdout or "").rstrip()
        if stdout:
            return stdout

        return ""

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def wait(
        self,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True,
        auto_install: bool = False,
    ) -> "SystemCommand":
        """
        Wait for process completion, capture outputs, and optionally raise on failure.

        Always returns ``self``.
        """
        if self.completed is not None:
            if raise_error:
                self.raise_for_status(
                    wait=wait,
                    raise_error=raise_error,
                    auto_install=auto_install,
                )
            return self

        wait_cfg = WaitingConfig.check_arg(wait)

        if wait_cfg:
            try:
                out, err = self.popen.communicate(timeout=wait_cfg.timeout_total_seconds)
            except subprocess.TimeoutExpired as exc:
                try:
                    self.popen.kill()
                    out, err = self.popen.communicate()
                except Exception:
                    out, err = "", ""
                self.completed = subprocess.CompletedProcess(
                    args=list(self.args),
                    returncode=self.popen.returncode if self.popen.returncode is not None else -9,
                    stdout=out,
                    stderr=err,
                )
                raise SystemCommandError(
                    command=self,
                    message=(
                        f"Command timed out after {wait_cfg.timeout_total_seconds} seconds"
                    ),
                ) from exc

            self.completed = subprocess.CompletedProcess(
                args=list(self.args),
                returncode=self.popen.returncode if self.popen.returncode is not None else 0,
                stdout=out,
                stderr=err,
            )

        if raise_error:
            self.raise_for_status(
                wait=wait_cfg,
                raise_error=raise_error,
                auto_install=auto_install,
            )

        return self

    def retry(
        self,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True,
        auto_install: bool = False,
    ) -> "SystemCommand":
        """
        Re-launch the same command, replacing internal process/completed state.
        """
        new_popen = subprocess.Popen(
            list(self.args),
            cwd=str(self.cwd) if self.cwd else None,
            env=self.env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.popen = new_popen
        self.completed = None

        return self.wait(wait=wait, raise_error=raise_error, auto_install=auto_install)

    def _maybe_auto_install_missing_module(
        self,
        wait: WaitingConfigArg | None = True,
    ) -> bool:
        """
        Attempt a one-shot recovery for ``ModuleNotFoundError``.

        Returns
        -------
        bool
            ``True`` if a missing module was installed and the command retried,
            ``False`` otherwise.
        """
        if self.python is None:
            return False

        module_err = self.find_module_not_found_error()
        if not isinstance(module_err, ModuleNotFoundError) or not module_err.name:
            return False

        if self.installed_python_modules is None:
            self.installed_python_modules = set()

        if module_err.name in self.installed_python_modules:
            return False

        # Prefer PyEnv's import/install mapping logic if available.
        try:
            self.python.install(module_err.name)
        except Exception:
            return False

        self.installed_python_modules.add(module_err.name)
        self.retry(wait=wait, raise_error=False, auto_install=False)
        return self.returncode == 0

    def exception(
        self,
        wait: WaitingConfigArg | None = True,
        auto_install: bool = True,
    ) -> Optional["SystemCommandError"]:
        """
        Return a :class:`SystemCommandError` for a failed completed process, or ``None``.
        """
        if self.completed is None:
            self.wait(wait=wait, raise_error=False, auto_install=False)

        if self.returncode in (None, 0):
            return None

        if auto_install and self._maybe_auto_install_missing_module(wait=wait):
            return None

        return SystemCommandError(command=self)

    def raise_for_status(
        self,
        *,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True,
        auto_install: bool = True,
    ) -> "SystemCommand":
        """
        Raise :class:`SystemCommandError` if the command failed.

        Returns ``self`` when successful or when ``raise_error=False``.
        """
        if raise_error:
            error = self.exception(wait=wait, auto_install=auto_install)
            if error is not None:
                raise error
        return self


@dataclass(frozen=True, slots=True)
class SystemCommandError(RuntimeError):
    """
    Raised when a subprocess command fails.

    Attributes
    ----------
    command:
        The failing :class:`SystemCommand`.
    message:
        Optional high-level override message, used for special cases such as
        timeouts. When omitted, a summary is derived from stderr / traceback.
    """

    command: SystemCommand
    message: str | None = None

    def __str__(self) -> str:
        cmd = self.command
        lines: list[str] = []

        headline = self.message or cmd.summary()
        lines.append(headline)

        lines.append(f"Command: {cmd.command_str}")
        if cmd.cwd is not None:
            lines.append(f"Working directory: {cmd.cwd}")
        if cmd.returncode is not None:
            lines.append(f"Exit status: {cmd.returncode}")

        details = cmd.render_error_details()
        if details:
            lines.append("")
            lines.append(details)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()