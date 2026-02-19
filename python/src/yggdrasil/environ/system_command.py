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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


# Matches the canonical Python exception line at the start of a line:
#   ValueError: some message
#   pkg.mod.MyError: some message
_EXCEPTION_LINE_RE = re.compile(
    r"^(?P<etype>(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception|Warning|Interrupt|Exit))"
    r":\s*(?P<msg>.*)$",
    re.MULTILINE,
)

# Matches Python traceback header so we can extract the full block
_TRACEBACK_HEADER_RE = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)

# Matches a traceback frame that came from inline -c code:
#   File "<string>", line 3, in <module>
_INLINE_FRAME_RE = re.compile(
    r'^(?P<indent>\s*)File "<string>", line (?P<lineno>\d+)(?P<rest>.*)$',
    re.MULTILINE,
)

# ModuleNotFoundError patterns
_MOD_NOT_FOUND_QUOTED_RE = re.compile(
    r"ModuleNotFoundError:\s+No module named\s+['\"](?P<name>[^'\"]+)['\"]"
)
_MOD_NOT_FOUND_BARE_RE = re.compile(
    r"\bNo module named\s+(?P<name>[A-Za-z_][\w.]*)\b"
)


# ─────────────────────────────────────────────────────────────────────────────
# SystemCommand
# ─────────────────────────────────────────────────────────────────────────────

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
    installed_modules: set[str] | None = field(default=None, init=False, repr=False)
    completed: subprocess.CompletedProcess[str] | None = field(default=None, init=False, repr=False)

    def __getstate__(self) -> dict:
        # Popen is never picklable — snapshot only the observable state.
        # If the process hasn't been waited on yet, poll() to grab returncode.
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
            "installed_modules": self.installed_modules,
            "completed": self.completed,
            "_popen_returncode": returncode,  # preserved for .returncode property
        }

    def __setstate__(self, state: dict) -> None:
        import types

        # Reconstruct a dead stub so .returncode / .poll() still work.
        stub = types.SimpleNamespace(returncode=state.pop("_popen_returncode", None))
        stub.poll = lambda: stub.returncode  # type: ignore[attr-defined]

        object.__setattr__(self, "args", state["args"])
        object.__setattr__(self, "cwd", state["cwd"])
        object.__setattr__(self, "env", state["env"])
        object.__setattr__(self, "python", state["python"])
        object.__setattr__(self, "installed_modules", state["installed_modules"])
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
        proc = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if check and proc.returncode != 0:
            # Build a proper SystemCommand so SystemCommandError.__str__ works.
            # Popen is already finished; we wrap proc in a minimal stub.
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

    # ── stderr analysis ───────────────────────────────────────────────────────

    def find_module_not_found_error(self) -> Optional[ModuleNotFoundError]:
        """
        Best-effort extraction of a ModuleNotFoundError from captured stderr.

        Supports common stderr shapes:
          - Standard Python traceback line:
              ModuleNotFoundError: No module named 'foo'
          - With dotted module path:
              ModuleNotFoundError: No module named 'foo.bar'
          - Alternative bare phrasing (rare but seen in some tools):
              No module named foo

        Returns:
          ModuleNotFoundError(name=<module>) when detected, None otherwise.
        """
        err = self.stderr
        if not err:
            return None

        m = _MOD_NOT_FOUND_QUOTED_RE.search(err)
        if m:
            name = m.group("name")
            return ModuleNotFoundError(f"No module named '{name}'", name=name)

        m = _MOD_NOT_FOUND_BARE_RE.search(err)
        if m:
            name = m.group("name")
            return ModuleNotFoundError(f"No module named '{name}'", name=name)

        return None

    def parse_python_exception(self) -> Optional[tuple[str, str]]:
        """
        Best-effort extraction of the terminal Python exception type and message
        from captured stderr, ignoring the traceback preamble.

        Handles:
          - Standard:    ValueError: invalid literal for int() with base 10
          - Namespaced:  pkg.errors.MyError: something went wrong
          - Chained:     returns the *last* exception in the chain
          - Multi-line:  captures only the header line of the message

        Returns:
          (exception_type, message) for the last raised exception, or None.
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
        Extract the last Python traceback block from stderr, including
        the exception line.  Returns the raw block as a string, or None
        if no traceback is present.

        When the command was run with ``python -c <code>``, frames that point
        to ``File "<string>", line N`` are replaced with the actual source
        line from the inline code so the error is immediately readable.
        """
        err = self.stderr
        if not err:
            return None

        starts = [m.start() for m in _TRACEBACK_HEADER_RE.finditer(err)]
        if not starts:
            return None

        tb = err[starts[-1]:].rstrip()

        # Check if any frame references inline code
        if _INLINE_FRAME_RE.search(tb):
            tb = self._annotate_inline_frames(tb)

        return tb

    def _annotate_inline_frames(self, tb: str) -> str:
        """
        Replace ``File "<string>", line N`` traceback frames with:

            File "<string>", line N
              <source line from the -c argument>
              ^  (caret pointing at the token, when derivable)

        This mirrors what Python itself prints for file-based tracebacks.
        The inline source is extracted from the ``-c`` argument in self.args.
        If no ``-c`` argument is found the traceback is returned unchanged.
        """
        # Extract the code passed via -c
        args = list(self.args)
        try:
            c_index = args.index("-c")
            source = args[c_index + 1]
        except (ValueError, IndexError):
            return tb

        source_lines = source.splitlines()

        def _replace_frame(m: re.Match) -> str:
            indent = m.group("indent")
            lineno = int(m.group("lineno"))
            rest = m.group("rest")
            frame = f'{indent}File "<string>", line {lineno}{rest}'
            if 1 <= lineno <= len(source_lines):
                code_line = source_lines[lineno - 1]
                frame += f"\n{indent}  {code_line}"
            return frame

        return _INLINE_FRAME_RE.sub(_replace_frame, tb)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def wait(
        self,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True,
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

            return self.raise_for_status(wait=wait, raise_error=raise_error)

        return self

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

            if install_python_modules and self.python is not None and isinstance(module_err, ModuleNotFoundError):
                if self.installed_modules is None:
                    self.installed_modules = set()

                if module_err.name in self.installed_modules:
                    raise module_err

                # Ask the bound PyEnv to pip-install the missing package, then retry once.
                self.python.install(module_err.name)

                self.installed_modules.add(module_err.name)

                return self.retry(wait=wait, raise_error=raise_error)

            e = SystemCommandError(command=self)

            if raise_error:
                raise e
            return e

        return self


# ─────────────────────────────────────────────────────────────────────────────
# SystemCommandError
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SystemCommandError(RuntimeError):
    """Raised when a subprocess command fails."""

    command: SystemCommand

    def __str__(self) -> str:
        cmd = self.command
        lines: list[str] = []

        # ── stdout: emitted as-is, no decoration ─────────────────────────────
        stdout = (cmd.stdout or "").rstrip()
        if stdout:
            lines.append(stdout)

        # ── stderr: prefer the extracted traceback block so the output looks
        #    identical to what Python itself would have printed.
        #    Fall back to raw stderr for non-Python processes (Rust, shell, …).
        tb = cmd.extract_traceback()
        if tb:
            lines.append(tb)
        else:
            stderr = (cmd.stderr or "").rstrip()
            if stderr:
                lines.append(stderr)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()