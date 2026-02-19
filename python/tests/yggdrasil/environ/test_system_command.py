"""
Integration tests for SystemCommand / SystemCommandError.

No mocking — every test spawns a real subprocess via sys.executable.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from yggdrasil.environ.system_command import SystemCommand, SystemCommandError

PY = sys.executable  # current interpreter, always available


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(code: str, *, check: bool = True) -> SystemCommand:
    """Run inline Python code via run_lazy().wait() and return the SystemCommand."""
    cmd = SystemCommand.run_lazy([PY, "-c", textwrap.dedent(code)])
    return cmd.wait(raise_error=check)


def run_sync(code: str, *, check: bool = True):
    """Run inline Python code via run_sync() and return CompletedProcess."""
    return SystemCommand.run_sync([PY, "-c", textwrap.dedent(code)], check=check)


# ─────────────────────────────────────────────────────────────────────────────
# run_sync — basic contract
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSync:
    def test_success_returncode(self):
        proc = run_sync("pass")
        assert proc.returncode == 0

    def test_stdout_captured(self):
        proc = run_sync("print('hello')")
        assert proc.stdout.strip() == "hello"

    def test_stderr_captured(self):
        proc = run_sync("import sys; sys.stderr.write('err\\n')")
        assert "err" in proc.stderr

    def test_failure_raises_system_command_error(self):
        with pytest.raises(SystemCommandError):
            run_sync("raise ValueError('boom')")

    def test_failure_error_message_contains_traceback(self):
        with pytest.raises(SystemCommandError) as exc_info:
            run_sync("raise ValueError('boom')")
        s = str(exc_info.value)
        assert "Traceback (most recent call last):" in s
        assert "ValueError: boom" in s

    def test_failure_no_raise_when_check_false(self):
        proc = run_sync("import sys; sys.exit(3)", check=False)
        assert proc.returncode == 3

    def test_cwd_passed_to_process(self, tmp_path):
        proc = run_sync("import os; print(os.getcwd())")
        # Just verifies it doesn't crash; cwd defaults to caller's cwd
        assert proc.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# run_lazy / wait — basic contract
# ─────────────────────────────────────────────────────────────────────────────

class TestRunLazy:
    def test_returns_system_command(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "pass"])
        assert isinstance(cmd, SystemCommand)
        cmd.wait()

    def test_completed_none_before_wait(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "pass"])
        assert cmd.completed is None
        cmd.wait()
        assert cmd.completed is not None

    def test_stdout_available_after_wait(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "print('lazy')"])
        cmd.wait()
        assert "lazy" in cmd.stdout

    def test_stderr_available_after_wait(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "import sys; sys.stderr.write('err\\n')"])
        cmd.wait(raise_error=False)
        assert "err" in cmd.stderr

    def test_returncode_zero_on_success(self):
        cmd = run("pass")
        assert cmd.returncode == 0

    def test_returncode_nonzero_on_failure(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "import sys; sys.exit(42)"])
        cmd.wait(raise_error=False)
        assert cmd.returncode == 42

    def test_raise_error_true_raises_on_failure(self):
        with pytest.raises(SystemCommandError):
            run("raise RuntimeError('fail')")

    def test_raise_error_false_returns_error(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "raise RuntimeError('fail')"])
        result = cmd.wait(raise_error=False)
        assert isinstance(result, SystemCommandError)
        assert cmd.returncode != 0  # error state accessible on the original cmd

    def test_wait_idempotent(self):
        """Calling wait() twice should not crash and returns completed state."""
        cmd = run("pass")
        first = cmd.completed
        cmd.wait()
        assert cmd.completed is first  # same object, not re-run

    def test_poll_returns_none_before_exit(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "import time; time.sleep(30)"])
        polled = cmd.poll()
        cmd.popen.kill()
        assert polled is None


# ─────────────────────────────────────────────────────────────────────────────
# SystemCommandError — __str__ output
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorStr:
    def _err(self, code: str) -> SystemCommandError:
        cmd = SystemCommand.run_lazy([PY, "-c", textwrap.dedent(code)])
        cmd.wait(raise_error=False)
        return SystemCommandError(command=cmd)

    def test_contains_traceback_header(self):
        s = str(self._err("raise ValueError('x')"))
        assert "Traceback (most recent call last):" in s

    def test_contains_exception_type_and_message(self):
        s = str(self._err("raise ValueError('specific_msg')"))
        assert "ValueError: specific_msg" in s

    def test_no_decoration(self):
        """No rulers, exit codes, cmd: prefixes — just raw output."""
        s = str(self._err("raise ValueError('x')"))
        assert "exit" not in s
        assert "cmd :" not in s
        assert "↳" not in s

    def test_stdout_before_traceback(self):
        code = """
            import sys
            print('build output')
            raise RuntimeError('after stdout')
        """
        s = str(self._err(code))
        assert "build output" in s
        assert s.index("build output") < s.index("Traceback")

    def test_no_stdout_section_when_clean(self):
        s = str(self._err("raise ValueError('x')"))
        # Only traceback — no extra stdout noise
        assert s.strip().startswith("Traceback")

    def test_empty_string_when_no_output(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "import sys; sys.exit(1)"])
        cmd.wait(raise_error=False)
        s = str(SystemCommandError(command=cmd))
        assert s == ""

    def test_non_python_process_raw_stderr(self):
        """Non-Python failure (ls / dir) emits raw stderr, no traceback."""
        import platform
        not_found = "/no_such_path_xyz_integration_test"
        if platform.system() == "Windows":
            args = ["cmd", "/c", f"dir {not_found}"]
        else:
            args = ["ls", not_found]

        cmd = SystemCommand.run_lazy(args)
        cmd.wait(raise_error=False)
        s = str(SystemCommandError(command=cmd))
        assert s  # something present
        assert "Traceback" not in s

    def test_repr_equals_str(self):
        e = self._err("raise TypeError('t')")
        assert repr(e) == str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Inline frame annotation (-c code)
# ─────────────────────────────────────────────────────────────────────────────

class TestInlineFrameAnnotation:
    def _tb(self, code: str) -> str:
        cmd = SystemCommand.run_lazy([PY, "-c", textwrap.dedent(code)])
        cmd.wait(raise_error=False)
        tb = cmd.extract_traceback()
        assert tb is not None, "No traceback found in stderr"
        return tb

    def test_single_line_source_injected(self):
        tb = self._tb("raise ValueError('oops')")
        assert "raise ValueError" in tb
        assert 'File "<string>", line 1' in tb

    def test_multiline_correct_line_shown(self):
        code = """
            x = 1
            y = 2
            raise RuntimeError('line 3')
        """
        tb = self._tb(code)
        assert "raise RuntimeError" in tb
        # lines before the raise should NOT appear
        assert "x = 1" not in tb

    def test_deep_call_stack_all_frames_annotated(self):
        code = """
            def a():
                raise TypeError('deep')
            def b():
                a()
            def c():
                b()
            c()
        """
        tb = self._tb(code)
        # All three <string> frames should carry injected source
        assert "raise TypeError" in tb
        assert "a()" in tb
        assert "b()" in tb

    def test_chained_exception_last_frame_annotated(self):
        code = """
            try:
                raise ValueError('inner')
            except ValueError:
                raise RuntimeError('outer')
        """
        tb = self._tb(code)
        # Last traceback block — RuntimeError frame annotated
        assert "raise RuntimeError" in tb
        assert "RuntimeError: outer" in tb

    def test_file_string_replaced_with_source(self):
        """The raw '<string>' frame line is still present but followed by source."""
        tb = self._tb("raise OSError('io')")
        lines = tb.splitlines()
        frame_idx = next(
            i for i, l in enumerate(lines) if 'File "<string>"' in l
        )
        # The line immediately after the frame header should be the source
        assert frame_idx + 1 < len(lines)
        assert "raise OSError" in lines[frame_idx + 1]


# ─────────────────────────────────────────────────────────────────────────────
# find_module_not_found_error
# ─────────────────────────────────────────────────────────────────────────────

class TestFindModuleNotFoundError:
    def _cmd(self, module: str) -> SystemCommand:
        cmd = SystemCommand.run_lazy([PY, "-c", f"import {module}"])
        cmd.wait(raise_error=False)
        return cmd

    def test_detects_missing_module(self):
        exc = self._cmd("totally_missing_pkg_xyz").find_module_not_found_error()
        assert isinstance(exc, ModuleNotFoundError)
        assert exc.name == "totally_missing_pkg_xyz"

    def test_returns_none_for_successful_import(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "import sys"])
        cmd.wait(raise_error=False)
        assert cmd.find_module_not_found_error() is None

    def test_returns_none_for_other_errors(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "raise ValueError('not a module error')"])
        cmd.wait(raise_error=False)
        assert cmd.find_module_not_found_error() is None

    def test_dotted_module_name_preserved(self):
        exc = self._cmd("missing_pkg_xyz.sub.module").find_module_not_found_error()
        assert exc is not None
        # Python reports the top-level name that's missing
        assert "missing_pkg_xyz" in exc.name


# ─────────────────────────────────────────────────────────────────────────────
# parse_python_exception
# ─────────────────────────────────────────────────────────────────────────────

class TestParsePythonException:
    def _cmd(self, code: str) -> SystemCommand:
        cmd = SystemCommand.run_lazy([PY, "-c", textwrap.dedent(code)])
        cmd.wait(raise_error=False)
        return cmd

    def test_value_error(self):
        etype, msg = self._cmd("raise ValueError('bad value')").parse_python_exception()
        assert etype == "ValueError"
        assert "bad value" in msg

    def test_runtime_error(self):
        etype, msg = self._cmd("raise RuntimeError('boom')").parse_python_exception()
        assert etype == "RuntimeError"
        assert "boom" in msg

    def test_type_error(self):
        etype, _ = self._cmd("1 + 'a'").parse_python_exception()
        assert etype == "TypeError"

    def test_os_error_with_errno(self):
        etype, msg = self._cmd("open('/no/such/file/xyz')").parse_python_exception()
        assert etype in ("FileNotFoundError", "OSError")
        assert msg

    def test_chained_returns_last(self):
        code = """
            try:
                raise ValueError('inner')
            except ValueError:
                raise RuntimeError('outer')
        """
        etype, msg = self._cmd(code).parse_python_exception()
        assert etype == "RuntimeError"
        assert "outer" in msg

    def test_none_when_process_succeeds(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "pass"])
        cmd.wait(raise_error=False)
        assert cmd.parse_python_exception() is None

    def test_none_when_clean_exit_nonzero(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "import sys; sys.exit(1)"])
        cmd.wait(raise_error=False)
        assert cmd.parse_python_exception() is None


# ─────────────────────────────────────────────────────────────────────────────
# extract_traceback
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTraceback:
    def _cmd(self, code: str) -> SystemCommand:
        cmd = SystemCommand.run_lazy([PY, "-c", textwrap.dedent(code)])
        cmd.wait(raise_error=False)
        return cmd

    def test_starts_with_traceback_header(self):
        tb = self._cmd("raise ValueError('x')").extract_traceback()
        assert tb.startswith("Traceback (most recent call last):")

    def test_ends_with_exception_line(self):
        tb = self._cmd("raise ValueError('end')").extract_traceback()
        assert tb.rstrip().endswith("ValueError: end")

    def test_none_when_no_traceback(self):
        cmd = self._cmd("import sys; sys.exit(1)")
        assert cmd.extract_traceback() is None

    def test_none_on_success(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "pass"])
        cmd.wait(raise_error=False)
        assert cmd.extract_traceback() is None

    def test_chained_returns_last_block(self):
        code = """
            try:
                raise ValueError('inner')
            except ValueError:
                raise RuntimeError('outer')
        """
        tb = self._cmd(code).extract_traceback()
        assert "RuntimeError: outer" in tb
        # The first traceback block for ValueError should not be in the extracted part
        assert "ValueError: inner" not in tb

    def test_deep_traceback_preserved(self):
        code = """
            def a(): raise TypeError('deep')
            def b(): a()
            def c(): b()
            c()
        """
        tb = self._cmd(code).extract_traceback()
        assert "TypeError: deep" in tb
        assert tb.count('File "<string>"') >= 3


# ─────────────────────────────────────────────────────────────────────────────
# retry
# ─────────────────────────────────────────────────────────────────────────────

class TestRetry:
    def test_retry_reruns_command(self):
        """retry() should re-execute the same command."""
        cmd = SystemCommand.run_lazy([PY, "-c", "pass"])
        cmd.wait(raise_error=False)
        first_pid = cmd.popen.pid

        cmd.retry(raise_error=False)
        assert cmd.popen.pid != first_pid  # new process spawned

    def test_retry_resets_completed(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "pass"])
        cmd.wait(raise_error=False)
        assert cmd.completed is not None

        # completed is reset and then re-populated by wait() inside retry
        cmd.retry(raise_error=False)
        assert cmd.completed is not None
        assert cmd.returncode == 0

    def test_retry_raises_on_persistent_failure(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "raise ValueError('always fails')"])
        cmd.wait(raise_error=False)
        with pytest.raises(SystemCommandError):
            cmd.retry(raise_error=True)


# ─────────────────────────────────────────────────────────────────────────────
# raise_for_status
# ─────────────────────────────────────────────────────────────────────────────

class TestRaiseForStatus:
    def test_returns_self_on_zero(self):
        cmd = run("pass")
        assert cmd.raise_for_status() is cmd

    def test_raises_on_nonzero(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "raise RuntimeError('bad')"])
        cmd.wait(raise_error=False)
        with pytest.raises(SystemCommandError):
            cmd.raise_for_status()

    def test_returns_error_when_raise_false(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "raise RuntimeError('bad')"])
        cmd.wait(raise_error=False)
        result = cmd.raise_for_status(raise_error=False)
        assert isinstance(result, SystemCommandError)

    def test_error_str_contains_traceback(self):
        cmd = SystemCommand.run_lazy([PY, "-c", "raise KeyError('missing')"])
        cmd.wait(raise_error=False)
        err = cmd.raise_for_status(raise_error=False)
        assert "KeyError" in str(err)
        assert "Traceback" in str(err)