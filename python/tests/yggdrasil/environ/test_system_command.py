# test_system_command.py
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

from yggdrasil.environ.system_command import SystemCommand, SystemCommandError
from yggdrasil.pyutils.waiting_config import WaitingConfig


def _make_wait_cfg(timeout_seconds: float) -> WaitingConfig:
    """
    Build a WaitingConfig instance without guessing its constructor too hard.
    We introspect __init__ params and fill what exists.

    This keeps the tests resilient if WaitingConfig evolves.
    """
    sig = inspect.signature(WaitingConfig)
    kwargs = {}
    for name, p in sig.parameters.items():
        if name in ("self",):
            continue

        # Common patterns seen in "waiting config" classes
        if name in ("timeout_total_seconds", "timeout_seconds", "timeout_s", "timeout"):
            # Some libs use timeout as bool; some as seconds.
            # If it's annotated bool, set True and also try to set seconds via another field.
            ann = p.annotation
            if ann is bool:
                kwargs[name] = True
            else:
                kwargs[name] = timeout_seconds

        elif name in ("total_seconds", "seconds"):
            kwargs[name] = timeout_seconds

        # Keep defaults for everything else (poll interval, jitter, etc.)
    return WaitingConfig(**kwargs)  # type: ignore[arg-type]


def _py(*code_lines: str) -> list[str]:
    """
    Cross-platform command that runs Python code.
    """
    code = "\n".join(code_lines)
    return [sys.executable, "-c", code]


def test_run_sync_success_captures_stdout_and_stderr():
    proc = SystemCommand.run_sync(
        _py(
            "print('hello')",
            "import sys; sys.stderr.write('warn\\n')",
        ),
        check=True,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "hello"
    assert proc.stderr.strip() == "warn"


def test_run_sync_failure_raises_system_command_error():
    with pytest.raises(SystemCommandError):
        SystemCommand.run_sync(
            _py(
                "import sys",
                "print('oops')",
                "sys.stderr.write('bad\\n')",
                "sys.exit(5)",
            ),
            check=True,
        )


def test_run_sync_failure_no_check_returns_completed_process():
    proc = SystemCommand.run_sync(
        _py(
            "import sys",
            "print('oops')",
            "sys.stderr.write('bad\\n')",
            "sys.exit(7)",
        ),
        check=False,
    )
    assert proc.returncode == 7
    assert proc.stdout.strip() == "oops"
    assert proc.stderr.strip() == "bad"


def test_run_sync_respects_cwd(tmp_path: Path):
    proc = SystemCommand.run_sync(
        _py(
            "import os",
            "print(os.getcwd())",
        ),
        cwd=tmp_path,
        check=True,
    )
    # normalize for platforms that may resolve symlinks etc.
    assert Path(proc.stdout.strip()).resolve() == tmp_path.resolve()


def test_run_sync_passes_env(tmp_path: Path):
    env = dict(os.environ)
    env["YGG_TEST_X"] = "123"

    proc = SystemCommand.run_sync(
        _py(
            "import os",
            "print(os.environ.get('YGG_TEST_X', 'MISSING'))",
        ),
        cwd=tmp_path,
        env=env,
        check=True,
    )
    assert proc.stdout.strip() == "123"


def test_run_lazy_wait_collects_output_and_sets_completed():
    cmd = SystemCommand.run_lazy(
        _py(
            "print('lazy-out')",
            "import sys; sys.stderr.write('lazy-err\\n')",
        )
    )

    # Make sure we actually collect stdout/stderr via wait with a real WaitingConfig.
    wait_cfg = _make_wait_cfg(2.0)
    res = cmd.wait(wait=wait_cfg, raise_error=True)

    # On success, implementation returns self (SystemCommand)
    assert res is cmd
    assert cmd.completed is not None
    assert cmd.returncode == 0
    assert cmd.stdout is not None and cmd.stdout.strip() == "lazy-out"
    assert cmd.stderr is not None and cmd.stderr.strip() == "lazy-err"


def test_run_lazy_wait_failure_returns_error_when_raise_error_false():
    cmd = SystemCommand.run_lazy(
        _py(
            "import sys",
            "print('nope')",
            "sys.stderr.write('kaboom\\n')",
            "sys.exit(9)",
        )
    )
    wait_cfg = _make_wait_cfg(2.0)
    res = cmd.wait(wait=wait_cfg, raise_error=False)

    assert isinstance(res, SystemCommandError)
    assert res.command is cmd
    assert cmd.completed is not None
    assert cmd.returncode == 9
    assert (cmd.stdout or "").strip() == "nope"
    assert (cmd.stderr or "").strip() == "kaboom"


def test_raise_for_status_raises_after_wait():
    cmd = SystemCommand.run_lazy(
        _py(
            "import sys",
            "sys.stderr.write('boom\\n')",
            "sys.exit(3)",
        )
    )
    wait_cfg = _make_wait_cfg(2.0)
    cmd.wait(wait=wait_cfg, raise_error=False)

    with pytest.raises(SystemCommandError):
        cmd.raise_for_status(raise_error=True)


def test_raise_for_status_returns_error_when_raise_error_false():
    cmd = SystemCommand.run_lazy(
        _py(
            "import sys",
            "sys.exit(4)",
        )
    )
    wait_cfg = _make_wait_cfg(2.0)
    cmd.wait(wait=wait_cfg, raise_error=False)

    err = cmd.raise_for_status(raise_error=False)
    assert isinstance(err, SystemCommandError)
    assert err.command is cmd


def test_system_command_error_str_and_repr_include_streams():
    cmd = SystemCommand.run_lazy(
        _py(
            "import sys",
            "print('OUT')",
            "sys.stderr.write('ERR\\n')",
            "sys.exit(2)",
        )
    )
    wait_cfg = _make_wait_cfg(2.0)
    err = cmd.wait(wait=wait_cfg, raise_error=False)
    assert isinstance(err, SystemCommandError)

    s = str(err)
    r = repr(err)
    assert "ERR" in s
    assert "ERR" in r
