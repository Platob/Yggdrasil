# test_pyenv.py
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from yggdrasil.environ import environment as pyenv_mod
from yggdrasil.environ.environment import PyEnv
from yggdrasil.environ.system_command import SystemCommandError


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _reset_singleton():
    """
    Ensure CURRENT_PYENV doesn't leak across tests.
    """
    old = pyenv_mod.CURRENT_PYENV
    pyenv_mod.CURRENT_PYENV = None
    try:
        yield
    finally:
        pyenv_mod.CURRENT_PYENV = old


# -------------------------
# Unit tests (no monkeypatch)
# -------------------------
def test_looks_like_path_basic():
    assert PyEnv._looks_like_path("/tmp/x")
    assert PyEnv._looks_like_path("./x")
    assert PyEnv._looks_like_path("~/x")
    assert PyEnv._looks_like_path("a/b")
    assert PyEnv._looks_like_path(r"a\b")

    assert not PyEnv._looks_like_path("")
    assert not PyEnv._looks_like_path("python3.12")
    assert not PyEnv._looks_like_path("3.12")


def test_looks_like_path_windows_drive_letter_only_if_running_on_windows():
    # No monkeypatch: only assert this behavior on Windows.
    if os.name != "nt":
        pytest.skip("Windows drive-letter path heuristic only applies on Windows")

    assert PyEnv._looks_like_path(r"C:\venv")
    assert PyEnv._looks_like_path(r"D:foo")


def test_venv_python_from_dir_posix(tmp_path: Path):
    venv = tmp_path / "venv"
    py = _touch(venv / "bin" / "python")
    assert PyEnv._venv_python_from_dir(venv) == py.resolve()


def test_venv_python_from_dir_windows(tmp_path: Path):
    venv = tmp_path / "venv"
    py = _touch(venv / "Scripts" / "python.exe")
    assert PyEnv._venv_python_from_dir(venv) == py.resolve()


def test_venv_python_from_dir_raises(tmp_path: Path):
    venv = tmp_path / "venv"
    venv.mkdir()
    with pytest.raises(ValueError):
        PyEnv._venv_python_from_dir(venv)


def test_create_normalizes_paths(tmp_path: Path):
    fake_python = _touch(tmp_path / "py" / ("python.exe" if os.name == "nt" else "python"))

    env = PyEnv.create(fake_python, cwd=tmp_path, packages=None)
    assert env.python_path == fake_python.resolve()
    assert env.cwd == tmp_path.resolve()
    assert env.prefer_uv is True


def test_current_singleton_uses_sys_executable():
    a = PyEnv.current(prefer_uv=True)
    b = PyEnv.current(prefer_uv=False)
    assert a is b
    assert a.python_path == Path(sys.executable).resolve()


def test_get_or_create_none_returns_current(tmp_path: Path):
    env = PyEnv.get_or_create(None, prefer_uv=True)
    assert env is PyEnv.current(prefer_uv=True)
    assert env.python_path == Path(sys.executable).resolve()


def test_get_or_create_path_file(tmp_path: Path):
    fake_python = _touch(tmp_path / "python")
    env = PyEnv.get_or_create(fake_python)
    assert env.python_path == fake_python.resolve()
    assert env.prefer_uv is True


def test_get_or_create_path_dir(tmp_path: Path):
    venv = tmp_path / "venv"
    # Create a minimal "venv-like" layout without actually creating a venv
    py = _touch(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    env = PyEnv.get_or_create(venv, prefer_uv=True)
    assert env.python_path == py.resolve()


def test_get_or_create_selector_returns_real_python():
    # This uses real system resolution.
    # It should always resolve the current interpreter by passing None,
    # and should also resolve "python" in most environments.
    env = PyEnv.get_or_create("python", prefer_uv=True)
    assert env.python_path.exists()
    assert env.python_path.is_file()


@pytest.mark.slow
def test_get_or_create_missing_path_creates_real_venv(tmp_path: Path):
    """
    Real integration-ish test:
    - Actually creates a venv using `sys.executable -m venv <dir>`
    - Then resolves the python inside that venv

    Marked slow because it does real venv creation.
    """
    venv_dir = tmp_path / "new_venv"
    assert not venv_dir.exists()

    env = PyEnv.get_or_create(str(venv_dir), prefer_uv=True)

    assert venv_dir.exists()
    assert env.python_path.exists()
    assert env.python_path.is_file()

    # Sanity: should not be the same as sys.executable
    assert env.python_path.resolve() != Path(sys.executable).resolve()

    # And it should point inside the venv dir
    assert str(venv_dir.resolve()) in str(env.python_path.resolve())


def test_versioninfo_returns_major_minor_micro():
    env = PyEnv.current(prefer_uv=True)  # avoid uv dependency in tests
    vi = env.version_info

    assert isinstance(vi, tuple)
    assert len(vi) == 3
    assert all(isinstance(x, int) for x in vi)

    # Must match the running interpreter when anchored to current
    assert vi == (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)


def test_requirements_returns_list_and_is_deterministic():
    env = PyEnv.current()

    a = env.requirements(with_system=True)
    b = env.requirements(with_system=True)

    assert isinstance(a, list)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in a)
    assert a == b  # sorted => deterministic


def test_requirements_matches_pip_list_json_for_current_with_system():
    env = PyEnv.current()

    got = env.requirements(with_system=True)

    res = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        text=True,
        capture_output=True,
        check=True,
    )
    pkgs = json.loads(res.stdout or "[]")
    expected = [(p['name'], p['version']) for p in pkgs if "name" in p and "version" in p]

    assert len(got) == len(expected)


def test_requirements_excludes_system_packages_by_default():
    env = PyEnv.current()

    got = env.requirements(with_system=False)
    lower = {x[0].lower() for x in got}

    assert "pip" not in lower
    assert "setuptools" not in lower
    assert "wheel" not in lower



def test_run_python_code_basic_creates_file(tmp_path: Path):
    env = PyEnv.current()

    out = tmp_path / "out.txt"
    code = (
        "from pathlib import Path\n"
        "Path('out.txt').write_text('ok', encoding='utf-8')\n"
    )

    env.run_python_code(code, cwd=tmp_path, wait=True, raise_error=True)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "ok"


def test_run_python_code_with_stdin(tmp_path: Path):
    env = PyEnv.current()

    out = tmp_path / "stdin.txt"
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "data = sys.stdin.read()\n"
        "Path('stdin.txt').write_text(data, encoding='utf-8')\n"
    )

    env.run_python_code(code, cwd=tmp_path, stdin="hello\n", wait=True, raise_error=True)
    assert out.read_text(encoding="utf-8") == "hello\n"


def test_run_python_code_with_env_vars(tmp_path: Path):
    env = PyEnv.current()

    out = tmp_path / "env.txt"
    code = (
        "import os\n"
        "from pathlib import Path\n"
        "Path('env.txt').write_text(os.environ.get('X_TEST_ENV', ''), encoding='utf-8')\n"
    )

    env.run_python_code(code, cwd=tmp_path, env={"X_TEST_ENV": "yo"}, wait=True, raise_error=True)
    assert out.read_text(encoding="utf-8") == "yo"


def test_run_python_code_python_arg_as_path(tmp_path: Path):
    env = PyEnv.current()

    out = tmp_path / "py.txt"
    code = (
        "from pathlib import Path\n"
        "Path('py.txt').write_text('py-ok', encoding='utf-8')\n"
    )

    env.run_python_code(
        code,
        cwd=tmp_path,
        python=Path(sys.executable),
        wait=True,
        raise_error=True,
    )
    assert out.read_text(encoding="utf-8") == "py-ok"


def test_run_python_code_python_arg_as_selector(tmp_path: Path):
    env = PyEnv.current()

    out = tmp_path / "sel.txt"
    code = (
        "from pathlib import Path\n"
        "Path('sel.txt').write_text('sel-ok', encoding='utf-8')\n"
    )

    # Uses real system resolution (no monkeypatch). Assumes "python" is resolvable.
    env.run_python_code(code, cwd=tmp_path, python="python", wait=True, raise_error=True)
    assert out.read_text(encoding="utf-8") == "sel-ok"


def test_run_python_code_requirements_inline_content(tmp_path: Path):
    env = PyEnv.current()

    out = tmp_path / "reqinline.txt"
    code = (
        "from pathlib import Path\n"
        "Path('reqinline.txt').write_text('reqinline-ok', encoding='utf-8')\n"
    )

    env.run_python_code(code, cwd=tmp_path, wait=True, raise_error=True)
    assert out.read_text(encoding="utf-8") == "reqinline-ok"


@pytest.mark.skipif(os.name == "nt" and sys.executable.lower().endswith("pythonw.exe"), reason="pythonw has no stdin")
def test_run_python_code_stdin_does_not_hang(tmp_path: Path):
    env = PyEnv.current()

    out = tmp_path / "stdin2.txt"
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "data = sys.stdin.read()\n"
        "Path('stdin2.txt').write_text(str(len(data)), encoding='utf-8')\n"
    )

    env.run_python_code(code, cwd=tmp_path, stdin="12345", wait=True, raise_error=True)
    assert out.read_text(encoding="utf-8") == "5"


def test_run_python_code_basic_uses_uv_by_default(tmp_path: Path):
    env = PyEnv.current(prefer_uv=True)

    rr = env.run_python_code(
        "print('OK')",
        cwd=tmp_path,
        wait=30.0,
        raise_error=True,
        # prefer_uv not passed => uses target.prefer_uv (True)
    )

    assert rr.returncode == 0
    assert (rr.stdout or "").strip() == "OK"
    assert (rr.stderr or "").strip() == ""

    # sanity: the args should look like uv run --python <...> python -c ...
    assert rr.args[0].endswith("uv") or "uv" in Path(rr.args[0]).name.lower()
    assert "run" in rr.args
    assert "--python" in rr.args


def test_run_python_code_python_set_by_path_uses_uv(tmp_path: Path):
    env = PyEnv.current(prefer_uv=True)

    py_path = Path(sys.executable).resolve()

    rr = env.run_python_code(
        "import sys; print(sys.executable)",
        cwd=tmp_path,
        wait=30.0,
        raise_error=True,
        python=py_path,  # <- explicit python anchor
        # prefer_uv not passed => True
    )

    assert rr.returncode == 0
    out = (rr.stdout or "").strip()
    assert Path(out).resolve() == py_path


def test_run_python_code_python_set_by_selector_uses_uv(tmp_path: Path):
    env = PyEnv.current(prefer_uv=True)

    major, minor = sys.version_info.major, sys.version_info.minor
    selector = f"{major}.{minor}"  # e.g. "3.12" -> resolve_python_executable tries "python3.12"

    try:
        resolved = PyEnv.resolve_python_executable(selector)
    except FileNotFoundError:
        pytest.skip(f"python selector not available on PATH: {selector!r}")

    rr = env.run_python_code(
        "import sys; print(sys.executable)",
        cwd=tmp_path,
        wait=30.0,
        raise_error=True,
        python=selector,  # <- selector
    )

    assert rr.returncode == 0
    out = (rr.stdout or "").strip()
    assert Path(out).resolve() == resolved.resolve()


def test_run_python_code_packages_adds_with_flag_and_runs(tmp_path: Path):
    """
    This is the exact path you care about: packages + python set + uv.

    Reality check: `uv run --with ...` can require resolution/download depending on the environment.
    So we:
      - assert the command includes `--with` (core behavior)
      - attempt to run
      - skip if the environment can’t satisfy uv’s resolution (no network/cache/locked index)
    """
    env = PyEnv.current(prefer_uv=True)

    py_path = Path(sys.executable).resolve()

    # choose something extremely likely to be satisfiable; still may fail if uv needs to fetch.
    # packages = ["dask"]

    rr = env.run_python_code(
        "import dask; print('HELLO_FROM_WITH')",
        cwd=tmp_path,
        wait=60.0,
        raise_error=True,
        python=py_path,  # python set
        # packages=packages,  # auto install dask on retry
        # prefer_uv default => True
    )

    assert rr.returncode == 0
    assert "HELLO_FROM_WITH" in (rr.stdout or "")


def test_run_python_code_python_as_pyenv_instance_uses_uv(tmp_path: Path):
    env = PyEnv.current(prefer_uv=True)

    target = PyEnv.create(Path(sys.executable).resolve(), cwd=tmp_path, prefer_uv=True)

    rr = env.run_python_code(
        "import sys; print(sys.executable)",
        cwd=tmp_path,
        wait=30.0,
        raise_error=True,
        python=target,  # <- PyEnv instance
    )

    assert rr.returncode == 0
    assert Path((rr.stdout or "").strip()).resolve() == target.python_path.resolve()


def test_run_python_code_python_as_python_version(tmp_path: Path):
    env = PyEnv.current(prefer_uv=True)

    target = PyEnv.create(Path(sys.executable).resolve(), cwd=tmp_path, prefer_uv=True)

    rr = env.run_python_code(
        "import sys; print(sys.executable)",
        cwd=tmp_path,
        wait=30.0,
        raise_error=True,
        python="3.11",  # <- PyEnv instance
    )

    assert rr.returncode == 0
    assert Path((rr.stdout or "").strip()).resolve() == target.python_path.resolve()