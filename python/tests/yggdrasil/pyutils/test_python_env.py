from __future__ import annotations

import os
import re
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from yggdrasil.pyutils import python_env as pe
from yggdrasil.pyutils.python_env import PythonEnv, PythonEnvError


# ------------------------------------------------------------
# helpers (no mocking, real HOME)
# ------------------------------------------------------------

def _have_uv() -> bool:
    return shutil.which("uv") is not None


def _unique_name(prefix: str = "pytest-python-env") -> str:
    # safe for _safe_env_name (letters/digits/._-)
    return f"{prefix}-{os.getpid()}-{uuid.uuid4().hex}"


def _base_envs_dir() -> Path:
    # real user HOME
    return pe._user_envs_dir()


def _cleanup_env_artifacts(name_prefix: str) -> None:
    """
    Remove only env directories under ~/.python/envs that match our prefix:
      - <name_prefix>*
      - and any backups created by change_python_version: <name>.pychange-*
      - and broken backups from create/update: <name>.broken-*
    """
    base = _base_envs_dir()
    if not base.exists():
        return

    # remove direct env dirs that start with prefix
    for p in base.glob(f"{name_prefix}*"):
        # only touch dirs we own by naming convention
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    # remove pychange/broken backups for any env with prefix
    for p in base.glob(f"{name_prefix}*.pychange-*"):
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
    for p in base.glob(f"{name_prefix}*.broken-*"):
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


def _create_uv_env(root: Path) -> PythonEnv:
    """
    Create a real venv using uv, offline-friendly (no pip installs).
    """
    uv = PythonEnv.ensure_uv()
    pe._run_cmd([uv, "venv", str(root), "--python", sys.executable])
    env = PythonEnv(root)
    assert env.exists(), f"uv venv created but python missing at {env.python_executable}"

    env.update(
        packages=["pandas"]
    )

    return env


def _slug(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = re.sub(r"[^A-Za-z0-9._+-]+", "-", s)
    return s.strip("-") or "unknown"


# ------------------------------------------------------------
# session-level cleanup guard
# ------------------------------------------------------------

@pytest.fixture(scope="session")
def _session_prefix() -> str:
    prefix = _unique_name("pytest-python-env")
    yield prefix
    # best-effort cleanup at end of session
    _cleanup_env_artifacts(prefix)


@pytest.fixture
def env_name(_session_prefix: str) -> str:
    # a new env per test to avoid cross-test contamination
    return _unique_name(_session_prefix)


@pytest.fixture
def real_env(env_name: str) -> PythonEnv:
    """
    Real env in real HOME: ~/.python/envs/<env_name>
    Skips if uv isn't available.
    """
    if not _have_uv():
        pytest.skip("uv not on PATH; skipping real-env tests")

    root = _base_envs_dir() / env_name
    root.parent.mkdir(parents=True, exist_ok=True)

    env = _create_uv_env(root)
    try:
        yield env
    finally:
        _cleanup_env_artifacts(env_name)


# ------------------------------------------------------------
# pure helper/unit tests (no uv needed)
# ------------------------------------------------------------

def test_safe_env_name_accepts_common_inputs():
    assert pe._safe_env_name("my-env") == "my-env"
    assert pe._safe_env_name(" my env ") == "my-env"
    assert pe._safe_env_name("a/b\\c") == "a-b-c"
    assert pe._safe_env_name("a..b__c--d") == "a..b__c--d"


def test_safe_env_name_rejects_empty_and_dot_names():
    for bad in ["", "   ", ".", ".."]:
        with pytest.raises(PythonEnvError):
            pe._safe_env_name(bad)


def test_dedupe_keep_order():
    assert pe._dedupe_keep_order([" a ", "b", "a", "", "  ", "c", "b"]) == ["a", "b", "c"]


def test_split_on_tag_basic():
    before, payload = pe._split_on_tag("x\ny\nRESULT:123\nz\n", "RESULT:")
    assert before == ["x", "y"]
    assert payload == "123"

    before2, payload2 = pe._split_on_tag("x\ny\nz\n", "RESULT:")
    assert before2 == ["x", "y", "z"]
    assert payload2 is None


def test_norm_env_sets_unbuffered():
    out = pe._norm_env({"X": "1"})
    assert out["X"] == "1"
    assert out["PYTHONUNBUFFERED"] == "1"


# ------------------------------------------------------------
# module-level lock tests (no uv needed)
# ------------------------------------------------------------

def test_module_locks_are_singletons_for_same_root(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "a"  # same path
    lk1 = pe._get_env_lock(a)
    lk2 = pe._get_env_lock(b)
    assert lk1 is lk2


def test_module_locks_differ_for_different_roots(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    lk1 = pe._get_env_lock(a)
    lk2 = pe._get_env_lock(b)
    assert lk1 is not lk2


def test_locked_env_is_reentrant(tmp_path: Path):
    root = tmp_path / "x"
    with pe._locked_env(root):
        # RLock: should not deadlock if same thread re-enters
        with pe._locked_env(root):
            assert True


# ------------------------------------------------------------
# user env discovery / get / delete using REAL HOME (no uv needed)
# ------------------------------------------------------------

def _make_fake_env_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyvenv.cfg").write_text("home=fake\n", encoding="utf-8")
    if os.name == "nt":
        py = root / "Scripts" / "python.exe"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_bytes(b"x")
    else:
        py = root / "bin" / "python"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("#!/bin/sh\necho fake\n", encoding="utf-8")


def test_iter_user_envs_and_get_and_delete_real_home(_session_prefix: str):
    """
    This test does NOT require uv. It creates a minimal fake env layout in real HOME.
    It only asserts presence/absence of envs that match our session prefix.
    """
    base = _base_envs_dir()
    base.mkdir(parents=True, exist_ok=True)

    name_a = _unique_name(_session_prefix)
    name_b = _unique_name(_session_prefix)
    name_hidden = f".{_unique_name(_session_prefix)}"

    root_a = base / name_a
    root_b = base / "nested" / name_b
    root_hidden = base / name_hidden

    try:
        _make_fake_env_dir(root_a)
        _make_fake_env_dir(root_b)
        _make_fake_env_dir(root_hidden)

        # get() should find direct env by name under ~/.python/envs/<name>
        got = PythonEnv.get(name_a, require_python=True)
        assert got is not None
        assert got.root == root_a.resolve()
        assert got.exists()

        # iter_user_envs should include our envs (plus other user's envs, ignore them)
        found = list(PythonEnv.iter_user_envs(max_depth=2, include_hidden=False, require_python=True, dedupe=True))
        roots = {e.root for e in found}
        assert root_a.resolve() in roots
        assert root_b.resolve() in roots
        assert root_hidden.resolve() not in roots

        found_hidden = list(PythonEnv.iter_user_envs(max_depth=2, include_hidden=True, require_python=True, dedupe=True))
        roots_hidden = {e.root for e in found_hidden}
        assert root_hidden.resolve() in roots_hidden

        # delete by name removes only that env
        PythonEnv.delete(name_a, missing_ok=False)
        assert not root_a.exists()

        # missing_ok behavior
        PythonEnv.delete("definitely-does-not-exist-" + uuid.uuid4().hex, missing_ok=True)
        with pytest.raises(PythonEnvError):
            PythonEnv.delete("definitely-does-not-exist-" + uuid.uuid4().hex, missing_ok=False)

    finally:
        # cleanup only our session prefix artifacts
        _cleanup_env_artifacts(_session_prefix)


# ------------------------------------------------------------
# get_current() (no uv required)
# ------------------------------------------------------------

def test_get_current_sanity():
    cur = PythonEnv.get_current()
    assert cur.root.exists()
    # In normal runtimes, sys.executable exists:
    assert cur.python_executable.exists()


# ------------------------------------------------------------
# exec_code / exec_code_and_return (needs uv, real env)
# ------------------------------------------------------------

def test_exec_code_runs(real_env: PythonEnv):
    out = real_env.exec_code("print('hi')")
    assert out.strip() == "hi"


def test_exec_code_respects_cwd(real_env: PythonEnv, tmp_path: Path):
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / "x.txt").write_text("abc", encoding="utf-8")
    out = real_env.exec_code("import pathlib; print(pathlib.Path('x.txt').read_text())", cwd=wd)
    assert out.strip() == "abc"


def test_exec_code_respects_env_vars(real_env: PythonEnv):
    env = dict(os.environ)
    env["FOO_TEST"] = "BAR"
    out = real_env.exec_code("import os; print(os.environ.get('FOO_TEST',''))", env=env)
    assert out.strip() == "BAR"


def test_exec_code_and_return_list_literal(real_env: PythonEnv):
    code = "print('RESULT:' + '[1, 2, 3]')"
    val = real_env.exec_code_and_return(code, result_tag="RESULT:")
    assert val == "[1, 2, 3]"


def test_exec_code_and_return_parse_json(real_env: PythonEnv):
    code = "print('RESULT:' + '{\"a\": 1, \"b\": [2,3]}')"
    val = real_env.exec_code_and_return(code, parse_json=True)
    assert val == {"a": 1, "b": [2, 3]}


def test_exec_code_and_return_envelope_ok_raw(real_env: PythonEnv):
    code = (
        "import json\n"
        "env = {'ok': True, 'encoding': 'raw', 'return': {'x': 1}, 'error': None}\n"
        "print('RESULT:' + json.dumps(env))\n"
    )
    val = real_env.exec_code_and_return(code)
    assert val == {"x": 1}


def test_exec_code_and_return_envelope_error_raises(real_env: PythonEnv):
    code = (
        "import json\n"
        "env = {'ok': False, 'encoding': 'raw', 'return': None, 'error': 'nope'}\n"
        "print('RESULT:' + json.dumps(env))\n"
    )
    with pytest.raises(PythonEnvError, match="Remote code reported failure"):
        real_env.exec_code_and_return(code)


def test_exec_code_and_return_missing_tag_raises(real_env: PythonEnv):
    with pytest.raises(PythonEnvError, match="Result tag not found"):
        real_env.exec_code_and_return("print('no tag')", result_tag="RESULT:")


# ------------------------------------------------------------
# change_python_version (needs uv, real env)
# ------------------------------------------------------------

def test_change_python_version_same_version_returns_self(real_env: PythonEnv):
    # request a prefix version like "3.12" (matches current major.minor)
    major, minor, _patch = real_env.version_info
    req = f"{major}.{minor}"

    before_backups = set(real_env.root.parent.glob(f"{real_env.root.name}.pychange-*"))
    out = real_env.change_python_version(req, keep_packages=False)
    after_backups = set(real_env.root.parent.glob(f"{real_env.root.name}.pychange-*"))

    assert out is real_env
    assert after_backups == before_backups


def test_change_python_version_path_recreates_and_moves_aside(real_env: PythonEnv):
    # passing an interpreter path should force recreate (not an early-return)
    sentinel = real_env.root / "SENTINEL.txt"
    sentinel.write_text("yo", encoding="utf-8")

    out = real_env.change_python_version(sys.executable, keep_packages=False)

    assert out.root == real_env.root
    assert out.exists()
    assert not sentinel.exists()

    backups = list(out.root.parent.glob(f"{out.root.name}.pychange-*"))
    assert backups, "Expected pychange backup dir"
    # validate sentinel moved into one of the backups
    assert any((b / "SENTINEL.txt").exists() for b in backups)


# ------------------------------------------------------------
# export_requirements_matrix (needs uv, real env)
# ------------------------------------------------------------
def test_requirements_matrix_out_dir_none_returns_text_from_self(real_env: PythonEnv):
    res = real_env.requirements(
        out_dir=None,
        base_name="requirements",
        include_input=True,
        include_frozen=False,  # keep it stable/fast
    )

    assert isinstance(res, str)
    assert res


def test_installed_packages(real_env: PythonEnv):
    pkgs = real_env.installed_packages(parsed=True)
    names = [p[0] for p in pkgs]
    assert "pandas" in names


def test_export_requirements_matrix_out_dir_none_returns_text(real_env: PythonEnv):
    buffers: dict[str, str] = {}

    res = real_env.export_requirements_matrix(
        python_versions=[sys.executable],
        out_dir=None,
        base_name="requirements",
        include_input=True,
        include_frozen=False,  # keep it stable/fast
        buffers=buffers,
    )

    assert sys.executable in res
    assert isinstance(res[sys.executable], str)
    assert "requirements.in" in buffers
    # compiled file text should also be in buffers
    expected_name = f"requirements-py{_slug(str(sys.executable))}.txt"
    assert expected_name in buffers


def test_export_requirements_matrix_out_dir_writes_files(real_env: PythonEnv, tmp_path: Path):
    out_dir = tmp_path / "reqs-out"
    buffers: dict[str, str] = {}

    res = real_env.export_requirements_matrix(
        [sys.executable],
        out_dir=out_dir,
        base_name="reqs",
        include_input=True,
        include_frozen=True,
        buffers=buffers,
    )

    p = res[sys.executable]
    assert isinstance(p, Path)
    assert p.exists()

    assert (out_dir / "reqs.in").exists()
    assert (out_dir / "reqs.frozen.txt").exists()
    expected_compiled = out_dir / f"reqs-py{_slug(str(sys.executable))}.txt"
    assert expected_compiled.exists()

    assert "reqs.in" in buffers
    assert "reqs.frozen.txt" in buffers
    assert expected_compiled.name in buffers
