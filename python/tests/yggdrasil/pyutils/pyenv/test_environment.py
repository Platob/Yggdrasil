# tests/test_pyenv_integration.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from yggdrasil.pyutils.pyenv import PyEnv


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return (venv_dir / "Scripts" / "python.exe").resolve()
    return (venv_dir / "bin" / "python").resolve()


def _create_venv(venv_dir: Path) -> Path:
    # Use the current interpreter to create an isolated venv
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    py = _venv_python(venv_dir)
    assert py.exists(), f"venv python not found at {py}"
    return py


def _make_local_package(project_dir: Path) -> tuple[Path, str, str]:
    """
    Creates a tiny local setuptools project you can pip install without internet.
    Returns (project_dir, dist_name, import_name).
    """
    dist_name = "dummy-pkg"
    import_name = "dummy_pkg"

    pkg_dir = project_dir / import_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")

    (project_dir / "setup.py").write_text(
        (
            "from setuptools import setup, find_packages\n"
            "setup(\n"
            f'    name="{dist_name}",\n'
            '    version="0.1.0",\n'
            '    packages=find_packages(),\n'
            ")\n"
        ),
        encoding="utf-8",
    )

    return project_dir, dist_name, import_name


# ----------------------------
# Pure function tests (no uv)
# ----------------------------
def test_resolve_python_executable_none_uses_sys_executable():
    assert PyEnv.resolve_python_executable(None) == Path(sys.executable).resolve()


def test_resolve_python_executable_existing_path(tmp_path: Path):
    exe = tmp_path / "python"
    exe.write_text("stub", encoding="utf-8")
    assert PyEnv.current().resolve_python_executable(exe) == exe.resolve()


def test_import_module_present_stdlib():
    mod = PyEnv.current().import_module("json", install=False)
    assert mod.__name__ == "json"


def test_import_module_missing_install_false_raises():
    with pytest.raises(ModuleNotFoundError):
        PyEnv.current().import_module("this_module_should_not_exist_9f3f2b8c", install=False)


def test_import_module_missing_install_true():
    env = PyEnv.current()

    try:
        fv = env.import_module(
            pip_name="claude-agent-sdk",
            install=True
        )

        assert fv is not None
    finally:
        env.uninstall("claude-agent-sdk", wait=False)


# ----------------------------
# Integration tests (real uv + real venv + real pip)
# ----------------------------
def test_uv_bin_resolves_and_is_file(tmp_path: Path):
    # Uses current interpreter for runtime import of uv; no mocking.
    env = PyEnv(python_path=Path(sys.executable).resolve(), cwd=tmp_path.resolve(), prefer_uv=True)
    uv_bin = env.uv_bin
    assert uv_bin.exists()
    assert uv_bin.is_file()


def test_pip_list_runs_in_venv(tmp_path: Path):
    venv_dir = tmp_path / ".venv"
    vpy = _create_venv(venv_dir)

    env = PyEnv(python_path=vpy, cwd=tmp_path.resolve(), prefer_uv=True)

    # Uses your real SystemCommand plumbing; expects a CompletedProcess-like object
    cp = env.pip("list", wait=True)
    assert hasattr(cp, "returncode")
    assert cp.returncode == 0


def test_install_local_project_then_import_then_uninstall(tmp_path: Path):
    """
    Real end-to-end:
      - create venv
      - create a local package (no network)
      - env.install(<local path>)
      - verify import works inside that venv
      - env.uninstall(dist)
      - verify import fails
    """
    venv_dir = tmp_path / ".venv"
    vpy = _create_venv(venv_dir)

    project_dir = tmp_path / "proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    proj, dist_name, import_name = _make_local_package(project_dir)

    env = PyEnv(python_path=vpy, cwd=tmp_path.resolve(), prefer_uv=True)

    # Install from local path (no internet)
    r = env.install(str(proj), wait=True)
    assert r is not None

    # Verify import with the venv python directly (keeps the assertion independent of SystemCommand behavior)
    out = subprocess.run(
        [str(vpy), "-c", f"import {import_name}; print({import_name}.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "0.1.0"

    # Also exercise run_python_code (real) if your SystemCommand supports it
    rr = env.run_python_code(f"import {import_name}; print({import_name}.__version__)", wait=True)
    assert getattr(rr, "returncode", 0) in (0, None)  # SystemCommand may or may not expose returncode

    # Uninstall distribution name
    u = env.uninstall(dist_name, wait=True)
    assert u is not None

    # Confirm it is really gone
    p = subprocess.run(
        [str(vpy), "-c", f"import {import_name}"],
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0


def test_install_with_requirements_content_tempfile_cleanup(tmp_path: Path):
    """
    Real-ish test of the 'requirements content' branch:
      - create local project
      - pass requirements as content (a line pointing to local path)
      - ensure install succeeds
    Note: we cannot directly assert tempfile deletion without peeking into internals,
    but success path + no leftover files in cwd is a practical check.
    """
    venv_dir = tmp_path / ".venv"
    vpy = _create_venv(venv_dir)

    project_dir = tmp_path / "proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    proj, dist_name, import_name = _make_local_package(project_dir)

    env = PyEnv(python_path=vpy, cwd=tmp_path.resolve(), prefer_uv=True)

    before = {p.name for p in tmp_path.iterdir() if p.is_file()}

    # requirements treated as *content* because Path("...") doesn't exist as a file
    r = env.install(requirements=str(proj), wait=True)
    assert r is not None

    # Validate import works
    subprocess.run(
        [str(vpy), "-c", f"import {import_name}; print('ok')"],
        check=True,
        capture_output=True,
        text=True,
    )

    after = {p.name for p in tmp_path.iterdir() if p.is_file()}
    # best-effort: no new requirements_*.txt file left behind in cwd
    leftovers = [n for n in (after - before) if n.startswith("requirements_") and n.endswith(".txt")]
    assert leftovers == []
