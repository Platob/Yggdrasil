"""Comprehensive unit tests for :mod:`yggdrasil.environ.environment`.

Covers:
  - ``safe_pip_name`` — string mapping, tuple versioning, iterable input
  - ``SYSTEM_LIBS`` — type, membership, case-sensitivity
  - ``PyEnv`` construction — ``__post_init__``, ``__getstate__``/``__setstate__``
  - Python resolution — ``resolve_python_executable``, ``_find_python_in_dir``, ``_venv_python_from_dir``
  - Venv lifecycle — ``venv()``, ``create()``, ``get_or_create()``
  - Properties — ``is_current``, ``is_windows``, ``bin_path``, ``root_path``, ``version_info``
  - uv resolution — ``has_uv``, ``ensure_uv``, ``uv_path``, ``_uv_base_cmd``
  - Package management — ``requirements``, ``install``, ``update``, ``uninstall``, ``pip``
  - Deletion — ``delete``
  - Execution — ``run_python_code``
  - Import — ``import_module``, ``runtime_import_module``
  - Helpers — ``_looks_like_path``, ``_is_externally_managed_failure``, ``get_root_module_directory``
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yggdrasil.environ.environment as mod
from yggdrasil.environ.environment import (
    PyEnv,
    SYSTEM_LIBS,
    safe_pip_name,
)


# ===================================================================
# Dummy / mock helpers
# ===================================================================

class DummyVersionInfo:
    """Lightweight stand-in for :class:`VersionInfo`."""

    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class DummyWaitingConfig:
    """Stand-in for :class:`WaitingConfig` — ``check_arg`` returns a bool."""

    @staticmethod
    def check_arg(value):
        return bool(value)


class DummyLazyResult:
    """Fake lazy result returned by :class:`DummySystemCommandModule`."""

    def __init__(self):
        self.wait_calls = []

    def wait(self, wait=True, raise_error=True, auto_install=False):
        self.wait_calls.append(
            {"wait": wait, "raise_error": raise_error, "auto_install": auto_install}
        )
        return self


class DummyProc:
    """Minimal Popen substitute with a writable stdin."""

    def __init__(self):
        self.stdin = DummyStdin()


class DummyStdin:
    """In-memory stdin capture."""

    def __init__(self):
        self.buffer = []
        self.closed = False

    def write(self, value):
        self.buffer.append(value)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class DummySystemCommandModule:
    """Intercepts all :meth:`SystemCommand.run_lazy` calls."""

    def __init__(self):
        self.calls: list[dict] = []
        self.next_result = DummyLazyResult()

    def run_lazy(self, cmd, cwd=None, env=None, python=None):
        self.calls.append({"cmd": cmd, "cwd": cwd, "env": env, "python": python})
        result = self.next_result
        result.popen = DummyProc()
        return result


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(autouse=True)
def _reset_current_pyenv(monkeypatch):
    """Reset the module-level singleton before and after each test."""
    monkeypatch.setattr(mod, "CURRENT_PYENV", None)
    yield
    monkeypatch.setattr(mod, "CURRENT_PYENV", None)


@pytest.fixture
def dummy_system_command(monkeypatch):
    """Patch SystemCommand and WaitingConfig with dummies."""
    fake = DummySystemCommandModule()
    monkeypatch.setattr(mod, "SystemCommand", fake)
    monkeypatch.setattr(mod, "WaitingConfig", DummyWaitingConfig)
    monkeypatch.setattr(mod, "VersionInfo", DummyVersionInfo)
    return fake


@pytest.fixture
def env(tmp_path):
    """A minimal PyEnv pointing at a fake python binary."""
    py = tmp_path / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("", encoding="utf-8")
    return PyEnv(python_path=py, cwd=tmp_path, prefer_uv=True)


# ===================================================================
# safe_pip_name
# ===================================================================

class TestSafePipName:
    """Test the ``safe_pip_name`` free function."""

    def test_known_mapping(self):
        assert safe_pip_name("yaml") == "PyYAML"
        assert safe_pip_name("jwt") == "PyJWT"
        assert safe_pip_name("dotenv") == "python-dotenv"
        assert safe_pip_name("dateutil") == "python-dateutil"
        assert safe_pip_name("yggdrasil") == "ygg"

    def test_unknown_passthrough(self):
        assert safe_pip_name("pyarrow") == "pyarrow"
        assert safe_pip_name("requests") == "requests"

    def test_tuple_version(self):
        assert safe_pip_name(("yaml", "6.0.2")) == "PyYAML==6.0.2"
        assert safe_pip_name(("pyarrow", "18.1.0")) == "pyarrow==18.1.0"

    def test_iterable_mixed(self):
        result = safe_pip_name(["yaml", ("jwt", "2.9.0"), "pyarrow"])
        assert result == ["PyYAML", "PyJWT==2.9.0", "pyarrow"]

    def test_empty_iterable(self):
        assert safe_pip_name([]) == []

    def test_single_string_not_in_mapping(self):
        assert safe_pip_name("some-random-pkg") == "some-random-pkg"


# ===================================================================
# SYSTEM_LIBS
# ===================================================================

class TestSystemLibs:
    """Verify ``SYSTEM_LIBS`` type and content."""

    def test_is_frozenset(self):
        assert isinstance(SYSTEM_LIBS, frozenset)

    def test_membership_spot_checks(self):
        assert "pip" in SYSTEM_LIBS
        assert "setuptools" in SYSTEM_LIBS
        assert "requests" in SYSTEM_LIBS
        assert "pyspark" in SYSTEM_LIBS
        assert "pyarrow" not in SYSTEM_LIBS

    def test_all_entries_are_lowercase(self):
        for entry in SYSTEM_LIBS:
            assert entry == entry.lower(), f"SYSTEM_LIBS entry {entry!r} is not lowercase"

    def test_not_empty(self):
        assert len(SYSTEM_LIBS) > 100


# ===================================================================
# PyEnv construction
# ===================================================================

class TestPyEnvConstruction:
    """Test ``PyEnv`` creation, serialisation, and singleton."""

    def test_post_init_resolves_paths(self, tmp_path):
        py = tmp_path / "venv" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")

        obj = PyEnv(python_path=py, cwd=tmp_path / ".")
        assert obj.python_path.is_absolute()
        assert obj.cwd.is_absolute()

    def test_getstate_setstate_roundtrip(self, tmp_path):
        py = tmp_path / "venv" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")

        env1 = PyEnv(python_path=py, cwd=tmp_path, prefer_uv=False)
        state = env1.__getstate__()
        env2 = object.__new__(PyEnv)
        env2.__setstate__(state)

        assert env2.python_path == py.resolve()
        assert env2.cwd == tmp_path.resolve()
        assert env2.prefer_uv is False
        assert env2._version_info is None
        assert env2._uv_bin is None

    def test_instance_factory(self, tmp_path):
        py = tmp_path / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")

        env = PyEnv.instance(py, cwd=tmp_path, prefer_uv=False)
        assert env.python_path == py.resolve()
        assert env.prefer_uv is False

    def test_current_singleton(self, monkeypatch, tmp_path):
        py = tmp_path / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")

        monkeypatch.setattr(
            PyEnv, "resolve_python_executable",
            staticmethod(lambda python=None: py),
        )
        env1 = PyEnv.current(prefer_uv=False)
        env2 = PyEnv.current(prefer_uv=True)  # returns cached singleton

        assert env1 is env2
        assert mod.CURRENT_PYENV is env1

    def test_current_sets_version_info_for_running_interpreter(self, monkeypatch):
        py = Path(sys.executable).resolve()
        monkeypatch.setattr(
            PyEnv, "resolve_python_executable",
            staticmethod(lambda python=None: py),
        )
        monkeypatch.setattr(mod, "VersionInfo", DummyVersionInfo)

        env = PyEnv.current()
        assert env._version_info is not None
        assert env._version_info.major == sys.version_info.major

    def test_get_or_create_with_pyenv_identifier(self, monkeypatch, env):
        called = {}
        monkeypatch.setattr(env, "install", lambda *pkgs: called.update({"pkgs": pkgs}))
        out = PyEnv.get_or_create(env, packages=["pyarrow"])
        assert out is env
        assert called["pkgs"] == ("pyarrow",)


# ===================================================================
# Python resolution
# ===================================================================

class TestResolvePython:
    """Test ``resolve_python_executable`` and related helpers."""

    def test_none_returns_sys_executable(self, monkeypatch):
        monkeypatch.setattr(sys, "executable", "/usr/bin/python-real")
        path = PyEnv.resolve_python_executable(None)
        assert path == Path("/usr/bin/python-real").resolve()

    def test_empty_string_returns_sys_executable(self, monkeypatch):
        monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
        assert PyEnv.resolve_python_executable("") == Path("/usr/bin/python3").resolve()

    def test_existing_file(self, tmp_path):
        py = tmp_path / "python3.12"
        py.write_text("", encoding="utf-8")
        assert PyEnv.resolve_python_executable(py) == py.resolve()

    def test_directory(self, tmp_path):
        venv_dir = tmp_path / "venv"
        py = venv_dir / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")
        assert PyEnv.resolve_python_executable(venv_dir) == py.resolve()

    def test_version_selector_via_which(self, monkeypatch):
        monkeypatch.setattr(
            mod.shutil, "which",
            lambda s: "/usr/bin/python3.12" if s == "python3.12" else None,
        )
        assert PyEnv.resolve_python_executable("3.12") == Path("/usr/bin/python3.12").resolve()

    def test_not_found_raises(self, monkeypatch):
        monkeypatch.setattr(mod.shutil, "which", lambda s: None)
        with pytest.raises(FileNotFoundError):
            PyEnv.resolve_python_executable("3.99")

    def test_find_python_in_dir_standard_layout(self, tmp_path):
        root = tmp_path / "env"
        py = root / "bin" / "python3"
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")
        assert PyEnv._find_python_in_dir(root) == py.resolve()

    def test_find_python_in_dir_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PyEnv._find_python_in_dir(tmp_path / "missing")

    def test_venv_python_from_dir_posix(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod.os, "name", "posix")
        venv_dir = tmp_path / "venv"
        py = venv_dir / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")
        assert PyEnv._venv_python_from_dir(venv_dir) == py.resolve()

    def test_venv_python_from_dir_missing_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod.os, "name", "posix")
        with pytest.raises(ValueError, match="No Python executable found inside venv"):
            PyEnv._venv_python_from_dir(tmp_path / "missing")

    def test_venv_python_from_dir_missing_no_raise(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod.os, "name", "posix")
        result = PyEnv._venv_python_from_dir(tmp_path / "missing", raise_error=False)
        assert isinstance(result, Path)


# ===================================================================
# Venv and Create
# ===================================================================

class TestVenvAndCreate:
    """Test ``venv()``, ``create()``, and ``get_or_create()``."""

    def test_venv_current_alias(self, monkeypatch, env):
        sentinel = object.__new__(PyEnv)
        monkeypatch.setattr(
            PyEnv, "current",
            classmethod(lambda cls, prefer_uv=True: sentinel),
        )
        assert env.venv("current") is sentinel

    def test_venv_system_alias(self, monkeypatch, env):
        sentinel = object.__new__(PyEnv)
        monkeypatch.setattr(
            PyEnv, "current",
            classmethod(lambda cls, prefer_uv=True: sentinel),
        )
        assert env.venv("system") is sentinel

    def test_venv_none_returns_current(self, monkeypatch, env):
        sentinel = object.__new__(PyEnv)
        monkeypatch.setattr(
            PyEnv, "current",
            classmethod(lambda cls, prefer_uv=True: sentinel),
        )
        assert env.venv(None) is sentinel

    def test_venv_pyenv_instance_passthrough(self, monkeypatch, env):
        other = object.__new__(PyEnv)
        monkeypatch.setattr(other, "install", lambda *a: None)
        out = env.venv(other)
        assert out is other

    def test_venv_existing_file(self, monkeypatch, env, tmp_path):
        py = tmp_path / "some" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")

        captured = {}
        monkeypatch.setattr(
            PyEnv, "instance",
            classmethod(lambda cls, pp, cwd=None, prefer_uv=True, packages=None: captured.update(pp=pp) or "ENV"),
        )
        assert env.venv(py) == "ENV"
        assert Path(captured["pp"]) == py.resolve()

    def test_venv_existing_dir(self, monkeypatch, env, tmp_path):
        folder = tmp_path / "myenv"
        folder.mkdir()
        py = folder / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")

        monkeypatch.setattr(
            PyEnv, "_venv_python_from_dir",
            staticmethod(lambda p, raise_error=False: py),
        )
        monkeypatch.setattr(
            PyEnv, "instance",
            classmethod(lambda cls, pp, cwd=None, prefer_uv=True, packages=None: ("ENV", pp)),
        )
        out = env.venv(folder)
        assert out[0] == "ENV"
        assert out[1] == py

    def test_venv_missing_calls_create(self, monkeypatch, env, tmp_path):
        folder = tmp_path / "newenv"
        monkeypatch.setattr(
            PyEnv, "_venv_python_from_dir",
            staticmethod(lambda p, raise_error=False: p / "bin" / "python"),
        )
        created = {}
        monkeypatch.setattr(env, "create", lambda path, **kw: created.update(path=path) or "CREATED")
        assert env.venv(folder) == "CREATED"

    def test_create_builds_uv_command(self, monkeypatch, env, tmp_path, dummy_system_command):
        folder = tmp_path / "created-env"
        py = folder / "bin" / "python"

        monkeypatch.setattr(env, "_uv_base_cmd", lambda install_runtime=True: ["uv"])
        object.__setattr__(env, "_version_info", DummyVersionInfo(3, 12, 7))
        monkeypatch.setattr(
            PyEnv, "_venv_python_from_dir",
            staticmethod(lambda p, raise_error=True: py),
        )

        created_env = PyEnv(python_path=py, cwd=tmp_path, prefer_uv=True)
        install_calls = []
        monkeypatch.setattr(
            PyEnv, "instance",
            classmethod(lambda cls, pp, cwd=None, prefer_uv=True, packages=None: created_env),
        )
        monkeypatch.setattr(created_env, "install", lambda *p, wait=True: install_calls.append(p))

        env.create(folder, packages=["pyarrow"], wait=True)

        cmd = dummy_system_command.calls[0]["cmd"]
        assert cmd[:2] == ["uv", "venv"]
        assert str(folder) in cmd
        assert "--python" in cmd
        assert "3.12.7" in cmd
        assert "--seed" in cmd
        assert "--native-tls" in cmd
        assert "--clear" in cmd
        assert install_calls == [("pyarrow",)]


# ===================================================================
# Properties
# ===================================================================

class TestProperties:
    """Test ``PyEnv`` computed properties."""

    def test_is_current_true(self, env):
        mod.CURRENT_PYENV = env
        assert env.is_current is True

    def test_is_current_false(self, env):
        assert env.is_current is False

    def test_is_windows(self, env):
        assert env.is_windows == (os.name == "nt")

    def test_bin_path(self, env):
        assert env.bin_path == env.python_path.parent

    def test_root_path(self, env):
        assert env.root_path == env.python_path.parent.parent

    def test_version_info_cached(self, env):
        env._version_info = DummyVersionInfo(3, 11, 9)
        assert env.version_info.patch == 9

    def test_version_info_subprocess(self, monkeypatch, env):
        monkeypatch.setattr(
            mod.subprocess, "run",
            lambda *a, **kw: SimpleNamespace(stdout=json.dumps([3, 12, 5])),
        )
        monkeypatch.setattr(mod, "VersionInfo", DummyVersionInfo)
        vi = env.version_info
        assert (vi.major, vi.minor, vi.patch) == (3, 12, 5)

    def test_version_info_wraps_error(self, monkeypatch, env):
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(ValueError("boom")))
        with pytest.raises(RuntimeError, match="Failed to get version info"):
            _ = env.version_info


# ===================================================================
# uv resolution
# ===================================================================

class TestUvResolution:
    """Test ``has_uv``, ``ensure_uv``, ``uv_path``, ``_uv_base_cmd``."""

    def test_has_uv_local_binary(self, env):
        uv = env.bin_path / ("uv.exe" if env.is_windows else "uv")
        uv.write_text("", encoding="utf-8")
        assert env.has_uv() is True

    def test_has_uv_on_path(self, monkeypatch, env):
        monkeypatch.setattr(mod.shutil, "which", lambda n: "/usr/bin/uv" if n == "uv" else None)
        assert env.has_uv() is True

    def test_has_uv_python_module(self, monkeypatch, env):
        monkeypatch.setattr(mod.shutil, "which", lambda n: None)
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
        assert env.has_uv() is True

    def test_has_uv_all_fail(self, monkeypatch, env):
        monkeypatch.setattr(mod.shutil, "which", lambda n: None)
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no")))
        assert env.has_uv() is False

    def test_ensure_uv_cached(self, env):
        env._uv_bin = env.python_path
        assert env.ensure_uv() == env.python_path

    def test_ensure_uv_local_binary(self, env):
        uv = env.bin_path / ("uv.exe" if env.is_windows else "uv")
        uv_bin = uv.resolve()
        uv.write_text("", encoding="utf-8")
        assert env.ensure_uv() == uv_bin
        assert env._uv_bin == uv_bin

    def test_ensure_uv_path_binary(self, monkeypatch, env):
        monkeypatch.setattr(mod.shutil, "which", lambda n: "/usr/bin/uv" if n == "uv" else None)
        assert env.ensure_uv() == Path("/usr/bin/uv").resolve()

    def test_ensure_uv_python_module(self, monkeypatch, env):
        monkeypatch.setattr(mod.shutil, "which", lambda n: None)
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
        assert env.ensure_uv() == env.python_path

    def test_ensure_uv_no_install_returns_none(self, monkeypatch, env):
        monkeypatch.setattr(mod.shutil, "which", lambda n: None)
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no")))
        assert env.ensure_uv(install_runtime=False) is None

    def test_uv_path_raises_when_unavailable(self, monkeypatch, env):
        monkeypatch.setattr(env, "ensure_uv", lambda install_runtime=True: None)
        with pytest.raises(FileNotFoundError):
            _ = env.uv_path

    def test_uv_base_cmd_binary(self, monkeypatch, env):
        uv = Path("/usr/bin/uv")
        monkeypatch.setattr(env, "ensure_uv", lambda install_runtime=True: uv)
        assert env._uv_base_cmd() == [str(uv)]

    def test_uv_base_cmd_module(self, monkeypatch, env):
        monkeypatch.setattr(env, "ensure_uv", lambda install_runtime=True: env.python_path)
        assert env._uv_base_cmd() == [str(env.python_path), "-m", "uv"]

    def test_pip_cmd_prefers_uv(self, monkeypatch, env):
        monkeypatch.setattr(env, "_uv_base_cmd", lambda install_runtime=True: ["uv"])
        assert env._pip_cmd_args() == ["uv", "pip", "--python", str(env.python_path)]

    def test_pip_cmd_falls_back(self, monkeypatch, env):
        monkeypatch.setattr(env, "_uv_base_cmd", lambda **kw: (_ for _ in ()).throw(RuntimeError()))
        assert env._pip_cmd_args() == [str(env.python_path), "-m", "pip"]

    def test_uv_run_prefix(self, monkeypatch, env):
        monkeypatch.setattr(env, "_uv_base_cmd", lambda install_runtime=True: ["uv"])
        assert env._uv_run_prefix() == ["uv", "run", "--python", str(env.python_path)]


# ===================================================================
# Package management
# ===================================================================

class TestPackageManagement:
    """Test ``requirements``, ``install``, ``update``, ``uninstall``, ``pip``."""

    def test_requirements_filters_system(self, monkeypatch, env):
        monkeypatch.setattr(env, "_pip_cmd_args", lambda prefer_uv=None: ["python", "-m", "pip"])
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: SimpleNamespace(
            stdout=json.dumps([
                {"name": "pip", "version": "24.0"},
                {"name": "pyarrow", "version": "19.0.0"},
                {"name": "test-helper", "version": "1.0"},
                {"name": "PyWin32", "version": "999"},
            ])
        ))
        out = env.requirements(with_system=False)
        assert out == [("pyarrow", "19.0.0")]

    def test_requirements_with_system_keeps_all(self, monkeypatch, env):
        monkeypatch.setattr(env, "_pip_cmd_args", lambda prefer_uv=None: ["python", "-m", "pip"])
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: SimpleNamespace(
            stdout=json.dumps([
                {"name": "pip", "version": "24.0"},
                {"name": "pyarrow", "version": "19.0.0"},
            ])
        ))
        out = env.requirements(with_system=True)
        assert out == [("pip", "24.0"), ("pyarrow", "19.0.0")]

    def test_requirements_bad_json_shape(self, monkeypatch, env):
        monkeypatch.setattr(env, "_pip_cmd_args", lambda prefer_uv=None: ["python", "-m", "pip"])
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: SimpleNamespace(
            stdout=json.dumps({"nope": 1})
        ))
        with pytest.raises(ValueError, match="Unexpected pip output"):
            env.requirements()

    def test_install_returns_none_when_no_inputs(self, env):
        assert env.install() is None

    def test_install_packages_success(self, dummy_system_command, env):
        env.prefer_uv = False
        out = env.install(
            "yaml", "pyarrow",
            extra_args=["--upgrade-strategy", "eager"],
            target=env.cwd / "target",
        )
        assert out is dummy_system_command.next_result
        cmd = dummy_system_command.calls[0]["cmd"]
        assert cmd[:4] == [str(env.python_path), "-m", "pip", "install"]
        assert "PyYAML" in cmd
        assert "pyarrow" in cmd
        assert "--target" in cmd

    def test_install_inline_requirements_creates_temp_file(self, dummy_system_command, env):
        env.prefer_uv = False
        env.install(requirements="pyarrow==19.0.0\npandas==2.2.3")
        cmd = dummy_system_command.calls[0]["cmd"]
        assert "-r" in cmd

    def test_install_fallback_to_internal_pip(self, monkeypatch, env):
        env.prefer_uv = False
        monkeypatch.setattr(mod.SystemCommand, "run_lazy", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()))
        monkeypatch.setattr(env, "_is_current_interpreter", lambda: True)

        calls = {}
        monkeypatch.setattr(env, "_run_pip_internal", lambda *a: calls.update(args=a))
        env.install("yaml", wait=True, raise_error=True)
        assert calls["args"] == ("install", "PyYAML")

    def test_install_returns_none_when_raise_error_false(self, monkeypatch, env):
        env.prefer_uv = False
        monkeypatch.setattr(mod.SystemCommand, "run_lazy", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()))
        assert env.install("pyarrow", raise_error=False) is None

    def test_update_success(self, dummy_system_command, env):
        env.prefer_uv = False
        env.update("yaml")
        cmd = dummy_system_command.calls[0]["cmd"]
        assert "--upgrade" in cmd
        assert "PyYAML" in cmd

    def test_update_no_packages_returns_none(self, env):
        assert env.update() is None

    def test_update_fallback_to_internal_pip(self, monkeypatch, env):
        env.prefer_uv = False
        monkeypatch.setattr(mod.SystemCommand, "run_lazy", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()))
        monkeypatch.setattr(env, "_is_current_interpreter", lambda: True)
        calls = {}
        monkeypatch.setattr(env, "_run_pip_internal", lambda *a: calls.update(args=a))
        env.update("yaml", wait=True)
        assert calls["args"] == ("install", "--upgrade", "PyYAML")

    def test_uninstall_builds_command(self, dummy_system_command, env):
        env.prefer_uv = False
        env.uninstall("yaml", extra_args=["-y"])
        cmd = dummy_system_command.calls[0]["cmd"]
        assert cmd[:4] == [str(env.python_path), "-m", "pip", "uninstall"]
        assert "PyYAML" in cmd
        assert "-y" in cmd

    def test_uninstall_no_packages_returns_none(self, env):
        assert env.uninstall() is None

    def test_pip_arbitrary_command(self, dummy_system_command, env):
        env.prefer_uv = False
        env.pip("show", "pyarrow")
        cmd = dummy_system_command.calls[0]["cmd"]
        assert cmd == [str(env.python_path), "-m", "pip", "show", "pyarrow"]


# ===================================================================
# Delete
# ===================================================================

class TestDelete:
    """Test ``PyEnv.delete``."""

    def test_raises_for_current_singleton(self, env):
        mod.CURRENT_PYENV = env
        with pytest.raises(ValueError, match="Cannot delete the current singleton"):
            env.delete()

    def test_removes_venv_root(self, monkeypatch, tmp_path):
        root = tmp_path / "venv"
        py = root / "bin" / "python"
        cfg = root / "pyvenv.cfg"
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")
        cfg.write_text("", encoding="utf-8")

        env = PyEnv(py, cwd=tmp_path)
        removed = {}
        monkeypatch.setattr(mod.shutil, "rmtree", lambda p, ignore_errors=False: removed.update(path=p))
        env.delete()
        assert removed["path"] == root

    def test_raises_when_no_pyvenv_cfg(self, tmp_path):
        py = tmp_path / "x" / "y" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("", encoding="utf-8")
        env = PyEnv(py, cwd=tmp_path)
        with pytest.raises(ValueError, match="Cannot determine venv root"):
            env.delete()


# ===================================================================
# Execution
# ===================================================================

class TestExecution:
    """Test ``run_python_code``."""

    def test_with_uv_and_globs(self, dummy_system_command, monkeypatch, env):
        monkeypatch.setattr(env, "_uv_run_prefix", lambda: ["uv", "run", "--python", str(env.python_path)])
        env.run_python_code("print(x + 1)", globs={"x": 41}, env={"A": "1"})
        call = dummy_system_command.calls[0]
        assert call["cmd"][:4] == ["uv", "run", "--python", str(env.python_path)]
        assert "x = 41" in call["cmd"][6]
        assert "print(x + 1)" in call["cmd"][6]
        assert call["env"]["A"] == "1"

    def test_fallback_when_uv_unavailable(self, dummy_system_command, monkeypatch, env):
        monkeypatch.setattr(env, "_uv_run_prefix", lambda: (_ for _ in ()).throw(RuntimeError()))
        env.run_python_code("print('hi')")
        assert dummy_system_command.calls[0]["cmd"] == [str(env.python_path), "-c", "print('hi')"]

    def test_writes_stdin(self, dummy_system_command, env):
        result = env.run_python_code("print(input())", stdin="hello\n")
        assert result.popen.stdin.buffer == ["hello\n"]
        assert result.popen.stdin.closed is True

    def test_installs_packages_before_run(self, monkeypatch, dummy_system_command, env):
        calls = []
        monkeypatch.setattr(env, "install", lambda *p: calls.append(p))
        monkeypatch.setattr(env, "_uv_run_prefix", lambda: ["uv", "run", "--python", str(env.python_path)])
        env.run_python_code("print('x')", packages=["pyarrow"])
        assert calls == [("pyarrow",)]

    def test_prefer_uv_false_uses_bare_python(self, dummy_system_command, env):
        env.prefer_uv = False
        env.run_python_code("1+1")
        assert dummy_system_command.calls[0]["cmd"][:2] == [str(env.python_path), "-c"]


# ===================================================================
# Import module
# ===================================================================

class TestImportModule:
    """Test ``import_module`` and ``runtime_import_module``."""

    def test_direct_import(self, monkeypatch, env):
        sentinel = object()
        monkeypatch.setattr(importlib, "import_module", lambda n: sentinel)
        assert env.import_module("json", install=False) is sentinel

    def test_missing_raises_when_install_false(self, monkeypatch, env):
        monkeypatch.setattr(importlib, "import_module", lambda n: (_ for _ in ()).throw(ModuleNotFoundError(name=n)))
        with pytest.raises(ModuleNotFoundError):
            env.import_module("missing_pkg", install=False)

    def test_auto_installs(self, monkeypatch, env):
        call_count = {"n": 0}

        def fake_import(name):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ModuleNotFoundError(name=name)
            return {"module": name}

        install_calls = []
        monkeypatch.setattr(importlib, "import_module", fake_import)
        monkeypatch.setattr(importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(env, "install", lambda *a, **kw: install_calls.append(a))
        out = env.import_module("yaml")
        assert out == {"module": "yaml"}
        assert install_calls[0] == ("PyYAML",)

    def test_wraps_failure(self, monkeypatch, env):
        monkeypatch.setattr(importlib, "import_module", lambda n: (_ for _ in ()).throw(ModuleNotFoundError(name=n)))
        monkeypatch.setattr(env, "install", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")))
        with pytest.raises(ModuleNotFoundError, match="Failed to import module"):
            env.import_module("yaml")

    def test_pip_name_override(self, monkeypatch, env):
        install_calls = []

        def fake_import(name):
            if install_calls:
                return {"module": name}
            raise ModuleNotFoundError(name=name)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        monkeypatch.setattr(importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(env, "install", lambda *a, **kw: install_calls.append(a))
        env.import_module("my_mod", pip_name="my-distribution")
        assert install_calls[0] == ("my-distribution",)

    def test_use_cache(self, monkeypatch, env):
        monkeypatch.setattr(importlib, "import_module", lambda n: object())
        env.import_module("json", use_cache=True)
        assert "json" in env._checked_modules

    def test_module_name_derived_from_pip_name(self, monkeypatch, env):
        sentinel = object()
        monkeypatch.setattr(importlib, "import_module", lambda n: sentinel if n == "my_pkg" else None)
        out = env.import_module(pip_name="my-pkg", install=False)
        assert out is sentinel

    def test_no_name_raises(self, env):
        with pytest.raises(ValueError, match="Provide at least one"):
            env.import_module()


class TestRuntimeImportModule:
    """Test the module-level ``runtime_import_module`` function."""

    def test_delegates_to_current(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(
            PyEnv, "current",
            classmethod(lambda cls: SimpleNamespace(import_module=lambda **kw: sentinel)),
        )
        assert mod.runtime_import_module("yaml") is sentinel


# ===================================================================
# get_root_module_directory
# ===================================================================

class TestGetRootModuleDirectory:
    """Test ``PyEnv.get_root_module_directory``."""

    def test_package(self, monkeypatch, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        spec = SimpleNamespace(submodule_search_locations=[str(pkg)], origin=None)
        monkeypatch.setattr(importlib.util, "find_spec", lambda n: spec)
        assert PyEnv.get_root_module_directory("mypkg.sub") == pkg.resolve()

    def test_module_file(self, monkeypatch, tmp_path):
        f = tmp_path / "single.py"
        f.write_text("", encoding="utf-8")
        spec = SimpleNamespace(submodule_search_locations=None, origin=str(f))
        monkeypatch.setattr(importlib.util, "find_spec", lambda n: spec)
        assert PyEnv.get_root_module_directory("single") == tmp_path.resolve()

    def test_missing_raises(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", lambda n: None)
        with pytest.raises(ModuleNotFoundError):
            PyEnv.get_root_module_directory("missing_pkg")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            PyEnv.get_root_module_directory("")

    def test_builtin_raises(self, monkeypatch):
        spec = SimpleNamespace(submodule_search_locations=None, origin="built-in")
        monkeypatch.setattr(importlib.util, "find_spec", lambda n: spec)
        with pytest.raises(FileNotFoundError, match="no filesystem directory"):
            PyEnv.get_root_module_directory("builtins")


# ===================================================================
# Helpers
# ===================================================================

class TestHelpers:
    """Test small internal helpers."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("~/env", True),
            ("./env", True),
            ("/tmp/env", True),
            ("envs/myenv", True),
            ("envs\\myenv", True),
            ("myenv", False),
            ("3.12", False),
            ("python3.12", False),
            ("", False),
        ],
    )
    def test_looks_like_path(self, value, expected):
        assert PyEnv._looks_like_path(value) is expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("externally-managed-environment", True),
            ("error: externally managed environment detected", True),
            ("This environment is externally managed", True),
            ("some other error", False),
            ("", False),
        ],
    )
    def test_is_externally_managed_failure(self, text, expected):
        exc = RuntimeError(text)
        assert PyEnv._is_externally_managed_failure(exc) is expected

