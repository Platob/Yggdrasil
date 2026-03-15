from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import yggdrasil.environ.environment as mod
from yggdrasil.environ.environment import PyEnv, safe_pip_name


class DummyVersionInfo:
    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class DummyWaitingConfig:
    @staticmethod
    def check_arg(value):
        return bool(value)


class DummyLazyResult:
    def __init__(self):
        self.wait_calls = []

    def wait(self, wait=True, raise_error=True, auto_install=False):
        self.wait_calls.append(
            {
                "wait": wait,
                "raise_error": raise_error,
                "auto_install": auto_install,
            }
        )
        return self


class DummyProc:
    def __init__(self):
        self.stdin = DummyStdin()


class DummyStdin:
    def __init__(self):
        self.buffer = []
        self.closed = False

    def write(self, value):
        self.buffer.append(value)

    def flush(self):
        return None

    def close(self):
        self.closed = True


class DummySystemCommandModule:
    def __init__(self):
        self.calls = []
        self.next_result = DummyLazyResult()

    def run_lazy(self, cmd, cwd=None, env=None, python=None):
        self.calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "env": env,
                "python": python,
            }
        )
        result = self.next_result
        result.popen = DummyProc()
        return result


@pytest.fixture(autouse=True)
def reset_current_pyenv(monkeypatch):
    monkeypatch.setattr(mod, "CURRENT_PYENV", None)
    yield
    monkeypatch.setattr(mod, "CURRENT_PYENV", None)


@pytest.fixture
def dummy_system_command(monkeypatch):
    fake = DummySystemCommandModule()
    monkeypatch.setattr(mod, "SystemCommand", fake)
    monkeypatch.setattr(mod, "WaitingConfig", DummyWaitingConfig)
    monkeypatch.setattr(mod, "VersionInfo", DummyVersionInfo)
    return fake


@pytest.fixture
def env(tmp_path):
    py = tmp_path / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("", encoding="utf-8")
    return PyEnv(python_path=py, cwd=tmp_path, prefer_uv=True)


def test_safe_pip_name_string_mapping():
    assert safe_pip_name("yaml") == "PyYAML"
    assert safe_pip_name("pyarrow") == "pyarrow"


def test_safe_pip_name_tuple_version():
    assert safe_pip_name(("yaml", "6.0.2")) == "PyYAML==6.0.2"


def test_safe_pip_name_iterable_mixed():
    result = safe_pip_name(["yaml", ("jwt", "2.9.0"), "pyarrow"])
    assert result == ["PyYAML", "PyJWT==2.9.0", "pyarrow"]


def test_post_init_resolves_paths(tmp_path):
    py = tmp_path / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("", encoding="utf-8")

    obj = PyEnv(python_path=py, cwd=tmp_path / ".")
    assert obj.python_path.is_absolute()
    assert obj.cwd.is_absolute()


def test_getstate_setstate_roundtrip(tmp_path):
    py = tmp_path / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
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


def test_resolve_python_executable_none(monkeypatch):
    monkeypatch.setattr(sys, "executable", "/usr/bin/python-real")
    path = PyEnv.resolve_python_executable(None)
    assert path == Path("/usr/bin/python-real").resolve()


def test_resolve_python_executable_file(tmp_path):
    py = tmp_path / "python3.12"
    py.write_text("", encoding="utf-8")
    assert PyEnv.resolve_python_executable(py) == py.resolve()


def test_resolve_python_executable_dir(monkeypatch, tmp_path):
    venv_dir = tmp_path / "venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    py = bin_dir / "python"
    py.write_text("", encoding="utf-8")

    assert PyEnv.resolve_python_executable(venv_dir) == py.resolve()


def test_resolve_python_executable_which(monkeypatch):
    monkeypatch.setattr(mod.shutil, "which", lambda s: "/usr/bin/python3.12" if s == "python3.12" else None)
    assert PyEnv.resolve_python_executable("3.12") == Path("/usr/bin/python3.12").resolve()


def test_resolve_python_executable_not_found(monkeypatch):
    monkeypatch.setattr(mod.shutil, "which", lambda s: None)
    with pytest.raises(FileNotFoundError):
        PyEnv.resolve_python_executable("3.99")


def test_find_python_in_dir_prefers_standard_layout(tmp_path):
    root = tmp_path / "env"
    py = root / "bin" / "python3"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")

    assert PyEnv._find_python_in_dir(root) == py.resolve()


def test_find_python_in_dir_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        PyEnv._find_python_in_dir(tmp_path / "missing")


def test_current_singleton(monkeypatch, tmp_path):
    py = tmp_path / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")

    monkeypatch.setattr(PyEnv, "resolve_python_executable", staticmethod(lambda python=None: py))
    env1 = PyEnv.current(prefer_uv=False)
    env2 = PyEnv.current(prefer_uv=True)

    assert env1 is env2
    assert mod.CURRENT_PYENV is env1


def test_current_sets_version_info_for_current_interpreter(monkeypatch, tmp_path):
    py = Path(sys.executable).resolve()
    monkeypatch.setattr(PyEnv, "resolve_python_executable", staticmethod(lambda python=None: py))
    monkeypatch.setattr(mod, "VersionInfo", DummyVersionInfo)

    env = PyEnv.current()
    assert env._version_info is not None
    assert env._version_info.major == sys.version_info.major


def test_get_or_create_with_pyenv_identifier_installs(monkeypatch, env):
    called = {}

    def fake_install(*packages):
        called["packages"] = packages

    monkeypatch.setattr(env, "install", fake_install)
    out = PyEnv.get_or_create(env, packages=["pyarrow", "pandas"])

    assert out is env
    assert called["packages"] == ("pyarrow", "pandas")


def test_venv_current_alias_returns_current(monkeypatch, env):
    current_env = object.__new__(PyEnv)
    monkeypatch.setattr(PyEnv, "current", classmethod(lambda cls, prefer_uv=True: current_env))

    out = env.venv("current")
    assert out is current_env


def test_venv_existing_python_file(monkeypatch, env, tmp_path):
    py = tmp_path / "some" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")

    calls = {}

    def fake_instance(python_path, cwd=None, prefer_uv=True, packages=None):
        calls["python_path"] = python_path
        calls["cwd"] = cwd
        calls["prefer_uv"] = prefer_uv
        return "ENV"

    monkeypatch.setattr(PyEnv, "instance", classmethod(lambda cls, python_path, cwd=None, prefer_uv=True, packages=None: fake_instance(python_path, cwd, prefer_uv, packages)))

    out = env.venv(py)
    assert out == "ENV"
    assert Path(calls["python_path"]) == py.resolve()


def test_venv_existing_dir_uses_venv_python(monkeypatch, env, tmp_path):
    folder = tmp_path / "myenv"
    folder.mkdir()

    py = folder / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("", encoding="utf-8")

    monkeypatch.setattr(PyEnv, "_venv_python_from_dir", staticmethod(lambda p, raise_error=False: py))

    monkeypatch.setattr(
        PyEnv,
        "instance",
        classmethod(
            lambda cls, python_path, cwd=None, prefer_uv=True, packages=None: (
                "ENV",
                python_path,
                cwd,
                prefer_uv,
            )
        ),
    )

    out = env.venv(folder)
    assert out[0] == "ENV"
    assert out[1] == py


def test_venv_missing_creates(monkeypatch, env, tmp_path):
    folder = tmp_path / "newenv"
    called = {}

    monkeypatch.setattr(PyEnv, "_venv_python_from_dir", staticmethod(lambda p, raise_error=False: p / "bin" / "python"))

    def fake_create(path, **kwargs):
        called["path"] = path
        called["kwargs"] = kwargs
        return "CREATED"

    monkeypatch.setattr(env, "create", fake_create)

    out = env.venv(folder)
    assert out == "CREATED"
    assert Path(called["path"]) == folder


def test_create_builds_uv_command_and_installs_packages(monkeypatch, env, tmp_path, dummy_system_command):
    folder = tmp_path / "created-env"
    py = folder / "bin" / "python"

    monkeypatch.setattr(env, "_uv_base_cmd", lambda install_runtime=True: ["uv"])
    object.__setattr__(env, "_version_info", DummyVersionInfo(3, 12, 7))
    monkeypatch.setattr(PyEnv, "_venv_python_from_dir", staticmethod(lambda p, raise_error=True: py))

    created_env = PyEnv(python_path=py, cwd=tmp_path, prefer_uv=True)
    install_calls = []

    monkeypatch.setattr(
        PyEnv,
        "instance",
        classmethod(lambda cls, python_path, cwd=None, prefer_uv=True, packages=None: created_env),
    )
    monkeypatch.setattr(created_env, "install", lambda *packages, wait=True: install_calls.append((packages, wait)))

    out = env.create(folder, packages=["pyarrow"], wait=True)

    assert out is created_env
    cmd = dummy_system_command.calls[0]["cmd"]
    assert cmd[:2] == ["uv", "venv"]
    assert str(folder) in cmd
    assert "--python" in cmd
    assert "3.12.7" in cmd
    assert "--seed" in cmd
    assert "--native-tls" in cmd
    assert "--clear" in cmd
    assert install_calls == [(("pyarrow",), True)]


def test_has_uv_local_binary(monkeypatch, env, tmp_path):
    uv_bin = env.bin_path / ("uv.exe" if env.is_windows else "uv")
    uv_bin.write_text("", encoding="utf-8")
    assert env.has_uv() is True


def test_has_uv_on_path(monkeypatch, env):
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    assert env.has_uv() is True


def test_has_uv_python_module(monkeypatch, env):
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    assert env.has_uv() is True


def test_has_uv_false_when_all_fail(monkeypatch, env):
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)

    def fail(*args, **kwargs):
        raise RuntimeError("no uv")

    monkeypatch.setattr(mod.subprocess, "run", fail)
    assert env.has_uv() is False


def test_ensure_uv_returns_cached(env):
    env._uv_bin = env.python_path
    assert env.ensure_uv() == env.python_path


def test_ensure_uv_prefers_local_binary(monkeypatch, env):
    uv_bin = env.bin_path / ("uv.exe" if env.is_windows else "uv")
    uv_bin.write_text("", encoding="utf-8")

    out = env.ensure_uv()
    assert out == uv_bin.resolve()
    assert env._uv_bin == uv_bin.resolve()


def test_ensure_uv_returns_path_binary(monkeypatch, env):
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    out = env.ensure_uv()
    assert out == Path("/usr/bin/uv").resolve()


def test_ensure_uv_returns_python_when_module_available(monkeypatch, env):
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    out = env.ensure_uv()
    assert out == env.python_path


def test_ensure_uv_no_install_runtime_returns_none(monkeypatch, env):
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)

    def fail(*args, **kwargs):
        raise RuntimeError("missing")

    monkeypatch.setattr(mod.subprocess, "run", fail)
    assert env.ensure_uv(install_runtime=False) is None


def test_uv_path_raises_when_unavailable(monkeypatch, env):
    monkeypatch.setattr(env, "ensure_uv", lambda install_runtime=True: None)
    with pytest.raises(FileNotFoundError):
        _ = env.uv_path


def test_uv_base_cmd_binary(monkeypatch, env):
    uv_path = Path("/usr/bin/uv")
    monkeypatch.setattr(env, "ensure_uv", lambda install_runtime=True: uv_path)
    assert env._uv_base_cmd() == [str(uv_path)]


def test_uv_base_cmd_module(monkeypatch, env):
    monkeypatch.setattr(env, "ensure_uv", lambda install_runtime=True: env.python_path)
    assert env._uv_base_cmd() == [str(env.python_path), "-m", "uv"]


def test_version_info_cached(env):
    env._version_info = DummyVersionInfo(3, 11, 9)
    assert env.version_info.patch == 9


def test_version_info_runs_subprocess(monkeypatch, env):
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps([3, 12, 5])),
    )
    monkeypatch.setattr(mod, "VersionInfo", DummyVersionInfo)

    out = env.version_info
    assert (out.major, out.minor, out.patch) == (3, 12, 5)


def test_version_info_wraps_error(monkeypatch, env):
    def fail(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(mod.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Failed to get version info"):
        _ = env.version_info


def test_pip_cmd_args_prefers_uv(monkeypatch, env):
    monkeypatch.setattr(env, "_uv_base_cmd", lambda install_runtime=True: ["uv"])
    assert env._pip_cmd_args() == ["uv", "pip", "--python", str(env.python_path)]


def test_pip_cmd_args_falls_back_to_python_pip(monkeypatch, env):
    def fail(*args, **kwargs):
        raise RuntimeError("no uv")

    monkeypatch.setattr(env, "_uv_base_cmd", fail)
    assert env._pip_cmd_args() == [str(env.python_path), "-m", "pip"]


def test_uv_run_prefix(monkeypatch, env):
    monkeypatch.setattr(env, "_uv_base_cmd", lambda install_runtime=True: ["uv"])
    assert env._uv_run_prefix() == ["uv", "run", "--python", str(env.python_path)]


def test_requirements_filters_system_packages(monkeypatch, env):
    monkeypatch.setattr(env, "_pip_cmd_args", lambda prefer_uv=None: ["python", "-m", "pip"])
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(
                [
                    {"name": "pip", "version": "24.0"},
                    {"name": "pyarrow", "version": "19.0.0"},
                    {"name": "test-helper", "version": "1.0"},
                    {"name": "PyWin32", "version": "999"},
                ]
            )
        ),
    )

    out = env.requirements(with_system=False)
    assert out == [("pyarrow", "19.0.0")]


def test_requirements_with_system_keeps_all(monkeypatch, env):
    monkeypatch.setattr(env, "_pip_cmd_args", lambda prefer_uv=None: ["python", "-m", "pip"])
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(
                [
                    {"name": "pip", "version": "24.0"},
                    {"name": "pyarrow", "version": "19.0.0"},
                ]
            )
        ),
    )

    out = env.requirements(with_system=True)
    assert out == [("pip", "24.0"), ("pyarrow", "19.0.0")]


def test_requirements_bad_json_shape(monkeypatch, env):
    monkeypatch.setattr(env, "_pip_cmd_args", lambda prefer_uv=None: ["python", "-m", "pip"])
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps({"nope": 1})),
    )
    with pytest.raises(ValueError, match="Unexpected pip output"):
        env.requirements()


def test_install_returns_none_when_no_inputs(env):
    assert env.install() is None


def test_install_packages_success(dummy_system_command, env):
    env.prefer_uv = False
    out = env.install("yaml", "pyarrow", extra_args=["--upgrade-strategy", "eager"], target=env.cwd / "target")
    assert out is dummy_system_command.next_result

    call = dummy_system_command.calls[0]
    assert call["cmd"][:4] == [str(env.python_path), "-m", "pip", "install"]
    assert "PyYAML" in call["cmd"]
    assert "pyarrow" in call["cmd"]
    assert "--upgrade-strategy" in call["cmd"]
    assert "eager" in call["cmd"]
    assert "--target" in call["cmd"]


def test_install_with_inline_requirements_creates_temp_file(monkeypatch, dummy_system_command, env):
    env.prefer_uv = False

    out = env.install(requirements="pyarrow==19.0.0\npandas==2.2.3")
    assert out is dummy_system_command.next_result

    cmd = dummy_system_command.calls[0]["cmd"]
    assert "-r" in cmd
    req_file = Path(cmd[cmd.index("-r") + 1])

    # install() should remove temp file after synchronous wait
    assert not req_file.exists()


def test_install_fallback_to_internal_pip(monkeypatch, env):
    env.prefer_uv = False

    class FailingRunLazy:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("subprocess pip failed")

    monkeypatch.setattr(mod.SystemCommand, "run_lazy", FailingRunLazy())
    monkeypatch.setattr(env, "_is_current_interpreter", lambda: True)

    calls = {}

    def fake_run_pip_internal(*args):
        calls["args"] = args

    monkeypatch.setattr(env, "_run_pip_internal", fake_run_pip_internal)

    out = env.install("yaml", wait=True, raise_error=True)
    assert out is None
    assert calls["args"] == ("install", "PyYAML")


def test_install_returns_none_when_raise_error_false(monkeypatch, env):
    env.prefer_uv = False

    class FailingRunLazy:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("subprocess pip failed")

    monkeypatch.setattr(mod.SystemCommand, "run_lazy", FailingRunLazy())

    out = env.install("pyarrow", raise_error=False)
    assert out is None


def test_update_success(dummy_system_command, env):
    env.prefer_uv = False
    out = env.update("yaml")
    assert out is dummy_system_command.next_result

    cmd = dummy_system_command.calls[0]["cmd"]
    assert cmd[:4] == [str(env.python_path), "-m", "pip", "install"]
    assert "--upgrade" in cmd
    assert "PyYAML" in cmd


def test_update_falls_back_to_internal_pip(monkeypatch, env):
    env.prefer_uv = False

    class FailingRunLazy:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(mod.SystemCommand, "run_lazy", FailingRunLazy())
    monkeypatch.setattr(env, "_is_current_interpreter", lambda: True)

    calls = {}

    def fake_run_pip_internal(*args):
        calls["args"] = args

    monkeypatch.setattr(env, "_run_pip_internal", fake_run_pip_internal)

    out = env.update("yaml", wait=True)
    assert out is None
    assert calls["args"] == ("install", "--upgrade", "PyYAML")


def test_uninstall_builds_command(dummy_system_command, env):
    env.prefer_uv = False
    out = env.uninstall("yaml", extra_args=["-y"])
    assert out is dummy_system_command.next_result

    cmd = dummy_system_command.calls[0]["cmd"]
    assert cmd[:4] == [str(env.python_path), "-m", "pip", "uninstall"]
    assert "PyYAML" in cmd
    assert "-y" in cmd


def test_pip_arbitrary_command(dummy_system_command, env):
    env.prefer_uv = False
    env.pip("show", "pyarrow")
    cmd = dummy_system_command.calls[0]["cmd"]
    assert cmd == [str(env.python_path), "-m", "pip", "show", "pyarrow"]


def test_delete_raises_for_current(env):
    mod.CURRENT_PYENV = env
    with pytest.raises(ValueError, match="Cannot delete the current singleton"):
        env.delete()


def test_delete_removes_venv_root(monkeypatch, tmp_path):
    root = tmp_path / "venv"
    py = root / "bin" / "python"
    cfg = root / "pyvenv.cfg"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    cfg.write_text("", encoding="utf-8")

    env = PyEnv(py, cwd=tmp_path)

    removed = {}

    def fake_rmtree(path, ignore_errors=False):
        removed["path"] = path
        removed["ignore_errors"] = ignore_errors

    monkeypatch.setattr(mod.shutil, "rmtree", fake_rmtree)

    env.delete()
    assert removed["path"] == root
    assert removed["ignore_errors"] is False


def test_delete_raises_when_no_pyvenv_cfg(monkeypatch, tmp_path):
    py = tmp_path / "x" / "y" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")

    env = PyEnv(py, cwd=tmp_path)
    with pytest.raises(ValueError, match="Cannot determine venv root"):
        env.delete()


def test_run_python_code_with_uv_and_globs(dummy_system_command, monkeypatch, env):
    monkeypatch.setattr(env, "_uv_run_prefix", lambda: ["uv", "run", "--python", str(env.python_path)])

    env.run_python_code("print(x + 1)", globs={"x": 41}, env={"A": "1"})
    call = dummy_system_command.calls[0]

    assert call["cmd"][:4] == ["uv", "run", "--python", str(env.python_path)]
    assert call["cmd"][4:6] == ["python", "-c"]
    assert "x = 41" in call["cmd"][6]
    assert "print(x + 1)" in call["cmd"][6]
    assert call["env"]["A"] == "1"
    assert call["python"] is env


def test_run_python_code_falls_back_when_uv_unavailable(dummy_system_command, monkeypatch, env):
    def fail():
        raise RuntimeError("uv nope")

    monkeypatch.setattr(env, "_uv_run_prefix", fail)
    env.run_python_code("print('hi')")

    cmd = dummy_system_command.calls[0]["cmd"]
    assert cmd == [str(env.python_path), "-c", "print('hi')"]


def test_run_python_code_writes_stdin(dummy_system_command, env):
    result = env.run_python_code("print(input())", stdin="hello\n")
    stdin_obj = result.popen.stdin
    assert stdin_obj.buffer == ["hello\n"]
    assert stdin_obj.closed is True


def test_run_python_code_installs_packages(monkeypatch, dummy_system_command, env):
    calls = []

    monkeypatch.setattr(env, "install", lambda *packages: calls.append(packages))
    monkeypatch.setattr(env, "_uv_run_prefix", lambda: ["uv", "run", "--python", str(env.python_path)])

    env.run_python_code("print('x')", packages=["pyarrow", "pandas"])
    assert calls == [("pyarrow", "pandas")]


def test_import_module_imports_directly_when_present(monkeypatch, env):
    sentry = object()
    monkeypatch.setattr(importlib, "import_module", lambda name: sentry)

    out = env.import_module("json", install=False)
    assert out is sentry


def test_import_module_raises_when_missing_and_install_false(monkeypatch, env):
    def fake_import(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(ModuleNotFoundError):
        env.import_module("definitely_missing_pkg", install=False)


def test_import_module_auto_installs(monkeypatch, env):
    calls = {"count": 0}

    def fake_import(name):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ModuleNotFoundError(name=name)
        return {"module": name}

    install_calls = []

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(env, "install", lambda *args, **kwargs: install_calls.append((args, kwargs)))

    out = env.import_module("yaml")
    assert out == {"module": "yaml"}
    assert install_calls[0][0] == ("PyYAML",)


def test_import_module_wraps_failure(monkeypatch, env):
    monkeypatch.setattr(importlib, "import_module", lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=name)))
    monkeypatch.setattr(env, "install", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("install failed")))

    with pytest.raises(ModuleNotFoundError, match="Failed to import module"):
        env.import_module("yaml")


def test_runtime_import_module_delegates(monkeypatch):
    sentry = object()

    monkeypatch.setattr(
        PyEnv,
        "current",
        classmethod(lambda cls: SimpleNamespace(import_module=lambda **kwargs: sentry)),
    )

    assert mod.runtime_import_module("yaml") is sentry


def test_get_root_module_directory_package(monkeypatch, tmp_path):
    package_dir = tmp_path / "mypkg"
    package_dir.mkdir()

    spec = SimpleNamespace(
        submodule_search_locations=[str(package_dir)],
        origin=None,
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)

    assert PyEnv.get_root_module_directory("mypkg.submod") == package_dir.resolve()


def test_get_root_module_directory_module_file(monkeypatch, tmp_path):
    file_path = tmp_path / "single.py"
    file_path.write_text("", encoding="utf-8")

    spec = SimpleNamespace(
        submodule_search_locations=None,
        origin=str(file_path),
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)

    assert PyEnv.get_root_module_directory("single") == tmp_path.resolve()


def test_get_root_module_directory_raises_when_missing(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError):
        PyEnv.get_root_module_directory("missing_pkg")


def test_venv_python_from_dir_non_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(mod.os, "name", "posix")
    venv_dir = tmp_path / "venv"
    py = venv_dir / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")

    assert PyEnv._venv_python_from_dir(venv_dir) == py.resolve()


def test_venv_python_from_dir_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(mod.os, "name", "posix")
    with pytest.raises(ValueError, match="No Python executable found inside venv"):
        PyEnv._venv_python_from_dir(tmp_path / "missing")


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
def test_looks_like_path(value, expected):
    assert PyEnv._looks_like_path(value) is expected