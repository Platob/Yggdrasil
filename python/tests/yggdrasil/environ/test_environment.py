"""
test_pyenv.py — Unit tests for PyEnv.

No mocking. Every test exercises real code paths against the live interpreter,
real filesystem, and real uv/pip. Tests that need isolation use tmp_path and
real venvs created and destroyed within the test.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

import yggdrasil.environ.environment as _env_module
from yggdrasil.environ.environment import PyEnv, safe_pip_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_singleton():
    _env_module.CURRENT_PYENV = None


# ---------------------------------------------------------------------------
# TestSafePipName
# ---------------------------------------------------------------------------

class TestSafePipName:
    def test_yaml(self):
        assert safe_pip_name("yaml") == "PyYAML"

    def test_jwt(self):
        assert safe_pip_name("jwt") == "PyJWT"

    def test_dotenv(self):
        assert safe_pip_name("dotenv") == "python-dotenv"

    def test_dateutil(self):
        assert safe_pip_name("dateutil") == "python-dateutil"

    def test_yggdrasil(self):
        assert safe_pip_name("yggdrasil") == "ygg"

    def test_unknown_passes_through(self):
        assert safe_pip_name("pyarrow") == "pyarrow"

    def test_list_input(self):
        assert safe_pip_name(["yaml", "jwt", "numpy"]) == ["PyYAML", "PyJWT", "numpy"]

    def test_empty_list(self):
        assert safe_pip_name([]) == []


# ---------------------------------------------------------------------------
# TestResolveExecutable
# ---------------------------------------------------------------------------

class TestResolveExecutable:
    def test_none_returns_sys_executable(self):
        assert PyEnv.resolve_python_executable(None) == Path(sys.executable).resolve()

    def test_path_object_to_real_file(self):
        p = Path(sys.executable)
        assert PyEnv.resolve_python_executable(p) == p.resolve()

    def test_string_path_to_real_file(self):
        assert PyEnv.resolve_python_executable(sys.executable) == Path(sys.executable).resolve()

    def test_result_is_a_real_file(self):
        assert PyEnv.resolve_python_executable(None).is_file()

    def test_version_selector_resolves_via_which(self):
        # Use the current major.minor — guaranteed to be on PATH
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        result = PyEnv.resolve_python_executable(version)
        assert result.is_file()

    def test_nonexistent_name_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PyEnv.resolve_python_executable("python_does_not_exist_xyz999")

    def test_tmp_file_resolved_directly(self, tmp_path):
        py = tmp_path / "python"
        py.write_text("#!/usr/bin/env python3")
        assert PyEnv.resolve_python_executable(py) == py.resolve()


# ---------------------------------------------------------------------------
# TestLooksLikePath
# ---------------------------------------------------------------------------

class TestLooksLikePath:
    def test_tilde(self):          assert PyEnv._looks_like_path("~/envs/x") is True
    def test_dot(self):            assert PyEnv._looks_like_path("./venv") is True
    def test_abs(self):            assert PyEnv._looks_like_path("/usr/bin/python") is True
    def test_slash_in_middle(self): assert PyEnv._looks_like_path("some/path") is True
    def test_backslash(self):      assert PyEnv._looks_like_path("some\\path") is True
    def test_bare_name(self):      assert PyEnv._looks_like_path("myenv") is False
    def test_version(self):        assert PyEnv._looks_like_path("3.12") is False
    def test_empty(self):          assert PyEnv._looks_like_path("") is False


# ---------------------------------------------------------------------------
# TestVenvPythonFromDir
# ---------------------------------------------------------------------------

class TestVenvPythonFromDir:
    def test_bin_python(self, tmp_path):
        (tmp_path / "bin").mkdir()
        py = tmp_path / "bin" / "python"
        py.write_text("")
        assert PyEnv._venv_python_from_dir(tmp_path) == py.resolve()

    def test_bin_python3_fallback(self, tmp_path):
        (tmp_path / "bin").mkdir()
        py3 = tmp_path / "bin" / "python3"
        py3.write_text("")
        assert PyEnv._venv_python_from_dir(tmp_path) == py3.resolve()

    def test_scripts_python_exe(self, tmp_path):
        (tmp_path / "Scripts").mkdir()
        py = tmp_path / "Scripts" / "python.exe"
        py.write_text("")
        assert PyEnv._venv_python_from_dir(tmp_path) == py.resolve()

    def test_missing_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No Python executable found"):
            PyEnv._venv_python_from_dir(tmp_path)

    def test_bin_python_preferred_over_python3(self, tmp_path):
        (tmp_path / "bin").mkdir()
        py = tmp_path / "bin" / "python"
        py3 = tmp_path / "bin" / "python3"
        py.write_text("")
        py3.write_text("")
        assert PyEnv._venv_python_from_dir(tmp_path) == py.resolve()


# ---------------------------------------------------------------------------
# TestPyEnvCreate
# ---------------------------------------------------------------------------

class TestPyEnvCreate:
    def test_python_path_resolved(self):
        env = PyEnv.create(Path(sys.executable))
        assert env.python_path == Path(sys.executable).resolve()

    def test_prefer_uv_default_true(self):
        env = PyEnv.create(Path(sys.executable))
        assert env.prefer_uv is True

    def test_prefer_uv_false(self):
        env = PyEnv.create(Path(sys.executable), prefer_uv=False)
        assert env.prefer_uv is False

    def test_cwd_resolved(self, tmp_path):
        env = PyEnv.create(Path(sys.executable), cwd=tmp_path)
        assert env.cwd == tmp_path.resolve()

    def test_packages_triggers_real_install(self):
        # "pip" is always present — a no-op reinstall that proves the pathway works
        env = PyEnv.create(Path(sys.executable), packages=["pip"])
        reqs = {n for n, _ in env.requirements(with_system=True)}
        assert "pip" in reqs


# ---------------------------------------------------------------------------
# TestPyEnvCurrent
# ---------------------------------------------------------------------------

class TestPyEnvCurrent:
    def setup_method(self):   _reset_singleton()
    def teardown_method(self): _reset_singleton()

    def test_creates_singleton(self):
        env = PyEnv.current()
        assert _env_module.CURRENT_PYENV is env

    def test_idempotent(self):
        e1 = PyEnv.current()
        e2 = PyEnv.current()
        assert e1 is e2

    def test_points_to_sys_executable(self):
        assert PyEnv.current().python_path == Path(sys.executable).resolve()

    def test_is_current_true(self):
        assert PyEnv.current().is_current is True

    def test_other_instance_is_not_current(self):
        PyEnv.current()
        other = PyEnv(python_path=Path(sys.executable), cwd=Path("/tmp"))
        assert other.is_current is False


# ---------------------------------------------------------------------------
# TestPyEnvGetOrCreate
# ---------------------------------------------------------------------------

class TestPyEnvGetOrCreate:
    def setup_method(self):   _reset_singleton()
    def teardown_method(self): _reset_singleton()

    def test_none_returns_current(self):
        result = PyEnv.get_or_create(None)
        assert result is _env_module.CURRENT_PYENV

    def test_empty_string_returns_current(self):
        assert PyEnv.get_or_create("") is _env_module.CURRENT_PYENV

    def test_current_keyword_returns_current(self):
        assert PyEnv.get_or_create("current") is _env_module.CURRENT_PYENV

    def test_sys_keyword_returns_current(self):
        assert PyEnv.get_or_create("sys") is _env_module.CURRENT_PYENV

    def test_pyenv_instance_returned_unchanged(self):
        existing = PyEnv.current()
        assert PyEnv.get_or_create(existing) is existing

    def test_real_executable_path(self):
        env = PyEnv.get_or_create(sys.executable)
        assert env.python_path == Path(sys.executable).resolve()

    def test_packages_installed_after_resolution(self):
        # Install pip (no-op) to prove the packages arg is forwarded
        env = PyEnv.get_or_create(sys.executable, packages=["pip"])
        reqs = {n for n, _ in env.requirements(with_system=True)}
        assert "pip" in reqs


# ---------------------------------------------------------------------------
# TestPyEnvResolveEnv
# ---------------------------------------------------------------------------

class TestPyEnvResolveEnv:
    def setup_method(self):   _reset_singleton()
    def teardown_method(self): _reset_singleton()

    def test_none_returns_current(self):
        assert PyEnv.resolve_env(None) is _env_module.CURRENT_PYENV

    def test_current_keyword(self):
        assert PyEnv.resolve_env("current") is _env_module.CURRENT_PYENV

    def test_sys_executable_string(self):
        env = PyEnv.resolve_env(sys.executable)
        assert env.python_path == Path(sys.executable).resolve()

    def test_path_object(self):
        env = PyEnv.resolve_env(Path(sys.executable))
        assert env.python_path == Path(sys.executable).resolve()

    def test_nonexistent_dir_creates_venv(self, tmp_path):
        venv_dir = tmp_path / "new_env"
        env = PyEnv.resolve_env(venv_dir)
        assert venv_dir.exists()
        assert env.python_path.is_file()
        env.delete()


# ---------------------------------------------------------------------------
# TestPyEnvProperties
# ---------------------------------------------------------------------------

class TestPyEnvProperties:
    def setup_method(self):   _reset_singleton()
    def teardown_method(self): _reset_singleton()

    def test_is_current_false_before_singleton(self):
        env = PyEnv(python_path=Path(sys.executable), cwd=Path.cwd())
        assert env.is_current is False

    def test_version_info_matches_running_interpreter(self):
        vi = PyEnv.current().version_info
        assert vi == (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)

    def test_version_info_is_tuple_of_ints(self):
        vi = PyEnv.current().version_info
        assert len(vi) == 3
        assert all(isinstance(x, int) for x in vi)

    def test_uv_bin_is_real_file(self):
        assert PyEnv.current().uv_bin.is_file()

    def test_uv_bin_cached(self):
        env = PyEnv.current()
        assert env.uv_bin is env.uv_bin


# ---------------------------------------------------------------------------
# TestPipCmdArgs
# ---------------------------------------------------------------------------

class TestPipCmdArgs:
    def test_prefer_uv_true(self):
        env = PyEnv.current()
        cmd = env._pip_cmd_args(prefer_uv=True)
        assert "uv" in cmd[0]
        assert cmd[1] == "pip"

    def test_prefer_uv_false(self):
        env = PyEnv.current()
        cmd = env._pip_cmd_args(prefer_uv=False)
        assert cmd[0] == str(env.python_path)
        assert "-m" in cmd
        assert "pip" in cmd

    def test_uv_run_prefix(self):
        env = PyEnv.current()
        prefix = env._uv_run_prefix()
        assert "uv" in prefix[0]
        assert "run" in prefix
        assert "--python" in prefix
        assert str(env.python_path) in prefix

    def test_uv_run_prefix_custom_python(self):
        env = PyEnv.current()
        prefix = env._uv_run_prefix(python="/alt/python3")
        assert prefix[-1] == "/alt/python3"


# ---------------------------------------------------------------------------
# TestRequirements
# ---------------------------------------------------------------------------

class TestRequirements:
    def test_returns_nonempty_list(self):
        reqs = PyEnv.current().requirements()
        assert isinstance(reqs, list)
        assert len(reqs) > 0

    def test_each_item_is_name_version_tuple(self):
        for name, version in PyEnv.current().requirements():
            assert isinstance(name, str) and name
            assert isinstance(version, str) and version

    def test_system_packages_excluded_by_default(self):
        names = {n for n, _ in PyEnv.current().requirements()}
        assert "pip" not in names
        assert "setuptools" not in names

    def test_with_system_is_superset(self):
        env = PyEnv.current()
        without = {n for n, _ in env.requirements(with_system=False)}
        with_ = {n for n, _ in env.requirements(with_system=True)}
        assert without.issubset(with_)

    def test_prefer_uv_false_works(self):
        assert isinstance(PyEnv.current().requirements(prefer_uv=False), list)


# ---------------------------------------------------------------------------
# TestInstall
# ---------------------------------------------------------------------------

class TestInstall:
    def test_returns_none_with_nothing(self):
        assert PyEnv.current().install() is None

    def test_install_existing_package(self):
        assert PyEnv.current().install("pip") is not None

    def test_install_maps_import_name(self):
        # "yaml" maps to "PyYAML"; both should be present already
        assert PyEnv.current().install("yaml") is not None

    def test_install_from_requirements_file(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("pip\n")
        assert PyEnv.current().install(requirements=req) is not None

    def test_install_from_raw_requirements_string(self, tmp_path):
        env = PyEnv(
            python_path=PyEnv.current().python_path,
            cwd=tmp_path,
            prefer_uv=True,
        )
        assert env.install(requirements="pip\n") is not None

    def test_install_multiple(self):
        assert PyEnv.current().install("pip", "setuptools") is not None

    def test_extra_args(self):
        assert PyEnv.current().install("pip", extra_args=["--quiet"]) is not None


# ---------------------------------------------------------------------------
# TestUpdate
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_returns_none_with_nothing(self):
        assert PyEnv.current().update() is None

    def test_update_existing_package(self):
        assert PyEnv.current().update("pip") is not None


# ---------------------------------------------------------------------------
# TestUninstall
# ---------------------------------------------------------------------------

class TestUninstall:
    def test_returns_none_with_nothing(self):
        assert PyEnv.current().uninstall() is None

    def test_uninstall_and_reinstall_in_isolated_venv(self, tmp_path):
        """
        Create a throwaway venv, install a small package, uninstall it,
        then confirm it's gone from requirements().
        """
        venv_dir = tmp_path / "iso_venv"
        env = PyEnv.create_venv(venv_dir, cwd=tmp_path, seed=False)
        env.install("pip")  # ensure pip is available
        env.install("six")
        before = {n for n, _ in env.requirements()}
        assert "six" in before

        env.uninstall("six")
        after = {n for n, _ in env.requirements()}
        assert "six" not in after

        env.delete()


# ---------------------------------------------------------------------------
# TestPip
# ---------------------------------------------------------------------------

class TestPip:
    def test_list_json(self):
        assert PyEnv.current().pip("list", "--format=json") is not None

    def test_show_pip(self):
        assert PyEnv.current().pip("show", "pip") is not None


# ---------------------------------------------------------------------------
# TestDelete
# ---------------------------------------------------------------------------

class TestDelete:
    def setup_method(self):   _reset_singleton()
    def teardown_method(self): _reset_singleton()

    def test_raises_for_current_singleton(self):
        with pytest.raises(ValueError, match="Cannot delete the current singleton"):
            PyEnv.current().delete()

    def test_raises_when_no_pyvenv_cfg(self, tmp_path):
        py = tmp_path / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        with pytest.raises(ValueError, match="No pyvenv.cfg found"):
            PyEnv(python_path=py, cwd=tmp_path).delete()

    def test_real_venv_deleted(self, tmp_path):
        venv_dir = tmp_path / "del_venv"
        env = PyEnv.create_venv(venv_dir, cwd=tmp_path, seed=False)
        assert venv_dir.exists()
        env.delete()
        assert not venv_dir.exists()

    def test_raises_runtime_error_when_directory_not_removable(self, tmp_path):
        """
        Make the venv root read-only so rmtree fails, then restore permissions
        in the finaliser so pytest can clean up tmp_path.
        """
        venv_dir = tmp_path / "locked_venv"
        env = PyEnv.create_venv(venv_dir, cwd=tmp_path, seed=False)

        # Remove write permission on the venv root
        original_mode = venv_dir.stat().st_mode
        venv_dir.chmod(stat.S_IREAD | stat.S_IEXEC)

        try:
            with pytest.raises((RuntimeError, PermissionError, OSError)):
                env.delete()
        finally:
            # Restore so pytest can remove tmp_path
            venv_dir.chmod(original_mode)


# ---------------------------------------------------------------------------
# TestRunPythonCode
# ---------------------------------------------------------------------------

class TestRunPythonCode:
    def test_simple_code(self):
        assert PyEnv.current().run_python_code("x = 1 + 1") is not None

    def test_globs_injected(self):
        PyEnv.current().run_python_code("assert x == 42", globs={"x": 42})

    def test_multiple_globs(self):
        PyEnv.current().run_python_code("assert a + b == 10", globs={"a": 3, "b": 7})

    def test_string_glob(self):
        PyEnv.current().run_python_code("assert name == 'crude_oil'", globs={"name": "crude_oil"})

    def test_env_var_visible_in_subprocess(self):
        code = "import os, sys; sys.exit(0 if os.environ.get('_YGG_TEST') == 'ok' else 1)"
        PyEnv.current().run_python_code(code, env={"_YGG_TEST": "ok"})

    def test_raise_error_false_on_failure(self):
        result = PyEnv.current().run_python_code("raise ValueError('boom')", raise_error=False)
        assert result is not None

    def test_raise_error_true_on_failure(self):
        with pytest.raises(Exception):
            PyEnv.current().run_python_code("raise ValueError('boom')", raise_error=True)

    def test_custom_cwd(self, tmp_path):
        code = f"import os; assert os.getcwd() == {str(tmp_path)!r}"
        PyEnv.current().run_python_code(code, cwd=tmp_path)

    def test_stdin(self):
        code = "import sys; assert sys.stdin.read().strip() == 'hello'"
        PyEnv.current().run_python_code(code, stdin="hello\n")

    def test_prefer_uv_false(self):
        result = PyEnv.current().run_python_code("pass", prefer_uv=False)
        assert result is not None

    def test_error_raise_module_not_found(self):
        cmd = PyEnv.resolve_env("tmp", version="3.10").run_python_code("import dill").wait(raise_error=True)
        cmd


# ---------------------------------------------------------------------------
# TestImportModule
# ---------------------------------------------------------------------------

class TestImportModule:
    def setup_method(self):   _reset_singleton()
    def teardown_method(self): _reset_singleton()

    def test_import_json(self):
        import json
        assert PyEnv.current().import_module("json") is json

    def test_import_via_pip_name(self):
        import json
        assert PyEnv.current().import_module(pip_name="json") is json

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Provide at least one"):
            PyEnv.current().import_module()

    def test_install_false_raises_for_missing(self):
        with pytest.raises(ModuleNotFoundError):
            PyEnv.current().import_module("_nonexistent_xyz123", install=False)

    def test_runtime_import_module(self):
        import json
        PyEnv.current()
        assert PyEnv.runtime_import_module("json") is json

    def test_dateutil_via_mapped_name(self):
        import dateutil
        assert PyEnv.current().import_module("dateutil") is dateutil


# ---------------------------------------------------------------------------
# TestIntegrationSmoke
# ---------------------------------------------------------------------------

class TestIntegrationSmoke:
    def setup_method(self):   _reset_singleton()
    def teardown_method(self): _reset_singleton()

    def test_current_has_packages(self):
        assert len(PyEnv.current().requirements()) > 0

    def test_version_info_matches_sys(self):
        vi = PyEnv.current().version_info
        assert vi == (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)

    def test_resolve_env_round_trip(self):
        env = PyEnv.resolve_env(sys.executable)
        assert env.python_path == Path(sys.executable).resolve()
        assert env.version_info[0] >= 3

    def test_get_or_create_then_run_code(self):
        env = PyEnv.get_or_create(sys.executable)
        env.run_python_code("assert __import__('sys').version_info.major >= 3")

    def test_install_then_import(self):
        env = PyEnv.current()
        env.install("pip")
        assert env.import_module("json") is not None

    def test_full_venv_lifecycle(self, tmp_path):
        venv_dir = tmp_path / "full_lifecycle"
        env = PyEnv.create_venv(venv_dir, cwd=tmp_path, seed=False)
        assert venv_dir.exists()
        assert env.python_path.is_file()
        assert env.version_info[0] >= 3
        reqs = env.requirements(with_system=True)
        assert isinstance(reqs, list)
        env.delete()
        assert not venv_dir.exists()

    def test_pip_list_parseable(self):
        result = PyEnv.current().pip("list", "--format=json")
        # The SystemCommand result should have accessible stdout
        assert result is not None