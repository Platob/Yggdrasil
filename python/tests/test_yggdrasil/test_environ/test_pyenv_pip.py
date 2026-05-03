"""Test PyEnv pip install functionality on Windows."""
from __future__ import annotations

import pytest
from pathlib import Path
from yggdrasil.environ.environment import PyEnv, safe_pip_name


class TestPyEnvPip:
    """Test suite for PyEnv pip operations on Windows."""

    def test_pyenv_current(self) -> None:
        """Test that PyEnv.current() initializes correctly."""
        env = PyEnv.current()
        assert env is not None
        assert env.python_path.exists()
        assert env.python_path.is_file()
        assert "python" in env.python_path.name.lower()

    def test_pyenv_windows_detection(self) -> None:
        """Test Windows platform detection."""
        env = PyEnv.current()
        assert isinstance(env.is_windows, bool)
        # On Windows, should be True
        import os
        assert env.is_windows == (os.name == "nt")

    def test_version_info(self) -> None:
        """Test version_info property."""
        env = PyEnv.current()
        version = env.version_info
        assert version.major >= 3
        assert version.minor >= 10

    def test_has_uv(self) -> None:
        """Test uv availability detection."""
        env = PyEnv.current()
        has_uv = env.has_uv()
        assert isinstance(has_uv, bool)

    def test_safe_pip_name_mapping(self) -> None:
        """Test safe_pip_name handles common name mappings."""
        test_cases = {
            "yaml": "PyYAML",
            "dateutil": "python-dateutil",
            "jwt": "PyJWT",
            "requests": "requests",
        }
        for module_name, expected_pip_name in test_cases.items():
            assert safe_pip_name(module_name) == expected_pip_name

    def test_pip_cmd_args_default(self) -> None:
        """Test _pip_cmd_args returns valid command list."""
        env = PyEnv.current()
        cmd = env._pip_cmd_args("install")
        assert isinstance(cmd, list)
        assert len(cmd) > 0
        assert "pip" in cmd
        assert "install" in cmd

    def test_pip_cmd_args_with_python_path(self) -> None:
        """Test _pip_cmd_args with explicit python path."""
        env = PyEnv.current()
        cmd = env._pip_cmd_args("install", python=env.python_path)
        assert isinstance(cmd, list)
        assert len(cmd) > 0
        assert any(str(env.python_path) in str(part) for part in cmd)

    def test_pip_cmd_args_prefer_uv_false(self) -> None:
        """Test _pip_cmd_args with prefer_uv=False."""
        env = PyEnv.current()
        cmd = env._pip_cmd_args("install", prefer_uv=False)
        assert isinstance(cmd, list)
        # Should be [<python>, '-m', 'pip', 'install']
        assert cmd == [str(env.python_path), "-m", "pip", "install"]

    def test_pip_cmd_args_prefer_uv_true(self) -> None:
        """Test _pip_cmd_args with prefer_uv=True."""
        env = PyEnv.current()
        if env.has_uv():
            cmd = env._pip_cmd_args("install", prefer_uv=True)
            assert isinstance(cmd, list)
            # Should be [<uv>, 'pip', 'install', '--python', <p>]
            assert "pip" in cmd
            assert "install" in cmd
            assert "--python" in cmd
            assert str(env.python_path) in cmd

    def test_pip_list_execution(self) -> None:
        """Test that pip list command can execute."""
        env = PyEnv.current()
        # Run in a temp directory to avoid project-specific issues
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd_backup = env.cwd
            try:
                # Create a new PyEnv with temp working directory
                env_tmp = PyEnv(python_path=env.python_path, cwd=Path(tmpdir))
                result = env_tmp.pip("list", "--format=json").wait(
                    wait=True, raise_error=True
                )
                assert result is not None
            finally:
                pass

    def test_bin_path_windows(self) -> None:
        """Test bin_path property on Windows."""
        env = PyEnv.current()
        bin_path = env.bin_path
        assert bin_path.exists()
        assert bin_path.is_dir()
        # On Windows, should contain Scripts
        if env.is_windows:
            assert "Scripts" in str(bin_path)

    def test_root_path(self) -> None:
        """Test root_path property."""
        env = PyEnv.current()
        root_path = env.root_path
        assert root_path.exists()
        assert root_path.is_dir()
        # root_path should be parent of bin_path
        assert root_path == env.bin_path.parent


class TestPipCommandGeneration:
    """Test pip command generation for Windows compatibility."""

    def test_uv_pip_syntax_with_python(self) -> None:
        """``uv pip <subcommand> --python <p>`` is used so installs land
        in the venv that owns ``<p>`` (instead of an ephemeral ``uv run``
        env)."""
        env = PyEnv.current()
        if env.has_uv():
            cmd = env._pip_cmd_args("install", prefer_uv=True)
            assert "pip" in cmd
            assert "install" in cmd
            assert "--python" in cmd
            assert str(env.python_path) in cmd

    def test_python_fallback_syntax(self) -> None:
        """Test that python -m pip fallback syntax is correct."""
        env = PyEnv.current()
        cmd = env._pip_cmd_args("install", prefer_uv=False)
        assert cmd == [str(env.python_path), "-m", "pip", "install"]

    def test_pip_args_with_packages(self) -> None:
        """Test pip install command generation with package names."""
        env = PyEnv.current()
        # Don't actually install, just test command generation
        base_cmd = env._pip_cmd_args("install")
        assert "pip" in base_cmd
        assert "install" in base_cmd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

