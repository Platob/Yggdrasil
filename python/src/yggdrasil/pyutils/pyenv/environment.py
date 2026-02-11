# pyenv.py
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .system_command import SystemCommand
from ..waiting_config import WaitingConfig, WaitingConfigArg

__all__ = ["PyEnv", "PIP_MODULE_NAME_MAPPINGS"]


PIP_MODULE_NAME_MAPPINGS: dict[str, str] = {
    # import name -> pip distribution name
    "jwt": "PyJWT",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
    "yggdrasil": "ygg",
}


CURRENT_PYENV: "PyEnv | None" = None


@dataclass(slots=True)
class PyEnv:
    """
    Tiny env manager.

    Constraints:
      - `python_path` is the python executable used as the anchor
      - NO venv_dir stored on the object
      - location-based via `cwd`

    This module prefers uv for both pip operations and execution when `prefer_uv=True`.

    Execution strategy (managed envs):
      - prefer_uv=True  -> `uv run --python <python_path> python ...`
      - prefer_uv=False -> `<python_path> ...` directly
    """

    python_path: Path
    cwd: Path = field(default_factory=lambda: Path.cwd().resolve())
    prefer_uv: bool = True

    _uv_bin_cache: Path | None = field(default=None, init=False, repr=False)

    # -----------------------
    # Python resolution
    # -----------------------
    @staticmethod
    def resolve_python_executable(python: str | Path | None) -> Path:
        """
        Accepts:
          - None -> sys.executable
          - Path to python executable
          - "python", "python3.12", "3.12" style selectors (best-effort)
        """
        if python is None:
            return Path(sys.executable).resolve()

        p = Path(python)
        if p.exists() and p.is_file():
            return p.resolve()

        s = str(python).strip()
        if s and s[0].isdigit():
            s = f"python{s}"

        found = shutil.which(s)
        if not found:
            raise FileNotFoundError(f"Python executable not found: {python!r}")
        return Path(found).resolve()

    # -----------------------
    # Singleton
    # -----------------------
    @classmethod
    def current(
        cls,
        *,
        python: str | Path | None = None,
        prefer_uv: bool = True,
    ) -> "PyEnv":
        global CURRENT_PYENV

        if CURRENT_PYENV is not None:
            return CURRENT_PYENV

        py = cls.resolve_python_executable(python)
        CURRENT_PYENV = cls(python_path=py, cwd=Path.cwd().resolve(), prefer_uv=prefer_uv)
        return CURRENT_PYENV

    @property
    def is_current(self) -> bool:
        return CURRENT_PYENV is self

    # -----------------------
    # uv resolution (cached)
    # -----------------------
    @property
    def uv_bin(self) -> Path:
        if self._uv_bin_cache:
            return self._uv_bin_cache

        uv_mod = self.import_module(module_name="uv", pip_name="uv", upgrade=False)
        self._uv_bin_cache = Path(uv_mod.find_uv_bin())
        assert self._uv_bin_cache.is_file(), f"uv found but is not a file: {self._uv_bin_cache}"
        return self._uv_bin_cache

    # -----------------------
    # Command plumbing
    # -----------------------
    def _pip_cmd_args(self, python: Optional[str | Path] = None) -> list[str]:
        """
        Build a pip command.

        prefer_uv=True:
          uv pip --python <python>

        prefer_uv=False:
          <python> -m pip
        """
        p = python or self.python_path

        if self.prefer_uv:
            return [str(self.uv_bin), "pip"]
        return [str(p), "-m", "pip"]

    def _uv_run_prefix(self, python: Optional[str | Path] = None) -> list[str]:
        """
        Prefix for running commands inside uv's managed environment resolution.

        uv run --python <python_path> <command...>
        """
        p = python or self.python_path
        return [str(self.uv_bin), "run", "--python", str(p)]

    # -----------------------
    # Public API (packages)
    # -----------------------
    def install(
        self,
        *packages: str,
        requirements: str | Path | None = None,
        extra_args: Sequence[str] = (),
        wait: WaitingConfigArg | None = True,
    ) -> SystemCommand | None:
        """
        Lazy installation into the environment anchored by python_path.

        requirements behavior:
          - if `requirements` points to an existing file -> use it
          - else treat it as requirements *content* and write a temp file
        """
        if not packages and requirements is None:
            return None

        cmd = self._pip_cmd_args() + ["install"]
        wait_cfg = WaitingConfig.check_arg(wait)

        tmp_req: Path | None = None
        if requirements is not None:
            p = Path(requirements).expanduser()
            if p.exists():
                cmd += ["-r", str(p)]
            else:
                import tempfile

                self.cwd.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix="requirements_", suffix=".txt", dir=str(self.cwd))
                tmp_req = Path(name)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(str(requirements).strip() + "\n")
                cmd += ["-r", str(tmp_req)]

        if packages:
            cmd += [*packages]
        if extra_args:
            cmd += [*extra_args]

        result = SystemCommand.run_lazy(cmd, cwd=self.cwd)

        if wait_cfg:
            result.wait(wait=wait_cfg)
            if tmp_req is not None:
                try:
                    tmp_req.unlink(missing_ok=True)
                except Exception:
                    pass

        return result

    def update(
        self,
        *packages: str,
        extra_args: Sequence[str] = (),
        wait: WaitingConfigArg | None = True,
    ) -> SystemCommand | None:
        if not packages:
            return None

        cmd = self._pip_cmd_args() + ["install", "--upgrade", *packages, *extra_args]
        return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait)

    def uninstall(
        self,
        *packages: str,
        extra_args: Sequence[str] = (),
        wait: WaitingConfigArg | None = True,
    ) -> SystemCommand | None:
        if not packages:
            return None

        cmd = self._pip_cmd_args() + ["uninstall", *packages, *extra_args]

        return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait)

    def pip(
        self,
        *args: str,
        wait: WaitingConfigArg | None = True,
    ) -> subprocess.CompletedProcess[str]:
        cmd = self._pip_cmd_args() + list(args)
        return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait)

    # -----------------------
    # Public API (execution)
    # -----------------------
    def run_python_code(
        self,
        code: str,
        *,
        args: Sequence[str] | None = None,
        kwargs: Mapping[str, Any] | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True,
        stdin: str | None = None,
    ) -> SystemCommand:
        """
        Run arbitrary Python code.

        prefer_uv=True (managed env):
          uv run --python <python_path> python -c <code> -- <args...>

        prefer_uv=False:
          <python_path> -c <code> -- <args...>
        """
        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)

        if args or kwargs:
            pass

        if self.prefer_uv:
            cmd = self._uv_run_prefix() + ["python", "-c", code]
        else:
            cmd = [str(self.python_path), "-c", code]

        rr = SystemCommand.run_lazy(cmd, cwd=cwd or self.cwd, env=merged_env)

        if stdin is not None:
            try:
                rr.popen.stdin.write(stdin)  # type: ignore[union-attr]
                rr.popen.stdin.flush()       # type: ignore[union-attr]
                rr.popen.stdin.close()       # type: ignore[union-attr]
            except Exception:
                pass

        return rr.wait(wait=wait, raise_error=raise_error)

    # -----------------------
    # Runtime import + auto-install (current interpreter)
    # -----------------------
    @classmethod
    def runtime_import_module(
        cls,
        module_name: str | None = None,
        *,
        install: bool = True,
        pip_name: str | None = None,
        upgrade: bool = False,
    ):
        return (
            cls.current()
            .import_module(
                module_name=module_name,
                install=install,
                pip_name=pip_name,
                upgrade=upgrade
            )
        )

    def import_module(
        self,
        module_name: str | None = None,
        *,
        install: bool = True,
        pip_name: str | None = None,
        upgrade: bool = False,
    ):
        """
        Import a module, installing it at runtime with pip if missing (into *current* interpreter).

        module_name : name used in `import`
        pip_name    : name used for pip install (defaults to mapping or module_name)
        upgrade     : pip install --upgrade
        """
        if module_name is None:
            if not pip_name:
                raise ValueError(
                    "Need at least module_name or pip_name to import"
                )

            module_name = pip_name.replace("-", "_")

        if not upgrade:
            try:
                return importlib.import_module(module_name)
            except ModuleNotFoundError:
                if not install:
                    raise

        if pip_name is None:
            pip_name = PIP_MODULE_NAME_MAPPINGS.get(module_name, module_name)

        result = self.install(pip_name)
        error = result.raise_for_status(raise_error=False)

        if isinstance(error, Exception):
            raise ModuleNotFoundError(
                "No module named '%s'" % module_name,
                name=module_name,
            ) from error

        importlib.invalidate_caches()
        mod = importlib.import_module(module_name)

        return mod
