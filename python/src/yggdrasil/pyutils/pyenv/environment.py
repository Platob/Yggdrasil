# pyenv.py
from __future__ import annotations

import importlib
import logging
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

logger = logging.getLogger(__name__)

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
            resolved = Path(sys.executable).resolve()
            logger.debug("resolve_python_executable: using sys.executable=%s", resolved)
            return resolved

        p = Path(python)
        if p.exists() and p.is_file():
            resolved = p.resolve()
            logger.debug("resolve_python_executable: using explicit path=%s", resolved)
            return resolved

        s = str(python).strip()
        if s and s[0].isdigit():
            s = f"python{s}"

        found = shutil.which(s)
        if not found:
            logger.error("resolve_python_executable: not found selector=%r", python)
            raise FileNotFoundError(f"Python executable not found: {python!r}")

        resolved = Path(found).resolve()
        logger.debug("resolve_python_executable: selector=%r resolved=%s", python, resolved)
        return resolved

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
            logger.debug("PyEnv.current: returning cached CURRENT_PYENV=%s", CURRENT_PYENV.python_path)
            return CURRENT_PYENV

        py = cls.resolve_python_executable(python)
        CURRENT_PYENV = cls(python_path=py, cwd=Path.cwd().resolve(), prefer_uv=prefer_uv)
        logger.debug(
            "PyEnv.current: created CURRENT_PYENV python_path=%s cwd=%s prefer_uv=%s",
            CURRENT_PYENV.python_path,
            CURRENT_PYENV.cwd,
            CURRENT_PYENV.prefer_uv,
        )
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

        logger.debug("uv_bin: resolving uv binary via runtime import")
        uv_mod = self.import_module(module_name="uv", pip_name="uv", upgrade=False)
        self._uv_bin_cache = Path(uv_mod.find_uv_bin())
        assert self._uv_bin_cache.is_file(), f"uv found but is not a file: {self._uv_bin_cache}"
        logger.debug("uv_bin: resolved=%s", self._uv_bin_cache)
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
            args = [str(self.uv_bin), "pip"]
            logger.debug("_pip_cmd_args: prefer_uv=True args=%s", args)
            return args

        args = [str(p), "-m", "pip"]
        logger.debug("_pip_cmd_args: prefer_uv=False args=%s", args)
        return args

    def _uv_run_prefix(self, python: Optional[str | Path] = None) -> list[str]:
        """
        Prefix for running commands inside uv's managed environment resolution.

        uv run --python <python_path> <command...>
        """
        p = python or self.python_path
        prefix = [str(self.uv_bin), "run", "--python", str(p)]
        logger.debug("_uv_run_prefix: %s", prefix)
        return prefix

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
            logger.debug("install: nothing to do (no packages, no requirements)")
            return None

        cmd = self._pip_cmd_args() + ["install"]
        wait_cfg = WaitingConfig.check_arg(wait)

        tmp_req: Path | None = None
        if requirements is not None:
            p = Path(requirements).expanduser()
            if p.exists():
                cmd += ["-r", str(p)]
                logger.debug("install: using requirements file=%s", p)
            else:
                import tempfile

                self.cwd.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix="requirements_", suffix=".txt", dir=str(self.cwd))
                tmp_req = Path(name)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(str(requirements).strip() + "\n")
                cmd += ["-r", str(tmp_req)]
                logger.debug("install: wrote temp requirements file=%s", tmp_req)

        if packages:
            cmd += [*packages]
        if extra_args:
            cmd += [*extra_args]

        logger.info("install: cmd=%s cwd=%s wait=%s", cmd, self.cwd, bool(wait_cfg))
        result = SystemCommand.run_lazy(cmd, cwd=self.cwd)

        if wait_cfg:
            result.wait(wait=wait_cfg)
            if tmp_req is not None:
                try:
                    tmp_req.unlink(missing_ok=True)
                    logger.debug("install: removed temp requirements file=%s", tmp_req)
                except Exception:
                    logger.warning("install: failed to remove temp requirements file=%s", tmp_req, exc_info=True)

        return result

    def update(
        self,
        *packages: str,
        extra_args: Sequence[str] = (),
        wait: WaitingConfigArg | None = True,
    ) -> SystemCommand | None:
        if not packages:
            logger.debug("update: nothing to do (no packages)")
            return None

        cmd = self._pip_cmd_args() + ["install", "--upgrade", *packages, *extra_args]
        logger.info("update: cmd=%s cwd=%s", cmd, self.cwd)
        return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait)

    def uninstall(
        self,
        *packages: str,
        extra_args: Sequence[str] = (),
        wait: WaitingConfigArg | None = True,
    ) -> SystemCommand | None:
        if not packages:
            logger.debug("uninstall: nothing to do (no packages)")
            return None

        cmd = self._pip_cmd_args() + ["uninstall", *packages, *extra_args]
        logger.info("uninstall: cmd=%s cwd=%s", cmd, self.cwd)
        return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait)

    def pip(
        self,
        *args: str,
        wait: WaitingConfigArg | None = True,
    ) -> subprocess.CompletedProcess[str]:
        cmd = self._pip_cmd_args() + list(args)
        logger.debug("pip: cmd=%s cwd=%s", cmd, self.cwd)
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
            # (kept as-is; looks like a hook for future behavior)
            logger.debug("run_python_code: args/kwargs provided but currently unused args=%s kwargs=%s", args, kwargs)

        if self.prefer_uv:
            cmd = self._uv_run_prefix() + ["python", "-c", code]
        else:
            cmd = [str(self.python_path), "-c", code]

        logger.debug(
            "run_python_code: cmd=%s cwd=%s prefer_uv=%s raise_error=%s wait=%s",
            cmd,
            cwd or self.cwd,
            self.prefer_uv,
            raise_error,
            wait,
        )
        rr = SystemCommand.run_lazy(cmd, cwd=cwd or self.cwd, env=merged_env)

        if stdin is not None:
            try:
                rr.popen.stdin.write(stdin)  # type: ignore[union-attr]
                rr.popen.stdin.flush()       # type: ignore[union-attr]
                rr.popen.stdin.close()       # type: ignore[union-attr]
                logger.debug("run_python_code: wrote stdin (%d bytes)", len(stdin))
            except Exception:
                logger.warning("run_python_code: failed writing stdin", exc_info=True)

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
                raise ValueError("Need at least module_name or pip_name to import")
            module_name = pip_name.replace("-", "_")

        logger.debug(
            "import_module: module_name=%s pip_name=%s install=%s upgrade=%s",
            module_name,
            pip_name,
            install,
            upgrade,
        )

        if not upgrade:
            try:
                mod = importlib.import_module(module_name)
                logger.debug("import_module: imported without install module_name=%s", module_name)
                return mod
            except ModuleNotFoundError:
                logger.debug("import_module: module missing module_name=%s", module_name)
                if not install:
                    raise

        if pip_name is None:
            pip_name = PIP_MODULE_NAME_MAPPINGS.get(module_name, module_name)

        logger.info("import_module: installing pip_name=%s for module_name=%s", pip_name, module_name)
        result = self.install(pip_name)
        error = result.raise_for_status(raise_error=False)

        if isinstance(error, Exception):
            logger.error(
                "import_module: install failed pip_name=%s module_name=%s",
                pip_name,
                module_name,
                exc_info=True,
            )
            raise ModuleNotFoundError(
                "No module named '%s'" % module_name,
                name=module_name,
            ) from error

        importlib.invalidate_caches()
        mod = importlib.import_module(module_name)
        logger.debug("import_module: imported after install module_name=%s", module_name)
        return mod
