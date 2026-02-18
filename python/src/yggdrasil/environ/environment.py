# yggdrasil.environ.environment.py
from __future__ import annotations

import json

"""
Tiny environment manager + runtime import helper.

Design constraints:
- `python_path` is the interpreter anchor (a specific python executable)
- No venv_dir stored on the object
- Location-based via `cwd`
- Prefers `uv` for pip + execution when `prefer_uv=True`

Execution strategy (managed envs):
- prefer_uv=True  -> `uv run --python <python_path> python ...`
- prefer_uv=False -> `<python_path> ...` directly

Also includes a best-effort "current user info" probe:
- Always returns OS-level runtime identity (username, home, host, etc.)
- Optionally returns Databricks identity when available (via Spark `current_user()` or env vars)
"""

import re
import importlib
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union, Iterable

from .userinfo import UserInfo
from .system_command import SystemCommand
from ..pyutils.waiting_config import WaitingConfig, WaitingConfigArg

__all__ = [
    "PyEnv",
    "PIP_MODULE_NAME_MAPPINGS",
    "CURRENT_PYENV",
]

logger = logging.getLogger(__name__)

# import name -> pip distribution name
PIP_MODULE_NAME_MAPPINGS: dict[str, str] = {
    "jwt": "PyJWT",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
    "yggdrasil": "ygg",
}
# matches:
#   "3" / "3.12" / "3.13.12"
#   "python3" / "python3.12" / "python3.13.12"
_PY_VERSION_RE = re.compile(
    r"^\s*(?:python)?\s*(\d+(?:\.\d+){0,2})\s*$",
    flags=re.IGNORECASE,
)
CURRENT_PYENV: "PyEnv | None" = None


@dataclass(slots=True)
class PyEnv:
    """
    Tiny env manager.

    Constraints:
      - `python_path` is the python executable used as the anchor
      - NO venv_dir stored on the object
      - location-based via `cwd`

    Prefers uv for both pip operations and execution when `prefer_uv=True`.

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
        Resolve a python interpreter executable.

        Accepts:
          - None -> sys.executable
          - Path to python executable
          - "python", "python3.12", "3.12" style selectors (best-effort)

        Raises:
          - FileNotFoundError if selector cannot be resolved.
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
            logger.error("resolve_python_executable: not found selector=%r", python)
            raise FileNotFoundError(f"Python executable not found: {python!r}")

        return Path(found).resolve()

    # -----------------------
    # Construction funnel
    # -----------------------
    @classmethod
    def create(
        cls,
        python_path: Path,
        *,
        cwd: Path | None = None,
        prefer_uv: bool = True,
        packages: Optional[list[str]] = None,
    ) -> "PyEnv":
        """
        Single constructor funnel.

        Normalizes:
          - python_path (resolved)
          - cwd (resolved; defaults to current process cwd)

        Applies:
          - prefer_uv

        Optional:
          - installs packages into the resolved env
        """
        env = cls(
            python_path=python_path.resolve(),
            cwd=(cwd or Path.cwd()).resolve(),
            prefer_uv=prefer_uv,
        )

        if packages:
            env.install(*packages)

        return env

    # -----------------------
    # Helpers for get_or_create
    # -----------------------
    @staticmethod
    def _venv_python_from_dir(venv_dir: Path) -> Path:
        """
        Given a venv directory, resolve the python executable inside it.
        """
        candidates = [
            venv_dir / "bin" / "python",
            venv_dir / "bin" / "python3",
            venv_dir / "Scripts" / "python.exe",
            venv_dir / "Scripts" / "python",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        raise ValueError(f"Cannot find python executable inside venv dir: {venv_dir}")

    @staticmethod
    def _looks_like_path(s: str) -> bool:
        """
        Heuristic: any path separator, starts with '.' or '~', or looks like an absolute path.
        """
        if not s:
            return False
        if s.startswith(("~", ".", "/")):
            return True
        if os.name == "nt" and len(s) >= 2 and s[1] == ":":
            return True
        return ("/" in s) or ("\\" in s)

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
        """
        Return a singleton "current interpreter" PyEnv.

        The first call creates the singleton; subsequent calls reuse it.
        """
        global CURRENT_PYENV

        if CURRENT_PYENV is not None:
            return CURRENT_PYENV

        py = cls.resolve_python_executable(python)
        CURRENT_PYENV = cls.create(py, cwd=Path.cwd(), prefer_uv=prefer_uv)

        logger.debug(
            "PyEnv.current: created CURRENT_PYENV python_path=%s cwd=%s prefer_uv=%s",
            CURRENT_PYENV.python_path,
            CURRENT_PYENV.cwd,
            CURRENT_PYENV.prefer_uv,
        )
        return CURRENT_PYENV

    @classmethod
    def get_or_create(
        cls,
        identifier: str | Path | None = None,
        *,
        version: str | None = None,
        packages: Optional[list[str]] = None,
        prefer_uv: bool = True,
        seed: bool = True,
    ) -> "PyEnv":
        """
        Resolve or create a Python environment from a flexible identifier.

        Resolution order:
          - None / "current" / "sys" / ""  -> singleton current interpreter
          - PyEnv instance                 -> returned as-is
          - Version selector ("3.12", "python3.12", "3.13.1", ...)
                                           -> resolved interpreter
          - Path to python executable      -> used directly
          - Path to existing venv dir      -> python extracted from venv
          - Path to non-existing dir       -> venv created via `uv venv`
          - Bare name / token              -> resolved via shutil.which
        """
        if isinstance(identifier, PyEnv):
            if packages:
                identifier.install(*packages)
            return identifier

        cwd = Path.cwd().resolve()

        env = cls.resolve_env(
            identifier,
            cwd=cwd,
            prefer_uv=prefer_uv,
            seed=seed,
            version=version
        )

        if packages:
            env.install(*packages)

        return env

    @classmethod
    def resolve_env(
        cls,
        identifier: str | Path | None,
        *,
        cwd: Path,
        prefer_uv: bool,
        seed: bool,
        version: str | None = None,
    ) -> "PyEnv":
        """Pure resolution logic — no package installation side-effects."""

        # --- None / empty string -> current singleton ---
        if not identifier:
            return cls.current(prefer_uv=prefer_uv)

        if isinstance(identifier, str):
            s = identifier.strip()

            if not s or s.lower() in {"current", "sys", "sys.executable"}:
                return cls.current(prefer_uv=prefer_uv)

            if "/" not in s and "\\" not in s:
                identifier = Path.home() / ".local" / "yggdrasil" / "python" / "envs" / s
            else:
                # Version selector: "3.12", "python3.13.1", etc.
                m = _PY_VERSION_RE.match(s)
                if m:
                    version = version or m.group(1)

                # Promote to Path if it looks like one, else treat as a bare name/selector
                elif cls._looks_like_path(s):
                    identifier = Path(s)
                else:
                    py = cls.resolve_python_executable(s)
                    return cls.create(py, cwd=cwd, prefer_uv=prefer_uv)

        # --- Path resolution ---
        path = Path(identifier).expanduser()  # type: ignore[arg-type]

        if path.is_file() and "python" in path.name:
            return cls.create(path, cwd=cwd, prefer_uv=prefer_uv)

        # Non-existent path -> create venv
        return cls.create_venv(
            path,
            cwd=cwd,
            prefer_uv=prefer_uv,
            seed=seed,
            version=version
        )

    @classmethod
    def create_venv(
        cls,
        venv_dir: Path,
        *,
        cwd: Path,
        prefer_uv: bool = True,
        seed: bool = True,
        version: str | None = None,
    ) -> "PyEnv":
        """Create a new venv at `venv_dir` via uv and return a PyEnv anchored to it."""
        anchor = cls.current()
        venv_dir.parent.mkdir(parents=True, exist_ok=True)

        # version kwarg pins the interpreter; falls back to current anchor
        python_pin = version or str(anchor.python_path)

        cmd = [
            str(anchor.uv_bin), "venv", str(venv_dir),
            "--python", python_pin,
            *(["--seed"] if seed else []),
        ]
        logger.info("_create_venv_env: cmd=%s", cmd)
        SystemCommand.run_lazy(cmd, cwd=cwd).wait(True)

        py = cls._venv_python_from_dir(venv_dir)
        return cls.create(py, cwd=cwd, prefer_uv=prefer_uv)

    @property
    def is_current(self) -> bool:
        """True if this instance is the module singleton returned by PyEnv.current()."""
        return CURRENT_PYENV is not None and CURRENT_PYENV is self

    @property
    def userinfo(self):
        return UserInfo.current()

    @property
    def version_info(self) -> tuple[int, int, int]:
        """
        Return (major, minor, micro) for this environment's python interpreter.
        """
        code = (
            "import sys, json; "
            "print(json.dumps([sys.version_info.major, sys.version_info.minor, sys.version_info.micro]))"
        )

        if self.prefer_uv:
            cmd = self._uv_run_prefix() + ["python", "-c", code]
        else:
            cmd = [str(self.python_path), "-c", code]

        # We want stdout; SystemCommand doesn't expose a nice CompletedProcess here,
        # so use subprocess directly for this tiny query.
        res = subprocess.run(
            cmd,
            cwd=str(self.cwd),
            env=dict(os.environ),
            text=True,
            capture_output=True,
            check=True,
        )
        major, minor, micro = json.loads(res.stdout.strip())
        return int(major), int(minor), int(micro)

    # -----------------------
    # uv resolution (cached)
    # -----------------------
    @property
    def uv_bin(self) -> Path:
        """
        Resolve `uv` binary location, installing uv into the *current* interpreter if needed.

        Caches the resolved path on first access.
        """
        if self._uv_bin_cache:
            return self._uv_bin_cache

        logger.debug("uv_bin: resolving uv binary via runtime import")

        try:
            import uv as uv_mod
        except ImportError:
            uv_mod = self.install("uv", prefer_uv=False)

        self._uv_bin_cache = Path(uv_mod.find_uv_bin())
        assert self._uv_bin_cache.is_file(), f"uv found but is not a file: {self._uv_bin_cache}"

        logger.debug("uv_bin: resolved=%s", self._uv_bin_cache)
        return self._uv_bin_cache

    def requirements(
        self,
        prefer_uv: bool | None = None,
        *,
        with_system: bool = False,
    ) -> list[tuple[str, str]]:
        """
        Return installed packages as requirements.txt-style lines: ["name==version", ...].

        prefer_uv:
          - None  -> use self.prefer_uv
          - True  -> use `uv pip ...`
          - False -> use `<python> -m pip ...`

        with_system:
          - False: excludes baseline tooling packages (pip/setuptools/wheel)
          - True: include everything

        Implementation notes:
          - Uses `pip list --format=json` (portable & parseable).
          - When prefer_uv=True, uses `uv pip` (NOT `uv run`) to avoid project-resolution behavior.
          - Output is sorted for determinism.
        """
        prefer_uv = self.prefer_uv if prefer_uv is None else prefer_uv

        # IMPORTANT: when prefer_uv=True, use "uv pip" (not "uv run python -m pip")
        cmd = self._pip_cmd_args(prefer_uv=prefer_uv) + ["list", "--format=json"]

        res = subprocess.run(
            cmd,
            cwd=str(self.cwd),
            env=dict(os.environ),
            text=True,
            capture_output=True,
            check=True,
        )

        pkgs = json.loads(res.stdout or "[]")
        if not isinstance(pkgs, list):
            raise ValueError("Unexpected pip output: expected JSON list")

        system_names = {"pip", "setuptools", "wheel"}  # minimal, cross-platform
        out: list[str] = []
        for item in pkgs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            version = str(item.get("version", "")).strip()
            if not name or not version:
                continue
            if (not with_system) and (name.lower() in system_names):
                continue
            out.append((name, version))

        return out

    # -----------------------
    # Command plumbing
    # -----------------------
    def _pip_cmd_args(
        self,
        python: Optional[str | Path] = None,
        prefer_uv: Optional[bool] = None,
    ) -> list[str]:
        """
        Build a pip command.

        prefer_uv=True:
          uv pip

        prefer_uv=False:
          <python> -m pip
        """
        p = python or self.python_path
        prefer_uv = self.prefer_uv if prefer_uv is None else prefer_uv

        if prefer_uv:
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
        prefer_uv: Optional[bool] = None,
    ) -> SystemCommand | None:
        """
        Lazy installation into the environment anchored by python_path.

        requirements behavior:
          - if `requirements` points to an existing file -> use it
          - else treat it as requirements *content* and write a temp file
        """
        if not packages and requirements is None:
            return None

        cmd = self._pip_cmd_args(prefer_uv=prefer_uv) + ["install"]
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
            cmd += [safe_pip_name(_) for _ in packages]

        if extra_args:
            cmd += [*extra_args]

        logger.info("install: cmd=%s cwd=%s wait=%s", cmd, self.cwd, bool(wait_cfg))
        result = SystemCommand.run_lazy(cmd, cwd=self.cwd)

        if wait_cfg:
            result.wait(wait=wait_cfg)
            if tmp_req is not None:
                try:
                    tmp_req.unlink(missing_ok=True)
                except Exception:
                    logger.warning(
                        "install: failed to remove temp requirements file=%s",
                        tmp_req,
                        exc_info=True,
                    )

        return result

    def update(
        self,
        *packages: str,
        extra_args: Sequence[str] = (),
        wait: WaitingConfigArg | None = True,
    ) -> SystemCommand | None:
        """Upgrade packages in the anchored environment."""
        if not packages:
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
        """Uninstall packages from the anchored environment."""
        if not packages:
            return None

        cmd = self._pip_cmd_args() + ["uninstall", *packages, *extra_args]
        logger.info("uninstall: cmd=%s cwd=%s", cmd, self.cwd)
        return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait)

    def pip(
        self,
        *args: str,
        wait: WaitingConfigArg | None = True,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run a pip command and wait for completion.

        Example:
            env.pip("list")
            env.pip("install", "polars")
        """
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
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True,
        stdin: str | None = None,
        python: Union["PyEnv", Path, str, None] = None,
        packages: list[str] | None = None,
        prefer_uv: bool | None = None,
    ) -> SystemCommand:
        """
        Run arbitrary Python code, optionally targeting a different interpreter and/or
        ensuring requirements are installed before execution.

        python:
          - None: use self.python_path
          - PyEnv: use that env's python_path
          - Path/str: resolved via resolve_python_executable (if selector) or used as path

        requirements:
          - None: don't install anything
          - list[str]: pip install those packages
          - str/Path:
              * if Path exists -> treat as requirements file (-r)
              * else treat as requirements content and write a temp file (existing install() behavior)
        """
        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)

        # Resolve target env/interpreter for execution + optional installs
        target: PyEnv = self.get_or_create(identifier=python) if python is not None else self

        # Build exec command using the *target* interpreter
        prefer_uv = target.prefer_uv if prefer_uv is None else prefer_uv

        if packages:
            self.install(
                *packages,
            )

        if prefer_uv:
            cmd = target._uv_run_prefix() + ["python", "-c", code]
        else:
            cmd = [str(target.python_path), "-c", code]

        rr = SystemCommand.run_lazy(cmd, cwd=cwd or target.cwd, env=merged_env, python=self)

        if stdin is not None:
            try:
                rr.popen.stdin.write(stdin)  # type: ignore[union-attr]
                rr.popen.stdin.flush()  # type: ignore[union-attr]
                rr.popen.stdin.close()  # type: ignore[union-attr]
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
        """
        Convenience wrapper that always targets the current interpreter.

        Calls:
            PyEnv.current().import_module(...)
        """
        return (
            cls.current()
            .import_module(
                module_name=module_name,
                install=install,
                pip_name=pip_name,
                upgrade=upgrade,
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
        Import a module, installing it at runtime with pip if missing
        (into the *current* interpreter).

        Args:
            module_name: name used in `import`
            install: if True, install missing modules via pip
            pip_name: name used for pip install (defaults to mapping or module_name)
            upgrade: if True, do a pip upgrade install even if import succeeds

        Returns:
            Imported module object.

        Raises:
            ModuleNotFoundError if module import and/or installation fails.
        """
        if not module_name:
            if not pip_name:
                raise ValueError("Need at least module_name or pip_name to import")
            module_name = pip_name.replace("-", "_")

        if not upgrade:
            try:
                mod = importlib.import_module(module_name)
                return mod
            except ModuleNotFoundError:
                if not install:
                    raise

        if pip_name is None:
            pip_name = safe_pip_name(module_name)

        result = self.install(pip_name, wait=False)
        error = result.raise_for_status(raise_error=False)

        if isinstance(error, Exception):
            raise ModuleNotFoundError(
                "No module named '%s'" % module_name,
                name=module_name,
            ) from error

        importlib.invalidate_caches()
        mod = importlib.import_module(module_name)
        return mod


def safe_pip_name(value: str | Iterable[str]):
    if isinstance(value, str):
        return PIP_MODULE_NAME_MAPPINGS.get(value, value)

    return [
        safe_pip_name(_)
        for _ in value
    ]