"""
pyenv.py — Thin wrapper around a Python interpreter for environment management.

Provides :class:`PyEnv`, a dataclass that anchors all package-management and
subprocess-execution operations to a single Python interpreter path.  It
preferentially delegates to ``uv`` for speed but falls back to ``pip``/``python``
when ``uv`` is unavailable or explicitly disabled.

Typical usage
-------------
::

    # Use the current interpreter (singleton)
    env = PyEnv.current()
    env.install("pyarrow", "pandas")

    # Resolve or create a named venv
    env = PyEnv.get_or_create("3.12", packages=["pyarrow"])

    # Execute arbitrary Python in a subprocess
    env.run_python_code("import pyarrow; print(pyarrow.__version__)")

Module-level globals
--------------------
CURRENT_PYENV : PyEnv | None
    Singleton for the running interpreter; populated on first call to
    :meth:`PyEnv.current`.

PIP_MODULE_NAME_MAPPINGS : dict[str, str]
    Maps Python import names to their pip distribution names where they differ
    (e.g. ``"yaml"`` → ``"PyYAML"``).
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .system_command import SystemCommand
from .userinfo import UserInfo
from ..pyutils.waiting_config import WaitingConfig, WaitingConfigArg

__all__ = [
    "PyEnv",
    "PIP_MODULE_NAME_MAPPINGS",
    "CURRENT_PYENV",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# import name -> pip distribution name
# ---------------------------------------------------------------------------
#: Maps Python import names to their pip distribution names where they differ.
#: Extend this dict when adding new dependencies whose import name differs from
#: the installable package name.
PIP_MODULE_NAME_MAPPINGS: dict[str, str] = {
    "jwt": "PyJWT",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
    "yggdrasil": "ygg",
}

# Matches: "3" / "3.12" / "3.13.12" / "python3" / "python3.12" / "python3.13.12"
_PY_VERSION_RE = re.compile(
    r"^\s*(?:python)?\s*(\d+(?:\.\d+){0,2})\s*$",
    flags=re.IGNORECASE,
)

#: Module-level singleton — set on first call to :meth:`PyEnv.current`.
CURRENT_PYENV: PyEnv | None = None


def safe_pip_name(value: str | tuple[str, str] | Iterable[str | tuple[str, str]]) -> str | list[str]:
    """
    Map an import name to its pip distribution name, or return it unchanged.

    Looks up *value* in :data:`PIP_MODULE_NAME_MAPPINGS`.  Accepts either a
    single string or an iterable of strings; the return type mirrors the input.

    Parameters
    ----------
    value:
        A single import-name string, or an iterable of import-name strings.

    Returns
    -------
    str | list[str]
        The mapped pip distribution name(s).

    Examples
    --------
    ::

        safe_pip_name("yaml")          # → "PyYAML"
        safe_pip_name("numpy")         # → "numpy"  (no mapping needed)
        safe_pip_name(["yaml", "jwt"]) # → ["PyYAML", "PyJWT"]
    """
    if isinstance(value, str):
        return PIP_MODULE_NAME_MAPPINGS.get(value, value)
    elif isinstance(value, tuple) and len(value) == 2 and value[1].isdigit():
        return "%s==%s" % (value[0], value[1])
    return [safe_pip_name(v) for v in value]


# ─────────────────────────────────────────────────────────────────────────────
# PyEnv
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PyEnv:
    """
    Thin wrapper around a single Python interpreter path.

    :class:`PyEnv` is the central primitive for all environment-related
    operations: package management (install / update / uninstall), subprocess
    execution, and dynamic module imports with auto-install.

    Design constraints
    ------------------
    * ``python_path`` is the sole interpreter anchor — no ``venv_dir`` is
      stored separately.  The venv root is discovered at runtime by walking up
      the directory tree looking for ``pyvenv.cfg``.
    * All operations are working-directory–relative via ``cwd``.
    * When ``prefer_uv=True`` (the default), ``uv`` is used for pip operations
      and subprocess execution, falling back to standard pip / the interpreter
      directly when ``prefer_uv=False``.

    Execution strategy
    ------------------
    * ``prefer_uv=True``  → ``uv run --python <python_path> python …``
    * ``prefer_uv=False`` → ``<python_path> …``

    Attributes
    ----------
    python_path : Path
        Absolute path to the Python interpreter for this environment.
    cwd : Path
        Working directory used for all subprocess invocations.
    prefer_uv : bool
        When ``True`` (default), ``uv`` is preferred over bare pip/python calls.

    Notes
    -----
    The ``_uv_bin_cache`` field is internal; it is populated lazily on first
    access to :attr:`uv_bin` and is not part of the public interface.
    """

    python_path: Path
    cwd: Path = field(default_factory=lambda: Path.cwd().resolve())
    prefer_uv: bool = True

    # Internal cache — not exposed in __repr__ or __init__
    _version_info: tuple[int, int, int] | None = field(default=None, init=False, repr=False)
    _uv_bin_cache: Path | None = field(default=None, init=False, repr=False)

    def __getstate__(self) -> dict:
        return {
            "python_path": self.python_path,
            "cwd": self.cwd,
            "prefer_uv": self.prefer_uv,
            # Drop _version_info and _uv_bin_cache — both are lazy, cheap to recompute
        }

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "python_path", state["python_path"])
        object.__setattr__(self, "cwd", state["cwd"])
        object.__setattr__(self, "prefer_uv", state["prefer_uv"])
        object.__setattr__(self, "_version_info", None)
        object.__setattr__(self, "_uv_bin_cache", None)

    # ── Python resolution ─────────────────────────────────────────────────────

    @staticmethod
    def _databricks_python_path() -> Path | None:
        """Return the Python interpreter path for the current Databricks cluster.

        Detection relies on the ``DATABRICKS_RUNTIME_VERSION`` environment
        variable, which is always set by the Databricks runtime.  The
        interpreter path is read from ``PYSPARK_PYTHON`` (preferred) or
        ``PYSPARK_DRIVER_PYTHON`` as a fallback.

        Returns
        -------
        Path | None
            Absolute path to the cluster Python interpreter, or ``None`` when
            not running inside a Databricks runtime or when neither env var
            resolves to an existing file.
        """
        if os.getenv("DATABRICKS_RUNTIME_VERSION") is None:
            return None

        for var in ("PYSPARK_PYTHON", "PYSPARK_DRIVER_PYTHON"):
            value = os.getenv(var)
            if not value:
                continue
            p = Path(value)
            if p.exists() and p.is_file():
                logger.debug(
                    "PyEnv._databricks_python_path: resolved via %s=%s",
                    var,
                    value,
                )
                return p.resolve()
            logger.warning(
                "PyEnv._databricks_python_path: %s=%r does not point to a file",
                var,
                value,
            )

        logger.warning(
            "PyEnv._databricks_python_path: DATABRICKS_RUNTIME_VERSION is set but "
            "neither PYSPARK_PYTHON nor PYSPARK_DRIVER_PYTHON resolves to a file; "
            "falling back to sys.executable",
        )
        return None

    @staticmethod
    def resolve_python_executable(python: str | Path | None) -> Path:
        """
        Resolve a Python interpreter selector to an absolute :class:`~pathlib.Path`.

        Accepts several forms of selector:

        * ``None``           → ``sys.executable`` (the running interpreter)
        * :class:`~pathlib.Path` → resolved directly when the file exists
        * ``"python3.12"``   → looked up via :func:`shutil.which`
        * ``"3.12"``         → prepended with ``"python"`` then which'd
        * ``"/usr/bin/python3"`` → resolved as-is

        Parameters
        ----------
        python:
            Selector for the desired interpreter.

        Returns
        -------
        Path
            Absolute path to the resolved interpreter.

        Raises
        ------
        FileNotFoundError
            If the selector cannot be resolved to a file on ``PATH`` or disk.
        """
        if python is None:
            databricks_py = PyEnv._databricks_python_path()
            if databricks_py is not None:
                return databricks_py
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

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        python_path: Path,
        *,
        cwd: Path | None = None,
        prefer_uv: bool = True,
        packages: list[str] | None = None,
    ) -> PyEnv:
        """
        Primary constructor — resolves and normalises all paths, then
        optionally installs packages into the new environment.

        This is the preferred low-level factory when you already have a
        resolved :class:`~pathlib.Path` to an interpreter.  Higher-level
        callers should use :meth:`get_or_create` instead.

        Parameters
        ----------
        python_path:
            Path to the Python interpreter.  Resolved to an absolute path
            internally.
        cwd:
            Working directory for subprocesses.  Defaults to ``Path.cwd()``.
        prefer_uv:
            Use ``uv`` for pip operations when ``True`` (default).
        packages:
            Optional list of packages to install immediately after creation.

        Returns
        -------
        PyEnv
            A fully initialized environment instance.
        """
        env = cls(
            python_path=python_path.resolve(),
            cwd=(cwd or Path.cwd()).resolve(),
            prefer_uv=prefer_uv,
        )
        if packages:
            env.install(*packages)
        return env

    @classmethod
    def current(
        cls,
        *,
        python: str | Path | None = None,
        prefer_uv: bool = True,
    ) -> PyEnv:
        """
        Return the module-level singleton for the current interpreter.

        Created on first call; subsequent calls always return the same
        instance regardless of arguments.  This makes it safe to call
        ``PyEnv.current()`` repeatedly without incurring repeated setup cost.

        Parameters
        ----------
        python:
            Interpreter selector passed to :meth:`resolve_python_executable`.
            Ignored on subsequent calls once the singleton exists.
        prefer_uv:
            Forwarded to :meth:`create` on first construction.

        Returns
        -------
        PyEnv
            The module-level :data:`CURRENT_PYENV` singleton.
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
        packages: list[str] | None = None,
        prefer_uv: bool = True,
        seed: bool = True,
    ) -> PyEnv:
        """
        Resolve or create a Python environment from a flexible identifier.

        This is the primary high-level factory.  It handles all common
        forms of environment specification and returns a ready-to-use
        :class:`PyEnv`.

        Resolution order
        ----------------
        1. ``None`` / ``"current"`` / ``"sys"`` / ``""``
               → return the :meth:`current` singleton.
        2. A :class:`PyEnv` instance
               → returned as-is (packages installed if provided).
        3. Version selector (``"3.12"``, ``"python3.12"``, …)
               → venv created/located under
               ``~/.local/yggdrasil/python/envs/<selector>``.
        4. Path to a Python executable
               → used directly.
        5. Path to an existing venv directory
               → Python executable extracted from it.
        6. Path to a non-existing directory
               → venv created via ``uv venv``.
        7. Bare name / token
               → resolved via :func:`shutil.which`.

        Parameters
        ----------
        identifier:
            Environment specifier — see resolution order above.
        version:
            Pin a specific Python version when creating a new venv
            (e.g. ``"3.12"``).  Overrides the version implied by *identifier*.
        packages:
            Packages to install after resolving the environment.
        prefer_uv:
            Prefer ``uv`` for all operations (default: ``True``).
        seed:
            Pass ``--seed`` to ``uv venv`` when creating a new environment,
            which pre-installs pip/setuptools/wheel.

        Returns
        -------
        PyEnv
            A resolved and optionally package-populated environment.
        """
        if isinstance(identifier, PyEnv):
            if packages:
                identifier.install(*packages)
            return identifier

        env = cls.resolve_env(
            identifier,
            cwd=Path.cwd().resolve(),
            prefer_uv=prefer_uv,
            seed=seed,
            version=version,
        )

        if packages:
            env.install(*packages)

        return env

    @classmethod
    def resolve_env(
        cls,
        identifier: str | Path | None,
        *,
        cwd: Path | None = None,
        prefer_uv: bool = True,
        seed: bool = True,
        version: str | None = None,
        packages: list[str] | None = None
    ) -> PyEnv:
        """
        Pure resolution logic with no package-installation side-effects.

        Implements the same resolution order as :meth:`get_or_create` but
        without installing packages.  Intended for internal use and for
        callers that want to separate resolution from installation.

        Parameters
        ----------
        identifier:
            Environment specifier (see :meth:`get_or_create` for full details).
        cwd:
            Working directory for the returned :class:`PyEnv`.
        prefer_uv:
            Prefer ``uv`` for operations.
        seed:
            Pass ``--seed`` when creating a new venv.
        version:
            Override or supply a Python version for venv creation.

        Returns
        -------
        PyEnv
            Resolved environment.
        """
        if not identifier:
            return cls.current(prefer_uv=prefer_uv)

        if isinstance(identifier, str):
            s = identifier.strip()

            if not s or s.lower() in {"current", "sys", "system"}:
                return cls.current(prefer_uv=prefer_uv)

            m = _PY_VERSION_RE.match(s)
            if m:
                version = version or m.group(1)
                # Fall through to create_venv with a pinned version
            elif cls._looks_like_path(s):
                identifier = Path(s)

            else:
                # Version-only string: place venv under the standard location
                identifier = Path.home() / ".local" / "yggdrasil" / "python" / "envs" / s

                py = cls._venv_python_from_dir(identifier, raise_error=False)

                if py.is_file() and "python" in py.name:
                    return cls.create(py, cwd=cwd, prefer_uv=prefer_uv)

        path = Path(identifier).expanduser()  # type: ignore[arg-type]

        # If the path is already a Python executable, use it directly
        if path.is_file() and "python" in path.name:
            return cls.create(path, cwd=cwd, prefer_uv=prefer_uv)

        # Otherwise treat as a venv directory (existing or to be created)
        return cls.create_venv(
            path,
            cwd=cwd or Path.cwd().resolve(),
            prefer_uv=prefer_uv,
            seed=seed,
            version=version,
            packages=packages
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
        packages: list[str] | None = None
    ) -> PyEnv:
        """
        Create a new virtual environment at *venv_dir* via ``uv`` and return
        a :class:`PyEnv` anchored to it.

        After creation the environment is automatically seeded with
        ``ygg``, ``pandas``, and ``dill``.

        Parameters
        ----------
        venv_dir:
            Target directory for the new venv.  Parent directories are
            created automatically.
        cwd:
            Working directory for the returned :class:`PyEnv`.
        prefer_uv:
            Prefer ``uv`` for all operations in the returned environment.
        seed:
            Pass ``--seed`` to ``uv venv`` (pre-installs pip/setuptools/wheel).
        version:
            Python version to pin for the new venv (e.g. ``"3.12"``).
            Falls back to the current interpreter's version if not provided.

        Returns
        -------
        PyEnv
            Environment anchored to the newly created venv.
        """
        anchor = cls.current()
        venv_dir.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(anchor.uv_bin), "venv", str(venv_dir),
            "--python", version or str(anchor.python_path),
            *(["--seed"] if seed else []),
        ]
        logger.info("create_venv: cmd=%s", cmd)
        SystemCommand.run_lazy(cmd, cwd=cwd).wait(True)

        py = cls._venv_python_from_dir(venv_dir)
        env = cls.create(py, cwd=cwd, prefer_uv=prefer_uv)
        # Seed with baseline packages expected by the Yggdrasil framework

        if packages:
            env.install(*packages)

        return env

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_current(self) -> bool:
        """
        ``True`` if this instance is the module-level singleton.

        Useful for guard-clauses (e.g. in :meth:`delete`) where operating on
        the active interpreter would be dangerous.
        """
        return CURRENT_PYENV is not None and CURRENT_PYENV is self

    @property
    def userinfo(self) -> UserInfo:
        """Return the current :class:`~.userinfo.UserInfo` singleton."""
        return UserInfo.current()

    @property
    def version_info(self) -> tuple[int, int, int]:
        """
        Return the interpreter's version as ``(major, minor, micro)``.

        Executes a short Python one-liner in a subprocess to query
        ``sys.version_info``, so it reflects the *environment's* interpreter,
        not necessarily the calling process.

        Returns
        -------
        tuple[int, int, int]
            ``(major, minor, micro)`` — e.g. ``(3, 12, 4)``.

        Raises
        ------
        subprocess.CalledProcessError
            If the subprocess exits with a non-zero status.
        """
        if self._version_info:
            return self._version_info

        code = (
            "import sys, json; "
            "print(json.dumps([sys.version_info.major, sys.version_info.minor, sys.version_info.micro]))"
        )
        cmd = (
            self._uv_run_prefix() + ["python", "-c", code]
            if self.prefer_uv
            else [str(self.python_path), "-c", code]
        )
        res = subprocess.run(
            cmd,
            cwd=str(self.cwd),
            env=dict(os.environ),
            text=True,
            capture_output=True,
            check=True,
        )
        major, minor, micro = json.loads(res.stdout.strip())
        self._version_info = int(major), int(minor), int(micro)

        return self._version_info

    @property
    def uv_bin(self) -> Path:
        """
        Resolve the ``uv`` binary, installing it into the current interpreter
        if it is absent.  The result is cached after the first access.

        ``uv`` is used as the preferred backend for all pip and venv operations
        when :attr:`prefer_uv` is ``True``.

        Returns
        -------
        Path
            Absolute path to the ``uv`` executable.

        Raises
        ------
        FileNotFoundError
            If ``uv`` resolves but the reported path is not an actual file.
        """
        if self._uv_bin_cache:
            return self._uv_bin_cache

        logger.debug("uv_bin: resolving via runtime import")
        try:
            import uv as uv_mod
        except ImportError:
            # Auto-install uv using the plain pip fallback to avoid recursion
            uv_mod = self.install("uv", prefer_uv=False)

        self._uv_bin_cache = Path(uv_mod.find_uv_bin())
        if not self._uv_bin_cache.is_file():
            raise FileNotFoundError(f"uv resolved but is not a file: {self._uv_bin_cache}")

        logger.debug("uv_bin: resolved=%s", self._uv_bin_cache)
        return self._uv_bin_cache

    # ── Package management ────────────────────────────────────────────────────

    def requirements(
        self,
        prefer_uv: bool | None = None,
        *,
        with_system: bool = False,
    ) -> list[tuple[str, str]]:
        """
        Return installed packages as ``(name, version)`` tuples, sorted by name.

        Delegates to ``uv pip list`` or ``python -m pip list`` depending on
        *prefer_uv*.

        Parameters
        ----------
        prefer_uv:
            ``None`` → use :attr:`self.prefer_uv`; ``True`` → ``uv pip list``;
            ``False`` → ``<python> -m pip list``.
        with_system:
            When ``True``, includes baseline tooling (pip / setuptools / wheel)
            in the results.  Excluded by default to keep output clean.

        Returns
        -------
        list[tuple[str, str]]
            Alphabetically sorted ``(name, version)`` pairs.

        Raises
        ------
        subprocess.CalledProcessError
            If the pip invocation fails.
        ValueError
            If the pip JSON output is not a list.
        """
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

        _system = {"pip", "setuptools", "wheel"}
        out: list[tuple[str, str]] = []
        for item in pkgs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            version = str(item.get("version", "")).strip()
            if not name or not version:
                continue
            if not with_system and name.lower() in _system:
                continue
            out.append((name, version))
        return out

    def install(
        self,
        *packages: str,
        requirements: str | Path | None = None,
        extra_args: Sequence[str] = (),
        wait: WaitingConfigArg | None = True,
        prefer_uv: bool | None = None,
    ) -> SystemCommand | None:
        """
        Install packages into the environment anchored by :attr:`python_path`.

        Supports both positional package names and a requirements file (or raw
        requirements-file content written to a temp file automatically).

        Parameters
        ----------
        *packages:
            Package names to install (import names are mapped via
            :func:`safe_pip_name` automatically).
        requirements:
            Path to an existing requirements file **or** raw requirements-file
            content as a string.  When raw content is provided it is written to
            a temporary file inside :attr:`cwd`.
        extra_args:
            Additional arguments forwarded verbatim to ``pip install``.
        wait:
            Waiting strategy passed to :class:`~.system_command.SystemCommand`.
            Defaults to synchronous wait (``True``).
        prefer_uv:
            Override :attr:`self.prefer_uv` for this single call.

        Returns
        -------
        SystemCommand | None
            The running/completed command, or ``None`` if there was nothing to
            install.

        Notes
        -----
        Temporary requirements files are automatically cleaned up after a
        synchronous install completes.  For async installs (``wait=False``)
        the caller is responsible for cleanup.
        """
        if not packages and requirements is None:
            return None

        cmd = self._pip_cmd_args(prefer_uv=prefer_uv) + ["install"]
        wait_cfg = WaitingConfig.check_arg(wait)
        tmp_req: Path | None = None

        if requirements is not None:
            req_path = Path(requirements).expanduser()
            if req_path.exists():
                cmd += ["-r", str(req_path)]
            else:
                # Treat the value as raw requirements content; write to a temp file
                import tempfile
                self.cwd.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(
                    prefix="requirements_", suffix=".txt", dir=str(self.cwd)
                )
                tmp_req = Path(name)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(str(requirements).strip() + "\n")
                cmd += ["-r", str(tmp_req)]

        if packages:
            cmd += [safe_pip_name(p) for p in packages]
        if extra_args:
            cmd += list(extra_args)

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
        """
        Upgrade one or more packages in the anchored environment.

        Equivalent to ``pip install --upgrade <packages>``.

        Parameters
        ----------
        *packages:
            Names of packages to upgrade.
        extra_args:
            Additional arguments forwarded to ``pip install --upgrade``.
        wait:
            Waiting strategy.  Defaults to synchronous wait.

        Returns
        -------
        SystemCommand | None
            The running/completed command, or ``None`` if *packages* is empty.
        """
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
        """
        Uninstall one or more packages from the anchored environment.

        Equivalent to ``pip uninstall <packages>``.

        Parameters
        ----------
        *packages:
            Names of packages to remove.
        extra_args:
            Additional arguments forwarded to ``pip uninstall``.
        wait:
            Waiting strategy.  Defaults to synchronous wait.

        Returns
        -------
        SystemCommand | None
            The running/completed command, or ``None`` if *packages* is empty.
        """
        if not packages:
            return None
        cmd = self._pip_cmd_args() + ["uninstall", *packages, *extra_args]
        logger.info("uninstall: cmd=%s cwd=%s", cmd, self.cwd)
        return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait)

    def pip(
        self,
        *args: str,
        wait: WaitingConfigArg | None = True,
    ) -> SystemCommand:
        """
        Run an arbitrary pip subcommand and return the result.

        A thin escape-hatch for pip invocations not covered by the higher-level
        helpers (:meth:`install`, :meth:`update`, :meth:`uninstall`).

        Parameters
        ----------
        *args:
            Arguments forwarded verbatim to the pip invocation.
        wait:
            Waiting strategy.  Defaults to synchronous wait.

        Returns
        -------
        SystemCommand
            The running/completed command.

        Examples
        --------
        ::

            env.pip("list")
            env.pip("install", "polars")
            env.pip("show", "pyarrow")
        """
        cmd = self._pip_cmd_args() + list(args)
        logger.debug("pip: cmd=%s cwd=%s", cmd, self.cwd)
        return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait)

    def delete(self) -> None:
        """
        Delete the virtual environment that contains this interpreter.

        Walks up from :attr:`python_path` looking for ``pyvenv.cfg`` (the venv
        root marker) and removes that entire directory tree with
        :func:`shutil.rmtree`.

        Raises
        ------
        ValueError
            If this environment is the :meth:`current` singleton, or if no
            ``pyvenv.cfg`` is found within 4 parent levels of
            :attr:`python_path`.
        RuntimeError
            If the directory removal fails.

        Warnings
        --------
        This operation is irreversible.  Ensure no active processes are using
        the environment before calling this method.
        """
        if self.is_current:
            raise ValueError("Cannot delete the current singleton PyEnv.")

        # Walk up to find the venv root (indicated by pyvenv.cfg)
        candidate = self.python_path.parent
        venv_root: Path | None = None
        for _ in range(4):
            if (candidate / "pyvenv.cfg").exists():
                venv_root = candidate
                break
            candidate = candidate.parent

        if venv_root is None:
            raise ValueError(
                f"Cannot determine venv root for python_path={self.python_path!r}. "
                "No pyvenv.cfg found within 4 parent levels."
            )

        logger.debug("PyEnv.delete: removing venv_root=%s", venv_root)
        try:
            shutil.rmtree(venv_root)
        except Exception as exc:
            raise RuntimeError(f"Failed to delete venv at {venv_root}: {exc}") from exc

        logger.info("PyEnv.delete: removed venv_root=%s", venv_root)

    # ── Execution ─────────────────────────────────────────────────────────────

    def run_python_code(
        self,
        code: str,
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True,
        stdin: str | None = None,
        python: PyEnv | Path | str | None = None,
        packages: list[str] | None = None,
        prefer_uv: bool | None = None,
        globs: dict[str, Any] | None = None,
    ) -> SystemCommand:
        """
        Execute Python source code in a subprocess under this (or another)
        environment.

        The code is passed via the ``-c`` flag.  Variables in *globs* are
        serialised as ``name = repr(value)`` and prepended to the code block,
        making simple primitives, dicts, and base64 strings available inside
        the subprocess without any IPC overhead.

        Parameters
        ----------
        code:
            Python source string to execute.
        cwd:
            Override the working directory.  Defaults to :attr:`self.cwd`.
        env:
            Extra environment variables merged *over* ``os.environ``.
        wait:
            Waiting strategy.  Defaults to synchronous wait.
        raise_error:
            Raise :class:`~.system_command.SystemCommandError` on non-zero exit
            when ``True`` (default).
        stdin:
            Text written to the subprocess stdin pipe immediately after launch.
        python:
            Override the target interpreter — accepts a :class:`PyEnv`
            instance, a path, or a selector string.  Defaults to ``self``.
        packages:
            Install these packages into *self* before running the code.
        prefer_uv:
            Override :attr:`self.prefer_uv` for this call.
        globs:
            Mapping of variable names to values injected at the top of the
            code string via ``repr()``.  Safe for primitives, dicts, and
            base64-encoded strings.

        Returns
        -------
        SystemCommand
            The running or completed command object.

        Notes
        -----
        * When *python* is provided, package installation still targets
          *self*, not the override interpreter.
        * stdin write errors are swallowed with a warning to avoid masking
          the primary error from the code execution itself.
        """
        merged_env = {**os.environ, **(env or {})}
        target = self.get_or_create(identifier=python) if python is not None else self
        prefer_uv = target.prefer_uv if prefer_uv is None else prefer_uv

        if packages:
            self.install(*packages)

        # Inject globals as literal assignments at the top of the code block
        if globs:
            prefix = "\n".join(f"{k} = {v!r}" for k, v in globs.items())
            code = prefix + "\n" + code

        cmd = (
            target._uv_run_prefix() + ["python", "-c", code]
            if prefer_uv
            else [str(target.python_path), "-c", code]
        )

        proc = SystemCommand.run_lazy(cmd, cwd=cwd or target.cwd, env=merged_env, python=self)

        if stdin is not None:
            try:
                proc.popen.stdin.write(stdin)
                proc.popen.stdin.flush()
                proc.popen.stdin.close()
            except Exception:
                logger.warning("run_python_code: failed writing stdin", exc_info=True)

        return proc.wait(wait=wait, raise_error=raise_error)

    # ── Runtime import + auto-install ─────────────────────────────────────────

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
        Class-level convenience wrapper for :meth:`import_module`.

        Delegates to ``PyEnv.current().import_module(...)`` so callers don't
        need to obtain a :class:`PyEnv` instance first.

        Parameters
        ----------
        module_name:
            Name used in ``import``.  Derived from *pip_name* if omitted.
        install:
            Auto-install via pip when the module is not found.
        pip_name:
            Distribution name for pip.  Defaults to the mapping in
            :data:`PIP_MODULE_NAME_MAPPINGS` or *module_name* as-is.
        upgrade:
            Force a pip upgrade even if the import already succeeds.

        Returns
        -------
        types.ModuleType
            The imported module object.
        """
        return cls.current().import_module(
            module_name=module_name,
            install=install,
            pip_name=pip_name,
            upgrade=upgrade,
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
        Import a module into the current interpreter, installing it if missing.

        Combines :func:`importlib.import_module` with an automatic pip install
        fallback, making it straightforward to use optional dependencies
        without pre-populating the environment.

        Parameters
        ----------
        module_name:
            Name used in ``import``; derived from *pip_name* if omitted.
        install:
            Auto-install via pip when the module is not found.  Set to
            ``False`` to propagate :class:`ModuleNotFoundError` immediately.
        pip_name:
            Distribution name for pip (defaults to the
            :data:`PIP_MODULE_NAME_MAPPINGS` lookup or *module_name* as-is).
        upgrade:
            Run a pip upgrade even if the import already succeeds.

        Returns
        -------
        types.ModuleType
            The imported module object.

        Raises
        ------
        ValueError
            If neither *module_name* nor *pip_name* is provided.
        ModuleNotFoundError
            If both the import and the installation attempt fail.

        Examples
        --------
        ::

            pa = env.import_module("pyarrow")
            yaml = env.import_module("yaml")          # maps to PyYAML
            toml = env.import_module(pip_name="toml") # derives module_name
        """
        if not module_name:
            if not pip_name:
                raise ValueError("Provide at least one of module_name or pip_name.")
            module_name = pip_name.replace("-", "_")

        if not upgrade:
            try:
                return importlib.import_module(module_name)
            except ModuleNotFoundError:
                if not install:
                    raise

        pip_name = pip_name or safe_pip_name(module_name)
        result = self.install(pip_name, wait=False)
        error = result.raise_for_status(raise_error=False)

        if isinstance(error, Exception):
            raise ModuleNotFoundError(
                f"No module named '{module_name}'", name=module_name
            ) from error

        importlib.invalidate_caches()
        return importlib.import_module(module_name)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pip_cmd_args(
        self,
        python: str | Path | None = None,
        prefer_uv: bool | None = None,
    ) -> list[str]:
        """
        Build the base pip invocation prefix as a list of strings.

        * ``prefer_uv=True``  → ``["<uv>", "pip"]``
        * ``prefer_uv=False`` → ``["<python>", "-m", "pip"]``

        Parameters
        ----------
        python:
            Override the interpreter path.  Defaults to :attr:`python_path`.
        prefer_uv:
            Override :attr:`self.prefer_uv` for this call.

        Returns
        -------
        list[str]
            Command prefix ready for extension with pip sub-commands.
        """
        prefer_uv = self.prefer_uv if prefer_uv is None else prefer_uv
        p = python or self.python_path

        if prefer_uv:
            return [str(self.uv_bin), "pip"]
        return [str(p), "-m", "pip"]

    def _uv_run_prefix(self, python: str | Path | None = None) -> list[str]:
        """
        Return the ``uv run --python <path>`` prefix for subprocess execution.

        Parameters
        ----------
        python:
            Override the interpreter path.  Defaults to :attr:`python_path`.

        Returns
        -------
        list[str]
            ``["<uv>", "run", "--python", "<python_path>"]``
        """
        return [str(self.uv_bin), "run", "--python", str(python or self.python_path)]

    @staticmethod
    def _venv_python_from_dir(venv_dir: Path, raise_error: bool = True) -> Path:
        """
        Locate the Python executable inside a venv directory.

        Checks the standard platform-specific locations in order:

        * ``bin/python``        (POSIX)
        * ``bin/python3``       (POSIX fallback)
        * ``Scripts/python.exe`` (Windows)
        * ``Scripts/python``    (Windows bare)

        Parameters
        ----------
        venv_dir:
            Root directory of the virtual environment.
        raise_error:
            Raises error if not found

        Returns
        -------
        Path
            Absolute path to the Python executable.

        Raises
        ------
        ValueError
            If no Python executable is found in any expected location.
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

        if raise_error:
            raise ValueError(f"No Python executable found inside venv: {venv_dir}")

        if os.name == "nt":
            return venv_dir / "Scripts" / "python.exe"
        else:
            return venv_dir / "bin" / "python"

    @staticmethod
    def _looks_like_path(s: str) -> bool:
        """
        Return ``True`` if *s* resembles a filesystem path rather than a bare
        name or version selector.

        Heuristics
        ----------
        * Starts with ``~``, ``.``, or ``/``
        * On Windows, starts with a drive letter (``C:``)
        * Contains ``/`` or ``\\``

        Parameters
        ----------
        s:
            String to classify.

        Returns
        -------
        bool
        """
        if not s:
            return False
        if s.startswith(("~", ".", "/")):
            return True
        if os.name == "nt" and len(s) >= 2 and s[1] == ":":
            return True
        return "/" in s or "\\" in s