"""Python environment management with uv-first toolchain.

:class:`PyEnv` wraps a single Python interpreter path and provides:

* **Resolution** — locate interpreters by path, version selector, or venv dir.
* **Virtual-env lifecycle** — create, delete, and reuse virtual environments
  via ``uv venv`` (with automatic ``uv`` bootstrap).
* **Package management** — install / update / uninstall via ``uv pip`` or
  ``python -m pip`` with private-API fallback.
* **Subprocess execution** — run Python code under ``uv run`` or bare Python.
* **Runtime imports** — import-or-install a module at call time.

The module prefers ``uv`` for all pip and subprocess operations but
falls back to plain pip / python transparently when ``uv`` is
unavailable.

Public API
----------
.. autosummary::

   PyEnv
   safe_pip_name
   runtime_import_module
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, MISSING
from pathlib import Path
from typing import Any, Iterable, Sequence, TYPE_CHECKING, ClassVar

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.version import VersionInfo

from .system_command import SystemCommand
from .userinfo import UserInfo

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

__all__ = [
    "runtime_import_module",
    "PyEnv",
    "safe_pip_name",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# import name -> pip distribution name
# ---------------------------------------------------------------------------
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

CURRENT_PYENV: PyEnv | None = None


def safe_pip_name(
    value: str | tuple[str, str] | Iterable[str | tuple[str, str]],
) -> str | list[str]:
    if isinstance(value, str):
        return PIP_MODULE_NAME_MAPPINGS.get(value, value)

    if isinstance(value, tuple) and len(value) == 2:
        name, version = value
        return f"{safe_pip_name(name)}=={version}"

    return [safe_pip_name(v) for v in value]


@dataclass
class PyEnv:
    """
    Thin wrapper around a single Python interpreter path.

    :class:`PyEnv` is the central primitive for environment-related operations:
    package management (install / update / uninstall), subprocess execution,
    and dynamic imports with auto-install.

    Design constraints
    ------------------
    * ``python_path`` is the sole interpreter anchor — no explicit ``venv_dir``
      is stored.
    * All operations are working-directory–relative via ``cwd``.
    * When ``prefer_uv=True`` (default), ``uv`` is preferred for pip operations
      and subprocess execution, but calls fall back to plain pip / python when
      ``uv`` is unavailable.

    Execution strategy
    ------------------
    * ``prefer_uv=True``  -> ``uv run --python <python_path> python ...``
    * ``prefer_uv=False`` -> ``<python_path> ...``
    """

    python_path: Path
    cwd: Path = field(default_factory=lambda: Path.cwd())
    prefer_uv: bool = True

    _version_info: VersionInfo | None = field(default=None, init=False, repr=False)
    _uv_bin: Path | None = field(default=None, init=False, repr=False)
    _checked_modules: set[str] = field(default_factory=set, init=False, repr=False)

    _SPARK_SESSION: ClassVar[object] = MISSING

    def __post_init__(self) -> None:
        self.python_path = Path(self.python_path).expanduser().resolve()
        self.cwd = Path(self.cwd).expanduser().resolve()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "python_path": self.python_path,
            "cwd": self.cwd,
            "prefer_uv": self.prefer_uv,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        object.__setattr__(self, "python_path", Path(state["python_path"]).expanduser().resolve())
        object.__setattr__(self, "cwd", Path(state["cwd"]).expanduser().resolve())
        object.__setattr__(self, "prefer_uv", bool(state["prefer_uv"]))
        object.__setattr__(self, "_version_info", None)
        object.__setattr__(self, "_uv_bin", None)

    # ---------------------------------------------------------------------
    # Construction / resolution
    # ---------------------------------------------------------------------

    @staticmethod
    def resolve_python_executable(python: str | Path | None) -> Path:
        """
        Resolve a Python selector to an absolute executable path.

        Accepts:
        * ``None`` -> ``sys.executable``
        * ``Path`` to an executable
        * version selectors such as ``'3.12'`` or ``'python3.12'``
        * directory paths containing a Python executable
        * raw executable paths
        """
        if python is None or python == "":
            return Path(sys.executable).resolve()

        p = Path(python).expanduser()

        if p.is_file() and "python" in p.name.lower():
            return p.resolve()

        if p.is_dir():
            return PyEnv._find_python_in_dir(p).resolve()

        s = str(python).strip()
        if s and s[0].isdigit():
            s = f"python{s}"

        found = shutil.which(s)
        if found:
            return Path(found).resolve()

        raise FileNotFoundError(f"Python executable not found: {python!r}")

    @staticmethod
    def _find_python_in_dir(folder: Path) -> Path:
        """
        Search *folder* for a Python executable, preferring standard venv layouts.
        """
        folder = folder.expanduser().resolve()

        candidates = [
            folder / "bin" / "python",
            folder / "bin" / "python3",
            folder / "Scripts" / "python.exe",
            folder / "Scripts" / "python",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        patterns = ["**/python", "**/python3", "**/python3.*", "**/python.exe"]
        found: list[Path] = []
        for pattern in patterns:
            for match in folder.glob(pattern):
                if match.is_file() and os.access(match, os.X_OK):
                    found.append(match.resolve())

        if not found:
            raise FileNotFoundError(f"No Python executable found under directory: {folder}")

        def _rank(p: Path) -> tuple[int, tuple[int, ...], int]:
            version_match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", p.name)
            version = tuple(int(x) for x in version_match.groups("0")) if version_match else (0, 0, 0)
            return (
                0 if "python3" in p.name.lower() else 1,
                tuple(-v for v in version),
                len(p.parts),
            )

        return sorted(found, key=_rank)[0]

    @classmethod
    def instance(
        cls,
        python_path: Path,
        *,
        cwd: Path | None = None,
        prefer_uv: bool = True,
        packages: list[str] | None = None,
    ) -> PyEnv:
        env = cls(
            python_path=Path(python_path).expanduser().resolve(),
            cwd=(cwd or Path.cwd()).expanduser().resolve(),
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
        """
        global CURRENT_PYENV

        if CURRENT_PYENV is not None:
            return CURRENT_PYENV

        py = cls.resolve_python_executable(python)
        env = cls.instance(py, prefer_uv=prefer_uv)

        if py.resolve() == Path(sys.executable).resolve():
            vinfo = sys.version_info
            object.__setattr__(
                env,
                "_version_info",
                VersionInfo(major=vinfo.major, minor=vinfo.minor, patch=vinfo.micro),
            )

        CURRENT_PYENV = env
        return env

    @classmethod
    def get_or_create(
        cls,
        identifier: str | Path | PyEnv | None = None,
        *,
        version: str | None = None,
        packages: list[str] | None = None,
        prefer_uv: bool = True,
        seed: bool = True,
        cwd: Path | None = None,
    ) -> PyEnv:
        """
        Resolve or create a Python environment from a flexible identifier.
        """
        if isinstance(identifier, PyEnv):
            env = identifier
        else:
            anchor = cls.current(prefer_uv=prefer_uv)
            env = anchor.venv(
                identifier,
                cwd=cwd or Path.cwd(),
                prefer_uv=prefer_uv,
                seed=seed,
                version=version,
            )

        if packages:
            env.install(*packages)

        return env

    def venv(
        self,
        identifier: str | Path | None,
        *,
        cwd: Path | None = None,
        prefer_uv: bool = True,
        seed: bool = True,
        version: str | None = None,
        packages: list[str] | None = None,
    ) -> PyEnv:
        """
        Resolve an environment identifier or create a venv when needed.
        """
        if not identifier:
            return self.current(prefer_uv=prefer_uv)

        if isinstance(identifier, PyEnv):
            env = identifier
            if packages:
                env.install(*packages)
            return env

        if isinstance(identifier, str):
            s = identifier.strip()

            if not s or s.lower() in {"current", "sys", "system"}:
                env = self.current(prefer_uv=prefer_uv)
                if packages:
                    env.install(*packages)
                return env

            match = _PY_VERSION_RE.match(s)
            if match:
                version = version or match.group(1)
                identifier = Path.home() / ".local" / "yggdrasil" / "python" / "envs" / s
            elif self._looks_like_path(s):
                identifier = Path(s)
            else:
                identifier = Path.home() / ".local" / "yggdrasil" / "python" / "envs" / s

        path = Path(identifier).expanduser()

        if path.is_file() and "python" in path.name.lower():
            env = self.instance(path, cwd=cwd, prefer_uv=prefer_uv)
            if packages:
                env.install(*packages)
            return env

        py = self._venv_python_from_dir(path, raise_error=False)
        if py.is_file():
            env = self.instance(py, cwd=cwd, prefer_uv=prefer_uv)
            if packages:
                env.install(*packages)
            return env

        return self.create(
            path,
            cwd=cwd or Path.cwd(),
            prefer_uv=prefer_uv,
            seed=seed,
            version=version,
            packages=packages,
        )

    def create(
        self,
        folder: Path | str,
        *,
        cwd: Path | None = None,
        prefer_uv: bool = True,
        seed: bool = True,
        version: str | None = None,
        packages: list[str] | None = None,
        linked: bool = False,
        native_tls: bool = True,
        wait: WaitingConfigArg = True,
        clear: bool = True,
    ) -> PyEnv:
        """
        Create a new virtual environment at *folder* via ``uv venv``.
        """
        if isinstance(folder, str):
            if self._looks_like_path(folder):
                folder = Path(folder)
            else:
                folder = Path.home() / ".local" / "yggdrasil" / "python" / "envs" / folder

        folder = folder.expanduser().resolve()
        folder.parent.mkdir(parents=True, exist_ok=True)

        anchor = self

        if not version:
            if linked:
                version = str(anchor.python_path)
            else:
                version = str(anchor.version_info)

        cmd = [
            *anchor._uv_base_cmd(install_runtime=True),
            "venv",
            str(folder),
            "--python",
            str(version),
            *(["--seed"] if seed else []),
            *(["--native-tls"] if native_tls else []),
            *(["--clear"] if clear else []),
        ]
        logger.info("PyEnv.create: cmd=%s", cmd)
        SystemCommand.run_lazy(cmd, cwd=cwd or self.cwd).wait(True)

        py = self._venv_python_from_dir(folder)
        env = self.instance(py, cwd=cwd, prefer_uv=prefer_uv)

        if packages:
            env.install(*packages, wait=wait)

        return env

    # ---------------------------------------------------------------------
    # Basic properties
    # ---------------------------------------------------------------------

    @property
    def is_current(self) -> bool:
        """``True`` when this instance is the module-level :data:`CURRENT_PYENV` singleton."""
        return CURRENT_PYENV is not None and CURRENT_PYENV is self

    @property
    def is_windows(self) -> bool:
        """``True`` when running on Windows (``os.name == 'nt'``)."""
        return os.name == "nt"

    @property
    def bin_path(self) -> Path:
        """Directory containing the Python executable (``Scripts`` on Windows, ``bin`` elsewhere)."""
        return self.python_path.parent

    @property
    def root_path(self) -> Path:
        """Parent of :attr:`bin_path` — typically the venv root."""
        return self.bin_path.parent

    @property
    def userinfo(self) -> UserInfo:
        """Return the current :class:`~yggdrasil.environ.userinfo.UserInfo`."""
        return UserInfo.current()

    @property
    def version_info(self) -> VersionInfo:
        """
        Return the interpreter version for this environment.
        """
        if self._version_info is not None:
            return self._version_info

        try:
            code = (
                "import sys, json; "
                "print(json.dumps([sys.version_info.major, sys.version_info.minor, sys.version_info.micro]))"
            )
            cmd = [str(self.python_path), "-c", code]
            res = subprocess.run(
                cmd,
                cwd=str(self.cwd),
                env=dict(os.environ),
                text=True,
                capture_output=True,
                check=True,
            )
            major, minor, micro = json.loads(res.stdout.strip())
            object.__setattr__(
                self,
                "_version_info",
                VersionInfo(int(major), int(minor), int(micro)),
            )
            return self._version_info
        except Exception as exc:
            raise RuntimeError(f"Failed to get version info for Python at {self.python_path}: {exc}") from exc

    @classmethod
    def in_databricks(cls):
        return os.getenv("DATABRICKS_RUNTIME_VERSION")

    @classmethod
    def in_databricks_notebook(cls) -> bool:
        """``True`` when running inside a Databricks **notebook** cell.

        Notebook execution drives Python via an IPython kernel; a
        Databricks job's plain entry point does not. The combination
        of ``DATABRICKS_RUNTIME_VERSION`` + a live IPython instance is
        the standard heuristic — the same signal ``dbutils`` itself
        uses to surface its notebook-only helpers.
        """
        if not cls.in_databricks():
            return False
        try:
            from IPython import get_ipython
        except ImportError:
            return False
        return get_ipython() is not None

    @classmethod
    def in_aws_lambda(cls) -> bool:
        """``True`` when running inside the AWS Lambda runtime.

        ``AWS_LAMBDA_FUNCTION_NAME`` is set on every Lambda invocation
        by the runtime bootstrap and is documented as reserved — it is
        not user-settable and never leaks to non-Lambda environments,
        making it the canonical detector.
        """
        return "AWS_LAMBDA_FUNCTION_NAME" in os.environ

    @classmethod
    def in_aws_batch(cls) -> bool:
        """``True`` when running inside an AWS Batch job container.

        ``AWS_BATCH_JOB_ID`` is injected into every Batch container by
        the Batch agent and is the canonical detector.
        """
        return "AWS_BATCH_JOB_ID" in os.environ

    @classmethod
    def in_aws(cls) -> bool:
        """``True`` when running on an AWS-managed compute surface.

        Detects AWS Lambda, AWS Batch, AWS ECS / Fargate, and AWS
        CodeBuild via the env vars those services inject. Bare EC2 is
        **not** covered — there's no environment-side signal for it
        (callers needing that should hit IMDS).
        """
        env = os.environ
        # AWS_EXECUTION_ENV is set by Lambda, CodeBuild, and some
        # CodePipeline / SageMaker images; ECS injects the metadata
        # URI; the credentials-relative-URI is set on any compute
        # surface using a task / instance role.
        return (
            "AWS_LAMBDA_FUNCTION_NAME" in env
            or "AWS_BATCH_JOB_ID" in env
            or "AWS_EXECUTION_ENV" in env
            or "ECS_CONTAINER_METADATA_URI_V4" in env
            or "ECS_CONTAINER_METADATA_URI" in env
            or "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" in env
        )

    @classmethod
    def should_use_databricks_connect(cls) -> bool:
        """``True`` when the caller is outside Databricks but configured to reach a workspace.

        Inside Databricks runtime the regular ``SparkSession`` is already
        wired by the runtime — no Connect needed. Outside Databricks, the
        presence of ``DATABRICKS_HOST`` is the canonical signal that the
        caller wants to talk to a remote workspace; the SDK reads the same
        env vars to resolve auth and target compute.
        """
        env = os.environ
        return "DATABRICKS_RUNTIME_VERSION" not in env and "DATABRICKS_HOST" in env

    @classmethod
    def spark_session(
        cls,
        obj: Any = None,
        *,
        create: bool = False,
        connect: bool | None = None,
        import_error: bool = False,
        install_spark: bool = False,
        install_java: bool = False,
        local_setup: bool = True,
        extra_config: dict[str, str] | None = None,
    ) -> "SparkSession | None":
        """Return a cached SparkSession, creating one if needed.

        Resolution order:
        1. Return the cached session if already resolved.
        2. Check for an active SparkSession in the current process.
        3. When *create* is True, pick the build path:
           - ``connect=True`` (or ``None`` + :meth:`should_use_databricks_connect`)
             → :meth:`_bootstrap_connect_session` (``databricks.connect``).
           - Otherwise → :meth:`_bootstrap_session` (local PySpark with the
             ``yggdrasil.spark.setup`` helpers when *local_setup* is True).

        For richer Databricks Connect wiring (wheel publishing, ``DatabricksEnv``,
        ``addArtifacts``), use :meth:`DatabricksClient.spark` — it delegates the
        final ``getOrCreate()`` back here so the cache stays consistent.
        """
        # ------------------------------------------------------------------
        # Explicit-argument dispatch
        # ------------------------------------------------------------------
        if obj is not None:
            return cls._spark_session_from_obj(obj, import_error=import_error)

        # ------------------------------------------------------------------
        # Cached session
        # ------------------------------------------------------------------
        # A cached non-None session is final. A cached ``None`` (left behind
        # by an earlier ``create=False`` probe) should NOT block a later
        # ``create=True`` call from actually bringing up a session — that
        # bites every caller that probes the cache before asking for a real
        # one (e.g. :func:`yggdrasil.spark.tests._get_test_spark`).
        if cls._SPARK_SESSION is not MISSING and cls._SPARK_SESSION is not None:
            return cls._SPARK_SESSION
        if cls._SPARK_SESSION is None and not create:
            return None

        # ------------------------------------------------------------------
        # Ensure PySpark is importable
        # ------------------------------------------------------------------
        SparkSession = cls._import_spark_session(
            import_error=import_error, install_spark=install_spark
        )
        if SparkSession is None:
            cls._SPARK_SESSION = None
            return None

        # ------------------------------------------------------------------
        # Resolve a session: active → bootstrap → bare builder → None
        # ------------------------------------------------------------------
        try:
            active = SparkSession.getActiveSession()
        except Exception:
            active = None

        if active is not None:
            cls._SPARK_SESSION = active
        elif create:
            use_connect = connect if connect is not None else cls.should_use_databricks_connect()
            if use_connect:
                cls._SPARK_SESSION = cls._bootstrap_connect_session(
                    import_error=connect is True or import_error,
                    fallback_local=connect is None,
                    SparkSession=SparkSession,
                    local_setup=local_setup,
                    extra_config=extra_config,
                    install_java=install_java,
                )
            else:
                cls._SPARK_SESSION = cls._bootstrap_session(
                    SparkSession,
                    local_setup=local_setup,
                    extra_config=extra_config,
                    install_java=install_java,
                )
        else:
            cls._SPARK_SESSION = None

        return cls._SPARK_SESSION

    @classmethod
    def _spark_session_from_obj(
        cls, obj: Any, *, import_error: bool
    ) -> "SparkSession | None":
        """Resolve `obj` argument forms: Ellipsis, bool, or a SparkSession."""
        if obj is ...:
            if cls.in_databricks():
                return cls.spark_session(create=True, import_error=False)
            return None

        if isinstance(obj, bool):
            if obj:
                return cls.spark_session(create=True, import_error=False)
            return None

        SparkSession = cls._import_spark_session(
            import_error=import_error, install_spark=False
        )
        if SparkSession is not None and isinstance(obj, SparkSession):
            if cls._SPARK_SESSION is MISSING:
                cls._SPARK_SESSION = obj
            return obj

        raise TypeError(f"Invalid argument for spark_session: {obj!r}")

    @classmethod
    def _import_spark_session(
        cls, *, import_error: bool, install_spark: bool
    ) -> "type[SparkSession] | None":
        """Import pyspark.sql.SparkSession, optionally pip-installing first."""
        try:
            from pyspark.sql import SparkSession
            return SparkSession
        except ImportError:
            if not install_spark:
                if import_error:
                    raise
                return None
        except Exception:
            return None

        # install_spark path: install, then re-import
        runtime_import_module(module_name="pyspark", pip_name="pyspark", install=True)
        try:
            from pyspark.sql import SparkSession
            return SparkSession
        except Exception:
            if import_error:
                raise
            return None

    @classmethod
    def _bootstrap_session(
        cls,
        SparkSession: "type[SparkSession]",
        *,
        local_setup: bool,
        extra_config: dict[str, str] | None,
        install_java: bool = False,
    ) -> "SparkSession":
        """Create a local SparkSession, preferring yggdrasil.spark.setup helpers."""
        if not local_setup:
            return SparkSession.builder.getOrCreate()

        try:
            from yggdrasil.spark.setup import (
                configure_java_compat,
                create_local_session,
                ensure_hadoop_home,
                ensure_java,
                quiet_spark_loggers,
            )

            if install_java:
                ensure_java(auto_download=True)
            ensure_hadoop_home()
            configure_java_compat()
            # PySpark 4.x workers need an explicit Python interpreter
            # otherwise they can pick up a different one and crash on startup.
            os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
            os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
            session = create_local_session(extra_config=extra_config)
            quiet_spark_loggers()
            return session
        except Exception:
            logger.warning(
                "PyEnv.spark_session: local setup bootstrap failed, "
                "falling back to bare SparkSession.builder",
                exc_info=True,
            )
            return SparkSession.builder.getOrCreate()

    @classmethod
    def _bootstrap_connect_session(
        cls,
        *,
        import_error: bool,
        fallback_local: bool,
        SparkSession: "type[SparkSession]",
        local_setup: bool,
        extra_config: dict[str, str] | None,
        install_java: bool,
    ) -> "SparkSession":
        """Build a Databricks Connect SparkSession.

        Uses the ``DATABRICKS_*`` env vars the SDK already reads to resolve
        host / auth / target compute. When ``databricks-connect`` isn't
        installed and *fallback_local* is True, fall back to a local
        session; otherwise raise (or swallow per *import_error*).
        """
        try:
            from databricks.connect import DatabricksSession
        except ImportError:
            if not fallback_local:
                if import_error:
                    raise
                return cls._bootstrap_session(
                    SparkSession,
                    local_setup=local_setup,
                    extra_config=extra_config,
                    install_java=install_java,
                )
            logger.debug(
                "PyEnv.spark_session: databricks-connect not installed, "
                "falling back to local PySpark"
            )
            return cls._bootstrap_session(
                SparkSession,
                local_setup=local_setup,
                extra_config=extra_config,
                install_java=install_java,
            )

        from yggdrasil.databricks import DatabricksClient
        return DatabricksClient.current().spark()

    @classmethod
    def set_spark_session(cls, spark_session: "SparkSession"):
        cls._SPARK_SESSION = spark_session

    # ---------------------------------------------------------------------
    # uv resolution / runtime installation
    # ---------------------------------------------------------------------

    def _is_current_interpreter(self) -> bool:
        """``True`` when :attr:`python_path` points to the running interpreter."""
        return self.python_path.resolve() == Path(sys.executable).resolve()

    def _locate_uv(self) -> Path | None:
        """Look up ``uv`` without attempting to install it.

        Probe order: env-local binary, ``uv`` on PATH, ``python -m uv``.
        Returns the resolved binary path, the python_path sentinel for
        the module-launched form, or ``None`` if uv isn't reachable.
        """
        local = self.bin_path / ("uv.exe" if self.is_windows else "uv")
        if local.is_file():
            return local.resolve()

        which_uv = shutil.which("uv")
        if which_uv:
            return Path(which_uv).resolve()

        try:
            subprocess.run(
                [str(self.python_path), "-m", "uv", "--version"],
                cwd=str(self.cwd),
                env=dict(os.environ),
                text=True,
                capture_output=True,
                check=True,
            )
            return self.python_path
        except Exception:
            return None

    def has_uv(self) -> bool:
        """``True`` if ``uv`` is reachable for this interpreter."""
        return self._locate_uv() is not None

    def _run_pip_internal(self, *args: str) -> None:
        """Run pip through its internal API as a private fallback.

        Only safe for the current interpreter — pip's internal API
        mutates the running process, so it can't target a different
        interpreter.
        """
        if not self._is_current_interpreter():
            raise RuntimeError(
                "pip internal fallback is only supported for the current interpreter"
            )

        try:
            from pip._internal.cli.main import main as pip_main  # type: ignore
        except Exception as exc:
            raise RuntimeError("pip internal API is unavailable") from exc

        rc = pip_main(list(args))
        if rc != 0:
            raise RuntimeError(f"pip internal API failed with exit code {rc}")

    def ensure_uv(self, *, install_runtime: bool = True) -> Path | None:
        """
        Resolve ``uv`` for this environment and optionally install it at runtime.

        Resolution order:
        1. cached path
        2. env-local binary / PATH / ``python -m uv``
        3. install via ``python -m pip install uv`` (subprocess), with
           pip-internal-API fallback for the current interpreter.
        """
        if self._uv_bin and self._uv_bin.exists():
            return self._uv_bin

        located = self._locate_uv()
        if located is not None:
            self._uv_bin = located
            return self._uv_bin

        if not install_runtime:
            return None

        logger.info("PyEnv.ensure_uv: installing uv into runtime interpreter %s", self.python_path)

        try:
            subprocess.run(
                [str(self.python_path), "-m", "pip", "install", "uv"],
                cwd=str(self.cwd),
                env=dict(os.environ),
                text=True,
                capture_output=True,
                check=True,
            )
        except Exception as exc:
            # Private in-process pip API fallback — only safe for the
            # running interpreter.
            if not self._is_current_interpreter():
                raise RuntimeError(
                    "Failed to install uv with pip subprocess, and pip internal fallback "
                    "is only supported for the current interpreter."
                ) from exc

            try:
                self._run_pip_internal("install", "uv")
            except Exception as inner_exc:
                raise RuntimeError(
                    "Failed to install uv with both pip subprocess and pip internal fallback"
                ) from inner_exc

        located = self._locate_uv()
        if located is None:
            raise FileNotFoundError(
                f"Installed uv attempt completed for {self.python_path}, but no usable uv command was found"
            )
        self._uv_bin = located
        return self._uv_bin

    @property
    def uv_path(self) -> Path:
        """
        Resolve the ``uv`` command, installing it into the runtime interpreter
        if needed.
        """
        uv_ref = self.ensure_uv(install_runtime=True)
        if uv_ref is None:
            raise FileNotFoundError("uv is unavailable")
        return uv_ref

    def _uv_base_cmd(self, *, install_runtime: bool = True) -> list[str]:
        """
        Return the command prefix for invoking ``uv``.

        If ``uv`` exists as a binary, returns ``[<uv>]``.
        If only the module is available, returns ``[<python>, '-m', 'uv']``.
        """
        uv_ref = self.ensure_uv(install_runtime=install_runtime)
        if uv_ref is None:
            raise FileNotFoundError("uv is unavailable")

        if uv_ref.resolve() == self.python_path.resolve():
            return [str(self.python_path), "-m", "uv"]

        return [str(uv_ref)]

    def install(
        self,
        *packages: str,
        requirements: str | Path | None = None,
        extra_args: Sequence[str] = (),
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        prefer_uv: bool | None = None,
        target: Path | str | None = None,
        break_system_packages: bool = False,
    ) -> SystemCommand | None:
        """
        Install packages into the environment anchored by :attr:`python_path`.

        Fallback behavior
        -----------------
        1. Try normal subprocess install (uv pip or python -m pip)
        2. If that fails and this env is the current interpreter, try pip internal API
        """
        result: SystemCommand | None = None
        prefer_uv = self.prefer_uv if prefer_uv is None else prefer_uv

        if not packages and requirements is None:
            return None

        wait_cfg = WaitingConfig.from_(wait)
        pip_cmd = self._pip_cmd_args("install", prefer_uv=prefer_uv)
        tmp_req: Path | None = None

        if requirements is not None:
            req_path = Path(requirements).expanduser()
            if req_path.exists():
                pip_cmd += ["-r", str(req_path)]
            else:
                self.cwd.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(
                    prefix="requirements_",
                    suffix=".txt",
                    dir=str(self.cwd),
                )
                tmp_req = Path(name)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(str(requirements).strip() + "\n")
                pip_cmd += ["-r", str(tmp_req)]

        if packages:
            pip_cmd += [str(safe_pip_name(p)) for p in packages]

        if extra_args:
            pip_cmd += list(extra_args)

        if target:
            target_path = Path(target).expanduser()
            target_path.mkdir(parents=True, exist_ok=True)
            pip_cmd += ["--target", str(target_path)]

        try:
            result = SystemCommand.run_lazy(pip_cmd, cwd=self.cwd).wait(
                wait=wait_cfg,
                raise_error=True,
            )
        except Exception as exc:
            if self._is_externally_managed_failure(exc):
                if break_system_packages:
                    pip_break_cmd = pip_cmd + ["--break-system-packages"]
                    return SystemCommand.run_lazy(pip_break_cmd, cwd=self.cwd).wait(
                        wait=wait_cfg,
                        raise_error=raise_error,
                    )

                raise RuntimeError(
                    f"Cannot install into externally managed interpreter: {self.python_path}. "
                    "Use a virtualenv, pass target=..., or set break_system_packages=True "
                    "to explicitly allow --break-system-packages."
                ) from exc

            if not raise_error:
                logger.warning(
                    "PyEnv.install: subprocess install failed and raise_error=False",
                    exc_info=True,
                )
                return None

            # Private fallback only for current interpreter and synchronous execution
            if not wait_cfg:
                raise

            if not self._is_current_interpreter():
                raise

            logger.warning(
                "PyEnv.install: subprocess install failed; falling back to pip internal API",
                exc_info=True,
            )

            fallback_args = ["install"]

            if requirements is not None:
                if tmp_req is not None:
                    fallback_args += ["-r", str(tmp_req)]
                else:
                    fallback_args += ["-r", str(req_path)]

            if packages:
                fallback_args += [str(safe_pip_name(p)) for p in packages]

            if extra_args:
                fallback_args += list(extra_args)

            if target:
                fallback_args += ["--target", str(target_path)]

            try:
                self._run_pip_internal(*fallback_args)
            except Exception as inner_exc:
                raise RuntimeError(
                    "Failed to install packages with both subprocess pip and pip internal fallback"
                ) from inner_exc

        finally:
            if wait_cfg and tmp_req is not None:
                try:
                    tmp_req.unlink(missing_ok=True)
                except Exception:
                    logger.warning(
                        "PyEnv.install: failed to remove temp requirements file=%s",
                        tmp_req,
                        exc_info=True,
                    )

        return result

    def update(
        self,
        *packages: str,
        extra_args: Sequence[str] = (),
        wait: WaitingConfigArg = True,
        prefer_uv: bool | None = None,
    ) -> SystemCommand | None:
        if not packages:
            return None

        cmd = self._pip_cmd_args("install", prefer_uv=prefer_uv) + [
            "--upgrade",
            *[str(safe_pip_name(p)) for p in packages],
            *extra_args,
        ]

        wait_cfg = WaitingConfig.from_(wait)

        try:
            return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait_cfg, raise_error=True)
        except Exception:
            if not wait_cfg or not self._is_current_interpreter():
                raise

            logger.warning(
                "PyEnv.update: subprocess update failed; falling back to pip internal API",
                exc_info=True,
            )
            self._run_pip_internal(
                "install",
                "--upgrade",
                *[str(safe_pip_name(p)) for p in packages],
                *extra_args,
            )
            return None

    def uninstall(
        self,
        *packages: str,
        extra_args: Sequence[str] = (),
        wait: WaitingConfigArg = True,
        prefer_uv: bool | None = None,
    ) -> SystemCommand | None:
        """
        Uninstall one or more packages from the anchored environment.
        """
        if not packages:
            return None

        cmd = self._pip_cmd_args("uninstall", prefer_uv=prefer_uv) + [
            *[str(safe_pip_name(p)) for p in packages],
            *extra_args,
        ]
        return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait)

    def pip(
        self,
        *args: str,
        wait: WaitingConfigArg = True,
        prefer_uv: bool | None = None,
    ) -> SystemCommand:
        """
        Run an arbitrary pip subcommand against this environment.

        First positional ``args`` element is the subcommand
        (``install``, ``freeze``, ``list``, …); the rest are passed
        through verbatim. With ``prefer_uv=True`` the call goes
        through ``uv pip <subcommand> --python <p>`` so installs land
        in the venv that owns ``python_path``.
        """
        if not args:
            raise ValueError("pip() requires at least the subcommand argument")
        subcommand, rest = args[0], args[1:]
        cmd = self._pip_cmd_args(subcommand, prefer_uv=prefer_uv) + list(rest)
        logger.debug("PyEnv.pip: cmd=%s cwd=%s", cmd, self.cwd)
        return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait)

    def delete(self, raise_error: bool = True) -> None:
        """
        Delete the virtual environment that contains this interpreter.
        """
        if self.is_current:
            raise ValueError("Cannot delete the current singleton PyEnv.")

        candidate = self.python_path.parent
        venv_root: Path | None = None

        for _ in range(4):
            if (candidate / "pyvenv.cfg").exists():
                venv_root = candidate
                break
            candidate = candidate.parent

        if venv_root is None:
            if raise_error:
                raise ValueError(
                    f"Cannot determine venv root for python_path={self.python_path!r}. "
                    "No pyvenv.cfg found within 4 parent levels."
                )
            venv_root = self.python_path.parent.parent

        try:
            shutil.rmtree(venv_root, ignore_errors=not raise_error)
        except Exception as exc:
            raise RuntimeError(f"Failed to delete venv at {venv_root}: {exc}") from exc

        logger.info("PyEnv.delete: removed venv_root=%s", venv_root)

    # ---------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------

    def run_python_code(
        self,
        code: str,
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        stdin: str | None = None,
        python: PyEnv | Path | str | None = None,
        packages: list[str] | None = None,
        prefer_uv: bool | None = None,
        globs: dict[str, Any] | None = None,
        auto_install: bool = False,
    ) -> SystemCommand:
        """
        Execute Python source code in a subprocess under this or another env.
        """
        merged_env = {**os.environ, **(env or {})}

        if python is None:
            target = self
        elif isinstance(python, PyEnv):
            target = python
        else:
            target = self.get_or_create(identifier=python)

        prefer_uv = target.prefer_uv if prefer_uv is None else prefer_uv

        if packages:
            target.install(*packages)

        if globs:
            prefix = "\n".join(f"{k} = {v!r}" for k, v in globs.items())
            code = prefix + "\n" + code

        if prefer_uv:
            try:
                cmd = target._uv_run_prefix() + ["python", "-c", code]
            except Exception:
                logger.warning(
                    "PyEnv.run_python_code: uv unavailable for %s, falling back to bare python",
                    target.python_path,
                    exc_info=True,
                )
                cmd = [str(target.python_path), "-c", code]
        else:
            cmd = [str(target.python_path), "-c", code]

        proc = SystemCommand.run_lazy(
            cmd,
            cwd=cwd or target.cwd,
            env=merged_env,
            python=target,
        )

        if stdin is not None:
            try:
                proc.popen.stdin.write(stdin)
                proc.popen.stdin.flush()
                proc.popen.stdin.close()
            except Exception:
                logger.warning("PyEnv.run_python_code: failed writing stdin", exc_info=True)

        return proc.wait(wait=wait, raise_error=raise_error, auto_install=auto_install)

    # ---------------------------------------------------------------------
    # Runtime import + auto-install
    # ---------------------------------------------------------------------

    @classmethod
    def runtime_import_module(
        cls,
        module_name: str | None = None,
        *,
        install: bool = True,
        pip_name: str | None = None,
        upgrade: bool = False,
        warn: bool = True,
        use_cache: bool = True,
    ):
        """
        Class-level convenience wrapper for :meth:`import_module`.
        """
        return cls.current().import_module(
            module_name=module_name,
            install=install,
            pip_name=pip_name,
            upgrade=upgrade,
            warn=warn,
            use_cache=use_cache
        )

    def import_module(
        self,
        module_name: str | None = None,
        *,
        wait: WaitingConfigArg = True,
        install: bool = True,
        pip_name: str | None = None,
        upgrade: bool = False,
        warn: bool = False,
        use_cache: bool = False,
    ):
        """
        Import a module into the current interpreter, installing it if missing.
        """
        if not module_name:
            if not pip_name:
                raise ValueError("Provide at least one of module_name or pip_name.")
            module_name = pip_name.replace("-", "_")

        if use_cache and not upgrade and module_name in self._checked_modules:
            cached = sys.modules.get(module_name)
            if cached is not None:
                return cached

        if not upgrade:
            try:
                imported = importlib.import_module(module_name)
                if use_cache:
                    self._checked_modules.add(module_name)
                return imported
            except ModuleNotFoundError:
                if "pyspark" in module_name or not install:
                    raise

        try:
            resolved_pip_name = pip_name or str(safe_pip_name(module_name))

            if warn:
                print(
                    f"Auto-installing '{resolved_pip_name}' into environment {self.python_path} "
                    f"because module '{module_name}' was not found.",
                    file=sys.stderr,
                )

            self.install(resolved_pip_name, wait=wait, raise_error=True)

            importlib.invalidate_caches()
            imported = importlib.import_module(module_name)
            if use_cache:
                self._checked_modules.add(module_name)
            return imported
        except Exception as exc:
            raise ModuleNotFoundError(
                f"No module named '{module_name}' and auto-install package '{pip_name or module_name}'",
                name=module_name,
            ) from exc

    @staticmethod
    def get_root_module_directory(module_name: str) -> Path:
        """
        Return the filesystem directory of the root package/module.
        """
        if not module_name or not module_name.strip():
            raise ValueError("module_name must be a non-empty string")

        root_module = module_name.split(".", 1)[0]

        # Special-case __main__ because find_spec("__main__") may raise
        # ValueError when __main__.__spec__ is None (common for direct script execution).
        if root_module == "__main__":
            main_mod = sys.modules.get("__main__")
            main_file = getattr(main_mod, "__file__", None)
            if main_file:
                return Path(main_file).resolve().parent
            raise FileNotFoundError(
                "Module '__main__' has no filesystem directory "
                "(likely running interactively or in an environment without __file__)"
            )

        try:
            spec = importlib.util.find_spec(root_module)
        except ValueError as e:
            # Fallback for odd import states / partially initialized modules
            mod = sys.modules.get(root_module)
            mod_file = getattr(mod, "__file__", None) if mod else None
            if mod_file:
                return Path(mod_file).resolve().parent
            raise ModuleNotFoundError(f"Cannot determine module spec for '{root_module}'") from e

        if spec is None:
            raise ModuleNotFoundError(f"Cannot find module '{root_module}'")

        if spec.submodule_search_locations:
            return Path(next(iter(spec.submodule_search_locations))).resolve()

        if spec.origin and spec.origin not in {"built-in", "frozen"}:
            return Path(spec.origin).resolve().parent

        raise FileNotFoundError(
            f"Module '{root_module}' has no filesystem directory (origin={spec.origin!r})"
        )

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _is_externally_managed_failure(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "externally-managed-environment" in text
            or "externally managed" in text
            or "this environment is externally managed" in text
        )

    def _pip_cmd_args(
        self,
        subcommand: str,
        *,
        python: str | Path | None = None,
        prefer_uv: bool | None = None,
    ) -> list[str]:
        """
        Build a ``pip <subcommand>`` invocation prefix anchored on ``python_path``.

        * prefer_uv=True  -> ``uv pip <subcommand> --python <python>``
        * prefer_uv=False -> ``<python> -m pip <subcommand>``

        ``uv run --python <p> python -m pip`` was the historical form,
        but ``uv run`` provisions an ephemeral managed env when ``<p>``
        isn't a project venv, so installs leaked out of the caller's
        venv. ``uv pip --python <p>`` targets the venv that owns
        ``<p>`` directly.
        """
        prefer_uv = self.prefer_uv if prefer_uv is None else prefer_uv
        p = Path(python).expanduser().resolve() if python is not None else self.python_path

        if prefer_uv:
            try:
                return [*self._uv_base_cmd(install_runtime=True), "pip", subcommand, "--python", str(p)]
            except Exception:
                logger.warning(
                    "PyEnv._pip_cmd_args: uv unavailable for %s, falling back to pip",
                    p,
                    exc_info=True,
                )

        return [str(p), "-m", "pip", subcommand]

    def _uv_run_prefix(self, python: str | Path | None = None) -> list[str]:
        """
        Return the ``uv run --python <path>`` prefix for subprocess execution.
        """
        p = Path(python).expanduser().resolve() if python is not None else self.python_path
        return [*self._uv_base_cmd(install_runtime=True), "run", "--python", str(p)]

    @staticmethod
    def _venv_python_from_dir(venv_dir: Path, raise_error: bool = True) -> Path:
        """
        Locate the Python executable inside a venv directory.
        """
        venv_dir = venv_dir.expanduser().resolve()

        if os.name == "nt":
            candidates = [
                venv_dir / "Scripts" / "python.exe",
                venv_dir / "Scripts" / "python",
            ]
        else:
            candidates = [
                venv_dir / "bin" / "python",
                venv_dir / "bin" / "python3",
            ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        if raise_error:
            raise ValueError(f"No Python executable found inside venv: {venv_dir}")

        return candidates[0]

    @staticmethod
    def _looks_like_path(s: str) -> bool:
        """
        Return ``True`` if *s* resembles a filesystem path rather than a bare
        name or version selector.
        """
        if not s:
            return False
        if s.startswith(("~", ".", "/")):
            return True
        if os.name == "nt" and len(s) >= 2 and s[1] == ":":
            return True
        return "/" in s or "\\" in s


def runtime_import_module(
    module_name: str | None = None,
    *,
    install: bool = True,
    pip_name: str | None = None,
    upgrade: bool = False,
    warn: bool = True,
    use_cache: bool = True,
):
    """
    Module-level convenience wrapper for :meth:`PyEnv.runtime_import_module`.
    """
    return PyEnv.runtime_import_module(
        module_name=module_name,
        install=install,
        pip_name=pip_name,
        upgrade=upgrade,
        warn=warn,
        use_cache=use_cache
    )