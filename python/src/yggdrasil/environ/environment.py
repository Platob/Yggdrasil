"""
pyenv.py — Thin wrapper around a Python interpreter for environment management.

Provides :class:`PyEnv`, a dataclass that anchors all package-management and
subprocess-execution operations to a single Python interpreter path. It
preferentially delegates to ``uv`` for speed but falls back to ``pip`` / bare
``python`` when ``uv`` is unavailable or explicitly disabled.

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
    (e.g. ``'yaml'`` → ``'PyYAML'``).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.version import VersionInfo
from .system_command import SystemCommand
from .userinfo import UserInfo

__all__ = [
    "runtime_import_module",
    "PyEnv",
    "PIP_MODULE_NAME_MAPPINGS",
    "CURRENT_PYENV",
    "SYSTEM_LIBS",
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

SYSTEM_LIBS: list[str] = [
    # Core Python / packaging
    "pip",
    "setuptools",
    "wheel",
    "packaging",
    "importlib-metadata",
    "zipp",
    "typing-extensions",
    "typing-inspection",
    "annotated-types",
    "annotated-doc",
    "iniconfig",
    "pluggy",
    "platformdirs",
    "pathspec",
    "filelock",
    "distro",
    "six",
    "decorator",
    "wrapt",
    "deprecated",
    "future",
    "more-itertools",
    "toolz",
    "attrs",
    "rpds-py",
    "referencing",
    "jsonschema",
    "jsonschema-specifications",
    "blessed",
    # Networking / HTTP
    "certifi",
    "charset-normalizer",
    "idna",
    "urllib3",
    "requests",
    "requests-toolbelt",
    "httpx-sse",
    "httpcore",
    "h11",
    "anyio",
    "sniffio",
    "aiofiles",
    "aiosignal",
    "aiohappyeyeballs",
    "frozenlist",
    "multidict",
    "yarl",
    "propcache",
    "websocket-client",
    "dnspython",
    "brotli",
    "greenlet",
    "gpustat",
    "h5py",
    "librt",
    "nvidia-ml-py",
    "opencv-python",
    "proglog",
    "pydub",
    "requests-kerberos",
    "watch",
    "wcwidth",
    "av",
    "uvicorn",
    "ansicon",
    "absl-py",
    # Crypto / Auth
    "cryptography",
    "cffi",
    "pycparser",
    "pyopenssl",
    "oauthlib",
    "requests-oauthlib",
    "pyasn1",
    "pyasn1-modules",
    "rsa",
    "id",
    # Serialization / Data formats
    "pyyaml",
    "tomlkit",
    "jiter",
    "jinja2",
    "markupsafe",
    "grpcio",
    "grpcio-status",
    "googleapis-common-protos",
    "proto-plus",
    "cloudpickle",
    "multiprocess",
    "locket",
    "partd",
    # Stdlib extensions
    "python-dateutil",
    "tzlocal",
    "isodate",
    "aniso8601",
    "regex",
    "click",
    "rich",
    "pygments",
    "markdown",
    "markdown-it-py",
    "mdurl",
    "nh3",
    "colorama",
    "tqdm",
    "psutil",
    "gitpython",
    "gitdb",
    "smmap",
    "loguru",
    "sentry-sdk",
    "python-dotenv",
    "pydantic-core",
    "pydantic-settings",
    "aiohttp",
    "cachetools",
    # Build / Dev / CI tools
    "build",
    "pyproject-hooks",
    "black",
    "mypy",
    "mypy-extensions",
    "ruff",
    "pytest",
    "pytest-asyncio",
    "pathlib2",
    "semantic-version",
    "twine",
    "readme-renderer",
    "docutils",
    "rfc3986",
    "keyring",
    "jaraco-classes",
    "jaraco-context",
    "jaraco-functools",
    "uv",
    # Numeric / Scientific
    "sympy",
    "mpmath",
    "arro3-core",
    "polars-runtime-32",
    "xarray",
    "contourpy",
    "cycler",
    "fonttools",
    "kiwisolver",
    "pyparsing",
    "matplotlib",
    "pillow",
    "imageio",
    # Cloud / Object storage
    "botocore",
    "jmespath",
    "s3transfer",
    "google-auth",
    "google-auth-oauthlib",
    "google-api-core",
    "google-cloud-core",
    "google-cloud-storage",
    "google-cloud-storage-control",
    "google-resumable-media",
    "google-crc32c",
    "googleapis-common-protos",
    "grpc-google-iam-v1",
    # ML / AI
    "torch",
    "torchvision",
    "torchdata",
    "tokenizers",
    "safetensors",
    "sentencepiece",
    "datasets",
    "huggingface-hub",
    "hf-xet",
    "accelerate",
    "peft",
    "diffusers",
    "timm",
    "einops",
    "tensorboard",
    "tensorboard-data-server",
    "wandb",
    "networkx",
    # Web / API frameworks
    "starlette",
    "uvicorn",
    "werkzeug",
    "itsdangerous",
    "blinker",
    "python-multipart",
    "sse-starlette",
    "gradio",
    "gradio-client",
    "safehttpx",
    "mcp",
    "openai",
    # Spark / Big data
    "pyspark",
    "py4j",
    "delta-spark",
    # Misc utilities
    "beautifulsoup4",
    "soupsieve",
    "filelock",
    "typer",
    "typer-slim",
    "shellingham",
    "remote-pdb",
    "groovy",
    "pytokens",
    "unitycatalog",
    "pyproject-hooks",
]


def safe_pip_name(
    value: str | tuple[str, str] | Iterable[str | tuple[str, str]],
) -> str | list[str]:
    """
    Map an import name to its pip distribution name, or return it unchanged.

    Accepts either a single string, a ``(name, version)`` tuple, or an iterable
    of either. The return type mirrors the input shape.

    Examples
    --------
    ::

        safe_pip_name("yaml")                # -> "PyYAML"
        safe_pip_name(("pyarrow", "18.1.0")) # -> "pyarrow==18.1.0"
        safe_pip_name(["yaml", "jwt"])       # -> ["PyYAML", "PyJWT"]
    """
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
        wait: WaitingConfigArg | None = True,
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
        return CURRENT_PYENV is not None and CURRENT_PYENV is self

    @property
    def is_windows(self) -> bool:
        return os.name == "nt"

    @property
    def bin_path(self) -> Path:
        return self.python_path.parent

    @property
    def root_path(self) -> Path:
        return self.bin_path.parent

    @property
    def userinfo(self) -> UserInfo:
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

    # ---------------------------------------------------------------------
    # uv resolution / runtime installation
    # ---------------------------------------------------------------------

    def has_uv(self) -> bool:
        """
        Return ``True`` if ``uv`` appears usable for this interpreter.
        """
        local = self.bin_path / ("uv.exe" if self.is_windows else "uv")
        if local.is_file():
            return True

        if shutil.which("uv"):
            return True

        try:
            subprocess.run(
                [str(self.python_path), "-m", "uv", "--version"],
                cwd=str(self.cwd),
                env=dict(os.environ),
                text=True,
                capture_output=True,
                check=True,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _install_package_with_pip_api(package: str) -> None:
        """
        Install a package into this interpreter using pip's internal Python API
        instead of spawning a public CLI command.

        Notes
        -----
        This is intentionally a fallback path. pip's internal API is not stable,
        but it works well enough as a last-resort bootstrap mechanism.
        """
        try:
            from pip._internal.cli.main import main as pip_main  # type: ignore
        except Exception as exc:
            raise RuntimeError("pip internal API is unavailable") from exc

        argv = ["install", package]
        rc = pip_main(argv)
        if rc != 0:
            raise RuntimeError(f"pip internal API failed installing {package!r} with exit code {rc}")

    def _is_current_interpreter(self) -> bool:
        return self.python_path.resolve() == Path(sys.executable).resolve()

    def ensure_uv(self, *, install_runtime: bool = True) -> Path | None:
        """
        Resolve ``uv`` for this environment and optionally install it at runtime.

        Resolution order:
        1. local env script
        2. uv on PATH
        3. python -m uv
        4. install via pip subprocess
        5. install via pip internal API fallback
        """
        if self._uv_bin and self._uv_bin.exists():
            return self._uv_bin

        local = self.bin_path / ("uv.exe" if self.is_windows else "uv")
        if local.is_file():
            self._uv_bin = local.resolve()
            return self._uv_bin

        which_uv = shutil.which("uv")
        if which_uv:
            self._uv_bin = Path(which_uv).resolve()
            return self._uv_bin

        try:
            subprocess.run(
                [str(self.python_path), "-m", "uv", "--version"],
                cwd=str(self.cwd),
                env=dict(os.environ),
                text=True,
                capture_output=True,
                check=True,
            )
            self._uv_bin = self.python_path
            return self._uv_bin
        except Exception:
            pass

        if not install_runtime:
            return None

        logger.info("PyEnv.ensure_uv: installing uv into runtime interpreter %s", self.python_path)

        install_errors: list[Exception] = []

        # First try normal isolated subprocess install
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
            install_errors.append(exc)

            # Fallback: private in-process pip API, only when this PyEnv points to
            # the current interpreter. We cannot safely mutate a different interpreter
            # in-process.
            if self.python_path.resolve() != Path(sys.executable).resolve():
                raise RuntimeError(
                    "Failed to install uv with pip subprocess, and pip internal fallback "
                    "is only supported for the current interpreter."
                ) from exc

            try:
                self._install_package_with_pip_api("uv")
            except Exception as inner_exc:
                install_errors.append(inner_exc)
                raise RuntimeError(
                    "Failed to install uv with both pip subprocess and pip internal fallback"
                ) from inner_exc

        if local.is_file():
            self._uv_bin = local.resolve()
            return self._uv_bin

        which_uv = shutil.which("uv")
        if which_uv:
            self._uv_bin = Path(which_uv).resolve()
            return self._uv_bin

        try:
            subprocess.run(
                [str(self.python_path), "-m", "uv", "--version"],
                cwd=str(self.cwd),
                env=dict(os.environ),
                text=True,
                capture_output=True,
                check=True,
            )
            self._uv_bin = self.python_path
            return self._uv_bin
        except Exception as exc:
            raise FileNotFoundError(
                f"Installed uv attempt completed for {self.python_path}, but no usable uv command was found"
            ) from exc

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

    # ---------------------------------------------------------------------
    # Package management
    # ---------------------------------------------------------------------

    def requirements(
        self,
        prefer_uv: bool | None = None,
        *,
        with_system: bool = False,
    ) -> list[tuple[str, str]]:
        """
        Return installed packages as ``(name, version)`` tuples, sorted by name.
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

        out: list[tuple[str, str]] = []
        for item in pkgs:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()
            version = str(item.get("version", "")).strip()
            if not name or not version:
                continue

            name_l = name.lower()
            if not with_system:
                if name_l in SYSTEM_LIBS:
                    continue
                if name_l.startswith(("test-", "test_", "win32", "pywin32")):
                    continue

            out.append((name, version))

        return sorted(out, key=lambda x: x[0].lower())

    def _run_pip_internal(self, *args: str) -> None:
        """
        Run pip through its internal API as a private fallback.

        Only safe for the current interpreter.
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

    def install(
        self,
        *packages: str,
        requirements: str | Path | None = None,
        extra_args: Sequence[str] = (),
        wait: WaitingConfigArg | None = True,
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
        prefer_uv = self.prefer_uv if prefer_uv is None else prefer_uv

        if not packages and requirements is None:
            return None

        wait_cfg = WaitingConfig.check_arg(wait)
        pip_cmd = self._pip_cmd_args(prefer_uv=prefer_uv) + ["install"]
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
            return result
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

            result = None

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
        wait: WaitingConfigArg | None = True,
        prefer_uv: bool | None = None,
    ) -> SystemCommand | None:
        if not packages:
            return None

        cmd = self._pip_cmd_args(prefer_uv=prefer_uv) + [
            "install",
            "--upgrade",
            *[str(safe_pip_name(p)) for p in packages],
            *extra_args,
        ]

        wait_cfg = WaitingConfig.check_arg(wait)

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
        wait: WaitingConfigArg | None = True,
        prefer_uv: bool | None = None,
    ) -> SystemCommand | None:
        """
        Uninstall one or more packages from the anchored environment.
        """
        if not packages:
            return None

        cmd = self._pip_cmd_args(prefer_uv=prefer_uv) + [
            "uninstall",
            *[str(safe_pip_name(p)) for p in packages],
            *extra_args,
        ]
        return SystemCommand.run_lazy(cmd, cwd=self.cwd).wait(wait)

    def pip(
        self,
        *args: str,
        wait: WaitingConfigArg | None = True,
        prefer_uv: bool | None = None,
    ) -> SystemCommand:
        """
        Run an arbitrary pip subcommand against this environment.
        """
        cmd = self._pip_cmd_args(prefer_uv=prefer_uv) + list(args)
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
        wait: WaitingConfigArg | None = True,
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

        if not upgrade:
            try:
                imported = importlib.import_module(module_name)
                if use_cache:
                    self._checked_modules.add(module_name)
                return imported
            except ModuleNotFoundError:
                if not install:
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
                f"Failed to import module '{module_name}' and auto-install package '{pip_name or module_name}'",
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
        spec = importlib.util.find_spec(root_module)

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
        python: str | Path | None = None,
        prefer_uv: bool | None = None,
    ) -> list[str]:
        """
        Build the base pip invocation prefix.

        * prefer_uv=True  -> ``uv pip --python <python>``
        * prefer_uv=False -> ``<python> -m pip``
        """
        prefer_uv = self.prefer_uv if prefer_uv is None else prefer_uv
        p = Path(python).expanduser().resolve() if python is not None else self.python_path

        if prefer_uv:
            try:
                return [*self._uv_base_cmd(install_runtime=True), "pip", "--python", str(p)]
            except Exception:
                logger.warning(
                    "PyEnv._pip_cmd_args: uv unavailable for %s, falling back to pip",
                    p,
                    exc_info=True,
                )

        return [str(p), "-m", "pip"]

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