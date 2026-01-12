"""Manage isolated Python environments with uv/pip helpers."""

# yggdrasil/pyutils/python_env.py
from __future__ import annotations

import ast
import base64
import datetime as _dt
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Optional, Union, List, Tuple

from yggdrasil.pyutils.modules import PipIndexSettings

log = logging.getLogger(__name__)


class PythonEnvError(RuntimeError):
    """Raised when Python environment operations fail."""

    pass


# -----------------------
# module-level locks (requested)
# -----------------------
_ENV_LOCKS: dict[str, threading.RLock] = {}
_UV_LOCK: threading.RLock = threading.RLock()
_LOCKS_GUARD: threading.RLock = threading.RLock()


# -----------------------
# tiny helpers
# -----------------------

# Installed into newly created envs (and on create() upsert when env already exists)
DEFAULT_CREATE_PACKAGES: tuple[str, ...] = ("uv", "ygg")
_NON_PIPABLE_NAMES = (
    "python-apt",
    "python3-apt",
    "unattended-upgrades",
    "apt",
    "apt-utils",
    "dpkg",
    "adduser",
    "lsb-release",
    "software-properties-common",
    "systemd",
    "udev",
    "dbus",
)

# Matches a requirement line that starts with one of those names, followed by:
#  - end of line
#  - a version specifier (==,>=, etc)
#  - extras ([...])
#  - an environment marker (; ...)
_NON_PIPABLE_RE = re.compile(
    r"^\s*(?:"
    + "|".join(re.escape(n) for n in _NON_PIPABLE_NAMES)
    + r")(?=\s*(?:$|==|~=|!=|<=|>=|<|>|\[|;))",
    re.IGNORECASE,
)



def _filter_non_pipable_linux_packages(requirements: Iterable[str]) -> List[str]:
    """
    Remove Linux OS-level packages that aren't realistically installable via pip/uv on Databricks.
    Keeps comments/blank lines as-is.
    """
    out: List[str] = []
    for line in requirements:
        s = line.strip()

        # keep empty lines and comments
        if not s or s.startswith("#"):
            out.append(line)
            continue

        if _NON_PIPABLE_RE.match(s):
            continue

        out.append(line)

    return out


def _is_windows() -> bool:
    """Return True when running on Windows."""
    return os.name == "nt"


def _norm_env(base: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return a normalized environment dict with common defaults."""
    env = dict(base or os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _split_on_tag(stdout: str, tag: str) -> tuple[list[str], Optional[str]]:
    """Split stdout into lines before the tag and the tag payload."""
    lines = (stdout or "").splitlines()
    before: list[str] = []
    payload: Optional[str] = None
    for line in lines:
        if payload is None and line.startswith(tag):
            payload = line[len(tag) :]
            continue
        if payload is None:
            before.append(line)
    return before, payload


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    """Return a de-duplicated list while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        x = str(x).strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    native_tls: bool = False,
    extra_index_url: Optional[Union[str, List[str]]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command and raise a PythonEnvError on failure."""
    cmd_s = [str(x) for x in cmd]

    if native_tls and "--native-tls" not in cmd_s:
        cmd_s = cmd_s + ["--native-tls"]

    if extra_index_url and "--extra-index-url" not in cmd_s:
        if isinstance(extra_index_url, list):
            extra_index_url = " ".join(extra_index_url)

        cmd_s = cmd_s + ["--extra-index-url", extra_index_url]

    log.debug("exec: %s", " ".join(cmd_s))
    if cwd:
        log.debug("cwd: %s", str(cwd))

    p = subprocess.run(
        cmd_s,
        cwd=str(cwd) if cwd else None,
        env=_norm_env(env),
        text=True,
        capture_output=True,
    )
    if p.returncode != 0:
        raise PythonEnvError(
            f"Command failed ({p.returncode})\n"
            f"--- cmd ---\n{cmd_s}\n"
            f"--- stdout ---\n{p.stdout}\n"
            f"--- stderr ---\n{p.stderr}\n"
        )
    return p


# -----------------------
# user dirs + naming
# -----------------------

def _user_python_dir() -> Path:
    """Return the base directory for user Python tooling."""
    return (Path.home() / ".python").expanduser().resolve()


def _user_envs_dir() -> Path:
    """Return the directory where user environments are stored."""
    return (_user_python_dir() / "envs").resolve()


def _safe_env_name(name: str) -> str:
    """Normalize and validate an environment name."""
    n = (name or "").strip()
    if not n:
        raise PythonEnvError("Env name cannot be empty")
    n = re.sub(r"[^A-Za-z0-9._-]+", "-", n).strip("-")
    if not n or n in (".", ".."):
        raise PythonEnvError(f"Invalid env name after sanitizing: {name!r}")
    return n


# -----------------------
# uv install (pip-based)
# -----------------------

def _uv_exe_on_path() -> Optional[str]:
    """Return the uv executable path if available."""
    uv = shutil.which("uv")
    return uv


def _current_env_script(name: str) -> Optional[Path]:
    """Return a script path from the current environment, if present."""
    exe = Path(sys.executable).resolve()
    bindir = exe.parent
    if _is_windows():
        if bindir.name.lower() != "scripts":
            scripts = Path(sys.prefix).resolve() / "Scripts"
            bindir = scripts if scripts.is_dir() else bindir
        cand = bindir / (name + ".exe")
        return cand if cand.exists() else None
    cand = bindir / name
    return cand if cand.exists() else None


def _ensure_pip_available(*, check: bool = True) -> None:
    """Ensure pip is available, optionally raising on failure."""
    log.debug("checking pip availability")
    p = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        text=True,
        capture_output=True,
        env=_norm_env(),
    )
    if p.returncode == 0:
        return

    log.warning("pip not available; attempting ensurepip bootstrap")
    b = subprocess.run(
        [sys.executable, "-m", "ensurepip", "--upgrade"],
        text=True,
        capture_output=True,
        env=_norm_env(),
    )
    if check and b.returncode != 0:
        raise PythonEnvError(
            "pip is not available and ensurepip failed.\n"
            f"--- stdout ---\n{b.stdout}\n"
            f"--- stderr ---\n{b.stderr}\n"
        )


def _pip_install_uv_in_current(
    *,
    upgrade: bool = True,
    user: bool = True,
    index_url: Optional[str] = None,
    extra_pip_args: Optional[Iterable[str]] = None,
    check: bool = True,
) -> None:
    """Install uv in the current environment using pip."""
    _ensure_pip_available(check=check)

    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("-U")
    if user:
        cmd.append("--user")
    if index_url:
        cmd += ["--index-url", index_url]
    if extra_pip_args:
        cmd += list(extra_pip_args)
    cmd.append("uv")

    log.info("installing uv via pip (upgrade=%s user=%s)", upgrade, user)
    _run_cmd(cmd, env=None)


# -----------------------
# locking (module-level)
# -----------------------

def _env_lock_key(root: Path) -> str:
    """Return a normalized lock key for an environment path."""
    return str(Path(root).expanduser().resolve())


def _get_env_lock(root: Path) -> threading.RLock:
    """Return a re-entrant lock for a specific environment root."""
    key = _env_lock_key(root)
    with _LOCKS_GUARD:
        lk = _ENV_LOCKS.get(key)
        if lk is None:
            lk = threading.RLock()
            _ENV_LOCKS[key] = lk
        return lk


@contextmanager
def _locked_env(root: Path):
    """Context manager that guards environment operations with a lock."""
    lk = _get_env_lock(root)
    lk.acquire()
    try:
        yield
    finally:
        lk.release()


# -----------------------
# PythonEnv
# -----------------------

@dataclass(frozen=True)
class PythonEnv:
    """Represent a managed Python environment rooted at a filesystem path."""
    root: Path

    def __post_init__(self) -> None:
        """Normalize the root path after dataclass initialization."""
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    # -----------------------
    # current env + singleton
    # -----------------------

    @classmethod
    def get_current(cls) -> "PythonEnv":
        """Return the current active environment inferred from the process."""
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            log.debug("current env from VIRTUAL_ENV=%s", venv)
            return cls(Path(venv))

        exe = Path(sys.executable).expanduser().resolve()
        parent = exe.parent
        if parent.name in ("bin", "Scripts"):
            log.debug("current env inferred from sys.executable=%s", str(exe))
            return cls(parent.parent)

        log.debug("current env fallback to sys.prefix=%s", sys.prefix)
        return cls(Path(sys.prefix))

    @classmethod
    def ensure_uv(
        cls,
    ) -> str:
        """Ensure uv is installed and return its executable path."""
        uv = _uv_exe_on_path()
        if uv:
            return uv

        with _UV_LOCK:
            uv = _uv_exe_on_path()
            if uv:
                return uv

            log.info("uv not found; attempting pip install (thread-safe)")
            _pip_install_uv_in_current()

        uv = _uv_exe_on_path()
        if uv:
            return uv

        uv_path = _current_env_script("uv")
        if uv_path:
            log.debug("uv found in current env scripts: %s", str(uv_path))
            return str(uv_path)

        try:
            import site

            user_base = Path(site.USER_BASE).expanduser().resolve()
            scripts = user_base / ("Scripts" if _is_windows() else "bin")
            cand = scripts / ("uv.exe" if _is_windows() else "uv")
            if cand.exists():
                log.debug("uv found in site.USER_BASE scripts: %s", str(cand))
                return str(cand)
        except Exception as e:
            log.debug("failed checking site.USER_BASE for uv: %r", e)

        raise PythonEnvError("uv install completed, but uv executable still not found")

    # -----------------------
    # user env discovery
    # -----------------------

    @classmethod
    def iter_user_envs(
        cls,
        *,
        max_depth: int = 2,
        include_hidden: bool = False,
        require_python: bool = True,
        dedupe: bool = True,
    ) -> Iterator["PythonEnv"]:
        """Yield PythonEnv instances from the user env directory."""
        base = _user_envs_dir()
        if not base.exists() or not base.is_dir():
            return

        seen: set[str] = set()

        def _python_exe(d: Path) -> Path:
            """Return the python executable path for a candidate env."""
            if os.name == "nt":
                return d / "Scripts" / "python.exe"
            return d / "bin" / "python"

        base_parts = len(base.parts)

        for p in base.rglob("*"):
            if not p.is_dir():
                continue
            depth = len(p.parts) - base_parts
            if depth > max_depth:
                continue
            if not include_hidden and p.name.startswith("."):
                continue

            if require_python and not _python_exe(p).exists():
                continue
            if not require_python and not (p / "pyvenv.cfg").exists() and not _python_exe(p).exists():
                continue

            env = cls(p)
            if require_python and not env.exists():
                continue

            key = str(env.root)
            if dedupe:
                if key in seen:
                    continue
                seen.add(key)

            yield env

    # -----------------------
    # CRUD by name in ~/.python/envs (thread-safe)
    # -----------------------

    @classmethod
    def _user_env_root(cls, name: str) -> Path:
        """Return the root path for a named user environment."""
        return _user_envs_dir() / _safe_env_name(name)

    @classmethod
    def get(cls, name: str, *, require_python: bool = False) -> Optional["PythonEnv"]:
        """Return a PythonEnv by name when it exists."""
        root = cls._user_env_root(name)
        if not root.exists() or not root.is_dir():
            return None
        env = cls(root)
        if require_python and not env.exists():
            return None
        return env

    @classmethod
    def create(
        cls,
        name: str,
        *,
        python: Union[str, Path] = "python",
        clear: bool = False,
        packages: Optional[Iterable[str]] = None,
        requirements: Optional[Union[str, Path]] = None,
        pip_args: Optional[Iterable[str]] = None,
        cwd: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        index_url: Optional[str] = None,
        extra_index_url: str | list[str] | None = None,
    ) -> "PythonEnv":
        """
        Create env under ~/.python/envs/<name>.

        Important behavior:
        - If env already exists and clear=False: UPSERT (install baseline + requested deps).
        - Always installs DEFAULT_CREATE_PACKAGES + user packages, plus optional requirements.
        - Thread-safe per-env root.
        """
        root = cls._user_env_root(name)
        root.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(packages, str):
            packages = [packages]

        install_pkgs = _dedupe_keep_order([*DEFAULT_CREATE_PACKAGES, *(packages or [])])

        with _locked_env(root):
            if root.exists():
                if clear:
                    log.info("removing existing env (clear=True): %s", str(root))
                    shutil.rmtree(root)
                else:
                    env_obj = cls(root)
                    if not env_obj.exists():
                        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                        moved = root.with_name(root.name + f".broken-{ts}")
                        log.warning("env exists but python missing; moving aside: %s -> %s", str(root), str(moved))
                        root.rename(moved)
                    else:
                        env_obj.update(
                            packages=install_pkgs,
                            requirements=requirements,
                            pip_args=pip_args,
                            python=python,
                            create_if_missing=False,
                            recreate_if_broken=True,
                            cwd=cwd,
                            env=env,
                            index_url=index_url,
                            extra_index_url=extra_index_url,
                        )
                        return env_obj

            uv = cls.ensure_uv()

            py = str(Path(python).expanduser()) if isinstance(python, Path) else str(python)
            log.info("creating env: name=%s root=%s python=%s", name, str(root), py)
            _run_cmd([uv, "venv", str(root), "--python", py], cwd=cwd, env=env, native_tls=True)

            env_obj = cls(root)
            if not env_obj.exists():
                raise PythonEnvError(f"Created env but python missing: {env_obj.python_executable}")

            env_obj.update(
                packages=install_pkgs,
                requirements=requirements,
                pip_args=pip_args,
                python=python,
                create_if_missing=False,
                recreate_if_broken=True,
                cwd=cwd,
                env=env,
                index_url=index_url,
                extra_index_url=extra_index_url,
            )

            return env_obj

    def update(
        self,
        *,
        packages: Optional[Iterable[str], str] = None,
        requirements: Optional[Union[str, Path]] = None,
        pip_args: Optional[Iterable[str]] = None,
        python: Union[str, Path] = "python",
        create_if_missing: bool = True,
        recreate_if_broken: bool = True,
        cwd: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        index_url: Optional[str] = None,
        extra_index_url: str | list[str] | None = None,
    ) -> "PythonEnv":
        """
        Install deps into *this* env (uv-only).
        Thread-safe per-env root.
        Returns the (possibly recreated) env object.
        """
        root = self.root

        with _locked_env(root):
            env_obj: PythonEnv = self

            if (not root.exists() or not root.is_dir()) and create_if_missing:
                name = root.name or "env"
                log.info("env root missing; creating: %s", str(root))
                env_obj = self.__class__.create(
                    name,
                    python=python,
                    clear=False,
                    cwd=cwd,
                    env=env,
                    index_url=index_url,
                    extra_index_url=extra_index_url,
                )

            if not env_obj.exists():
                if not recreate_if_broken:
                    raise PythonEnvError(f"Env exists but python missing: {env_obj.root}")

                ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                moved = env_obj.root.with_name(env_obj.root.name + f".broken-{ts}")
                log.warning("env broken; moving aside: %s -> %s", str(env_obj.root), str(moved))
                env_obj.root.rename(moved)

                name = env_obj.root.name or "env"
                env_obj = self.__class__.create(
                    name,
                    python=python,
                    clear=False,
                    cwd=cwd,
                    env=env,
                    index_url=index_url,
                )

            uv = self.__class__.ensure_uv()

            extra = list(pip_args or [])
            base_uv = [uv, "pip", "install", "--python", str(env_obj.python_executable)]

            if packages:
                pkgs = [packages] if isinstance(packages, str) else list(packages)

                if pkgs:
                    log.info("installing packages into env %s: %s", str(env_obj.root), pkgs)
                    _run_cmd(base_uv + ["-U"] + pkgs + extra, cwd=cwd, env=env)

            if requirements:
                # requirements can be:
                #   - Path / str path to an existing file
                #   - raw requirements content (str) if path doesn't exist
                tmp_ctx: Optional[tempfile.TemporaryDirectory] = None

                try:
                    if isinstance(requirements, Path):
                        req_path = requirements.expanduser().resolve()
                    elif isinstance(requirements, str):
                        s = requirements.strip()
                        if not s:
                            raise PythonEnvError("requirements cannot be empty")

                        # treat as path if it exists, otherwise treat as raw content
                        if os.path.exists(s):
                            req_path = Path(s).expanduser().resolve()
                        else:
                            tmp_ctx = tempfile.TemporaryDirectory(prefix="pythonenv-req-")
                            req_path = Path(tmp_ctx.name) / "requirements.txt"
                            req_path.write_text(s + ("\n" if not s.endswith("\n") else ""), encoding="utf-8")
                    else:
                        raise PythonEnvError("requirements must be a path-like string/Path or raw requirements text")

                    log.info("installing requirements into env %s: %s", str(env_obj.root), str(req_path))
                    _run_cmd(
                        base_uv + ["-U", "-r", str(req_path)] + extra,
                        cwd=cwd, env=env,
                        extra_index_url=PipIndexSettings.default_settings().extra_index_url
                    )

                finally:
                    if tmp_ctx is not None:
                        tmp_ctx.cleanup()

            return env_obj

    @classmethod
    def delete(cls, name: str, *, missing_ok: bool = True) -> None:
        """Delete a user environment by name."""
        root = cls._user_env_root(name)
        with _locked_env(root):
            if not root.exists():
                if missing_ok:
                    return
                raise PythonEnvError(f"Env not found: {root}")
            log.info("deleting env: %s", str(root))
            shutil.rmtree(root)

    # -----------------------
    # env info
    # -----------------------

    @property
    def bindir(self) -> Path:
        """Return the scripts/bin directory for this environment."""
        if _is_windows():
            scripts = self.root / "Scripts"
            return scripts if scripts.is_dir() else self.root
        return self.root / "bin"

    @property
    def name(self) -> str:
        """Return the environment name derived from its root path."""
        n = self.root.name
        return n if n else str(self.root)

    @property
    def python_executable(self) -> Path:
        """Return the full path to the environment's Python executable."""
        exe = "python.exe" if _is_windows() else "python"
        return self.bindir / exe

    def exists(self) -> bool:
        """Return True when the environment's Python executable exists."""
        return self.python_executable.exists()

    @property
    def version(self) -> str:
        """Return the Python version string for the environment."""
        out = self.exec_code("import sys; print(sys.version.split()[0])", check=True)
        return out.strip()

    @property
    def version_info(self) -> tuple[int, int, int]:
        """Return the parsed (major, minor, patch) version tuple."""
        v = self.version
        m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)\s*$", v)
        if not m:
            raise PythonEnvError(f"Unexpected python version string: {v!r}")
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    # -----------------------
    # python version switch
    # -----------------------

    def change_python_version(self, version: str | tuple[int, ...], *, keep_packages: bool = False) -> "PythonEnv":
        """
        Recreate this env with a different python interpreter/version.

        If the requested version matches the current env's Python version, returns self.

        version:
          - "3.12" / "3.12.1" (version spec)
          - tuple like (3, 12) or (3, 12, 1)
          - OR an interpreter request like "python3.12" or full path (will recreate)

        keep_packages:
          - If True, freezes current env and re-installs into the new one.
          - Default False to avoid surprise network/resolution in CI.
        """
        req_str: str
        req_parts: Optional[tuple[int, ...]] = None

        if isinstance(version, tuple):
            if len(version) < 2:
                raise PythonEnvError("version tuple must be at least (major, minor)")
            if not all(isinstance(x, int) and x >= 0 for x in version):
                raise PythonEnvError(f"version tuple must be non-negative ints: {version!r}")
            req_parts = tuple(version)
            req_str = ".".join(str(x) for x in req_parts)
        else:
            req_str = str(version).strip()
            if not req_str:
                raise PythonEnvError("version cannot be empty")

            m = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?\s*$", req_str)
            if m:
                parts = [int(m.group(1))]
                if m.group(2) is not None:
                    parts.append(int(m.group(2)))
                if m.group(3) is not None:
                    parts.append(int(m.group(3)))
                req_parts = tuple(parts)

        if self.exists() and req_parts is not None:
            cur = self.version_info  # (major, minor, patch)
            if tuple(cur[: len(req_parts)]) == req_parts:
                return self

        def _slug(s: str) -> str:
            """Return a filesystem-friendly slug from a string."""
            s = (s or "").strip()
            s = re.sub(r"[^A-Za-z0-9._+-]+", "-", s)
            return s.strip("-") or "unknown"

        root = Path(self.root).expanduser().resolve()
        parent = root.parent

        uv = self.__class__.ensure_uv()

        frozen_text: Optional[str] = None
        if keep_packages and self.exists():
            p = _run_cmd([uv, "pip", "freeze", "--python", str(self.python_executable)])
            frozen_text = p.stdout or ""

        with _locked_env(root):
            if root.exists():
                ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                backup = parent / f"{root.name}.pychange-{_slug(req_str)}-{ts}"
                log.info("changing python version; moving aside: %s -> %s", str(root), str(backup))
                root.rename(backup)

            log.info("recreating env with python=%s at %s", req_str, str(root))
            _run_cmd([uv, "venv", str(root), "--python", req_str])

            new_env = self.__class__(root)
            if not new_env.exists():
                raise PythonEnvError(f"Recreated env but python missing: {new_env.python_executable}")

            if keep_packages and frozen_text and frozen_text.strip():
                ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                req_path = parent / f".{root.name}.freeze-{ts}.txt"
                req_path.write_text(frozen_text, encoding="utf-8")
                try:
                    _run_cmd([uv, "pip", "install", "--python", str(new_env.python_executable), "-r", str(req_path)])
                finally:
                    try:
                        req_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            return new_env

    # -----------------------
    # requirements matrix
    # -----------------------
    def requirements(
        self,
        out_dir: Optional[Union[str, Path]] = None,
        base_name: str = "requirements",
        include_frozen: bool = True,
        include_input: bool = True,
        buffers: Optional[MutableMapping[str, str]] = None,
    ):
        """Export requirements for the current environment's Python executable."""
        return self.export_requirements_matrix(
            python_versions=[self.python_executable],
            out_dir=out_dir, base_name=base_name, include_frozen=include_frozen,
            include_input=include_input, buffers=buffers or {},
        )[str(self.python_executable)]

    def export_requirements_matrix(
        self,
        python_versions: Iterable[Union[str, Path]],
        *,
        out_dir: Optional[Union[str, Path]] = None,
        base_name: str = "requirements",
        include_frozen: bool = True,
        include_input: bool = True,
        buffers: Optional[MutableMapping[str, str]] = None,
    ) -> Union[str, dict[str, Union[Path, str]]]:
        """
        Generate requirements to duplicate this env across multiple Python versions.

        If out_dir is provided:
          - writes files into out_dir
          - returns dict[python_req -> Path]

        If out_dir is None:
          - writes to a temp workdir (needed for uv compile input/output files)
          - returns dict[python_req -> compiled requirements text (str)]

        Outputs (when out_dir given):
          - {base_name}.in
          - {base_name}.frozen.txt
          - {base_name}-py<slug>.txt
        """
        def _slug(s: str) -> str:
            """Return a filesystem-friendly slug for a Python version label."""
            s = (s or "").strip()
            if not s:
                return "unknown"
            s = re.sub(r"[^A-Za-z0-9._+-]+", "-", s)
            return s.strip("-") or "unknown"

        targets = _dedupe_keep_order([str(v) for v in python_versions])
        if not targets:
            raise PythonEnvError("python_versions cannot be empty")

        if not self.exists():
            raise PythonEnvError(f"Python executable not found in env: {self.python_executable}")

        uv = self.__class__.ensure_uv()

        write_files = out_dir is not None

        tmp_ctx: Optional[tempfile.TemporaryDirectory] = None
        if write_files:
            out_root = Path(out_dir).expanduser().resolve()
            out_root.mkdir(parents=True, exist_ok=True)
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="pythonenv-reqs-")
            out_root = Path(tmp_ctx.name).resolve()

        try:
            with _locked_env(self.root):
                # 1) top-level deps
                req_in_path = out_root / f"{base_name}.in"
                if include_input:
                    log.info("exporting top-level requirements: %s", str(req_in_path))

                    code = r"""import importlib.metadata as md, json, re

def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()

dists = list(md.distributions())
installed = {}
for d in dists:
    n = d.metadata.get("Name") or d.metadata.get("Summary") or ""
    n = (n or "").strip()
    if not n:
        continue
    installed[norm(n)] = n

required = set()
for d in dists:
    for r in (d.requires or []):
        r = (r or "").strip()
        if not r:
            continue
        name = re.split(r"[ ;<>=!~\[\]]", r, 1)[0].strip()
        if name:
            required.add(norm(name))

top_level = [installed[k] for k in installed.keys() if k not in required]
drop = {norm("pip"), norm("setuptools"), norm("wheel")}
top_level = [x for x in top_level if norm(x) not in drop]
top_level = sorted(set(top_level), key=lambda s: s.lower())
print("RESULT:" + json.dumps(top_level))""".strip()

                    top_level = self.exec_code_and_return(
                        code,
                        result_tag="RESULT:",
                        parse_json=True,
                        print_prefix_lines=False,
                        strip_payload=True,
                    )

                    if not isinstance(top_level, list) or not all(isinstance(x, str) for x in top_level):
                        raise PythonEnvError(f"Unexpected top-level requirements payload: {top_level!r}")

                    filtered = _filter_non_pipable_linux_packages(top_level)

                    req_in_text = "\n".join(filtered) + "\n"
                    req_in_path.write_text(req_in_text, encoding="utf-8")
                    if buffers is not None:
                        buffers[f"{base_name}.in"] = req_in_text

                # 2) frozen snapshot
                frozen_path = out_root / f"{base_name}.frozen.txt"
                if include_frozen:
                    log.info("exporting frozen requirements: %s", str(frozen_path))
                    p = _run_cmd([uv, "pip", "freeze", "--python", str(self.python_executable)])
                    frozen_text = p.stdout or ""
                    if write_files:
                        frozen_path.write_text(frozen_text, encoding="utf-8")
                    if buffers is not None:
                        buffers[f"{base_name}.frozen.txt"] = frozen_text

                # 3) compile per target
                if include_input and not req_in_path.exists():
                    raise PythonEnvError(f"Missing requirements input file: {req_in_path}")

                compiled_paths: dict[str, Path] = {}
                compiled_texts: dict[str, str] = {}

                for py_req in targets:
                    out_path = out_root / f"{base_name}-py{_slug(py_req)}.txt"
                    log.info("compiling requirements for python=%s -> %s", py_req, str(out_path))

                    cmd = [uv, "pip", "compile", str(req_in_path), "--python", py_req, "-o", str(out_path)]
                    _run_cmd(cmd)

                    if write_files:
                        compiled_paths[py_req] = out_path
                        if buffers is not None:
                            buffers[out_path.name] = out_path.read_text(encoding="utf-8")
                    else:
                        txt = out_path.read_text(encoding="utf-8")
                        compiled_texts[py_req] = txt
                        if buffers is not None:
                            buffers[out_path.name] = txt

                return compiled_paths if write_files else compiled_texts

        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    def installed_packages(self, parsed: bool = False) -> List[Tuple[str, str]]:
        """Return installed packages, optionally parsed into (name, version)."""
        req = self.requirements()

        r = [
            line.strip()
            for line in req.splitlines() if line.strip() and not line.strip().startswith("#")
        ]

        if parsed:
            return [
                tuple(re.split(r"==|>=|<=|>|<|~=|!=", line, 1)) if re.search(r"==|>=|<=|>|<|~=|!=", line) else (line, "")
                for line in r
            ]
        return r

    # -----------------------
    # execute (uv run always)
    # -----------------------

    def exec_code(
        self,
        code: str,
        *,
        python: Optional[Union[str, Path]] = None,
        cwd: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        check: bool = True,
    ) -> str:
        """Execute Python code inside the environment and return stdout."""
        # pick interpreter (default = env python)
        if python is None:
            if not self.exists():
                raise PythonEnvError(f"Python executable not found in env: {self.python_executable}")
            py = self.python_executable
        else:
            py = Path(python).expanduser().resolve() if isinstance(python, Path) else Path(str(python)).expanduser().resolve()
            if not py.exists():
                raise PythonEnvError(f"Python executable not found: {py}")

        uv = PythonEnv.ensure_uv()
        cmd = [uv, "run", "--python", str(py), "--", "python", "-c", code]
        log.debug("exec_code env=%s python=%s", str(self.root), str(py))
        p = _run_cmd(cmd, cwd=cwd, env=env)
        return p.stdout or ""

    def exec_code_and_return(
        self,
        code: str,
        *,
        result_tag: str = "RESULT:",
        python: Optional[Union[str, Path]] = None,
        cwd: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        check: bool = True,
        parse_json: bool = False,
        print_prefix_lines: bool = True,
        strip_payload: bool = True,
    ) -> Any:
        """Execute code and parse a tagged payload from stdout."""
        stdout = self.exec_code(
            code,
            python=python,
            cwd=cwd,
            env=env,
            check=check,
        )

        before_lines, payload = _split_on_tag(stdout, result_tag)

        if print_prefix_lines and before_lines:
            for line in before_lines:
                print(line)

        if payload is None:
            raise PythonEnvError(
                f"Result tag not found in stdout (tag={result_tag!r}).\n"
                f"--- stdout ---\n{stdout}\n"
            )

        if strip_payload:
            payload = payload.strip()

        def _try_parse_obj(s: str) -> Optional[Any]:
            """Try parsing a string as JSON or Python literal."""
            s2 = s.strip()
            if not s2:
                return None
            try:
                return json.loads(s2)
            except Exception:
                pass
            try:
                return ast.literal_eval(s2)
            except Exception:
                return None

        def _decode_value(val: Any, encoding: Optional[str]) -> Any:
            """Decode a serialized value based on an encoding hint."""
            enc = (encoding or "").strip().lower()
            if enc in ("", "none", "raw", "plain"):
                return val
            if enc in ("text", "str", "utf-8"):
                if isinstance(val, (bytes, bytearray)):
                    return bytes(val).decode("utf-8", errors="replace")
                return str(val)
            if enc in ("dill+b64", "dill_base64", "dill-b64"):
                if val is None:
                    return None
                if isinstance(val, str):
                    b = base64.b64decode(val.encode("utf-8"))
                elif isinstance(val, (bytes, bytearray)):
                    b = base64.b64decode(bytes(val))
                else:
                    raise PythonEnvError(f"Cannot decode dill+b64 from type: {type(val)}")

                try:
                    import dill
                except Exception as e:
                    raise PythonEnvError("dill is required to decode dill+b64 payloads") from e

                try:
                    return dill.loads(b)
                except Exception as e:
                    raise PythonEnvError(f"Failed to dill-decode payload: {e}") from e

            raise PythonEnvError(f"Unknown encoding: {encoding!r}")

        obj = _try_parse_obj(payload)
        if isinstance(obj, dict) and "ok" in obj and "encoding" in obj:
            ok = bool(obj.get("ok"))
            encoding = obj.get("encoding")

            ret_raw = obj.get("return")
            err_raw = obj.get("error")

            decoded_return = _decode_value(ret_raw, encoding) if ret_raw is not None else None

            decoded_error = None
            if err_raw is not None:
                try:
                    decoded_error = _decode_value(err_raw, encoding)
                except Exception:
                    decoded_error = _decode_value(err_raw, "text")

            if ok:
                return decoded_return

            raise PythonEnvError(
                "Remote code reported failure.\n"
                f"error={decoded_error!r}\n"
                f"envelope={obj!r}"
            )

        if parse_json:
            try:
                return json.loads(payload)
            except json.JSONDecodeError as e:
                raise PythonEnvError(
                    f"Failed to parse JSON payload after tag {result_tag!r}: {e}\n"
                    f"--- payload ---\n{payload}\n"
                ) from e

        return payload

    # -----------------------
    # CLI (uv everywhere)
    # -----------------------

    @classmethod
    def cli(cls, argv: Optional[list[str]] = None) -> int:
        """Run the PythonEnv CLI command."""
        import argparse

        parser = argparse.ArgumentParser(prog="python_env", description="User env CRUD + exec (uv everywhere)")
        sub = parser.add_subparsers(dest="cmd", required=True)

        p_uv = sub.add_parser("ensure-uv", help="Ensure uv is installed via pip")
        p_uv.add_argument("--no-upgrade", action="store_true", help="Do not upgrade uv if already installed")
        p_uv.add_argument("--no-user", action="store_true", help="Install uv without --user")
        p_uv.add_argument("--index-url", default=None, help="pip --index-url")
        p_uv.add_argument("--pip-arg", action="append", default=[], help="Extra pip args for installing uv")

        p_list = sub.add_parser("list", help="List envs under ~/.python/envs")
        p_list.add_argument("--max-depth", type=int, default=2)
        p_list.add_argument("--include-hidden", action="store_true")
        p_list.add_argument("--no-require-python", action="store_true")

        p_create = sub.add_parser("create", help="Create env (or update if it already exists)")
        p_create.add_argument("name")
        p_create.add_argument("--python", default="python")
        p_create.add_argument("--clear", action="store_true")
        p_create.add_argument("--pkg", action="append", default=[], help="Package to install (repeatable)")
        p_create.add_argument("--requirements", default=None)
        p_create.add_argument("--pip-arg", action="append", default=[], help="Extra uv pip args (repeatable)")

        p_delete = sub.add_parser("delete", help="Delete env under ~/.python/envs/<name>")
        p_delete.add_argument("name")
        p_delete.add_argument("--missing-ok", action="store_true")

        p_exec = sub.add_parser("exec", help="Exec python -c in an env (uv run)")
        p_exec.add_argument("name")
        p_exec.add_argument("--code", required=True)

        p_ret = sub.add_parser("exec-return", help="Exec and parse RESULT: payload (uv run)")
        p_ret.add_argument("name")
        p_ret.add_argument("--code", required=True)
        p_ret.add_argument("--tag", default="RESULT:")
        p_ret.add_argument("--json", action="store_true")

        if argv is None:
            argv = sys.argv[1:]
        if not argv:
            parser.print_help()
            return 0

        args = parser.parse_args(argv)

        try:
            if args.cmd == "ensure-uv":
                uv = cls.ensure_uv()
                print(uv)
                return 0

            if args.cmd == "list":
                require_python = not args.no_require_python
                for e in cls.iter_user_envs(
                    max_depth=args.max_depth,
                    include_hidden=args.include_hidden,
                    require_python=require_python,
                    dedupe=True,
                ):
                    py = str(e.python_executable) if e.exists() else "-"
                    print(f"{e.name}\t{e.root}\t{py}")
                return 0

            if args.cmd == "create":
                env_obj = cls.create(
                    args.name,
                    python=args.python,
                    clear=args.clear,
                    packages=args.pkg or None,
                    requirements=Path(args.requirements).expanduser() if args.requirements else None,
                    pip_args=args.pip_arg or None,
                )
                print(env_obj.root)
                return 0

            if args.cmd == "delete":
                cls.delete(args.name, missing_ok=args.missing_ok)
                return 0

            if args.cmd == "exec":
                env_obj = cls.get(args.name, require_python=True)
                if env_obj is None:
                    raise PythonEnvError(f"Env not found: {args.name!r}")
                out = env_obj.exec_code(args.code)
                print(out, end="")
                return 0

            if args.cmd == "exec-return":
                env_obj = cls.get(args.name, require_python=True)
                if env_obj is None:
                    raise PythonEnvError(f"Env not found: {args.name!r}")
                val = env_obj.exec_code_and_return(
                    args.code,
                    result_tag=args.tag,
                    parse_json=args.json,
                )
                if isinstance(val, (dict, list, int, float, bool)) or val is None:
                    print(json.dumps(val))
                else:
                    print(val)
                return 0

            if args.cmd == "current":
                cur = cls.get_current()
                print(f"root={cur.root}")
                print(f"python={cur.python_executable}")
                return 0

            raise PythonEnvError(f"Unknown command: {args.cmd}")

        except PythonEnvError as e:
            log.error("python_env CLI error: %s", e)
            print(f"ERROR: {e}", file=sys.stderr)
            return 2


# Snapshot singleton (import-time)
CURRENT_PYTHON_ENV: PythonEnv = PythonEnv.get_current()


if __name__ == "__main__":
    raise SystemExit(PythonEnv.cli())
