"""Wheel registry — uniform CRUD over the workspace PyPI-like index.

A wheel is identified by ``(project, version)``. The registry is a PEP 503-style
layout under ``/Workspace/Shared/pypi/<dist>/<version>/`` — distribution and
version are folder levels — holding the ``.whl`` files.
Every project — ``ygg`` included — is handled identically; there is no special
casing.

``create`` **fetches** the wheel and uploads it:

- a **local path** (a directory/file with a ``pyproject.toml``) is built from
  source (``uv build``), so "deploy the project I'm working on" just works;
- anything else is treated as a **PyPI** project and downloaded
  (``pip download``).

Fetches land in a local on-disk cache first (``~ tmp/yggdrasil/wheel-cache``) so
repeated builds/downloads/uploads are cheap. The :class:`Wheels` service
(``dbc.wheels``) is the front door; versions are parsed and compared with
:class:`yggdrasil.version.VersionInfo`.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from yggdrasil.version import VersionInfo

from ..service import DatabricksService
from .wheel import Wheel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "WORKSPACE_PYPI_DIR",
    "WORKSPACE_ENV_DIR",
    "SUPPORTED_PYTHONS",
    "SERVERLESS_ENVIRONMENT_VERSION",
    "SERVERLESS_ENVIRONMENT_VERSIONS",
    "SERVERLESS_WHEEL_PLATFORMS",
    "BUNDLE_EXCLUDE",
    "PROJECT_ALIASES",
    "parse_version",
    "wheel_parts",
    "serverless_environment_version",
    "environment_key_for",
    "wheel_for_python",
    "distribution_for",
    "runtime_dependencies",
    "find_pyproject",
    "read_pyproject",
    "project_display_name",
    "fetch_wheels",
    "registry_upload",
    "Wheels",
]

#: Root of the workspace's PyPI-like wheel registry — one ``<dist>/`` folder per
#: distribution + version folder levels (``pypi/<dist>/<version>/<dist>-<version>-…whl``).
WORKSPACE_PYPI_DIR = "/Workspace/Shared/pypi"
#: Where reusable base environments live (one ``<proj>/`` folder per project).
WORKSPACE_ENV_DIR = "/Workspace/Shared/environment"

#: Python minors we build/download wheels for. Pure-python projects collapse to a
#: single ``py3-none-any`` wheel reused across all of them. Capped at
#: :data:`MAX_PYTHON` — Databricks (serverless + DBR) doesn't run 3.13+ yet.
SUPPORTED_PYTHONS: "tuple[str, ...]" = ("3.10", "3.11", "3.12")

#: Highest Python minor Databricks can run today. Wheel/env targets are clamped
#: to it (see :func:`_py_minor`) so a build on a newer local interpreter (3.13+)
#: still produces an artifact the cluster can actually install.
MAX_PYTHON: str = "3.12"

#: Latest serverless environment version; per-Python overrides below.
SERVERLESS_ENVIRONMENT_VERSION = "5"
SERVERLESS_ENVIRONMENT_VERSIONS: "dict[str, str]" = {"3.10": "1", "3.11": "2"}

#: Linux-x86_64 manylinux tags the Databricks compute installs — the dependency
#: closure is pinned to these so it is platform-correct regardless of build host.
SERVERLESS_WHEEL_PLATFORMS: "tuple[str, ...]" = (
    "manylinux2014_x86_64",
    "manylinux_2_28_x86_64",
)
#: Runtime-provided distributions never shipped in the closure (a second copy
#: conflicts with the serverless image's own).
BUNDLE_EXCLUDE: "frozenset[str]" = frozenset({"certifi"})

#: Project-name aliases applied before the metadata lookup in
#: :func:`distribution_for` — the import name a project is known by mapped to the
#: distribution (pip) name its wheels actually ship under. ``yggdrasil`` is the
#: import package; ``ygg`` is the distribution, so a deploy named for either lands
#: in the same registry folder / environment. Keys are normalized on lookup.
#: Override / extend via ``YGG_DATABRICKS_PROJECT_ALIASES`` (``import=dist,…``).
PROJECT_ALIASES: "dict[str, str]" = {"yggdrasil": "ygg"}


# --------------------------------------------------------------------------- #
# Naming / version helpers
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def _serverless_wheel_platforms() -> "list[str]":
    raw = os.environ.get("YGG_DATABRICKS_WHEEL_PLATFORMS")
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else list(SERVERLESS_WHEEL_PLATFORMS)


def _bundle_exclude() -> "set[str]":
    raw = os.environ.get("YGG_DATABRICKS_BUNDLE_EXCLUDE")
    return {_norm(p) for p in raw.split(",") if p.strip()} if raw else set(BUNDLE_EXCLUDE)


def _py_minor(python: "str | None" = None) -> str:
    """``"3.X"`` for *python* (default: the local interpreter; accepts ``3.11`` /
    ``311`` / ``py311`` / ``3.11.7``), **clamped to** :data:`MAX_PYTHON` so a
    build on a newer interpreter (3.13+) still targets a Python Databricks runs."""
    if python is None:
        minor = f"3.{sys.version_info[1]}"
    else:
        digits = re.sub(r"[^0-9.]", "", python)
        parts = digits.split(".")
        if len(parts) >= 2:
            minor = f"{parts[0]}.{parts[1]}"
        elif digits.startswith("3") and len(digits) >= 3:
            minor = f"3.{digits[1:]}"
        else:
            minor = digits or f"3.{sys.version_info[1]}"
    return _clamp_python(minor)


def _clamp_python(minor: str) -> str:
    """Cap *minor* (``"3.X"``) at :data:`MAX_PYTHON` — Databricks can't run 3.13+
    yet, so a newer target would build an unusable wheel / environment."""
    try:
        if tuple(map(int, minor.split("."))) > tuple(map(int, MAX_PYTHON.split("."))):
            logger.debug(
                "Python %s exceeds the Databricks max (%s); building for %s.",
                minor, MAX_PYTHON, MAX_PYTHON,
            )
            return MAX_PYTHON
    except ValueError:
        pass
    return minor


def serverless_environment_version(python: "str | None" = None) -> str:
    """The serverless ``environment_version`` whose runtime Python matches *python*."""
    return SERVERLESS_ENVIRONMENT_VERSIONS.get(_py_minor(python), SERVERLESS_ENVIRONMENT_VERSION)


def environment_key_for(python: "str | None" = None) -> str:
    """The serverless ``environment_key`` for a Python (``3.11`` → ``"py311"``)."""
    return "py" + _py_minor(python).replace(".", "")


def parse_version(text: "str | VersionInfo | None") -> "Optional[VersionInfo]":
    """Parse *text* into a :class:`VersionInfo`, or ``None`` if it can't be read.

    Tolerates wheel-filename escaping (``0_8_57`` / ``0-8-57``) and a PEP 440
    local segment (``0.8.57+host.x`` → ``0.8.57``)."""
    if text is None or isinstance(text, VersionInfo):
        return text
    head = str(text).split("+", 1)[0].strip()
    head = re.sub(r"[-_]", ".", head)
    try:
        return VersionInfo.from_string(head)
    except (ValueError, IndexError):
        return None


def wheel_parts(filename: "str | Path") -> "tuple[str, Optional[VersionInfo], str]":
    """``(dist, version, pytag)`` parsed from a wheel *filename*."""
    name = Path(str(filename)).name
    stem = name[:-4] if name.endswith(".whl") else name
    parts = stem.split("-")
    dist = _norm(parts[0]) if parts else stem
    version = parse_version(parts[1]) if len(parts) >= 2 else None
    tag = "-".join(parts[2:]) if len(parts) > 2 else ""
    return dist, version, tag


def wheel_for_python(wheels: "list", python: "str | None" = None) -> str:
    """Pick the wheel matching *python* (a ``cp3XX`` build, else the universal
    ``py3-none-any`` wheel, else the first). Returns a string path."""
    tag = "cp" + _py_minor(python).replace(".", "")
    items = [str(w) for w in wheels]
    return next(
        (w for w in items if tag in w),
        next((w for w in items if "-py3-none-any.whl" in w), items[0] if items else ""),
    )


def distribution_for(package: str) -> str:
    """The distribution (pip) name providing import *package* (``yggdrasil`` →
    ``ygg``); falls back to *package*.

    An explicit :data:`PROJECT_ALIASES` entry (extendable via
    ``YGG_DATABRICKS_PROJECT_ALIASES``) wins first, so a project deploys under one
    canonical distribution regardless of whether its metadata is installed locally;
    otherwise the import → distribution mapping is read from installed metadata."""
    import importlib.metadata as ilmd

    aliases = dict(PROJECT_ALIASES)
    raw = os.environ.get("YGG_DATABRICKS_PROJECT_ALIASES")
    if raw:
        for pair in raw.split(","):
            src, _, dst = pair.partition("=")
            if src.strip() and dst.strip():
                aliases[_norm(src.strip())] = dst.strip()
    alias = aliases.get(_norm(package))
    if alias:
        return alias

    dists = ilmd.packages_distributions().get(package)
    return dists[0] if dists else package


def runtime_dependencies(project: str = "ygg", extras: "tuple[str, ...] | list[str]" = ("databricks",)) -> "list[str]":
    """*project*'s declared runtime dependencies as **index requirement** specs
    (names + version pins), for a Spark Connect / cluster install that resolves
    them from the index. Bare names are pinned to their installed version so the
    registry doesn't mistake an unpinned dep for a local package."""
    import importlib.metadata as ilmd
    import re

    dist = distribution_for(project)
    wants = set(extras)
    ops = ("==", ">=", "<=", "!=", "~=", "===", ">", "<")
    out: "list[str]" = []
    for req in ilmd.requires(dist) or []:
        head, _, marker = req.partition(";")
        head = head.strip()
        extra = re.search(r'extra\s*==\s*["\']([^"\']+)["\']', marker)
        if extra is not None and extra.group(1) not in wants:
            continue
        if "[" in head or any(op in head for op in ops):
            out.append(req if extra is None else head)
            continue
        try:
            out.append(f"{head}=={ilmd.version(head)}")
        except Exception:  # noqa: BLE001
            out.append(head)
    return out



# --------------------------------------------------------------------------- #
# Project discovery + local build
# --------------------------------------------------------------------------- #
def find_pyproject(start: "str | Path | None" = None) -> "Optional[Path]":
    """The nearest ``pyproject.toml`` at or above *start* (cwd by default), or
    ``None`` when *start* isn't a real local path / has no project on the way up."""
    if start is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start)
        if not start_path.exists():
            return None
        start_path = start_path.resolve()
    if start_path.is_file():
        if start_path.name == "pyproject.toml":
            return start_path
        start_path = start_path.parent
    for directory in (start_path, *start_path.parents):
        candidate = directory / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def project_display_name(name: str) -> str:
    """A nice, capitalized display name for a project — each word capitalized
    (``my-app`` → ``My App``, ``meteologica`` → ``Meteologica``)."""
    words = [w for w in re.split(r"[^0-9A-Za-z]+", str(name)) if w]
    return " ".join(w.capitalize() for w in words) or str(name)


def read_pyproject(path: "str | Path") -> "dict[str, Any]":
    """Parse a ``pyproject.toml``'s ``[project]`` into ``name`` / ``version`` /
    ``dependencies`` / ``optional_dependencies`` / ``requires_python`` / ``dir``."""
    try:
        import tomllib as toml
    except ModuleNotFoundError:                       # Python 3.10
        import tomli as toml

    path = Path(path).resolve()
    project = (toml.loads(path.read_text(encoding="utf-8")).get("project") or {})
    name = project.get("name")
    if not name:
        raise ValueError(f"{path} has no [project].name — not a deployable project")
    return {
        "name": name,
        "version": project.get("version", "0.0.0"),
        "dependencies": list(project.get("dependencies") or []),
        "optional_dependencies": {
            extra: list(reqs)
            for extra, reqs in (project.get("optional-dependencies") or {}).items()
        },
        "requires_python": project.get("requires-python"),
        "dir": path.parent,
    }


def _run(cmd: "list[str]") -> None:
    """Run a build/download subprocess quietly; fold a failure's tail into the error."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "").strip()
        logger.error("command failed (%s): %s\n%s", exc.returncode, " ".join(cmd), tail)
        raise RuntimeError(
            f"command failed (exit {exc.returncode}): {' '.join(cmd[:4])} … — {tail[-2000:]}"
        ) from exc


def _cache_dir(name: str, version: "str | None", python: "str | None", deps: bool) -> Path:
    """A stable local cache slot for a ``(name, version, python, deps)`` fetch."""
    slug = f"{_norm(name)}-{version or 'latest'}-{environment_key_for(python)}-{'deps' if deps else 'nodeps'}"
    out = Path(tempfile.gettempdir()) / "yggdrasil" / "wheel-cache" / slug
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_local(project_dir: "str | Path", *, python: "str | None", deps: bool,
                 extras: "tuple[str, ...] | list[str]", out: Path) -> "list[Path]":
    """Build the on-disk project at *project_dir* (and, with *deps*, download its
    declared dependency closure as Linux wheels) into *out*. Project wheel first."""
    project_dir = Path(project_dir).resolve()
    cmd = ["uv", "build", "--wheel"]
    if python:
        cmd += ["--python", _py_minor(python)]
    cmd += ["--out-dir", str(out), str(project_dir)]
    try:
        _run(cmd)
    except FileNotFoundError:
        _run([sys.executable, "-m", "pip", "wheel", str(project_dir), "--no-deps",
              "--wheel-dir", str(out)])
    wheels = sorted(out.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"no wheel produced for project {project_dir}")
    if deps:
        meta = read_pyproject(project_dir / "pyproject.toml") if (project_dir / "pyproject.toml").exists() \
            else read_pyproject(find_pyproject(project_dir))
        wanted = list(meta["dependencies"])
        for extra in extras:
            wanted += meta["optional_dependencies"].get(extra, [])
        wheels += _download_deps(wanted, python=python, out=out)
    return _ordered(wheels, _norm(wheel_parts(wheels[0])[0]))


def _download_pypi(name: str, version: "str | None", *, python: "str | None",
                   deps: bool, extras: "tuple[str, ...] | list[str]", out: Path) -> "list[Path]":
    """Download *name* (``==version`` when given) from PyPI as wheel(s) into *out*.
    With *deps* the whole transitive closure is fetched as Linux wheels."""
    spec = f"{name}=={version}" if version else name
    if extras:
        spec = f"{name}[{','.join(extras)}]" + (f"=={version}" if version else "")
    if deps:
        cmd = [sys.executable, "-m", "pip", "download", spec, "--only-binary=:all:",
               "--python-version", _py_minor(python), "--implementation", "cp",
               "--abi", "cp" + _py_minor(python).replace(".", ""), "--abi", "abi3", "--abi", "none"]
        for tag in _serverless_wheel_platforms():
            cmd += ["--platform", tag]
        cmd += ["--dest", str(out)]
        _run(cmd)
    else:
        try:
            _run([sys.executable, "-m", "pip", "download", spec, "--no-deps",
                  "--only-binary=:all:", "--dest", str(out)])
        except RuntimeError:                          # no wheel on the index — build one
            _run([sys.executable, "-m", "pip", "wheel", spec, "--no-deps", "--wheel-dir", str(out)])
    wheels = sorted(out.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"pip produced no wheel for {spec!r}")
    proj = next((w for w in wheels if wheel_parts(w)[0] == _norm(name)), wheels[0])
    return _ordered(wheels, wheel_parts(proj)[0])


def _download_deps(dependencies: "list[str]", *, python: "str | None", out: Path) -> "list[Path]":
    """Download *dependencies* as Linux wheels (runtime-provided dists dropped)."""
    deps = [d for d in dependencies if d]
    if not deps:
        return []
    cmd = [sys.executable, "-m", "pip", "download", "--only-binary=:all:",
           "--python-version", _py_minor(python), "--implementation", "cp",
           "--abi", "cp" + _py_minor(python).replace(".", ""), "--abi", "abi3", "--abi", "none"]
    for tag in _serverless_wheel_platforms():
        cmd += ["--platform", tag]
    cmd += ["--dest", str(out), *deps]
    _run(cmd)
    return [w for w in sorted(out.glob("*.whl")) if wheel_parts(w)[0] not in _bundle_exclude()]


def _ordered(wheels: "list[Path]", project_dist: str) -> "list[Path]":
    """Project wheel(s) first, then the dependency wheels — both de-duplicated."""
    seen: "set[str]" = set()
    proj, rest = [], []
    for w in wheels:
        if w.name in seen:
            continue
        seen.add(w.name)
        (proj if wheel_parts(w)[0] == project_dist else rest).append(w)
    return sorted(proj) + sorted(rest)


def fetch_wheels(
    spec: "str | Path",
    version: "str | None" = None,
    *,
    python: "str | None" = None,
    deps: bool = False,
    extras: "tuple[str, ...] | list[str]" = (),
    rebuild: bool = False,
) -> "list[Path]":
    """Resolve *spec* into local wheel files (project wheel first), via the local
    cache. A path with a ``pyproject.toml`` is built; otherwise *spec* is a PyPI
    project downloaded by name. With *deps* the full closure is fetched."""
    pyproject = find_pyproject(spec) if isinstance(spec, (str, Path)) else None
    if pyproject is not None and find_pyproject(spec):
        meta = read_pyproject(pyproject)
        name, version = meta["name"], str(meta["version"])
        out = _cache_dir(name, version, python, deps)
        cached = sorted(out.glob("*.whl"))
        if cached and not rebuild:
            return _ordered(cached, _norm(name))
        return _build_local(meta["dir"], python=python, deps=deps, extras=extras, out=out)
    name = str(spec)
    out = _cache_dir(name, version, python, deps)
    cached = sorted(out.glob("*.whl"))
    if cached and not rebuild:
        return _ordered(cached, _norm(name))
    return _download_pypi(name, version, python=python, deps=deps, extras=extras, out=out)


def registry_upload(client: Any, wheel: "str | Path", *, workspace_dir: str = WORKSPACE_PYPI_DIR,
                    overwrite: bool = False) -> str:
    """Upload *wheel* to ``<workspace_dir>/<dist>/<version>/<wheel>`` and return
    the path — distribution **and** version are folder levels, so the registry
    browses like a PEP 503 index. An already-present, immutable wheel is reused
    unless *overwrite*."""
    from ..path import DatabricksPath

    wheel = Path(wheel)
    dist, version, _ = wheel_parts(wheel)
    dest = f"{workspace_dir.rstrip('/')}/{dist}/{version or 'unknown'}/{wheel.name}"
    path = DatabricksPath.from_(dest, client=client)
    if not overwrite and path.exists():
        logger.info("reusing deployed wheel %s", dest)
        return dest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wheel.read_bytes())
    logger.info("uploaded wheel to %s", dest)
    return dest


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class Wheels(DatabricksService):
    """CRUD over the workspace wheel registry (``dbc.wheels``).

    A wheel is keyed by ``(project, version)``. :meth:`create` fetches it from a
    local project or PyPI and uploads it; :meth:`find` returns it, building it on
    a miss; :meth:`update` / :meth:`delete` re-fetch / remove it.
    """

    default_dir: ClassVar[str] = WORKSPACE_PYPI_DIR

    def _wheel(self, path: str) -> Wheel:
        return Wheel(self, path=path)

    # -- create / update ---------------------------------------------------
    def create(
        self,
        project: "str | Path" = "ygg",
        version: "str | VersionInfo | None" = None,
        *,
        python: "str | None" = None,
        extras: "tuple[str, ...] | list[str]" = (),
        deps: bool = False,
        workspace_dir: "str | None" = None,
        overwrite: bool = True,
        rebuild: bool = False,
    ) -> "list[Wheel]":
        """Fetch *project* (a local pyproject path or a PyPI name) at *version* and
        upload its wheel(s) to the registry — project wheel first. With *deps* the
        whole zero-PyPI closure is uploaded too."""
        root = workspace_dir or self.default_dir
        ver = parse_version(version)
        files = fetch_wheels(project, str(ver) if ver else None, python=python,
                             deps=deps, extras=extras, rebuild=rebuild)
        return [self._wheel(registry_upload(self.client, w, workspace_dir=root, overwrite=overwrite))
                for w in files]

    def update(
        self,
        project: "str | Path" = "ygg",
        version: "str | VersionInfo | None" = None,
        **kwargs: Any,
    ) -> "list[Wheel]":
        """Re-fetch and **overwrite** *project*'s wheel(s) in the registry."""
        kwargs.setdefault("overwrite", True)
        kwargs.setdefault("rebuild", True)
        return self.create(project, version, **kwargs)

    # -- read --------------------------------------------------------------
    def list(
        self,
        project: "str | None" = None,
        *,
        workspace_dir: "str | None" = None,
    ) -> "list[Wheel] | list[str]":
        """Browse the registry: the :class:`Wheel`\\ s under *project*'s folder, or
        the distribution folder names when *project* is ``None``."""
        from ..path import DatabricksPath

        root = (workspace_dir or self.default_dir).rstrip("/")
        if project is not None:
            folder = DatabricksPath.from_(f"{root}/{_norm(distribution_for(str(project)))}", client=self.client)
            if not folder.exists():
                return []
            wheels: "list[Wheel]" = []
            for child in folder.iterdir():
                if str(child.name).endswith(".whl"):           # loose (legacy layout)
                    wheels.append(self._wheel(child.full_path()))
                elif child.is_dir():                           # <version>/ folder
                    wheels += [self._wheel(w.full_path()) for w in child.iterdir()
                               if str(w.name).endswith(".whl")]
            return wheels
        registry = DatabricksPath.from_(root, client=self.client)
        if not registry.exists():
            return []
        return [str(c.name) for c in registry.iterdir() if c.is_dir()]

    def _select(self, project: str, version: "Optional[VersionInfo]", python: "str | None",
                workspace_dir: "str | None") -> "Optional[Wheel]":
        wheels = [w for w in self.list(project, workspace_dir=workspace_dir) if isinstance(w, Wheel)]
        if version is not None:
            wheels = [w for w in wheels if w.version == version]
        if python is not None:                         # keep only wheels that run on *python*
            tag = "cp" + _py_minor(python).replace(".", "")
            compat = [w for w in wheels if (not w.tag) or "py3-none-any" in w.tag or tag in w.tag]
            if compat:
                wheels = compat
        if version is None and wheels:                 # no pin → the newest version present
            latest = max((w.version for w in wheels if w.version is not None), default=None)
            if latest is not None:
                wheels = [w for w in wheels if w.version == latest]
        if not wheels:
            return None
        chosen = wheel_for_python([w.path for w in wheels], python=python)
        return next((w for w in wheels if w.path == chosen), wheels[0])

    def get(
        self,
        project: "str | Path" = "ygg",
        version: "str | VersionInfo | None" = None,
        *,
        python: "str | None" = None,
        workspace_dir: "str | None" = None,
    ) -> "Optional[Wheel]":
        """The deployed wheel for *project* (matching *version* / *python*), or
        ``None`` — never builds. See :meth:`find` to build on a miss."""
        return self.find(project, parse_version(version), install=False,
                          python=python, workspace_dir=workspace_dir)

    def find(
        self,
        project: "str | Path" = "ygg",
        version: "VersionInfo | str | None" = None,
        *,
        install: bool = True,
        python: "str | None" = None,
        extras: "tuple[str, ...] | list[str]" = (),
        workspace_dir: "str | None" = None,
    ) -> "Optional[Wheel]":
        """Find *project*'s wheel in the registry; build + upload it (from a local
        pyproject or PyPI) when missing and *install* (the default)."""
        ver = parse_version(version)
        hit = self._select(_project_name(project), ver, python, workspace_dir)
        if hit is not None or not install:
            return hit
        built = self.create(project, ver, python=python, extras=extras,
                            workspace_dir=workspace_dir, overwrite=False)
        return built[0] if built else None

    # -- delete ------------------------------------------------------------
    def delete(
        self,
        project: "str | Path" = "ygg",
        version: "str | VersionInfo | None" = None,
        *,
        workspace_dir: "str | None" = None,
    ) -> "list[Wheel]":
        """Delete *project*'s wheel(s) from the registry (a specific *version*, or
        every version when omitted). Returns the wheels removed."""
        ver = parse_version(version)
        wheels = [w for w in self.list(project, workspace_dir=workspace_dir) if isinstance(w, Wheel)]
        if ver is not None:
            wheels = [w for w in wheels if w.version == ver]
        for w in wheels:
            w.delete()
        return wheels


def _project_name(spec: "str | Path") -> str:
    """The distribution name for a CRUD *spec* — a local project's
    ``[project].name`` when *spec* is a path, else *spec* itself."""
    pyproject = find_pyproject(spec) if isinstance(spec, (str, Path)) else None
    if pyproject is not None:
        try:
            return read_pyproject(pyproject)["name"]
        except Exception:  # noqa: BLE001
            pass
    return str(spec)
