# modules.py
from __future__ import annotations

import dataclasses as dc
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    # py3.8+
    import importlib.metadata as ilm  # type: ignore
except Exception:  # pragma: no cover
    ilm = None  # type: ignore

try:
    # Usually present because pip depends on it (but treat as optional)
    from packaging.requirements import Requirement  # type: ignore
except Exception:  # pragma: no cover
    Requirement = None  # type: ignore


__all__ = [
    "DependencyMetadata",
    "PipIndexSettings",
    "module_name_to_project_name",
    "resolve_local_lib_path",
    "module_dependencies",
    "get_pip_index_settings",
]


MODULE_PROJECT_NAMES_ALIASES = {
    "yggdrasil": "ygg",
    "jwt": "PyJWT",
}


def module_name_to_project_name(module_name: str) -> str:
    return MODULE_PROJECT_NAMES_ALIASES.get(module_name, module_name)


def resolve_local_lib_path(lib: Union[str, ModuleType]) -> Path:
    """
    Resolve a lib spec (path string, module name, or module object)
    into a concrete filesystem path.

    Package-walk rule:
    - If the resolved path is inside a Python package (dir containing __init__.py),
      walk upward and return the *top-most* directory that still contains __init__.py.
    - If not in a package context, return the resolved file/dir.
    """
    if isinstance(lib, ModuleType):
        mod_file = getattr(lib, "__file__", None)
        if not mod_file:
            raise ValueError(f"Module {lib.__name__!r} has no __file__; cannot determine path")
        path = Path(mod_file).resolve()
    else:
        p = Path(lib)
        if p.exists():
            path = p.resolve()
        else:
            try:
                mod = importlib.import_module(lib)
            except ImportError as e:
                raise ModuleNotFoundError(
                    f"'{lib}' is neither an existing path nor an importable module"
                ) from e
            mod_file = getattr(mod, "__file__", None)
            if not mod_file:
                raise ModuleNotFoundError(
                    f"Module {mod.__name__!r} has no __file__; cannot determine path"
                )
            path = Path(mod_file).resolve()

    # Determine a directory to start package-walk from
    start_dir = path.parent if path.is_file() else path

    top_pkg_dir: Optional[Path] = None
    current = start_dir

    while True:
        if (current / "__init__.py").exists():
            top_pkg_dir = current
            parent = current.parent
            if parent == current:
                break
            current = parent
        else:
            break

    return top_pkg_dir.resolve() if top_pkg_dir is not None else path


# Fallback regex (only used when packaging isn't available)
_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


@dc.dataclass(frozen=True)
class DependencyMetadata:
    project: str
    requirement: str
    installed: bool
    version: Optional[str]
    dist_root: Optional[Path]
    metadata_path: Optional[Path]


def _req_project_name(req_line: str) -> Optional[str]:
    """
    Best-effort extraction of the project name from a Requires-Dist line.
    Prefer a real PEP 508 parser (packaging) when available; fall back to regex.
    """
    left = req_line.split(";", 1)[0].strip()
    if not left:
        return None

    if Requirement is not None:
        try:
            r = Requirement(left)
            return r.name
        except Exception:
            return None

    m = _REQ_NAME_RE.match(left)
    if not m:
        return None
    name = m.group(1)
    return name.split("[", 1)[0]


def _distribution_for_module(mod: Union[str, ModuleType]):
    if ilm is None:
        raise RuntimeError("importlib.metadata is not available")

    if isinstance(mod, ModuleType):
        module_name = mod.__name__
    else:
        module_name = mod
        importlib.import_module(module_name)

    top = module_name.split(".", 1)[0]
    mapping = ilm.packages_distributions()
    dists = mapping.get(top)
    if not dists:
        raise ModuleNotFoundError(
            f"Can't find installed distribution that provides top-level module '{top}'"
        )
    return ilm.distribution(dists[0])


def module_dependencies(lib: Union[str, ModuleType]) -> List[DependencyMetadata]:
    """
    Return DependencyMetadata for all Requires-Dist deps of `lib`'s distribution.
    """
    if ilm is None:
        return []

    dist = _distribution_for_module(lib)
    reqs = list(dist.requires or [])

    out: List[DependencyMetadata] = []
    for req in reqs:
        project = _req_project_name(req)
        if not project:
            continue

        try:
            dep_dist = ilm.distribution(project)
            version = dep_dist.version
            dist_root = Path(dep_dist.locate_file("")).resolve()

            metadata_path = None
            try:
                if dep_dist.read_text("METADATA") is not None:
                    p = getattr(dep_dist, "_path", None)
                    if p is not None:
                        mp = Path(p) / "METADATA"
                        if mp.exists():
                            metadata_path = mp.resolve()
            except Exception:
                pass

            out.append(
                DependencyMetadata(
                    project=project,
                    requirement=req,
                    installed=True,
                    version=version,
                    dist_root=dist_root,
                    metadata_path=metadata_path,
                )
            )
        except ilm.PackageNotFoundError:
            out.append(
                DependencyMetadata(
                    project=project,
                    requirement=req,
                    installed=False,
                    version=None,
                    dist_root=None,
                    metadata_path=None,
                )
            )

    return out


def _run_pip(*args: str) -> Tuple[int, str, str]:
    p = subprocess.run(
        [sys.executable, "-m", "pip", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


@dc.dataclass(frozen=True)
class PipIndexSettings:
    index_url: Optional[str] = None
    extra_index_urls: List[str] = dc.field(default_factory=list)
    sources: Dict[str, Dict[str, Any]] = dc.field(default_factory=dict)  # {"env": {...}, "config": {...}}

    @classmethod
    def default_settings(cls):
        return DEFAULT_PIP_INDEX_SETTINGS

    def as_dict(self) -> dict:
        return dc.asdict(self)


def get_pip_index_settings() -> PipIndexSettings:
    """
    Inspect pip settings:
      - env (PIP_INDEX_URL / PIP_EXTRA_INDEX_URL)
      - pip config (merged view from `pip config list`)

    Precedence:
      env overrides config.
    """
    sources: Dict[str, Dict[str, Any]] = {"env": {}, "config": {}}

    env_index = os.environ.get("PIP_INDEX_URL")
    env_extra = os.environ.get("PIP_EXTRA_INDEX_URL")

    if env_index:
        sources["env"]["PIP_INDEX_URL"] = env_index
    if env_extra:
        sources["env"]["PIP_EXTRA_INDEX_URL"] = env_extra

    env_extra_urls: List[str] = shlex.split(env_extra) if env_extra else []

    # Read pip config (best-effort)
    rc, out, _err = _run_pip("config", "list", "--format=json")
    config_index_url: Optional[str] = None
    config_extra_raw: List[Any] = []

    if rc == 0 and out:
        cfg = json.loads(out)
        for k, v in cfg.items():
            lk = k.lower()
            if lk.endswith("index-url"):
                sources["config"][k] = v
                if lk.endswith("index-url") and not lk.endswith("extra-index-url"):
                    config_index_url = str(v)
                elif lk.endswith("extra-index-url"):
                    if isinstance(v, list):
                        config_extra_raw.extend(v)
                    else:
                        config_extra_raw.append(v)
    else:
        rc2, out2, _ = _run_pip("config", "list")
        if rc2 == 0 and out2:
            for line in out2.splitlines():
                if "=" not in line:
                    continue
                k, v = [x.strip() for x in line.split("=", 1)]
                lk = k.lower()
                if lk.endswith("extra-index-url"):
                    sources["config"][k] = v
                    config_extra_raw.append(v)
                elif lk.endswith("index-url") and not lk.endswith("extra-index-url"):
                    sources["config"][k] = v
                    config_index_url = v

    # Apply precedence
    index_url = env_index or config_index_url

    # extras: if env is set, it replaces config (pip behavior)
    if env_extra_urls:
        candidates = list(env_extra_urls)  # already tokenized; do NOT split again
    else:
        # config entries might contain multiple URLs in a single string => split them
        candidates: List[str] = []
        for item in config_extra_raw:
            if item is None:
                continue
            candidates.extend(shlex.split(str(item)))

    # Dedup preserving order
    seen = set()
    extra_index_urls: List[str] = []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            extra_index_urls.append(u)

    return PipIndexSettings(index_url=index_url, extra_index_urls=extra_index_urls, sources=sources)


try:
    DEFAULT_PIP_INDEX_SETTINGS = get_pip_index_settings()
except:
    DEFAULT_PIP_INDEX_SETTINGS = PipIndexSettings()