import dataclasses as dc
import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import Union, Optional, List

try:
    # py3.8+
    import importlib.metadata as ilm
except Exception:  # pragma: no cover
    ilm = None  # type: ignore


__all__ = [
    "DependencyMetadata",
    "module_name_to_project_name",
    "resolve_local_lib_path",
    "module_dependencies"
]


MODULE_PROJECT_NAMES_ALIASES = {
    "yggdrasil": "ygg",
    "jwt": "PyJWT"
}


def module_name_to_project_name(module_name: str):
    return MODULE_PROJECT_NAMES_ALIASES.get(module_name, module_name)


def resolve_local_lib_path(
    lib: Union[str, ModuleType],
) -> Path:
    """
    Resolve a lib spec (path string, module name, or module object)
    into a concrete filesystem path.

    Rules:
    - If it's a path and exists:
        * if it's a file:
            - if it's __init__.py -> treat as start of a package, then walk up
              until top-most directory that still has __init__.py
            - else -> just that file
        * if it's a dir:
            - if it (or parents) contain __init__.py, walk up to the
              top-most dir that still has __init__.py
            - else -> that dir as-is
    - Else treat as module name:
        * import, use __file__, apply same logic as above.
    """
    # Module object case
    if isinstance(lib, ModuleType):
        mod = lib
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            raise ValueError(
                f"Module {mod.__name__!r} has no __file__; cannot determine path"
            )
        path = Path(mod_file).resolve()
    else:
        # First, treat as path
        p = Path(lib)
        if p.exists():
            path = p.resolve()
        else:
            # Not a path -> try as module name
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

    # If it's a file, maybe part of a package
    if path.is_file():
        # If it's __init__.py, start from its directory
        if path.name == "__init__.py":
            pkg_dir = path.parent
        else:
            # Regular module: see if its parent is a package
            pkg_dir = path.parent
    else:
        # Directory given. Might be a package root or inside a package.
        pkg_dir = path

    # Walk up to the top-most directory that still looks like a package
    # (has __init__.py). If none found, just use the original dir/file.
    top_pkg_dir = None
    current = pkg_dir

    while True:
        init_file = current / "__init__.py"
        if init_file.exists():
            top_pkg_dir = current
            # try going higher
            parent = current.parent
            # stop at filesystem root
            if parent == current:
                break
            current = parent
        else:
            break

    if top_pkg_dir is not None:
        return top_pkg_dir

    # Not in a package context -> return original path
    return path


_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")  # PEP 508-ish


@dc.dataclass(frozen=True)
class DependencyMetadata:
    project: str                 # pip/distribution name (e.g. "cryptography")
    requirement: str             # raw Requires-Dist line
    installed: bool              # is the dist installed locally
    version: Optional[str]       # installed version if present
    dist_root: Optional[Path]    # install root (site-packages path for that dist)
    metadata_path: Optional[Path]  # .../<name>-*.dist-info/METADATA if available


def _req_project_name(req_line: str) -> Optional[str]:
    left = req_line.split(";", 1)[0].strip()
    m = _REQ_NAME_RE.match(left)
    if not m:
        return None
    name = m.group(1)
    return name.split("[", 1)[0]  # drop extras


def _distribution_for_module(mod: Union[str, ModuleType]) -> Optional["ilm.Distribution"]:
    if isinstance(mod, ModuleType):
        module_name = mod.__name__
    else:
        module_name = mod
        importlib.import_module(module_name)

    top = module_name.split(".", 1)[0]
    mapping = ilm.packages_distributions()
    dists = mapping.get(top)
    if not dists:
        raise ModuleNotFoundError(f"Can't find installed distribution that provides top-level module '{top}'")

    return ilm.distribution(dists[0])


def module_dependencies(lib: Union[str, ModuleType]) -> List[DependencyMetadata]:
    """
    Return a list of DepMetadata for all Requires-Dist dependencies of `lib`'s distribution.
    - Works only when importlib.metadata is available (py3.8+).
    - If a dependency is not installed locally, installed=False and fields are None.
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

            # Best-effort METADATA path
            metadata_path = None
            try:
                meta_rel = dep_dist.read_text("METADATA")
                if meta_rel is not None:
                    # read_text worked, now try to locate the actual file path
                    # Many dists have _path pointing at the dist-info dir (private but practical).
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