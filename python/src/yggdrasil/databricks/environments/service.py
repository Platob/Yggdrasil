"""Base environments — uniform CRUD over reusable serverless + cluster images.

An environment is identified by ``(project, version)`` (and a Python tag). It is
a project's reusable image under ``/Workspace/Shared/environment/<proj>/``: a
serverless ``<proj>-<version>-py3XX.yml`` (``environment_version`` + wheel
dependency paths) and a classic-cluster ``<proj>-<version>-py3XX.requirements.txt``
that sit side by side. The dependency list is the project wheel **plus its whole
closure as wheels in the shared registry** — environments never resolve from a
live PyPI index, so the runtime installs with zero network access.

``create`` fetches the project + closure (local build or PyPI download, via
:mod:`yggdrasil.databricks.wheels`), uploads the wheels, and writes the spec
files. :meth:`Environments.find` builds on a miss; :meth:`update` / :meth:`delete`
re-create / remove. Every project — ``ygg`` included — is handled identically.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.version import VersionInfo

from ..service import DatabricksService
from ..wheels.service import (
    SUPPORTED_PYTHONS,
    WORKSPACE_ENV_DIR,
    WORKSPACE_PYPI_DIR,
    _norm,
    distribution_for,
    environment_key_for,
    fetch_wheels,
    find_pyproject,
    parse_version,
    read_pyproject,
    registry_upload,
    serverless_environment_version,
    wheel_parts,
)
from .environment import Environment

if TYPE_CHECKING:
    from ..client import DatabricksClient

logger = logging.getLogger(__name__)

__all__ = [
    "WORKSPACE_ENV_DIR",
    "environment_folder",
    "environment_stem",
    "environment_folder_of",
    "ensure_named_environment",
    "ensure_cluster_requirements",
    "deployed_environments",
    "Environments",
]

#: TTL (seconds) for the in-process deployed-environment listing cache. Walking
#: the workspace tree is a remote round-trip per folder; ``get`` / ``find`` /
#: ``resolve`` all funnel through :meth:`Environments.list`, so a short-lived
#: snapshot turns repeated lookups into a dict hit. In-process mutations
#: (``create`` / ``update`` / ``delete``) invalidate eagerly; the TTL only
#: bounds staleness from *other* processes deploying environments.
ENVIRONMENT_LIST_TTL = 300.0

# host -> ExpiringDict(workspace_root -> list[Environment])
_LIST_CACHE: "dict[str, ExpiringDict[str, list[Environment]]]" = {}


def _list_bucket(client: Any) -> "ExpiringDict[str, list[Environment]]":
    host = client.base_url.to_string() if getattr(client, "base_url", None) else "default"
    bucket = _LIST_CACHE.get(host)
    if bucket is None:
        bucket = _LIST_CACHE[host] = ExpiringDict(default_ttl=ENVIRONMENT_LIST_TTL)
    return bucket


def environment_folder(project: str = "ygg") -> str:
    """The workspace folder an environment lives in — the normalized dist name."""
    return _norm(distribution_for(project))


def environment_stem(project: str = "ygg", *, python: "str | None" = None,
                     version: "str | VersionInfo | None" = None) -> str:
    """The version-tagged file stem ``<proj>-<version>-py3XX`` (``ygg-0.8.57-py311``)."""
    folder = environment_folder(project)
    ver = parse_version(version)
    if ver is None:
        from yggdrasil.version import __version_info__
        import importlib.metadata as ilmd
        try:
            ver = parse_version(ilmd.version(distribution_for(project))) or __version_info__
        except Exception:  # noqa: BLE001
            ver = __version_info__
    return f"{folder}-{ver}-{environment_key_for(python)}"


def environment_folder_of(stem: str) -> str:
    """The project folder a versioned *stem* belongs to (``ygg-0.8.57-py311`` →
    ``ygg``); a bare name is returned unchanged."""
    import re

    parts = stem.split("-")
    if len(parts) >= 3 and re.fullmatch(r"py3\d+", parts[-1]):
        return "-".join(parts[:-2])
    return stem


def _workspace_text_unchanged(path: Any, body: str) -> bool:
    try:
        return path.exists() and path.read_text() == body
    except Exception:  # noqa: BLE001
        return False


def ensure_named_environment(client: Any, name: str, *, dependencies, environment_version=None,
                             workspace_dir: str = WORKSPACE_ENV_DIR, filename: "str | None" = None) -> str:
    """Create-or-update a serverless base-environment YAML; return its path. Upsert
    that only writes when the content differs."""
    from ..path import DatabricksPath

    version = environment_version or serverless_environment_version()
    body = "\n".join([f"environment_version: '{version}'", "dependencies:",
                      *[f"  - {dep}" for dep in dependencies]]) + "\n"
    dest = f"{workspace_dir.rstrip('/')}/{name}/{filename or f'{name}.env.yaml'}"
    path = DatabricksPath.from_(dest, client=client)
    if _workspace_text_unchanged(path, body):
        return dest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return dest


def ensure_cluster_requirements(client: Any, name: str, *, dependencies,
                                workspace_dir: str = WORKSPACE_ENV_DIR, filename: "str | None" = None) -> str:
    """Create-or-update a classic-cluster ``requirements.txt``; return its path."""
    from ..path import DatabricksPath

    body = "\n".join(str(dep) for dep in dependencies) + "\n"
    dest = f"{workspace_dir.rstrip('/')}/{name}/{filename or f'{name}.requirements.txt'}"
    path = DatabricksPath.from_(dest, client=client)
    if _workspace_text_unchanged(path, body):
        return dest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return dest


def deployed_environments(client: Any, *, workspace_dir: str = WORKSPACE_ENV_DIR) -> "list[str]":
    """Workspace paths of persisted environment files (``*.yml`` /
    ``*.requirements.txt``), walking the ``<proj>/<version>/`` layout (and the
    looser legacy ones)."""
    from ..path import DatabricksPath

    suffixes = (".env.yaml", ".yml", ".requirements.txt")
    found: "list[str]" = []

    def _walk(node: Any, depth: int) -> None:
        for child in node.iterdir():
            if str(child.name).endswith(suffixes):
                found.append(child.full_path())
            elif depth and child.is_dir():
                _walk(child, depth - 1)

    folder = DatabricksPath.from_(workspace_dir, client=client)
    if folder.exists():
        _walk(folder, depth=2)            # root → <proj>/ → <version>/ → files
    return found


class Environments(DatabricksService):
    """CRUD over reusable base environments (``dbc.environments``).

    An environment is keyed by ``(project, version)``. :meth:`create` fetches the
    project + its wheel closure and writes the serverless + cluster spec files;
    :meth:`find` builds on a miss; :meth:`update` / :meth:`delete` re-create /
    remove. The dependency closure is bundled as wheels (zero-PyPI).
    """

    default_dir: ClassVar[str] = WORKSPACE_ENV_DIR

    # -- create / update ---------------------------------------------------
    def create(
        self,
        project: "str | Path" = "ygg",
        version: "str | VersionInfo | None" = None,
        *,
        python: "str | None" = None,
        extras: "tuple[str, ...] | list[str]" = (),
        workspace_dir: "str | None" = None,
        pypi_dir: str = WORKSPACE_PYPI_DIR,
        overwrite: bool = True,
        rebuild: bool = False,
    ) -> Environment:
        """Fetch *project* + its dependency closure (local build or PyPI), upload
        the wheels, and write the serverless + cluster base environment."""
        root = workspace_dir or self.default_dir
        ver = parse_version(version)

        # Resolve project name/version (local pyproject wins; else the spec/name).
        pyproject = find_pyproject(project)
        if pyproject is not None:
            meta = read_pyproject(pyproject)
            name, ver = meta["name"], parse_version(meta["version"])
        else:
            name = str(project)

        files = fetch_wheels(project, str(ver) if ver else None, python=python,
                            deps=True, extras=extras, rebuild=rebuild)
        if ver is None:                               # derive version from the built project wheel
            ver = wheel_parts(files[0])[1]
        dependencies = [registry_upload(self.client, w, workspace_dir=pypi_dir,
                                        overwrite=overwrite and wheel_parts(w)[0] == _norm(name))
                        for w in files]

        folder = environment_folder(name)
        stem = f"{folder}-{ver}-{environment_key_for(python)}"
        # ``<proj>/<version>/`` folder levels, mirroring the wheel registry.
        subdir = f"{folder}/{ver}"
        serverless = ensure_named_environment(
            self.client, subdir, dependencies=dependencies,
            environment_version=serverless_environment_version(python),
            workspace_dir=root, filename=f"{stem}.yml",
        )
        cluster = ensure_cluster_requirements(
            self.client, subdir, dependencies=dependencies,
            workspace_dir=root, filename=f"{stem}.requirements.txt",
        )
        self.invalidate_cache()                        # newly-written env supersedes any cached listing
        return Environment(self, name=stem, project=name, version=ver, python=python,
                          env_dir=serverless.rsplit("/", 1)[0],
                          serverless=serverless, cluster=cluster, dependencies=dependencies)

    def update(self, project: "str | Path" = "ygg", version=None, **kwargs: Any) -> Environment:
        """Re-fetch and overwrite *project*'s environment."""
        kwargs.setdefault("overwrite", True)
        kwargs.setdefault("rebuild", True)
        return self.create(project, version, **kwargs)

    # -- read --------------------------------------------------------------
    def list(self, *, workspace_dir: "str | None" = None,
             refresh: bool = False) -> "list[Environment]":
        """The deployed base environments, one :class:`Environment` per stem.

        Cached per ``(host, workspace_dir)`` for :data:`ENVIRONMENT_LIST_TTL`
        seconds so the ``get`` / ``find`` / ``resolve`` hot path skips the
        workspace walk; pass ``refresh=True`` to force a fresh read."""
        root = workspace_dir or self.default_dir
        bucket = _list_bucket(self.client)
        if not refresh:
            cached = bucket.get(root)
            if cached is not None:
                return cached
        by_stem: "dict[str, dict[str, Any]]" = {}
        for path in deployed_environments(self.client, workspace_dir=root):
            folder, fname = path.rsplit("/", 1)
            for suffix in (".env.yaml", ".requirements.txt", ".yml"):
                if fname.endswith(suffix):
                    stem = fname[: -len(suffix)]
                    slot = by_stem.setdefault(stem, {"stem": stem, "folder": folder})
                    slot["cluster" if suffix == ".requirements.txt" else "serverless"] = path
                    break
        out: "list[Environment]" = []
        for info in by_stem.values():
            stem = info["stem"]
            project = environment_folder_of(stem)
            parts = stem.split("-")
            version = parse_version(parts[-2]) if len(parts) >= 3 else None
            out.append(Environment(self, name=stem, project=project, version=version,
                                  env_dir=info["folder"], serverless=info.get("serverless"),
                                  cluster=info.get("cluster")))
        bucket[root] = out
        return out

    def get(self, project: "str | Path" = "ygg", version=None, *, python=None,
            workspace_dir: "str | None" = None, refresh: bool = False) -> "Optional[Environment]":
        """The deployed environment for *project* (matching *version* / *python*),
        or ``None`` — never builds."""
        return self.find(project, version, install=False, python=python,
                         workspace_dir=workspace_dir, refresh=refresh)

    def find(
        self,
        project: "str | Path" = "ygg",
        version: "VersionInfo | str | None" = None,
        *,
        install: bool = True,
        python: "str | None" = None,
        extras: "tuple[str, ...] | list[str]" = (),
        workspace_dir: "str | None" = None,
        refresh: bool = False,
    ) -> "Optional[Environment]":
        """Find *project*'s base environment **for a Python** (its ``py3XX`` tag,
        defaulting to the local interpreter); build + write it (from a local
        pyproject or PyPI) when missing and *install* (the default)."""
        name = _project_name(project)
        ver = parse_version(version)
        key = environment_key_for(python)              # py3XX — the env's Python tag
        envs = [e for e in self.list(workspace_dir=workspace_dir, refresh=refresh)
                if environment_folder_of(e.name) == environment_folder(name)
                and e.name.endswith(f"-{key}")
                and (ver is None or e.version == ver)]
        if envs:
            return max(envs, key=lambda e: e.version or VersionInfo(0, 0, 0))
        if not install:
            return None
        return self.create(project, ver, python=python, extras=extras,
                          workspace_dir=workspace_dir, overwrite=False)

    def invalidate_cache(self) -> None:
        """Drop this client's cached :meth:`list` snapshots — call after the
        workspace's environments change out-of-band (the in-process ``create`` /
        ``update`` / ``delete`` paths do this for you)."""
        host = (self.client.base_url.to_string()
                if getattr(self.client, "base_url", None) else "default")
        bucket = _LIST_CACHE.get(host)
        if bucket is not None:
            bucket.clear()

    def client_project(self, *, workspace_dir: "str | None" = None,
                       refresh: bool = False) -> "Optional[Environment]":
        """The **running client project's** deployed environment — discovered
        from the nearest ``pyproject.toml`` (walking up from the cwd), matched by
        its ``[project].name`` / ``version`` for the local Python. ``None`` when
        there's no pyproject or its environment isn't deployed (best-effort)."""
        try:
            meta = read_pyproject(find_pyproject())
        except Exception:  # noqa: BLE001 — no/unreadable pyproject → no project default
            return None
        return self.get(meta["name"], meta["version"], workspace_dir=workspace_dir, refresh=refresh)

    def resolve(self, ref: "str | None" = None, *,
                workspace_dir: "str | None" = None,
                refresh: bool = False) -> "Optional[Environment]":
        """Resolve a base-environment *reference* to a deployed :class:`Environment`.

        - a ``str`` carrying a ``/`` or a ``.yml`` / ``.yaml`` suffix — a **direct
          workspace path** to a serverless spec (``None`` when it's absent);
        - any other ``str`` — a deployed **stem name** (``ygg-<version>-py3XX``),
          looked up among :meth:`list`;
        - ``None`` — **auto**: the running :meth:`client_project`, else the ``ygg``
          base environment for the current Python, else ``None``.

        Discovery rides the cached :meth:`list` snapshot; pass ``refresh=True`` to
        bypass it.
        """
        from ..path import DatabricksPath

        if isinstance(ref, str) and ("/" in ref or ref.endswith((".yml", ".yaml"))):
            path = DatabricksPath.from_(ref, client=self.client)
            if not path.exists():
                return None
            full = path.full_path()
            name = full.rsplit("/", 1)[-1].removesuffix(".yaml").removesuffix(".yml").removesuffix(".env")
            return Environment(self, name=name, env_dir=full.rsplit("/", 1)[0], serverless=full)
        if isinstance(ref, str):
            for env in self.list(workspace_dir=workspace_dir, refresh=refresh):
                if env.name == ref:
                    return env
            return None
        return (self.client_project(workspace_dir=workspace_dir, refresh=refresh)
                or self.get("ygg", workspace_dir=workspace_dir, refresh=refresh))

    # -- delete ------------------------------------------------------------
    def delete(self, project: "str | Path" = "ygg", version=None, *,
               workspace_dir: "str | None" = None) -> "list[Environment]":
        """Delete *project*'s environment file(s) — a specific *version*, or all."""
        name = _project_name(project)
        ver = parse_version(version)
        envs = [e for e in self.list(workspace_dir=workspace_dir)
                if environment_folder_of(e.name) == environment_folder(name)
                and (ver is None or e.version == ver)]
        for e in envs:
            e.delete()
        if envs:
            self.invalidate_cache()
        return envs


def _project_name(spec: "str | Path") -> str:
    pyproject = find_pyproject(spec)
    if pyproject is not None:
        try:
            return read_pyproject(pyproject)["name"]
        except Exception:  # noqa: BLE001
            pass
    return str(spec)
