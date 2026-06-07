"""Assemble serverless base environments + cluster requirements, and deploy projects.

A *base environment* is the reusable spec a Databricks job/cluster installs from
— a serverless ``<proj>-<version>-py3XX.yml`` (``environment_version`` + wheel
dependency paths) plus a classic-cluster ``<proj>-<version>-py3XX.requirements.txt``
— living under ``/Workspace/Shared/environment/<proj>/``. This module owns
writing them (:func:`ensure_named_environment` / :func:`ensure_cluster_requirements`
/ :func:`ensure_environment`), the project deploy that discovers a
``pyproject.toml`` and builds its own image (:func:`ensure_project_environment`),
and the ygg ``JobEnvironment`` assembly (:func:`ygg_environment`).

Wheel building lives next door in :mod:`yggdrasil.databricks.wheels`; this module
composes it. The :class:`Environments` service (``dbc.environments``) is the OO
front door, returning :class:`~yggdrasil.databricks.environments.environment.Environment`
handles.
"""
from __future__ import annotations

import importlib.metadata as ilmd
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from yggdrasil.enums.mode import Mode

from ..service import DatabricksService
from ..wheels.service import (
    SUPPORTED_PYTHONS,
    WORKSPACE_PYPI_DIR,
    _norm,
    _py_minor,
    _project_dependencies,
    _workspace_text_unchanged,
    build_project_wheel,
    deployed_wheels,
    distribution_for,
    download_dependency_wheels,
    ensure_bundle,
    ensure_ygg_wheel,
    ensure_ygg_wheels,
    environment_key_for,
    registry_upload,
    serverless_environment_version,
    wheel_for_python,
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
    "ygg_base_environment_name",
    "ensure_named_environment",
    "ensure_cluster_requirements",
    "ensure_environment",
    "ensure_environments",
    "find_pyproject",
    "read_pyproject",
    "ensure_project_environment",
    "deployed_environments",
    "ygg_runtime_dependencies",
    "ygg_environment",
    "ygg_environments",
    "Environments",
]



#: Where reusable serverless **base environments** live. The layout mirrors the
#: wheel registry (:data:`WORKSPACE_PYPI_DIR`): one folder **per project**, named
#: for the project/dist (no version, no python tag), holding its *version-tagged*
#: spec files + the zero-PyPI wheel closure::
#:
#:     /Workspace/Shared/environment/<proj>/
#:         <proj>-<version>-py3XX.yml             serverless base_environment
#:         <proj>-<version>-py3XX.requirements.txt   classic-cluster requirements
#:
#: e.g. ``environment/ygg/ygg-0.8.57-py311.yml``. Every version / python lands in
#: the **same** ``<proj>/`` folder (the filenames carry the version + ``py3XX``
#: tag), exactly like ``pypi/<dist>/<dist>-<version>-…whl`` — uniform with the
#: wheel creations.
#:
#: A job references the ``.yml`` by file path via ``Environment.base_environment``
#: instead of inlining the whole dependency list (see
#: :func:`ensure_named_environment`); its dependencies are **wheels in the shared
#: pypi registry** (:data:`WORKSPACE_PYPI_DIR`, built by :func:`ensure_environment`),
#: so the env is self-describing and the runtime installs with zero PyPI access.
WORKSPACE_ENV_DIR = "/Workspace/Shared/environment"


def environment_folder(project: str = "ygg") -> str:
    """The workspace **folder** an environment lives in: the project/dist name,
    normalized (``yggdrasil`` / ``ygg`` → ``ygg``) — **no version, no python
    tag**, mirroring the wheel registry's ``<dist>/`` folder. Accepts an import
    package or a distribution name."""
    return _norm(distribution_for(project))


def environment_stem(
    project: str = "ygg",
    *,
    python: "str | None" = None,
    version: "str | None" = None,
) -> str:
    """The version-tagged **file stem** of an environment — ``<proj>-<version>-py3XX``
    (e.g. ``ygg-0.8.57-py311``).

    The ``.yml`` / ``.requirements.txt`` files inside the project folder carry
    this stem, uniform with the wheel registry's ``<dist>-<version>-…whl``.
    *version* defaults to the installed *project* version (falling back to the
    in-tree ygg version), *python* to the local interpreter."""
    dist = distribution_for(project)
    if version is None:
        try:
            version = ilmd.version(dist)
        except Exception:  # noqa: BLE001 — fall back to the in-tree version
            from yggdrasil.version import __version__ as version
    return f"{_norm(dist)}-{version}-{environment_key_for(python)}"


def environment_folder_of(stem: str) -> str:
    """The project folder a versioned environment *stem* belongs to: the stem with
    its trailing ``-<version>-py3XX`` stripped (``ygg-0.8.57-py311`` → ``ygg``,
    ``my-proj-1.2.3-py312`` → ``my-proj``).

    A bare name with no ``py3XX`` python-tag suffix (a hand-written named env) is
    returned unchanged, so the old ``<name>/<name>.yml`` layout still resolves."""
    parts = stem.split("-")
    if len(parts) >= 3 and re.fullmatch(r"py3\d+", parts[-1]):
        return "-".join(parts[:-2])
    return stem


def ygg_base_environment_name(python: "str | None" = None) -> str:
    """Canonical name (version-tagged file stem) of the reusable serverless
    **base environment** for the running ygg image — ``ygg-<version>-py3XX``.

    This is exactly the stem ``ygg databricks seed`` writes under
    :data:`WORKSPACE_ENV_DIR` in the project folder
    (``environment/ygg/ygg-<version>-py3XX.yml``), so a job that points its
    ``base_environment_name`` here reuses the seeded, wheel-built image when the
    seed has run — and self-provisions the identical file (same wheel closure,
    same path) when it hasn't. The version-pinned name is the single source of
    truth for "the correct ygg environment", replacing the old static
    ``yellow`` env."""
    return environment_stem("ygg", python=python)


def ensure_named_environment(
    client: Any,
    name: str = "yellow",
    *,
    dependencies: "list[str] | tuple[str, ...]",
    environment_version: "str | None" = None,
    workspace_dir: str = WORKSPACE_ENV_DIR,
    filename: "str | None" = None,
) -> str:
    """Create-or-update a reusable serverless **base environment** *name* as a
    YAML file in the workspace and return its path.

    A serverless job can reference this file via
    ``Environment.base_environment`` instead of inlining the whole dependency
    list — so one shared, named environment is defined once and every ygg job
    points at it. The file is the documented serverless env spec; its
    *dependencies* are **built wheels in the workspace pypi registry**
    (:func:`ensure_bundle`) so the runtime installs with zero PyPI access::

        environment_version: '5'
        dependencies:
          - /Workspace/Shared/pypi/ygg/ygg-0.8.54-py3-none-any.whl
          - /Workspace/Shared/pypi/pyarrow/pyarrow-...-cp312-...-.whl

    Here *name* is the **project folder** (e.g. ``ygg``); the file is
    ``<name>.env.yaml`` unless *filename* overrides it — the seed writes a
    version-pinned ``<proj>-<version>-py3XX.yml`` so jobs can point at an exact
    image and every version coexists in the one project folder. **Upsert that
    only writes when the content differs**: an existing file whose body already
    matches is left untouched (no needless re-stamp / churn); a missing or
    drifted file is (over)written. *dependencies* are wheel workspace paths
    (and/or pip requirement lines, when an index resolve is wanted instead)."""
    from yggdrasil.databricks.path import DatabricksPath

    version = environment_version or serverless_environment_version()
    lines = [f"environment_version: '{version}'", "dependencies:"]
    lines += [f"  - {dep}" for dep in dependencies]
    body = "\n".join(lines) + "\n"

    # The environment lives under the project's ``<workspace_dir>/<name>/`` folder
    # (mirroring the wheel registry's ``<dist>/``); *filename* carries the version
    # + ``py3XX`` tag so every build coexists rather than overwriting a flat file.
    dest = f"{workspace_dir.rstrip('/')}/{name}/{filename or f'{name}.env.yaml'}"
    path = DatabricksPath.from_(dest, client=client)
    if _workspace_text_unchanged(path, body):
        logger.info("serverless base environment %r unchanged -> %s", name, dest)
        return dest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    logger.info(
        "wrote serverless base environment %r -> %s (%d deps, env v%s)",
        name, dest, len(dependencies), version,
    )
    return dest


def ensure_cluster_requirements(
    client: Any,
    name: str = "yellow",
    *,
    dependencies: "list[str] | tuple[str, ...]",
    workspace_dir: str = WORKSPACE_ENV_DIR,
    filename: "str | None" = None,
) -> str:
    """Create-or-update a plain ``<name>.requirements.txt`` in the workspace and
    return its path — the **classic-cluster** counterpart of
    :func:`ensure_named_environment`.

    Serverless references a base environment by path (``environment_version`` +
    dependencies); a classic cluster has no such concept — it installs from a
    pip requirements file via ``Library(requirements=<path>)``. So the same ygg
    image is written here as a flat requirements list (wheel workspace paths +
    pinned index requirements, no ``environment_version`` line)::

        /Workspace/Shared/pypi/ygg/ygg-0.8.54-py3-none-any.whl
        pyarrow==...

    **Upsert that only writes when the content differs** — an existing file whose
    body already matches is left untouched; a missing or drifted file is
    (over)written. *dependencies* are wheel workspace paths and/or pip
    requirement lines (typically the same list fed to
    :func:`ensure_named_environment`)."""
    from yggdrasil.databricks.path import DatabricksPath

    body = "\n".join(str(dep) for dep in dependencies) + "\n"
    # Sits beside the serverless ``.yml`` in the project's ``<name>/`` folder;
    # *filename* (a version-tagged ``<stem>.requirements.txt``) overrides the
    # default flat name so every version coexists in the one project folder.
    dest = f"{workspace_dir.rstrip('/')}/{name}/{filename or f'{name}.requirements.txt'}"
    path = DatabricksPath.from_(dest, client=client)
    if _workspace_text_unchanged(path, body):
        logger.info("cluster requirements %r unchanged -> %s", name, dest)
        return dest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    logger.info(
        "wrote cluster requirements %r -> %s (%d deps)",
        name, dest, len(dependencies),
    )
    return dest


def ensure_environment(
    client: Any,
    *,
    python: "str | None" = None,
    version: "str | None" = None,
    workspace_dir: str = WORKSPACE_ENV_DIR,
    rebuild: bool = False,
    mode: Mode = Mode.AUTO,
) -> "dict[str, Any]":
    """Build + persist one **self-contained** ygg base environment for a single
    Python version, returning a small descriptor of what was written.

    Lays the environment out under the **project folder** (``ygg``), mirroring
    the wheel registry — every version / python coexists in the one folder, the
    filenames carry the ``<version>-py3XX`` tag, and the env holds **only its
    spec files** (no per-env binaries)::

        <workspace_dir>/ygg/
            ygg-<version>-py3XX.yml             serverless base_environment
            ygg-<version>-py3XX.requirements.txt   classic-cluster requirements

    The wheel closure is built (:func:`build_bundle`) and uploaded into the
    **shared pypi registry** (:data:`WORKSPACE_PYPI_DIR` — ``pypi/<dist>/<wheel>``),
    so dependency wheels are shared across images/versions instead of being
    duplicated per environment. The serverless ``.yml`` and cluster
    ``requirements.txt`` list those pypi wheel paths, so the runtime installs
    with zero PyPI access.

    *mode* sets the env-config-file policy (the wheel closure is **get-or-create**
    unless *rebuild*): :data:`Mode.APPEND` writes the ``.yml`` / ``.requirements.txt``
    only when they don't exist yet; :data:`Mode.AUTO` (default) and
    :data:`Mode.OVERWRITE` upsert them — **overwriting only when the content
    differs** so an unchanged redeploy leaves the files untouched.

    Returns ``{python, key, env_name, env_dir, n_wheels, serverless, cluster}``
    where ``env_name`` is the version-tagged stem and ``env_dir`` the project
    folder.
    """
    from yggdrasil.databricks.path import DatabricksPath

    overwrite_env = Mode.from_(mode) is not Mode.APPEND
    version = version or ilmd.version("ygg")
    key = environment_key_for(python)
    folder = environment_folder("ygg")                     # project folder: ``ygg``
    env_name = f"{folder}-{version}-{key}"                 # versioned file stem
    env_dir = f"{workspace_dir.rstrip('/')}/{folder}"

    # Wheels go to the shared pypi registry (not a per-env ``binaries/``), so the
    # dependency closure is reused across images/versions; the env files just
    # reference those pypi paths.
    bundle = ensure_bundle(client, "ygg", python=python, rebuild=rebuild)
    serverless_dest = f"{env_dir}/{env_name}.yml"
    cluster_dest = f"{env_dir}/{env_name}.requirements.txt"
    if overwrite_env or not DatabricksPath.from_(serverless_dest, client=client).exists():
        serverless = ensure_named_environment(
            client, folder, dependencies=bundle,
            environment_version=serverless_environment_version(python),
            workspace_dir=workspace_dir, filename=f"{env_name}.yml",
        )
    else:
        serverless = serverless_dest
    if overwrite_env or not DatabricksPath.from_(cluster_dest, client=client).exists():
        cluster = ensure_cluster_requirements(
            client, folder, dependencies=bundle, workspace_dir=workspace_dir,
            filename=f"{env_name}.requirements.txt",
        )
    else:
        cluster = cluster_dest
    return {
        "python": python,
        "key": key,
        "env_name": env_name,
        "env_dir": env_dir,
        "n_wheels": len(bundle),
        "serverless": serverless,
        "cluster": cluster,
    }


def ensure_environments(
    client: Any,
    *,
    versions: "tuple[str | None, ...] | list[str | None]" = (None,),
    workspace_dir: str = WORKSPACE_ENV_DIR,
    rebuild: bool = False,
    mode: Mode = Mode.AUTO,
    max_workers: "int | None" = None,
) -> "list[dict[str, Any]]":
    """:func:`ensure_environment` for several Python versions, **in parallel**.

    Each version's environment is an independent folder with its own wheel
    closure, so the builds share nothing and run concurrently on a
    :class:`~concurrent.futures.ThreadPoolExecutor` (the work is subprocess-bound
    — uv / pip — so threads give real overlap). Results are returned in the input
    order regardless of completion order. A single version skips the pool and
    runs inline. *mode* is forwarded to each :func:`ensure_environment`."""
    versions = list(versions) or [None]
    if len(versions) == 1:
        return [ensure_environment(
            client, python=versions[0], workspace_dir=workspace_dir,
            rebuild=rebuild, mode=mode,
        )]

    results: "dict[Any, dict[str, Any]]" = {}
    with ThreadPoolExecutor(
        max_workers=max_workers or len(versions), thread_name_prefix="ygg-env",
    ) as pool:
        futures = {
            pool.submit(
                ensure_environment,
                client, python=py, workspace_dir=workspace_dir,
                rebuild=rebuild, mode=mode,
            ): py
            for py in versions
        }
        for future in as_completed(futures):
            py = futures[future]
            results[py] = future.result()
    return [results[py] for py in versions]


# ---------------------------------------------------------------------------
# Arbitrary on-disk projects — discover a pyproject.toml, build it, and write a
# project-named environment (the user-project counterpart of the ygg image).
# ---------------------------------------------------------------------------


def find_pyproject(start: str | Path | None = None) -> Path:
    """The nearest ``pyproject.toml`` at or above *start* (cwd by default).

    *start* may point at the file itself, at its directory, or at any nested
    directory — the search walks up to the first ``pyproject.toml``. Raises
    :class:`FileNotFoundError` when none exists on the way up to the root."""
    start_path = Path(start).resolve() if start else Path.cwd()
    if start_path.is_file():
        if start_path.name == "pyproject.toml":
            return start_path
        start_path = start_path.parent
    for directory in (start_path, *start_path.parents):
        candidate = directory / "pyproject.toml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no pyproject.toml found at or above {start_path}")


def read_pyproject(path: str | Path) -> dict[str, Any]:
    """Parse a ``pyproject.toml``'s ``[project]`` table into what the deploy
    needs: ``name``, ``version``, base ``dependencies``, ``optional_dependencies``
    (keyed by extra), ``requires_python``, and the project ``dir``."""
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:                       # Python 3.10 has no tomllib
        import tomli as toml_reader

    path = Path(path).resolve()
    data = toml_reader.loads(path.read_text(encoding="utf-8"))
    project = data.get("project") or {}
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


def ensure_project_environment(
    client: Any,
    pyproject: str | Path | None = None,
    *,
    python: str | None = None,
    extras: tuple[str, ...] | list[str] = (),
    bundle: bool = False,
    mode: Mode = Mode.AUTO,
    workspace_dir: str = WORKSPACE_ENV_DIR,
    pypi_dir: str = WORKSPACE_PYPI_DIR,
) -> dict[str, Any]:
    """Discover a project's ``pyproject.toml``, build its wheel, and write a
    serverless **base environment** + classic-cluster **requirements** named for
    the project (``<name>-<version>``) — the user-project counterpart of
    :func:`ensure_environment`.

    The environment's dependency list is the **project wheel** plus the
    project's own ``[project].dependencies`` (and any requested *extras*' deps).
    With ``bundle=True`` those dependencies are downloaded as Linux-x86_64 wheels
    into the **shared pypi registry** (``pypi/<dist>/<wheel>``) and listed by
    workspace path, so the runtime installs with zero PyPI access; otherwise
    they're listed as index requirements resolved at install time.

    *mode* (a :class:`~yggdrasil.enums.Mode`) sets the idempotency policy:

    - :data:`Mode.OVERWRITE` — rebuild the wheel(s) and **overwrite** everything
      (the deployed wheel and the env config files).
    - :data:`Mode.APPEND` — **add only what's missing**: reuse an already-deployed
      wheel, and write the env config files only when they don't exist yet.
    - :data:`Mode.AUTO` (default) — **get-or-create** the wheel(s) (reuse when
      already deployed, build when not) and **upsert** the env config files,
      overwriting them only when the content differs so an unchanged redeploy is
      a no-op.

    Returns a descriptor with the project name/version, the env name, the written
    file paths, the dependency list, and the resolved *mode*.
    """
    from yggdrasil.databricks.path import DatabricksPath

    mode = Mode.from_(mode)
    rebuild = mode is Mode.OVERWRITE           # OVERWRITE rebuilds wheels
    overwrite_env = mode is not Mode.APPEND    # OVERWRITE + AUTO rewrite env files

    meta = read_pyproject(find_pyproject(pyproject))
    name, version = meta["name"], meta["version"]
    proj = _norm(name)
    # Folder = project name (mirrors the wheel registry's ``<dist>/``); the env
    # files carry the version + ``py3XX`` tag, so versions coexist in the folder.
    key = environment_key_for(python)
    env_name = f"{proj}-{version}-{key}"
    env_dir = f"{workspace_dir.rstrip('/')}/{proj}"

    # The project's declared deps, with any requested extras flattened in.
    deps = list(meta["dependencies"])
    for extra in extras:
        deps += meta["optional_dependencies"].get(extra, [])

    if bundle:
        # Zero-PyPI: project wheel + dependency closure, all uploaded to the
        # shared pypi registry and listed by workspace path. A ``.manifest`` under
        # the project's registry folder records the full path set so a
        # get-or-create (non-OVERWRITE) deploy can reuse the closure without
        # rebuilding.
        manifest = DatabricksPath.from_(
            f"{pypi_dir.rstrip('/')}/{proj}/{env_name}.manifest", client=client,
        )
        reused = (
            [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
            if (not rebuild and manifest.exists()) else []
        )
        if reused and DatabricksPath.from_(reused[0], client=client).exists():
            logger.info("reusing %d-wheel project bundle for %s", len(reused), env_name)
            dependencies = reused
        else:
            dependencies = [
                registry_upload(client, w, workspace_dir=pypi_dir, overwrite=True)
                for w in build_project_wheel(meta["dir"], python=python)
            ]
            dependencies += [
                registry_upload(client, w, workspace_dir=pypi_dir)
                for w in download_dependency_wheels(deps, python=python)
            ]
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("\n".join(dependencies) + "\n")
    else:
        # Project wheel by path + its declared deps resolved from the index.
        proj_dir = f"{pypi_dir.rstrip('/')}/{proj}"
        existing = (
            [] if rebuild
            else deployed_wheels(client, name, version, workspace_dir=proj_dir, dist_only=True)
        )
        if existing:
            logger.info("reusing deployed project wheel(s) for %s", env_name)
            wheel_paths = existing
        else:
            wheel_paths = [
                registry_upload(client, w, workspace_dir=pypi_dir, overwrite=True)
                for w in build_project_wheel(meta["dir"], python=python)
            ]
        dependencies = wheel_paths + deps

    # Env config files: OVERWRITE/AUTO always rewrite; APPEND writes only the
    # ones that don't exist yet ("add missing").
    serverless_dest = f"{env_dir}/{env_name}.yml"
    cluster_dest = f"{env_dir}/{env_name}.requirements.txt"
    if overwrite_env or not DatabricksPath.from_(serverless_dest, client=client).exists():
        serverless = ensure_named_environment(
            client, proj, dependencies=dependencies,
            environment_version=serverless_environment_version(python),
            workspace_dir=workspace_dir, filename=f"{env_name}.yml",
        )
    else:
        serverless = serverless_dest
    if overwrite_env or not DatabricksPath.from_(cluster_dest, client=client).exists():
        cluster = ensure_cluster_requirements(
            client, proj, dependencies=dependencies, workspace_dir=workspace_dir,
            filename=f"{env_name}.requirements.txt",
        )
    else:
        cluster = cluster_dest
    return {
        "name": name,
        "version": version,
        "env_name": env_name,
        "env_dir": env_dir,
        "dependencies": dependencies,
        "n_wheels": len(dependencies),
        "serverless": serverless,
        "cluster": cluster,
        "requires_python": meta["requires_python"],
        "mode": mode.name,
    }


def deployed_environments(client: Any, *, workspace_dir: str = WORKSPACE_ENV_DIR) -> list[str]:
    """Workspace paths of persisted environment files under *workspace_dir* —
    serverless base environments (``*.env.yaml`` / ``*.yml``, e.g. the
    version-pinned ``ygg-<version>-py3XX.yml``) and cluster requirement files
    (``*.requirements.txt``).

    Each environment lives in its **own ``<env-name>/`` folder** now
    (:func:`ensure_environment`), so this descends one level into those folders;
    loose files left directly under *workspace_dir* by older deploys are still
    picked up for back-compat. The environment-layer counterpart of
    :func:`deployed_wheels`: lets ``ygg databricks seed --check`` report whether
    the reusable environment files were written. Empty when the directory is
    absent or holds none."""
    from yggdrasil.databricks.path import DatabricksPath

    folder = DatabricksPath.from_(workspace_dir, client=client)
    if not folder.exists():
        return []

    suffixes = (".env.yaml", ".yml", ".requirements.txt")
    found: list[str] = []
    for child in folder.iterdir():
        if str(child.name).endswith(suffixes):     # legacy flat file
            found.append(child.full_path())
        elif child.is_dir():                        # per-env folder
            for sub in child.iterdir():
                if str(sub.name).endswith(suffixes):
                    found.append(sub.full_path())
    return found


def ygg_runtime_dependencies() -> list[str]:
    """The ygg image's runtime dependency **requirements** (names + version
    pins), for a serverless env to resolve from the workspace index.

    The live ``yggdrasil`` package's declared dependencies plus its
    ``[databricks]`` extra (which pins the latest ``databricks-sdk``). Shipped
    as names — not wheels — so the serverless runtime installs platform-correct
    builds. ``pyarrow`` / ``numpy`` and other binary deps therefore resolve on
    the cluster instead of being bundled from the build host.

    A bare, unpinned name (e.g. ``"xxhash"``) is pinned to its installed
    version so it reads as an unambiguous **index** requirement — otherwise the
    Spark Connect registry mistakes an installed-but-unpinned dep for a *local*
    package and tries to build a wheel for it."""
    _OPS = ("==", ">=", "<=", "!=", "~=", "===", ">", "<")
    out: list[str] = []
    for dep in _project_dependencies("ygg", {"databricks"}):
        head = dep.split(";", 1)[0].strip()
        if "[" in head or any(op in head for op in _OPS):
            out.append(dep)
            continue
        try:
            out.append(f"{head}=={ilmd.version(head)}")
        except Exception:  # noqa: BLE001 - not locally installed; ship the bare name
            out.append(dep)
    return out


def ygg_environment(
    client: Any,
    *,
    environment_key: str = "default",
    environment_version: "str | None" = None,
    rebuild: bool = False,
    workspace_dir: str = WORKSPACE_PYPI_DIR,
) -> Any:
    """The serverless ``JobEnvironment`` for the **versioned ygg image**.

    Pairs the serverless runtime — ``environment_version``, defaulting to
    :func:`serverless_environment_version` so the cluster Python matches the
    local interpreter (the locally-built wheel installs and UDFs run) — with:

    - the get-or-created pure-python ygg wheel (:func:`ensure_ygg_wheel`),
      installed **by path**; and
    - its runtime dependencies (:func:`ygg_runtime_dependencies`) as **index
      requirements**, so ``pyarrow`` / ``polars`` / ``databricks-sdk`` / … land
      as platform-correct builds the serverless runtime can actually install.

    Drop this into any serverless job's ``environments=[...]`` so its
    python-wheel tasks run the ``ygg`` CLI against a pinned image."""
    from databricks.sdk.service.compute import Environment
    from databricks.sdk.service.jobs import JobEnvironment

    wheels = ensure_ygg_wheel(
        client, workspace_dir=workspace_dir, rebuild=rebuild,
    )
    dependencies = list(wheels) + ygg_runtime_dependencies()
    return JobEnvironment(
        environment_key=environment_key,
        spec=Environment(
            environment_version=environment_version or serverless_environment_version(),
            dependencies=dependencies,
        ),
    )


def ygg_environments(
    client: Any,
    *,
    versions: "tuple[str, ...] | list[str]" = SUPPORTED_PYTHONS,
    default_python: "str | None" = None,
    rebuild: bool = False,
    workspace_dir: str = WORKSPACE_PYPI_DIR,
) -> list:
    """A serverless ``JobEnvironment`` **for every Python version** — the matrix
    counterpart of :func:`ygg_environment`.

    Returns ``[default, py310, py311, py312, py313]``: a ``"default"`` env pinned
    to *default_python* (the local interpreter unless given) plus one keyed
    ``py3XX`` per :data:`SUPPORTED_PYTHONS`, each pairing the matching ygg wheel
    (:func:`wheel_for_python`, installed by path) with the runtime deps from the
    index. Attach the whole list to a job's ``environments=[...]`` and point each
    task at the ``environment_key`` for the Python it needs; the default keeps the
    local-matched behaviour."""
    from databricks.sdk.service.compute import Environment
    from databricks.sdk.service.jobs import JobEnvironment

    wheels = ensure_ygg_wheels(
        client, versions=versions, workspace_dir=workspace_dir, rebuild=rebuild,
    )
    runtime = ygg_runtime_dependencies()
    default_python = _py_minor(default_python)

    def _env(key: str, python: str) -> Any:
        return JobEnvironment(
            environment_key=key,
            spec=Environment(
                environment_version=serverless_environment_version(python),
                dependencies=[wheel_for_python(wheels, python)] + runtime,
            ),
        )

    envs = [_env("default", default_python)]
    envs += [_env(environment_key_for(v), v) for v in versions]
    return envs


class Environments(DatabricksService):
    """Assemble + deploy base environments and project images.

    The OO front door (``dbc.environments``) over this module's functions. CRUD
    verbs return :class:`Environment` handles (or, for the job-spec helpers, the
    SDK ``JobEnvironment`` a job's ``environments=[…]`` takes directly).
    """

    #: Default environment root (overridable per call via ``workspace_dir=``).
    default_dir: ClassVar[str] = WORKSPACE_ENV_DIR

    def _from_info(self, info: "dict[str, Any]") -> Environment:
        return Environment(
            self,
            name=info["env_name"],
            project=info.get("name"),
            env_dir=info.get("env_dir"),
            serverless=info.get("serverless"),
            cluster=info.get("cluster"),
            dependencies=info.get("dependencies", ()),
            version=info.get("version"),
            python=info.get("python"),
        )

    # -- deploy ------------------------------------------------------------
    def deploy_project(
        self,
        path: "str | Path | None" = None,
        *,
        extras: "tuple[str, ...] | list[str]" = (),
        bundle: bool = False,
        mode: Mode = Mode.AUTO,
        workspace_dir: Optional[str] = None,
        pypi_dir: str = WORKSPACE_PYPI_DIR,
    ) -> Environment:
        """Discover a project's ``pyproject.toml`` (from *path* or the cwd), build
        its wheel, and write its serverless + cluster base environment."""
        info = ensure_project_environment(
            self.client, path, extras=extras, bundle=bundle, mode=mode,
            workspace_dir=workspace_dir or self.default_dir, pypi_dir=pypi_dir,
        )
        return self._from_info(info)

    def deploy_ygg(
        self,
        *,
        versions: "tuple[str | None, ...] | list[str | None]" = (None,),
        rebuild: bool = False,
        mode: Mode = Mode.AUTO,
        workspace_dir: Optional[str] = None,
    ) -> list[Environment]:
        """Build + persist the self-contained ygg base environment(s) — one per
        requested Python version (default: the local interpreter)."""
        infos = ensure_environments(
            self.client, versions=versions, workspace_dir=workspace_dir or self.default_dir,
            rebuild=rebuild, mode=mode,
        )
        return [self._from_info(i) for i in infos]

    def named(
        self,
        name: str,
        *,
        dependencies: "list[str] | tuple[str, ...]",
        environment_version: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Create-or-update a serverless base environment YAML; return its path."""
        return ensure_named_environment(
            self.client, name, dependencies=dependencies,
            environment_version=environment_version,
            workspace_dir=workspace_dir or self.default_dir, filename=filename,
        )

    def cluster_requirements(
        self,
        name: str,
        *,
        dependencies: "list[str] | tuple[str, ...]",
        workspace_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Create-or-update a classic-cluster requirements file; return its path."""
        return ensure_cluster_requirements(
            self.client, name, dependencies=dependencies,
            workspace_dir=workspace_dir or self.default_dir, filename=filename,
        )

    # -- job specs ---------------------------------------------------------
    def job_environment(
        self,
        *,
        environment_key: str = "default",
        environment_version: Optional[str] = None,
        rebuild: bool = False,
        workspace_dir: str = WORKSPACE_PYPI_DIR,
    ) -> Any:
        """The serverless ``JobEnvironment`` for the versioned ygg image — drop
        into a job's ``environments=[…]``."""
        return ygg_environment(
            self.client, environment_key=environment_key,
            environment_version=environment_version, rebuild=rebuild,
            workspace_dir=workspace_dir,
        )

    def job_environments(
        self,
        *,
        versions: "tuple[str, ...] | list[str]" = SUPPORTED_PYTHONS,
        default_python: Optional[str] = None,
        rebuild: bool = False,
        workspace_dir: str = WORKSPACE_PYPI_DIR,
    ) -> list:
        """A serverless ``JobEnvironment`` per supported Python (plus a default)."""
        return ygg_environments(
            self.client, versions=versions, default_python=default_python,
            rebuild=rebuild, workspace_dir=workspace_dir,
        )

    # -- read --------------------------------------------------------------
    def list(self, *, workspace_dir: Optional[str] = None) -> list[Environment]:
        """The deployed base environments under *workspace_dir*, one
        :class:`Environment` per ``<stem>`` (its ``.yml`` + ``.requirements.txt``)."""
        root = workspace_dir or self.default_dir
        by_stem: "dict[str, dict[str, Any]]" = {}
        for path in deployed_environments(self.client, workspace_dir=root):
            fname = path.rsplit("/", 1)[-1]
            for suffix in (".env.yaml", ".requirements.txt", ".yml"):
                if fname.endswith(suffix):
                    stem = fname[: -len(suffix)]
                    slot = by_stem.setdefault(stem, {"env_name": stem,
                                                     "env_dir": path.rsplit("/", 1)[0]})
                    slot["cluster" if suffix == ".requirements.txt" else "serverless"] = path
                    break
        return [self._from_info(info) for info in by_stem.values()]

    def get(
        self,
        project: str = "ygg",
        *,
        version: Optional[str] = None,
        python: Optional[str] = None,
        workspace_dir: Optional[str] = None,
    ) -> Optional[Environment]:
        """The deployed base environment for *project* matching *version* /
        *python* — by default the ygg image for the local interpreter."""
        stem = (ygg_base_environment_name(python) if _norm(distribution_for(project)) == "ygg"
                else environment_stem(project, python=python, version=version))
        return next((e for e in self.list(workspace_dir=workspace_dir) if e.name == stem), None)

    # -- helpers -----------------------------------------------------------
    def base_environment_name(self, python: Optional[str] = None) -> str:
        """Canonical version-tagged stem of the ygg base environment."""
        return ygg_base_environment_name(python)

    def runtime_dependencies(self) -> list[str]:
        """The ygg image's runtime dependency requirements (index specs)."""
        return ygg_runtime_dependencies()
