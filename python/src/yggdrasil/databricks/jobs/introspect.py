"""
Static introspection helpers for staged Python job tasks.

Used by :meth:`JobTask.from_callable` to derive task-level defaults
from the function's source instead of asking the user to spell them
out twice:

- :func:`sniff_imports` — walks the script AST and returns the
  top-level module names the function actually imports. Drives the
  auto-dependency path on :class:`JobTask`.
- :func:`sniff_env_vars` — walks the AST for ``os.getenv("X")`` /
  ``os.environ["X"]`` / ``os.environ.get("X")`` reads so the staged
  task can surface its required env vars on the parent job (run as
  ``spark_env_vars`` or as a UI hint in the description).
- :func:`resolve_module_dependency` — maps a top-level module name to
  its :mod:`importlib.metadata` distribution and classifies it
  (``pypi`` / ``editable`` / ``local`` / ``stdlib`` / ``unknown``).
- :func:`dependencies_to_pip_specs` — composes the two: takes a set
  of import names, resolves each, and renders the right pip
  requirement string (``name==version`` for public PyPI installs,
  direct workspace URLs for local/editable wheels when a
  :class:`~yggdrasil.databricks.jobs.workspace_pypi.WorkspacePyPI`
  publisher is supplied).
"""
from __future__ import annotations

import ast
import dataclasses as dc
import importlib.metadata as ilm
import json
import logging
import sys
from typing import Iterable, Optional, Sequence, TYPE_CHECKING

from yggdrasil.environ.modules import (
    module_name_to_project_name,
    packages_distributions_cached,
)

if TYPE_CHECKING:
    from .workspace_pypi import WorkspacePyPI


__all__ = [
    "ModuleDependency",
    "sniff_imports",
    "sniff_env_vars",
    "sniff_script",
    "resolve_module_dependency",
    "dependencies_to_pip_specs",
]

LOGGER = logging.getLogger(__name__)


#: PyPI distributions that aren't expected to be reachable from a
#: Databricks runtime — typically pre-installed by the runtime itself
#: (PySpark / dbutils stubs / Databricks SDK) or stdlib aliases.
#: Pulled out of the auto-dep set so the env spec stays clean.
DEFAULT_EXCLUDED_MODULES: frozenset[str] = frozenset({
    "pyspark", "databricks", "dbutils", "mlflow",
    "yggdrasil",
})


@dc.dataclass(frozen=True)
class ModuleDependency:
    """Resolved provenance for a top-level Python import name.

    ``kind`` summarizes where the package came from:

    - ``"stdlib"`` — part of the Python standard library; never
      shipped as a dependency.
    - ``"pypi"`` — installed normally; reproducible from the public
      index as ``project==version``.
    - ``"editable"`` — installed with ``pip install -e .`` (carries a
      ``direct_url.json`` with ``dir_info.editable == true``); pin to
      the local source so we can wheel-and-upload it.
    - ``"local"`` — installed from a local path / VCS URL; same
      treatment as ``editable``.
    - ``"unknown"`` — module isn't installed (e.g. namespace-only or
      typo); rendered as a bare name so pip can surface the failure
      with a real message instead of us guessing.
    """

    module: str
    project: Optional[str]
    version: Optional[str]
    kind: str
    source_path: Optional[str] = None


def sniff_script(source: str) -> tuple[set[str], set[str]]:
    """Return ``(top_level_imports, env_var_names)`` from one AST walk.

    Combined walk so callers that need both (notably
    :meth:`JobTask.from_callable`) parse the script exactly once
    instead of twice. The semantics match :func:`sniff_imports` +
    :func:`sniff_env_vars`: parse errors return ``(set(), set())``,
    relative imports are skipped, and only string-literal env-var
    keys are picked up.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        LOGGER.debug("sniff_script: source did not parse; returning empty sets")
        return set(), set()

    imports: set[str] = set()
    env_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top:
                    imports.add(top)
        elif isinstance(node, ast.ImportFrom):
            if not node.level and node.module:
                top = node.module.split(".", 1)[0]
                if top:
                    imports.add(top)
        elif isinstance(node, ast.Call):
            name = _extract_env_var_from_call(node)
            if name:
                env_vars.add(name)
        elif isinstance(node, ast.Subscript):
            name = _extract_env_var_from_subscript(node)
            if name:
                env_vars.add(name)
    return imports, env_vars


def sniff_imports(source: str) -> set[str]:
    """Return the set of top-level modules imported by *source*.

    Walks the AST once and collapses ``import foo.bar`` /
    ``from foo.bar.baz import qux`` to ``foo`` — pip resolves on the
    distribution that owns the top-level package, so any deeper
    suffix is noise. Relative imports (``from .helpers import x``)
    are skipped: they resolve inside the same source tree and don't
    add a wheel-level dependency.

    Returns an empty set when *source* fails to parse — the staged
    script still runs unaltered, the auto-dep path just stays
    conservative.
    """
    return sniff_script(source)[0]


def sniff_env_vars(source: str) -> set[str]:
    """Return env-var names referenced as string literals in *source*.

    Detects the three idiomatic shapes and only those — anything
    dynamic (``os.getenv(f"{prefix}_KEY")``) is intentionally skipped
    because we can't resolve it statically:

    - ``os.getenv("NAME"[, default])``
    - ``os.environ["NAME"]`` (Subscript with constant string)
    - ``os.environ.get("NAME"[, default])``
    """
    return sniff_script(source)[1]


def _extract_env_var_from_call(node: ast.Call) -> Optional[str]:
    func = node.func
    # os.getenv("NAME") / os.environ.get("NAME")
    if isinstance(func, ast.Attribute) and node.args:
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            return None
        # os.getenv("X")
        if (
            func.attr == "getenv"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        ):
            return first.value
        # os.environ.get("X")
        if (
            func.attr == "get"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "environ"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "os"
        ):
            return first.value
    return None


def _extract_env_var_from_subscript(node: ast.Subscript) -> Optional[str]:
    # os.environ["NAME"]
    value, slc = node.value, node.slice
    if not (
        isinstance(value, ast.Attribute)
        and value.attr == "environ"
        and isinstance(value.value, ast.Name)
        and value.value.id == "os"
    ):
        return None
    if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
        return slc.value
    return None


def resolve_module_dependency(module: str) -> ModuleDependency:
    """Classify *module* by where its installed distribution came from.

    Order of resolution mirrors what ``pip`` itself would do:

    1. Stdlib (``sys.stdlib_module_names``, 3.10+) — short-circuit
       with ``kind="stdlib"``.
    2. ``importlib.metadata.packages_distributions()`` lookup — if
       it has no entry, the top-level isn't owned by an installed
       wheel (namespace pkg, vendored, etc.) and we emit
       ``kind="unknown"``.
    3. Pull the matching :class:`Distribution`, peek at
       ``direct_url.json`` — present + ``editable=true`` ⇒
       ``editable``; present otherwise ⇒ ``local``; absent ⇒
       ``pypi`` (the standard, reproducible case).
    """
    top = module.split(".", 1)[0]

    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if stdlib_names is not None and top in stdlib_names:
        return ModuleDependency(
            module=top, project=None, version=None, kind="stdlib",
        )

    dist_names = packages_distributions_cached().get(top)
    if not dist_names:
        return ModuleDependency(
            module=top, project=None, version=None, kind="unknown",
        )

    dist_name = dist_names[0]
    try:
        dist = ilm.distribution(dist_name)
    except ilm.PackageNotFoundError:
        return ModuleDependency(
            module=top, project=dist_name, version=None, kind="unknown",
        )

    version = dist.version
    direct_url_raw = None
    try:
        direct_url_raw = dist.read_text("direct_url.json")
    except (FileNotFoundError, OSError):
        direct_url_raw = None

    if direct_url_raw:
        try:
            payload = json.loads(direct_url_raw)
        except json.JSONDecodeError:
            payload = {}
        dir_info = payload.get("dir_info") or {}
        url = payload.get("url") or ""
        is_editable = bool(dir_info.get("editable"))
        if is_editable:
            return ModuleDependency(
                module=top, project=dist_name, version=version,
                kind="editable",
                source_path=_url_to_local_path(url),
            )
        if url.startswith("file://"):
            return ModuleDependency(
                module=top, project=dist_name, version=version,
                kind="local", source_path=_url_to_local_path(url),
            )
        # Non-file VCS / direct-URL install: treat as PyPI-reachable
        # only when we have a version; otherwise tag as local so the
        # publisher tries to upload from the local checkout.
        return ModuleDependency(
            module=top, project=dist_name, version=version,
            kind="pypi" if version else "local",
        )

    return ModuleDependency(
        module=top, project=dist_name, version=version, kind="pypi",
    )


def _url_to_local_path(url: str) -> Optional[str]:
    """Decode a ``file://`` URL into a local filesystem path.

    Uses stdlib :func:`urllib.parse.urlparse` for the percent-decoding
    + authority strip rather than rolling the shape by hand — pip's
    ``direct_url.json`` follows RFC 8089 strictly, so the standard
    parser gets the right answer for all of ``file:/path``,
    ``file:///path`` and ``file://host/path`` (host dropped — we only
    care about the local-filesystem case).
    """
    if not url.startswith("file:"):
        return None
    from urllib.parse import unquote, urlparse
    parsed = urlparse(url)
    path = unquote(parsed.path)
    return path or None


def dependencies_to_pip_specs(
    modules: Iterable[str],
    *,
    exclude: Sequence[str] = (),
    workspace_pypi: Optional["WorkspacePyPI"] = None,
    pin_pypi: bool = True,
) -> list[str]:
    """Render a set of imported module names as pip requirement strings.

    Each module is resolved through :func:`resolve_module_dependency`
    and rendered according to its kind:

    - ``stdlib`` / excluded — dropped silently.
    - ``pypi`` — ``"project==version"`` when ``pin_pypi`` and we have
      a version; bare ``"project"`` otherwise. The aliases in
      :data:`yggdrasil.environ.modules.MODULE_PROJECT_NAMES_ALIASES`
      cover module-vs-distribution name mismatches (e.g. ``yggdrasil``
      → ``ygg``, ``jwt`` → ``PyJWT``).
    - ``editable`` / ``local`` — when *workspace_pypi* is supplied,
      build a wheel from the local source and upload it; the rendered
      spec is a PEP 440 direct reference
      (``project @ /Workspace/.../wheel.whl``) that pip can install
      from a serverless env or cluster library. Without a publisher
      we fall back to ``project==version`` and let the user wire up
      the wheel themselves (the surface stays informative).
    - ``unknown`` — rendered as a bare project name and warned about.

    Output order is deterministic (sorted by module name) so the
    serialized environment spec stays diffable.
    """
    excluded_set: set[str] = {m.lower() for m in exclude}
    excluded_set.update(m.lower() for m in DEFAULT_EXCLUDED_MODULES)

    specs: list[str] = []
    for module in sorted(set(modules)):
        if module.lower() in excluded_set:
            continue
        dep = resolve_module_dependency(module)
        if dep.kind == "stdlib":
            continue

        project = dep.project or module_name_to_project_name(module)
        project_norm = (project or module).lower()
        if project_norm in excluded_set:
            continue

        if dep.kind == "pypi":
            if pin_pypi and dep.version:
                specs.append(f"{project}=={dep.version}")
            else:
                specs.append(project)
            continue

        if dep.kind in ("editable", "local"):
            if workspace_pypi is not None:
                try:
                    published = workspace_pypi.publish(
                        module, source_path=dep.source_path,
                        version=dep.version,
                    )
                    # ``PathPyPI.publish`` returns a :class:`Path`;
                    # older shims may hand back a raw string. Accept
                    # either, project through ``full_path()`` when it
                    # exists so the requirement line carries the
                    # backend-native location (workspace path, s3 URL,
                    # …) pip can resolve.
                    location = (
                        published.full_path()
                        if hasattr(published, "full_path")
                        else str(published)
                    )
                    specs.append(f"{project} @ {location}")
                    continue
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "Failed to publish %r to workspace PyPI %r (%s); "
                        "falling back to bare requirement",
                        module, workspace_pypi, exc,
                    )
            if dep.version:
                specs.append(f"{project}=={dep.version}")
            else:
                specs.append(project)
            continue

        # unknown
        LOGGER.warning(
            "Sniffed import %r is not installed locally; emitting bare "
            "requirement %r — pip will surface the resolution error.",
            module, project,
        )
        specs.append(project)

    return specs
