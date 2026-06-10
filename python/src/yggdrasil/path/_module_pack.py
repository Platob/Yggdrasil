"""Module → ``.zip`` archive packaging shared by :class:`Path` callers.

A single helper, :func:`build_module_archive`, mirrors what
``CommandExecution._zip_local_module`` has been doing for cluster
command uploads: walk a local package directory, drop the standard
junk (``__pycache__``, ``.dist-info``, ``.egg-info``, ``.DS_Store``),
and emit a deflated ``.zip`` whose top-level entry IS the package
directory (so the archive can be added to ``sys.path`` directly).

Lives under :mod:`yggdrasil.io.path` rather than the legacy
``databricks.compute`` so the abstract :class:`Path` upload/import
hooks — and the Spark Connect builder — can reuse it without
pulling the cluster-command code in.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path as _LocalPath
from typing import Callable, Iterable, Union

__all__ = [
    "ModuleSpec",
    "resolve_module_root",
    "build_module_archive",
]


ModuleSpec = Union[str, os.PathLike, Callable]
"""Anything :func:`resolve_module_root` accepts.

- ``str`` — importable module name (``"yggdrasil.io"``) OR a
  filesystem path string. Path-shaped strings (containing
  separators / pointing at a real file or directory) are taken as
  paths; the rest are resolved as module names.
- :class:`os.PathLike` — concrete filesystem path.
- callable — uses ``obj.__module__``.
"""


#: Path components that get skipped on archive emit. Mirrors the
#: long-standing convention from
#: :meth:`CommandExecution._zip_local_module` so callers see the
#: same archive shape they'd get from cluster-command uploads.
_SKIP_DIR_SUFFIXES = (".dist-info", ".egg-info")
_SKIP_DIR_NAMES = frozenset({"__pycache__"})
_SKIP_FILE_NAMES = frozenset({".DS_Store"})


def resolve_module_root(obj: ModuleSpec) -> _LocalPath:
    """Resolve *obj* to a concrete local path (file or directory).

    Strings that are valid filesystem paths win: ``"./mypkg"`` and
    ``"/abs/path/to/pkg"`` resolve as-is. Anything else is treated
    as an importable module name and routed through
    :meth:`PyEnv.get_root_module_directory`.

    Raises :class:`FileNotFoundError` when the resolved path does
    not exist on disk, and :class:`ValueError` when *obj* has no
    usable module / path information.
    """
    if isinstance(obj, os.PathLike):
        path = _LocalPath(os.fspath(obj))
        if not path.exists():
            raise FileNotFoundError(
                f"upload_module: local path {str(path)!r} does not exist. "
                f"Pass an existing package directory, archive file, or "
                f"importable module name."
            )
        return path.resolve()

    if isinstance(obj, str):
        # Path-shaped strings (separator present OR resolves to a
        # real fs entry) are taken as paths. Bare names like
        # ``"yggdrasil"`` route through the module resolver.
        if os.sep in obj or "/" in obj or obj.startswith("."):
            path = _LocalPath(obj)
            if path.exists():
                return path.resolve()
        module_name = obj
    else:
        module_name = getattr(obj, "__module__", None)
        if not module_name:
            raise ValueError(
                f"upload_module: cannot resolve {obj!r} to a module — "
                f"expected a module name string, an os.PathLike, or a "
                f"callable with __module__."
            )

    from yggdrasil.environ import PyEnv

    root = module_name.split(".", 1)[0]
    resolved = _LocalPath(PyEnv.get_root_module_directory(module_name=root)).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"upload_module: resolved module root for {root!r} does not "
            f"exist: {resolved}"
        )
    return resolved


def _iter_module_files(local_root: _LocalPath) -> Iterable[tuple[_LocalPath, str]]:
    """Yield ``(absolute_path, arcname)`` pairs for files to archive.

    Directories surface their files with paths *relative to the
    package's parent* — so ``mypkg/__init__.py`` lands as
    ``mypkg/__init__.py`` inside the zip, and adding the archive
    to ``sys.path`` exposes ``mypkg`` directly.
    """
    if local_root.is_file():
        yield local_root, local_root.name
        return

    parent = local_root.parent
    for path in local_root.rglob("*"):
        if path.is_dir():
            continue
        parts = path.parts
        if path.name in _SKIP_FILE_NAMES:
            continue
        if any(seg in _SKIP_DIR_NAMES for seg in parts):
            continue
        if any(seg.endswith(_SKIP_DIR_SUFFIXES) for seg in parts):
            continue
        yield path, str(path.relative_to(parent))


def build_module_archive(
    obj: ModuleSpec,
    dest: Union[str, os.PathLike, None] = None,
) -> _LocalPath:
    """Zip a local module / package into a deflated ``.zip`` archive.

    Resolves *obj* via :func:`resolve_module_root`, then walks the
    directory (skipping ``__pycache__`` / ``.dist-info`` /
    ``.egg-info`` / ``.DS_Store``) and emits a deflated zip. When
    the resolved root is already a single file (e.g. an existing
    ``.whl`` or ``.zip``) the file is copied verbatim.

    *dest* picks the output location:

    - ``None`` — write to a fresh
      ``LocalPath.staging_path()`` so the file gets unlinked when
      the caller closes its holder.
    - directory — write ``<dir>/<module>.zip`` inside it.
    - file ending in ``.zip`` / ``.whl`` — write there exactly.

    Returns the local :class:`pathlib.Path` of the produced
    archive.
    """
    local_root = resolve_module_root(obj)

    if local_root.is_file() and local_root.suffix.lower() in (".zip", ".whl"):
        if dest is None:
            return local_root
        out = _coerce_dest(dest, default_name=local_root.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(local_root.read_bytes())
        return out

    archive_name = f"{local_root.name}.zip"
    if dest is None:
        from yggdrasil.path.local_path import LocalPath
        # Staging file is closed (no fd) — fine to write via stdlib
        # zipfile.
        staging = LocalPath.staging_path()
        out = _LocalPath(staging.os_path).with_suffix(".zip")
    else:
        out = _coerce_dest(dest, default_name=archive_name)

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in _iter_module_files(local_root):
            zf.write(src, arcname)
    return out


def _coerce_dest(dest: Union[str, os.PathLike], default_name: str) -> _LocalPath:
    """Resolve *dest* into a concrete output file path.

    A directory becomes ``<dest>/<default_name>``; everything else
    is taken as the exact destination file path.
    """
    out = _LocalPath(os.fspath(dest))
    if out.exists() and out.is_dir():
        return out / default_name
    # Trailing slash is also "directory-shaped" — handle even when
    # the dir doesn't exist yet.
    raw = os.fspath(dest)
    if raw.endswith(("/", os.sep)):
        return out / default_name
    return out
