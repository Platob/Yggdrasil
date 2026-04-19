"""Ensure pyarrow can locate a tzdata database on Windows.

pyarrow ships without a bundled IANA timezone database. On Windows it looks
at ``%PYARROW_TZDATA_PATH%`` (or ``%USERPROFILE%\\Downloads\\tzdata`` if
unset) and fails with ``ArrowInvalid: Unable to get Timezone database
version`` when that directory is missing, empty, or stale. This module
repairs that state at import time and exposes a manual re-run entry point.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from .lib import pyarrow

__all__ = ["ensure_tzdata"]

_log = logging.getLogger("yggdrasil")
_ENSURED: bool | None = None


def _default_tzdata_path() -> Path:
    """Return the path pyarrow consults for tzdata on Windows."""
    env = os.environ.get("PYARROW_TZDATA_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Downloads" / "tzdata"


def _canary() -> bool:
    """Exercise the tzdata path the real error hits.

    ``pa.array(..., type=timestamp(tz=...))`` only tags metadata — it doesn't
    touch the tz database. ``compute.assume_timezone`` actually resolves the
    zone, which is the call that raises
    ``ArrowInvalid: Unable to get Timezone database version`` on Windows
    when tzdata is missing.
    """
    try:
        import pyarrow.compute as pc

        naive = pyarrow.array([0], type=pyarrow.timestamp("ns"))
        pc.assume_timezone(naive, "UTC")
        return True
    except Exception:
        return False


def _looks_partial(path: Path) -> bool:
    """A tzdata directory without a ``version`` file is the exact failure mode
    from the reported error — arrow can't read the DB version and bails."""
    return path.is_dir() and not (path / "version").is_file()


def _install_tzdata_package() -> bool:
    """Best-effort install of the ``tzdata`` PyPI package.

    Does not repair pyarrow's Windows tzdata directory (pyarrow wants IANA
    source files; the PyPI package ships pre-compiled zoneinfo). It does
    keep stdlib ``zoneinfo`` working for the rest of yggdrasil, so it's a
    useful secondary safety net when pyarrow's downloader is blocked.
    """
    try:
        from yggdrasil.environ import PyEnv

        PyEnv.runtime_import_module(
            module_name="tzdata", pip_name="tzdata", install=True, warn=False,
        )
        return True
    except Exception as exc:
        _log.debug("Could not import or install 'tzdata' package: %s", exc)
        return False


def ensure_tzdata(*, force: bool = False) -> bool:
    """Make pyarrow's tzdata lookup succeed on Windows.

    Returns ``True`` when a timezone-aware timestamp cast works at the end of
    the call, ``False`` otherwise. No-op (returns ``True``) on non-Windows
    platforms and on repeat calls once a positive result has been cached.

    Set ``force=True`` to bypass the cache and rerun the probe + repair.
    """
    global _ENSURED

    if os.name != "nt":
        _ENSURED = True
        return True

    if _ENSURED and not force:
        return True

    if _canary():
        _ENSURED = True
        return True

    path = _default_tzdata_path()

    if _looks_partial(path):
        _log.warning(
            "Removing partial tzdata directory at %s (no 'version' file). "
            "pyarrow will re-download.",
            path,
        )
        shutil.rmtree(path, ignore_errors=True)

    try:
        pyarrow.util.download_tzdata_on_windows()
    except Exception as exc:
        _log.warning(
            "pyarrow.util.download_tzdata_on_windows() failed: %s. "
            "Timezone-aware operations may raise ArrowInvalid. "
            "Set PYARROW_TZDATA_PATH to a populated IANA tzdata directory, "
            "or run pyarrow.util.download_tzdata_on_windows() manually.",
            exc,
        )
        _install_tzdata_package()
        _ENSURED = False
        return False

    if _canary():
        _ENSURED = True
        return True

    _log.warning(
        "pyarrow tzdata download completed but the canary cast still fails. "
        "Inspected path: %s. Set PYARROW_TZDATA_PATH to a valid IANA tzdata "
        "directory, or reinstall pyarrow.",
        path,
    )
    _install_tzdata_package()
    _ENSURED = False
    return False
