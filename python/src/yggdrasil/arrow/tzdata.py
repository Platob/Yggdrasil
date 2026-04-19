"""Ensure pyarrow can locate a tzdata database on Windows.

pyarrow ships without a bundled IANA timezone database. On Windows it looks
at ``%PYARROW_TZDATA_PATH%`` (or ``%USERPROFILE%\\Downloads\\tzdata`` if
unset) and fails with ``ArrowInvalid: Unable to get Timezone database
version`` when that directory is missing, empty, or stale.

pyarrow's own ``pyarrow.util.download_tzdata_on_windows()`` has no retry,
no timeout, and no meaningful error — it silently leaves the target
directory in an unusable state when a corporate proxy blocks the IANA
download. This module repairs that state: it probes with
``pyarrow.compute.assume_timezone`` (the call that actually hits the tz
database), cleans up partial installs, runs pyarrow's downloader, and
falls back to a manual stdlib download with retries if that fails.
"""
from __future__ import annotations

import logging
import os
import shutil
import tarfile
import time
from pathlib import Path

from .lib import pyarrow

__all__ = ["ensure_tzdata"]

_log = logging.getLogger("yggdrasil")
_ENSURED: bool | None = None

_TZDATA_URL = "https://data.iana.org/time-zones/tzdata-latest.tar.gz"
_WINDOWS_ZONES_URL = (
    "https://raw.githubusercontent.com/unicode-org/cldr/master/"
    "common/supplemental/windowsZones.xml"
)


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


def _urlretrieve(url: str, out_path: Path, *, timeout: float, attempts: int) -> None:
    """Download ``url`` to ``out_path`` via stdlib urllib.

    Retries with exponential backoff on any exception. Honors standard
    proxy env vars (``HTTPS_PROXY``, ``HTTP_PROXY``) — urllib picks them
    up automatically. A User-Agent is set because some corporate
    proxies block the default python-urllib agent.
    """
    from urllib.request import Request, urlopen

    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = Request(url, headers={"User-Agent": "yggdrasil-tzdata"})
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) < 1024:
                raise IOError(
                    f"Downloaded only {len(data)} bytes from {url}; "
                    f"expected a real payload"
                )
            out_path.write_bytes(data)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    assert last_exc is not None
    raise last_exc


def _manual_download(path: Path) -> None:
    """Populate ``path`` with tzdata via a direct stdlib download.

    Used as a fallback when ``pyarrow.util.download_tzdata_on_windows``
    raises or leaves the directory unusable. Structure matches what
    pyarrow's own helper produces: ``tzdata.tar.gz`` plus
    ``windowsZones.xml`` at the root, with the tarball extracted in place.
    """
    path.mkdir(parents=True, exist_ok=True)
    tarball = path / "tzdata.tar.gz"
    windows_zones = path / "windowsZones.xml"

    _urlretrieve(_TZDATA_URL, tarball, timeout=30, attempts=3)
    _urlretrieve(_WINDOWS_ZONES_URL, windows_zones, timeout=30, attempts=3)

    with tarfile.open(tarball) as tf:
        tf.extractall(path)


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


def _manual_instructions(path: Path) -> str:
    return (
        f"pyarrow cannot read an IANA tzdata database at {path}. "
        f"To fix manually: download {_TZDATA_URL} and extract it to that "
        f"directory, and also drop {_WINDOWS_ZONES_URL} as "
        f"'windowsZones.xml' alongside the extracted files. "
        f"Alternatively set PYARROW_TZDATA_PATH to a directory that already "
        f"contains a valid tzdata source tree."
    )


def ensure_tzdata(*, force: bool = False, raise_on_failure: bool = False) -> bool:
    """Make pyarrow's tzdata lookup succeed on Windows.

    Returns ``True`` when a timezone-aware timestamp operation works at the
    end of the call, ``False`` otherwise. No-op (returns ``True``) on
    non-Windows platforms and on repeat calls once a positive result has
    been cached.

    Parameters
    ----------
    force:
        Bypass the cached positive result and rerun the probe + repair.
    raise_on_failure:
        Raise ``RuntimeError`` with manual-fix instructions when the final
        canary still fails. Useful at the top of scripts/tests that depend
        on timezone support.
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
            "Removing partial tzdata directory at %s (no 'version' file).",
            path,
        )
        shutil.rmtree(path, ignore_errors=True)

    primary_exc: BaseException | None = None
    try:
        pyarrow.util.download_tzdata_on_windows()
    except Exception as exc:
        primary_exc = exc
        _log.warning(
            "pyarrow.util.download_tzdata_on_windows() failed: %s. "
            "Falling back to manual download.",
            exc,
        )

    if _canary():
        _ENSURED = True
        return True

    if _looks_partial(path):
        shutil.rmtree(path, ignore_errors=True)

    try:
        _manual_download(path)
    except Exception as exc:
        _log.warning(
            "Manual tzdata download failed: %s (primary error: %s). %s",
            exc, primary_exc, _manual_instructions(path),
        )
        _install_tzdata_package()
        _ENSURED = False
        if raise_on_failure:
            raise RuntimeError(_manual_instructions(path)) from exc
        return False

    if _canary():
        _ENSURED = True
        return True

    _log.warning(
        "tzdata files are present at %s but pyarrow's canary still fails. %s",
        path, _manual_instructions(path),
    )
    _install_tzdata_package()
    _ENSURED = False
    if raise_on_failure:
        raise RuntimeError(_manual_instructions(path))
    return False
