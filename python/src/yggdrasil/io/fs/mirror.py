"""Local mirror for remote :class:`Path` instances.

Some Path subclasses (S3Path, DBFSPath, VolumePath, WorkspacePath, …)
reach across the network on every read. When the same object is
read repeatedly — Parquet footer probes, DeltaIO replay, repeated
config loads, hot dataframe reloads — the round-trips compound.

:meth:`Path.local_mirror` returns a :class:`LocalPath` that mirrors
the remote bytes under ``~/.yggdrasil/mirror/<scheme>/<host>/<key>``.
A small JSON sidecar (``.<name>.ygmirror.json``, written next to
the mirror file) records the remote :class:`PathStats` summary
``(size, mtime, kind)`` from the last download.

Refresh model
-------------

Two layers, cheapest first:

1. **In-process verdict cache.** A module-level :class:`ExpiringDict`
   keyed on :meth:`Path.full_path` records the ``(size, mtime)`` we
   most recently validated. While the entry is live (default 60s),
   :meth:`local_mirror` returns the on-disk mirror without any
   remote round-trip. The TTL is per-call tunable.
2. **Sidecar comparison.** On a verdict miss, we issue exactly one
   remote ``stat()`` and compare ``size`` + ``mtime`` to the sidecar.
   Match → bump the in-process verdict, return the mirror as-is.
   Mismatch (or no sidecar) → re-download, write the sidecar, bump
   the verdict.

For local paths this is the identity — the mirror IS the source.

Crash safety
------------

The mirror is written via ``LocalPath.write_bytes``; the sidecar
is written after the mirror lands. A torn sidecar at worst forces
one extra download next call. The mirror layout never escapes its
root — scheme/host/path segments are sanitized for path-safety.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Optional, Tuple

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.io.path_stat import PathKind, PathStats

if TYPE_CHECKING:
    from yggdrasil.io.fs.local_path import LocalPath
    from yggdrasil.io.fs.path import Path


__all__ = [
    "default_mirror_root",
    "ensure_local_mirror",
    "invalidate_mirror",
    "mirror_path_for",
    "sweep_mirror_root",
]


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-global freshness verdict cache
# ---------------------------------------------------------------------------

#: Default TTL for the in-process freshness verdict (seconds). Long
#: enough to collapse hot loops (parquet footer + body, repeated
#: config loads), short enough that an external writer's update is
#: picked up in under a minute by default.
_MIRROR_FRESH_TTL_S: float = 60.0

#: Cap on the verdict cache so traversing an enormous remote tree
#: doesn't pin unbounded state.
_MIRROR_FRESH_MAX_KEYS: int = 4096

#: Key: remote ``full_path()``. Value: the ``(size, mtime)`` tuple
#: we validated against. The presence of a live entry is the
#: "fresh" signal; the value lets diagnostic tooling observe what
#: was last seen without a stat round-trip.
_MIRROR_FRESH: ExpiringDict[str, Tuple[int, float]] = ExpiringDict(
    default_ttl=_MIRROR_FRESH_TTL_S,
    max_size=_MIRROR_FRESH_MAX_KEYS,
)


_SIDECAR_SUFFIX: str = ".ygmirror.json"


# ---------------------------------------------------------------------------
# Rate-limited "sweep old mirror files" — same pattern as the staging
# sweep in :mod:`yggdrasil.io.fs.path`. Fires at most once per root per
# :data:`_MIRROR_SWEEP_INTERVAL_S`, so a hot caller never pays the
# ``ls(recursive=True)`` cost on the mirror tree more than once per day.
# ---------------------------------------------------------------------------

#: Rate-limit interval for mirror-tree sweeps (seconds). One sweep per
#: process per day per root by default, matching the staging sweep's
#: "fire once and forget" cadence.
_MIRROR_SWEEP_INTERVAL_S: float = 24 * 60 * 60.0

#: Cap on tracked roots. Exotic configurations using many distinct
#: mirror roots stay bounded.
_MIRROR_SWEEP_MAX_KEYS: int = 64

#: Default max age of a mirror file before the sweep deletes it.
#: Seven days — long enough for repeat-read workflows, short enough
#: that an abandoned mirror tree doesn't grow without bound.
_MIRROR_MAX_AGE_S: float = 7 * 24 * 60 * 60.0

_MIRROR_SWEPT: ExpiringDict[str, bool] = ExpiringDict(
    default_ttl=_MIRROR_SWEEP_INTERVAL_S,
    max_size=_MIRROR_SWEEP_MAX_KEYS,
)


def sweep_mirror_root(
    root: "Optional[Path]" = None,
    *,
    max_age: float = _MIRROR_MAX_AGE_S,
    force: bool = False,
) -> bool:
    """Best-effort sweep of *root*: remove mirror files older than *max_age*.

    Rate-limited per-root via the module-level :data:`_MIRROR_SWEPT`
    :class:`ExpiringDict` — at most one sweep per
    :data:`_MIRROR_SWEEP_INTERVAL_S` per root per process. Mirrors
    the contract of :meth:`Path._sweep_expired_staging`: stamp
    before walking so concurrent callers converge to one sweep,
    and swallow per-file errors so one bad sibling doesn't abort
    the whole pass.

    Returns ``True`` if the sweep ran, ``False`` if rate-limited.
    """
    base = root if root is not None else default_mirror_root()
    from yggdrasil.io.fs.local_path import LocalPath
    if not isinstance(base, LocalPath):
        base = LocalPath.from_(base)

    key = base.full_path()
    if not force and key in _MIRROR_SWEPT:
        return False

    # Stamp BEFORE the walk so concurrent callers see "swept" and
    # skip; even a partial walk shouldn't cost a second pass.
    _MIRROR_SWEPT[key] = True

    if not base.exists():
        return True

    cutoff = time.time() - max_age
    try:
        for candidate in base.ls(recursive=True, allow_not_found=True):
            try:
                stats = candidate.stat()
                if stats.kind != PathKind.FILE:
                    continue
                if (stats.mtime or 0.0) >= cutoff:
                    continue
                candidate.remove(recursive=False, allow_not_found=True)
            except Exception:
                LOGGER.debug(
                    "Failed sweeping mirror entry %s",
                    candidate, exc_info=True,
                )
    except Exception:
        LOGGER.debug("Mirror sweep failed at %s", base, exc_info=True)
    return True


def reset_mirror_sweep_state(root: Optional[str] = None) -> None:
    """Test/maintenance hook: drop the rate-limit stamp for one root,
    or for every tracked root when ``root`` is ``None``."""
    if root is None:
        _MIRROR_SWEPT.clear()
    else:
        _MIRROR_SWEPT.pop(root, None)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def default_mirror_root() -> "LocalPath":
    """The default mirror root: ``~/.yggdrasil/mirror``.

    Built lazily so importing this module does not touch the
    filesystem.
    """
    from yggdrasil.io.fs.local_path import LocalPath
    return LocalPath.from_(os.path.expanduser("~/.yggdrasil/mirror"))


def _safe_segment(s: str) -> str:
    """Sanitize a scheme/host segment so the mirror path can't escape root."""
    if not s:
        return "_"
    # NUL is illegal on every supported FS; backslashes confuse Windows
    # path math when nested inside a forward-slash URL key.
    return s.replace("\x00", "_").replace("\\", "_")


def mirror_path_for(
    path: "Path",
    *,
    root: "Optional[Path]" = None,
) -> "Path":
    """Map *path* to its canonical local mirror location.

    Pure mapping — no I/O, no directory creation. Layout:
    ``<root>/<scheme>/<host>/<url-path>``. For a local *path* this
    is the identity (the mirror IS the source).
    """
    from yggdrasil.io.fs.local_path import LocalPath

    if path.is_local:
        return path

    base = root if root is not None else default_mirror_root()
    if not isinstance(base, LocalPath):
        base = LocalPath.from_(base)

    scheme = _safe_segment(path.url.scheme or "")
    host = _safe_segment(path.url.host or "")
    rel = (path.url.path or "/").lstrip("/")
    if not rel:
        # Root of a remote namespace mirrors to a marker dir; the
        # caller almost certainly doesn't want this for reads, but
        # it keeps the mapping total.
        rel = "_root_"
    return base / scheme / host / rel


def _sidecar_for(local: "LocalPath") -> "LocalPath":
    """Sidecar JSON path for a mirror file."""
    return local.parent / f".{local.name}{_SIDECAR_SUFFIX}"


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------

def _read_sidecar(sidecar: "LocalPath") -> Optional[PathStats]:
    """Read a sidecar JSON. Returns ``None`` on any error or empty file."""
    try:
        raw = sidecar.read_bytes(raise_error=False)
    except Exception:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    try:
        kind_raw = data.get("kind", "file")
        kind = PathKind(kind_raw) if isinstance(kind_raw, str) else PathKind.FILE
        return PathStats(
            size=int(data.get("size", 0)),
            mtime=float(data.get("mtime", 0.0)),
            kind=kind,
            mode=int(data.get("mode", 0)),
        )
    except Exception:
        return None


def _write_sidecar(sidecar: "LocalPath", stats: PathStats) -> None:
    """Write a sidecar JSON. Failures are logged but never raised —
    a missing sidecar just means the next call re-downloads."""
    payload = json.dumps(
        {
            "size": int(stats.size),
            "mtime": float(stats.mtime or 0.0),
            "kind": str(
                stats.kind.value
                if isinstance(stats.kind, PathKind)
                else stats.kind
            ),
            "mode": int(stats.mode or 0),
        }
    ).encode("utf-8")
    try:
        sidecar.write_bytes(payload, parents=True)
    except Exception as exc:
        LOGGER.debug("Failed writing mirror sidecar %s: %s", sidecar, exc)


def _stats_match(cached: PathStats, fresh: PathStats) -> bool:
    """A cached sidecar matches a fresh remote stat when size and
    mtime are identical and both are files."""
    if cached.kind != fresh.kind:
        return False
    if int(cached.size) != int(fresh.size):
        return False
    return float(cached.mtime or 0.0) == float(fresh.mtime or 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_local_mirror(
    path: "Path",
    *,
    ttl: float = _MIRROR_FRESH_TTL_S,
    force_refresh: bool = False,
    root: "Optional[Path]" = None,
    sweep: bool = True,
    max_age: float = _MIRROR_MAX_AGE_S,
) -> "Path":
    """Refresh and return the local mirror for *path*.

    Public surface: :meth:`Path.local_mirror`. See the module
    docstring for the freshness model.

    When ``sweep=True`` (default) the first call against a given
    mirror root in this process triggers a one-shot sweep that
    deletes mirror files older than *max_age* seconds. The sweep
    is rate-limited via :data:`_MIRROR_SWEPT` (default: one sweep
    per root per day per process) so hot callers never pay for
    the directory walk twice.
    """
    if path.is_local:
        return path

    local = mirror_path_for(path, root=root)
    cache_key = path.full_path()

    # Opportunistic, rate-limited sweep of stale mirror files. Runs
    # at most once per process per root per day; the staging-sweep
    # pattern guarantees concurrent callers converge to one sweep.
    if sweep:
        try:
            sweep_mirror_root(root=root, max_age=max_age)
        except Exception:
            LOGGER.debug(
                "Mirror sweep failed; continuing with fetch", exc_info=True,
            )

    # ── Layer 1: in-process verdict cache ────────────────────────────
    if not force_refresh and ttl > 0:
        cached_verdict = _MIRROR_FRESH.get(cache_key)
        if cached_verdict is not None and local.exists():
            return local

    # ── Layer 2: stat the remote, compare to sidecar ─────────────────
    try:
        remote_stats = path.stat()
    except Exception as exc:
        # Serve stale rather than fail outright when the remote
        # is briefly unreachable but we already have a copy.
        if local.exists():
            LOGGER.debug(
                "Remote stat failed for %s (%s); serving stale mirror %s.",
                cache_key, exc, local.full_path(),
            )
            return local
        raise

    if remote_stats.kind == PathKind.MISSING:
        raise FileNotFoundError(
            f"Cannot mirror missing remote path {cache_key!r}."
        )
    if remote_stats.kind != PathKind.FILE:
        raise IsADirectoryError(
            f"local_mirror is for files; {cache_key!r} is a "
            f"{remote_stats.kind.value}. Use ls()/walk() for trees."
        )

    sidecar = _sidecar_for(local)
    cached_summary = _read_sidecar(sidecar) if not force_refresh else None
    fresh_summary = (
        int(remote_stats.size),
        float(remote_stats.mtime or 0.0),
    )

    if (
        not force_refresh
        and local.exists()
        and cached_summary is not None
        and _stats_match(cached_summary, remote_stats)
    ):
        # Sidecar matches → mirror is still good. Bump the in-process
        # verdict so the next caller in the TTL window short-circuits
        # without even a stat round-trip.
        if ttl > 0:
            _MIRROR_FRESH.set(cache_key, fresh_summary, ttl=ttl)
        return local

    # Out of date (or never downloaded): pull through.
    local.parent.mkdir(parents=True, exist_ok=True)
    payload = path.read_bytes()
    local.write_bytes(payload, parents=True)
    _write_sidecar(sidecar, remote_stats)

    if ttl > 0:
        _MIRROR_FRESH.set(cache_key, fresh_summary, ttl=ttl)
    return local


def invalidate_mirror(path: "Path") -> None:
    """Drop the in-process freshness verdict for *path*.

    The next :meth:`Path.local_mirror` call will re-stat the remote
    and possibly re-download. Does NOT delete the on-disk mirror —
    use :meth:`Path.local_mirror(force_refresh=True)` to overwrite,
    or :meth:`Path.mirror_path` to obtain the local handle and
    :meth:`Path.unlink` it explicitly.
    """
    try:
        _MIRROR_FRESH.pop(path.full_path(), None)
    except Exception:
        pass
