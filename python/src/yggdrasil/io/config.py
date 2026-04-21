# yggdrasil/pyutils/dynamic_buffer/_config.py
"""Spill-to-disk buffer configuration."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Optional

__all__ = ["BufferConfig", "DEFAULT_CONFIG", "get_tmp_dir"]


_TMP_DIR: Path | None = None
_TMP_DIR_LOCK = RLock()
_TMP_DIR_CLEANED = False


def _safe_tmp_root() -> Path:
    try:
        path = Path(tempfile.gettempdir())
    except Exception:
        path = Path("~/tmp").expanduser()

    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Last-ditch fallback; if this also fails, let caller use the path anyway.
        path = Path(".").resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    return path


def _cleanup_expired_tmp_files(
    tmp_dir: Path, *, now: int, max_age_seconds: int = 24 * 3600
) -> None:
    """
    Best-effort cleanup of expired files named like: tmp-{start}-{end}.*
    Never raises.
    """
    try:
        for path in tmp_dir.glob("tmp-*.*"):
            try:
                if not path.is_file():
                    continue

                # Expect filename stem like: tmp-<start>-<end>
                parts = path.stem.split("-")
                if len(parts) != 3 or parts[0] != "tmp":
                    continue

                _start = int(parts[1])
                end = int(parts[2])

                # Expire by explicit end timestamp first; otherwise fall back to age.
                expired = end <= now or (now - end) > max_age_seconds
                if expired:
                    path.unlink(missing_ok=True)
            except Exception:
                # Swallow per-file issues. Temp cleanup should be invisible.
                continue
    except Exception:
        # Swallow directory scan issues too.
        pass


def get_tmp_dir() -> Path:
    """
    Return the process temp directory and lazily perform one cleanup pass
    per process for expired tmp-{start}-{end}.* files.

    Cleanup is best-effort and never raises.
    """
    global _TMP_DIR, _TMP_DIR_CLEANED

    if _TMP_DIR is not None and _TMP_DIR_CLEANED:
        return _TMP_DIR

    with _TMP_DIR_LOCK:
        if _TMP_DIR is None:
            _TMP_DIR = _safe_tmp_root()

        if not _TMP_DIR_CLEANED:
            _cleanup_expired_tmp_files(_TMP_DIR, now=int(time.time()))
            _TMP_DIR_CLEANED = True

        return _TMP_DIR


@dataclass(frozen=True, slots=True)
class BufferConfig:
    """Immutable configuration for spill-to-disk buffers."""

    spill_bytes: int = 128 * 1024 * 1024
    prefix: str = "tmp-"
    suffix: str = ".bin"
    keep_spilled_file: bool = False
    tmp_dir: Optional[Path | Any] = None

    @classmethod
    def default(cls) -> "BufferConfig":
        return DEFAULT_CONFIG

    def create_spill_path(self, lifetime_seconds: int | None = None):
        """Return a fresh spill path — a :class:`LocalPath` by default.

        When ``tmp_dir`` is set to any path-like object (``pathlib.Path``,
        ``str``, or a :class:`yggdrasil.io.fs.Path` subclass like a
        remote :class:`DatabricksPath`), the spill path is produced by
        joining against that object. That lets callers spill to DBFS /
        S3 / whatever, as long as the resulting :class:`Path` subclass
        implements :meth:`open`. Local-fs callers get zero-copy
        ``mmap`` / ``os.pread`` through :class:`LocalPath`.
        """
        from yggdrasil.io.fs import LocalPath, Path as FsPath

        tmp_dir = self.tmp_dir if self.tmp_dir is not None else get_tmp_dir()
        now = int(time.time())
        end = (
            now + int(lifetime_seconds)
            if lifetime_seconds is not None
            else now + 24 * 3600
        )
        name = f"{self.prefix}{now}-{end}-{os.urandom(8).hex()}{self.suffix}"

        # A yggdrasil Path composes natively via ``/``.
        if isinstance(tmp_dir, FsPath):
            return tmp_dir / name
        # Duck-typed path (custom remote types, test fakes, …) — anything
        # that already knows how to join ``/`` goes through untouched so
        # users can plug in their own backends without subclassing
        # :class:`Path`.
        if hasattr(tmp_dir, "__truediv__") and not isinstance(tmp_dir, (str, bytes)):
            return tmp_dir / name
        # str / pathlib.Path / os.PathLike → LocalPath so spilled files
        # get the zero-copy ``mmap`` / ``os.pread`` fast paths.
        return LocalPath.from_any(tmp_dir) / name


DEFAULT_CONFIG = BufferConfig(
    spill_bytes=128 * 1024 * 1024,
    prefix="tmp-",
    suffix=".bin",
    keep_spilled_file=False,
    tmp_dir=None,
)
