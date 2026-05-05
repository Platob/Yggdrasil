"""Janitor for self-owned :class:`BytesIO` spill temp files.

Spill files mint by ``BytesIO._spill`` follow the
``tmp-<start>-<end>-<seed>.<ext>`` naming convention, with both
timestamps zero-padded to 12 digits so a lexical scan is also a
chronological scan. Crashed workers can leave their spill files
behind; this module sweeps them.

Two entry points:

- :func:`cleanup_stale_spill_files` — direct, full scan-and-unlink.
- :func:`maybe_cleanup_stale_spill_files` — in-process throttled
  wrapper safe to call from hot paths.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from typing import Optional


__all__ = [
    "cleanup_stale_spill_files",
    "maybe_cleanup_stale_spill_files",
]


# ``tmp-{start_epoch:012d}-{end_epoch:012d}-{seed:16hex}.<ext>``
_SPILL_FILENAME_RE = re.compile(
    r"^tmp-(?P<start>\d+)-(?P<end>\d+)-[0-9a-f]+\.[^/\\\s]+$"
)

# In-process throttle so the cleanup probe can't dominate the hot
# path of a tight spill loop.
_CLEANUP_INTERVAL_S = 600.0
_last_cleanup_at: float = 0.0


def cleanup_stale_spill_files(
    directory: Optional[str] = None,
    *,
    now: Optional[float] = None,
    grace_seconds: float = 0.0,
) -> int:
    """Unlink expired spill temp files in *directory*.

    Files matching the ``tmp-<start>-<end>-<seed>.<ext>`` pattern
    are inspected; those whose encoded ``end`` epoch second is below
    ``now - grace_seconds`` are unlinked. Errors on individual files
    are swallowed (another process may have just deleted the file,
    or the file may be in use on Windows). Returns the count of
    files actually unlinked.
    """
    if directory is None:
        directory = tempfile.gettempdir()
    if now is None:
        now = time.time()

    try:
        entries = os.listdir(directory)
    except OSError:
        return 0

    threshold = now - grace_seconds
    removed = 0
    for name in entries:
        m = _SPILL_FILENAME_RE.match(name)
        if m is None:
            continue
        try:
            end_epoch = int(m.group("end"))
        except ValueError:
            continue
        if end_epoch > threshold:
            continue
        full = os.path.join(directory, name)
        try:
            os.unlink(full)
            removed += 1
        except OSError:
            continue
    return removed


def maybe_cleanup_stale_spill_files(
    directory: Optional[str] = None,
    *,
    interval_s: Optional[float] = None,
) -> int:
    """Throttled :func:`cleanup_stale_spill_files`.

    Safe to call from hot paths: bounded by an in-process timer so
    repeat calls within ``interval_s`` (default 10 minutes) return
    immediately.
    """
    global _last_cleanup_at
    period = _CLEANUP_INTERVAL_S if interval_s is None else float(interval_s)
    monotonic_now = time.monotonic()
    if monotonic_now - _last_cleanup_at < period:
        return 0
    _last_cleanup_at = monotonic_now
    try:
        return cleanup_stale_spill_files(directory)
    except Exception:
        return 0
