"""Self-describing scratch storage for a node.

A node has two scratch roots directly under its home:

* ``tmp/`` — short-lived: SQL spill, download staging, Arrow overflow.
* ``stg/`` — staging: holds results other nodes wrote here; swept far less
  often because it backs in-flight remote work.

Every entry is named ``{prefix}-{start_ms}-{end_ms}-{suffix}`` where the two
epoch-millisecond stamps are UTC. The expiry is *in the name*, so the janitor
decides what to reclaim by parsing the filename — never a ``stat()`` per entry.
That keeps a sweep of a large scratch dir to one ``scandir`` and string math.
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

# prefix - start_ms - end_ms - suffix(rest)
_NAME_RE = re.compile(r"^(tmp|stg)-(\d+)-(\d+)-(.*)$")


def now_ms() -> int:
    return int(time.time() * 1000)


def make_name(prefix: str, *, ttl_seconds: float, suffix: str = "") -> str:
    """``{prefix}-{start_ms}-{end_ms}-{suffix}`` with UTC epoch-ms stamps."""
    start = now_ms()
    end = start + int(max(0.0, ttl_seconds) * 1000)
    safe = suffix.strip().lstrip(".") or "x"
    return f"{prefix}-{start}-{end}-{safe}"


def new_path(root: Path, prefix: str, *, ttl_seconds: float, suffix: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / make_name(prefix, ttl_seconds=ttl_seconds, suffix=suffix)


def expiry_ms(name: str) -> int | None:
    """The encoded end_ms, or ``None`` if the name isn't self-describing."""
    m = _NAME_RE.match(name)
    return int(m.group(2 + 1)) if m else None


def sweep(root: Path, *, now: int | None = None, fallback_ttl_seconds: float | None = None) -> int:
    """Delete expired entries under *root*. Returns the count removed.

    Self-describing entries expire when ``now > end_ms``. Foreign entries
    (no encoded expiry) are only touched when *fallback_ttl_seconds* is given,
    in which case their mtime decides — the one place we pay a ``stat()``.
    """
    if not root.exists():
        return 0
    now = now if now is not None else now_ms()
    removed = 0
    import os

    with os.scandir(root) as it:
        for de in it:
            end = expiry_ms(de.name)
            if end is not None:
                if now <= end:
                    continue
            elif fallback_ttl_seconds is not None:
                try:
                    if de.stat().st_mtime * 1000 + fallback_ttl_seconds * 1000 > now:
                        continue
                except OSError:
                    continue
            else:
                continue  # foreign entry, no fallback → leave it
            try:
                if de.is_dir(follow_symlinks=False):
                    shutil.rmtree(de.path, ignore_errors=True)
                else:
                    Path(de.path).unlink(missing_ok=True)
                removed += 1
            except OSError:
                continue
    return removed
