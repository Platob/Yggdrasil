"""Loki session workspaces — isolated, self-purging scratch under ``~/.loki``.

Each interactive (or driven) Loki run gets its own directory tree::

    ~/.loki/session/<id>/
        workspace/   # the agent's default root — files it writes land here
        memory/      # synthesized context + transcript (see loki.memory)
        cache/       # session-scoped caches

Isolating IO here keeps a session's files together, out of the user's cwd,
and safe to wipe. To avoid an unbounded pile of old sessions, :meth:`start`
auto-purges on creation — dropping sessions older than :data:`MAX_AGE_DAYS`
and keeping at most :data:`KEEP` of the most recent.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["LokiSession", "BASE", "KEEP", "MAX_AGE_DAYS"]

#: Root for every session tree.
BASE = Path.home() / ".loki" / "session"
#: Keep at most this many recent sessions; older ones are purged.
KEEP = 20
#: Purge sessions whose directory is older than this many days.
MAX_AGE_DAYS = 14


@dataclass(frozen=True)
class LokiSession:
    """One isolated session directory tree."""

    id: str
    dir: Path

    @property
    def workspace(self) -> Path:
        return self.dir / "workspace"

    @property
    def memory_dir(self) -> Path:
        return self.dir / "memory"

    @property
    def cache_dir(self) -> Path:
        return self.dir / "cache"

    @property
    def memory_file(self) -> Path:
        return self.memory_dir / "memory.json"

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def start(cls, *, purge: bool = True) -> "LokiSession":
        """Create a fresh session tree (and purge stale ones first)."""
        if purge:
            cls.purge()
        sid = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
        session = cls(id=sid, dir=BASE / sid)
        for sub in (session.workspace, session.memory_dir, session.cache_dir):
            sub.mkdir(parents=True, exist_ok=True)
        return session

    @classmethod
    def list(cls) -> "list[Path]":
        """Existing session directories, newest first."""
        if not BASE.is_dir():
            return []
        dirs = [p for p in BASE.iterdir() if p.is_dir()]
        return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)

    @classmethod
    def purge(
        cls,
        *,
        keep: int = KEEP,
        max_age_days: float = MAX_AGE_DAYS,
        exclude: "Path | None" = None,
    ) -> "list[str]":
        """Delete stale session trees; return the ids removed.

        A session is purged when it's older than *max_age_days* OR falls
        outside the *keep* most-recent. The *exclude* directory (the live
        session) is never touched.
        """
        sessions = cls.list()
        cutoff = time.time() - max_age_days * 86400
        removed: list[str] = []
        for i, path in enumerate(sessions):
            if exclude is not None and path == exclude:
                continue
            too_old = path.stat().st_mtime < cutoff
            too_many = i >= keep
            if too_old or too_many:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path.name)
        return removed

    def remove(self) -> None:
        """Delete this session's tree."""
        shutil.rmtree(self.dir, ignore_errors=True)
