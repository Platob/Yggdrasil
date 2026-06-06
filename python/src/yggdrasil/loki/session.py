"""Loki session workspaces — per-user, named, resumable, self-purging.

Sessions live under the current user's tree::

    ~/.loki/users/<user>/session/<id>/
        meta.json    # id, name, user, first_prompt, created_at, last_used_at
        workspace/   # the agent's default root — files it writes land here
        memory/      # synthesized context + transcript (loki.memory)
        cache/       # session-scoped caches

The user comes from :meth:`UserInfo.current`, so each user only sees and
resumes their own sessions. A session is **named from its first prompt** and
tracks **last usage**: every turn (and a resume) touches ``last_used_at``,
which resets its purge clock — :meth:`start` drops sessions only when they go
stale (older than :data:`MAX_AGE_DAYS` since last use) or fall outside the
:data:`KEEP` most-recent, so active work is never collected.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = ["LokiSession", "BASE", "KEEP", "MAX_AGE_DAYS"]

#: Root for every user's session trees.
BASE = Path.home() / ".loki"
#: Keep at most this many recent sessions per user; older ones are purged.
KEEP = 20
#: Purge sessions untouched for longer than this many days.
MAX_AGE_DAYS = 14


def _slug(text: str, *, maxlen: int = 48) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "-", (text or "").strip().lower()).strip("-")
    return s[:maxlen] or "session"


def _current_user() -> str:
    try:
        from yggdrasil.environ.userinfo import UserInfo

        u = UserInfo.current()
        return _slug(u.key or u.email or u.hostname or "default")
    except Exception:
        return "default"


@dataclass
class LokiSession:
    """One isolated, named, per-user session directory tree."""

    id: str
    dir: Path
    user: str
    name: str = ""
    first_prompt: str = ""
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)

    # -- paths -------------------------------------------------------------

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

    @property
    def meta_file(self) -> Path:
        return self.dir / "meta.json"

    @property
    def label(self) -> str:
        return f"{self.id}{' · ' + self.name if self.name else ''}"

    # -- registry ----------------------------------------------------------

    @classmethod
    def user_base(cls, user: Optional[str] = None) -> Path:
        return BASE / "users" / (user or _current_user()) / "session"

    @classmethod
    def start(cls, *, user: Optional[str] = None, purge: bool = True) -> "LokiSession":
        """Create a fresh session for the user (purging stale ones first)."""
        user = user or _current_user()
        if purge:
            cls.purge(user=user)
        sid = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
        now = time.time()
        s = cls(id=sid, dir=cls.user_base(user) / sid, user=user,
                created_at=now, last_used_at=now)
        for sub in (s.workspace, s.memory_dir, s.cache_dir):
            sub.mkdir(parents=True, exist_ok=True)
        s._save_meta()
        return s

    @classmethod
    def list(cls, *, user: Optional[str] = None) -> "list[LokiSession]":
        """The user's sessions, most-recently-used first."""
        base = cls.user_base(user)
        if not base.is_dir():
            return []
        out: list[LokiSession] = []
        for p in base.iterdir():
            if p.is_dir():
                s = cls._load(p)
                if s is not None:
                    out.append(s)
        return sorted(out, key=lambda s: s.last_used_at, reverse=True)

    @classmethod
    def resume(cls, id: str, *, user: Optional[str] = None) -> "Optional[LokiSession]":
        """Reopen an existing session by id and reset its purge clock."""
        d = cls.user_base(user) / id
        s = cls._load(d) if d.is_dir() else None
        if s is not None:
            s.touch()
        return s

    @classmethod
    def latest(cls, *, user: Optional[str] = None) -> "Optional[LokiSession]":
        sessions = cls.list(user=user)
        return sessions[0] if sessions else None

    @classmethod
    def purge(cls, *, user: Optional[str] = None, keep: int = KEEP,
              max_age_days: float = MAX_AGE_DAYS,
              exclude: "Optional[Path]" = None) -> "list[str]":
        """Delete the user's stale session trees; return the ids removed.

        Stale = untouched for > *max_age_days* (by ``last_used_at``) OR outside
        the *keep* most-recently-used. *exclude* (the live session) is spared.
        """
        sessions = cls.list(user=user)
        cutoff = time.time() - max_age_days * 86400
        removed: list[str] = []
        for i, s in enumerate(sessions):
            if exclude is not None and s.dir == exclude:
                continue
            if s.last_used_at < cutoff or i >= keep:
                shutil.rmtree(s.dir, ignore_errors=True)
                removed.append(s.id)
        return removed

    # -- mutation ----------------------------------------------------------

    def touch(self) -> None:
        """Mark the session used now — resets its purge clock."""
        self.last_used_at = time.time()
        self._save_meta()
        try:
            os.utime(self.dir, None)
        except OSError:
            pass

    def name_from_prompt(self, prompt: str) -> str:
        """Name the session from its first user prompt (slug); persist + return."""
        self.first_prompt = prompt.strip()
        self.name = _slug(prompt, maxlen=40)
        self._save_meta()
        return self.name

    def remove(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "user": self.user, "name": self.name,
            "first_prompt": self.first_prompt,
            "created_at": self.created_at, "last_used_at": self.last_used_at,
        }

    def _save_meta(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_file.write_text(json.dumps(self.to_dict(), indent=2), "utf-8")

    @classmethod
    def _load(cls, dir: Path) -> "Optional[LokiSession]":
        meta = dir / "meta.json"
        try:
            data = json.loads(meta.read_text("utf-8")) if meta.exists() else {}
        except (OSError, json.JSONDecodeError):
            data = {}
        mtime = dir.stat().st_mtime
        return cls(
            id=data.get("id", dir.name),
            dir=dir,
            user=data.get("user", dir.parent.parent.name),
            name=data.get("name", ""),
            first_prompt=data.get("first_prompt", ""),
            created_at=data.get("created_at", mtime),
            last_used_at=data.get("last_used_at", mtime),
        )
