# userinfo.py
"""
User identity + runtime context probe.

Goal:
- One place to ask: "who am I and where am I running?"
- Cross-platform best-effort:
  - OS-level identity always attempted
  - Windows: try UPN via `whoami /UPN`
  - Unix: use `whoami` (or fall back to $USER)

Caching:
- Module-level cache avoids repeated subprocess calls.
"""

from __future__ import annotations

import os
import socket
import subprocess
from dataclasses import dataclass
from typing import Sequence

__all__ = ["UserInfo", "get_user_info"]

_CURRENT_CACHE: "UserInfo | None" = None


@dataclass(frozen=True, slots=True)
class UserInfo:
    email: str | None
    sam: str
    hostname: str

    @classmethod
    def current(cls, *, refresh: bool = False) -> "UserInfo":
        global _CURRENT_CACHE
        if _CURRENT_CACHE is not None and not refresh:
            return _CURRENT_CACHE

        hostname = socket.gethostname()
        sam = _get_username()
        email = _get_upn_email()

        _CURRENT_CACHE = cls(email=email, sam=sam, hostname=hostname)
        return _CURRENT_CACHE


def get_user_info(*, refresh: bool = False) -> UserInfo:
    """Convenience wrapper + cache control."""
    return UserInfo.current(refresh=refresh)


def _clear_cache() -> None:
    """Internal helper (mainly for tests)."""
    global _CURRENT_CACHE
    _CURRENT_CACHE = None


def _get_username() -> str:
    name = _run_quiet(["whoami"])
    if name:
        return name.strip()

    return (
        os.getenv("USERNAME")
        or os.getenv("USER")
        or os.getenv("LOGNAME")
        or "UNKNOWN"
    )


def _get_upn_email() -> str | None:
    upn = _run_quiet(["whoami", "/UPN"])
    if not upn:
        return None

    upn = upn.strip()
    if not upn or upn.lower() == "null":
        return None

    return upn if "@" in upn else None


def _run_quiet(cmd: Sequence[str]) -> str | None:
    try:
        out = subprocess.check_output(list(cmd), text=True, stderr=subprocess.DEVNULL)
        out = out.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None
