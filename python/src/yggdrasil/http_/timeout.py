"""Per-call timeout config — :class:`Timeout` + :func:`_resolve_timeout`.

``connect`` / ``read`` / ``total`` map onto ``http.client`` connect
and read deadlines. ``None`` means "no timeout"; a number means
"seconds". The full urllib3 DEFAULT-sentinel machinery is intentionally
not reproduced — every yggdrasil call site treats ``None`` and bare
numbers the same way.
"""
from __future__ import annotations

import socket
from typing import Any, Optional, Tuple


__all__ = ["Timeout", "_resolve_timeout"]


class Timeout:
    """Per-call timeout config — ``urllib3.Timeout`` shaped subset."""

    DEFAULT_TIMEOUT = socket._GLOBAL_DEFAULT_TIMEOUT  # type: ignore[attr-defined]

    def __init__(
        self,
        total: Optional[float] = None,
        connect: Optional[float] = None,
        read: Optional[float] = None,
    ) -> None:
        self.total = total
        self.connect = connect
        self.read = read

    @property
    def connect_timeout(self) -> Optional[float]:
        if self.connect is not None:
            return self.connect
        return self.total

    @property
    def read_timeout(self) -> Optional[float]:
        if self.read is not None:
            return self.read
        return self.total

    def __repr__(self) -> str:
        return f"Timeout(total={self.total!r}, connect={self.connect!r}, read={self.read!r})"


def _resolve_timeout(timeout: Any) -> Tuple[Optional[float], Optional[float]]:
    """Normalize a ``timeout`` argument to ``(connect, read)`` seconds."""
    if timeout is None:
        return None, None
    if isinstance(timeout, Timeout):
        return timeout.connect_timeout, timeout.read_timeout
    if isinstance(timeout, (int, float)):
        return float(timeout), float(timeout)
    if isinstance(timeout, tuple) and len(timeout) == 2:
        return timeout[0], timeout[1]
    return None, None
