"""Lightweight cookie jar used by :class:`HTTPSession` browser-style methods."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Mapping, Optional

if TYPE_CHECKING:
    from .response import HTTPResponse

__all__ = ["Cookies"]


class Cookies:
    """Mapping-like cookie jar with ``Cookie`` / ``Set-Cookie`` plumbing.

    Stored as ``name -> value`` pairs without attribute parsing
    (``Path``, ``Expires``, ``HttpOnly``, ``Secure``, ``Domain`` are ignored).
    That matches the rest of the browser-mode helpers, which intentionally
    stay close to a "browser-as-a-string-bag" model rather than reimplementing
    :mod:`http.cookiejar` semantics.
    """

    __slots__ = ("_jar",)

    def __init__(self, initial: Optional[Mapping[str, str]] = None) -> None:
        self._jar: dict[str, str] = dict(initial) if initial else {}

    # ── Mapping protocol ──────────────────────────────────────────────────

    def __getitem__(self, name: str) -> str:
        return self._jar[name]

    def __setitem__(self, name: str, value: str) -> None:
        self._jar[name] = value

    def __delitem__(self, name: str) -> None:
        del self._jar[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._jar)

    def __len__(self) -> int:
        return len(self._jar)

    def __contains__(self, name: object) -> bool:
        return name in self._jar

    def __bool__(self) -> bool:
        return bool(self._jar)

    def __repr__(self) -> str:
        return f"Cookies({self._jar!r})"

    # ── Convenience accessors ─────────────────────────────────────────────

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self._jar.get(name, default)

    def set(self, name: str, value: str) -> None:
        self._jar[name] = value

    def delete(self, name: str) -> None:
        self._jar.pop(name, None)

    def clear(self) -> None:
        self._jar.clear()

    def update(self, other: Mapping[str, str]) -> None:
        self._jar.update(other)

    def items(self):
        return self._jar.items()

    def keys(self):
        return self._jar.keys()

    def values(self):
        return self._jar.values()

    def as_dict(self) -> dict[str, str]:
        return dict(self._jar)

    # ── HTTP serialization ────────────────────────────────────────────────

    def to_header(self) -> str:
        """Serialize the jar into a ``Cookie`` request-header value.

        Returns an empty string when the jar is empty so callers can
        ``if header:`` skip emitting the header entirely.
        """
        return "; ".join(f"{k}={v}" for k, v in self._jar.items())

    @staticmethod
    def parse_set_cookie(header: str) -> tuple[str, str]:
        """Parse a single ``Set-Cookie`` header value into ``(name, value)``.

        Attribute pairs after the first ``;`` are dropped on purpose — see
        the class docstring.
        """
        name_value, _, _ = header.partition(";")
        name_value = name_value.strip()
        if "=" in name_value:
            name, _, value = name_value.partition("=")
            return name.strip(), value.strip()
        return name_value, ""

    def update_from_set_cookie(self, raw: str) -> None:
        """Merge a raw ``Set-Cookie`` header value into the jar.

        ``urllib3`` collapses multiple ``Set-Cookie`` values with a comma,
        so we split and best-effort parse each piece.
        """
        if not raw:
            return
        for piece in raw.split(","):
            try:
                name, value = self.parse_set_cookie(piece.strip())
                if name:
                    self._jar[name] = value
            except Exception:
                continue

    def update_from_response(self, response: "HTTPResponse") -> None:
        """Pull cookies out of *response*'s ``Set-Cookie`` header(s)."""
        raw = (
            response.headers.get("Set-Cookie")
            or response.headers.get("set-cookie")
        )
        if raw:
            self.update_from_set_cookie(raw)
