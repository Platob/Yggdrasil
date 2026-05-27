"""Abstract base class for HTTP authorization providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


__all__ = ["Authorization"]


class Authorization(ABC):
    """Contract for objects that produce an HTTP ``Authorization`` header value.

    Concrete subclasses own credential storage, refresh logic, and any token
    cache. Callers only need :attr:`authorization` — the fully-formed header
    value (e.g. ``"Bearer <token>"``) ready to drop into
    :class:`yggdrasil.io.headers.Headers` or :class:`PreparedRequest.authorization`.
    """

    @property
    @abstractmethod
    def authorization(self) -> str:
        """Return the value for the HTTP ``Authorization`` header."""

    def __str__(self) -> str:
        return self.authorization
