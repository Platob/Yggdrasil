"""``ProxyPathMixin`` — be a :class:`Path` by delegating to an inner one.

A mixin for objects that *are* a filesystem path conceptually but aren't
constructed like one — their real backing :class:`Path` is built lazily from
other state (credentials, a fetched URL, a remote handle). Implement the single
abstract :meth:`_internal_path` hook and the whole :class:`Path` surface
(``ls`` / ``read_bytes`` / ``write_bytes`` / ``stat`` / ``exists`` / ``parent`` /
``iterdir`` / ``/`` / ``open`` / context-manager / iteration / …) is mirrored
straight to that inner path, so every call lands on the inner path's own
(possibly overridden) implementation::

    class ExternalLocation(DatabricksResource, ProxyPathMixin, Singleton):
        def _internal_path(self) -> Path:
            return self._build_credential_backed_s3_path()

    el.ls()              # → inner_path.ls()
    el / "sub/file.txt"  # → inner_path / "sub/file.txt"  (an inner-typed Path)
    el.read_bytes()      # → inner_path.read_bytes()

The mixin owns only delegation. Identity (``__repr__`` / ``__eq__`` /
``__hash__``) and any domain metadata are left to the consuming class — keep
this mixin *before* the identity-bearing base in the MRO only for the path
surface, never for identity. Navigation (``parent`` / ``joinpath`` / ``/``)
returns inner-typed paths: stepping off the proxy leaves the wrapper behind and
hands back the plain inner :class:`Path`, which is usually what you want.

Delegation works through :meth:`__getattr__` (so every non-dunder attribute the
consuming class doesn't define itself forwards to the inner path) plus explicit
mirrors for the protocol/operator dunders Python resolves on the *type* and
therefore never routes through ``__getattr__`` (``__truediv__``, ``__fspath__``,
``__iter__``, ``with`` support, …). A consuming class keeps full control: any
attribute it defines (a ``url`` / ``name`` / ``info`` metadata property) shadows
the delegated one, so domain state stays on the wrapper while I/O flows through.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.path import Path

__all__ = ["ProxyPathMixin"]


class ProxyPathMixin(ABC):
    """Mirror the full :class:`Path` surface onto an inner path.

    Implement :meth:`_internal_path`; everything else delegates. The
    inner path is resolved on every access (the implementation is free
    to cache it) so a refreshed / rebuilt inner path is picked up
    transparently.
    """

    @abstractmethod
    def _internal_path(self) -> "Path":
        """The backing :class:`Path` every filesystem op delegates to."""

    @property
    def inner_path(self) -> "Path":
        """The backing :class:`Path` (alias for :meth:`_internal_path`)."""
        return self._internal_path()

    # -- generic mirror -------------------------------------------------
    def __getattr__(self, item: str) -> Any:
        # Only fires for names the consuming class / mixin doesn't define
        # itself — forward the whole filesystem surface (ls / read_* /
        # write_* / stat / exists / parent / iterdir / size / open / …)
        # to the inner path. Never delegate private / dunder names: they'd
        # recurse while the inner path is still being built, and identity
        # dunders must stay with the consuming class.
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._internal_path(), item)

    # -- operator / protocol dunders (bypass __getattr__) ---------------
    # Python looks these up on the type, never the instance, so they must
    # be mirrored explicitly to reach the inner path's overrides.
    def __truediv__(self, other: Any) -> "Path":
        return self._internal_path() / other

    def __rtruediv__(self, other: Any) -> "Path":
        return other / self._internal_path()

    def __fspath__(self) -> str:
        return self._internal_path().__fspath__()

    def __iter__(self) -> Any:
        return iter(self._internal_path())

    def __len__(self) -> int:
        return len(self._internal_path())

    def __enter__(self) -> Any:
        return self._internal_path().__enter__()

    def __exit__(self, *exc: Any) -> Any:
        return self._internal_path().__exit__(*exc)

    def __str__(self) -> str:
        return str(self._internal_path())
