"""Remote function registry for ``yggdrasil.node``.

Decorate a function with :func:`remote` to expose it over ``/api/call``.
The registry is a flat ``name → callable`` dict; names default to the
function's ``__name__`` but can be namespaced (``"ns:fn"``) explicitly.
"""
from __future__ import annotations

from typing import Any, Callable

from yggdrasil.exceptions.api import NotFoundError

__all__ = [
    "remote",
    "get_registered",
    "list_registered",
    "call_registered",
]

_REGISTRY: dict[str, Callable] = {}


def remote(*, name: str | None = None) -> Callable:
    """Decorator registering a function in the remote-call registry.

    Usage::

        @remote()
        def add(x, y): ...          # registered as "add"

        @remote(name="math:add")
        def add(x, y): ...          # registered as "math:add"
    """
    def _decorator(fn: Callable) -> Callable:
        key = name or fn.__name__
        _REGISTRY[key] = fn
        return fn

    return _decorator


def get_registered(name: str) -> Callable | None:
    """Return the registered callable for *name*, or ``None``."""
    return _REGISTRY.get(name)


def list_registered() -> list[str]:
    """Sorted list of registered function names."""
    return sorted(_REGISTRY)


def call_registered(name: str, args: tuple, kwargs: dict) -> Any:
    """Invoke a registered function by *name*.

    Raises :class:`NotFoundError` with a near-match hint when *name* is not
    registered.
    """
    fn = _REGISTRY.get(name)
    if fn is None:
        import difflib

        suggestions = difflib.get_close_matches(name, list(_REGISTRY), n=3)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        registered = ", ".join(list_registered()) or "(none)"
        raise NotFoundError(
            f"No remote function registered as {name!r}.{hint} "
            f"Registered functions: {registered}."
        )
    return fn(*args, **(kwargs or {}))
