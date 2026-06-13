"""Remote function registry.

Decorate a function with ``@remote("name")`` to expose it through the node's
``/api/call`` endpoint. The registry is process-global so it can be populated
at import time from anywhere in the codebase.
"""
from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable] = {}


def remote(name: str) -> Callable[[Callable], Callable]:
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_registry() -> dict[str, Callable]:
    return _REGISTRY


def call_remote(name: str, args: tuple, kwargs: dict) -> Any:
    fn = _REGISTRY.get(name)
    if fn is None:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"no remote function named {name!r}. Registered functions: {known}. "
            f"Decorate one with @remote({name!r}) and import it before serving."
        )
    return fn(*args, **(kwargs or {}))
