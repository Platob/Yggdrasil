"""Remote-callable function registry behind ``/api/call``.

A function decorated with :func:`remote` is registered under a name; the
node's ``/api/call`` endpoint looks it up, runs it with the caller's
``args``/``kwargs``, and ships the return value through the transport
layer. Names default to the function's qualified name.
"""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}

__all__ = ["remote", "get", "list_all"]


def remote(name: str | None = None) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name or fn.__qualname__] = fn
        return fn

    return decorator


def get(name: str) -> Callable | None:
    return _REGISTRY.get(name)


def list_all() -> list[str]:
    return list(_REGISTRY)
