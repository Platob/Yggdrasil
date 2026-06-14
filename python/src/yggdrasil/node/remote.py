"""Remote dispatch registry — functions exposed over ``POST /api/call``.

Decorate a function with ``@remote("name")`` and it becomes callable from a
peer node: the caller sends ``{"func": name, "args": ..., "kwargs": ...}`` as
yggdrasil pickle and gets the result back in the transport format that fits it.
"""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def remote(name: str) -> Callable[[Callable], Callable]:
    """Register *fn* under *name* for ``/api/call`` dispatch."""

    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return decorator
