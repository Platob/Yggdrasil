"""Remote function registry behind ``POST /api/call``.

A function decorated with :func:`remote` is callable over the wire by name.
The registry is process-global — the node serves whatever was registered at
import time plus anything registered at runtime by a caller.
"""
from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., Any]] = {}


def remote(name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register ``fn`` for remote dispatch. ``name`` defaults to the qualname."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        key = name or fn.__qualname__
        _REGISTRY[key] = fn
        return fn
    return decorator


def call(func_name: str, args: Any, kwargs: Any) -> Any:
    fn = _REGISTRY.get(func_name)
    if fn is None:
        available = sorted(_REGISTRY)
        hint = f" Registered: {available}." if available else " No functions registered yet."
        raise KeyError(f"No remote function {func_name!r}.{hint}")
    return fn(*(args or ()), **(kwargs or {}))
