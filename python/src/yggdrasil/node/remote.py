"""``@remote`` registry for callable node functions.

A function decorated with ``@remote(name=...)`` is registered under its name
and becomes invokable over ``POST /api/call`` (pickle in, pickle/Arrow out).
The registry is a flat process-global dict; registration is idempotent on
name (re-decorating replaces the entry).
"""
from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., Any]] = {}


def remote(name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register *fn* under *name* (defaults to ``module:qualname``)."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        key = name or f"{fn.__module__}:{fn.__qualname__}"
        _REGISTRY[key] = fn
        fn._remote_name = key  # type: ignore[attr-defined]
        return fn

    return _decorator


def get_remote(name: str) -> Callable[..., Any]:
    fn = _REGISTRY.get(name)
    if fn is None:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"No remote function named {name!r}. Registered names: {known}. "
            f"Decorate a function with @remote(name={name!r}) to expose it."
        )
    return fn


def list_remotes() -> list[str]:
    return sorted(_REGISTRY)
