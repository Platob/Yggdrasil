from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, TypeVar, overload

LOGGER = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_REGISTRY: dict[str, Callable[..., Any]] = {}


def _func_key(func: Callable) -> str:
    module = getattr(func, "__module__", None) or ""
    qualname = getattr(func, "__qualname__", None) or func.__name__
    return f"{module}:{qualname}"


@overload
def remote(func: F) -> F: ...


@overload
def remote(*, name: str | None = ..., timeout: float | None = ...) -> Callable[[F], F]: ...


def remote(
    func: F | None = None,
    *,
    name: str | None = None,
    timeout: float | None = None,
) -> F | Callable[[F], F]:
    """Register a function for remote execution via a bot server.

    Usage::

        @remote
        def compute(x: int, y: int) -> int:
            return x + y

        # With options
        @remote(timeout=30)
        def slow_compute(data: list) -> dict:
            ...

    The decorated function works normally when called locally.
    Use ``BotClient.call(func, *args, **kwargs)`` to invoke it
    on a remote bot node.
    """
    def _wrap(f: F) -> F:
        key = name or _func_key(f)
        _REGISTRY[key] = f

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)

        wrapper._remote_key = key  # type: ignore[attr-defined]
        wrapper._remote_timeout = timeout  # type: ignore[attr-defined]
        wrapper._remote_func = f  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if func is not None:
        return _wrap(func)
    return _wrap


def get_registered(key: str) -> Callable[..., Any] | None:
    return _REGISTRY.get(key)


def list_registered() -> dict[str, str]:
    result = {}
    for key, func in _REGISTRY.items():
        sig = inspect.signature(func)
        result[key] = str(sig)
    return result
