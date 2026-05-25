from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, TypeVar, overload

LOGGER = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_REGISTRY: dict[str, _RemoteSpec] = {}


class _RemoteSpec:
    __slots__ = ("func", "key", "timeout", "modules")

    def __init__(
        self,
        func: Callable,
        key: str,
        timeout: float | None,
        modules: list[str] | None,
    ) -> None:
        self.func = func
        self.key = key
        self.timeout = timeout
        self.modules = modules or []


def _func_key(func: Callable) -> str:
    module = getattr(func, "__module__", None) or ""
    qualname = getattr(func, "__qualname__", None) or func.__name__
    return f"{module}:{qualname}"


@overload
def remote(func: F) -> F: ...


@overload
def remote(
    *,
    name: str | None = ...,
    timeout: float | None = ...,
    modules: list[str] | None = ...,
) -> Callable[[F], F]: ...


def remote(
    func: F | None = None,
    *,
    name: str | None = None,
    timeout: float | None = None,
    modules: list[str] | None = None,
) -> F | Callable[[F], F]:
    """Register a function for remote execution via a bot server.

    Usage::

        @remote
        def compute(x: int, y: int) -> int:
            return x + y

        @remote(timeout=30, modules=["numpy", "scipy"])
        def ml_predict(data: list) -> dict:
            import numpy as np
            ...

    ``modules`` lists pip packages the function needs. When called on a
    bot node that lacks them, the server auto-installs before execution.

    The decorated function works normally when called locally.
    Use ``BotClient.call(func, *args, **kwargs)`` to invoke it on a
    remote bot node.
    """
    def _wrap(f: F) -> F:
        key = name or _func_key(f)
        spec = _RemoteSpec(func=f, key=key, timeout=timeout, modules=modules)
        _REGISTRY[key] = spec

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)

        wrapper._remote_key = key  # type: ignore[attr-defined]
        wrapper._remote_timeout = timeout  # type: ignore[attr-defined]
        wrapper._remote_func = f  # type: ignore[attr-defined]
        wrapper._remote_modules = modules  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if func is not None:
        return _wrap(func)
    return _wrap


def get_registered(key: str) -> _RemoteSpec | None:
    return _REGISTRY.get(key)


def list_registered() -> dict[str, str]:
    result = {}
    for key, spec in _REGISTRY.items():
        sig = inspect.signature(spec.func)
        result[key] = str(sig)
    return result


def ensure_modules(spec: _RemoteSpec) -> None:
    """Auto-install any modules declared by the @remote spec."""
    if not spec.modules:
        return

    for mod_name in spec.modules:
        try:
            __import__(mod_name.replace("-", "_").split("[")[0])
        except ImportError:
            LOGGER.info("Auto-installing module %r for %r", mod_name, spec.key)
            try:
                from yggdrasil.environ.environment import PyEnv
                PyEnv.current().install(mod_name, wait=True, raise_error=True)
            except Exception:
                import subprocess, sys
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", mod_name],
                    check=True,
                    capture_output=True,
                )
            __import__(mod_name.replace("-", "_").split("[")[0])
            LOGGER.info("Installed module %r", mod_name)
