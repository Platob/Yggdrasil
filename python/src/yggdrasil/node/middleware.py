"""Lightweight response caching for read-heavy endpoints.

Uses ExpiringDict to cache endpoint responses by function name and arguments,
avoiding repeated computation for frequently-called GET endpoints.

Only apply to read-only endpoints -- never cache mutations (POST/PUT/DELETE
that modify state).
"""
from __future__ import annotations

from functools import wraps
from typing import Callable

from yggdrasil.dataclasses.expiring import ExpiringDict

# Module-level response cache shared across all decorated endpoints.
_response_cache: ExpiringDict[str, object] = ExpiringDict(default_ttl=5_000_000_000)  # 5s default


def cached_response(ttl_seconds: float = 5.0) -> Callable:
    """Decorator for FastAPI endpoint functions. Caches response by function + kwargs.

    Parameters
    ----------
    ttl_seconds :
        Time-to-live for cached responses in seconds (converted to nanoseconds
        internally since ExpiringDict uses nanosecond TTLs).

    Notes
    -----
    - Only use on GET/read endpoints.
    - The cache key is built from the function name and the hashable subset of
      keyword arguments (Query params, path params injected by FastAPI).
    - Dependency-injected services are excluded from the cache key automatically
      (they are not hashable).
    """
    ttl_ns = int(ttl_seconds * 1_000_000_000)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key from function name + hashable kwargs only
            hashable_kwargs = {}
            for k, v in kwargs.items():
                try:
                    hash(v)
                    hashable_kwargs[k] = v
                except TypeError:
                    pass  # skip non-hashable args (services, Request objects, etc.)
            key = f"{func.__module__}:{func.__name__}:{hash(frozenset(hashable_kwargs.items()) if hashable_kwargs else ())}"
            cached = _response_cache.get(key)
            if cached is not None:
                return cached
            result = await func(*args, **kwargs)
            _response_cache.set(key, result, ttl=ttl_ns)
            return result
        return wrapper
    return decorator


def invalidate_response_cache() -> None:
    """Clear all cached responses. Call after mutations that affect cached data."""
    _response_cache.clear()
