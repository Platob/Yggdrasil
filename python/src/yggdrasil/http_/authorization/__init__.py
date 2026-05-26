"""HTTP authorization providers for ``yggdrasil.io`` sessions and requests."""

from __future__ import annotations

from .base import Authorization


__all__ = ["Authorization", "MSALAuth"]


def __getattr__(name: str):
    if name == "MSALAuth":
        from .msal import MSALAuth as value
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
