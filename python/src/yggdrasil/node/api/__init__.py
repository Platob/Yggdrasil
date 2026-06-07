"""Node API package — schemas, services, v2 routers, and the app factory.

The ``create_api`` factory lives in :mod:`yggdrasil.node.app`; import it
from there (or from ``yggdrasil.node``). This package stays import-light so
``node.app`` can pull in ``node.api.schemas`` without a cycle.
"""
from __future__ import annotations

__all__ = ["create_api"]


def __getattr__(name: str):
    if name == "create_api":
        from .app import create_api

        return create_api
    raise AttributeError(f"module 'yggdrasil.node.api' has no attribute {name!r}")
