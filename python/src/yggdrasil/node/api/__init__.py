"""Node API package — v2 surface, schemas, and services."""
from __future__ import annotations

__all__ = ["create_api"]


def create_api():
    from .app import create_api as _create_api

    return _create_api()
