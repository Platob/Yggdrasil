"""yggdrasil.node — the FastAPI bot backend.

A node serves a filesystem-rooted workspace, tabular inspection + analysis,
an in-memory messenger, function registry, and market data (FX + energy).
``create_app`` builds the full app (REST + ``/api/call`` remote dispatch);
``create_api`` builds just the REST surface.
"""
from __future__ import annotations

from yggdrasil.node.config import Settings
from yggdrasil.node.remote import remote

__all__ = ["Settings", "create_app", "create_api", "remote"]


def create_app(settings: Settings | None = None):
    from yggdrasil.node.app import create_app as _create_app

    return _create_app(settings)


def create_api(settings: Settings | None = None):
    from yggdrasil.node.api.app import create_api as _create_api

    return _create_api(settings)
