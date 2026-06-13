"""yggdrasil.node — the local YGG FastAPI node.

Serves filesystem browsing, tabular inspection, lazy analysis (aggregate /
series / OHLC / pivot / forecast / finance), chat, function management, system
monitoring, and a pickle/Arrow remote-call transport. Runs as ``ygg node serve``.
"""
from __future__ import annotations

from .config import Settings

__all__ = ["Settings", "create_app", "create_api", "remote"]


def create_app(settings: Settings | None = None):
    from .app import create_app as _create_app

    return _create_app(settings)


def create_api():
    from .api.app import create_api as _create_api

    return _create_api()


def remote(name: str):
    from .remote import remote as _remote

    return _remote(name)
