"""Legacy / launcher entry point.

``create_app(settings)`` is the back-compat alias for :func:`create_api`, and
``app`` is the module-level ASGI callable so ``uvicorn yggdrasil.node.app:app``
boots a node configured from the ``YGG_NODE_*`` environment (the shape the
subprocess-spawning benches use).
"""
from __future__ import annotations

from fastapi import FastAPI

from .api.app import create_api
from .config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    return create_api(settings)


app = create_api()
