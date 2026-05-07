"""FastAPI dependencies — engine + settings accessors.

The app stashes its :class:`TabularEngine` and :class:`Settings` on
``app.state``; routers grab them via :func:`Depends`. Tests can swap
the engine on a per-app basis without touching the process-wide
:data:`yggdrasil.io.tabular.SYSTEM_ENGINE`.
"""

from __future__ import annotations

from fastapi import Request

from yggdrasil.io.tabular import TabularEngine

from .config import Settings


def get_settings(request: Request) -> Settings:
    """Return the :class:`Settings` bound on ``app.state``."""
    return request.app.state.settings


def get_engine(request: Request) -> TabularEngine:
    """Return the :class:`TabularEngine` bound on ``app.state``.

    Defaults to :data:`yggdrasil.io.tabular.SYSTEM_ENGINE` when the
    app is built without an explicit engine, so process-wide
    registrations are visible to the API immediately.
    """
    return request.app.state.engine
