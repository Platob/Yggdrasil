"""``/api/v2`` route handlers — thin, service-backed, no business logic.

Each module owns one ``APIRouter`` already carrying its ``/v2/...`` prefix.
:func:`all_routers` returns them in the order the app should mount them;
``ping_router`` (the prefix-less liveness probe) is exported separately.
"""
from __future__ import annotations

from fastapi import APIRouter

from . import analysis, fs, health, market, portfolio

__all__ = ["all_routers", "ping_router"]

ping_router = health.ping_router


def all_routers() -> list[APIRouter]:
    return [
        health.router,
        market.router,
        portfolio.router,
        analysis.router,
        fs.router,
    ]
