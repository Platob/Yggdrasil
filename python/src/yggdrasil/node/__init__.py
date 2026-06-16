"""yggdrasil.node — FastAPI backend for the YGG bot/node service.

The node is a small, dependency-light FastAPI service exposing the bot
surface (messenger, python functions, monitor) plus a v2 analysis API
(filesystem reads, audit log, trading-focused parquet analytics).

Two apps live here:

- :func:`yggdrasil.node.app.create_app` — the bot app (`/api/messenger`,
  `/api/function`, `/api/monitor`, `/api/hello`, `/api/ping`).
- :func:`yggdrasil.node.api.app.create_api` — the v2 analysis API.
"""
from __future__ import annotations

from .config import Settings

__all__ = ["Settings"]
