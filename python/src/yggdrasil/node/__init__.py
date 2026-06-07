"""Yggdrasil node — a FastAPI trading backend.

Synthetic market data (candles/ticks/order books), an in-memory portfolio
book with live mark-to-market, polars-backed analytics over node-local
files, and an Arrow-IPC transport for tabular responses. Build the app with
:func:`create_api` and serve it with ``ygg node serve``.
"""
from __future__ import annotations

from .app import create_api
from .config import Settings

__all__ = ["create_api", "Settings"]
