"""YGG Bot — FastAPI trading + AI backend.

Exports:
    create_app  FastAPI application factory
    BotSettings configuration (pydantic BaseSettings)
"""
from __future__ import annotations

from .app import create_app
from .config import BotSettings

__all__ = ["create_app", "BotSettings"]
