"""API app factory — the v2 surface lives in the same FastAPI app."""
from __future__ import annotations

from fastapi import FastAPI

from ..app import create_app


def create_api() -> FastAPI:
    return create_app()
