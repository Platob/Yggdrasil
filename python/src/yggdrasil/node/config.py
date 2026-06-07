"""Node server settings.

Plain pydantic ``BaseModel`` so a caller can construct a fully-explicit
``Settings`` in code, or let ``create_api(None)`` fall back to the
defaults below. ``node_home`` is where the analysis/fs services look for
parquet/arrow files; it is created lazily by the services that write to it.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

__all__ = ["Settings"]


class Settings(BaseModel):
    node_id: str = "ygg"
    node_home: Path = Path.home() / ".ygg" / "node"
    front_home: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    cors_origins: list[str] = ["*"]
