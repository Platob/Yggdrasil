"""Settings for ``yggdrasil.node``.

A plain pydantic ``BaseModel`` (``pydantic-settings`` is not a runtime dep);
``from_env`` reads ``YGG_NODE_*`` environment variables explicitly.
"""
from __future__ import annotations

import os

from pydantic import BaseModel

__all__ = ["Settings"]

_ENV_PREFIX = "YGG_NODE_"


def _env_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Node server settings.

    Populated from constructor kwargs or, via :meth:`from_env`, from
    ``YGG_NODE_*`` environment variables.
    """

    host: str = "0.0.0.0"
    port: int = 8765
    allow_remote: bool = False
    api_key: str | None = None  # if set, require matching X-API-Key header
    market_cache_ttl: int = 60  # seconds
    cors_origins: list[str] = ["*"]

    @classmethod
    def from_env(cls) -> "Settings":
        """Build :class:`Settings` from ``YGG_NODE_*`` environment variables."""
        values: dict[str, object] = {}

        if (v := os.getenv(f"{_ENV_PREFIX}HOST")) is not None:
            values["host"] = v
        if (v := os.getenv(f"{_ENV_PREFIX}PORT")) is not None:
            values["port"] = int(v)
        if (v := os.getenv(f"{_ENV_PREFIX}ALLOW_REMOTE")) is not None:
            values["allow_remote"] = _env_bool(v)
        if (v := os.getenv(f"{_ENV_PREFIX}API_KEY")) is not None:
            values["api_key"] = v
        if (v := os.getenv(f"{_ENV_PREFIX}MARKET_CACHE_TTL")) is not None:
            values["market_cache_ttl"] = int(v)
        if (v := os.getenv(f"{_ENV_PREFIX}CORS_ORIGINS")) is not None:
            values["cors_origins"] = [o.strip() for o in v.split(",") if o.strip()]

        return cls(**values)
