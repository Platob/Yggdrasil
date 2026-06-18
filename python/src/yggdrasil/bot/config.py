"""Bot service configuration — driven by env vars with YGG_BOT_ prefix."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YGG_BOT_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Market data
    market_cache_ttl: int = 300         # seconds; ENTSOE day-ahead prices don't change often
    fx_cache_ttl: int = 60              # FX rates refresh faster
    entsoe_token: str | None = None     # ENTSOE_API_TOKEN env var also checked by loki.entsoe

    # AI / Loki
    loki_engine: str | None = None      # pin a specific engine; None = auto-select
    ai_max_steps: int = 8

    # WebSocket
    ws_tick_interval: float = 5.0       # seconds between broadcast ticks
