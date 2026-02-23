from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # API
    api_v1_prefix: str = "/api/v1"

    # Frontend proxy – upstream URL that /* requests are forwarded to
    frontend_upstream: str = "http://localhost:5173"

    # Optional: strip/add a path prefix when forwarding to the upstream
    frontend_strip_prefix: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
