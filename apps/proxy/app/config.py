from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Server
    host: str = "0.0.0.0"
    # Databricks Apps injects DATABRICKS_APP_PORT at runtime; fall back to PORT
    # or the hardcoded default when running locally.
    port: int = Field(
        default=8000,
        validation_alias=AliasChoices("DATABRICKS_APP_PORT", "PORT", "port"),
    )
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
