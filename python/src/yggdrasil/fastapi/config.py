from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_cache_root() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(base) / "yggdrasil" / "cache"
    return Path.home() / ".cache" / "yggdrasil"


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "yggdrasil"
    app_version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 8000
    allow_remote: bool = False
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    api_prefix: str = "/api"
    python_prefix: str = "/python"
    system_prefix: str = "/system"
    excel_prefix: str = "/excel"
    env_home: Path = field(default_factory=lambda: Path.home() / ".local" / "yggdrasil" / "python" / "envs")
    cache_home: Path = field(default_factory=_default_cache_root)
    excel_app_version: str = "excel-exec-v3"

    @property
    def local_clients(self) -> set[str]:
        return {"127.0.0.1", "::1", "localhost"}

    @property
    def excel_env_root(self) -> Path:
        return self.cache_home / "excel" / "envs"

    @property
    def excel_run_root(self) -> Path:
        return self.cache_home / "excel" / "runs"


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("YGG_FASTAPI_APP_NAME", "yggdrasil"),
        app_version=os.getenv("YGG_FASTAPI_APP_VERSION", "0.1.0"),
        host=os.getenv("YGG_FASTAPI_HOST", "127.0.0.1"),
        port=int(os.getenv("YGG_FASTAPI_PORT", "8000")),
        allow_remote=_as_bool(os.getenv("YGG_FASTAPI_ALLOW_REMOTE"), False),
        docs_url=os.getenv("YGG_FASTAPI_DOCS_URL", "/docs"),
        redoc_url=os.getenv("YGG_FASTAPI_REDOC_URL", "/redoc"),
        openapi_url=os.getenv("YGG_FASTAPI_OPENAPI_URL", "/openapi.json"),
        api_prefix=os.getenv("YGG_FASTAPI_API_PREFIX", "/api"),
        python_prefix=os.getenv("YGG_FASTAPI_PYTHON_PREFIX", "/python"),
        system_prefix=os.getenv("YGG_FASTAPI_SYSTEM_PREFIX", "/system"),
        excel_prefix=os.getenv("YGG_FASTAPI_EXCEL_PREFIX", "/excel"),
        env_home=Path(
            os.getenv(
                "YGG_FASTAPI_ENV_HOME",
                str(Path.home() / ".local" / "yggdrasil" / "python" / "envs"),
            )
        ).expanduser().resolve(),
        cache_home=Path(
            os.getenv(
                "YGG_FASTAPI_CACHE_HOME",
                str(_default_cache_root()),
            )
        ).expanduser().resolve(),
        excel_app_version=os.getenv("YGG_FASTAPI_EXCEL_APP_VERSION", "excel-exec-v3"),
    )
