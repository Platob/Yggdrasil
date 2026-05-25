from __future__ import annotations

import os
import platform
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_root() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(base) / "yggdrasil" / "bot"
    return Path.home() / ".local" / "yggdrasil" / "bot"


def _default_node_id() -> str:
    return os.getenv("YGG_BOT_NODE_ID", f"{platform.node()}-{uuid.uuid4().hex[:8]}")


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "yggdrasil-bot"
    app_version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 8100
    allow_remote: bool = False
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    api_prefix: str = "/api"
    node_id: str = field(default_factory=_default_node_id)
    data_root: Path = field(default_factory=_default_data_root)
    max_cmd_timeout: float = 300.0
    max_python_timeout: float = 600.0
    max_concurrent_jobs: int = 16
    job_ttl: int = 3600
    job_max_history: int = 256

    @property
    def local_clients(self) -> set[str]:
        return {"127.0.0.1", "::1", "localhost"}

    @property
    def jobs_root(self) -> Path:
        return self.data_root / "jobs"


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("YGG_BOT_APP_NAME", "yggdrasil-bot"),
        app_version=os.getenv("YGG_BOT_APP_VERSION", "0.1.0"),
        host=os.getenv("YGG_BOT_HOST", "127.0.0.1"),
        port=int(os.getenv("YGG_BOT_PORT", "8100")),
        allow_remote=_as_bool(os.getenv("YGG_BOT_ALLOW_REMOTE"), False),
        docs_url=os.getenv("YGG_BOT_DOCS_URL", "/docs"),
        redoc_url=os.getenv("YGG_BOT_REDOC_URL", "/redoc"),
        openapi_url=os.getenv("YGG_BOT_OPENAPI_URL", "/openapi.json"),
        api_prefix=os.getenv("YGG_BOT_API_PREFIX", "/api"),
        node_id=_default_node_id(),
        data_root=Path(
            os.getenv("YGG_BOT_DATA_ROOT", str(_default_data_root()))
        ).expanduser().resolve(),
        max_cmd_timeout=float(os.getenv("YGG_BOT_MAX_CMD_TIMEOUT", "300")),
        max_python_timeout=float(os.getenv("YGG_BOT_MAX_PYTHON_TIMEOUT", "600")),
        max_concurrent_jobs=int(os.getenv("YGG_BOT_MAX_CONCURRENT_JOBS", "16")),
        job_ttl=int(os.getenv("YGG_BOT_JOB_TTL", "3600")),
        job_max_history=int(os.getenv("YGG_BOT_JOB_MAX_HISTORY", "256")),
    )
