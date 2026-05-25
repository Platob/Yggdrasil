from __future__ import annotations

import hashlib
import os
import platform
import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def _user_key() -> str:
    raw = f"{os.getlogin() if hasattr(os, 'login') else 'user'}-{platform.node()}"
    try:
        raw = os.getlogin() + "@" + platform.node()
    except OSError:
        raw = os.environ.get("USER", os.environ.get("USERNAME", "user")) + "@" + platform.node()
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _bot_home() -> Path:
    return Path.home() / ".bot" / _user_key()


def _default_node_id() -> str:
    return os.getenv("YGG_BOT_NODE_ID", f"{platform.node()}-{uuid.uuid4().hex[:8]}")


def _find_open_port(start: int = 8100, end: int = 8200) -> int:
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    return start


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
    bot_home: Path = field(default_factory=_bot_home)
    max_cmd_timeout: float = 300.0
    max_python_timeout: float = 600.0
    max_concurrent_jobs: int = 16
    job_ttl: int = 3600
    job_max_history: int = 256
    log_retention_days: int = 7

    @property
    def local_clients(self) -> set[str]:
        return {"127.0.0.1", "::1", "localhost"}

    @property
    def data_root(self) -> Path:
        return self.bot_home / "data"

    @property
    def jobs_root(self) -> Path:
        return self.data_root / "jobs"

    @property
    def cache_root(self) -> Path:
        return self.bot_home / "cache"

    @property
    def logs_root(self) -> Path:
        return self.bot_home / "logs"

    @property
    def spill_root(self) -> Path:
        return self.bot_home / "spill"


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
        bot_home=Path(
            os.getenv("YGG_BOT_HOME", str(_bot_home()))
        ).expanduser().resolve(),
        max_cmd_timeout=float(os.getenv("YGG_BOT_MAX_CMD_TIMEOUT", "300")),
        max_python_timeout=float(os.getenv("YGG_BOT_MAX_PYTHON_TIMEOUT", "600")),
        max_concurrent_jobs=int(os.getenv("YGG_BOT_MAX_CONCURRENT_JOBS", "16")),
        job_ttl=int(os.getenv("YGG_BOT_JOB_TTL", "3600")),
        job_max_history=int(os.getenv("YGG_BOT_JOB_MAX_HISTORY", "256")),
        log_retention_days=int(os.getenv("YGG_BOT_LOG_RETENTION_DAYS", "7")),
    )
