from __future__ import annotations

import os
import platform
import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from yggdrasil.version import __version__

_NODE_ID_FILE = ".ygg_node_id"


def _stable_node_id() -> str:
    """Return a stable node ID, persisted in ~/.node/.ygg_node_id.

    On first call generates ``{hostname}-{8hex}`` and writes it to disk.
    Subsequent calls return the same ID. Override with YGG_NODE_NODE_ID env var.
    """
    env_id = os.getenv("YGG_NODE_NODE_ID")
    if env_id:
        return env_id

    node_root = Path.home() / ".node"
    id_file = node_root / _NODE_ID_FILE
    if id_file.exists():
        try:
            return id_file.read_text().strip()
        except OSError:
            pass

    node_id = f"{platform.node()}-{uuid.uuid4().hex[:8]}"
    node_root.mkdir(parents=True, exist_ok=True)
    try:
        id_file.write_text(node_id)
    except OSError:
        pass
    return node_id


def _node_home() -> Path:
    return Path.home() / ".node" / _stable_node_id()


def _find_open_port(start: int = 8100, end: int = 8200) -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", start))
            return start
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def _default_front_home() -> Path:
    return Path(__file__).resolve().parents[4] / "nextjs"


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "yggdrasil-node"
    # Single source of truth: the yggdrasil package version.
    app_version: str = __version__
    host: str = "0.0.0.0"
    port: int = 8100
    front_port: int = 3000
    allow_remote: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    api_prefix: str = "/api"
    node_id: str = field(default_factory=_stable_node_id)
    node_home: Path = field(default_factory=_node_home)
    front_home: Path = field(default_factory=_default_front_home)
    max_cmd_timeout: float = 300.0
    max_python_timeout: float = 600.0
    max_concurrent_jobs: int = 16
    job_ttl: int = 3600
    job_max_history: int = 256
    log_retention_days: int = 7
    max_environments: int = 32
    max_functions: int = 256
    max_runs_history: int = 512
    max_log_lines_per_stream: int = 1000
    run_heartbeat_interval: float = 2.0
    run_cancel_grace_seconds: float = 1.5
    # Largest slice of a file the /fs/read preview will pull into memory.
    # Bigger files are read up to this cap and flagged ``truncated`` — the
    # node never loads a multi-GB file whole just to preview it.
    max_read_bytes: int = 4 * 1024 * 1024
    # Upper bound on tree nodes a single recursive fs walk (du / search / grep)
    # visits before it stops and reports a partial result — keeps a scan of a
    # huge tree bounded in both time and memory.
    du_max_entries: int = 200_000
    # Seconds a PyEnv's resolved interpreter version + installed-library
    # listing stays cached before the next ``pip list`` subprocess runs —
    # keeps the UI's per-env package view from flooding the node.
    pyenv_packages_cache_ttl: float = 60.0
    # Seed a ``default`` PyEnv + starter PyFuncs on startup so a fresh node
    # is immediately useful. The env builds in the background; functions are
    # registered instantly and run on the node interpreter until it's ready.
    seed_defaults: bool = True

    @property
    def local_clients(self) -> set[str]:
        return {"127.0.0.1", "::1", "localhost"}

    @property
    def data_root(self) -> Path:
        return self.node_home / "data"

    @property
    def jobs_root(self) -> Path:
        return self.data_root / "jobs"

    @property
    def files_root(self) -> Path:
        return self.data_root / "files"

    @property
    def mirrors_root(self) -> Path:
        return self.node_home / "mirrors"

    @property
    def cache_root(self) -> Path:
        return self.node_home / "cache"

    @property
    def logs_root(self) -> Path:
        return self.node_home / "logs"

    @property
    def spill_root(self) -> Path:
        return self.node_home / "spill"


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("YGG_NODE_APP_NAME", "yggdrasil-node"),
        app_version=os.getenv("YGG_NODE_APP_VERSION", __version__),
        host=os.getenv("YGG_NODE_HOST", "0.0.0.0"),
        port=int(os.getenv("YGG_NODE_PORT", "8100")),
        front_port=int(os.getenv("YGG_NODE_FRONT_PORT", "3000")),
        allow_remote=_as_bool(os.getenv("YGG_NODE_ALLOW_REMOTE"), True),
        docs_url=os.getenv("YGG_NODE_DOCS_URL", "/docs"),
        redoc_url=os.getenv("YGG_NODE_REDOC_URL", "/redoc"),
        openapi_url=os.getenv("YGG_NODE_OPENAPI_URL", "/openapi.json"),
        api_prefix=os.getenv("YGG_NODE_API_PREFIX", "/api"),
        node_id=_stable_node_id(),
        node_home=Path(
            os.getenv("YGG_NODE_HOME", str(_node_home()))
        ).expanduser().resolve(),
        front_home=Path(
            os.getenv("YGG_NODE_FRONT_HOME", str(_default_front_home()))
        ).expanduser().resolve(),
        max_cmd_timeout=float(os.getenv("YGG_NODE_MAX_CMD_TIMEOUT", "300")),
        max_python_timeout=float(os.getenv("YGG_NODE_MAX_PYTHON_TIMEOUT", "600")),
        max_concurrent_jobs=int(os.getenv("YGG_NODE_MAX_CONCURRENT_JOBS", "16")),
        job_ttl=int(os.getenv("YGG_NODE_JOB_TTL", "3600")),
        job_max_history=int(os.getenv("YGG_NODE_JOB_MAX_HISTORY", "256")),
        log_retention_days=int(os.getenv("YGG_NODE_LOG_RETENTION_DAYS", "7")),
        max_environments=int(os.getenv("YGG_NODE_MAX_ENVIRONMENTS", "32")),
        max_functions=int(os.getenv("YGG_NODE_MAX_FUNCTIONS", "256")),
        max_runs_history=int(os.getenv("YGG_NODE_MAX_RUNS_HISTORY", "512")),
        max_log_lines_per_stream=int(os.getenv("YGG_NODE_MAX_LOG_LINES", "1000")),
        run_heartbeat_interval=float(os.getenv("YGG_NODE_RUN_HEARTBEAT", "2.0")),
        run_cancel_grace_seconds=float(os.getenv("YGG_NODE_RUN_CANCEL_GRACE", "1.5")),
        max_read_bytes=int(os.getenv("YGG_NODE_MAX_READ_BYTES", str(4 * 1024 * 1024))),
        du_max_entries=int(os.getenv("YGG_NODE_DU_MAX_ENTRIES", "200000")),
        pyenv_packages_cache_ttl=float(os.getenv("YGG_NODE_PYENV_PKG_TTL", "60")),
        seed_defaults=_as_bool(os.getenv("YGG_NODE_SEED_DEFAULTS"), True),
    )
