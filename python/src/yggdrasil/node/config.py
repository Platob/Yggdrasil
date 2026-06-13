"""Node settings.

The YGG node serves a local FastAPI app rooted at a home directory. All
filesystem-facing services resolve paths relative to ``node_home``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    node_id: str = "local"
    node_home: Path = field(default_factory=lambda: Path.home() / ".ygg" / "node")
    front_home: Path = field(default_factory=lambda: Path.home() / ".ygg" / "front")
    logs_root: Path = field(default_factory=lambda: Path.home() / ".ygg" / "logs")
    allow_remote: bool = False
    max_read_bytes: int = 4 * 1024 * 1024  # 4 MB
    tabular_preview_max_rows: int = 10_000
    host: str = "127.0.0.1"
    port: int = 8100

    def __post_init__(self) -> None:
        # Callers routinely pass strings from CLI args / env; coerce to Path
        # so downstream services can rely on Path semantics.
        if not isinstance(self.node_home, Path):
            self.node_home = Path(self.node_home)
        if not isinstance(self.front_home, Path):
            self.front_home = Path(self.front_home)
        if not isinstance(self.logs_root, Path):
            self.logs_root = Path(self.logs_root)
