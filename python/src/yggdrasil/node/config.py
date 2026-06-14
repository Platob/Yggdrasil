"""Node settings — a plain pydantic model so it stays trivially constructible.

Not :class:`BaseSettings`: the node is driven from code and tests that hand it
explicit homes, so we want a model with sane defaults and no env-var magic.
``model_post_init`` ensures the home + log directories exist on construction.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime configuration for a yggdrasil node."""

    node_id: str = "ygg-node"
    node_home: Path = Field(default_factory=lambda: Path.home() / ".ygg" / "node")
    front_home: Path = Field(default_factory=lambda: Path.home() / ".ygg" / "front")
    logs_root: Path | None = None
    allow_remote: bool = False
    tabular_preview_max_rows: int = 1000
    history_size: int = 1000
    max_messages_per_channel: int = 10_000
    max_read_bytes: int = 4 * 1024 * 1024

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: object) -> None:
        self.node_home.mkdir(parents=True, exist_ok=True)
        if self.logs_root is None:
            object.__setattr__(self, "logs_root", self.node_home / "logs")
        self.logs_root.mkdir(parents=True, exist_ok=True)
