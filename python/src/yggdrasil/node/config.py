"""Node configuration.

:class:`Settings` is the single config object threaded through every
service. It carries identity (`node_id`), filesystem roots (`node_home`,
`front_home`, `logs_root`), the read cap (`max_read_bytes`), and whether
remote callers are accepted (`allow_remote`).
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    """Runtime settings for a node instance.

    ``logs_root`` defaults to ``node_home / "logs"`` when not supplied.
    """

    allow_remote: bool = False
    node_id: str = "default"
    node_home: Path = Path(".")
    front_home: Path = Path(".")
    max_read_bytes: int = 4 * 1024 * 1024
    logs_root: Path | None = None

    @model_validator(mode="after")
    def _default_logs_root(self) -> "Settings":
        if self.logs_root is None:
            object.__setattr__(self, "logs_root", self.node_home / "logs")
        return self
