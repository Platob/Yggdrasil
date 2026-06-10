"""Node runtime settings — the single config object every service takes.

A :class:`Settings` carries the node identity, the confined home roots
(``node_home`` for server-owned state, ``front_home`` for the UI bundle),
and the logs root. Services resolve paths relative to ``node_home``.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    node_id: str = "ygg"
    node_home: Path = Path.home() / ".ygg" / "node"
    front_home: Path = Path.home() / ".ygg" / "front"
    logs_root: Path | None = None
    allow_remote: bool = False

    # Cap on rows a tabular preview / inspect will pull eagerly before it
    # falls back to footer metadata (parquet) for the row count.
    tabular_preview_max_rows: int = 1000

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: object) -> None:
        if self.logs_root is None:
            self.logs_root = self.node_home / "logs"
