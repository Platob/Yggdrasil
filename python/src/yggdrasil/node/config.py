"""Runtime settings for the yggdrasil node server.

Plain object, not pydantic-settings — the node is constructed both directly
(``Settings(node_home=...)`` in benchmarks/tests) and from the environment
(``Settings.from_env()`` for the ``uvicorn yggdrasil.node.app:app`` path). All
path fields are coerced to :class:`pathlib.Path` so downstream services can
``/`` and ``.mkdir()`` without re-coercing.
"""
from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(
        self,
        node_id: str = "default",
        node_home: Path | str = Path.home() / ".ygg" / "node",
        front_home: Path | str | None = None,
        saga_home: Path | str | None = None,
        allow_remote: bool = False,
        tabular_preview_max_rows: int = 10_000,
        saga_result_spill_rows: int = 1_000_000,
        max_read_bytes: int = 4 * 1024 * 1024,
        logs_root: Path | str | None = None,
        port: int = 8100,
        seed_defaults: bool = True,
    ) -> None:
        self.node_id = node_id
        self.node_home = Path(node_home)
        # front_home / saga_home / logs_root all hang off node_home when unset
        # so a caller only has to point one knob at a temp dir in a bench.
        self.front_home = Path(front_home) if front_home is not None else self.node_home
        self.saga_home = Path(saga_home) if saga_home is not None else self.node_home / ".saga"
        self.logs_root = Path(logs_root) if logs_root is not None else self.node_home / "logs"
        self.allow_remote = allow_remote
        self.tabular_preview_max_rows = tabular_preview_max_rows
        self.saga_result_spill_rows = saga_result_spill_rows
        self.max_read_bytes = max_read_bytes
        self.port = port
        self.seed_defaults = seed_defaults

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        """Build settings from ``YGG_NODE_*`` environment variables.

        This is the entry point the ``app:app`` module callable uses so a
        ``uvicorn yggdrasil.node.app:app`` launch picks up the same config the
        spawning bench passed through the subprocess environment.
        """
        e = os.environ if env is None else env
        kwargs: dict[str, object] = {}
        if (v := e.get("YGG_NODE_NODE_ID")) is not None:
            kwargs["node_id"] = v
        if (v := e.get("YGG_NODE_HOME")) is not None:
            kwargs["node_home"] = v
        if (v := e.get("YGG_NODE_FRONT_HOME")) is not None:
            kwargs["front_home"] = v
        if (v := e.get("YGG_NODE_SAGA_HOME")) is not None:
            kwargs["saga_home"] = v
        if (v := e.get("YGG_NODE_PORT")) is not None:
            kwargs["port"] = int(v)
        if (v := e.get("YGG_NODE_SEED_DEFAULTS")) is not None:
            kwargs["seed_defaults"] = v not in ("0", "false", "False", "")
        if (v := e.get("YGG_NODE_ALLOW_REMOTE")) is not None:
            kwargs["allow_remote"] = v not in ("0", "false", "False", "")
        return cls(**kwargs)
