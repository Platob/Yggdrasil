"""Loki — the global yggdrasil agent.

Loki is one agent that adapts to wherever it runs. It detects the backends
it can reach (:mod:`yggdrasil.loki.capability`), acts as a **token /
credential provider** for them (chiefly Databricks — when a session is
present Loki hands its authenticated client to whatever it drives), and
dispatches :class:`~yggdrasil.loki.behavior.LokiBehavior` actions. The CLI
(`ygg loki`) is a thin shell over this object.

    from yggdrasil.loki import Loki

    loki = Loki.current()
    loki.card()                      # who am I + what can I reach
    loki.databricks                  # the live DatabricksClient, or None
    loki.run("genie", space="01ef…", question="revenue by region")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from . import behavior as _behavior
from .capability import Backend, detect

if TYPE_CHECKING:
    from yggdrasil.databricks import DatabricksClient

__all__ = ["Loki"]


class Loki:
    """The global yggdrasil agent — capability-aware, token-providing."""

    name = "loki"

    _CURRENT: "Optional[Loki]" = None

    def __init__(self) -> None:
        import getpass
        import socket

        self.user = _safe(getpass.getuser)
        self.host = _safe(socket.gethostname)
        self._backends: "Optional[list[Backend]]" = None

    # -- singleton ---------------------------------------------------------

    @classmethod
    def current(cls) -> "Loki":
        """The process-global Loki (created on first use)."""
        if cls._CURRENT is None:
            cls._CURRENT = cls()
        return cls._CURRENT

    # -- identity ----------------------------------------------------------

    @property
    def agent_id(self) -> int:
        """Stable int64 id derived from ``user@host`` (xxhash, not crypto)."""
        import xxhash

        return xxhash.xxh64_intdigest(f"{self.user}@{self.host}") & 0x7FFFFFFFFFFFFFFF

    # -- backends / capabilities ------------------------------------------

    def backends(self, *, refresh: bool = False) -> list[Backend]:
        """Detected backends (cached; pass ``refresh=True`` to re-sniff)."""
        if self._backends is None or refresh:
            self._backends = detect()
        return self._backends

    def backend(self, name: str, *, refresh: bool = False) -> "Optional[Backend]":
        for b in self.backends(refresh=refresh):
            if b.name == name:
                return b
        return None

    def has(self, name: str) -> bool:
        b = self.backend(name)
        return bool(b and b.available)

    # -- Databricks token provider ----------------------------------------

    @property
    def databricks(self) -> "Optional[DatabricksClient]":
        """The authenticated Databricks client when a session is present.

        This is Loki acting as a token provider: behaviors and downstream
        code take this client to reach Databricks service endpoints (SQL,
        Genie, jobs, serving, …) under the agent's resolved credentials.
        Returns ``None`` when no Databricks session is detected.
        """
        if not self.has("databricks"):
            return None
        from yggdrasil.databricks import DatabricksClient

        return DatabricksClient.current()

    def token_info(self) -> dict[str, Any]:
        """Non-secret summary of the Databricks credentials Loki provides."""
        b = self.backend("databricks")
        if not b or not b.available:
            return {"backend": "databricks", "available": False}
        return {
            "backend": "databricks",
            "available": True,
            "host": b.detail.get("host"),
            "auth_type": b.detail.get("auth_type"),
            "catalog": b.detail.get("catalog"),
            "schema": b.detail.get("schema"),
        }

    def whoami(self, *, probe: bool = False) -> "Optional[str]":
        """The Databricks user, if reachable. ``probe`` allows one network call."""
        client = self.databricks
        if client is None or not probe:
            return None
        try:
            return client.workspace_client().current_user.me().user_name
        except Exception:
            return None

    # -- behaviors ---------------------------------------------------------

    def behaviors(self) -> list["_behavior.LokiBehavior"]:
        return _behavior.registry()

    def behavior(self, name: str) -> "Optional[_behavior.LokiBehavior]":
        return _behavior.get(name)

    def run(self, name: str, **kwargs: Any) -> Any:
        """Dispatch behavior *name* with *kwargs*, using self as the provider."""
        b = self.behavior(name)
        if b is None:
            known = ", ".join(x.name for x in self.behaviors()) or "(none)"
            raise KeyError(f"unknown behavior {name!r}; registered: {known}")
        if not b.available(self):
            raise RuntimeError(
                f"behavior {name!r} needs backend {b.requires!r}, "
                f"which is not available here"
            )
        return b.run(self, **kwargs)

    # -- self-description --------------------------------------------------

    def card(self, *, refresh: bool = False) -> dict[str, Any]:
        """Everything Loki knows about itself — identity, reach, behaviors."""
        return {
            "agent": self.name,
            "agent_id": self.agent_id,
            "user": self.user,
            "host": self.host,
            "backends": [b.to_dict() for b in self.backends(refresh=refresh)],
            "token": self.token_info(),
            "behaviors": [b.to_dict() for b in self.behaviors()],
        }

    def __repr__(self) -> str:
        reach = ",".join(b.name for b in self.backends() if b.available)
        return f"Loki(user={self.user!r}, host={self.host!r}, reach=[{reach}])"


def _safe(fn) -> str:
    try:
        return fn()
    except Exception:
        return "unknown"
