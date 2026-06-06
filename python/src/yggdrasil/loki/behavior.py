"""Loki behaviors — the pluggable unit of agent action.

A :class:`LokiBehavior` is one thing Loki can *do*: ask Genie, run a SQL
query, ingest an HTTP source, replicate itself, message a peer. Behaviors
declare whether they're :meth:`available` in the current environment (so a
Databricks-only behavior stays dark on a bare shell) and implement
:meth:`run`. They register into a global table so the CLI and the agent can
discover and dispatch them by name.

This module is the **abstraction**; concrete behaviors (replication,
inter-agent messaging, serving, ingestion) land on top of it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .agent import Loki

__all__ = ["LokiBehavior", "register", "registry", "get", "REGISTRY"]

# Global name → behavior-instance registry. Behaviors are stateless dispatch
# objects, so a single shared instance per name is the right granularity.
REGISTRY: dict[str, "LokiBehavior"] = {}


class LokiBehavior(ABC):
    """One discoverable, environment-aware action Loki can perform."""

    #: Unique dispatch name (``loki run <name>``).
    name: ClassVar[str]
    #: One-line human description.
    description: ClassVar[str] = ""
    #: Backend this behavior needs (``"databricks"``, ``"node"``, …) or
    #: ``None`` when it runs anywhere. Drives the default :meth:`available`.
    requires: ClassVar[str | None] = None

    def available(self, agent: "Loki") -> bool:
        """True when this behavior can run in *agent*'s environment.

        Default: available everywhere, unless :attr:`requires` names a
        backend that isn't detected. Override for finer checks.
        """
        if self.requires is None:
            return True
        backend = agent.backend(self.requires)
        return bool(backend and backend.available)

    @abstractmethod
    def run(self, agent: "Loki", **kwargs: Any) -> Any:
        """Perform the behavior, using *agent* as the capability/token provider."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "requires": self.requires,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, requires={self.requires!r})"


def register(behavior: "type[LokiBehavior] | LokiBehavior") -> "type[LokiBehavior] | LokiBehavior":
    """Register a behavior (class or instance) by its ``name``.

    Usable as a decorator on a :class:`LokiBehavior` subclass::

        @register
        class Echo(LokiBehavior):
            name = "echo"
            def run(self, agent, **kw): return kw
    """
    instance = behavior() if isinstance(behavior, type) else behavior
    REGISTRY[instance.name] = instance
    return behavior


def get(name: str) -> "LokiBehavior | None":
    return REGISTRY.get(name)


def registry() -> list["LokiBehavior"]:
    """All registered behaviors, sorted by name."""
    return [REGISTRY[name] for name in sorted(REGISTRY)]
