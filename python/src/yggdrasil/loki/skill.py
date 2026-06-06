"""Loki skills — the pluggable unit of agent capability.

A :class:`LokiSkill` is one thing Loki can *do*: ask Genie, run SQL, fetch a
table, browse the web, scaffold a project, drive a cloud service. Skills
declare whether they're :meth:`available` in the current environment (so a
Databricks-only skill stays dark on a bare shell) and implement :meth:`run`.
They register into a global table so the CLI and the agent can discover and
dispatch them by name.

This is the **abstraction**; concrete skills land on top of it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .agent import Loki

__all__ = ["LokiSkill", "register", "registry", "get", "REGISTRY"]

#: Global name → skill-instance registry. Skills are stateless dispatch
#: objects, so a single shared instance per name is the right granularity.
REGISTRY: dict[str, "LokiSkill"] = {}


class LokiSkill(ABC):
    """One discoverable, environment-aware capability Loki can perform."""

    #: Unique dispatch name (``loki run <name>``).
    name: ClassVar[str]
    #: One-line human description.
    description: ClassVar[str] = ""
    #: Backend this skill needs (``"databricks"``, ``"aws"``, …) or ``None``
    #: when it runs anywhere. Drives the default :meth:`available`.
    requires: ClassVar[str | None] = None

    def available(self, agent: "Loki") -> bool:
        """True when this skill can run in *agent*'s environment.

        Default: available everywhere, unless :attr:`requires` names a backend
        that isn't detected. Override for finer checks.
        """
        if self.requires is None:
            return True
        backend = agent.backend(self.requires)
        return bool(backend and backend.available)

    @abstractmethod
    def run(self, agent: "Loki", **kwargs: Any) -> Any:
        """Perform the skill, using *agent* as the capability/token provider."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "requires": self.requires,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, requires={self.requires!r})"


def register(skill: "type[LokiSkill] | LokiSkill") -> "type[LokiSkill] | LokiSkill":
    """Register a skill (class or instance) by its ``name``.

    Usable as a decorator on a :class:`LokiSkill` subclass::

        @register
        class Echo(LokiSkill):
            name = "echo"
            def run(self, agent, **kw): return kw
    """
    instance = skill() if isinstance(skill, type) else skill
    REGISTRY[instance.name] = instance
    return skill


def get(name: str) -> "LokiSkill | None":
    return REGISTRY.get(name)


def registry() -> list["LokiSkill"]:
    """All registered skills, sorted by name."""
    return [REGISTRY[name] for name in sorted(REGISTRY)]
