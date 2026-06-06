"""Loki — the global yggdrasil agent.

Loki adapts to wherever it runs: it detects the backends it can reach
(Databricks session, node, local), acts as a token/credential provider for
them, and dispatches pluggable :class:`LokiBehavior` actions. Driven from
code via :class:`Loki` or from the terminal via ``ygg loki``.

    from yggdrasil.loki import Loki
    Loki.current().card()
"""
from .agent import Loki
from .behavior import LokiBehavior, register, registry
from .capability import Backend, detect
from .engine import AgentResponse, Completion, EngineType, TokenEngine
from .model import (
    Complexity,
    ModelSpec,
    Provider,
    TokenModel,
    models,
    register_model,
    select_model,
    unregister_model,
)
from .replica import Replica

# Import the built-in behaviors so they register on package import.
from . import behaviors as _behaviors  # noqa: F401

__all__ = [
    "Loki",
    "LokiBehavior",
    "Backend",
    "detect",
    "register",
    "registry",
    "TokenEngine",
    "EngineType",
    "Completion",
    "AgentResponse",
    "Replica",
    "TokenModel",
    "ModelSpec",
    "Provider",
    "Complexity",
    "models",
    "select_model",
    "register_model",
    "unregister_model",
]
