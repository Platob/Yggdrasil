"""Loki — the global yggdrasil agent.

Loki adapts to wherever it runs: it detects the backends it can reach
(Databricks session, node, local), acts as a token/credential provider for
them, and dispatches pluggable :class:`LokiSkill` actions. Driven from
code via :class:`Loki` or from the terminal via ``ygg loki``.

    from yggdrasil.loki import Loki
    Loki.current().card()
"""
from .agent import Loki
from .skill import LokiSkill, register, registry
from .capability import Backend, detect
from .engine import Completion, TokenEngine
from .tools import Tool, Toolbox, filesystem_toolbox
from .usage import METER, ModelPricing, ModelUsage, TokenMeter, price_for
from .session import LokiSession
from .memory import LokiMemory

# Import the built-in skills so they register on package import.
from . import skills as _skills  # noqa: F401
from . import web

__all__ = [
    "Loki",
    "LokiSkill",
    "Backend",
    "detect",
    "register",
    "registry",
    "TokenEngine",
    "Completion",
    "Tool",
    "Toolbox",
    "filesystem_toolbox",
    "METER",
    "TokenMeter",
    "ModelUsage",
    "ModelPricing",
    "price_for",
    "web",
    "LokiSession",
    "LokiMemory",
]
