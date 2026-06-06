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
from .engine import Completion, TokenEngine
from .tools import Tool, Toolbox, filesystem_toolbox

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
    "Completion",
    "Tool",
    "Toolbox",
    "filesystem_toolbox",
]
