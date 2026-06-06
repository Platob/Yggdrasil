"""Built-in Loki skills (canonical module).

Re-exports the built-in skills. The implementations currently live in
:mod:`yggdrasil.loki.behaviors` (the historical name); this module is the
canonical ``skills`` surface and the place new built-ins should land.
"""
from __future__ import annotations

from .behaviors import (  # noqa: F401
    AgentBehavior,
    GenieBehavior,
    PythonProjectBehavior,
    TabularBehavior,
    WebBehavior,
)

#: Alias-friendly handles under the skill name.
AgentSkill = AgentBehavior
GenieSkill = GenieBehavior
PythonProjectSkill = PythonProjectBehavior
TabularSkill = TabularBehavior
WebSkill = WebBehavior

__all__ = [
    "AgentSkill", "GenieSkill", "PythonProjectSkill", "TabularSkill", "WebSkill",
    "AgentBehavior", "GenieBehavior", "PythonProjectBehavior", "TabularBehavior", "WebBehavior",
]
