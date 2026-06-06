"""Back-compat shim — ``LokiBehavior`` was renamed to :class:`LokiSkill`.

Import from :mod:`yggdrasil.loki.skill` instead; this module re-exports the
same objects (and the shared :data:`REGISTRY`) so existing imports keep working.
"""
from __future__ import annotations

from .skill import REGISTRY, LokiBehavior, LokiSkill, get, register, registry

__all__ = ["LokiBehavior", "LokiSkill", "register", "registry", "get", "REGISTRY"]
