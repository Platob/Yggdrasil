"""Wheel registry — build/upload/deploy/browse wheels in the workspace.

``dbc.wheels`` (the :class:`Wheels` service) is the front door; the module-level
functions in :mod:`yggdrasil.databricks.wheels.service` carry the build/upload
machinery and stay importable for the internals that compose them.
"""
from __future__ import annotations

from .service import *  # noqa: F401,F403 — re-export the public function/const surface
from .service import Wheels
from .wheel import Wheel

__all__ = ["Wheels", "Wheel"]
