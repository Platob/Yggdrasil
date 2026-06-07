"""Base environments — assemble/deploy serverless + cluster images, deploy projects.

``dbc.environments`` (the :class:`Environments` service) is the front door; the
module-level functions in :mod:`yggdrasil.databricks.environments.service` carry
the assembly machinery and stay importable for the internals that compose them.
"""
from __future__ import annotations

from .service import *  # noqa: F401,F403 — re-export the public function/const surface
from .service import Environments
from .environment import Environment

__all__ = ["Environments", "Environment"]
