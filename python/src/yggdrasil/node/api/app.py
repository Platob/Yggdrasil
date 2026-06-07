"""Compat re-export — the app factory lives in :mod:`yggdrasil.node.app`.

Older callers and the perf bench import ``create_api`` from
``yggdrasil.node.api.app``; keep that path working by re-exporting the
canonical factory rather than defining a second one.
"""
from __future__ import annotations

from ..app import create_api

__all__ = ["create_api"]
