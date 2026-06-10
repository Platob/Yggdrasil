"""yggdrasil.node — the FastAPI backend for the ygg node.

A confined, in-process server exposing the node's services over HTTP:
a remote-call endpoint (:mod:`.remote` + :mod:`.transport`), a chat
messenger, a pyfunc registry, a confined filesystem, an audit log, and
the tabular/analysis engine (lazy polars scans, projection pushdown,
forecasting). :func:`create_app` wires the full app; :func:`create_api`
(:mod:`.api.app`) builds the lean v2 surface used by the hot-path benches.
"""
from __future__ import annotations

from .config import Settings

__all__ = ["create_app", "Settings"]


def create_app(settings: Settings | None = None):
    from .app import create_app as _create_app

    return _create_app(settings)
