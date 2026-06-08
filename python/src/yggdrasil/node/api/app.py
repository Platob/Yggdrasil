"""Alias module: ``create_api`` → :func:`yggdrasil.node.app.create_app`.

Some callers/benchmarks import the app factory as
``yggdrasil.node.api.app.create_api``. Keep that path working without
duplicating the factory.
"""
from __future__ import annotations

from yggdrasil.node.app import create_app as create_api

__all__ = ["create_api"]
