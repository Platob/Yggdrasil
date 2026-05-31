"""Databricks Unity Catalog **external data** resources.

Namespace for the securables that bind UC to outside storage / systems. Today:
external locations (:mod:`yggdrasil.databricks.external.location`); room to grow
(storage credentials, connections) without disturbing the API.
"""
from __future__ import annotations

from .location import ExternalLocation, ExternalLocations

__all__ = ["ExternalLocation", "ExternalLocations"]
