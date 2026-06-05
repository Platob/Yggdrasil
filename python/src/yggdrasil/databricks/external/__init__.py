"""Databricks Unity Catalog **external data** resources.

Namespace for the securables that bind UC to outside storage / systems:
external locations (:mod:`yggdrasil.databricks.external.location`) and storage
credentials, grouped under the :class:`DatabricksExternal` umbrella service
(``client.external.locations`` / ``client.external.credentials``).
"""
from __future__ import annotations

from .location import ExternalLocation, ExternalLocations
from .service import DatabricksExternal

__all__ = ["DatabricksExternal", "ExternalLocation", "ExternalLocations"]
