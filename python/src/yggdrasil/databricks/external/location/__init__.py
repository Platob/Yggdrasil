"""Unity Catalog external location — resource + service (folder module).

* :class:`ExternalLocation` (``location.py``) — a single external location.
* :class:`ExternalLocations` (``locations.py``) — the workspace collection
  service, reachable as ``client.external_locations``.

API: https://docs.databricks.com/api/workspace/externallocations
"""
from __future__ import annotations

from .location import ExternalLocation
from .locations import ExternalLocations

__all__ = ["ExternalLocation", "ExternalLocations"]
