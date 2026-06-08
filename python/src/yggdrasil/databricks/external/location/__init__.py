"""Unity Catalog external location — resource + service (folder module).

* :class:`ExternalLocation` (``resource.py``) — a single external location.
* :class:`ExternalLocations` (``service.py``) — the workspace collection
  service, reachable as ``client.external_locations``.

API: https://docs.databricks.com/api/workspace/externallocations
"""
from __future__ import annotations

from .resource import ExternalLocation
from .service import ExternalLocations

__all__ = ["ExternalLocation", "ExternalLocations"]
