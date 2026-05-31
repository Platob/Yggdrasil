"""Collection-level service for Unity Catalog **external locations**.

Wraps the Databricks ``external_locations`` workspace API
(https://docs.databricks.com/api/workspace/externallocations) — list / get /
create / update / delete — and hands back :class:`ExternalLocation` resources.
Reach it as ``client.external_locations``.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import ExternalLocationInfo

from yggdrasil.databricks.client import DatabricksService
from yggdrasil.databricks.external.location.resource import ExternalLocation

__all__ = ["ExternalLocations"]

logger = logging.getLogger(__name__)


class ExternalLocations(DatabricksService):
    """Service over a workspace's Unity Catalog external locations."""

    @property
    def _api(self):
        return self.client.workspace_client().external_locations

    # -- reads ----------------------------------------------------------
    def get_info(self, name: str, *, include_browse: Optional[bool] = None) -> ExternalLocationInfo:
        """Raw :class:`ExternalLocationInfo` for *name* (one GET)."""
        return self._api.get(name, include_browse=include_browse)

    def location(self, name: str) -> ExternalLocation:
        """A lazy :class:`ExternalLocation` handle (no API call until used)."""
        return ExternalLocation(name, service=self)

    __getitem__ = location

    def get(self, name: str) -> ExternalLocation:
        """Fetch *name* eagerly (raises :class:`NotFound` if absent)."""
        return ExternalLocation(name, service=self, info=self.get_info(name))

    def list(self, **kwargs: Any) -> Iterator[ExternalLocation]:
        """Iterate every external location in the metastore."""
        for info in self._api.list(**kwargs):
            yield ExternalLocation(info.name, service=self, info=info)

    def names(self, **kwargs: Any) -> "list[str]":
        return [info.name for info in self._api.list(**kwargs)]

    def exists(self, name: str) -> bool:
        try:
            self.get_info(name)
            return True
        except NotFound:
            return False

    # -- writes ---------------------------------------------------------
    def create(self, name: str, url: str, credential_name: str, **kwargs: Any) -> ExternalLocation:
        """Create an external location bound to *url* + *credential_name*.

        ``**kwargs`` forward to the SDK (``comment`` / ``read_only`` /
        ``skip_validation`` / ``fallback`` / ``enable_file_events`` / …)."""
        info = self._api.create(name=name, url=url, credential_name=credential_name, **kwargs)
        return ExternalLocation(name, service=self, info=info)

    def update(self, name: str, **changes: Any) -> ExternalLocation:
        """Patch an external location (``url`` / ``credential_name`` /
        ``comment`` / ``read_only`` / ``owner`` / …)."""
        info = self._api.update(name, **changes)
        return ExternalLocation(info.name or name, service=self, info=info)

    def delete(self, name: str, *, force: bool = False) -> None:
        self._api.delete(name, force=force)
