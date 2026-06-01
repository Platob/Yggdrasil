"""Databricks **external data** umbrella service.

Centralizes the Unity Catalog securables that bind UC to outside storage —
external locations and storage credentials — under one client-level entry
point. Reach each via ``client.external.<service>``::

    client.external.locations["raw_zone"]                 # ExternalLocation
    client.external.locations.create(name, url, cred)
    client.external.credentials.create_aws("prod_s3", "arn:aws:iam::123:role/R")
    client.external.credentials["prod_s3"].aws_client(region="us-east-1")

The legacy flat accessors ``client.external_locations`` / ``client.credentials``
remain as thin aliases onto the same cached sub-services, so existing callers
keep working while new code can group them under ``client.external``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from yggdrasil.databricks.service import DatabricksService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.credentials import Credentials
    from yggdrasil.databricks.external.location import ExternalLocations

__all__ = ["DatabricksExternal"]


class DatabricksExternal(DatabricksService):
    """Umbrella for Databricks external-data sub-services on a single client."""

    def __init__(self, client=None):
        super().__init__(client=client)
        self._locations: "Optional[ExternalLocations]" = None
        self._credentials: "Optional[Credentials]" = None

    @property
    def locations(self) -> "ExternalLocations":
        """External-locations service (lazy + cached on this umbrella)."""
        cached = self._locations
        if cached is None:
            from yggdrasil.databricks.external.location import ExternalLocations

            cached = ExternalLocations(client=self.client)
            self._locations = cached
        return cached

    @property
    def credentials(self) -> "Credentials":
        """Storage-credentials service (lazy + cached on this umbrella)."""
        cached = self._credentials
        if cached is None:
            from yggdrasil.databricks.credentials import Credentials

            cached = Credentials(client=self.client)
            self._credentials = cached
        return cached
