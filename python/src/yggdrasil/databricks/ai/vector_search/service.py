"""Databricks Vector Search service.

``client.ai.vector_search.endpoint("rag")`` and
``client.ai.vector_search.index("main.rag.docs")`` are the two most-used
entry points. The service holds a :class:`VectorSearchDefaults` so
callers can set ``endpoint_name`` / ``endpoint_type`` /
``embedding_model_endpoint_name`` once and stop repeating them on every
call.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Optional

from yggdrasil.databricks.service import DatabricksService
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg

from .resources import (
    DEFAULT_VS_WAIT,
    VectorSearchDefaults,
    VectorSearchEndpoint,
    VectorSearchIndex,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk.service.vectorsearch import (
        VectorSearchEndpointsAPI,
        VectorSearchIndexesAPI,
    )


__all__ = ["VectorSearch"]


LOGGER = logging.getLogger(__name__)


class VectorSearch(DatabricksService):
    """High-level wrapper around Databricks Vector Search APIs.

    Attributes
    ----------
    defaults
        :class:`VectorSearchDefaults` — service-wide configuration.
        Replace via ``client.ai.vector_search.defaults = replace(...)``
        the same way :class:`~yggdrasil.databricks.genie.Genie` does.
    """

    def __init__(
        self,
        client=None,
        defaults: Optional[VectorSearchDefaults] = None,
    ):
        super().__init__(client=client)
        self.defaults: VectorSearchDefaults = (
            defaults if defaults is not None else VectorSearchDefaults()
        )

    # ------------------------------------------------------------------ #
    # SDK boundaries
    # ------------------------------------------------------------------ #
    @property
    def endpoints_api(self) -> "VectorSearchEndpointsAPI":
        return self.client.workspace_client().vector_search_endpoints

    @property
    def indexes_api(self) -> "VectorSearchIndexesAPI":
        return self.client.workspace_client().vector_search_indexes

    # ------------------------------------------------------------------ #
    # Endpoint resolution
    # ------------------------------------------------------------------ #
    def endpoint(self, endpoint_name: Optional[str] = None) -> VectorSearchEndpoint:
        """Return a :class:`VectorSearchEndpoint` handle.

        ``endpoint_name`` defaults to :attr:`VectorSearchDefaults.endpoint_name`.
        """
        name = endpoint_name or self.defaults.endpoint_name
        if not name:
            raise ValueError(
                "No endpoint_name given and VectorSearch.defaults.endpoint_name "
                "is unset. Pass endpoint_name=... or set the default."
            )
        return VectorSearchEndpoint(service=self, endpoint_name=name)

    def list_endpoints(self) -> Iterator[VectorSearchEndpoint]:
        """Iterate over vector-search endpoints visible in this workspace."""
        for info in self.endpoints_api.list_endpoints():
            ep_name = getattr(info, "name", None)
            if not ep_name:
                continue
            yield VectorSearchEndpoint(
                service=self,
                endpoint_name=ep_name,
                details=info,
            )

    def find_endpoint(
        self,
        *,
        name: Optional[str] = None,
    ) -> Optional[VectorSearchEndpoint]:
        """Return the endpoint with this name, or ``None`` when missing."""
        target = name or self.defaults.endpoint_name
        if not target:
            return None
        for ep in self.list_endpoints():
            if ep.endpoint_name == target:
                return ep
        return None

    # ------------------------------------------------------------------ #
    # Index resolution
    # ------------------------------------------------------------------ #
    def index(
        self,
        index_name: str,
        *,
        endpoint_name: Optional[str] = None,
    ) -> VectorSearchIndex:
        """Return a :class:`VectorSearchIndex` handle.

        ``endpoint_name`` resolves to the explicit arg, then
        :attr:`VectorSearchDefaults.endpoint_name`, then the value
        carried by the cached :class:`VectorIndex` infos on first
        :meth:`VectorSearchIndex.refresh`.
        """
        return VectorSearchIndex(
            service=self,
            index_name=index_name,
            endpoint_name=endpoint_name or self.defaults.endpoint_name,
        )

    def list_indexes(
        self,
        *,
        endpoint_name: Optional[str] = None,
    ) -> Iterator[VectorSearchIndex]:
        """Iterate over indexes hosted on a given endpoint."""
        name = endpoint_name or self.defaults.endpoint_name
        if not name:
            raise ValueError(
                "No endpoint_name given and VectorSearch.defaults.endpoint_name "
                "is unset — list_indexes requires an endpoint to scope by."
            )
        for info in self.indexes_api.list_indexes(endpoint_name=name):
            idx_name = getattr(info, "name", None)
            if not idx_name:
                continue
            yield VectorSearchIndex(
                service=self,
                index_name=idx_name,
                endpoint_name=getattr(info, "endpoint_name", None) or name,
            )

    # ------------------------------------------------------------------ #
    # Wait resolution
    # ------------------------------------------------------------------ #
    def _resolve_wait(self, override: WaitingConfigArg) -> WaitingConfig:
        if override is None:
            return self.defaults.wait or DEFAULT_VS_WAIT
        return WaitingConfig.from_(override)
