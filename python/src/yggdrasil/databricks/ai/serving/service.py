"""Databricks model-serving service.

``client.ai.serving`` is the entry point. The two most-used calls are
``client.ai.serving.chat([...])`` and ``client.ai.serving.complete("...")``,
which hit :attr:`ServingDefaults.endpoint_name` (a Foundation Model endpoint
by default) unless a specific ``endpoint_name`` is passed. Reach a named
endpoint handle via ``client.ai.serving.endpoint("my-endpoint")``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

from yggdrasil.databricks.service import DatabricksService

from .resources import (
    ChatResult,
    MessageLike,
    ServingDefaults,
    ServingEndpoint,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk.service.serving import ServingEndpointsAPI


__all__ = ["ModelServing"]


LOGGER = logging.getLogger(__name__)


class ModelServing(DatabricksService):
    """High-level wrapper around Databricks serving endpoints.

    Attributes
    ----------
    defaults
        :class:`ServingDefaults` — service-wide endpoint + generation
        configuration. Replace via
        ``client.ai.serving.defaults = replace(client.ai.serving.defaults, ...)``.
    """

    def __init__(
        self,
        client=None,
        defaults: Optional[ServingDefaults] = None,
    ):
        super().__init__(client=client)
        self.defaults: ServingDefaults = (
            defaults if defaults is not None else ServingDefaults()
        )

    # ------------------------------------------------------------------ #
    # SDK boundary
    # ------------------------------------------------------------------ #
    @property
    def endpoints_api(self) -> "ServingEndpointsAPI":
        return self.client.workspace_client().serving_endpoints

    # ------------------------------------------------------------------ #
    # Endpoint resolution
    # ------------------------------------------------------------------ #
    def endpoint(self, endpoint_name: Optional[str] = None) -> ServingEndpoint:
        """Return a :class:`ServingEndpoint` handle.

        ``endpoint_name`` defaults to :attr:`ServingDefaults.endpoint_name`.
        """
        name = endpoint_name or self.defaults.endpoint_name
        if not name:
            raise ValueError(
                "No endpoint_name given and ModelServing.defaults.endpoint_name "
                "is unset. Pass endpoint_name=... or set the default."
            )
        return ServingEndpoint(service=self, endpoint_name=name)

    def list_endpoints(self) -> Iterator[ServingEndpoint]:
        """Iterate over serving endpoints visible in this workspace."""
        for info in self.endpoints_api.list():
            ep_name = getattr(info, "name", None)
            if not ep_name:
                continue
            yield ServingEndpoint(service=self, endpoint_name=ep_name, details=info)

    def find_endpoint(self, *, name: Optional[str] = None) -> Optional[ServingEndpoint]:
        """Return the endpoint with this name, or ``None`` when missing."""
        target = name or self.defaults.endpoint_name
        if not target:
            return None
        ep = ServingEndpoint(service=self, endpoint_name=target)
        return ep if ep.exists() else None

    # ------------------------------------------------------------------ #
    # Inference shortcuts (default endpoint)
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: Sequence[MessageLike],
        *,
        endpoint_name: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
        **extra,
    ) -> ChatResult:
        """Chat completion against the default (or named) endpoint."""
        return self.endpoint(endpoint_name).chat(
            messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **extra,
        )

    def complete(
        self,
        prompt: str,
        *,
        endpoint_name: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **extra,
    ) -> ChatResult:
        """Single-prompt completion against the default (or named) endpoint."""
        return self.endpoint(endpoint_name).complete(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )
