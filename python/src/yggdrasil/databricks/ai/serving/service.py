"""Databricks Model Serving service.

``client.ai.serving.endpoint("my-llm")`` is the single entry point. The
service holds a :class:`ServingDefaults` so callers configure the
"max-config" knobs (scale-to-zero, AI Gateway usage tracking, inference
tables, workload size) once and stop repeating them::

    serving = client.ai.serving

    # Front an OpenAI model behind a governed endpoint
    serving.endpoint("gpt-4o").serve_openai(
        "gpt-4o", api_key_secret="llm/openai_key", wait=True,
    )
    print(serving.endpoint("gpt-4o").chat("Hello!").text)

    # Serve a Unity Catalog agent / custom model
    serving.endpoint("rag-agent").serve_uc_model(
        "main.agents.rag", 3, wait=True,
    )

    # Query a built-in foundation model (no create needed)
    serving.endpoint("databricks-meta-llama-3-3-70b-instruct").chat(
        "Summarise the CAP theorem in one sentence.",
    ).text
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Optional

from yggdrasil.databricks.service import DatabricksService
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg

from .resources import (
    DEFAULT_SERVING_WAIT,
    Served,
    ServingDefaults,
    ServingEndpoint,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk.service.serving import ServingEndpointsAPI


__all__ = ["ModelServing"]


LOGGER = logging.getLogger(__name__)


class ModelServing(DatabricksService):
    """High-level wrapper around the Databricks Model Serving API.

    Serves LLMs, Mosaic AI agents, foundation models, external models,
    and classic ML models behind stable endpoints.

    Attributes
    ----------
    defaults
        :class:`ServingDefaults` — service-wide configuration. Replace via
        ``client.ai.serving.defaults = replace(...)`` the same way
        :class:`~yggdrasil.databricks.ai.vector_search.VectorSearch` does.
    """

    def __init__(self, client=None, defaults: Optional[ServingDefaults] = None):
        super().__init__(client=client)
        self.defaults: ServingDefaults = defaults if defaults is not None else ServingDefaults()

    # ------------------------------------------------------------------ #
    # SDK boundary
    # ------------------------------------------------------------------ #
    @property
    def endpoints_api(self) -> "ServingEndpointsAPI":
        return self.client.workspace_client().serving_endpoints

    # Builders namespace — ``client.ai.serving.served.openai(...)`` reads as
    # well as the bare ``Served.openai(...)`` import.
    served = Served

    # ------------------------------------------------------------------ #
    # Endpoint resolution
    # ------------------------------------------------------------------ #
    def endpoint(self, name: str) -> ServingEndpoint:
        """Return a :class:`ServingEndpoint` handle (no API call)."""
        return ServingEndpoint(service=self, name=name)

    def list_endpoints(self) -> Iterator[ServingEndpoint]:
        """Iterate over serving endpoints visible in this workspace."""
        for info in self.endpoints_api.list():
            ep_name = getattr(info, "name", None)
            if not ep_name:
                continue
            # ``list`` returns the lighter ``ServingEndpoint`` summary, not the
            # ``ServingEndpointDetailed`` ``infos`` expects — so leave the
            # handle's cache empty and let ``infos`` lazily fetch the detail.
            yield ServingEndpoint(service=self, name=ep_name)

    def find_endpoint(self, *, name: str) -> Optional[ServingEndpoint]:
        """Return the endpoint with this name, or ``None`` when missing."""
        ep = self.endpoint(name)
        return ep if ep.exists() else None

    # ------------------------------------------------------------------ #
    # Wait resolution
    # ------------------------------------------------------------------ #
    def _resolve_wait(self, override: WaitingConfigArg) -> WaitingConfig:
        if override is None or override is True:
            return self.defaults.wait or DEFAULT_SERVING_WAIT
        return WaitingConfig.from_(override)
