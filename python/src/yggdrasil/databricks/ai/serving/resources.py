"""Model Serving resources and default configuration.

Mirrors the Databricks Model Serving hierarchy as Yggdrasil resources.
A *serving endpoint* is the unit that serves LLMs, agents, foundation
models, external models, and classic ML models behind a single stable
URL; it routes traffic across one or more *served entities*.

- :class:`ServingDefaults` — frozen config attached to the
  :class:`~.service.ModelServing` service (``service.defaults``).
  Carries the "max-config-by-default" knobs every create inherits:
  scale-to-zero on, AI Gateway usage tracking on, inference-table
  payload capture on (when a catalog/schema is resolvable), the
  default workload size/type, and the polling budget for the
  long-running create / update operations.
- :class:`Served` — namespace of builders that turn a model reference
  into a :class:`ServedEntityInput`: a Unity Catalog registered model
  / agent (:meth:`Served.uc_model`), a pay-per-token or
  provisioned-throughput foundation model, or an external LLM
  (:meth:`Served.openai`, :meth:`Served.anthropic`,
  :meth:`Served.amazon_bedrock`, :meth:`Served.cohere`,
  :meth:`Served.google_vertex`, :meth:`Served.external`).
- :class:`ServingEndpoint` — a single serving endpoint. Exposes
  :meth:`create` / :meth:`ensure_created` / :meth:`update_config` /
  :meth:`delete` / :meth:`wait_ready` plus the query data-plane
  (:meth:`query` / :meth:`chat` / :meth:`complete` / :meth:`embed`)
  and the ops surface (:meth:`logs` / :meth:`build_logs` /
  :meth:`metrics` / :meth:`openapi` / tag + gateway management).
- :class:`ServingQueryResult` — wrapper around the
  ``QueryEndpointResponse`` body that exposes the chat / completion /
  embedding / prediction shapes without the caller poking at the raw
  SDK object.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

from yggdrasil.databricks.resource import DatabricksResource
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg
from yggdrasil.url import URL

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk.service.serving import (
        AiGatewayConfig,
        ChatMessage,
        EndpointTag,
        ExportMetricsResponse,
        ExternalModel,
        GetOpenApiResponse,
        QueryEndpointResponse,
        RateLimit,
        ServedEntityInput,
        ServerLogsResponse,
        ServingEndpointDetailed,
        TrafficConfig,
    )

    from .service import ModelServing


MessagesLike = Sequence[Union[Mapping[str, Any], "ChatMessage"]]


__all__ = [
    "DEFAULT_SERVING_WAIT",
    "Served",
    "ServingDefaults",
    "ServingEndpoint",
    "ServingQueryResult",
]


LOGGER = logging.getLogger(__name__)

#: Default wait budget for endpoint create / config-update operations.
#: Matches the SDK's own ``create_and_wait`` / ``update_config_and_wait``
#: default of 20 minutes — a fresh endpoint pulls the model image and
#: provisions compute, which routinely takes 5-15 minutes (longer for
#: GPU workloads), so most callers want a generous ceiling.
DEFAULT_SERVING_WAIT: WaitingConfig = WaitingConfig(timeout=1200.0, interval=10.0)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServingDefaults:
    """Default configuration for :class:`~.service.ModelServing`.

    The defaults lean *maximal*: every knob that makes an endpoint more
    observable and more production-ready is on out of the box, so a bare
    ``service.endpoint("x").serve_openai(...)`` lands a fully governed
    endpoint (usage tracking + payload capture + scale-to-zero) without
    the caller listing it all. Set once on the service and every
    subsequent call inherits these values unless overridden inline::

        from dataclasses import replace
        client.ai.serving.defaults = replace(
            client.ai.serving.defaults,
            workload_size="Medium",
            inference_table_catalog="main",
            inference_table_schema="serving_logs",
        )

    Attributes
    ----------
    workload_size
        Concurrency band for served entities (``"Small"`` / ``"Medium"``
        / ``"Large"``). Databricks maps each band to a provisioned
        concurrency range. ``"Small"`` by default.
    workload_type
        Compute type for served entities — ``"CPU"`` or one of the GPU
        tiers (``"GPU_SMALL"`` … ``"MULTIGPU_MEDIUM"``). Accepts the SDK
        :class:`ServedModelInputWorkloadType` enum or its string name.
    scale_to_zero_enabled
        Whether served entities scale to zero replicas when idle. ``True``
        by default — the cheap, bursty path most callers want.
    route_optimized
        Opt into route-optimized serving (lower-latency routing for
        provisioned-throughput / external endpoints). ``None`` leaves it
        to the platform default.
    enable_usage_tracking
        Turn on AI Gateway usage tracking (per-request token / cost
        accounting in system tables). ``True`` by default.
    enable_inference_table
        Turn on AI Gateway request/response payload capture into a Delta
        table. Only applied when a catalog **and** schema are resolvable
        (from these defaults or the client's ``catalog_name`` /
        ``schema_name``). ``True`` by default.
    inference_table_catalog, inference_table_schema, inference_table_prefix
        Destination for captured payloads. Catalog / schema fall back to
        the client's bound catalog / schema; the prefix defaults to the
        endpoint name.
    rate_limit_calls
        Optional AI Gateway rate-limit ceiling (calls per
        :attr:`rate_limit_renewal_period`). ``None`` (default) leaves the
        endpoint unthrottled.
    rate_limit_key
        Scope the rate limit applies to — ``"endpoint"`` (default) or
        ``"user"``.
    rate_limit_renewal_period
        Window the rate limit resets over. ``"minute"`` (the only value
        the API currently accepts).
    default_task
        Task hint used by :meth:`ServingEndpoint.serve_external` /
        :meth:`ServingEndpoint.serve_foundation` when the caller does not
        pass one. ``"llm/v1/chat"`` — the agent / chat path.
    wait
        :class:`~yggdrasil.dataclasses.WaitingConfig` budget for
        long-running create / update operations. Defaults to
        :data:`DEFAULT_SERVING_WAIT` (20 minutes / 10 seconds). Override
        per-call by passing ``wait=`` — anything :meth:`WaitingConfig.from_`
        accepts works (seconds, timedelta, deadline, dict, full config).
    tags
        Optional default endpoint tags merged with the service's
        :meth:`~yggdrasil.databricks.service.DatabricksService.default_tags`
        on create.
    """

    workload_size: str = "Small"
    workload_type: str = "CPU"
    scale_to_zero_enabled: bool = True
    route_optimized: Optional[bool] = None
    enable_usage_tracking: bool = True
    enable_inference_table: bool = True
    inference_table_catalog: Optional[str] = None
    inference_table_schema: Optional[str] = None
    inference_table_prefix: Optional[str] = None
    rate_limit_calls: Optional[int] = None
    rate_limit_key: str = "endpoint"
    rate_limit_renewal_period: str = "minute"
    default_task: str = "llm/v1/chat"
    wait: WaitingConfig = DEFAULT_SERVING_WAIT
    tags: Optional[Mapping[str, str]] = None


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_workload_type(value: Any) -> "Any":
    from databricks.sdk.service.serving import ServedModelInputWorkloadType

    if value is None:
        return None
    if isinstance(value, ServedModelInputWorkloadType):
        return value
    return ServedModelInputWorkloadType(str(value).upper())


def _tags_to_list(tags: Optional[Mapping[str, str]]) -> "Optional[list[EndpointTag]]":
    if not tags:
        return None
    from databricks.sdk.service.serving import EndpointTag

    return [EndpointTag(key=str(k), value=str(v)) for k, v in tags.items()]


# ---------------------------------------------------------------------------
# Served-entity builders
# ---------------------------------------------------------------------------


class Served:
    """Builders that turn a model reference into a :class:`ServedEntityInput`.

    Each classmethod returns the SDK ``ServedEntityInput`` ready to drop
    into :meth:`ServingEndpoint.create` / :meth:`ServingEndpoint.update_config`
    ``served_entities=[...]``. Splitting them out keeps the per-provider
    plumbing (which secret field, which config dataclass) in one place and
    the endpoint resource focused on lifecycle.

    Secrets always travel by reference — ``{{secrets/scope/key}}`` — never
    plaintext: the ``*_plaintext`` SDK fields are deliberately not exposed,
    matching ``CLAUDE.md``'s "never persist API keys" posture.
    """

    @staticmethod
    def uc_model(
        model_name: str,
        model_version: Union[str, int],
        *,
        name: Optional[str] = None,
        workload_size: Optional[str] = None,
        workload_type: Any = None,
        scale_to_zero_enabled: Optional[bool] = None,
        min_provisioned_throughput: Optional[int] = None,
        max_provisioned_throughput: Optional[int] = None,
        provisioned_model_units: Optional[int] = None,
        environment_vars: Optional[Mapping[str, str]] = None,
        instance_profile_arn: Optional[str] = None,
    ) -> "ServedEntityInput":
        """Serve a Unity Catalog registered model — a custom model, a
        Mosaic AI **agent**, or a fine-tuned / provisioned foundation
        model referenced as ``catalog.schema.model``.

        Pass ``provisioned_model_units`` (or the min/max throughput pair)
        for a provisioned-throughput foundation model; leave them unset
        and pass ``workload_size`` for a standard custom-model / agent
        deployment.
        """
        from databricks.sdk.service.serving import ServedEntityInput

        return ServedEntityInput(
            entity_name=model_name,
            entity_version=str(model_version),
            name=name,
            workload_size=workload_size,
            workload_type=_coerce_workload_type(workload_type),
            scale_to_zero_enabled=scale_to_zero_enabled,
            min_provisioned_throughput=min_provisioned_throughput,
            max_provisioned_throughput=max_provisioned_throughput,
            provisioned_model_units=provisioned_model_units,
            environment_vars=dict(environment_vars) if environment_vars else None,
            instance_profile_arn=instance_profile_arn,
        )

    @staticmethod
    def external(
        external_model: "ExternalModel",
        *,
        name: Optional[str] = None,
    ) -> "ServedEntityInput":
        """Wrap a pre-built :class:`ExternalModel` into a served entity.

        The per-provider helpers (:meth:`openai`, :meth:`anthropic`, …)
        call through here; reach for this directly only when you need a
        provider the helpers don't cover yet. ``name`` defaults to the
        external model's own name.
        """
        from databricks.sdk.service.serving import ServedEntityInput

        return ServedEntityInput(
            name=name or external_model.name,
            external_model=external_model,
        )

    @staticmethod
    def openai(
        model: str,
        *,
        task: str = "llm/v1/chat",
        api_key_secret: Optional[str] = None,
        api_base: Optional[str] = None,
        api_type: Optional[str] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        name: Optional[str] = None,
    ) -> "ServedEntityInput":
        """Serve an OpenAI (or Azure OpenAI) model as an external model.

        ``api_key_secret`` is a ``{{secrets/scope/key}}`` reference (or a
        bare ``scope/key`` shorthand) — the key is read from a Databricks
        secret scope at request time, never stored on the endpoint.
        """
        from databricks.sdk.service.serving import (
            ExternalModel,
            ExternalModelProvider,
            OpenAiConfig,
        )

        cfg = OpenAiConfig(
            openai_api_key=_secret_ref(api_key_secret),
            openai_api_base=api_base,
            openai_api_type=api_type,
            openai_api_version=api_version,
            openai_organization=organization,
        )
        em = ExternalModel(
            provider=ExternalModelProvider.OPENAI,
            name=model,
            task=task,
            openai_config=cfg,
        )
        return Served.external(em, name=name)

    @staticmethod
    def anthropic(
        model: str,
        *,
        task: str = "llm/v1/chat",
        api_key_secret: Optional[str] = None,
        name: Optional[str] = None,
    ) -> "ServedEntityInput":
        """Serve an Anthropic Claude model as an external model."""
        from databricks.sdk.service.serving import (
            AnthropicConfig,
            ExternalModel,
            ExternalModelProvider,
        )

        em = ExternalModel(
            provider=ExternalModelProvider.ANTHROPIC,
            name=model,
            task=task,
            anthropic_config=AnthropicConfig(anthropic_api_key=_secret_ref(api_key_secret)),
        )
        return Served.external(em, name=name)

    @staticmethod
    def amazon_bedrock(
        model: str,
        *,
        region: str,
        bedrock_provider: str,
        access_key_id_secret: str,
        secret_access_key_secret: str,
        task: str = "llm/v1/chat",
        name: Optional[str] = None,
    ) -> "ServedEntityInput":
        """Serve an Amazon Bedrock model as an external model."""
        from databricks.sdk.service.serving import (
            AmazonBedrockConfig,
            AmazonBedrockConfigBedrockProvider,
            ExternalModel,
            ExternalModelProvider,
        )

        cfg = AmazonBedrockConfig(
            aws_region=region,
            bedrock_provider=AmazonBedrockConfigBedrockProvider(bedrock_provider.lower()),
            aws_access_key_id=_secret_ref(access_key_id_secret),
            aws_secret_access_key=_secret_ref(secret_access_key_secret),
        )
        em = ExternalModel(
            provider=ExternalModelProvider.AMAZON_BEDROCK,
            name=model,
            task=task,
            amazon_bedrock_config=cfg,
        )
        return Served.external(em, name=name)

    @staticmethod
    def cohere(
        model: str,
        *,
        api_key_secret: str,
        api_base: Optional[str] = None,
        task: str = "llm/v1/chat",
        name: Optional[str] = None,
    ) -> "ServedEntityInput":
        """Serve a Cohere model as an external model."""
        from databricks.sdk.service.serving import (
            CohereConfig,
            ExternalModel,
            ExternalModelProvider,
        )

        em = ExternalModel(
            provider=ExternalModelProvider.COHERE,
            name=model,
            task=task,
            cohere_config=CohereConfig(
                cohere_api_key=_secret_ref(api_key_secret),
                cohere_api_base=api_base,
            ),
        )
        return Served.external(em, name=name)

    @staticmethod
    def google_vertex(
        model: str,
        *,
        project_id: str,
        region: str,
        private_key_secret: str,
        task: str = "llm/v1/chat",
        name: Optional[str] = None,
    ) -> "ServedEntityInput":
        """Serve a Google Cloud Vertex AI model as an external model."""
        from databricks.sdk.service.serving import (
            ExternalModel,
            ExternalModelProvider,
            GoogleCloudVertexAiConfig,
        )

        cfg = GoogleCloudVertexAiConfig(
            project_id=project_id,
            region=region,
            private_key=_secret_ref(private_key_secret),
        )
        em = ExternalModel(
            provider=ExternalModelProvider.GOOGLE_CLOUD_VERTEX_AI,
            name=model,
            task=task,
            google_cloud_vertex_ai_config=cfg,
        )
        return Served.external(em, name=name)


def _secret_ref(value: Optional[str]) -> Optional[str]:
    """Normalise a secret reference into the ``{{secrets/scope/key}}`` form.

    Accepts the full ``{{secrets/scope/key}}`` template (returned as-is),
    a bare ``scope/key`` shorthand (wrapped), or ``None``. Plaintext keys
    are intentionally unsupported on this path — secrets ride a Databricks
    secret scope, never the endpoint spec.
    """
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{{") and text.endswith("}}"):
        return text
    return "{{secrets/" + text + "}}"


# ---------------------------------------------------------------------------
# ServingEndpoint
# ---------------------------------------------------------------------------


class ServingEndpoint(DatabricksResource):
    """A Databricks Model Serving endpoint.

    One endpoint fronts one or more *served entities* (a custom model, an
    agent, a foundation model, an external LLM) behind a stable URL and
    splits traffic across them. Provisioning is asynchronous; use
    :meth:`wait_ready` (or pass ``wait=True`` to :meth:`create` /
    :meth:`update_config`) when the next step needs the endpoint serving
    queries.
    """

    def __init__(
        self,
        service: "ModelServing",
        name: str,
        *,
        details: "Optional[ServingEndpointDetailed]" = None,
    ):
        super().__init__(service=service)
        self.service: "ModelServing" = service
        self.name = name
        self._details: "Optional[ServingEndpointDetailed]" = details

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

    @property
    def api(self):
        return self.client.workspace_client().serving_endpoints

    @property
    def explore_url(self) -> URL:
        return URL.from_str(
            f"{self.client.base_url.to_string().rstrip('/')}"
            f"/ml/endpoints/{self.name}"
        )

    # ------------------------------------------------------------------ #
    # Identity / cached infos
    # ------------------------------------------------------------------ #
    @property
    def infos(self) -> "ServingEndpointDetailed":
        """Cached :class:`ServingEndpointDetailed` — fetched on first access."""
        cached = self._details
        if cached is not None:
            return cached
        LOGGER.debug("Fetching serving endpoint %r from remote", self)
        info = self.api.get(name=self.name)
        self._details = info
        return info

    def refresh(self) -> "ServingEndpoint":
        """Re-fetch :attr:`infos` from the API."""
        LOGGER.debug("Refreshing serving endpoint %r", self)
        self._details = self.api.get(name=self.name)
        return self

    def exists(self) -> bool:
        """``True`` when the endpoint exists in the workspace."""
        from databricks.sdk.errors import NotFound

        try:
            _ = self.infos
            return True
        except NotFound:
            return False

    @property
    def state(self) -> Optional[str]:
        """Config-update state (``NOT_UPDATING`` / ``IN_PROGRESS`` / ``UPDATE_FAILED`` …)."""
        st = getattr(self.infos, "state", None)
        cu = getattr(st, "config_update", None) if st is not None else None
        return cu.value if cu is not None and hasattr(cu, "value") else cu

    @property
    def ready(self) -> Optional[str]:
        """Readiness state (``READY`` / ``NOT_READY``)."""
        st = getattr(self.infos, "state", None)
        rd = getattr(st, "ready", None) if st is not None else None
        return rd.value if rd is not None and hasattr(rd, "value") else rd

    @property
    def is_ready(self) -> bool:
        from databricks.sdk.service.serving import EndpointStateReady

        st = getattr(self.infos, "state", None)
        rd = getattr(st, "ready", None) if st is not None else None
        return rd == EndpointStateReady.READY

    @property
    def endpoint_url(self) -> Optional[str]:
        return getattr(self.infos, "endpoint_url", None)

    @property
    def task(self) -> Optional[str]:
        return getattr(self.infos, "task", None)

    @property
    def served_entity_names(self) -> tuple[str, ...]:
        """Names of the served entities in the active config."""
        cfg = getattr(self.infos, "config", None)
        entities = getattr(cfg, "served_entities", None) if cfg is not None else None
        return tuple(e.name or "" for e in (entities or []))

    # ------------------------------------------------------------------ #
    # Config assembly
    # ------------------------------------------------------------------ #
    def _default_ai_gateway(self) -> "Optional[AiGatewayConfig]":
        """Assemble the AI Gateway config from the service defaults.

        Mirrors the "max-config" defaults: usage tracking + inference-table
        payload capture + an optional rate limit. Inference-table capture
        needs a catalog *and* schema (from the defaults or the client's
        bound catalog/schema); without one it's silently skipped — there's
        nowhere governed to write the Delta table. Returns ``None`` when
        every gateway feature is off so no gateway block is sent.
        """
        from databricks.sdk.service.serving import (
            AiGatewayConfig,
            AiGatewayInferenceTableConfig,
            AiGatewayRateLimit,
            AiGatewayRateLimitKey,
            AiGatewayRateLimitRenewalPeriod,
            AiGatewayUsageTrackingConfig,
        )

        d = self.service.defaults
        usage = (
            AiGatewayUsageTrackingConfig(enabled=True) if d.enable_usage_tracking else None
        )

        inference = None
        if d.enable_inference_table:
            catalog = d.inference_table_catalog or getattr(self.client, "catalog_name", None)
            schema = d.inference_table_schema or getattr(self.client, "schema_name", None)
            if catalog and schema:
                inference = AiGatewayInferenceTableConfig(
                    enabled=True,
                    catalog_name=catalog,
                    schema_name=schema,
                    table_name_prefix=d.inference_table_prefix or self.name.replace("-", "_"),
                )
            else:
                LOGGER.debug(
                    "Skipping inference-table capture for %r: no catalog/schema "
                    "resolvable (set ServingDefaults.inference_table_catalog/_schema "
                    "or bind the client to a catalog/schema).",
                    self,
                )

        rate_limits = None
        if d.rate_limit_calls:
            rate_limits = [
                AiGatewayRateLimit(
                    calls=d.rate_limit_calls,
                    key=AiGatewayRateLimitKey(d.rate_limit_key),
                    renewal_period=AiGatewayRateLimitRenewalPeriod(d.rate_limit_renewal_period),
                )
            ]
        if usage is None and inference is None and rate_limits is None:
            return None
        return AiGatewayConfig(
            usage_tracking_config=usage,
            inference_table_config=inference,
            rate_limits=rate_limits,
        )

    def _default_traffic(self, entities: "Sequence[ServedEntityInput]") -> "Optional[TrafficConfig]":
        """Route 100% to a single entity; leave multi-entity routing to the caller."""
        from databricks.sdk.service.serving import Route, TrafficConfig

        named = [e for e in entities if getattr(e, "name", None)]
        if len(named) != 1:
            return None
        return TrafficConfig(
            routes=[Route(served_model_name=named[0].name, traffic_percentage=100)]
        )

    def _apply_defaults_to_entities(
        self,
        entities: "Sequence[ServedEntityInput]",
    ) -> "list[ServedEntityInput]":
        """Fill unset workload / scale-to-zero on custom-model entities.

        External-model entities (those with ``external_model`` set) have
        no workload knobs, so they are left untouched.
        """
        d = self.service.defaults
        out = []
        for e in entities:
            if getattr(e, "external_model", None) is not None:
                out.append(e)
                continue
            if e.workload_size is None:
                e.workload_size = d.workload_size
            if e.workload_type is None:
                e.workload_type = _coerce_workload_type(d.workload_type)
            if e.scale_to_zero_enabled is None and e.provisioned_model_units is None \
                    and e.max_provisioned_throughput is None:
                e.scale_to_zero_enabled = d.scale_to_zero_enabled
            out.append(e)
        return out

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def create(
        self,
        *,
        served_entities: "Optional[Sequence[ServedEntityInput]]" = None,
        traffic_config: "Optional[TrafficConfig]" = None,
        ai_gateway: "Optional[AiGatewayConfig]" = None,
        rate_limits: "Optional[Sequence[RateLimit]]" = None,
        tags: Optional[Mapping[str, str]] = None,
        route_optimized: Optional[bool] = None,
        description: Optional[str] = None,
        wait: WaitingConfigArg = None,
        missing_ok: bool = True,
    ) -> "ServingEndpoint":
        """Create this endpoint.

        Parameters
        ----------
        served_entities
            The entities to serve — build them with :class:`Served`
            (``Served.uc_model(...)``, ``Served.openai(...)``, …).
            Workload size / type / scale-to-zero are filled from
            :class:`ServingDefaults` when left unset on a custom-model
            entity.
        traffic_config
            Optional explicit traffic split. Defaults to 100% to the sole
            entity when exactly one is given.
        ai_gateway
            Explicit AI Gateway config. Defaults to the "max config"
            gateway assembled from :class:`ServingDefaults` (usage
            tracking + payload capture + optional rate limit).
        rate_limits
            Legacy top-level rate limits (prefer the gateway path).
        tags
            Endpoint tags, merged over the service default tags.
        route_optimized
            Opt into route-optimized serving. Defaults to
            :attr:`ServingDefaults.route_optimized`.
        description
            Optional human description.
        wait
            ``None`` (default) returns as soon as the create is accepted.
            Pass ``True`` to block with :attr:`ServingDefaults.wait`, or
            anything :meth:`WaitingConfig.from_` accepts.
        missing_ok
            When ``True`` (default), an existing endpoint with the same
            name is treated as success and :attr:`infos` is refreshed.
        """
        from databricks.sdk.errors import AlreadyExists, DatabricksError
        from databricks.sdk.service.serving import EndpointCoreConfigInput

        entities = self._apply_defaults_to_entities(list(served_entities or []))
        if not entities:
            raise ValueError(
                f"Cannot create {self!r}: no served_entities given. Build them "
                f"with Served.uc_model(...) / Served.openai(...) etc."
            )
        traffic = traffic_config or self._default_traffic(entities)
        gateway = ai_gateway if ai_gateway is not None else self._default_ai_gateway()
        merged_tags = dict(self.service.default_tags())
        if self.service.defaults.tags:
            merged_tags.update(self.service.defaults.tags)
        if tags:
            merged_tags.update(tags)
        ro = route_optimized if route_optimized is not None else self.service.defaults.route_optimized

        LOGGER.debug(
            "Creating serving endpoint %r (entities=%s, gateway=%s, route_optimized=%s)",
            self, [e.name for e in entities], gateway is not None, ro,
        )
        try:
            self._details = self.api.create(
                name=self.name,
                config=EndpointCoreConfigInput(
                    name=self.name,
                    served_entities=entities,
                    traffic_config=traffic,
                ),
                ai_gateway=gateway,
                rate_limits=list(rate_limits) if rate_limits else None,
                tags=_tags_to_list(merged_tags),
                route_optimized=ro,
                description=description,
            ).response
        except AlreadyExists:
            if not missing_ok:
                raise
            LOGGER.debug("Serving endpoint %r already exists — refreshing infos", self)
            self.refresh()
        except DatabricksError as exc:
            if missing_ok and "already exists" in str(exc).lower():
                LOGGER.debug("Serving endpoint %r already exists — refreshing infos", self)
                self.refresh()
            else:
                raise

        if wait is not None and wait is not False:
            self.wait_ready(wait=wait)
        LOGGER.info("Created serving endpoint %r", self)
        return self

    def ensure_created(
        self,
        *,
        served_entities: "Optional[Sequence[ServedEntityInput]]" = None,
        wait: WaitingConfigArg = None,
        **kwargs: Any,
    ) -> "ServingEndpoint":
        """Create this endpoint when missing, otherwise return ``self``."""
        if self.exists():
            return self
        return self.create(served_entities=served_entities, wait=wait, missing_ok=True, **kwargs)

    # ---- convenience single-entity creators ------------------------------ #
    def serve_uc_model(
        self,
        model_name: str,
        model_version: Union[str, int],
        *,
        wait: WaitingConfigArg = None,
        **entity_kwargs: Any,
    ) -> "ServingEndpoint":
        """Create the endpoint serving one Unity Catalog model / agent."""
        return self.create(
            served_entities=[Served.uc_model(model_name, model_version, **entity_kwargs)],
            wait=wait,
        )

    def serve_openai(
        self, model: str, *, wait: WaitingConfigArg = None, **kwargs: Any,
    ) -> "ServingEndpoint":
        """Create the endpoint fronting one OpenAI external model."""
        return self.create(served_entities=[Served.openai(model, **kwargs)], wait=wait)

    def serve_anthropic(
        self, model: str, *, wait: WaitingConfigArg = None, **kwargs: Any,
    ) -> "ServingEndpoint":
        """Create the endpoint fronting one Anthropic external model."""
        return self.create(served_entities=[Served.anthropic(model, **kwargs)], wait=wait)

    def update_config(
        self,
        *,
        served_entities: "Optional[Sequence[ServedEntityInput]]" = None,
        traffic_config: "Optional[TrafficConfig]" = None,
        wait: WaitingConfigArg = None,
    ) -> "ServingEndpoint":
        """Swap the served entities / traffic split on a live endpoint.

        This is the rolling-update path: the new config is provisioned
        alongside the old one and traffic shifts once it is ready.
        """
        entities = self._apply_defaults_to_entities(list(served_entities or []))
        traffic = traffic_config or (self._default_traffic(entities) if entities else None)
        LOGGER.debug("Updating config on serving endpoint %r", self)
        self._details = self.api.update_config(
            name=self.name,
            served_entities=entities or None,
            traffic_config=traffic,
        ).response
        if wait is not None and wait is not False:
            self.wait_ready(wait=wait)
        LOGGER.info("Updated config on serving endpoint %r", self)
        return self

    def delete(self, *, missing_ok: bool = False) -> None:
        """Delete this endpoint."""
        from databricks.sdk.errors import NotFound

        LOGGER.debug("Deleting serving endpoint %r", self)
        try:
            self.api.delete(name=self.name)
        except NotFound:
            if not missing_ok:
                raise
            LOGGER.debug("Serving endpoint %r already missing", self)
        self._details = None
        LOGGER.info("Deleted serving endpoint %r", self)

    def wait_ready(self, *, wait: WaitingConfigArg = None) -> "ServingEndpoint":
        """Block until the endpoint finishes updating (or the budget elapses)."""
        wait_cfg = self.service._resolve_wait(wait)
        LOGGER.debug(
            "Waiting for serving endpoint %r to settle (timeout=%ss)",
            self, wait_cfg.timeout,
        )
        info = self.api.wait_get_serving_endpoint_not_updating(
            name=self.name,
            timeout=wait_cfg.timeout_timedelta,
        )
        self._details = info
        return self

    # ------------------------------------------------------------------ #
    # Tags / gateway management
    # ------------------------------------------------------------------ #
    def add_tags(self, tags: Mapping[str, str]) -> "ServingEndpoint":
        """Add or overwrite endpoint tags."""
        self.api.patch(name=self.name, add_tags=_tags_to_list(tags))
        self._details = None
        return self

    def delete_tags(self, keys: Sequence[str]) -> "ServingEndpoint":
        """Delete endpoint tags by key."""
        self.api.patch(name=self.name, delete_tags=list(keys))
        self._details = None
        return self

    def set_ai_gateway(self, ai_gateway: "AiGatewayConfig") -> "ServingEndpoint":
        """Replace the AI Gateway config on a live endpoint."""
        self.api.put_ai_gateway(
            name=self.name,
            fallback_config=getattr(ai_gateway, "fallback_config", None),
            guardrails=getattr(ai_gateway, "guardrails", None),
            inference_table_config=getattr(ai_gateway, "inference_table_config", None),
            rate_limits=getattr(ai_gateway, "rate_limits", None),
            usage_tracking_config=getattr(ai_gateway, "usage_tracking_config", None),
        )
        self._details = None
        return self

    # ------------------------------------------------------------------ #
    # Query data-plane
    # ------------------------------------------------------------------ #
    def query(self, **kwargs: Any) -> "ServingQueryResult":
        """Low-level query passthrough to ``ServingEndpointsAPI.query``.

        Accepts every shape the endpoint task supports — ``messages=`` for
        chat, ``prompt=`` for completions, ``input=``/``inputs=`` for
        embeddings, ``dataframe_records=`` for classic ML — plus the
        sampling knobs (``max_tokens``, ``temperature``, ``n``, ``stop``,
        ``extra_params``). Prefer :meth:`chat` / :meth:`complete` /
        :meth:`embed` for the common LLM paths.
        """
        LOGGER.debug("Querying serving endpoint %r (keys=%s)", self, sorted(kwargs))
        response = self.api.query(name=self.name, **kwargs)
        return ServingQueryResult(endpoint=self, response=response)

    def chat(
        self,
        messages: "Union[str, MessagesLike]",
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        extra_params: Optional[Mapping[str, str]] = None,
    ) -> "ServingQueryResult":
        """Chat-complete against an ``llm/v1/chat`` endpoint.

        ``messages`` is either a bare ``str`` (sent as a single ``user``
        turn) or a sequence of ``{"role": ..., "content": ...}`` mappings
        / :class:`ChatMessage` objects.
        """
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        raw = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        chat_messages: list[ChatMessage] = []
        for m in raw:
            if isinstance(m, ChatMessage):
                chat_messages.append(m)
            else:
                role = m.get("role", "user")
                role = role if isinstance(role, ChatMessageRole) else ChatMessageRole(str(role))
                chat_messages.append(ChatMessage(role=role, content=m["content"]))
        return self.query(
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            stop=list(stop) if stop else None,
            extra_params=dict(extra_params) if extra_params else None,
        )

    def complete(
        self,
        prompt: Union[str, Sequence[str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        extra_params: Optional[Mapping[str, str]] = None,
    ) -> "ServingQueryResult":
        """Text-complete against an ``llm/v1/completions`` endpoint."""
        return self.query(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            stop=list(stop) if stop else None,
            extra_params=dict(extra_params) if extra_params else None,
        )

    def embed(
        self, text: Union[str, Sequence[str]],
    ) -> "ServingQueryResult":
        """Embed text against an ``llm/v1/embeddings`` endpoint."""
        return self.query(input=text)

    # ------------------------------------------------------------------ #
    # Ops / observability
    # ------------------------------------------------------------------ #
    def logs(self, served_model_name: Optional[str] = None) -> "ServerLogsResponse":
        """Tail the model-server logs for a served entity."""
        name = served_model_name or self._first_entity_name()
        return self.api.logs(name=self.name, served_model_name=name)

    def build_logs(self, served_model_name: Optional[str] = None) -> "Any":
        """Fetch the container build logs for a served entity."""
        name = served_model_name or self._first_entity_name()
        return self.api.build_logs(name=self.name, served_model_name=name)

    def metrics(self) -> "ExportMetricsResponse":
        """Export the endpoint's Prometheus metrics."""
        return self.api.export_metrics(name=self.name)

    def openapi(self) -> "GetOpenApiResponse":
        """Fetch the endpoint's OpenAPI schema."""
        return self.api.get_open_api(name=self.name)

    def _first_entity_name(self) -> str:
        names = self.served_entity_names
        if not names or not names[0]:
            raise ValueError(
                f"{self!r} has no served entity to target — pass "
                f"served_model_name explicitly."
            )
        return names[0]


# ---------------------------------------------------------------------------
# Query result
# ---------------------------------------------------------------------------


class ServingQueryResult:
    """Wrapper around a :class:`QueryEndpointResponse`.

    Surfaces the four response shapes a serving endpoint returns —
    chat / completion *choices*, embedding *data*, and classic ML
    *predictions* / *outputs* — through one ergonomic object so callers
    don't reach into the raw SDK body. :attr:`text` is the fast path for
    the common "one chat / completion turn out" case.
    """

    def __init__(self, endpoint: ServingEndpoint, response: "QueryEndpointResponse"):
        self.endpoint = endpoint
        self.response = response

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(endpoint={self.endpoint.name!r}, "
            f"model={self.model!r})"
        )

    # ---- chat / completion ------------------------------------------------ #
    @property
    def choices(self) -> tuple[Any, ...]:
        return tuple(getattr(self.response, "choices", None) or ())

    @property
    def text(self) -> Optional[str]:
        """First choice's content — chat ``message.content`` or completion ``text``."""
        for choice in self.choices:
            msg = getattr(choice, "message", None)
            if msg is not None and getattr(msg, "content", None) is not None:
                return msg.content
            txt = getattr(choice, "text", None)
            if txt is not None:
                return txt
        return None

    @property
    def message(self) -> Optional[dict[str, Any]]:
        """First choice's chat message as ``{"role": ..., "content": ...}``."""
        for choice in self.choices:
            msg = getattr(choice, "message", None)
            if msg is not None:
                role = getattr(msg, "role", None)
                return {
                    "role": role.value if role is not None and hasattr(role, "value") else role,
                    "content": getattr(msg, "content", None),
                }
        return None

    @property
    def texts(self) -> list[str]:
        """Every choice's text (for ``n`` > 1 sampling)."""
        out = []
        for choice in self.choices:
            msg = getattr(choice, "message", None)
            if msg is not None and getattr(msg, "content", None) is not None:
                out.append(msg.content)
            elif getattr(choice, "text", None) is not None:
                out.append(choice.text)
        return out

    # ---- embeddings ------------------------------------------------------- #
    @property
    def embeddings(self) -> list[list[float]]:
        """Embedding vectors from an ``llm/v1/embeddings`` response."""
        data = getattr(self.response, "data", None) or []
        return [list(getattr(e, "embedding", None) or []) for e in data]

    @property
    def embedding(self) -> Optional[list[float]]:
        """First embedding vector, or ``None``."""
        vecs = self.embeddings
        return vecs[0] if vecs else None

    # ---- classic ML ------------------------------------------------------- #
    @property
    def predictions(self) -> Any:
        return getattr(self.response, "predictions", None)

    @property
    def outputs(self) -> Any:
        return getattr(self.response, "outputs", None)

    # ---- metadata --------------------------------------------------------- #
    @property
    def usage(self) -> Optional[dict[str, Any]]:
        u = getattr(self.response, "usage", None)
        return u.as_dict() if u is not None and hasattr(u, "as_dict") else u

    @property
    def model(self) -> Optional[str]:
        return getattr(self.response, "model", None)

    @property
    def served_model_name(self) -> Optional[str]:
        return getattr(self.response, "served_model_name", None)

    @property
    def id(self) -> Optional[str]:
        return getattr(self.response, "id", None)

    def to_dict(self) -> dict[str, Any]:
        """Return the full response as a plain dict."""
        return self.response.as_dict() if hasattr(self.response, "as_dict") else dict(self.response)
