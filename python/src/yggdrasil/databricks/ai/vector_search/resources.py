"""Vector Search resources and default configuration.

Mirrors the Databricks Vector Search hierarchy as Yggdrasil resources:

- :class:`VectorSearchDefaults` — frozen config attached to the
  :class:`VectorSearch` service (``service.defaults``). Carries the
  workspace's default endpoint name, the endpoint / index / pipeline
  type to use when callers do not specify one, the polling budget for
  long-running create / sync operations, and the optional
  ``embedding_model_endpoint_name`` used to embed source columns on the
  managed-embedding path.
- :class:`VectorSearchEndpoint` — a single vector-search *endpoint*
  (the compute that backs one or more indexes). Exposes
  :meth:`create` / :meth:`ensure_created` / :meth:`delete` /
  :meth:`wait_online` plus the cached :attr:`infos` lookup.
- :class:`VectorSearchIndex` — a single vector-search *index*. Supports
  both the ``DELTA_SYNC`` shape (driven by a UC Delta source table) and
  the ``DIRECT_ACCESS`` shape (the caller upserts / deletes rows
  directly). Exposes :meth:`query` for similarity / hybrid / full-text
  search, :meth:`sync` for delta-sync refresh, and :meth:`upsert` /
  :meth:`delete_rows` / :meth:`scan` for direct-access maintenance.
- :class:`VectorSearchQueryResult` — Arrow-typed wrapper around the
  ``QueryVectorIndexResponse`` body. Carries pagination state and
  materialises the response as Arrow / Polars / pandas via the
  registered cast paths.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional, Sequence, Union

from yggdrasil.databricks.resource import DatabricksResource
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg
from yggdrasil.io.url import URL

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from databricks.sdk.service.vectorsearch import (
        ColumnInfo,
        EmbeddingSourceColumn,
        EmbeddingVectorColumn,
        EndpointInfo,
        QueryVectorIndexResponse,
        RerankerConfig,
        VectorIndex,
        VectorSearchEndpointsAPI,
        VectorSearchIndexesAPI,
    )

    from .service import VectorSearch


__all__ = [
    "DEFAULT_VS_WAIT",
    "VectorSearchDefaults",
    "VectorSearchEndpoint",
    "VectorSearchIndex",
    "VectorSearchQueryResult",
]


LOGGER = logging.getLogger(__name__)

#: Default wait budget for vector-search create / wait-online / sync
#: operations. Matches the SDK's own
#: ``wait_get_endpoint_vector_search_endpoint_online`` default of 20
#: minutes — endpoint provisioning routinely takes 5-10 minutes and
#: a fresh delta-sync index can take longer than that for the first
#: pipeline run.
DEFAULT_VS_WAIT: WaitingConfig = WaitingConfig(timeout=1200.0, interval=5.0)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorSearchDefaults:
    """Default configuration for :class:`VectorSearch`.

    Set once on the service and every subsequent call inherits these
    values unless overridden inline::

        from dataclasses import replace
        client.ai.vector_search.defaults = replace(
            client.ai.vector_search.defaults,
            endpoint_name="rag-endpoint",
            embedding_model_endpoint_name="databricks-bge-large-en",
        )

    Attributes
    ----------
    endpoint_name
        Default endpoint name used by :meth:`VectorSearch.index` /
        :meth:`VectorSearch.create_delta_sync_index` /
        :meth:`VectorSearch.create_direct_access_index` when none is
        passed inline.
    endpoint_type
        Endpoint type used by :meth:`VectorSearchEndpoint.create` /
        :meth:`VectorSearchEndpoint.ensure_created` when the caller
        does not specify one. ``"STANDARD"`` (cheap, shared) by
        default; ``"STORAGE_OPTIMIZED"`` is the larger-billing tier
        for cold storage workloads. Accepts the SDK enum or its
        string name.
    index_type
        Default :class:`VectorIndexType` for :meth:`VectorSearchIndex.create`.
        ``"DELTA_SYNC"`` is the most common path — the index is
        driven by a UC Delta source table; ``"DIRECT_ACCESS"`` lets
        the caller upsert / delete rows directly.
    pipeline_type
        Default :class:`PipelineType` for delta-sync indexes.
        ``"TRIGGERED"`` is cheaper (sync on demand);
        ``"CONTINUOUS"`` keeps the index in lockstep with the source
        Delta table.
    embedding_model_endpoint_name
        Name of a serving endpoint that produces embeddings (e.g.
        ``"databricks-bge-large-en"``). When set, delta-sync indexes
        created via the managed-embedding shape route source columns
        through this endpoint automatically — callers no longer have
        to pre-compute and store the embedding vector column.
    wait
        :class:`~yggdrasil.dataclasses.WaitingConfig` carrying the
        budget for long-running endpoint / index operations
        (``wait.timeout``) and the polling cadence (``wait.interval``).
        Defaults to :data:`DEFAULT_VS_WAIT` (20 minutes / 5 seconds).
        Override per-call by passing ``wait=`` — anything
        :meth:`WaitingConfig.from_` accepts works (seconds, timedelta,
        deadline, dict, full ``WaitingConfig``).
    """

    endpoint_name: Optional[str] = None
    endpoint_type: str = "STANDARD"
    index_type: str = "DELTA_SYNC"
    pipeline_type: str = "TRIGGERED"
    embedding_model_endpoint_name: Optional[str] = None
    wait: WaitingConfig = DEFAULT_VS_WAIT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_endpoint_type(value: Any) -> "Any":
    from databricks.sdk.service.vectorsearch import EndpointType

    if isinstance(value, EndpointType):
        return value
    if value is None:
        return EndpointType.STANDARD
    return EndpointType(str(value).upper())


def _coerce_pipeline_type(value: Any) -> "Any":
    from databricks.sdk.service.vectorsearch import PipelineType

    if isinstance(value, PipelineType):
        return value
    if value is None:
        return PipelineType.TRIGGERED
    return PipelineType(str(value).upper())


# ---------------------------------------------------------------------------
# VectorSearchEndpoint
# ---------------------------------------------------------------------------


class VectorSearchEndpoint(DatabricksResource):
    """A Databricks Vector Search endpoint.

    Endpoints are the compute layer — one endpoint hosts one or more
    :class:`VectorSearchIndex` instances. Provisioning is asynchronous;
    use :meth:`wait_online` (or pass ``wait=True`` to :meth:`create` /
    :meth:`ensure_created`) when the next step needs the endpoint to be
    serving queries.
    """

    def __init__(
        self,
        service: "VectorSearch",
        endpoint_name: str,
        *,
        details: "Optional[EndpointInfo]" = None,
    ):
        super().__init__(service=service)
        self.service: "VectorSearch" = service
        self.endpoint_name = endpoint_name
        self._details: "Optional[EndpointInfo]" = details

    def __repr__(self) -> str:
        return f"{type(self).__name__}(endpoint_name={self.endpoint_name!r})"

    @property
    def api(self) -> "VectorSearchEndpointsAPI":
        return self.client.workspace_client().vector_search_endpoints

    @property
    def explore_url(self) -> URL:
        return URL.from_str(
            f"{self.client.base_url.to_string().rstrip('/')}"
            f"/compute/vector-search/{self.endpoint_name}"
        )

    # ------------------------------------------------------------------ #
    # Identity / cached infos
    # ------------------------------------------------------------------ #
    @property
    def infos(self) -> "EndpointInfo":
        """Cached :class:`EndpointInfo` — fetched on first access."""
        cached = self._details
        if cached is not None:
            return cached
        LOGGER.debug("Fetching vector-search endpoint %r from remote", self)
        info = self.api.get_endpoint(endpoint_name=self.endpoint_name)
        self._details = info
        return info

    def refresh(self) -> "VectorSearchEndpoint":
        """Re-fetch :attr:`infos` from the API."""
        LOGGER.debug("Refreshing vector-search endpoint %r", self)
        self._details = self.api.get_endpoint(endpoint_name=self.endpoint_name)
        return self

    @property
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
        """Lifecycle state from the endpoint status (``ONLINE`` / ``PROVISIONING`` / …)."""
        status = getattr(self.infos, "endpoint_status", None)
        state = getattr(status, "state", None) if status is not None else None
        return state.value if state is not None and hasattr(state, "value") else state

    @property
    def is_online(self) -> bool:
        from databricks.sdk.service.vectorsearch import EndpointStatusState

        status = getattr(self.infos, "endpoint_status", None)
        state = getattr(status, "state", None) if status is not None else None
        return state == EndpointStatusState.ONLINE

    @property
    def endpoint_type(self) -> Optional[str]:
        et = getattr(self.infos, "endpoint_type", None)
        return et.value if et is not None and hasattr(et, "value") else et

    @property
    def num_indexes(self) -> Optional[int]:
        return getattr(self.infos, "num_indexes", None)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def create(
        self,
        *,
        endpoint_type: Any = None,
        budget_policy_id: Optional[str] = None,
        target_qps: Optional[int] = None,
        wait: WaitingConfigArg = None,
        if_not_exists: bool = True,
    ) -> "VectorSearchEndpoint":
        """Create this endpoint.

        Parameters
        ----------
        endpoint_type
            Endpoint type. Accepts the SDK enum or its string name.
            Defaults to :attr:`VectorSearchDefaults.endpoint_type`.
        budget_policy_id
            Optional budget-policy id to attribute usage to.
        target_qps
            Optional target queries-per-second the endpoint should
            scale to. Only applicable to ``STANDARD`` endpoints.
        wait
            Per-call wait budget. ``None`` (the default) returns as
            soon as the SDK accepts the create request — endpoint
            provisioning typically takes 5-10 minutes and most
            callers don't want to block. Pass ``True`` to wait with
            :attr:`VectorSearchDefaults.wait`, or anything
            :meth:`WaitingConfig.from_` accepts (seconds, timedelta,
            deadline, dict, full ``WaitingConfig``).
        if_not_exists
            When ``True`` (the default), an existing endpoint with
            the same name is treated as success and the cached
            :attr:`infos` is refreshed instead of raising.
        """
        from databricks.sdk.errors import AlreadyExists, DatabricksError

        endpoint_type_value = _coerce_endpoint_type(
            endpoint_type if endpoint_type is not None else self.service.defaults.endpoint_type,
        )

        LOGGER.debug(
            "Creating vector-search endpoint %r (endpoint_type=%s, target_qps=%s)",
            self, endpoint_type_value.value, target_qps,
        )
        try:
            self._details = self.api.create_endpoint(
                name=self.endpoint_name,
                endpoint_type=endpoint_type_value,
                budget_policy_id=budget_policy_id,
                target_qps=target_qps,
            ).response
        except AlreadyExists:
            if not if_not_exists:
                raise
            LOGGER.debug(
                "Vector-search endpoint %r already exists — refreshing infos",
                self,
            )
            self.refresh()
        except DatabricksError as exc:
            if if_not_exists and "already exists" in str(exc).lower():
                LOGGER.debug(
                    "Vector-search endpoint %r already exists — refreshing infos",
                    self,
                )
                self.refresh()
            else:
                raise

        if wait is not None and wait is not False:
            self.wait_online(wait=wait)
        LOGGER.info("Created vector-search endpoint %r", self)
        return self

    def ensure_created(
        self,
        *,
        endpoint_type: Any = None,
        budget_policy_id: Optional[str] = None,
        target_qps: Optional[int] = None,
        wait: WaitingConfigArg = None,
    ) -> "VectorSearchEndpoint":
        """Create this endpoint when missing, otherwise return ``self``."""
        if self.exists:
            return self
        return self.create(
            endpoint_type=endpoint_type,
            budget_policy_id=budget_policy_id,
            target_qps=target_qps,
            wait=wait,
            if_not_exists=True,
        )

    def delete(self, *, missing_ok: bool = False) -> None:
        """Delete this endpoint.

        Parameters
        ----------
        missing_ok
            When ``True``, a missing endpoint is treated as success.
        """
        from databricks.sdk.errors import NotFound

        LOGGER.debug("Deleting vector-search endpoint %r", self)
        try:
            self.api.delete_endpoint(endpoint_name=self.endpoint_name)
        except NotFound:
            if not missing_ok:
                raise
            LOGGER.debug("Vector-search endpoint %r already missing", self)
        self._details = None
        LOGGER.info("Deleted vector-search endpoint %r", self)

    def wait_online(self, *, wait: WaitingConfigArg = None) -> "VectorSearchEndpoint":
        """Block until the endpoint reaches ``ONLINE`` (or the budget elapses)."""
        wait_cfg = self.service._resolve_wait(wait)
        LOGGER.debug(
            "Waiting for vector-search endpoint %r to become ONLINE (timeout=%ss)",
            self, wait_cfg.timeout,
        )
        info = self.api.wait_get_endpoint_vector_search_endpoint_online(
            endpoint_name=self.endpoint_name,
            timeout=wait_cfg.timeout_timedelta,
        )
        self._details = info
        return self

    # ------------------------------------------------------------------ #
    # Indexes
    # ------------------------------------------------------------------ #
    def index(self, index_name: str) -> "VectorSearchIndex":
        """Return a :class:`VectorSearchIndex` handle bound to this endpoint."""
        return VectorSearchIndex(
            service=self.service,
            index_name=index_name,
            endpoint_name=self.endpoint_name,
        )

    def indexes(self) -> Iterator["VectorSearchIndex"]:
        """Iterate over indexes hosted on this endpoint."""
        return self.service.list_indexes(endpoint_name=self.endpoint_name)


# ---------------------------------------------------------------------------
# VectorSearchIndex
# ---------------------------------------------------------------------------


class VectorSearchIndex(DatabricksResource):
    """A Databricks Vector Search index.

    Indexes are UC-governed three-part identifiers (``catalog.schema.name``);
    callers refer to them by that full name. The :attr:`endpoint_name`
    is required for create / sync / pagination operations but is read
    back from :attr:`infos` for queries — the SDK only needs the
    index name for :meth:`query`.

    Two index *types* are supported:

    - ``DELTA_SYNC`` — driven by a UC Delta source table. Create via
      :meth:`create_delta_sync` (the source table's primary key is the
      index PK, and the embedding either comes from a precomputed
      vector column or via a managed-embedding endpoint).
    - ``DIRECT_ACCESS`` — caller-managed rows. Create via
      :meth:`create_direct_access`, then :meth:`upsert` / :meth:`delete_rows`.
    """

    def __init__(
        self,
        service: "VectorSearch",
        index_name: str,
        *,
        endpoint_name: Optional[str] = None,
        details: "Optional[VectorIndex]" = None,
    ):
        super().__init__(service=service)
        self.service: "VectorSearch" = service
        self.index_name = index_name
        # ``endpoint_name`` may be ``None`` for handles built off the
        # service's default endpoint; we resolve it lazily on
        # operations that need it.
        self._endpoint_name = endpoint_name
        self._details: "Optional[VectorIndex]" = details

    def __repr__(self) -> str:
        ep = self.endpoint_name
        if ep:
            return (
                f"{type(self).__name__}(index_name={self.index_name!r}, "
                f"endpoint_name={ep!r})"
            )
        return f"{type(self).__name__}(index_name={self.index_name!r})"

    @property
    def api(self) -> "VectorSearchIndexesAPI":
        return self.client.workspace_client().vector_search_indexes

    @property
    def endpoint_name(self) -> Optional[str]:
        """Endpoint this index is hosted on.

        Resolves from the most informative source available: an explicit
        constructor arg, the cached :attr:`_details`, then the service
        default. The infos-roundtrip path is left for the explicit
        :meth:`refresh` so attribute access stays free of API calls.
        """
        if self._endpoint_name:
            return self._endpoint_name
        if self._details is not None and self._details.endpoint_name:
            self._endpoint_name = self._details.endpoint_name
            return self._endpoint_name
        return self.service.defaults.endpoint_name

    @property
    def explore_url(self) -> URL:
        parts = [p for p in self.index_name.split(".") if p]
        path = "/".join(parts) if parts else self.index_name
        return URL.from_str(
            f"{self.client.base_url.to_string().rstrip('/')}/explore/data/{path}"
        )

    # ------------------------------------------------------------------ #
    # Identity / cached infos
    # ------------------------------------------------------------------ #
    @property
    def infos(self) -> "VectorIndex":
        """Cached :class:`VectorIndex` — fetched on first access."""
        cached = self._details
        if cached is not None:
            return cached
        LOGGER.debug("Fetching vector-search index %r from remote", self)
        info = self.api.get_index(index_name=self.index_name)
        self._details = info
        self._endpoint_name = self._endpoint_name or info.endpoint_name
        return info

    def refresh(self) -> "VectorSearchIndex":
        """Re-fetch :attr:`infos` from the API."""
        LOGGER.debug("Refreshing vector-search index %r", self)
        info = self.api.get_index(index_name=self.index_name)
        self._details = info
        self._endpoint_name = info.endpoint_name or self._endpoint_name
        return self

    @property
    def exists(self) -> bool:
        from databricks.sdk.errors import NotFound

        try:
            _ = self.infos
            return True
        except NotFound:
            return False

    @property
    def is_ready(self) -> bool:
        status = getattr(self.infos, "status", None)
        return bool(getattr(status, "ready", False)) if status is not None else False

    @property
    def indexed_row_count(self) -> Optional[int]:
        status = getattr(self.infos, "status", None)
        return getattr(status, "indexed_row_count", None) if status is not None else None

    @property
    def primary_key(self) -> Optional[str]:
        return getattr(self.infos, "primary_key", None)

    @property
    def index_type(self) -> Optional[str]:
        it = getattr(self.infos, "index_type", None)
        return it.value if it is not None and hasattr(it, "value") else it

    @property
    def index_subtype(self) -> Optional[str]:
        sub = getattr(self.infos, "index_subtype", None)
        return sub.value if sub is not None and hasattr(sub, "value") else sub

    @property
    def source_table(self) -> Optional[str]:
        spec = getattr(self.infos, "delta_sync_index_spec", None)
        return getattr(spec, "source_table", None) if spec is not None else None

    # ------------------------------------------------------------------ #
    # Lifecycle — create
    # ------------------------------------------------------------------ #
    def create_delta_sync(
        self,
        *,
        source_table: str,
        primary_key: str,
        embedding_source_columns: Optional[Sequence[Union[str, "EmbeddingSourceColumn"]]] = None,
        embedding_vector_columns: Optional[Sequence[Union[Mapping[str, Any], "EmbeddingVectorColumn"]]] = None,
        embedding_model_endpoint_name: Optional[str] = None,
        pipeline_type: Any = None,
        columns_to_sync: Optional[Sequence[str]] = None,
        embedding_writeback_table: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        wait: WaitingConfigArg = None,
        if_not_exists: bool = True,
    ) -> "VectorSearchIndex":
        """Create a ``DELTA_SYNC`` index backed by a UC Delta source table.

        Pick exactly one embedding shape:

        - **Managed embeddings** — pass ``embedding_source_columns`` (a
          list of column names) and either ``embedding_model_endpoint_name``
          or the service default. The endpoint embeds each source value
          automatically.
        - **Self-managed embeddings** — pass
          ``embedding_vector_columns`` (each entry being a mapping
          ``{"name": "<col>", "embedding_dimension": <int>}`` or a
          fully-built :class:`EmbeddingVectorColumn`). The vector
          column must already exist on the source table.
        """
        from databricks.sdk.errors import AlreadyExists, DatabricksError
        from databricks.sdk.service.vectorsearch import (
            DeltaSyncVectorIndexSpecRequest,
            EmbeddingSourceColumn,
            EmbeddingVectorColumn,
            VectorIndexType,
        )

        ep_name = endpoint_name or self.endpoint_name
        if not ep_name:
            raise ValueError(
                f"Cannot create {self!r}: no endpoint_name given and no service "
                f"default set. Pass endpoint_name=... or set "
                f"VectorSearch.defaults.endpoint_name."
            )

        if not embedding_source_columns and not embedding_vector_columns:
            raise ValueError(
                f"Cannot create {self!r}: pass embedding_source_columns "
                f"(managed embeddings) or embedding_vector_columns "
                f"(self-managed embeddings). Got neither."
            )
        if embedding_source_columns and embedding_vector_columns:
            raise ValueError(
                f"Cannot create {self!r}: pass embedding_source_columns OR "
                f"embedding_vector_columns, not both."
            )

        source_cols: Optional[list[EmbeddingSourceColumn]] = None
        if embedding_source_columns:
            model_endpoint = (
                embedding_model_endpoint_name
                or self.service.defaults.embedding_model_endpoint_name
            )
            if not model_endpoint:
                raise ValueError(
                    "embedding_source_columns requires "
                    "embedding_model_endpoint_name (or the service default) — "
                    "managed embeddings need a serving endpoint to call."
                )
            source_cols = []
            for col in embedding_source_columns:
                if isinstance(col, EmbeddingSourceColumn):
                    source_cols.append(col)
                else:
                    source_cols.append(EmbeddingSourceColumn(
                        name=str(col),
                        embedding_model_endpoint_name=model_endpoint,
                    ))

        vector_cols: Optional[list[EmbeddingVectorColumn]] = None
        if embedding_vector_columns:
            vector_cols = []
            for col in embedding_vector_columns:
                if isinstance(col, EmbeddingVectorColumn):
                    vector_cols.append(col)
                elif isinstance(col, Mapping):
                    vector_cols.append(EmbeddingVectorColumn(
                        name=col["name"],
                        embedding_dimension=col["embedding_dimension"],
                    ))
                else:
                    raise TypeError(
                        f"embedding_vector_columns entries must be "
                        f"EmbeddingVectorColumn or mappings with "
                        f"'name' + 'embedding_dimension'; got {type(col).__name__}"
                    )

        pipeline = _coerce_pipeline_type(
            pipeline_type if pipeline_type is not None else self.service.defaults.pipeline_type,
        )
        spec = DeltaSyncVectorIndexSpecRequest(
            source_table=source_table,
            pipeline_type=pipeline,
            embedding_source_columns=source_cols,
            embedding_vector_columns=vector_cols,
            columns_to_sync=list(columns_to_sync) if columns_to_sync else None,
            embedding_writeback_table=embedding_writeback_table,
        )

        LOGGER.debug(
            "Creating vector-search index %r (endpoint=%s, source=%s, pipeline=%s)",
            self, ep_name, source_table, pipeline.value,
        )
        try:
            info = self.api.create_index(
                name=self.index_name,
                endpoint_name=ep_name,
                primary_key=primary_key,
                index_type=VectorIndexType.DELTA_SYNC,
                delta_sync_index_spec=spec,
            )
            self._details = info
            self._endpoint_name = info.endpoint_name or ep_name
        except AlreadyExists:
            if not if_not_exists:
                raise
            LOGGER.debug("Vector-search index %r already exists — refreshing", self)
            self.refresh()
        except DatabricksError as exc:
            if if_not_exists and "already exists" in str(exc).lower():
                LOGGER.debug("Vector-search index %r already exists — refreshing", self)
                self.refresh()
            else:
                raise

        if wait is not None and wait is not False:
            self.wait_ready(wait=wait)
        LOGGER.info("Created vector-search index %r (endpoint=%s)", self, ep_name)
        return self

    def create_direct_access(
        self,
        *,
        primary_key: str,
        schema_json: str,
        embedding_source_columns: Optional[Sequence[Union[str, "EmbeddingSourceColumn"]]] = None,
        embedding_vector_columns: Optional[Sequence[Union[Mapping[str, Any], "EmbeddingVectorColumn"]]] = None,
        embedding_model_endpoint_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        wait: WaitingConfigArg = None,
        if_not_exists: bool = True,
    ) -> "VectorSearchIndex":
        """Create a ``DIRECT_ACCESS`` index that the caller upserts into.

        ``schema_json`` is a JSON string describing the row schema (see
        the Databricks Vector Search docs for the shape — typically
        ``{"<col>": "<type>", ...}``). Provide either
        ``embedding_source_columns`` (managed embeddings) or
        ``embedding_vector_columns`` (self-managed) the same way as
        :meth:`create_delta_sync`.
        """
        from databricks.sdk.errors import AlreadyExists, DatabricksError
        from databricks.sdk.service.vectorsearch import (
            DirectAccessVectorIndexSpec,
            EmbeddingSourceColumn,
            EmbeddingVectorColumn,
            VectorIndexType,
        )

        ep_name = endpoint_name or self.endpoint_name
        if not ep_name:
            raise ValueError(
                f"Cannot create {self!r}: no endpoint_name given and no service "
                f"default set. Pass endpoint_name=... or set "
                f"VectorSearch.defaults.endpoint_name."
            )

        source_cols: Optional[list[EmbeddingSourceColumn]] = None
        if embedding_source_columns:
            model_endpoint = (
                embedding_model_endpoint_name
                or self.service.defaults.embedding_model_endpoint_name
            )
            if not model_endpoint:
                raise ValueError(
                    "embedding_source_columns requires "
                    "embedding_model_endpoint_name (or the service default) — "
                    "managed embeddings need a serving endpoint to call."
                )
            source_cols = []
            for col in embedding_source_columns:
                if isinstance(col, EmbeddingSourceColumn):
                    source_cols.append(col)
                else:
                    source_cols.append(EmbeddingSourceColumn(
                        name=str(col),
                        embedding_model_endpoint_name=model_endpoint,
                    ))

        vector_cols: Optional[list[EmbeddingVectorColumn]] = None
        if embedding_vector_columns:
            vector_cols = []
            for col in embedding_vector_columns:
                if isinstance(col, EmbeddingVectorColumn):
                    vector_cols.append(col)
                elif isinstance(col, Mapping):
                    vector_cols.append(EmbeddingVectorColumn(
                        name=col["name"],
                        embedding_dimension=col["embedding_dimension"],
                    ))
                else:
                    raise TypeError(
                        f"embedding_vector_columns entries must be "
                        f"EmbeddingVectorColumn or mappings with "
                        f"'name' + 'embedding_dimension'; got {type(col).__name__}"
                    )

        if not source_cols and not vector_cols:
            raise ValueError(
                f"Cannot create {self!r}: pass embedding_source_columns or "
                f"embedding_vector_columns. Got neither."
            )

        spec = DirectAccessVectorIndexSpec(
            schema_json=schema_json,
            embedding_source_columns=source_cols,
            embedding_vector_columns=vector_cols,
        )

        LOGGER.debug(
            "Creating direct-access vector-search index %r (endpoint=%s)",
            self, ep_name,
        )
        try:
            info = self.api.create_index(
                name=self.index_name,
                endpoint_name=ep_name,
                primary_key=primary_key,
                index_type=VectorIndexType.DIRECT_ACCESS,
                direct_access_index_spec=spec,
            )
            self._details = info
            self._endpoint_name = info.endpoint_name or ep_name
        except AlreadyExists:
            if not if_not_exists:
                raise
            LOGGER.debug("Vector-search index %r already exists — refreshing", self)
            self.refresh()
        except DatabricksError as exc:
            if if_not_exists and "already exists" in str(exc).lower():
                LOGGER.debug("Vector-search index %r already exists — refreshing", self)
                self.refresh()
            else:
                raise

        if wait is not None and wait is not False:
            self.wait_ready(wait=wait)
        LOGGER.info("Created vector-search index %r (endpoint=%s)", self, ep_name)
        return self

    def delete(self, *, missing_ok: bool = False) -> None:
        """Delete this index."""
        from databricks.sdk.errors import NotFound

        LOGGER.debug("Deleting vector-search index %r", self)
        try:
            self.api.delete_index(index_name=self.index_name)
        except NotFound:
            if not missing_ok:
                raise
            LOGGER.debug("Vector-search index %r already missing", self)
        self._details = None
        LOGGER.info("Deleted vector-search index %r", self)

    # ------------------------------------------------------------------ #
    # Sync / wait
    # ------------------------------------------------------------------ #
    def sync(self) -> "VectorSearchIndex":
        """Trigger a sync for a ``DELTA_SYNC`` index (no-op for direct-access)."""
        LOGGER.debug("Triggering sync on vector-search index %r", self)
        self.api.sync_index(index_name=self.index_name)
        LOGGER.info("Triggered sync on vector-search index %r", self)
        return self

    def wait_ready(self, *, wait: WaitingConfigArg = None) -> "VectorSearchIndex":
        """Poll :attr:`infos` until ``status.ready`` is ``True`` (or budget elapses)."""
        wait_cfg = self.service._resolve_wait(wait)
        deadline = time.monotonic() + wait_cfg.timeout
        interval = wait_cfg.interval

        LOGGER.debug(
            "Waiting for vector-search index %r to be ready (timeout=%ss, interval=%ss)",
            self, wait_cfg.timeout, interval,
        )
        while True:
            self.refresh()
            if self.is_ready:
                return self
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Vector-search index {self.index_name!r} did not become "
                    f"ready within {wait_cfg.timeout}s. Last status: "
                    f"{getattr(self.infos, 'status', None)!r}"
                )
            time.sleep(min(interval, remaining))

    # ------------------------------------------------------------------ #
    # Direct-access data plane
    # ------------------------------------------------------------------ #
    def upsert(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        """Upsert rows into a ``DIRECT_ACCESS`` index.

        ``rows`` is a sequence of dicts (one per row) matching the
        ``schema_json`` passed at create time. The vector column is
        either pre-computed (self-managed embeddings) or filled in by
        the managed embedding endpoint.
        """
        from yggdrasil.pickle import json as ygg_json

        if not rows:
            LOGGER.debug("Skipping upsert into %r: empty rows", self)
            return None

        payload = ygg_json.dumps(list(rows), to_bytes=False)
        LOGGER.debug("Upserting %d rows into vector-search index %r", len(rows), self)
        response = self.api.upsert_data_vector_index(
            index_name=self.index_name,
            inputs_json=payload,
        )
        LOGGER.info("Upserted %d rows into vector-search index %r", len(rows), self)
        return response

    def delete_rows(self, primary_keys: Sequence[str]) -> Any:
        """Delete rows by primary key from a ``DIRECT_ACCESS`` index."""
        keys = list(primary_keys)
        if not keys:
            LOGGER.debug("Skipping delete on %r: empty primary_keys", self)
            return None
        LOGGER.debug("Deleting %d rows from vector-search index %r", len(keys), self)
        response = self.api.delete_data_vector_index(
            index_name=self.index_name,
            primary_keys=keys,
        )
        LOGGER.info("Deleted %d rows from vector-search index %r", len(keys), self)
        return response

    def scan(
        self,
        *,
        num_results: Optional[int] = None,
        last_primary_key: Optional[str] = None,
    ) -> Any:
        """Scan rows (direct-access maintenance) returning the raw response."""
        LOGGER.debug(
            "Scanning vector-search index %r (num_results=%s, last_pk=%s)",
            self, num_results, last_primary_key,
        )
        return self.api.scan_index(
            index_name=self.index_name,
            num_results=num_results,
            last_primary_key=last_primary_key,
        )

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def query(
        self,
        *,
        columns: Sequence[str],
        query_text: Optional[str] = None,
        query_vector: Optional[Sequence[float]] = None,
        num_results: Optional[int] = 10,
        filters: Optional[Union[str, Mapping[str, Any]]] = None,
        query_type: Optional[str] = None,
        columns_to_rerank: Optional[Sequence[str]] = None,
        reranker: "Optional[RerankerConfig]" = None,
        score_threshold: Optional[float] = None,
    ) -> "VectorSearchQueryResult":
        """Run a similarity / hybrid / full-text query against this index.

        Parameters
        ----------
        columns
            Result columns to return. The score column (``__db_score``)
            is appended automatically by Databricks.
        query_text
            Natural-language query — embedded by the index's managed
            embedding endpoint. Mutually exclusive with ``query_vector``.
        query_vector
            Pre-computed embedding vector. Mutually exclusive with
            ``query_text``.
        num_results
            Maximum number of results to return.
        filters
            Optional metadata filter. Either a JSON string (the SDK's
            ``filters_json``) or a mapping that is serialised to JSON
            via :mod:`yggdrasil.pickle.json`.
        query_type
            ``"ANN"`` (default), ``"HYBRID"``, or ``"FULL_TEXT"``.
        columns_to_rerank
            Columns whose values should be considered by the reranker.
        reranker
            Optional :class:`RerankerConfig`.
        score_threshold
            Optional minimum score; rows below the threshold are
            dropped server-side.
        """
        from yggdrasil.pickle import json as ygg_json

        if query_text is None and query_vector is None:
            raise ValueError(
                f"Cannot query {self!r}: pass query_text or query_vector. Got neither."
            )
        if query_text is not None and query_vector is not None:
            raise ValueError(
                f"Cannot query {self!r}: pass query_text OR query_vector, not both."
            )

        filters_json: Optional[str]
        if filters is None or isinstance(filters, str):
            filters_json = filters
        else:
            filters_json = ygg_json.dumps(dict(filters), to_bytes=False)

        LOGGER.debug(
            "Querying vector-search index %r (num_results=%s, query_type=%s, text=%s, vector_len=%s)",
            self, num_results, query_type,
            (query_text[:40] + "…") if query_text and len(query_text) > 40 else query_text,
            len(query_vector) if query_vector is not None else None,
        )
        response = self.api.query_index(
            index_name=self.index_name,
            columns=list(columns),
            query_text=query_text,
            query_vector=list(query_vector) if query_vector is not None else None,
            num_results=num_results,
            filters_json=filters_json,
            query_type=query_type,
            columns_to_rerank=list(columns_to_rerank) if columns_to_rerank else None,
            reranker=reranker,
            score_threshold=score_threshold,
        )
        return VectorSearchQueryResult(index=self, response=response)


# ---------------------------------------------------------------------------
# Query result
# ---------------------------------------------------------------------------

# ``data_array`` cells arrive as JSON-encoded strings (the wire shape is
# always ``List[List[str]]`` even for numeric / vector columns). Map the
# subset of Databricks type_text tokens vector-search actually emits to
# Arrow types so :meth:`to_arrow_table` can cast in one C-bridge hop.
_PA_PRIMITIVE_BY_TYPE_TEXT = {
    "string": "string",
    # ``pa.bool_`` is the canonical builder; ``pa.bool`` doesn't exist.
    "boolean": "bool_",
    "int": "int32",
    "integer": "int32",
    "bigint": "int64",
    "long": "int64",
    "smallint": "int16",
    "short": "int16",
    "tinyint": "int8",
    "byte": "int8",
    "float": "float32",
    "double": "float64",
    "binary": "binary",
}


def _resolve_arrow_type(type_text: Optional[str]) -> "Any":
    """Translate a Databricks ``type_text`` to a :class:`pyarrow.DataType`.

    Falls back to ``pa.string()`` for anything we do not recognise — the
    cell already arrived as a string, so leaving it as such preserves
    every byte the API returned (timestamps, complex maps, nested
    arrays the SDK serialised as JSON).
    """
    import pyarrow as pa

    if not type_text:
        return pa.string()
    text = type_text.strip().lower()
    primitive = _PA_PRIMITIVE_BY_TYPE_TEXT.get(text)
    if primitive is not None:
        return getattr(pa, primitive)()
    if text.startswith("decimal"):
        rest = text[len("decimal"):].strip()
        if rest.startswith("(") and rest.endswith(")"):
            try:
                precision, scale = (int(p.strip()) for p in rest[1:-1].split(","))
                return pa.decimal128(precision, scale)
            except (ValueError, TypeError):
                return pa.string()
        return pa.string()
    if text.startswith(("array<float", "array<double", "array<int", "array<long")):
        inner = text[len("array<"):-1].strip()
        return pa.list_(_resolve_arrow_type(inner))
    return pa.string()


class VectorSearchQueryResult:
    """Wrapper around a :class:`QueryVectorIndexResponse`.

    Carries the column manifest, the raw row data (as the API returned
    it — :class:`list` of :class:`list` of :class:`str`), and the
    pagination token. Materialises the result as Arrow / Polars /
    pandas through the registered cast paths.
    """

    def __init__(
        self,
        index: VectorSearchIndex,
        response: "QueryVectorIndexResponse",
    ):
        self.index = index
        self.response = response

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(index={self.index.index_name!r}, "
            f"row_count={self.row_count}, next_page_token={self.next_page_token!r})"
        )

    # ------------------------------------------------------------------ #
    # Raw accessors
    # ------------------------------------------------------------------ #
    @property
    def manifest(self) -> "Any":
        return self.response.manifest

    @property
    def columns(self) -> tuple["ColumnInfo", ...]:
        manifest = self.response.manifest
        cols = getattr(manifest, "columns", None) if manifest is not None else None
        return tuple(cols or ())

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name or "" for c in self.columns)

    @property
    def data_array(self) -> list[list[str]]:
        result = self.response.result
        return list(getattr(result, "data_array", None) or []) if result is not None else []

    @property
    def row_count(self) -> int:
        result = self.response.result
        if result is None:
            return 0
        explicit = getattr(result, "row_count", None)
        if explicit is not None:
            return int(explicit)
        return len(self.data_array)

    @property
    def next_page_token(self) -> Optional[str]:
        return getattr(self.response, "next_page_token", None)

    # ------------------------------------------------------------------ #
    # Pagination
    # ------------------------------------------------------------------ #
    def next_page(self) -> Optional["VectorSearchQueryResult"]:
        """Fetch the next page, or ``None`` when none is available."""
        token = self.next_page_token
        if not token:
            return None
        endpoint_name = self.index.endpoint_name
        response = self.index.api.query_next_page(
            index_name=self.index.index_name,
            endpoint_name=endpoint_name,
            page_token=token,
        )
        return VectorSearchQueryResult(index=self.index, response=response)

    def iter_pages(self) -> Iterator["VectorSearchQueryResult"]:
        """Yield ``self`` then every subsequent page."""
        page: Optional[VectorSearchQueryResult] = self
        while page is not None:
            yield page
            page = page.next_page()

    # ------------------------------------------------------------------ #
    # Materialisation
    # ------------------------------------------------------------------ #
    def to_arrow_table(self) -> "pa.Table":
        """Materialise the result as a :class:`pyarrow.Table`.

        Each column is cast to the Arrow type resolved from its
        ``type_text``; unknown / complex types stay as strings so the
        original byte payload is preserved.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        columns = self.columns
        rows = self.data_array
        if not columns:
            return pa.table({})

        # Transpose rows → columns at the C boundary. Cells arrive as
        # JSON-string scalars (``List[List[str]]`` is the wire shape);
        # we build a string array per column and then route the type
        # coercion through ``pyarrow.compute.cast`` so the numeric /
        # boolean parsing stays in C++ — no Python row loop.
        col_cells: list[list[Any]] = [[] for _ in columns]
        for row in rows:
            for i, _ in enumerate(columns):
                col_cells[i].append(row[i] if i < len(row) else None)

        arrays: list[pa.Array] = []
        names: list[str] = []
        for col, cells in zip(columns, col_cells):
            arrow_type = _resolve_arrow_type(col.type_text)
            string_arr = pa.array(cells, type=pa.string())
            if arrow_type.equals(pa.string()):
                arr = string_arr
            else:
                try:
                    arr = pc.cast(string_arr, target_type=arrow_type)
                except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                    # The SDK occasionally emits values that don't
                    # round-trip cleanly through the C++ cast kernel
                    # (e.g. an int column whose NULLs arrived as empty
                    # strings, or a complex ``array<…>`` shape the
                    # primitive cast doesn't cover). Keep the column as
                    # a string so the caller still sees the byte payload.
                    arr = string_arr
            arrays.append(arr)
            names.append(col.name or "")
        return pa.table(arrays, names=names)

    def to_polars(self) -> "pl.DataFrame":
        """Materialise the result as a :class:`polars.DataFrame`."""
        from yggdrasil.polars.lib import polars as pl

        return pl.from_arrow(self.to_arrow_table())

    def to_pandas(self) -> "pd.DataFrame":
        """Materialise the result as a :class:`pandas.DataFrame`."""
        return self.to_arrow_table().to_pandas()

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return the rows as a list of dicts keyed by column name.

        Genuine row endpoint: vector-search query results are typically
        consumed as ``[{"id": …, "text": …, "score": …}, ...]`` payloads
        handed straight to a downstream RAG prompt / JSON response.
        """
        names = self.column_names
        return [dict(zip(names, row)) for row in self.data_array]
