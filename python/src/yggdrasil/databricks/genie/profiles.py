"""Best-practice configuration profiles for Databricks resource creation.

Each profile is a frozen dataclass carrying the recommended defaults for one
resource shape.  :class:`AutonomousAgent` uses them as the ``profile=`` argument
on its resource-creation tools so callers get production-grade settings out of
the box — and can override any field via ``dataclasses.replace``.

Pre-built singletons cover the common patterns::

    from yggdrasil.databricks.genie.profiles import (
        SERVERLESS_WAREHOUSE,
        INGESTION_CLUSTER,
        STARTER_WAREHOUSE,
    )

    agent.create_warehouse(profile=SERVERLESS_WAREHOUSE)
    agent.create_cluster(profile=INGESTION_CLUSTER)

Custom profiles are just ``replace(INGESTION_CLUSTER, num_workers=4)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

__all__ = [
    "WarehouseProfile",
    "ClusterProfile",
    "StorageProfile",
    "SERVERLESS_WAREHOUSE",
    "STARTER_WAREHOUSE",
    "PRO_WAREHOUSE",
    "INGESTION_CLUSTER",
    "SINGLE_NODE_CLUSTER",
]


# ---------------------------------------------------------------------------
# Warehouse
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarehouseProfile:
    """Recommended settings for a SQL warehouse.

    ``cluster_size`` accepts the Databricks size tokens: ``2X-Small``,
    ``X-Small``, ``Small``, ``Medium``, ``Large``, ``X-Large``,
    ``2X-Large``, ``3X-Large``, ``4X-Large``.
    """

    cluster_size: str = "Small"
    min_num_clusters: int = 1
    max_num_clusters: int = 1
    auto_stop_mins: int = 10
    warehouse_type: str = "PRO"
    enable_serverless_compute: bool = True
    spot_instance_policy: str = "COST_OPTIMIZED"
    tags: Mapping[str, str] = field(default_factory=dict)


#: Serverless SQL warehouse — fastest cold start, minimal footprint, UC-only.
SERVERLESS_WAREHOUSE: WarehouseProfile = WarehouseProfile(
    cluster_size="Small",
    enable_serverless_compute=True,
    auto_stop_mins=5,
    warehouse_type="PRO",
)

#: Starter warehouse — small classic PRO for development / CI.
STARTER_WAREHOUSE: WarehouseProfile = WarehouseProfile(
    cluster_size="2X-Small",
    enable_serverless_compute=False,
    auto_stop_mins=10,
    warehouse_type="PRO",
)

#: Pro warehouse — medium classic PRO for production workloads.
PRO_WAREHOUSE: WarehouseProfile = WarehouseProfile(
    cluster_size="Medium",
    enable_serverless_compute=False,
    max_num_clusters=3,
    auto_stop_mins=15,
    warehouse_type="PRO",
)


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterProfile:
    """Recommended settings for an all-purpose cluster.

    When ``autoscale_min`` / ``autoscale_max`` are set, ``num_workers`` is
    ignored and Databricks auto-scales within the range.
    """

    node_type_id: str = "i3.xlarge"
    num_workers: int = 2
    autoscale_min: Optional[int] = None
    autoscale_max: Optional[int] = None
    spark_conf: Mapping[str, str] = field(default_factory=dict)
    custom_tags: Mapping[str, str] = field(default_factory=dict)
    init_scripts: Sequence[str] = ()
    single_user_name: Optional[str] = None
    data_security_mode: str = "SINGLE_USER"


#: Multi-node ingestion cluster — outbound internet, IP spread for rate
#: limits, SINGLE_USER security mode.  The default for API ingestion jobs.
INGESTION_CLUSTER: ClusterProfile = ClusterProfile(
    node_type_id="i3.xlarge",
    num_workers=2,
    autoscale_min=2,
    autoscale_max=8,
    data_security_mode="SINGLE_USER",
    spark_conf={
        "spark.databricks.cluster.profile": "singleNode",
    },
)

#: Single-node cluster — small dev/test workloads, no workers.
SINGLE_NODE_CLUSTER: ClusterProfile = ClusterProfile(
    node_type_id="i3.xlarge",
    num_workers=0,
    data_security_mode="SINGLE_USER",
    spark_conf={
        "spark.databricks.cluster.profile": "singleNode",
        "spark.master": "local[*]",
    },
)


# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StorageProfile:
    """Describes a ``<catalog>.<source>`` data layout.

    Feeds :meth:`AutonomousAgent.setup_storage` which creates the catalog,
    schema, raw tables, curated tables, and volumes in one call.

    ``raw_entities`` land as ``raw_<entity>`` in the schema;
    ``curated_entities`` as ``<entity>``.  ``meta_entities`` go into a
    ``_meta`` sub-schema.
    """

    catalog: str
    source: str
    raw_entities: tuple[str, ...] = ()
    curated_entities: tuple[str, ...] = ()
    meta_entities: tuple[str, ...] = ()
    create_volume: bool = True
    volume_name: str = "uploads"
    comment: Optional[str] = None
    properties: Mapping[str, str] = field(default_factory=dict)

    @property
    def schema_name(self) -> str:
        return f"{self.catalog}.{self.source}"

    @property
    def meta_schema_name(self) -> str:
        return f"{self.catalog}.{self.source}._meta"

    def raw_table_name(self, entity: str) -> str:
        return f"{self.schema_name}.raw_{entity}"

    def curated_table_name(self, entity: str) -> str:
        return f"{self.schema_name}.{entity}"

    def volume_full_name(self) -> str:
        return f"{self.schema_name}.{self.volume_name}"
