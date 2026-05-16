"""Top-level Databricks compute service aggregator.

Exposes :class:`Compute` — the wrapper attached to :class:`DatabricksClient`
as ``client.compute``. ``compute.clusters`` resolves to the
:class:`yggdrasil.databricks.cluster.Clusters` service (moved out of
this package so the cluster resource + service live together at
``yggdrasil.databricks.cluster``); ``compute.instance_pools`` resolves
to :class:`InstancePools` next door.
"""
from typing import TYPE_CHECKING

from ..client import DatabricksService

if TYPE_CHECKING:
    from .instance_pool import InstancePools
    from ..cluster.service import Clusters

__all__ = ["Compute"]


class Compute(DatabricksService):

    @property
    def clusters(self) -> "Clusters":
        from ..cluster.service import Clusters

        return self.client.lazy_property(
            self,
            cache_attr="_clusters",
            factory=lambda: Clusters(client=self.client),
            use_cache=True
        )

    @property
    def instance_pools(self) -> "InstancePools":
        from .instance_pool import InstancePools

        return self.client.lazy_property(
            self,
            cache_attr="_instance_pools",
            factory=lambda: InstancePools(client=self.client),
            use_cache=True,
        )
