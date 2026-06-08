"""Databricks cluster resource + service.

:class:`Cluster` is the cluster handle for all-purpose / classic compute.

:class:`ClusterStatementExecutor` exposes a cluster as a backing for
``yggdrasil.data.executor.StatementExecutor`` — i.e. lets the cluster
stand in for a SQL warehouse when an :class:`SQLEngine` needs a
cluster-driven SQL path.
"""

from .cluster import Cluster
from .service import Clusters, PYTHON_BY_DBR
from .statement import ClusterPreparedStatement, ClusterStatementBatch, ClusterStatementResult
from .statement_executor import ClusterStatementExecutor

__all__ = [
    "Cluster",
    "Clusters",
    "ClusterPreparedStatement",
    "ClusterStatementBatch",
    "ClusterStatementExecutor",
    "ClusterStatementResult",
    "PYTHON_BY_DBR",
]
