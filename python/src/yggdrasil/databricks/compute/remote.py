"""Convenience decorator for running functions on Databricks clusters."""

import logging
import os
from typing import (
    Callable,
    Optional,
    TypeVar,
    List, TYPE_CHECKING, Union,
)

from yggdrasil.data.cast.registry import identity

if TYPE_CHECKING:
    from .cluster import Cluster

from ..client import DatabricksClient


__all__ = [
    "databricks_remote_compute"
]


ReturnType = TypeVar("ReturnType")

logger = logging.getLogger(__name__)


def databricks_remote_compute(
    _func: Optional[Callable] = None,
    cluster_id: Optional[str] = None,
    cluster_name: Optional[str] = None,
    workspace: Optional[Union["DatabricksClient", str]] = None,
    cluster: Optional["Cluster"] = None,
    env_keys: Optional[List[str]] = None,
    force_local: bool = False,
) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    """Return a decorator that executes functions on a remote cluster.

    Args:
        _func: function to decorate
        cluster_id: Optional cluster id to target.
        cluster_name: Optional cluster name to target.
        workspace: Workspace instance or host string for lookup.
        cluster: Pre-configured Cluster instance to reuse.
        env_keys: Optional environment variable names to forward.
        force_local: Force local execution

    Returns:
        A decorator that runs functions on the resolved Databricks cluster.
    """
    if force_local or DatabricksClient.is_in_databricks_environment():
        return identity if _func is None else _func

    if workspace is None:
        workspace = os.getenv("DATABRICKS_HOST")

    if workspace is None:
        return identity if _func is None else _func

    workspace = DatabricksClient.parse(workspace)

    if cluster is None:
        if cluster_id or cluster_name:
            cluster = workspace.compute.clusters.find_cluster(
                cluster_id=cluster_id,
                cluster_name=cluster_name
            )
        else:
            cluster = workspace.compute.clusters.all_purpose_cluster(wait=False)

    return cluster.decorate(
        func=_func,
        environ=env_keys,
    )
