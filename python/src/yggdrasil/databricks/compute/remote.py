"""Convenience decorator for running functions on Databricks clusters."""

import datetime as dt
import logging
from typing import (
    Callable,
    Optional,
    TypeVar,
    List, TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .cluster import Cluster

from ..workspaces.workspace import Workspace

ReturnType = TypeVar("ReturnType")

logger = logging.getLogger(__name__)


def databricks_remote_compute(
    cluster_id: Optional[str] = None,
    cluster_name: Optional[str] = None,
    workspace: Optional[Workspace] = None,
    cluster: Optional["Cluster"] = None,
    timeout: Optional[dt.timedelta] = None,
    env_keys: Optional[List[str]] = None,
    force_local: bool = False,
    **options
) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    """Return a decorator that executes functions on a remote cluster.

    Args:
        cluster_id: Optional cluster id to target.
        cluster_name: Optional cluster name to target.
        workspace: Workspace instance or host string for lookup.
        cluster: Pre-configured Cluster instance to reuse.
        timeout: Optional execution timeout for remote calls.
        env_keys: Optional environment variable names to forward.
        force_local: Force local execution
        **options: Extra options forwarded to the execution decorator.

    Returns:
        A decorator that runs functions on the resolved Databricks cluster.
    """
    if force_local or Workspace.is_in_databricks_environment():
        def identity(x):
            return x

        return identity

    if isinstance(workspace, str):
        workspace = Workspace(host=workspace)

    if cluster is None:
        if cluster_id or cluster_name:
            cluster = workspace.clusters(
                cluster_id=cluster_id,
                cluster_name=cluster_name
            )
        else:
            cluster = workspace.clusters().replicated_current_environment(
                workspace=workspace,
                cluster_name=cluster_name
            )

    return cluster.execution_decorator(
        env_keys=env_keys,
        timeout=timeout,
        **options
    )


__all__ = [
    "databricks_remote_compute",
]
