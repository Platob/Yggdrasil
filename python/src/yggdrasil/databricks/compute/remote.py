"""databricks.compute.remote module documentation."""

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
    **options
) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    """
    databricks_remote_compute documentation.
    
    Args:
        cluster_id: Parameter.
        cluster_name: Parameter.
        workspace: Parameter.
        cluster: Parameter.
        timeout: Parameter.
        env_keys: Parameter.
        **options: Parameter.
    
    Returns:
        The result.
    """

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
