import datetime as dt
import logging
from typing import (
    Callable,
    Optional,
    TypeVar,
    List,
)

from ..workspaces.workspace import Workspace

ReturnType = TypeVar("ReturnType")

logger = logging.getLogger(__name__)


def databricks_remote_compute(
    cluster_id: Optional[str] = None,
    cluster_name: Optional[str] = None,
    workspace: Optional[Workspace] = None,
    cluster: Optional["Cluster"] = None,
    timeout: Optional[dt.timedelta] = None,
    env_keys: Optional[List[str]] = None
) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    from .. import Cluster

    if isinstance(workspace, str):
        workspace = Workspace(host=workspace)

    if cluster is None:
        if cluster_id:
            cluster = Cluster(workspace=workspace, cluster_id=cluster_id)
        else:
            cluster = Cluster.replicated_current_environment(
                workspace=workspace,
                cluster_name=cluster_name
            )

    return cluster.execution_decorator(
        env_keys=env_keys,
        timeout=timeout,
    )


__all__ = [
    "databricks_remote_compute",
]
