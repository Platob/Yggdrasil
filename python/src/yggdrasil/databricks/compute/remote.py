import datetime as dt
import functools
import logging
from typing import (
    Callable,
    Optional,
    TypeVar,
    List,
)

from ..workspaces.workspace import Workspace
from ...ser import SerializedFunction

ReturnType = TypeVar("ReturnType")

logger = logging.getLogger(__name__)


def databricks_remote_compute(
    cluster_id: Optional[str] = None,
    workspace: Optional[Workspace] = None,
    cluster: Optional["Cluster"] = None,
    timeout: Optional[dt.timedelta] = None,
    env_keys: Optional[List[str]] = None
) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    from .. import Cluster

    if isinstance(workspace, str):
        logger.debug("Creating Workspace from host '%s' for remote compute", workspace)
        workspace = Workspace(host=workspace)

    if cluster is None:
        if cluster_id:
            logger.info(
                "Initializing Cluster helper for cluster_id=%s via databricks_remote_compute",
                cluster_id,
            )
            cluster = Cluster(workspace=workspace, cluster_id=cluster_id)
        else:
            logger.info("Replicating current environment into Databricks cluster")
            cluster = Cluster.replicated_current_environment(workspace=workspace)

    return cluster.execution_decorator(
        env_keys=env_keys,
        timeout=timeout,
    )


__all__ = [
    "databricks_remote_compute",
]
