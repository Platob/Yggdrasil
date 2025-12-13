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

ReturnType = TypeVar("ReturnType")

logger = logging.getLogger(__name__)


def databricks_remote_compute(
    cluster_id: Optional[str] = None,
    workspace: Optional[Workspace] = None,
    cluster: Optional["Cluster"] = None,
    timeout: Optional[dt.timedelta] = None,
    force_local: Optional[bool] = None,
    env_keys: Optional[List[str]] = None
) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    """
    Decorator that executes the wrapped function on a Databricks cluster.

    Usage:
        @databricks_remote_compute(
            cluster_id="...",
            workspace="https://<workspace-url>",
            upload_paths=[".."],
        )
        def my_func(x, y):
            return x + y

        result = my_func(1, 2)  # executed remotely

    Args:
        cluster_id: Target cluster ID.
        workspace: Optional Workspace or host string. If None, a default
            Workspace is created.
        timeout: Optional timeout for remote execution (default 20 minutes).
        force_local: Optional bool to bypass remote compute
        env_keys: Environment keys

    Returns:
        A decorator that wraps the target function so calls are executed remotely.
    """
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

    context = cluster.execution_context()

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if force_local:
                logger.debug("Executing %s locally because force_local=True", func.__name__)
                return func(*args, **kwargs)

            logger.info(
                "Executing %s on remote Databricks cluster %s",
                func.__name__,
                getattr(cluster, "cluster_id", None),
            )
            return context.execute(
                obj=func,
                args=list(args),
                kwargs=kwargs,
                env_keys=env_keys,
                timeout=timeout,
            )

        return wrapper

    return decorator


__all__ = [
    "databricks_remote_compute",
]
