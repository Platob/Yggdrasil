import datetime as dt
import functools
from typing import (
    Callable,
    Optional,
    TypeVar,
    List,
)

from ..workspaces.workspace import Workspace

ReturnType = TypeVar("ReturnType")


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
        workspace = Workspace(host=workspace)

    if cluster is None:
        if cluster_id:
            cluster = Cluster(workspace=workspace, cluster_id=cluster_id)
        else:
            cluster = Cluster.replicated_current_environment(workspace=workspace)

    context = cluster.execution_context()

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if force_local:
                return func(*args, **kwargs)

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
