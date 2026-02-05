"""Convenience decorator for running functions on Databricks clusters."""

import logging
import os
from typing import (
    Callable,
    Optional,
    TypeVar,
    List, TYPE_CHECKING, Union,
)

if TYPE_CHECKING:
    from .cluster import Cluster

from ..workspaces.workspace import Workspace


__all__ = [
    "databricks_remote_compute"
]


ReturnType = TypeVar("ReturnType")

logger = logging.getLogger(__name__)


def identity(x):
    return x


def databricks_remote_compute(
    _func: Optional[Callable] = None,
    cluster_id: Optional[str] = None,
    cluster_name: Optional[str] = None,
    workspace: Optional[Union[Workspace, str]] = None,
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
    if force_local or Workspace.is_in_databricks_environment():
        return identity if _func is None else _func

    if workspace is None:
        workspace = os.getenv("DATABRICKS_HOST")

    if workspace is None:
        return identity if _func is None else _func

    if not isinstance(workspace, Workspace):
        if isinstance(workspace, str):
            workspace = Workspace(host=workspace).connect(clone=False)
        else:
            raise ValueError("Cannot initialize databricks workspace with %s" % type(workspace))

    if cluster is None:
        if cluster_id or cluster_name:
            cluster = workspace.clusters(
                cluster_id=cluster_id,
                cluster_name=cluster_name
            )
        else:
            cluster = workspace.clusters().replicated_current_environment(
                workspace=workspace,
                cluster_name=cluster_name,
                single_user_name=workspace.current_user.user_name,
                wait_update=False
            )

    return cluster.system_context.decorate(
        func=_func,
        environ=env_keys,
    )
