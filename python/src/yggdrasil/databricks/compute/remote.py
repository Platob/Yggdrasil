import base64
import concurrent
import concurrent.futures
import datetime as dt
import functools
import os
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
)

import dill

from ...libs.databrickslib import databricks_sdk

if databricks_sdk is not None:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.compute import State, Language

from ..workspaces.workspace import DBXWorkspace
from ...ser import EmbeddedFunction, DependencyInfo

if TYPE_CHECKING:  # pragma: no cover - hints only
    pass

ReturnType = TypeVar("ReturnType")

_MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024  # 4 MiB



def _ensure_cluster_running(
    client: "WorkspaceClient",
    cluster_id: str,
    start_timeout: int = 600,
    poll_interval: int = 10,
) -> None:
    """
    Make sure the cluster is RUNNING.
    If it's not, start it and wait up to `start_timeout` seconds.
    """
    cluster = client.clusters.get(cluster_id)
    state = cluster.state

    # Already fine
    if state in (State.RUNNING, State.RESIZING):
        return

    # If not running, try to start
    if state not in (State.PENDING, State.RESTARTING):
        # TERMINATED / ERROR / UNKNOWN etc → explicit start
        client.clusters.start(cluster_id=cluster_id)

    # Wait for RUNNING
    deadline = time.time() + start_timeout
    while time.time() < deadline:
        cluster = client.clusters.get(cluster_id)
        state = cluster.state
        if state == State.RUNNING:
            return
        if state in (State.ERROR, State.TERMINATED):
            raise RuntimeError(f"Cluster {cluster_id} entered bad state: {state}")
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Cluster {cluster_id} did not reach RUNNING within {start_timeout} seconds"
    )


def _create_command_with_timeout(
    client,
    cluster_id: str,
    language: Language,
    cmd_timeout: int = 10,
) -> any:
    """
    Wrap `client.command_execution.create` in a 10s timeout.
    On timeout:
      - ensure the cluster is running
      - retry once with the same timeout
    """

    def _do_create():
        return client.command_execution.create_and_wait(
            cluster_id=cluster_id,
            language=language,
        )

    # First attempt with timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_create)
        try:
            return future.result(timeout=cmd_timeout)
        except concurrent.futures.TimeoutError:
            # Timed out → check/start cluster and retry once
            _ensure_cluster_running(client, cluster_id)

    return _do_create()


def databricks_remote_compute(
    cluster_id: Optional[str] = None,
    workspace: Optional[Union[DBXWorkspace, str]] = None,
    timeout: Optional[dt.timedelta] = None,
    force_local: Optional[bool] = None
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
        workspace: Optional DBXWorkspace or host string. If None, a default
            DBXWorkspace is created.
        timeout: Optional timeout for remote execution (default 20 minutes).
        force_local: Optional bool to bypass remote compute

    Returns:
        A decorator that wraps the target function so calls are executed remotely.
    """
    if force_local is None:
        force_local = os.getenv("DATABRICKS_RUNTIME_VERSION") is not None

    def decorator(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            if force_local:
                return func(*args, **kwargs)

            return remote_invoke(
                cluster_id=cluster_id,
                func=func,
                args=args,
                kwargs=kwargs,
                workspace=workspace,
                timeout=timeout,
            )

        return wrapper

    return decorator


def remote_invoke(
    func: Callable[..., ReturnType],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    cluster_id: Optional[str] = None,
    workspace: Optional[Union[DBXWorkspace, str]] = None,
    timeout: Optional[dt.timedelta] = None,
) -> ReturnType:
    """
    Internal helper that actually performs the remote execution.

    Args:
        func: Local callable whose calls we want to execute remotely.
        args: Positional arguments for the call.
        kwargs: Keyword arguments for the call.
        cluster_id: Target cluster ID (required).
        workspace: Optional DBXWorkspace or host string.
        timeout: Optional timeout for remote execution.

    Returns:
        Deserialized return value from the remote function.
    """
    from databricks.sdk.service.compute import CommandStatus, Language, ResultType

    if workspace is None:
        ws = DBXWorkspace()
    elif isinstance(workspace, DBXWorkspace):
        ws = workspace
    else:
        ws = DBXWorkspace(host=str(workspace))

    if not cluster_id:
        raise ValueError("cluster_id is required for remote_invoke")

    timeout = timeout or dt.timedelta(minutes=20)

    # --- context acquisition with optional persistence ---
    with ws.connect() as connected:
        client = connected.sdk()
        current_user = client.current_user.me()

        create_cmd = _create_command_with_timeout(
            client=client,
            cluster_id=cluster_id,
            language=Language.PYTHON,
            cmd_timeout=10,  # seconds
        )

        context_id = create_cmd.id

        method = EmbeddedFunction.from_callable(func)

        new_deps = []
        for dep in method.dependencies_map:
            # If we have a local root_path, copy it to a user-scoped DBFS cache
            if dep.root_path:
                # Use the last path component as package name fallback
                package_name = os.path.basename(dep.root_path.rstrip(os.sep)) or dep.root_module
                # DBFS path for the package contents
                dbfs_root = f"/Workspace/Shared/.ygg/cache/{current_user.user_name}/pypkg/{package_name}"

                if os.path.isdir(dep.root_path):
                    connected.upload_local_folder(
                        local_dir=dep.root_path,
                        target_dir=dbfs_root,
                        only_if_size_diff=True,
                        parallel_pool=4
                    )

                # Point the dependency at the DBFS-backed local path so build() can
                # add it to sys.path on the cluster
                new_dep = DependencyInfo(
                    root_module=dep.root_module,
                    submodule=dep.submodule,
                    root_path=dbfs_root,
                )
            else:
                # Nothing to copy, just keep as-is
                new_dep = dep

            new_deps.append(new_dep)

        # Swap in the updated dependency map so to_command/build() use DBFS paths
        method.dependencies_map = new_deps

        command = method.to_command(args=args, kwargs=kwargs)

        result = client.command_execution.execute_and_wait(
            cluster_id=cluster_id,
            context_id=context_id,
            language=Language.PYTHON,
            command=command,
            timeout=timeout,
        )

        # --- result handling as you already have ---
        if not result.results:
            raise RuntimeError("Remote execution returned no results")

        if result.results.result_type == ResultType.ERROR:
            # Base message
            msg = result.results.cause or "Remote execution failed"

            # Try to grab traceback / details from various fields
            remote_tb = None

            # Most useful first: Databricks often returns the Python traceback in `data`
            if getattr(result.results, "data", None):
                remote_tb = result.results.data

            # Some SDKs / APIs expose it under a dedicated field
            elif getattr(result.results, "stack_trace", None):
                remote_tb = result.results.stack_trace

            elif getattr(result.results, "traceback", None):
                remote_tb = result.results.traceback

            if remote_tb:
                msg = f"{msg}\n\nRemote traceback:\n{remote_tb}"

            raise RuntimeError(msg)

        if result.results.result_type != ResultType.TEXT:
            raise RuntimeError(
                "Unexpected remote result type: "
                f"{result.results.result_type}. Expected text."
            )

        if result.status != CommandStatus.FINISHED:
            raise RuntimeError(
                f"Remote execution did not finish successfully (status={result.status})"
            )

        if result.results.data is None:
            raise RuntimeError("Remote execution returned empty data")

        try:
            # Parse JSON payload produced by to_command()
            content = base64.b64decode(result.results.data.encode("utf-8"))
            return dill.loads(content)
        finally:
            if context_id:
                try:
                    client.command_execution.destroy(
                        cluster_id=cluster_id,
                        context_id=context_id,
                    )
                except Exception:
                    pass


__all__ = [
    "databricks_remote_compute",
    "remote_invoke",
]
