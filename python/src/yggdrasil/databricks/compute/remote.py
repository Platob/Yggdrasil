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
    Union, List,
)

from ...libs.databrickslib import databricks_sdk

if databricks_sdk is not None:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.compute import State, Language

from ..workspaces.workspace import DBXWorkspace
from ...ser import EmbeddedFunction

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
        workspace: Optional DBXWorkspace or host string. If None, a default
            DBXWorkspace is created.
        timeout: Optional timeout for remote execution (default 20 minutes).
        force_local: Optional bool to bypass remote compute
        env_keys: Environment keys

    Returns:
        A decorator that wraps the target function so calls are executed remotely.
    """
    if os.getenv("DATABRICKS_RUNTIME_VERSION") is not None:
        force_local = True

    def decorator(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            if force_local:
                return func(*args, **kwargs)
            else:
                return remote_invoke(
                    cluster_id=cluster_id,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    workspace=workspace,
                    timeout=timeout,
                    env_keys=env_keys
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
    env_keys: Optional[List[str]] = None
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
        env_keys: Environment keys

    Returns:
        Deserialized return value from the remote function.
    """
    if os.getenv("DATABRICKS_RUNTIME_VERSION") is not None:
        return func(*args, **kwargs)

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

        if method.package_root:
            # Use the last path component as package name fallback
            package_name = os.path.basename(method.package_root.rstrip(os.sep)) or method.package_root
            # DBFS path for the package contents
            dbfs_root = f"/Workspace/Shared/.ygg/cache/{current_user.user_name}/pypkg/{package_name}"

            if os.path.isdir(method.package_root):
                connected.upload_local_folder(
                    local_dir=method.package_root,
                    target_dir=dbfs_root,
                    only_if_size_diff=True,
                    parallel_pool=4,
                    exclude_dir_names=[
                        "__pycache__",
                    ],
                    exclude_hidden=True
                )

            method.package_root = dbfs_root

        command = method.to_command(
            args=args, kwargs=kwargs,
            env_keys=env_keys
        )

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
            import json
            import base64
            import dill
            import sys as _sys

            text = result.results.data  # full stdout from remote command (string)

            start_tag = "<<<EMBEDDED_RESULT_START>>>"
            end_tag = "<<<EMBEDDED_RESULT_END>>>"

            start_idx = text.find(start_tag)
            end_idx = text.find(end_tag, start_idx + len(start_tag)) if start_idx != -1 else -1

            # --- print everything before the balise, line by line ---
            if start_idx == -1:
                # No marker at all -> just dump the whole thing as logs and bail/handle fallback
                prefix = text
            else:
                prefix = text[:start_idx]

            prefix = prefix or ""
            if prefix.strip():
                for line in prefix.splitlines():
                    print(line.rstrip())

            # --- validate markers with explicit errors ---
            if start_idx == -1 and end_idx == -1:
                snippet = text[-1000:] if len(text) > 1000 else text
                raise ValueError(
                    "Cannot find embedded result markers in remote stdout. "
                    f"Expected '{start_tag}' and '{end_tag}' but neither was found. "
                    "Last part of stdout:\n"
                    f"{snippet}"
                )

            if start_idx == -1:
                snippet = text[-1000:] if len(text) > 1000 else text
                raise ValueError(
                    f"Missing start marker '{start_tag}' in remote stdout. "
                    "Cannot locate embedded result payload. "
                    "Last part of stdout:\n"
                    f"{snippet}"
                )

            if end_idx == -1:
                snippet = text[start_idx:start_idx + 1000]
                raise ValueError(
                    f"Found start marker '{start_tag}' at index {start_idx} "
                    f"but missing end marker '{end_tag}'. "
                    "Partial content after start marker:\n"
                    f"{snippet}"
                )

            # --- extract JSON payload ---
            start_idx += len(start_tag)
            payload_json = text[start_idx:end_idx].strip()

            if not payload_json:
                raise ValueError(
                    "Embedded result payload between markers is empty. "
                    f"Markers used: '{start_tag}' ... '{end_tag}'."
                )

            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError as e:
                snippet = payload_json[:1000]
                raise ValueError(
                    "Failed to parse embedded result JSON between markers. "
                    f"JSON error: {e}. "
                    "First 1000 characters of raw payload:\n"
                    f"{snippet}"
                ) from e

            if "result_b64" not in payload:
                raise KeyError(
                    "Embedded result JSON does not contain required key 'result_b64'. "
                    f"Available keys: {list(payload.keys())}"
                )

            result_b64 = payload["result_b64"]

            if not isinstance(result_b64, str) or not result_b64.strip():
                raise ValueError(
                    "Field 'result_b64' in embedded result JSON is missing or empty."
                )

            # --- decode base64 + dill with explicit errors ---
            try:
                result_bytes = base64.b64decode(result_b64.encode("utf-8"))
            except Exception as e:
                snippet = result_b64[:200]
                raise ValueError(
                    "Failed to base64-decode 'result_b64' from embedded result JSON. "
                    f"Error: {e}. First 200 characters of base64 string:\n{snippet}"
                ) from e

            try:
                return dill.loads(result_bytes)
            except Exception as e:
                raise ValueError(
                    "Failed to unpickle embedded result via dill.loads(). "
                    "The remote side may have returned an incompatible or corrupted payload. "
                    f"Underlying error: {e}"
                ) from e

        finally:
            if context_id:
                try:
                    client.command_execution.destroy(
                        cluster_id=cluster_id,
                        context_id=context_id,
                    )
                except Exception:
                    # Best-effort cleanup; don't mask original error
                    pass


__all__ = [
    "databricks_remote_compute",
    "remote_invoke",
]
