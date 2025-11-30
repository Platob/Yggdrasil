import base64
import datetime as dt
import textwrap
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

import dill

from ...libs.databrickslib import require_databricks_sdk
from ..workspaces.workspace import DBXWorkspace

if TYPE_CHECKING:  # pragma: no cover - hints only
    from databricks.sdk import WorkspaceClient


ReturnType = TypeVar("ReturnType")


@require_databricks_sdk
def remote_pyeval(
    cluster_id: str,
    func: Callable[..., ReturnType],
    *args: Any,
    workspace: Union["WorkspaceClient", DBXWorkspace],
    timeout: Optional[dt.timedelta] = None,
    **kwargs: Any,
) -> ReturnType:
    """Execute a Python callable on a Databricks cluster.

    The callable and its arguments are serialized with :mod:`dill`, sent to the
    remote command execution API, and the returned value is deserialized
    locally. Any exception raised remotely is re-raised locally with the remote
    traceback attached.

    Args:
        cluster_id: Target cluster identifier.
        func: Python callable to execute on the cluster.
        *args: Positional arguments passed to ``func``.
        workspace: Either a :class:`databricks.sdk.WorkspaceClient` instance or
            :class:`~yggdrasil.databricks.workspaces.workspace.DBXWorkspace`.
        timeout: Optional timeout for command execution. Defaults to the SDK
            standard (20 minutes) when omitted.
        **kwargs: Keyword arguments passed to ``func``.

    Returns:
        The value returned by ``func`` when executed on the cluster.

    Raises:
        RuntimeError: If the remote execution reports an error or returns an
            unexpected result format.
    """
    from databricks.sdk.service.compute import CommandStatus, Language, ResultType

    client = _workspace_client(workspace)

    context = client.command_execution.create_and_wait(
        cluster_id=cluster_id,
        language=Language.PYTHON,
    )

    command = _build_remote_command(func=func, args=args, kwargs=kwargs)
    timeout = timeout or dt.timedelta(minutes=20)
    context_id = getattr(context, "id", None)

    try:
        result = client.command_execution.execute_and_wait(
            cluster_id=cluster_id,
            context_id=context_id,
            language=Language.PYTHON,
            command=command,
            timeout=timeout,
        )

        if not result.results:
            raise RuntimeError("Remote execution returned no results")

        if result.results.result_type == ResultType.ERROR:
            raise RuntimeError(result.results.cause or "Remote execution failed")

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

        return _decode_remote_result(result.results.data)

    finally:
        if context_id:
            try:
                client.command_execution.destroy(
                    cluster_id=cluster_id,
                    context_id=context_id,
                )
            except Exception:
                # Best-effort cleanup; avoid masking the original error
                pass


def _workspace_client(workspace: Union["WorkspaceClient", DBXWorkspace]):
    if isinstance(workspace, DBXWorkspace):
        return workspace.sdk()

    if hasattr(workspace, "sdk") and callable(getattr(workspace, "sdk")):
        return workspace.sdk()

    return workspace


def _build_remote_command(
    func: Callable[..., ReturnType],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    payload = dill.dumps({"func": func, "args": args, "kwargs": kwargs})
    encoded_payload = base64.b64encode(payload).decode("utf-8")

    return textwrap.dedent(
        f"""
        import base64
        import dill
        import traceback

        _payload = base64.b64decode("{encoded_payload}")
        _data = dill.loads(_payload)
        _func = _data["func"]
        _args = _data["args"]
        _kwargs = _data["kwargs"]

        try:
            _value = _func(*_args, **_kwargs)
            _encoded = dill.dumps({{"ok": True, "value": _value}})
        except Exception as _exc:  # noqa: BLE001 - bubble up remote exception details
            _encoded = dill.dumps({{
                "ok": False,
                "error": repr(_exc),
                "traceback": traceback.format_exc(),
            }})

        print(base64.b64encode(_encoded).decode("utf-8"))
        """
    )


def _decode_remote_result(encoded_data: str) -> ReturnType:
    payload = base64.b64decode(encoded_data)
    response = dill.loads(payload)

    if response.get("ok"):
        return response.get("value")

    raise RuntimeError(
        "Remote function raised an exception: "
        f"{response.get('error')}\n{response.get('traceback')}"
    )


__all__ = ["remote_pyeval"]
