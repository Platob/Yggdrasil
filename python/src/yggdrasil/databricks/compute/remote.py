import base64
import datetime as dt
import functools
import io
import os
import sys
import textwrap
import zipfile
import zlib
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, List, Union

import dill

from ..workspaces.workspace import DBXWorkspace
from ...libs.databrickslib import require_databricks_sdk

if TYPE_CHECKING:  # pragma: no cover - hints only
    pass

ReturnType = TypeVar("ReturnType")

_MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024  # 4 MiB
_RESULT_BEGIN = "<<<REMOTE_PYEVAL_RESULT_BEGIN>>>"
_RESULT_END = "<<<REMOTE_PYEVAL_RESULT_END>>>"


@require_databricks_sdk
def databricks_remote_compute(
    cluster_id: Optional[str] = None,
    workspace: Optional[Union[DBXWorkspace, str]] = None,
    timeout: Optional[dt.timedelta] = None,
    debug: bool = False,
    debug_host: Optional[str] = None,
    debug_port: int = 5678,
    debug_suspend: bool = True,
    upload_folders: Optional[List[str]] = None,
) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    """
    Decorator that executes the wrapped function on a Databricks cluster.

    Usage:
        @remote_pyeval(
            cluster_id="...",
            workspace="https://<workspace-url>",
            upload_folders=[".."],
        )
        def my_func(x, y):
            return x + y

        result = my_func(1, 2)  # executed remotely

    Args:
        cluster_id: Target cluster ID.
        workspace: Optional DBXWorkspace or host string. If None, a default
            DBXWorkspace is created.
        timeout: Optional timeout for remote execution (default 20 minutes).
        debug: If True, try to attach a PyCharm debugger remotely.
        debug_host: Debug server host.
        debug_port: Debug server port.
        debug_suspend: Whether the debugger should suspend on attach.
        upload_folders: Optional list of folders (relative to this file's
            directory or absolute) whose `.py` files will be shipped to the
            cluster.

    Returns:
        A decorator that wraps the target function so calls are executed remotely.
    """

    def decorator(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            return remote_invoke(
                cluster_id=cluster_id,
                func=func,
                args=args,
                kwargs=kwargs,
                workspace=workspace,
                timeout=timeout,
                debug=debug,
                debug_host=debug_host,
                debug_port=debug_port,
                debug_suspend=debug_suspend,
                upload_folders=upload_folders,
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
    debug: bool = False,
    debug_host: Optional[str] = None,
    debug_port: int = 5678,
    debug_suspend: bool = True,
    upload_folders: Optional[List[str]] = None,
) -> ReturnType:
    """
    Internal helper that actually performs the remote execution.
    """
    # Import here so @require_databricks_sdk can do its thing first
    from databricks.sdk.service.compute import CommandStatus, Language, ResultType

    if workspace is None:
        ws = DBXWorkspace()
    elif isinstance(workspace, DBXWorkspace):
        ws = workspace
    else:
        # treat as host string
        ws = DBXWorkspace(host=str(workspace))

    client = ws.sdk()

    context = client.command_execution.create_and_wait(
        cluster_id=cluster_id,
        language=Language.PYTHON,
    )

    command = _build_remote_command(
        func=func,
        args=args,
        kwargs=kwargs,
        debug=debug,
        debug_host=debug_host,
        debug_port=debug_port,
        debug_suspend=debug_suspend,
        upload_folders=upload_folders,
    )
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


def _maybe_attach_debugger(
    enabled: bool,
    host: Optional[str],
    port: int,
    suspend: bool,
) -> None:
    """Attach PyCharm debugger locally if enabled."""
    if not enabled:
        return

    try:
        import pydevd_pycharm  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Local debug setup failed (pydevd_pycharm import): {exc}", file=sys.stderr)
        return

    try:
        debug_host = host or "127.0.0.1"
        pydevd_pycharm.settrace(
            debug_host,
            port=port,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=suspend,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Local debug setup failed (settrace): {exc}", file=sys.stderr)


def _encode_envelope(data: bytes) -> str:
    """Encode payload bytes with a 1-byte compression flag + base64."""
    if len(data) > _MAX_UNCOMPRESSED_BYTES:
        body = zlib.compress(data)
        flag = b"\x01"
    else:
        body = data
        flag = b"\x00"

    envelope = flag + body
    return base64.b64encode(envelope).decode("utf-8")


def _decode_envelope(encoded: str) -> bytes:
    """Decode a payload created by :func:`_encode_envelope`."""
    raw = base64.b64decode(encoded)
    if not raw:
        raise RuntimeError("Empty envelope from remote")

    flag, body = raw[0], raw[1:]
    if flag == 0x00:
        return body
    if flag == 0x01:
        return zlib.decompress(body)

    raise RuntimeError(f"Unknown envelope compression flag: {flag}")


def _build_modules_zip(upload_folders: Optional[List[str]]) -> str:
    """Create a zip containing code under the given folders.

    - `upload_folders` entries may be relative or absolute paths.
    - Relative paths are resolved against this module's directory.
    - Only `.py` files are shipped.
    - Paths inside the zip are relative to this module's directory, so that
      adding the extracted temp dir to sys.path behaves like having this
      directory on sys.path.
    """
    if not upload_folders:
        return ""

    module_dir = os.path.abspath(os.path.dirname(__file__))

    ignore_dirs = {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        "venv",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in upload_folders:
            if not folder:
                continue

            if os.path.isabs(folder):
                base = os.path.abspath(folder)
            else:
                base = os.path.abspath(os.path.join(module_dir, folder))

            if not os.path.isdir(base):
                # Silently skip invalid entries
                continue

            for root, dirs, files in os.walk(base):
                dirs[:] = [
                    d for d in dirs
                    if d not in ignore_dirs and not d.startswith(".")
                ]

                for filename in files:
                    if not filename.endswith(".py"):
                        continue
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, module_dir)
                    zf.write(full_path, rel_path)

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_remote_command(
    func: Callable[..., ReturnType],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    debug: bool,
    debug_host: Optional[str],
    debug_port: int,
    debug_suspend: bool,
    upload_folders: Optional[List[str]] = None,
) -> str:
    # Inner payload: func + args/kwargs
    call_spec = {"func": func, "args": args, "kwargs": kwargs}
    inner_bytes = dill.dumps(call_spec, recurse=True)
    encoded_input = _encode_envelope(inner_bytes)

    # Zip up the requested folders
    modules_zip_b64 = _build_modules_zip(upload_folders)

    debug_snippet = ""
    if debug:
        debug_snippet = textwrap.dedent(
            f"""
            try:
                import pydevd_pycharm
                pydevd_pycharm.settrace(
                    "{debug_host}",
                    port={debug_port},
                    stdoutToServer=True,
                    stderrToServer=True,
                    suspend={repr(debug_suspend)},
                )
            except Exception as _dbg_exc:  # noqa: BLE001
                import sys
                print("Remote debug setup failed:", _dbg_exc, file=sys.stderr)
            """
        )

    return textwrap.dedent(
        f"""
        import base64
        import dill
        import traceback
        import zlib
        import sys
        import io
        import zipfile
        import tempfile

        {debug_snippet}

        _MODULES_ZIP_B64 = {modules_zip_b64!r}

        def _remote_decode_envelope(b64: str) -> bytes:
            raw = base64.b64decode(b64)
            if not raw:
                raise RuntimeError("Empty envelope from client")
            flag, body = raw[0], raw[1:]
            if flag == 0:
                return body
            if flag == 1:
                return zlib.decompress(body)
            raise RuntimeError(f"Unknown envelope compression flag: {{flag}}")

        def _remote_encode_envelope(data: bytes) -> str:
            if len(data) > {_MAX_UNCOMPRESSED_BYTES}:
                body = zlib.compress(data)
                flag = b"\\x01"
            else:
                body = data
                flag = b"\\x00"
            env = flag + body
            return base64.b64encode(env).decode("utf-8")

        def _remote_setup_modules():
            if not _MODULES_ZIP_B64:
                return
            try:
                zip_bytes = base64.b64decode(_MODULES_ZIP_B64)
                tmp_dir = tempfile.mkdtemp(prefix="remote_pyeval_mods_")
                with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                    zf.extractall(tmp_dir)
                if tmp_dir not in sys.path:
                    sys.path.insert(0, tmp_dir)
            except Exception as _mod_exc:  # noqa: BLE001
                # best-effort; don't crash the whole command
                print("Remote module setup failed:", _mod_exc, file=sys.stderr)

        _remote_setup_modules()

        _inner_bytes = _remote_decode_envelope("{encoded_input}")
        _payload = dill.loads(_inner_bytes)
        _func = _payload["func"]
        _args = _payload["args"]
        _kwargs = _payload["kwargs"]

        try:
            _value = _func(*_args, **_kwargs)
            _resp_inner = {{"ok": True, "value": _value}}
        except Exception as _exc:  # noqa: BLE001
            _resp_inner = {{
                "ok": False,
                "error": repr(_exc),
                "traceback": traceback.format_exc(),
            }}

        _resp_bytes = dill.dumps(_resp_inner, recurse=True)
        _resp_env_b64 = _remote_encode_envelope(_resp_bytes)

        print("{_RESULT_BEGIN}")
        print(_resp_env_b64)
        print("{_RESULT_END}")
        """
    )


def _decode_remote_result(raw_data: str) -> ReturnType:
    start = raw_data.find(_RESULT_BEGIN)
    stop = raw_data.find(_RESULT_END, start + len(_RESULT_BEGIN)) if start != -1 else -1

    if start == -1 or stop == -1:
        lines = [ln.strip() for ln in raw_data.splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError(
                "Remote execution returned no decodable payload:\n"
                f"{raw_data!r}"
            )
        encoded_env = lines[-1]
    else:
        encoded_env = raw_data[start + len(_RESULT_BEGIN): stop].strip()

    try:
        inner_bytes = _decode_envelope(encoded_env)
    except Exception as exc:
        raise RuntimeError(
            "Failed to decode remote envelope: "
            f"{exc}\nRaw payload:\n{encoded_env!r}"
        ) from exc

    response = dill.loads(inner_bytes)

    if response.get("ok"):
        return response.get("value")

    raise RuntimeError(
        "Remote function raised an exception: "
        f"{response.get('error')}\n{response.get('traceback')}"
    )


__all__ = [
    "databricks_remote_compute"
]
