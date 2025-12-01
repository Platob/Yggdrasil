import base64
import datetime as dt
import functools
import glob
import inspect
import io
import os
import sys
import textwrap
import zipfile
import zlib
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import dill

from ..workspaces.workspace import DBXWorkspace
from ...libs.databrickslib import require_databricks_sdk

if TYPE_CHECKING:  # pragma: no cover - hints only
    pass

ReturnType = TypeVar("ReturnType")

_MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024  # 4 MiB
_RESULT_BEGIN = "<<<REMOTE_PYEVAL_RESULT_BEGIN>>>"
_RESULT_END = "<<<REMOTE_PYEVAL_RESULT_END>>>"

# key: (host, cluster_id, context_key)
_PERSISTED_CONTEXTS: Dict[Tuple[Optional[str], str, Optional[str]], str] = {}


def _make_context_key(
    client: Any,
    cluster_id: str,
    context_key: Optional[str],
) -> Tuple[Optional[str], str, Optional[str]]:
    host = getattr(getattr(client, "config", None), "host", None)
    return (host, cluster_id, context_key)


def clear_persisted_context(
    workspace: Optional[Union[DBXWorkspace, str]],
    cluster_id: str,
    context_key: Optional[str] = None,
) -> None:
    """
    Drop a cached context_id mapping for a given (host, cluster_id, context_key).

    If context_key is None, the current process ID (os.getpid()) is used as the
    default, mirroring remote_invoke/databricks_remote_compute behavior.

    Note: this does NOT call destroy() on the remote context; it only clears the
    local cache. If you want to destroy the remote context, do so via the SDK.
    """
    if workspace is None:
        ws = DBXWorkspace()
    elif isinstance(workspace, DBXWorkspace):
        ws = workspace
    else:
        ws = DBXWorkspace(host=str(workspace))

    client = ws.sdk()

    # default to PID-based context key if not explicitly given
    if context_key is None:
        context_key = str(os.getpid())

    key = _make_context_key(client, cluster_id, context_key)
    _PERSISTED_CONTEXTS.pop(key, None)


def clear_all_persisted_contexts() -> None:
    """Drop all cached context IDs (local process only)."""
    _PERSISTED_CONTEXTS.clear()


@require_databricks_sdk
def databricks_remote_compute(
    cluster_id: Optional[str] = None,
    workspace: Optional[Union[DBXWorkspace, str]] = None,
    timeout: Optional[dt.timedelta] = None,
    debug: bool = False,
    debug_host: Optional[str] = None,
    debug_port: int = 5678,
    debug_suspend: bool = True,
    upload_paths: Optional[List[str]] = None,
    persist_context: bool = False,
    context_key: Optional[str] = None,
    remote_target: Optional[str] = None,
) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    """
    Decorator that executes the wrapped function on a Databricks cluster.

    Usage:
        @databricks_remote_compute(
            cluster_id="...",
            workspace="https://<workspace-url>",
            upload_paths=[".."],
            persist_context=True,
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
        upload_paths: Optional list of folders/files/globs whose `.py` files
            will be shipped to the cluster. If not provided, the package
            containing the decorated function is uploaded automatically.
        persist_context: If True, reuse a long-lived command context instead of
            creating/destroying one per call.
        context_key: Logical key to distinguish multiple persistent contexts on
            the same cluster (e.g. "feature_eng", "etl").
            If None and persist_context=True, the current process ID
            (str(os.getpid())) is used.
        remote_target: Optional "pkg.mod:func" string. If provided, the remote
            side will import and call that symbol by name instead of unpickling
            the function object from the client.

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
                upload_paths=upload_paths,
                persist_context=persist_context,
                context_key=context_key,
                remote_target=remote_target,
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
    upload_paths: Optional[List[str]] = None,
    persist_context: bool = False,
    context_key: Optional[str] = None,
    remote_target: Optional[str] = None,
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
        debug: Remote debugger toggle (PyCharm).
        debug_host: Debug server host.
        debug_port: Debug server port.
        debug_suspend: Whether the debugger should suspend on attach.
        upload_paths: Optional explicit paths/globs of modules to ship.
            If not provided, the package containing ``func`` is uploaded.
        persist_context: Reuse command context across calls.
        context_key: Optional logical name to distinguish multiple contexts.
            If None and persist_context=True, the current process ID
            (str(os.getpid())) is used.
        remote_target: If provided, remote code will import and call this
            "pkg.mod:func" on the cluster instead of using the pickled func.

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

    client = ws.sdk()

    if not cluster_id:
        raise ValueError("cluster_id is required for remote_invoke")

    timeout = timeout or dt.timedelta(minutes=20)

    # default context key based on current process id when using persistence
    if persist_context and context_key is None:
        context_key = str(os.getpid())

    # --- context acquisition with optional persistence ---
    context_id: Optional[str] = None
    created_here = False
    ctx_key_tuple: Optional[Tuple[Optional[str], str, Optional[str]]] = None

    if persist_context:
        ctx_key_tuple = _make_context_key(client, cluster_id, context_key)
        context_id = _PERSISTED_CONTEXTS.get(ctx_key_tuple)

    def _create_context() -> str:
        nonlocal created_here
        ctx = client.command_execution.create_and_wait(
            cluster_id=cluster_id,
            language=Language.PYTHON,
        )
        cid = getattr(ctx, "id", None)
        if not cid:
            raise RuntimeError("Failed to acquire remote command context")
        created_here = True
        if ctx_key_tuple is not None:
            _PERSISTED_CONTEXTS[ctx_key_tuple] = cid
        return cid

    if not context_id:
        context_id = _create_context()

    command = _build_remote_command(
        func=func,
        args=args,
        kwargs=kwargs,
        debug=debug,
        debug_host=debug_host,
        debug_port=debug_port,
        debug_suspend=debug_suspend,
        upload_paths=upload_paths,
        remote_target=remote_target,
    )

    def _execute_once(cid: str) -> Any:
        return client.command_execution.execute_and_wait(
            cluster_id=cluster_id,
            context_id=cid,
            language=Language.PYTHON,
            command=command,
            timeout=timeout,
        )

    tried_recreate = False

    while True:
        try:
            result = _execute_once(context_id)
            break
        except Exception as exc:
            msg = str(exc)
            if (
                persist_context
                and not tried_recreate
                and (
                    "ContextNotFound" in msg
                    or "does not exist" in msg
                    or "INVALID_STATE" in msg
                )
            ):
                # Drop cached, recreate once, then retry
                if ctx_key_tuple is not None:
                    _PERSISTED_CONTEXTS.pop(ctx_key_tuple, None)
                context_id = _create_context()
                tried_recreate = True
                continue
            raise

    # --- result handling ---
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

    try:
        return _decode_remote_result(result.results.data)
    finally:
        # Only tear down context if not persistent
        if context_id and not persist_context:
            try:
                client.command_execution.destroy(
                    cluster_id=cluster_id,
                    context_id=context_id,
                )
            except Exception:
                # best-effort cleanup; avoid masking the original error
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


def find_package_root(path: str) -> str:
    """
    Return the 'package root' directory for a given file or directory.

    - If `path` is a file, we start from its containing directory.
    - We walk upwards as long as there is an `__init__.py` in the directory.
    - If one or more package dirs are found, we return the *top-most* such dir.
    - If no `__init__.py` is found anywhere, we return the starting directory.
    """
    path = os.path.abspath(path)

    # Normalize to directory
    if os.path.isfile(path):
        current_dir = os.path.dirname(path)
    else:
        current_dir = path

    last_pkg_dir: Optional[str] = None

    # If current_dir itself is a package, mark it
    if os.path.isfile(os.path.join(current_dir, "__init__.py")):
        last_pkg_dir = current_dir

    # Walk upwards while parent also looks like a package
    while True:
        parent = os.path.dirname(current_dir)
        if parent == current_dir:  # reached filesystem root
            break

        init_path = os.path.join(parent, "__init__.py")
        if os.path.isfile(init_path):
            last_pkg_dir = parent
            current_dir = parent
        else:
            break

    # If we found any package dir, return the top-most one,
    # otherwise just return the original directory.
    return last_pkg_dir or (os.path.dirname(path) if os.path.isfile(path) else path)


def _resolve_upload_paths(
    func: Callable[..., Any], upload_paths: Optional[List[str]]
) -> List[str]:
    """Return the set of paths that should be shipped to the remote cluster."""

    # Explicit paths/globs provided by the caller take precedence.
    if upload_paths:
        return upload_paths

    # Otherwise, ship the package that defines ``func``.
    module_dir = _get_module_dir_for_func(func)
    pkg_root = find_package_root(module_dir)
    return [pkg_root]


def _build_modules_zip(
    upload_paths: Optional[List[str]],
    module_dir: str,
) -> str:
    """Create a zip containing code under the given paths.

    - `upload_paths` entries may be:
        * folders
        * single files
        * glob patterns (e.g. "src/**/*.py")
    - Relative paths are resolved against this module's directory.
    - Only `.py` files are shipped.
    - Paths inside the zip are relative to this module's directory, so that
      adding the extracted temp dir to sys.path behaves like having this
      directory on sys.path.
    """
    if not upload_paths:
        return ""

    module_dir = os.path.abspath(module_dir)

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
    written: set[str] = set()

    def _add_file(full_path: str, zf: zipfile.ZipFile) -> None:
        if not full_path.endswith(".py"):
            return
        full_path_abs = os.path.abspath(full_path)
        rel_path = os.path.relpath(full_path_abs, module_dir)
        if rel_path in written:
            return
        written.add(rel_path)
        zf.write(full_path_abs, rel_path)

    def _walk_dir(base: str, zf: zipfile.ZipFile) -> None:
        base = os.path.abspath(base)
        if not os.path.isdir(base):
            return
        for root, dirs, files in os.walk(base):
            dirs[:] = [
                d for d in dirs
                if d not in ignore_dirs and not d.startswith(".")
            ]
            for filename in files:
                _add_file(os.path.join(root, filename), zf)

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for spec in upload_paths:
            if not spec:
                continue

            # Resolve relative to module_dir
            if os.path.isabs(spec):
                pattern = spec
            else:
                pattern = os.path.abspath(spec)

            # If it looks like a glob, expand it
            if any(ch in pattern for ch in "*?[]"):
                matches = glob.glob(pattern, recursive=True)
                for m in matches:
                    if os.path.isdir(m):
                        _walk_dir(m, zf)
                    elif os.path.isfile(m):
                        _add_file(m, zf)
                continue

            # Not a glob: treat as concrete path
            path = os.path.abspath(pattern)
            if os.path.isdir(path):
                _walk_dir(path, zf)
            elif os.path.isfile(path):
                _add_file(path, zf)
            else:
                # silently ignore nonexistent path specs
                continue

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_module_dir_for_func(func: Callable[..., Any]) -> str:
    """
    Best-effort resolution of the directory containing the file that defines `func`.

    Falls back to CWD if it can't find a real file (e.g. builtins, REPL, etc.).
    """
    # Try via inspect first (handles functions defined in regular .py files)
    try:
        module_file = inspect.getsourcefile(func) or inspect.getfile(func)
    except (TypeError, OSError):
        module_file = None

    if not module_file:
        # Fallback: go via the module object
        module_name = getattr(func, "__module__", None)
        if module_name is not None:
            mod = sys.modules.get(module_name)
            if mod is not None:
                module_file = getattr(mod, "__file__", None)

    if module_file:
        return os.path.dirname(os.path.abspath(module_file))

    # Absolute last resort: current working directory
    return os.path.abspath(os.getcwd())


def _build_remote_command(
    func: Callable[..., ReturnType],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    debug: bool,
    debug_host: Optional[str],
    debug_port: int,
    debug_suspend: bool,
    upload_paths: Optional[List[str]],
    remote_target: Optional[str] = None,
) -> str:
    # Inner payload: func + args/kwargs (or target + args/kwargs)
    if remote_target:
        call_spec = {
            "mode": "by_name",
            "target": remote_target,
            "args": args,
            "kwargs": kwargs,
        }
    else:
        call_spec = {
            "mode": "by_object",
            "func": func,
            "args": args,
            "kwargs": kwargs,
        }

    inner_bytes = dill.dumps(call_spec, recurse=True)
    encoded_input = _encode_envelope(inner_bytes)

    module_dir = _get_module_dir_for_func(func)
    resolved_paths = _resolve_upload_paths(func, upload_paths)
    modules_zip_b64 = _build_modules_zip(resolved_paths, module_dir)

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

        _modules_zip = "{modules_zip_b64}"
        if _modules_zip:
            _mod_buf = io.BytesIO(base64.b64decode(_modules_zip))
            _tmp_mod_dir = tempfile.mkdtemp(prefix="remote_modules_")
            with zipfile.ZipFile(_mod_buf) as _zf:
                _zf.extractall(_tmp_mod_dir)
            sys.path.insert(0, _tmp_mod_dir)

        def _remote_decode_envelope(b64: str) -> bytes:
            raw = base64.b64decode(b64)
            if not raw:
                raise RuntimeError("Empty envelope from client")
            flag, body = raw[0], raw[1:]
            if flag == 0:
                return body
            if flag == 1:
                return zlib.decompress(body)
            raise RuntimeError("Unknown envelope compression flag: {{flag}}".format(flag=flag))

        def _remote_encode_envelope(data: bytes) -> str:
            if len(data) > {_MAX_UNCOMPRESSED_BYTES}:
                body = zlib.compress(data)
                flag = b"\\x01"
            else:
                body = data
                flag = b"\\x00"
            env = flag + body
            return base64.b64encode(env).decode("utf-8")

        _inner_bytes = _remote_decode_envelope("{encoded_input}")
        _payload = dill.loads(_inner_bytes)
        _mode = _payload.get("mode", "by_object")

        if _mode == "by_object":
            _func = _payload["func"]
        elif _mode == "by_name":
            _target = _payload["target"]
            _mod_name, _sep, _attr = _target.partition(":")
            if not _sep:
                raise RuntimeError("Invalid remote_target: %r" % (_target,))
            _mod = __import__(_mod_name, fromlist=[_attr])
            _func = getattr(_mod, _attr)
        else:
            raise RuntimeError("Unknown call mode: %r" % (_mode,))

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
    "databricks_remote_compute",
    "remote_invoke",
    "clear_persisted_context",
    "clear_all_persisted_contexts",
]
