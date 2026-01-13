"""Remote execution helpers for Databricks command contexts."""

import base64
import dataclasses as dc
import datetime as dt
import io
import json
import logging
import os
import posixpath
import re
import sys
import threading
import zipfile
from types import ModuleType
from typing import TYPE_CHECKING, Optional, Any, Callable, List, Dict, Union, Iterable, Tuple

from ...libs.databrickslib import databricks_sdk
from ...pyutils.exceptions import raise_parsed_traceback
from ...pyutils.modules import resolve_local_lib_path
from ...pyutils.callable_serde import CallableSerde

if TYPE_CHECKING:
    from .cluster import Cluster

if databricks_sdk is not None:
    from databricks.sdk.service.compute import Language, ResultType

__all__ = [
    "ExecutionContext"
]

logger = logging.getLogger(__name__)


@dc.dataclass
class RemoteMetadata:
    """Metadata describing the remote cluster execution environment."""
    site_packages_path: Optional[str] = dc.field(default=None)
    os_env: Dict[str, str] = dc.field(default_factory=dict)
    requirements: Optional[str] = dc.field(default=None)
    version_info: Tuple[int, int, int] = dc.field(default=(0, 0, 0))

    def os_env_diff(
        self,
        current: Optional[Dict] = None
    ):
        """Return environment variables present locally but missing remotely."""
        if current is None:
            current = os.environ

        return {
            k: v
            for k, v in current.items()
            if k not in self.os_env.keys()
        }


@dc.dataclass
class ExecutionContext:
    """
    Lightweight wrapper around Databricks command execution context for a cluster.

    Can be used directly:

        ctx = ExecutionContext(cluster=my_cluster)
        ctx.open()
        ctx.execute("print(1)")
        ctx.close()

    Or as a context manager to reuse the same remote context for multiple commands:

        with ExecutionContext(cluster=my_cluster) as ctx:
            ctx.execute("x = 1")
            ctx.execute("print(x + 1)")
    """
    cluster: "Cluster"
    language: Optional["Language"] = None
    context_id: Optional[str] = None

    _was_connected: Optional[bool] = dc.field(default=None, repr=False)
    _remote_metadata: Optional[RemoteMetadata] = dc.field(default=None, repr=False)

    _lock: threading.RLock = dc.field(default_factory=threading.RLock, init=False, repr=False)

    # --- Pickle / cloudpickle support (don’t serialize locks or cached remote metadata) ---
    def __getstate__(self):
        """Serialize context state, excluding locks and remote metadata."""
        state = self.__dict__.copy()

        # name-mangled field for _lock in instance dict:
        state.pop("_lock", None)

        return state

    def __setstate__(self, state):
        """Restore context state, rehydrating locks if needed."""
        state["_lock"] = state.get("_lock", threading.RLock())

        self.__dict__.update(state)

    def __enter__(self) -> "ExecutionContext":
        """Enter a context manager, opening a remote execution context."""
        self.cluster.__enter__()
        self._was_connected = self.context_id is not None
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close the remote context if created."""
        if not self._was_connected:
            self.close()
        self.cluster.__exit__(exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def __del__(self):
        """Best-effort cleanup for the remote execution context."""
        self.close()

    @property
    def remote_metadata(self) -> RemoteMetadata:
        """Fetch and cache remote environment metadata for the cluster."""
        # fast path (no lock)
        rm = self._remote_metadata
        if rm is not None:
            return rm

        # slow path guarded
        with self._lock:
            # double-check after acquiring lock
            if self._remote_metadata is None:
                cmd = r"""import glob
import json
import os
from yggdrasil.pyutils.python_env import PythonEnv

current_env = PythonEnv.get_current()
meta = {}

for path in glob.glob('/local_**/.ephemeral_nfs/cluster_libraries/python/lib/python*/site-*', recursive=False):
    if path.endswith('site-packages'):
        meta["site_packages_path"] = path
        break

os_env = meta["os_env"] = {}
for k, v in os.environ.items():
    os_env[k] = v
    
meta["requirements"] = current_env.requirements()
meta["version_info"] = current_env.version_info

print(json.dumps(meta))"""

                try:
                    content = self.execute_command(
                        command=cmd,
                        result_tag="<<RESULT>>",
                        print_stdout=False,
                    )
                except ImportError:
                    self.cluster.ensure_running()

                    content = self.execute_command(
                        command=cmd,
                        result_tag="<<RESULT>>",
                        print_stdout=False,
                    )

                self._remote_metadata = RemoteMetadata(**json.loads(content))

            return self._remote_metadata

    # ------------ internal helpers ------------
    def _workspace_client(self):
        """Return the Databricks SDK client for command execution.

        Returns:
            The underlying WorkspaceClient instance.
        """
        return self.cluster.workspace.sdk()

    def _create_command(
        self,
        language: "Language",
    ) -> any:
        """Create a command execution context, retrying if needed.

        Args:
            language: The Databricks command language to use.

        Returns:
            The created command execution context response.
        """
        self.cluster.ensure_running()

        logger.debug(
            "Creating Databricks command execution context for %s",
            self.cluster
        )

        created = self._workspace_client().command_execution.create_and_wait(
            cluster_id=self.cluster.cluster_id,
            language=language,
        )
        created = getattr(created, "response", created)

        return created

    def connect(
        self,
        language: Optional["Language"] = None
    ) -> "ExecutionContext":
        """Create a remote command execution context if not already open.

        Args:
            language: Optional language override for the context.

        Returns:
            The connected ExecutionContext instance.
        """
        if self.context_id is not None:
            logger.debug(
                "Execution context already open for %s",
                self
            )
            return self

        self.language = language or self.language

        if self.language is None:
            self.language = Language.PYTHON

        ctx = self._create_command(language=self.language)

        context_id = ctx.id
        if not context_id:
            raise RuntimeError("Failed to create command execution context")

        self.context_id = context_id
        logger.info(
            "Opened execution context for %s",
            self
        )
        return self

    def close(self) -> None:
        """Destroy the remote command execution context if it exists.

        Returns:
            None.
        """
        if self.context_id is None:
            return

        logger.debug(
            "Closing execution context for %s",
            self
        )
        try:
            self._workspace_client().command_execution.destroy(
                cluster_id=self.cluster.cluster_id,
                context_id=self.context_id,
            )
        except Exception:
            # non-fatal: context cleanup best-effort
            pass
        finally:
            self.context_id = None

    # ------------ public API ------------
    def execute(
        self,
        obj: Union[str, Callable],
        *,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        env_keys: Optional[List[str]] = None,
        env_variables: Optional[dict[str, str]] = None,
        timeout: Optional[dt.timedelta] = None,
        result_tag: Optional[str] = None,
        **options
    ):
        """Execute a string command or a callable in the remote context.

        Args:
            obj: Command string or callable to execute.
            args: Optional positional arguments for callables.
            kwargs: Optional keyword arguments for callables.
            env_keys: Environment variable names to forward.
            env_variables: Environment variables to inject remotely.
            timeout: Optional timeout for execution.
            result_tag: Optional result tag for parsing output.
            **options: Additional execution options.

        Returns:
            The decoded execution result.
        """
        if isinstance(obj, str):
            return self.execute_command(
                command=obj,
                timeout=timeout,
                result_tag=result_tag,
                **options
            )
        elif callable(obj):
            return self.execute_callable(
                func=obj,
                args=args,
                kwargs=kwargs,
                env_keys=env_keys,
                env_variables=env_variables,
                timeout=timeout,
                **options
            )
        raise ValueError(f"Cannot execute {type(obj)}")

    def is_in_databricks_environment(self):
        """Return True when running on a Databricks runtime."""
        return self.cluster.is_in_databricks_environment()

    def execute_callable(
        self,
        func: Callable | CallableSerde,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        env_keys: Optional[Iterable[str]] = None,
        env_variables: Optional[Dict[str, str]] = None,
        print_stdout: Optional[bool] = True,
        timeout: Optional[dt.timedelta] = None,
        command: Optional[str] = None,
    ) -> Any:
        """Execute a Python callable remotely and return the decoded result.

        Args:
            func: Callable or serialized callable to run remotely.
            args: Positional arguments for the callable.
            kwargs: Keyword arguments for the callable.
            env_keys: Environment variable names to forward.
            env_variables: Environment variables to inject remotely.
            print_stdout: Whether to print stdout from the command output.
            timeout: Optional timeout for execution.
            command: Optional prebuilt command string override.

        Returns:
            The decoded return value from the remote execution.
        """
        if self.is_in_databricks_environment():
            args = args or []
            kwargs = kwargs or {}
            return func(*args, **kwargs)

        self.connect(language=Language.PYTHON)

        logger.debug(
            "Executing callable %s with %s",
            getattr(func, "__name__", type(func)),
            self,
        )

        serialized = CallableSerde.from_callable(func)

        if serialized.pkg_root:
            self.install_temporary_libraries(libraries=serialized.pkg_root)

        current_version = (sys.version_info.major, sys.version_info.minor)

        if current_version != self.cluster.python_version[:2]:
            raise RuntimeError(
                f"Cannot execute callable: local Python version "
                f"{current_version[0]}.{current_version[1]} does not match "
                f"remote cluster Python version "
                f"{self.cluster.python_version[0]}.{self.cluster.python_version[1]}"
            )

        result_tag = "<<<RESULT>>>"

        command = serialized.to_command(
            args=args,
            kwargs=kwargs,
            result_tag=result_tag,
            env_keys=env_keys,
            env_variables=env_variables
        ) if not command else command

        raw_result = self.execute_command(
            command,
            timeout=timeout, result_tag=result_tag, print_stdout=print_stdout
        )

        try:
            result = serialized.parse_command_result(raw_result, result_tag=result_tag)
        except ModuleNotFoundError as remote_module_error:
            _MOD_NOT_FOUND_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
            module_name = _MOD_NOT_FOUND_RE.search(str(remote_module_error))
            module_name = module_name.group(1) if module_name else None
            module_name = module_name.split(".")[0]

            if module_name and "yggdrasil" not in module_name:
                self.close()

                self.cluster.install_libraries(
                    libraries=[module_name],
                    raise_error=True,
                    restart=True
                )

                return self.execute_callable(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    env_keys=env_keys,
                    env_variables=env_variables,
                    print_stdout=print_stdout,
                    timeout=timeout,
                    command=command,
                )
            raise remote_module_error

        return result

    def execute_command(
        self,
        command: str,
        *,
        timeout: Optional[dt.timedelta] = dt.timedelta(minutes=20),
        result_tag: Optional[str] = None,
        print_stdout: Optional[bool] = True,
    ) -> str:
        """Execute a command in this context and return decoded output.

        Args:
            command: The command string to execute.
            timeout: Optional timeout for execution.
            result_tag: Optional tag to extract a specific result segment.
            print_stdout: Whether to print stdout for tagged output.

        Returns:
            The decoded command output string.
        """
        self.connect()

        client = self._workspace_client()
        result = client.command_execution.execute_and_wait(
            cluster_id=self.cluster.cluster_id,
            context_id=self.context_id,
            language=self.language,
            command=command,
            timeout=timeout or dt.timedelta(minutes=20)
        )

        try:
            return self._decode_result(result, result_tag=result_tag, print_stdout=print_stdout)
        except ModuleNotFoundError as remote_module_error:
            _MOD_NOT_FOUND_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
            module_name = _MOD_NOT_FOUND_RE.search(str(remote_module_error))
            module_name = module_name.group(1) if module_name else None
            module_name = module_name.split(".")[0]

            if module_name and "yggdrasil" not in module_name:
                self.close()
                self.cluster.install_libraries(
                    libraries=[module_name],
                    raise_error=True,
                    restart=True
                )

                return self.execute_command(
                    command=command,
                    timeout=timeout,
                    result_tag=result_tag,
                    print_stdout=print_stdout
                )
            raise remote_module_error

    # ------------------------------------------------------------------
    # generic local → remote uploader, via remote python
    # ------------------------------------------------------------------
    def upload_local_path(self, local_path: str, remote_path: str) -> None:
        """
        Generic uploader.

        - If local_path is a file:
              remote_path is the *file* path on remote.
        - If local_path is a directory:
              remote_path is the *directory root* on remote; the directory
              contents are mirrored under it.
        Args:
            local_path: Local file or directory to upload.
            remote_path: Target path on the remote cluster.

        Returns:
            None.
        """
        local_path = os.path.abspath(local_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local path not found: {local_path}")

        # normalize to POSIX for remote (Linux)
        remote_path = remote_path.replace("\\", "/")

        if os.path.isfile(local_path):
            # ---------- single file ----------
            with open(local_path, "rb") as f:
                data_b64 = base64.b64encode(f.read()).decode("ascii")

            cmd = f"""import base64, os

remote_file = {remote_path!r}
data_b64 = {data_b64!r}

os.makedirs(os.path.dirname(remote_file), exist_ok=True)
with open(remote_file, "wb") as f:
    f.write(base64.b64decode(data_b64))
"""

            self.execute_command(command=cmd, print_stdout=False)
            return

        # ---------- directory ----------
        buf = io.BytesIO()
        local_root = local_path

        # zip local folder into memory
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(local_root):
                # skip __pycache__
                dirs[:] = [d for d in dirs if d != "__pycache__"]

                rel_root = os.path.relpath(root, local_root)
                if rel_root == ".":
                    rel_root = ""
                for name in files:
                    if name.endswith((".pyc", ".pyo")):
                        continue
                    full = os.path.join(root, name)
                    arcname = os.path.join(rel_root, name) if rel_root else name
                    zf.write(full, arcname=arcname)

        data_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        cmd = f"""import base64, io, os, zipfile

remote_root = {remote_path!r}
data_b64 = {data_b64!r}

os.makedirs(remote_root, exist_ok=True)

buf = io.BytesIO(base64.b64decode(data_b64))
with zipfile.ZipFile(buf, "r") as zf:
    for member in zf.infolist():
        rel_name = member.filename
        target_path = os.path.join(remote_root, rel_name)

        if member.is_dir() or rel_name.endswith("/"):
            os.makedirs(target_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                dst.write(src.read())
"""

        self.execute_command(command=cmd, print_stdout=False)

    # ------------------------------------------------------------------
    # upload local lib into remote site-packages
    # ------------------------------------------------------------------
    def install_temporary_libraries(
        self,
        libraries: str | ModuleType | List[str | ModuleType],
        with_dependencies: bool = True
    ) -> Union[str, ModuleType, List[str | ModuleType]]:
        """
        Upload a local Python lib/module into the remote cluster's
        site-packages.

        `local_lib` can be:
        - path to a folder  (e.g. "./ygg")
        - path to a file    (e.g. "./ygg/__init__.py")
        - module name       (e.g. "ygg")
        - module object     (e.g. import ygg; workspace.upload_local_lib(ygg))
        Args:
            libraries: Library path, name, module, or iterable of these.
            with_dependencies: Whether to include dependencies (unused).

        Returns:
            The resolved library or list of libraries uploaded.
        """
        if isinstance(libraries, (list, tuple, set)):
            return [
                self.install_temporary_libraries(l) for l in libraries
            ]

        resolved = resolve_local_lib_path(libraries)

        remote_site_packages_path = self.remote_metadata.site_packages_path
        if resolved.is_dir():
            # site-packages/<package_name>/
            remote_target = posixpath.join(remote_site_packages_path, resolved.name)
        else:
            # site-packages/<module_file>
            remote_target = posixpath.join(remote_site_packages_path, resolved.name)

        self.upload_local_path(resolved, remote_target)

        return libraries

    def _decode_result(
        self,
        result: Any,
        *,
        result_tag: Optional[str],
        print_stdout: Optional[bool] = True
    ) -> str:
        """Mirror the old Cluster.execute_command result handling.

        Args:
            result: Raw command execution response.
            result_tag: Optional tag to extract a segment from output.
            print_stdout: Whether to print stdout when using tags.

        Returns:
            The decoded output string.
        """
        if not getattr(result, "results", None):
            raise RuntimeError("Command execution returned no results")

        res = result.results

        # error handling
        if res.result_type == ResultType.ERROR:
            message = res.cause or "Command execution failed"

            if self.language == Language.PYTHON:
                raise_parsed_traceback(message)

            remote_tb = (
                getattr(res, "data", None)
                or getattr(res, "stack_trace", None)
                or getattr(res, "traceback", None)
            )
            if remote_tb:
                message = f"{message}\n{remote_tb}"

            raise RuntimeError(message)

        # normal output
        if res.result_type == ResultType.TEXT:
            output = getattr(res, "data", "") or ""
        elif getattr(res, "data", None) is not None:
            output = str(res.data)
        else:
            output = ""

        # result_tag slicing
        if result_tag:
            start = output.find(result_tag)
            if start != -1:
                content_start = start + len(result_tag)
                end = output.find(result_tag, content_start)
                if end != -1:
                    before = output[:start].strip()
                    if before and print_stdout:
                        print(before)
                    return output[content_start:end]

        return output
