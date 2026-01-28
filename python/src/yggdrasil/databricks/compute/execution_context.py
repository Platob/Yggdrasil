"""Remote execution helpers for Databricks command contexts."""

import base64
import dataclasses as dc
import io
import json
import logging
import os
import posixpath
import re
import sys
import threading
import zipfile
from threading import Thread
from types import ModuleType
from typing import TYPE_CHECKING, Optional, Any, Callable, List, Dict, Union, Iterable, Tuple

from .command_execution import CommandExecution
from .exceptions import CommandExecutionAborted
from ...libs.databrickslib import databricks_sdk, DatabricksDummyClass
from ...pyutils.callable_serde import CallableSerde
from ...pyutils.exceptions import raise_parsed_traceback
from ...pyutils.expiring_dict import ExpiringDict
from ...pyutils.modules import resolve_local_lib_path
from ...pyutils.waiting_config import WaitingConfig, WaitingConfigArg

if TYPE_CHECKING:
    from .cluster import Cluster

if databricks_sdk is not None:
    from databricks.sdk.service.compute import Language, ResultType, CommandStatusResponse
else:
    Language = DatabricksDummyClass
    ResultType = DatabricksDummyClass
    CommandStatusResponse = DatabricksDummyClass


__all__ = [
    "ExecutionContext"
]

LOGGER = logging.getLogger(__name__)
UPLOADED_PACKAGE_ROOTS: Dict[str, ExpiringDict] = {}


@dc.dataclass
class RemoteMetadata:
    """Metadata describing the remote cluster execution environment."""
    site_packages_path: Optional[str] = dc.field(default=None)
    os_env: Dict[str, str] = dc.field(default_factory=dict)
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
    language: Optional[Language] = None
    context_id: Optional[str] = None

    _was_connected: Optional[bool] = dc.field(default=None, repr=False)
    _remote_metadata: Optional[RemoteMetadata] = dc.field(default=None, repr=False)
    _uploaded_package_roots: Optional[ExpiringDict] = dc.field(default_factory=ExpiringDict, repr=False)

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
            self.close(wait=False)
        self.cluster.__exit__(exc_type, exc_val=exc_val, exc_tb=exc_tb)

    @property
    def workspace(self):
        return self.cluster.workspace

    @property
    def cluster_id(self):
        return self.cluster.cluster_id

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
                cmd = r"""import glob, json, os
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
    
meta["version_info"] = current_env.version_info

print(json.dumps(meta))"""

                content = self.execute_command(command=cmd)

                self._remote_metadata = RemoteMetadata(**json.loads(content))

            return self._remote_metadata

    # ------------ internal helpers ------------
    def workspace_client(self):
        """Return the Databricks SDK client for command execution.

        Returns:
            The underlying WorkspaceClient instance.
        """
        return self.cluster.workspace.sdk()

    def shared_cache_path(
        self,
        suffix: str
    ):
        return self.cluster.shared_cache_path(
            suffix="/context/%s/%s" % (self.context_id, suffix.lstrip("/"))
        )

    def create(
        self,
        language: "Language",
    ) -> any:
        """Create a command execution context, retrying if needed.

        Args:
            language: The Databricks command language to use.

        Returns:
            The created command execution context response.
        """
        LOGGER.debug(
            "Creating Databricks command execution context for %s",
            self.cluster
        )

        client = self.workspace_client().command_execution

        try:
            created = client.create_and_wait(
                cluster_id=self.cluster_id,
                language=language,
            )
        except:
            self.cluster.ensure_running()

            created = client.create_and_wait(
                cluster_id=self.cluster_id,
                language=language,
            )

        LOGGER.info(
            "Created Databricks command execution context %s",
            self
        )

        created = getattr(created, "response", created)

        self.context_id = created.id

        return self

    def connect(
        self,
        language: Optional[Language] = None,
        reset: bool = False
    ) -> "ExecutionContext":
        """Create a remote command execution context if not already open.

        Args:
            language: Optional language override for the context.
            reset: Reset existing if connected

        Returns:
            The connected ExecutionContext instance.
        """
        if self.context_id is not None:
            if not reset:
                return self

            LOGGER.info(
                "%s reset connection",
                self
            )

            self.close(wait=False)

        language = language or self.language

        if language is None:
            language = Language.PYTHON

        return self.create(language=language)

    def close(self, wait: bool = True) -> None:
        """Destroy the remote command execution context if it exists.

        Returns:
            None.
        """
        if not self.context_id:
            return

        client = self.workspace_client()

        try:
            if wait:
                client.command_execution.destroy(
                    cluster_id=self.cluster.cluster_id,
                    context_id=self.context_id,
                )
            else:
                Thread(
                    target=client.command_execution.destroy,
                    kwargs={
                        "cluster_id": self.cluster.cluster_id,
                        "context_id": self.context_id,
                    }
                ).start()
        except BaseException:
            # non-fatal: context cleanup best-effort
            pass
        finally:
            self.context_id = None

    # ------------ public API ------------
    def command(
        self,
        context: Optional["ExecutionContext"] = None,
        func: Optional[Callable] = None,
        command_id: Optional[str] = None,
        command: Optional[str] = None,
        language: Optional[Language] = None,
    ):
        context = self if context is None else context

        return CommandExecution(
            context=context,
            command_id=command_id,
            language=language,
            command=command,
        ).create(
            context=context,
            language=language,
            command=command,
            func=func,
        )

    def execute(
        self,
        obj: Union[str, Callable],
        *,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        env_keys: Optional[List[str]] = None,
        env_variables: Optional[dict[str, str]] = None,
        timeout: Optional[WaitingConfigArg] = True,
    ):
        """Execute a string command or a callable in the remote context.

        Args:
            obj: Command string or callable to execute.
            args: Optional positional arguments for callables.
            kwargs: Optional keyword arguments for callables.
            env_keys: Environment variable names to forward.
            env_variables: Environment variables to inject remotely.
            timeout: Optional timeout for execution.

        Returns:
            The decoded execution result.
        """
        if isinstance(obj, str):
            return self.execute_command(
                command=obj,
                timeout=timeout,
            )
        elif callable(obj):
            return self.execute_callable(
                func=obj,
                args=args,
                kwargs=kwargs,
                env_keys=env_keys,
                env_variables=env_variables,
                timeout=timeout,
            )
        else:
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
        timeout: Optional[WaitingConfigArg] = True,
        command: Optional[str] = None,
    ) -> Any:
        """Execute a Python callable remotely and return the decoded result.

        Args:
            func: Callable or serialized callable to run remotely.
            args: Positional arguments for the callable.
            kwargs: Keyword arguments for the callable.
            env_keys: Environment variable names to forward.
            env_variables: Environment variables to inject remotely.
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

        LOGGER.debug(
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
            timeout=timeout
        )

        result = serialized.parse_command_result(
            raw_result,
            result_tag=result_tag,
            workspace=self.cluster.workspace
        )

        return result

    def execute_command(
        self,
        command: str,
        *,
        language: Optional[Language] = None,
        timeout: Optional[WaitingConfigArg] = True,
    ) -> str:
        """Execute a command in this context and return decoded output.

        Args:
            command: The command string to execute.
            language: Language to execute
            timeout: Optional timeout for execution.

        Returns:
            The decoded command output string.
        """
        self.connect()

        client = self.workspace_client()

        timeout = WaitingConfig.default() if timeout is None else WaitingConfig.check_arg(timeout)
        language = self.language if language is None else Language.PYTHON
        language = Language.PYTHON if language is None else language

        if timeout.timeout:
            result = client.command_execution.execute_and_wait(
                cluster_id=self.cluster.cluster_id,
                context_id=self.context_id,
                language=language,
                command=command,
                timeout=timeout.timeout_timedelta
            )
        else:
            result = client.command_execution.execute(
                cluster_id=self.cluster.cluster_id,
                context_id=self.context_id,
                language=language,
                command=command,
            )

        try:
            return _decode_result(
                result=result,
                language=language
            )
        except CommandExecutionAborted:
            return self.connect(
                language=self.language,
                reset=True
            ).execute_command(
                command=command,
                timeout=timeout,
            )
        except ModuleNotFoundError as remote_module_error:
            _MOD_NOT_FOUND_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
            module_name = _MOD_NOT_FOUND_RE.search(str(remote_module_error))
            module_name = module_name.group(1) if module_name else None
            module_name = module_name.split(".")[0]

            if module_name and "yggdrasil" not in module_name:
                LOGGER.debug(
                    "Installing missing module %s from local environment",
                    module_name,
                )

                self.install_temporary_libraries(
                    libraries=[module_name],
                )

                LOGGER.warning(
                    "Installed missing module %s from local environment",
                    module_name,
                )

                return self.execute_command(
                    command=command,
                    timeout=timeout,
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

            self.execute_command(command=cmd)
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

        self.execute_command(command=cmd)

    # ------------------------------------------------------------------
    # upload local lib into remote site-packages
    # ------------------------------------------------------------------
    def install_temporary_libraries(
        self,
        libraries: str | ModuleType | List[str | ModuleType],
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

        Returns:
            The resolved library or list of libraries uploaded.
        """
        if isinstance(libraries, (list, tuple, set)):
            return [
                self.install_temporary_libraries(l) for l in libraries
            ]

        resolved = resolve_local_lib_path(libraries)
        str_resolved = str(resolved)
        existing = self._uploaded_package_roots.get(str_resolved)

        if not existing:
            remote_site_packages_path = self.remote_metadata.site_packages_path

            if resolved.is_dir():
                # site-packages/<package_name>/
                remote_target = posixpath.join(remote_site_packages_path, resolved.name)
            else:
                # site-packages/<module_file>
                remote_target = posixpath.join(remote_site_packages_path, resolved.name)

            self.upload_local_path(resolved, remote_target)

            self._uploaded_package_roots[str_resolved] = remote_target

        return libraries


def _decode_result(
    result: CommandStatusResponse,
    language: Language
) -> str:
    """Mirror the old Cluster.execute_command result handling.

    Args:
        result: Raw command execution response.

    Returns:
        The decoded output string.
    """
    res = result.results

    # error handling
    if res.result_type == ResultType.ERROR:
        message = res.cause or "Command execution failed"

        if "client terminated the session" in message:
            raise CommandExecutionAborted(message)

        if language == Language.PYTHON:
            raise_parsed_traceback(message)

        raise RuntimeError(message)

    # normal output
    if res.result_type == ResultType.TEXT:
        output = res.data or ""
    elif res.data is not None:
        output = str(res.data)
    else:
        output = ""

    return output
