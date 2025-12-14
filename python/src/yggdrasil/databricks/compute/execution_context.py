import base64
import dataclasses as dc
import datetime as dt
import importlib
import io
import logging
import os
import posixpath
import re
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Optional, Any, Callable, List, Dict, Union, Set

from ...libs.databrickslib import databricks_sdk
from ...ser import SerializedFunction, RemoteExecutionError

if TYPE_CHECKING:
    from .cluster import Cluster

if databricks_sdk is not None:
    from databricks.sdk.service.compute import Language, ResultType

__all__ = [
    "ExecutionContext"
]

logger = logging.getLogger(__name__)


@dc.dataclass()
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

    _remote_site_packages_path: Optional[str] = dc.field(default=None, init=False, repr=False)
    _remote_installed_local_libs: Optional[Set[str]] = dc.field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._remote_installed_local_libs = self._remote_installed_local_libs or set()

    def remote_site_packages_path(self) -> str:
        if self._remote_site_packages_path is None:
            cmd = """import glob
for path in glob.glob('/local_**/.ephemeral_nfs/cluster_libraries/python/lib/python*/site-*', recursive=False):
    if path.endswith('site-packages'):
        print(path)
        break
"""
            self._remote_site_packages_path = self.execute_command(
                command=cmd,
                result_tag="<<RESULT>>",
                print_stdout=False,
            ).strip()

            assert self._remote_site_packages_path, f"Cannot find remote_site_packages path in remote cluster {self.cluster}"

        return self._remote_site_packages_path

    # ------------ internal helpers ------------
    def _workspace_client(self):
        return self.cluster.workspace.sdk()

    def _create_command(
        self,
        language: "Language",
    ) -> any:
        """
        Wrap `client.command_execution.create` in a 10s timeout.
        On timeout:
          - ensure the cluster is running
          - retry once with the same timeout
        """
        self.cluster.ensure_running()

        logger.debug(
            "Creating Databricks command execution context for cluster_id=%s",
            self.cluster.cluster_id
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
        """Create a remote command execution context if not already open."""
        if self.context_id is not None:
            logger.debug(
                "Execution context already open for cluster_id=%s (context_id=%s)",
                self.cluster.cluster_id,
                self.context_id,
            )
            return self

        self.language = language or self.language

        if self.language is None:
            self.language = Language.PYTHON

        # if self.cluster.runtime_version < "15" and self.language == Language.PYTHON:
        #     raise RuntimeError(f"Cannot remote execute commands for runtime version {self.cluster.runtime_version} with {self.language} language")

        ctx = self._create_command(language=self.language)

        context_id = ctx.id
        if not context_id:
            raise RuntimeError("Failed to create command execution context")

        self.context_id = context_id
        logger.info(
            "Opened execution context for cluster_id=%s (context_id=%s, language=%s)",
            self.cluster.cluster_id,
            self.context_id,
            self.language,
        )
        return self

    def close(self) -> None:
        """Destroy the remote command execution context if it exists."""
        if self.context_id is None:
            return

        logger.debug(
            "Closing execution context for cluster_id=%s (context_id=%s)",
            self.cluster.cluster_id,
            self.context_id,
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

    def __enter__(self) -> "ExecutionContext":
        self.cluster.__enter__()
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self.cluster.__exit__(exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def __del__(self):
        self.close()

    # ------------ public API ------------
    def execute(
        self,
        obj: Union[str, Callable, SerializedFunction],
        *,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        env_keys: Optional[List[str]] = None,
        timeout: Optional[dt.timedelta] = None,
        result_tag: Optional[str] = None,
    ):
        if isinstance(obj, str):
            return self.execute_command(
                command=obj,
                timeout=timeout,
                result_tag=result_tag
            )
        elif callable(obj):
            return self.execute_callable(
                func=obj,
                args=args,
                kwargs=kwargs,
                env_keys=env_keys,
                timeout=timeout,
            )
        raise ValueError(f"Cannot execute {type(obj)}")

    def is_in_databricks_environment(self):
        return self.cluster.is_in_databricks_environment()

    def execute_callable(
        self,
        func: Callable | SerializedFunction,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        env_keys: Optional[List[str]] = None,
        print_stdout: Optional[bool] = True,
        timeout: Optional[dt.timedelta] = None,
        command: Optional[str] = None,
    ) -> str:
        if self.is_in_databricks_environment():
            args = args or []
            kwargs = kwargs or {}
            return func(*args, **kwargs)

        """Execute a command in this context and return decoded output."""
        self.connect(language=Language.PYTHON)

        logger.info(
            "Executing callable %s on cluster_id=%s",
            getattr(func, "__name__", type(func)),
            self.cluster.cluster_id,
        )

        serialized = func if isinstance(func, SerializedFunction) else SerializedFunction.from_callable(func)

        self.install_temporary_libraries(libraries=serialized.package_root)

        # Use dill of same version
        use_dill = sys.version_info[:2] == self.cluster.python_version
        result_tag = "<<<RESULT>>>"
        command = serialized.to_command(
            args=args,
            kwargs=kwargs,
            env_keys=env_keys,
            use_dill=use_dill,
            result_tag=result_tag,
        ) if not command else command

        raw_result = self.execute_command(
            command,
            timeout=timeout, result_tag=result_tag, print_stdout=print_stdout
        )
        result = serialized.parse_command_result(raw_result, raise_error=False)

        if isinstance(result, RemoteExecutionError):
            error: RemoteExecutionError = result
            message = error.remote_message

            m = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
            if m:
                missing_module = m.group(1)
                self.install_temporary_libraries(libraries=missing_module)
                logger.warning(f"{message}, installed on {self.cluster}")

                return self.execute_callable(
                    func=serialized,
                    args=args,
                    kwargs=kwargs,
                    print_stdout=print_stdout,
                    timeout=timeout,
                    command=command,
                )
            raise error

        return result

    def execute_command(
        self,
        command: str,
        *,
        timeout: Optional[dt.timedelta] = None,
        result_tag: Optional[str] = None,
        print_stdout: Optional[bool] = True,
    ) -> str:
        """Execute a command in this context and return decoded output."""
        self.connect()

        client = self._workspace_client()
        result = client.command_execution.execute_and_wait(
            cluster_id=self.cluster.cluster_id,
            context_id=self.context_id,
            language=self.language,
            command=command,
            timeout=timeout or dt.timedelta(minutes=20)
        )

        return self._decode_result(result, result_tag=result_tag, print_stdout=print_stdout)

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
    ) -> Union[str, ModuleType, List[str | ModuleType]]:
        """
        Upload a local Python lib/module into the remote cluster's
        site-packages.

        `local_lib` can be:
        - path to a folder  (e.g. "./ygg")
        - path to a file    (e.g. "./ygg/__init__.py")
        - module name       (e.g. "ygg")
        - module object     (e.g. import ygg; workspace.upload_local_lib(ygg))
        """
        if isinstance(libraries, (list, tuple, set)):
            return [self.install_temporary_libraries(l) for l in libraries]

        resolved = _resolve_local_lib_path(libraries)
        remote_site_packages = self.remote_site_packages_path()

        if resolved.is_dir():
            # site-packages/<package_name>/
            remote_target = posixpath.join(remote_site_packages, resolved.name)
        else:
            # site-packages/<module_file>
            remote_target = posixpath.join(remote_site_packages, resolved.name)

        resolved = str(resolved)

        if resolved not in self._remote_installed_local_libs:
            self._remote_installed_local_libs.add(resolved)
            try:
                self.upload_local_path(resolved, remote_target)
            except:
                self._remote_installed_local_libs.remove(resolved)
                raise

        return libraries

    def _decode_result(
        self,
        result: Any,
        *,
        result_tag: Optional[str],
        print_stdout: Optional[bool] = True
    ) -> str:
        """Mirror the old Cluster.execute_command result handling."""
        if not getattr(result, "results", None):
            raise RuntimeError("Command execution returned no results")

        res = result.results

        # error handling
        if res.result_type == ResultType.ERROR:
            message = res.cause or "Command execution failed"

            m = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
            if m:
                missing_module = m.group(1)
                self.cluster.install_libraries(libraries=[missing_module])
                logger.warning(f"{message}, just installed it on {self.cluster}, try again")

            remote_tb = (
                getattr(res, "data", None)
                or getattr(res, "stack_trace", None)
                or getattr(res, "traceback", None)
            )
            if remote_tb:
                message = f"{message}\n\nRemote traceback:\n{remote_tb}"

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
                    before = output[:start]
                    if before and print_stdout:
                        print(before)
                    return output[content_start:end]

        return output


def _resolve_local_lib_path(
    lib: Union[str, ModuleType],
) -> Path:
    """
    Resolve a lib spec (path string, module name, or module object)
    into a concrete filesystem path.

    Rules:
    - If it's a path and exists:
        * if it's a file:
            - if it's __init__.py -> treat as start of a package, then walk up
              until top-most directory that still has __init__.py
            - else -> just that file
        * if it's a dir:
            - if it (or parents) contain __init__.py, walk up to the
              top-most dir that still has __init__.py
            - else -> that dir as-is
    - Else treat as module name:
        * import, use __file__, apply same logic as above.
    """
    # Module object case
    if isinstance(lib, ModuleType):
        mod = lib
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            raise ValueError(
                f"Module {mod.__name__!r} has no __file__; cannot determine path"
            )
        path = Path(mod_file).resolve()
    else:
        # First, treat as path
        p = Path(lib)
        if p.exists():
            path = p.resolve()
        else:
            # Not a path -> try as module name
            try:
                mod = importlib.import_module(lib)
            except ImportError as e:
                raise FileNotFoundError(
                    f"'{lib}' is neither an existing path nor an importable module"
                ) from e

            mod_file = getattr(mod, "__file__", None)
            if not mod_file:
                raise ValueError(
                    f"Module {mod.__name__!r} has no __file__; cannot determine path"
                )
            path = Path(mod_file).resolve()

    # If it's a file, maybe part of a package
    if path.is_file():
        # If it's __init__.py, start from its directory
        if path.name == "__init__.py":
            pkg_dir = path.parent
        else:
            # Regular module: see if its parent is a package
            pkg_dir = path.parent
    else:
        # Directory given. Might be a package root or inside a package.
        pkg_dir = path

    # Walk up to the top-most directory that still looks like a package
    # (has __init__.py). If none found, just use the original dir/file.
    top_pkg_dir = None
    current = pkg_dir

    while True:
        init_file = current / "__init__.py"
        if init_file.exists():
            top_pkg_dir = current
            # try going higher
            parent = current.parent
            # stop at filesystem root
            if parent == current:
                break
            current = parent
        else:
            break

    if top_pkg_dir is not None:
        return top_pkg_dir

    # Not in a package context -> return original path
    return path
