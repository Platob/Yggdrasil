"""Remote execution helpers for Databricks command contexts."""

import dataclasses as dc
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional, Any, Callable, Dict, Union, Literal, TypeVar, \
    Mapping

from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.compute import Language

from yggdrasil.concurrent.threading import Job
from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io.headers import DEFAULT_HOSTNAME
from yggdrasil.io.url import URL

if TYPE_CHECKING:
    from .cluster import Cluster
    from .command_execution import CommandExecution


__all__ = [
    "ExecutionContext",
    "EXCLUDED_ENV_KEYS"
]

EXCLUDED_ENV_KEYS = frozenset([
    'ALLUSERSPROFILE', 'APPDATA',
    'ARM_CLIENT_ID', 'ARM_CLIENT_SECRET', 'ARM_ENVIRONMENT', 'ARM_RESOURCE_ID', 'ARM_TENANT_ID',
    'CLASSPATH', 'CLICOLOR', 'CLICOLOR_FORCE', 'CLUSTER_DB_HOME', 'COMMONPROGRAMFILES', 'COMMONPROGRAMFILES(X86)',
    'COMMONPROGRAMW6432', 'COMPUTERNAME', 'COMSPEC', 'DATABRICKS_CLUSTER_ID', 'DATABRICKS_CLUSTER_LIBS_PYTHON_ROOT_DIR',
    'DATABRICKS_CLUSTER_LIBS_ROOT_DIR', 'DATABRICKS_CLUSTER_LIBS_R_ROOT_DIR',
    'DATABRICKS_HOST', 'DATABRICKS_INSTANCE_ID', 'DATABRICKS_LIBS_NFS_ROOT_DIR',
    'DATABRICKS_LIBS_NFS_ROOT_PATH', 'DATABRICKS_ROOT_VIRTUALENV_ENV', 'DATABRICKS_RUNTIME_VERSION',
    'DATABRICKS_TOKEN', 'DATA_SECURITY_MODE', 'DBX_WORKSPACE_URL', 'DB_HOME', 'DEBUGINFOD_URLS',
    'DEFAULT_DATABRICKS_ROOT_VIRTUALENV_ENV', 'DEFAULT_PYTHON_ENVIRONMENT', 'DISABLE_LOCAL_FILESYSTEM',
    'DRIVERDATA', 'DRIVER_PID_FILE', 'DRIVER_REPL_ID', 'DRIVER_STARTUP_OBSERVABILITY_ENABLED',
    'EFC_12712_1262719628', 'EFC_12712_1592913036', 'EFC_12712_2283032206', 'EFC_12712_2775293581',
    'EFC_12712_3789132940', 'ENABLE_APPCDS', 'ENABLE_CLASSLOADING_LOGS', 'ENABLE_COMMAND_OUTPUT_TRUNCATION',
    'ENABLE_DRIVER_DEVELOPER_MODE', 'ENABLE_IPTABLES', 'ENABLE_KEEPALIVE_COMMAND_CONTEXT', 'ENABLE_REPL_LOGGING',
    'ENABLE_TRACEPARENT_REPL_PROPAGATION', 'FORCE_COLOR', 'GIT_PAGER', 'GRPC_GATEWAY_TOKEN',
    'HALT_VARIABLE_RESOLVE_THREADS_ON_STEP_RESUME', 'HF_DATASETS_CACHE', 'HIVE_HOME', 'HOME', 'HOMEDRIVE',
    'HOMEPATH', 'HTTPS_PROXY', 'HTTP_PROXY', 'ICU_DATA', 'IDE_PROJECT_ROOTS', 'IPYTHONENABLE', 'JAVA_HOME',
    'JAVA_OPTS', 'JUPYTER_WIDGETS_ECHO', 'KOALAS_USAGE_LOGGER', 'LANG', 'LIBRARY_ROOTS', 'LOCALAPPDATA',
    'LOGNAME', 'LOGONSERVER', 'MAIL', 'MASTER', 'MEDSITE', 'MLFLOW_CONDA_HOME', 'MLFLOW_DEPLOYMENTS_TARGET',
    'MLFLOW_GATEWAY_URI', 'MLFLOW_PYTHON_EXECUTABLE', 'MLFLOW_REGISTRY_URI', 'MLFLOW_TRACKING_URI', 'MPLBACKEND',
    'NEXTHINK', 'NO_PROXY', 'NUMBER_OF_PROCESSORS', 'OLDPWD', 'OMPI_MCA_btl_tcp_if_include', 'ONEDRIVE',
    'ONEDRIVECOMMERCIAL', 'OPENSSL_FORCE_FIPS_MODE', 'OS', 'PAGER', 'PATH', 'PATHEXT', 'PINNED_THREAD_MODE',
    'PIPELINE_UDS_CONNECT_MODE', 'PIP_NO_INPUT', 'PROCESSOR_ARCHITECTURE', 'PROCESSOR_IDENTIFIER', 'PROCESSOR_LEVEL',
    'PROCESSOR_REVISION', 'PROGRAMDATA', 'PROGRAMFILES', 'PROGRAMFILES(X86)', 'PROGRAMW6432', 'PROJ_DATA', 'PROMPT',
    'PS1', 'PSMODULEPATH', 'PUBLIC', 'PWD', 'PYCHARM_HELPERS_DIR', 'PYCHARM_HOSTED',
    'PYDEVD_DISABLE_FILE_VALIDATION', 'PYDEVD_INTERRUPT_THREAD_TIMEOUT', 'PYDEVD_LOAD_VALUES_ASYNC',
    'PYDEVD_USE_FRAME_EVAL', 'PYENV_ROOT', 'PYSPARK_GATEWAY_PORT', 'PYSPARK_GATEWAY_SECRET', 'PYSPARK_PYTHON',
    'PYTEST_CURRENT_TEST', 'PYTEST_RUN_CONFIG', 'PYTEST_VERSION', 'PYTHONHASHSEED', 'PYTHONIOENCODING',
    'PYTHONPATH', 'PYTHONUNBUFFERED', 'PYTHON_REPL_SAFE_CONFIG_MAP', 'RAY_TMPDIR', 'R_LIBS', 'SCALA_VERSION',
    'SESSIONNAME', 'SHELL', 'SHLVL', 'SPARK_AUTH_SOCKET_TIMEOUT', 'SPARK_BUFFER_SIZE', 'SPARK_CONF_DIR',
    'SPARK_DIST_CLASSPATH', 'SPARK_ENV_LOADED', 'SPARK_HOME', 'SPARK_LOCAL_DIRS', 'SPARK_LOCAL_IP', 'SPARK_PUBLIC_DNS',
    'SPARK_SCALA_VERSION', 'SPARK_WORKER_MEMORY', 'SUDO_COMMAND', 'SUDO_GID', 'SUDO_UID', 'SUDO_USER', 'SYSTEMDRIVE',
    'SYSTEMROOT', 'TEAMCITY_VERSION', 'TEMP', 'TERM', 'TMP', 'TS_ADSITE_DRIVE', 'TS_APP32', 'TS_APP64', 'TS_APPRW',
    'TS_CONDA', 'TS_DFSN_ROOT', 'TS_IOADDIN', 'TS_PYTHON', 'TZDIR', 'UATDATA', 'USER', 'USERDNSDOMAIN', 'USERDOMAIN',
    'USERDOMAIN_ROAMINGPROFILE', 'USERNAME', 'USERPROFILE', 'USE_LOW_IMPACT_MONITORING', 'VIRTUAL_ENV',
    'VIRTUAL_ENV_PROMPT', 'WINDIR', 'ZES_ENABLE_SYSMAN', '_JB_PPRINT_PRIMITIVES', '_OLD_VIRTUAL_PATH',
    '_OLD_VIRTUAL_PROMPT', '_PIP_USE_IMPORTLIB_METADATA', '_RJEM_MALLOC_CONF', 'container'
])

LOGGER = logging.getLogger(__name__)
UPLOADED_PACKAGE_ROOTS: Dict[str, ExpiringDict] = {}
BytesLike = Union[bytes, bytearray, memoryview]
F = TypeVar("F", bound=Callable[..., Any])

_CTX_RUNTIME_FIELDS = frozenset({"_lock",})
_CTX_RESET_FIELDS = frozenset({"_remote_metadata"})


@dc.dataclass
class RemoteMetadata:
    context_path: str
    tmp_path: str
    libs_path: str


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
    context_id: Optional[str] = None
    context_key: Optional[str] = None
    ephemeral: bool = dc.field(default=True, repr=False, compare=False, hash=False)
    language: Optional[Language] = dc.field(default=None, repr=False, compare=False, hash=False)

    _remote_metadata: Optional[RemoteMetadata] = dc.field(default=None, init=False, repr=False, compare=False, hash=False)
    _requirements: Optional[list[tuple[str]]] = dc.field(default=None, init=False, repr=False, compare=False, hash=False)
    _pyenv_check_timestamp: int = dc.field(default=0, init=False, repr=False, compare=False, hash=False)
    _lock: threading.RLock = dc.field(default_factory=threading.RLock, init=False, repr=False, compare=False, hash=False)

    def __getstate__(self) -> dict:
        """Serialize context state for pickling.

        Drops unpickable threading primitives and resets fields whose
        values are only meaningful in the originating process:

        - ``_lock``: RLock is not picklable and must always be reconstructed
        - ``_remote_metadata``: contains a ``temp_path`` that only exists on
          the remote cluster's filesystem; stale after transport.  The
          ``site_packages_path`` and ``os_env`` within it are also
          process/host-specific.  Drop it and let the lazy property
          re-fetch on first use.

        ``_uploaded_package_roots`` is preserved: remote paths remain valid
        across processes as long as the cluster session is alive, so we avoid
        redundant re-uploads.

        Returns:
            A compact, pickle-ready state dictionary.
        """
        state = {}

        for key, value in self.__dict__.items():
            if key in _CTX_RUNTIME_FIELDS:
                continue
            if key in _CTX_RESET_FIELDS:
                state[key] = None  # preserve key for attribute completeness
                continue
            state[key] = value

        return state

    def __setstate__(self, state: dict) -> None:
        """Restore context state after unpickling.

        Always constructs a fresh RLock — never attempts to restore a
        serialized one.  Ensures all expected attributes are present even
        when the state was produced by an older serialized form.

        Args:
            state: Serialized state dictionary.
        """
        state["_lock"] = threading.RLock()  # always fresh

        self.__dict__.update(state)

    def __enter__(self) -> "ExecutionContext":
        """Enter a context manager, opening a remote execution context."""
        if self.context_id is None:
            return self.create(
                language=self.language,
                context_key=self.context_key
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close the remote context if created."""
        self.close(wait=False, raise_error=False)

    def __repr__(self):
        return "%s(url=%s)" % (
            self.__class__.__name__,
            self.url()
        )

    def __str__(self):
        return self.url().to_string()

    def url(self) -> URL:
        url = self.cluster.url()

        return url.with_query_items({
            "context": self.context_id or "unknown"
        })

    @property
    def client(self):
        return self.cluster.client

    @property
    def cluster_id(self):
        return self.cluster.cluster_id

    @property
    def remote_metadata(self) -> RemoteMetadata:
        """Fetch and cache remote environment metadata for the cluster."""
        # fast path (no lock)
        if self._remote_metadata is not None:
            return self._remote_metadata

        if not self.context_key:
            self.context_key = DEFAULT_HOSTNAME

        context_path = f"/local_disk0/.ephemeral_nfs/context/{self.context_key}"
        tmp_path = context_path + "/tmp/"
        libs_path = context_path + "/python/lib/site-packages"

        self._remote_metadata = RemoteMetadata(
            context_path=context_path,
            tmp_path=tmp_path,
            libs_path=libs_path
        )

        return self._remote_metadata

    @property
    def requirements(self):
        if self._requirements is not None:
            return self._requirements

        command = f"uv pip --directory {str(self.remote_metadata.libs_path)!r} list --format=json"

        try:
            reqs = self.command(
                command_str=command,
                language="shell",
            ).start().result()

            self._requirements = [
                (kw["name"], kw["version"])
                for kw in reqs
            ]
        except Exception as e:
            if "exit code 2" in str(e):
                self._requirements = []
            else:
                raise e

        return self._requirements

    # ------------ internal helpers ------------
    def create(
        self,
        *,
        language: "Language",
        context_key: Optional[str] = None,
        wait: WaitingConfigArg = True,
        ephemeral: bool = True
    ) -> "ExecutionContext":
        """Create a command execution context, retrying if needed.

        Args:
            language: The Databricks command language to use.
            context_key: Constant string key value
            wait: Waiting config to update

        Returns:
            The created command execution context response.
        """
        if self.context_id and self.language == language:
            return self

        client = self.client.workspace_client().command_execution

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    client.create,
                    cluster_id=self.cluster_id,
                    language=language,
                )

                try:
                    created = fut.result(timeout=10).response
                except TimeoutError:
                    self.cluster.ensure_running(wait=True)

                    created = client.create(
                        cluster_id=self.cluster_id,
                        language=language,
                    ).response
        except Exception as e:
            LOGGER.warning(e)

            self.cluster.ensure_running(wait=True)

            created = client.create(
                cluster_id=self.cluster_id,
                language=language,
            ).response

        instance = ExecutionContext(
            cluster=self.cluster,
            context_id=created.id,
            context_key=context_key or self.context_key or os.urandom(8).hex(),
            language=language,
            ephemeral=ephemeral
        )

        return instance

    def connect(
        self,
        *,
        language: Optional[Language] = None,
        wait: WaitingConfigArg = True,
        reset: bool = False,
    ) -> "ExecutionContext":
        """Create a remote command execution context if not already open.

        Args:
            language: Optional language override for the context.
            wait: Wait config
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

        return self.create(
            language=language,
            context_key=self.context_key,
            wait=wait
        )

    def close(
        self,
        wait: bool = True,
        raise_error: bool = True
    ) -> None:
        """Destroy the remote command execution context if it exists.

        Returns:
            None.
        """
        if not self.context_id:
            return

        client = self.client.workspace_client()

        try:
            if wait:
                client.command_execution.destroy(
                    cluster_id=self.cluster.cluster_id,
                    context_id=self.context_id,
                )
            else:
                Job.make(
                    client.command_execution.destroy,
                    cluster_id=self.cluster_id,
                    context_id=self.context_id
                ).fire_and_forget()
        except DatabricksError:
            if raise_error:
                raise
        finally:
            self.context_id = None

    # ------------ public API ------------
    def syspath_lines(self, environ: Optional[Mapping[str, str]]) -> str:
        if environ:

            return f"""import base64, os, traceback, json, sys
_p = os.path.expanduser({self.remote_metadata.libs_path!r})
if _p not in sys.path:
    sys.path.insert(0, _p)
_env = {environ!r} or {{}}"""
        else:
            return f"""import base64, os, traceback, json, sys
_p = os.path.expanduser({self.remote_metadata.libs_path!r})
if _p not in sys.path:
    sys.path.insert(0, _p)"""
    
    def command(
        self,
        command: Optional[str | Callable] = None,
        *,
        command_str: Optional[str] = None,
        language: Optional[Language | Literal["python", "r", "sql", "scala", "shell"]] = None,
        context: Optional["ExecutionContext"] = None,
        command_id: Optional[str] = None,
        func: Optional[Callable] = None,
        environ: Optional[Mapping] = None
    ) -> "CommandExecution":
        from .command_execution import CommandExecution

        context = self if context is None else context

        out_environ = {
            k: v
            for k, v in os.environ.items()
            if k not in EXCLUDED_ENV_KEYS
        }

        if environ:
            if not isinstance(environ, Mapping):
                out_environ.update({
                    k: os.getenv(k)
                    for k in (str(_) for _ in environ if _)
                    if k and k not in EXCLUDED_ENV_KEYS and os.getenv(k)
                })
            else:
                out_environ.update({
                    str(k): str(v)
                    for k, v in environ.items()
                    if v is not None and k not in EXCLUDED_ENV_KEYS
                })

        if isinstance(command, str):
            command_str = command
        elif callable(command):
            func = command

        if language == "shell":
            language = Language.PYTHON
            command_str = f"""
import subprocess, sys, shlex, pathlib

cmd = shlex.split({str(command_str)!r})
cmd = [str(pathlib.Path(arg).expanduser()) if arg.startswith("~/") else arg for arg in cmd]

p = subprocess.run(cmd, text=True, capture_output=True)

if p.returncode != 0:
    raise RuntimeError(
        f"Command {{cmd}} failed with exit code {{p.returncode}}:\\n"
        f"stderr: {{p.stderr.strip()}}"
    )
"""
        elif isinstance(language, str):
            language = Language[language]

        if language == Language.PYTHON and command_str:
            command_str = self.syspath_lines(out_environ) + "\n" + command_str

        return CommandExecution(
            context=context,
            command_id=command_id,
            language=language,
            command=command_str,
            pyfunc=func,
            environ=out_environ
        )

    def make_python_function_command(
        self,
        func: Callable,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        tag: str = "__CALL_RESULT__",
        environ: Optional[Mapping] = None
    ) -> str:
        """
        Build a remote Python snippet that:
          - loads a CommandExecution proxy
          - loads args/kwargs
          - decodes the function payload + args/kwargs payloads
          - executes
          - ALWAYS prints a tagged payload (success or error)
          - restores environment variables afterwards

        Notes:
          - protocol=5 for perf + modern compatibility
          - strongly prefer byref=True for cross Python minor stability
        """
        from yggdrasil.pickle.ser import dumps

        payload = [func, args or (), kwargs or {}]
        payload = dumps(payload, b64=True)

        cmd = f"""\
{self.syspath_lines(environ)}

from yggdrasil.pickle.ser import loads, dumps
f,a,k=loads({payload!r})
r = dumps(f(*a, **k), b64=True)
print({tag!r} + r, flush=True)
"""
        return cmd

    def is_in_databricks_environment(self):
        """Return True when running on a Databricks runtime."""
        return self.cluster.client.is_in_databricks_environment()
