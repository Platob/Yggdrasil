"""Remote execution helpers for Databricks command contexts."""

import dataclasses as dc
import inspect
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
    "exclude_env_key"
]

EXCLUDED_ENV_EXACT_KEYS = frozenset([
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

EXCLUDED_ENV_PREFIXES = (
    "ARM_",
    "DATABRICKS_",
    "SPARK_",
    "PYSPARK_",
    "PYTHON_",
    "PYTEST_",
    "PYDEVD_",
    "PYCHARM_",
    "VSCODE_",
    "MLFLOW_",
    "TS_",
    "EFC_",
    "GIT_",
    "GITGUARDIAN_",
    "COMMONPROGRAMFILES",
    "FPS_BROWSER"
)

def exclude_env_key(key: str) -> bool:
    """Return True when an environment variable key should be excluded."""
    k = key.upper()

    return (
        k in EXCLUDED_ENV_EXACT_KEYS
        or k.startswith(EXCLUDED_ENV_PREFIXES)
    )


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


def _bind_call_args(
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> tuple[tuple, dict]:
    """Validate *args*/*kwargs* against *func*'s signature, fill in defaults.

    Parameters
    ----------
    func:
        The callable whose signature to inspect.
    args:
        Positional arguments intended for *func*.
    kwargs:
        Keyword arguments intended for *func*.

    Returns
    -------
    tuple[tuple, dict]
        ``(args, kwargs)`` with defaults for optional parameters filled in
        to the *kwargs* dict.

    Raises
    ------
    TypeError
        If required parameters are missing or unexpected keyword arguments
        are passed.
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        # Built-ins or C extensions may not expose a signature; skip validation.
        return args, kwargs

    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Invalid arguments for {getattr(func, '__qualname__', repr(func))!r}: {exc}"
        ) from exc

    bound.apply_defaults()

    # Reconstruct positional and keyword arguments from the bound mapping.
    # Parameters that were originally positional stay positional; the rest
    # (keyword-only + defaults that were filled in) go into kwargs.
    new_args: list = []
    new_kwargs: dict = {}

    for param_name, param in sig.parameters.items():
        if param_name not in bound.arguments:
            continue
        value = bound.arguments[param_name]
        kind = param.kind
        if kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            new_args.append(value)
        elif kind == inspect.Parameter.VAR_POSITIONAL:
            new_args.extend(value)
        elif kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        ):
            if kind == inspect.Parameter.VAR_KEYWORD:
                new_kwargs.update(value)
            else:
                new_kwargs[param_name] = value

    return tuple(new_args), new_kwargs


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
    def syspath_lines(self, environ: Optional[Mapping[str, str]] = None) -> str:
        """Build the preamble that sets ``sys.path`` and environment variables.

        The snippet is prepended to every Python command sent to the
        remote execution context.  It:

        1. Ensures the remote libs path is on ``sys.path``.
        2. Injects any extra environment variables into ``os.environ``.

        Parameters
        ----------
        environ:
            Optional mapping of environment variables to propagate.

        Returns
        -------
        str
            A multi-line Python snippet suitable for ``exec()``.
        """
        lines = f"""\
import base64, os, traceback, json, sys
_p = os.path.expanduser({self.remote_metadata.libs_path!r})
if _p not in sys.path:
    sys.path.insert(0, _p)"""

        if environ:
            lines += f"""
for _k, _v in {dict(environ)!r}.items():
    os.environ[_k] = str(_v)"""

        return lines

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
            if not exclude_env_key(k)
        }

        if environ:
            if not isinstance(environ, Mapping):
                out_environ.update({
                    k: os.getenv(k)
                    for k in (str(_) for _ in environ if _)
                    if k and not exclude_env_key(k)
                })
            else:
                out_environ.update({
                    str(k): str(v)
                    for k, v in environ.items()
                    if not exclude_env_key(k)
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

    # Prefix used to signal that the payload literal in the remote
    # command is a Databricks path to read from, not an inline value.
    _DBXPATH_PREFIX = "DBXPATH:"

    # Conservative payload-size threshold.  Databricks command execution
    # has a ~1 MB command-text limit; we leave headroom for the preamble
    # and wrapper code.
    _MAX_INLINE_PAYLOAD_BYTES = 900_000

    def make_python_function_command(
        self,
        func: Callable,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        tag: str = "__CALL_RESULT__",
        environ: Optional[Mapping] = None,
    ) -> tuple[str, Optional[str]]:
        """Build a remote Python command that executes a serialised callable.

        The generated snippet:

        1. Prepends ``sys.path`` / environment preamble.
        2. Checks whether the payload literal starts with ``DBXPATH:``.
           If so, reads the base64 payload from the Databricks path;
           otherwise treats it as an inline base64 string.
        3. Deserialises ``[func, args, kwargs]`` from the payload.
        4. Calls ``func(*args, **kwargs)``.
        5. Serialises the result (or the exception) and prints it after
           the *tag* sentinel so that
           :meth:`CommandExecution.decode_response` can split logs from
           the return value.

        The ``try/except`` wrapper guarantees the tag is **always**
        printed, even when the function raises.  The caller-side
        ``result()`` method deserialises the exception and re-raises it.

        When the serialised payload exceeds
        :attr:`_MAX_INLINE_PAYLOAD_BYTES` the payload is uploaded to a
        temporary Databricks path (Volumes / DBFS) and the command
        references it via the ``DBXPATH:`` prefix.  The caller is
        responsible for deleting this path after use (see
        :meth:`CommandExecution.result`).

        Parameters
        ----------
        func:
            The callable to execute remotely.
        args:
            Positional arguments for *func*.
        kwargs:
            Keyword arguments for *func*.
        tag:
            Sentinel string separating stdout logs from the serialised
            return value.
        environ:
            Extra environment variables to inject on the remote side.

        Returns
        -------
        tuple[str, str | None]
            ``(command_string, remote_payload_path_or_None)``.
            *remote_payload_path* is the Databricks path where the
            payload was uploaded, or ``None`` when the payload was
            embedded inline.

        Raises
        ------
        TypeError
            When *args* / *kwargs* do not satisfy *func*'s signature
            (missing required arguments, unexpected keyword arguments, etc.).
        """
        from yggdrasil.pickle.ser import dumps

        args, kwargs = _bind_call_args(func, args or (), kwargs or {})

        payload: str = dumps([func, args, kwargs], b64=True)
        remote_payload_path: Optional[str] = None

        if len(payload) > self._MAX_INLINE_PAYLOAD_BYTES:
            tmp = self.client.tmp_path(
                suffix="-pyfunc", extension="b64",
                max_lifetime=3600,
            )
            payload_bytes = (
                payload.encode("ascii")
                if isinstance(payload, str)
                else payload
            )
            tmp.write_bytes(payload_bytes)
            remote_payload_path = str(tmp)

            # The remote snippet receives a DBXPATH:-prefixed literal
            # and reads the actual base64 data from the path.
            payload_literal = f"{self._DBXPATH_PREFIX}{remote_payload_path}"
        else:
            payload_literal = payload

        cmd = f"""\
{self.syspath_lines(environ)}

from yggdrasil.pickle.ser import loads, dumps
_DBXPATH_PREFIX = {self._DBXPATH_PREFIX!r}
_MAX_INLINE = {self._MAX_INLINE_PAYLOAD_BYTES!r}
_raw = {payload_literal!r}

if _raw.startswith(_DBXPATH_PREFIX):
    from yggdrasil.databricks.client import DatabricksClient as _DBC
    _client = _DBC.current()
    _path = _raw[len(_DBXPATH_PREFIX):]
    _raw = _client.dbfs_path(_path).read_bytes().decode("ascii")
_f, _a, _k = loads(_raw)
_r = dumps(_f(*_a, **_k), b64=True)

if len(_r) > _MAX_INLINE:
    from yggdrasil.databricks.client import DatabricksClient as _DBC
    _client = _DBC.current()
    _tmp = _client.tmp_path(suffix="-pyresult", extension="b64", max_lifetime=3600)
    _tmp.write_bytes(_r.encode("ascii") if isinstance(_r, str) else _r)
    _r = _DBXPATH_PREFIX + str(_tmp)
    
print({tag!r} + _r, flush=True)
"""
        return cmd, remote_payload_path

    def is_in_databricks_environment(self):
        """Return True when running on a Databricks runtime."""
        return self.cluster.client.is_in_databricks_environment()
