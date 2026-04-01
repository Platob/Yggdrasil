"""Remote execution helpers for Databricks command contexts."""
import base64
import dataclasses as dc
import gzip
import inspect
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Optional, Any, Callable, Dict, Union, Literal, TypeVar, \
    Mapping

from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.compute import Language

from yggdrasil.concurrent.threading import Job
from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io.headers import DEFAULT_HOSTNAME
from yggdrasil.io.url import URL
from yggdrasil.pickle.ser import dumps, Serialized

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
    '_OLD_VIRTUAL_PROMPT', '_PIP_USE_IMPORTLIB_METADATA', '_RJEM_MALLOC_CONF', 'container',
    "UV", "UV_PATH", "UV_BIN", "UV_RUN_RECURSION_DEPTH",
])

EXCLUDED_ENV_PREFIXES = (
    "ARM_",
    "DATABRICKS_",
    "SPARK",
    "PYSPARK",
    "PYTHON",
    "PYTEST",
    "PYDEVD",
    "PYCHARM",
    "VSCODE",
    "MLFLOW",
    "TS_",
    "EFC_",
    "GIT_",
    "GITGUARDIAN_",
    "COMMONPROGRAMFILES",
    "FPS_BROWSER",
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


def _normalize_call_args(
    func: Callable,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
) -> tuple[tuple, dict]:
    args = tuple(args or ())
    kwargs = dict(kwargs or {})

    # For bound methods the normal inspect.unwrap() does not cross the
    # MethodType boundary, so __wrapped__ on __func__ is never followed.
    # Unwrap __func__ explicitly so that @wraps-decorated methods expose
    # the original function's signature (with real defaults), not the
    # wrapper's generic (*args, **kwargs).
    from types import MethodType as _MethodType  # noqa: PLC0415
    if isinstance(func, _MethodType):
        unwrapped_func = inspect.unwrap(func.__func__)
        # Re-bind so inspect.signature strips the leading `self` correctly.
        target = _MethodType(unwrapped_func, func.__self__)
    else:
        target = inspect.unwrap(func)

    try:
        sig = inspect.signature(target)
    except (ValueError, TypeError):
        # Signature not introspectable – pass args through unchanged.
        return tuple(args), dict(kwargs)

    # If the effective signature is only (*args, **kwargs) we cannot expand
    # defaults (wrapper whose inner signature is unknown at this point).
    # Return as-is; the remote function still has its own defaults stored in
    # the payload (fixed in _dump_function_payload to use inner_fn's defaults).
    _params = list(sig.parameters.values())
    _only_var = all(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in _params
    )
    if _only_var:
        return tuple(args), dict(kwargs)

    try:
        # Use partial binding so we only normalize arguments already supplied.
        # This avoids manufacturing a local TypeError for callables that still
        # rely on runtime-provided args or decorator/partial behavior.
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except TypeError:
        # Signature expansion is best-effort only. If introspection disagrees
        # with the callable's real invocation behavior, preserve the original
        # call shape and let the remote execution raise the underlying error.
        return tuple(args), dict(kwargs)

    out_args: list = []
    out_kwargs: dict = {}

    for name, param in sig.parameters.items():
        if name not in bound.arguments and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            out_args.append(bound.arguments[name])

        elif param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if name in kwargs:
                out_kwargs[name] = bound.arguments[name]
            else:
                out_args.append(bound.arguments[name])

        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            out_args.extend(bound.arguments.get(name, ()))

        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            if name in bound.arguments:
                out_kwargs[name] = bound.arguments[name]

        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            out_kwargs.update(bound.arguments.get(name, {}))

    return tuple(out_args), out_kwargs


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

        command = f"uv pip list --format=json"

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
                except FuturesTimeoutError:
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
        remote execution context. It:

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
import base64, gzip, os, traceback, json, sys, pandas as pd, numpy as np
_p = os.path.expanduser({self.remote_metadata.libs_path!r})
if _p not in sys.path:
    sys.path.insert(0, _p)"""

        if environ:
            env_json = json.dumps({str(k): "" if v is None else str(v) for k, v in dict(environ).items()})
            env_b64_gzip = base64.b64encode(gzip.compress(env_json.encode("utf-8"))).decode("ascii")

            lines += f"""
_env = json.loads(
    gzip.decompress(
        base64.b64decode({env_b64_gzip!r}.encode("ascii"))
    ).decode("utf-8")
)
for _k, _v in _env.items():
    os.environ[_k] = _v"""

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

    def make_python_function_command(
        self,
        serfunc: Serialized,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        tag: str = "__CALL_RESULT__",
        environ: Optional[Mapping] = None,
        pyfunc: Optional[Callable] = None,
    ) -> tuple[str, Optional[str]]:
        # Prefer the live callable (which still has __wrapped__ on @wraps
        # wrappers) for signature inspection.  The deserialized copy loses
        # __wrapped__ after the marshal round-trip, causing _normalize_call_args
        # to see (*args, **kwargs) and skip default expansion.
        func = pyfunc if pyfunc is not None else serfunc.as_cache_python()
        args, kwargs = _normalize_call_args(func, args=args, kwargs=kwargs)

        # Use the pre-serialized function (serfunc) in the payload instead of
        # re-serializing the live callable.  serfunc was created via
        # serialize(pyfunc) and is already a FunctionSerialized/MethodSerialized
        # that round-trips correctly.  Re-serializing `func` from scratch can
        # produce a broken payload when the callable is a wrapper, partial, or
        # other non-plain FunctionType (falls through to PickleSerialized which
        # may deserialize as a tuple on the remote cluster).
        payload: str = dumps([serfunc, args, kwargs], b64=True)
        remote_payload_path: Optional[str] = None

        if len(payload) > 900000:
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
            payload_literal = f"DBXPATH:{remote_payload_path}"
        else:
            payload_literal = payload

        cmd = f"""\
{self.syspath_lines(environ)}
 
from yggdrasil.pickle.ser import loads, dumps
_DBXPATH_PREFIX = "DBXPATH:"
_MAX_INLINE = 900000
_raw = {payload_literal!r}
 
if _raw.startswith(_DBXPATH_PREFIX):
    from yggdrasil.databricks.client import DatabricksClient as _DBC
    _client = _DBC.current()
    _path = _raw[len(_DBXPATH_PREFIX):]
    _raw = _client.dbfs_path(_path, temporary=True).read_bytes().decode("ascii")
 
_f, _a, _k = loads(_raw)
_r = dumps(_f(*list(_a), **_k), b64=True)
 
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