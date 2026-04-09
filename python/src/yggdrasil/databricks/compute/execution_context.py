from __future__ import annotations

import datetime as dt
import base64
import dataclasses as dc
import gzip
import inspect
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Optional, TypeVar

from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.compute import Language

from yggdrasil.concurrent.threading import Job
from yggdrasil.dataclasses import (
    WaitingConfigArg,
    restore_dataclass_state,
    serialize_dataclass_state,
)
from yggdrasil.environ import shutdown as yg_shutdown
from yggdrasil.io.headers import DEFAULT_HOSTNAME
from yggdrasil.io.url import URL
from yggdrasil.pickle.ser import Serialized, dumps

if TYPE_CHECKING:
    from .cluster import Cluster
    from .command_execution import CommandExecution

__all__ = [
    "ExecutionContext",
    "RemoteMetadata",
    "ContextPoolKey",
    "exclude_env_key",
    "close_all_pooled_contexts",
]

LOGGER = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


# ============================================================================
# Environment filtering
# ============================================================================

EXCLUDED_ENV_EXACT_KEYS = frozenset([
    "ALLUSERSPROFILE", "APPDATA", "ARM_CLIENT_ID", "ARM_CLIENT_SECRET", "ARM_ENVIRONMENT",
    "ARM_RESOURCE_ID", "ARM_TENANT_ID", "CLASSPATH", "CLICOLOR", "CLICOLOR_FORCE",
    "CLUSTER_DB_HOME", "COMMONPROGRAMFILES", "COMMONPROGRAMFILES(X86)", "COMMONPROGRAMW6432",
    "COMPUTERNAME", "COMSPEC", "DATABRICKS_CLUSTER_ID",
    "DATABRICKS_CLUSTER_LIBS_PYTHON_ROOT_DIR", "DATABRICKS_CLUSTER_LIBS_ROOT_DIR",
    "DATABRICKS_CLUSTER_LIBS_R_ROOT_DIR", "DATABRICKS_HOST", "DATABRICKS_INSTANCE_ID",
    "DATABRICKS_LIBS_NFS_ROOT_DIR", "DATABRICKS_LIBS_NFS_ROOT_PATH",
    "DATABRICKS_ROOT_VIRTUALENV_ENV", "DATABRICKS_RUNTIME_VERSION", "DATABRICKS_TOKEN",
    "DATA_SECURITY_MODE", "DBX_WORKSPACE_URL", "DB_HOME", "DEBUGINFOD_URLS",
    "DEFAULT_DATABRICKS_ROOT_VIRTUALENV_ENV", "DEFAULT_PYTHON_ENVIRONMENT",
    "DISABLE_LOCAL_FILESYSTEM", "DRIVERDATA", "DRIVER_PID_FILE", "DRIVER_REPL_ID",
    "DRIVER_STARTUP_OBSERVABILITY_ENABLED", "ENABLE_APPCDS", "ENABLE_CLASSLOADING_LOGS",
    "ENABLE_COMMAND_OUTPUT_TRUNCATION", "ENABLE_DRIVER_DEVELOPER_MODE", "ENABLE_IPTABLES",
    "ENABLE_KEEPALIVE_COMMAND_CONTEXT", "ENABLE_REPL_LOGGING",
    "ENABLE_TRACEPARENT_REPL_PROPAGATION", "FORCE_COLOR", "GIT_PAGER", "GRPC_GATEWAY_TOKEN",
    "HF_DATASETS_CACHE", "HIVE_HOME", "HOME", "HOMEDRIVE", "HOMEPATH", "HTTPS_PROXY",
    "HTTP_PROXY", "ICU_DATA", "IDE_PROJECT_ROOTS", "IPYTHONENABLE", "JAVA_HOME", "JAVA_OPTS",
    "JUPYTER_WIDGETS_ECHO", "KOALAS_USAGE_LOGGER", "LANG", "LIBRARY_ROOTS", "LOCALAPPDATA",
    "LOGNAME", "LOGONSERVER", "MAIL", "MASTER", "MLFLOW_CONDA_HOME",
    "MLFLOW_DEPLOYMENTS_TARGET", "MLFLOW_GATEWAY_URI", "MLFLOW_PYTHON_EXECUTABLE",
    "MLFLOW_REGISTRY_URI", "MLFLOW_TRACKING_URI", "MPLBACKEND", "NO_PROXY", "OLDPWD",
    "ONEDRIVE", "ONEDRIVECOMMERCIAL", "OS", "PAGER", "PATH", "PATHEXT", "PIP_NO_INPUT",
    "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "PROCESSOR_LEVEL", "PROCESSOR_REVISION",
    "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "PROMPT", "PS1",
    "PSMODULEPATH", "PUBLIC", "PWD", "PYCHARM_HELPERS_DIR", "PYCHARM_HOSTED",
    "PYDEVD_DISABLE_FILE_VALIDATION", "PYDEVD_INTERRUPT_THREAD_TIMEOUT",
    "PYDEVD_LOAD_VALUES_ASYNC", "PYDEVD_USE_FRAME_EVAL", "PYENV_ROOT",
    "PYSPARK_GATEWAY_PORT", "PYSPARK_GATEWAY_SECRET", "PYSPARK_PYTHON",
    "PYTEST_CURRENT_TEST", "PYTEST_RUN_CONFIG", "PYTEST_VERSION", "PYTHONHASHSEED",
    "PYTHONIOENCODING", "PYTHONPATH", "PYTHONUNBUFFERED", "R_LIBS", "SCALA_VERSION",
    "SESSIONNAME", "SHELL", "SHLVL", "SPARK_AUTH_SOCKET_TIMEOUT", "SPARK_BUFFER_SIZE",
    "SPARK_CONF_DIR", "SPARK_DIST_CLASSPATH", "SPARK_ENV_LOADED", "SPARK_HOME",
    "SPARK_LOCAL_DIRS", "SPARK_LOCAL_IP", "SPARK_PUBLIC_DNS", "SPARK_SCALA_VERSION",
    "SPARK_WORKER_MEMORY", "SUDO_COMMAND", "SUDO_GID", "SUDO_UID", "SUDO_USER",
    "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TERM", "TMP", "USER", "USERNAME",
    "USERPROFILE", "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "WINDIR", "container",
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
    """
    Return ``True`` when an environment-variable key should not be forwarded to
    remote execution payloads.

    The filter intentionally excludes:
    - host-specific process/runtime variables
    - Databricks / Spark / Python launcher internals
    - credentials or deployment-specific configuration
    - IDE / debugger / test-runner state

    Parameters
    ----------
    key:
        Environment-variable key to evaluate.

    Returns
    -------
    bool
        Whether the variable should be excluded from remote propagation.
    """
    normalized = str(key).upper()
    return (
        normalized in EXCLUDED_ENV_EXACT_KEYS
        or normalized.startswith(EXCLUDED_ENV_PREFIXES)
    )


def _build_forwarded_environ(
    environ: Optional[Mapping | list | tuple | set] = None,
) -> dict[str, str]:
    """
    Build the environment mapping forwarded to remote commands.

    The default baseline is the current process environment with excluded keys
    removed. Additional values may be supplied either as:

    - a mapping of explicit key/value pairs
    - an iterable of key names to copy from ``os.environ``

    ``None`` values are normalized to empty strings so the remote side can
    still receive an explicit key assignment.
    """
    forwarded: dict[str, str] = {
        str(k): str(v)
        for k, v in os.environ.items()
        if not exclude_env_key(k)
    }

    if environ is None:
        return forwarded

    if isinstance(environ, Mapping):
        forwarded.update({
            str(k): "" if v is None else str(v)
            for k, v in environ.items()
            if not exclude_env_key(str(k))
        })
        return forwarded

    forwarded.update({
        key: os.getenv(key, "")
        for key in (str(item) for item in environ if item)
        if not exclude_env_key(key)
    })
    return forwarded


# ============================================================================
# Context pool state
# ============================================================================

@dc.dataclass(frozen=True, slots=True)
class ContextPoolKey:
    """
    Stable key for process-global pooled execution contexts.

    A context is uniquely identified by the Databricks cluster, the execution
    language, and the logical ``context_key`` used to isolate remote state.
    """

    cluster_id: str
    language: str
    context_key: str


_CONTEXT_POOL: dict[ContextPoolKey, "ExecutionContext"] = {}
_CONTEXT_POOL_LOCK = threading.RLock()


# ============================================================================
# Idle-context reaper
# ============================================================================

_REAPER_INTERVAL: float = 60.0
_REAPER_THREAD: Optional[threading.Thread] = None
_REAPER_STOP = threading.Event()
_REAPER_LOCK = threading.Lock()


def _evict_idle_contexts() -> None:
    """
    Close and remove pooled contexts that have exceeded their idle timeout.

    Only contexts with a non-``None`` ``close_after`` value are eligible for
    automatic eviction. Contexts without an active ``context_id`` are skipped,
    since they are already effectively closed.
    """
    now = time.time()
    evicted: list[tuple[ContextPoolKey, "ExecutionContext"]] = []

    with _CONTEXT_POOL_LOCK:
        for key, ctx in list(_CONTEXT_POOL.items()):
            if ctx.close_after is None:
                continue
            if not ctx.context_id:
                continue
            if ctx._last_used_at <= 0:
                continue

            idle_seconds = now - ctx._last_used_at
            if idle_seconds >= ctx.close_after:
                evicted.append((key, ctx))

        for key, _ctx in evicted:
            _CONTEXT_POOL.pop(key, None)

    for _key, ctx in evicted:
        prev_raise = logging.raiseExceptions
        logging.raiseExceptions = False
        try:
            LOGGER.info(
                "Auto-closing idle context %s (idle=%.0fs, close_after=%.0fs)",
                ctx.context_id,
                now - ctx._last_used_at,
                ctx.close_after,
            )
            ctx.close(wait=False, raise_error=False)
        except Exception:
            LOGGER.debug("Error auto-closing idle context %s", ctx, exc_info=True)
        finally:
            logging.raiseExceptions = prev_raise


def _reaper_loop() -> None:
    """
    Background loop that periodically scans the global pool for idle contexts.
    """
    while not _REAPER_STOP.wait(timeout=_REAPER_INTERVAL):
        try:
            _evict_idle_contexts()
        except Exception:
            LOGGER.debug("Unexpected error in context reaper loop", exc_info=True)


def _ensure_reaper_running() -> None:
    """
    Start the idle-context reaper thread if it is not already alive.
    """
    global _REAPER_THREAD

    with _REAPER_LOCK:
        if _REAPER_THREAD is not None and _REAPER_THREAD.is_alive():
            return

        _REAPER_STOP.clear()
        _REAPER_THREAD = threading.Thread(
            target=_reaper_loop,
            name="ygg-context-reaper",
            daemon=True,
        )
        _REAPER_THREAD.start()
        LOGGER.debug(
            "Context reaper thread started (interval=%.0fs)",
            _REAPER_INTERVAL,
        )


# ============================================================================
# Remote metadata
# ============================================================================

@dc.dataclass
class RemoteMetadata:
    """
    Resolved filesystem paths associated with a remote execution context.

    Attributes
    ----------
    context_path:
        Root working directory for the logical context.
    tmp_path:
        Temporary-file directory within the context root.
    libs_path:
        Site-packages directory used to inject custom Python dependencies.
    """

    context_path: str
    tmp_path: str
    libs_path: str


# ============================================================================
# Execution context
# ============================================================================

@dc.dataclass
class ExecutionContext:
    """
    Databricks command-execution context bound to a specific cluster.

    An ``ExecutionContext`` represents a live remote REPL/session created via
    the Databricks command execution API. It owns enough metadata to:

    - lazily connect or reconnect to a remote context
    - participate in a global in-process pool
    - build command payloads with controlled environment propagation
    - serialize Python callables and arguments for remote execution

    Notes
    -----
    - Instances are thread-safe at the object level via an internal ``RLock``.
    - Pooled contexts are shared per ``(cluster, language, context_key)``.
    - Temporary contexts register a shutdown hook so they are closed on process
      teardown when possible.
    """

    cluster: "Cluster"
    context_id: str = ""
    context_key: Optional[str] = dc.field(default=None, repr=False, compare=False, hash=False)
    language: Optional[Language] = dc.field(default=None, repr=False, compare=False, hash=False)
    temporary: bool = dc.field(default=False, repr=False, compare=False, hash=False)
    close_after: Optional[float] = dc.field(default=1800.0, repr=False, compare=False, hash=False)

    _remote_metadata: Optional[RemoteMetadata] = dc.field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _lock: threading.RLock = dc.field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _created_at: float = dc.field(default=0.0, init=False, repr=False, compare=False, hash=False)
    _last_used_at: float = dc.field(default=0.0, init=False, repr=False, compare=False, hash=False)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __getstate__(self):
        """
        Serialize dataclass state while excluding runtime-only lock objects.
        """
        return serialize_dataclass_state(self)

    def __setstate__(self, state):
        """
        Restore dataclass state and recreate the local synchronization lock.
        """
        restore_dataclass_state(self, state)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExecutionContext":
        """
        Open the context on entry, defaulting to Python when no language is set.
        """
        return self.connect(language=self.language or Language.PYTHON)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Automatically close temporary contexts when leaving a ``with`` block.
        """
        if self.temporary:
            self.close(wait=False, raise_error=False)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<url={self.url()!r}>"

    def __str__(self) -> str:
        return self.__repr__()

    def url(self) -> URL:
        """
        Return a context-scoped URL for diagnostics and logging.
        """
        return self.cluster.url().with_query_items({
            "context": self.context_id or "unknown",
        })

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def client(self):
        """
        Convenience proxy to the cluster client.
        """
        return self.cluster.client

    @property
    def cluster_id(self):
        """
        Convenience proxy to the underlying Databricks cluster id.
        """
        return self.cluster.cluster_id

    @property
    def remote_metadata(self) -> RemoteMetadata:
        """
        Lazily resolve remote filesystem paths for this context.

        The path layout is derived from ``context_key``. If no explicit
        ``context_key`` is set, the default hostname-based key is used.
        """
        if self._remote_metadata is not None:
            return self._remote_metadata

        if not self.context_key:
            self.context_key = str(DEFAULT_HOSTNAME)

        context_path = f"/local_disk0/.ephemeral_nfs/context/{self.context_key}"
        self._remote_metadata = RemoteMetadata(
            context_path=context_path,
            tmp_path=f"{context_path}/tmp/",
            libs_path=f"{context_path}/python/lib/site-packages",
        )
        return self._remote_metadata

    @property
    def created_at(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(
            self._created_at,
            tz=dt.timezone.utc,
        )

    @property
    def last_used_at(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(
            self._last_used_at,
            tz=dt.timezone.utc,
        )

    # ------------------------------------------------------------------
    # Pool helpers
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """
        Record recent use of this context.

        This timestamp is used by the idle-context reaper to determine when a
        pooled context should be evicted and closed.
        """
        self._last_used_at = time.time()

    @classmethod
    def _pool_key(
        cls,
        *,
        cluster_id: str,
        language: Language,
        context_key: Optional[str],
    ) -> ContextPoolKey:
        """
        Build the canonical pool key for a cluster/language/context tuple.
        """
        return ContextPoolKey(
            cluster_id=str(cluster_id),
            language=language.value,
            context_key=str(context_key or DEFAULT_HOSTNAME),
        )

    @classmethod
    def get_or_create(
        cls,
        *,
        cluster: "Cluster",
        language: Language = Language.PYTHON,
        context_key: Optional[str] = None,
        temporary: bool = False,
        reset: bool = False,
        close_after: Optional[float] = 1800.0,
    ) -> "ExecutionContext":
        """
        Return a pooled execution context, creating it if needed.

        The pool is process-global and keyed by ``(cluster, language,
        context_key)``. Concurrent callers targeting the same key will share
        the same ``ExecutionContext`` instance.

        Parameters
        ----------
        cluster:
            Cluster that owns the remote command-execution context.
        language:
            Execution language for the remote context.
        context_key:
            Logical identifier used to isolate remote filesystem state.
        temporary:
            Whether the created context should register shutdown cleanup.
        reset:
            Whether to force-close the existing pooled context before reuse.
        close_after:
            Idle timeout in seconds used by the background reaper. Set to
            ``None`` to disable automatic eviction for this pooled context.

        Returns
        -------
        ExecutionContext
            Connected context instance from the global pool.
        """
        key = cls._pool_key(
            cluster_id=cluster.cluster_id,
            language=language,
            context_key=context_key,
        )

        with _CONTEXT_POOL_LOCK:
            ctx = _CONTEXT_POOL.get(key)
            if ctx is None:
                ctx = cls(
                    cluster=cluster,
                    context_key=context_key or str(DEFAULT_HOSTNAME),
                    language=language,
                    temporary=temporary,
                    close_after=close_after,
                )
                _CONTEXT_POOL[key] = ctx
                if close_after is not None:
                    _ensure_reaper_running()

        with ctx._lock:
            if reset:
                ctx.close(wait=False, raise_error=False)

            ctx.connect(language=language, reset=False)
            ctx.touch()
            return ctx

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        language: Language,
        context_key: Optional[str] = None,
        wait: WaitingConfigArg = True,
        temporary: bool = False,
    ) -> "ExecutionContext":
        """
        Create a new remote execution context.

        If the current instance already owns a live context with the requested
        language, it is reused and only the last-used timestamp is refreshed.

        Parameters
        ----------
        language:
            Databricks command-execution language for the remote context.
        context_key:
            Optional logical key used to derive remote filesystem paths.
        wait:
            Reserved lifecycle option for API consistency.
        temporary:
            Whether this context should be treated as short-lived and cleaned
            up automatically during shutdown.

        Returns
        -------
        ExecutionContext
            The current instance for fluent chaining.
        """
        del wait  # kept for API compatibility

        with self._lock:
            if self.context_id and self.language == language:
                self.touch()
                return self

            client = self.client.workspace_client().command_execution
            LOGGER.info(
                "Creating %s context on %s (key=%s)",
                language.value,
                self.cluster,
                context_key or self.context_key,
            )

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        client.create,
                        cluster_id=self.cluster_id,
                        language=language,
                    )
                    try:
                        created = future.result(timeout=10).response
                    except FuturesTimeoutError:
                        LOGGER.warning(
                            "Context creation timed out for %s — ensuring cluster running and retrying",
                            self.cluster,
                        )
                        self.cluster.ensure_running(wait=True)
                        created = client.create(
                            cluster_id=self.cluster_id,
                            language=language,
                        ).response
            except Exception as exc:
                LOGGER.warning(
                    "Context creation failed for %s — ensuring cluster running and retrying: %s",
                    self.cluster,
                    exc,
                )
                self.cluster.ensure_running(wait=True)
                created = client.create(
                    cluster_id=self.cluster_id,
                    language=language,
                ).response

            self.context_id = created.id
            self.language = language
            self.context_key = context_key or self.context_key or os.urandom(8).hex()
            self.temporary = temporary
            self._created_at = time.time()
            self.touch()

            LOGGER.info("Context created: id=%s on %s", self.context_id, self.cluster)

            if self.temporary:
                try:
                    yg_shutdown.register(self._unsafe_close)
                except Exception:
                    LOGGER.debug(
                        "Failed to register shutdown handler for context %s",
                        self.context_id,
                        exc_info=True,
                    )

            return self

    def connect(
        self,
        *,
        language: Optional[Language] = None,
        wait: WaitingConfigArg = True,
        reset: bool = False,
    ) -> "ExecutionContext":
        """
        Ensure this instance has an open remote execution context.

        Parameters
        ----------
        language:
            Desired execution language. Falls back to the existing instance
            language, then Python.
        wait:
            Reserved lifecycle option for API consistency.
        reset:
            When ``True``, force-close the current remote context before
            creating a replacement.

        Returns
        -------
        ExecutionContext
            The current instance for fluent chaining.
        """
        with self._lock:
            if self.context_id and not reset:
                LOGGER.debug(
                    "%s already connected (context=%s), reusing",
                    self,
                    self.context_id,
                )
                self.touch()
                return self

            if self.context_id and reset:
                LOGGER.info("%s resetting connection", self)
                self.close(wait=False, raise_error=False)

            resolved_language = language or self.language or Language.PYTHON
            LOGGER.debug("%s connecting with language=%s", self, resolved_language.value)

            return self.create(
                language=resolved_language,
                context_key=self.context_key,
                wait=wait,
                temporary=self.temporary,
            )

    def close(self, wait: bool = True, raise_error: bool = True) -> None:
        """
        Destroy the remote execution context associated with this instance.

        Parameters
        ----------
        wait:
            When ``True``, block until Databricks confirms destruction.
            Otherwise, dispatch the destroy call asynchronously.
        raise_error:
            Whether ``DatabricksError`` exceptions should be propagated.
        """
        with self._lock:
            if not self.context_id:
                return

            closing_id = self.context_id
            LOGGER.info(
                "Closing context %s on %s (wait=%s)",
                closing_id,
                self.cluster,
                wait,
            )

            client = self.client.workspace_client()
            try:
                if wait:
                    client.command_execution.destroy(
                        cluster_id=self.cluster.cluster_id,
                        context_id=closing_id,
                    )
                else:
                    Job.make(
                        client.command_execution.destroy,
                        cluster_id=self.cluster.cluster_id,
                        context_id=closing_id,
                    ).fire_and_forget()
            except DatabricksError:
                if raise_error:
                    raise
                LOGGER.debug(
                    "Suppressed DatabricksError while closing context %s",
                    closing_id,
                )
            finally:
                LOGGER.debug("Context %s closed", closing_id)
                self.context_id = ""
                if self.temporary:
                    try:
                        yg_shutdown.unregister(self._unsafe_close)
                    except Exception:
                        pass

    def _unsafe_close(self) -> None:
        """
        Shutdown-safe close helper used by process-exit hooks.

        This method always performs a blocking close so it does not spawn extra
        threads during interpreter teardown. Logging exception banners are
        suppressed because log streams may already be partially dismantled at
        shutdown time.
        """
        prev_raise = logging.raiseExceptions
        logging.raiseExceptions = False
        try:
            self.close(wait=True, raise_error=False)
        except Exception:
            pass
        finally:
            logging.raiseExceptions = prev_raise

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------

    def syspath_lines(self, environ: Optional[Mapping[str, str]] = None) -> str:
        """
        Build the Python preamble injected into Python commands.

        The preamble:
        - prepends the remote context-specific site-packages directory to
          ``sys.path``
        - optionally injects environment variables into ``os.environ``

        Environment values are compressed and base64-encoded so they can be
        embedded safely into the generated command string.
        """
        lines = f"""\
import base64, gzip, os, json, sys, pandas as pd, numpy as np
_p = os.path.expanduser({self.remote_metadata.libs_path!r})
if _p not in sys.path:
    sys.path.insert(0, _p)"""

        if environ:
            env_json = json.dumps({
                str(k): "" if v is None else str(v)
                for k, v in dict(environ).items()
            })
            env_b64_gzip = base64.b64encode(
                gzip.compress(env_json.encode("utf-8"))
            ).decode("ascii")
            lines += f"""
_env = json.loads(gzip.decompress(base64.b64decode({env_b64_gzip!r}.encode("ascii"))).decode("utf-8"))
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
        environ: Optional[Mapping] = None,
    ) -> "CommandExecution":
        """
        Build a :class:`CommandExecution` for this context.

        Parameters
        ----------
        command:
            Either a command string or a Python callable.
        command_str:
            Explicit command string. Used when ``command`` is not a string.
        language:
            Execution language. The special value ``"shell"`` is translated into
            a Python wrapper that executes a subprocess remotely.
        context:
            Alternate context to bind to the resulting command object.
        command_id:
            Optional command identifier.
        func:
            Explicit callable to execute remotely.
        environ:
            Additional environment overrides or environment keys to forward.

        Returns
        -------
        CommandExecution
            Configured command-execution wrapper.
        """
        from .command_execution import CommandExecution

        context = self if context is None else context
        out_environ = _build_forwarded_environ(environ)

        if isinstance(command, str):
            command_str = command
        elif callable(command):
            func = command

        if language == "shell":
            language = Language.PYTHON
            command_str = f"""
import subprocess, shlex, pathlib
cmd = shlex.split({str(command_str)!r})
cmd = [str(pathlib.Path(arg).expanduser()) if arg.startswith("~/") else arg for arg in cmd]
p = subprocess.run(cmd, text=True, capture_output=True)
if p.returncode != 0:
    raise RuntimeError(f"Command {{cmd}} failed (exit {{p.returncode}}):\\nstderr: {{p.stderr.strip()}}")
print(p.stdout, flush=True)
"""
        elif isinstance(language, str):
            language = Language[language.upper()]

        resolved_language = language or context.language or Language.PYTHON

        if resolved_language == Language.PYTHON and command_str:
            command_str = self.syspath_lines(out_environ) + "\n" + command_str

        return CommandExecution(
            context=context,
            command_id=command_id,
            language=resolved_language,
            command=command_str,
            pyfunc=func,
            environ=out_environ,
        )

    # ------------------------------------------------------------------
    # Python callable dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_call_args(
        func: Callable,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
    ) -> tuple[tuple, dict]:
        """
        Normalize positional and keyword arguments against a callable signature.

        This expands default values when possible so remote invocation more
        closely mirrors local call behavior, especially for partially bound
        methods and functions with mixed positional/keyword parameters.
        """
        args = tuple(args or ())
        kwargs = dict(kwargs or {})

        from types import MethodType as _MethodType

        if isinstance(func, _MethodType):
            target: Callable = _MethodType(
                inspect.unwrap(func.__func__),
                func.__self__,
            )
        else:
            target = inspect.unwrap(func)

        try:
            sig = inspect.signature(target)
        except (ValueError, TypeError):
            return args, kwargs

        params = list(sig.parameters.values())
        if all(
            p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            for p in params
        ):
            return args, kwargs

        try:
            bound = sig.bind_partial(*args, **kwargs)
        except TypeError:
            return args, kwargs

        positional_names: list[str] = []
        arg_index = 0
        for param in params:
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if arg_index < len(args):
                    positional_names.append(param.name)
                    arg_index += 1
            elif param.kind is inspect.Parameter.VAR_POSITIONAL:
                break

        bound.apply_defaults()

        out_args: list[Any] = []
        out_kwargs: dict[str, Any] = {}

        for param in params:
            name = param.name

            if name not in bound.arguments and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                out_args.append(bound.arguments[name])

            elif param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                if name in positional_names:
                    out_args.append(bound.arguments[name])
                else:
                    out_kwargs[name] = bound.arguments[name]

            elif param.kind is inspect.Parameter.VAR_POSITIONAL:
                out_args.extend(bound.arguments.get(name, ()))

            elif param.kind is inspect.Parameter.KEYWORD_ONLY:
                if name in bound.arguments:
                    out_kwargs[name] = bound.arguments[name]

            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                out_kwargs.update(bound.arguments.get(name, {}))

        return tuple(out_args), out_kwargs

    def make_python_function_command(
        self,
        serfunc: Serialized,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        tag: str = "__CALL_RESULT__",
        environ: Optional[Mapping] = None,
        pyfunc: Optional[Callable] = None,
    ) -> tuple[str, Optional[str]]:
        """
        Build a Python command string that invokes a serialized callable remotely.

        The serialized function and its arguments are inlined when the payload is
        small enough. Large payloads are uploaded to temporary DBFS storage and
        referenced by path instead.

        Parameters
        ----------
        serfunc:
            Serialized callable wrapper.
        args, kwargs:
            Positional and keyword arguments passed to the callable.
        tag:
            Prefix used to mark the printed serialized return value.
        environ:
            Optional environment additions for the generated preamble.
        pyfunc:
            Original callable used for argument normalization when available.

        Returns
        -------
        tuple[str, Optional[str]]
            Generated command string and optional remote DBFS payload path.
        """
        func = pyfunc if pyfunc is not None else serfunc.as_cache_python()
        args, kwargs = self._normalize_call_args(func, args=args, kwargs=kwargs)

        payload: str = dumps([serfunc, args, kwargs], b64=True)
        remote_payload_path: Optional[str] = None

        if len(payload) > 900_000:
            tmp = self.client.tmp_path(
                suffix="-pyfunc",
                extension="b64",
                max_lifetime=3600,
            )
            tmp.write_bytes(
                payload.encode("ascii") if isinstance(payload, str) else payload
            )
            remote_payload_path = str(tmp)
            LOGGER.debug(
                "Payload too large (%d bytes) — uploaded to DBFS: %s",
                len(payload),
                remote_payload_path,
            )
            payload_literal = f"DBXPATH:{remote_payload_path}"
        else:
            payload_literal = payload

        command = f"""\
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
_r = dumps(_f(*_a, **_k), b64=True)
if len(_r) > _MAX_INLINE:
    from yggdrasil.databricks.client import DatabricksClient as _DBC
    _client = _DBC.current()
    _tmp = _client.tmp_path(suffix="-pyresult", extension="b64", max_lifetime=3600)
    _tmp.write_bytes(_r.encode("ascii") if isinstance(_r, str) else _r)
    _r = _DBXPATH_PREFIX + str(_tmp)
print({tag!r} + _r, flush=True)
"""
        return command, remote_payload_path

    def is_in_databricks_environment(self) -> bool:
        """
        Return whether the current client is running inside a Databricks
        environment.
        """
        return self.cluster.client.is_in_databricks_environment()


# ============================================================================
# Pool shutdown helpers
# ============================================================================

def close_all_pooled_contexts() -> None:
    """
    Close and remove every context currently registered in the global pool.

    This helper uses ``wait=True`` for every context so shutdown is synchronous
    and does not rely on background threads. That makes it safe for use in
    interpreter shutdown and signal-handling paths.
    """
    with _CONTEXT_POOL_LOCK:
        contexts = list(_CONTEXT_POOL.values())
        _CONTEXT_POOL.clear()

    for ctx in contexts:
        try:
            ctx.close(wait=True, raise_error=False)
        except Exception:
            LOGGER.debug("Failed closing pooled context %s", ctx, exc_info=True)


yg_shutdown.register(close_all_pooled_contexts)