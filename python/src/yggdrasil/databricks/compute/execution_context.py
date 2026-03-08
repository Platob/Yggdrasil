"""Remote execution helpers for Databricks command contexts."""

import base64
import dataclasses as dc
import gzip
import io
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional, Any, Callable, Dict, Union, Tuple, Literal, TypeVar, \
    Mapping, Generator, Iterator

from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.compute import Language, ResultType, CommandStatusResponse

from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.environ import PyEnv, UserInfo
from yggdrasil.io.url import URL
from yggdrasil.pyutils.exceptions import raise_parsed_traceback
from .command_execution import CommandExecution
from .exceptions import ClientTerminatedSession
from ...concurrent.threading import Job
from ...dataclasses.expiring import ExpiringDict

if TYPE_CHECKING:
    from .cluster import Cluster


__all__ = [
    "ExecutionContext"
]

LOGGER = logging.getLogger(__name__)
UPLOADED_PACKAGE_ROOTS: Dict[str, ExpiringDict] = {}
BytesLike = Union[bytes, bytearray, memoryview]
F = TypeVar("F", bound=Callable[..., Any])

_CTX_RUNTIME_FIELDS = frozenset({"_lock",})
_CTX_RESET_FIELDS = frozenset({"_remote_metadata"})


@dc.dataclass(frozen=True)
class BytesSource:
    """
    Hashable wrapper for in-memory content so it can be used as a dict key.

    name: only used for debugging / metadata (not required to match remote basename)
    data: bytes-like payload
    """
    name: str
    data: bytes

LocalSpec = Union[
    str,
    os.PathLike,
    bytes,                       # raw bytes as key (works, but no name)
    BytesSource,                 # recommended for buffers
    Tuple[str, BytesLike],       # (name, data) helper
]

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

    language: Optional[Language] = dc.field(default=None, repr=False, compare=False, hash=False)

    _remote_metadata: Optional[RemoteMetadata] = dc.field(default=None, repr=False, compare=False, hash=False)
    _requirements: Optional[list[tuple[str]]] = dc.field(default=None, repr=False, compare=False, hash=False)
    _pyenv_check_timestamp: int = dc.field(default=0, repr=False, compare=False, hash=False)

    _uploaded_package_roots: Optional[ExpiringDict] = dc.field(default_factory=ExpiringDict, repr=False, compare=False, hash=False)
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
            usr, env = UserInfo.current(), PyEnv.current()
            vinfo = env.version_info

            self.context_key = f"{usr.hostname}-py{vinfo.major}.{vinfo.minor}"

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
            language=language
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
    def syspath_lines(self):
        return "\n".join([
            "import os, sys",
            f"_p = os.path.expanduser({self.remote_metadata.libs_path!r})",
            "if _p not in sys.path:",
            "    sys.path.insert(0, _p)",
        ])
    
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
        context = self if context is None else context

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

print(p.stdout)

if p.returncode != 0:
    raise RuntimeError(
        f"Command {{cmd}} failed with exit code {{p.returncode}}:\\n"
        f"stderr: {{p.stderr.strip()}}"
    )
"""
        elif isinstance(language, str):
            language = Language[language]

        if language == Language.PYTHON and command_str:
            command_str = self.syspath_lines() + "\n" + command_str

        if environ:
            if not isinstance(environ, Mapping):
                environ = {
                    k: os.environ.get(k)
                    for k in environ
                    if k
                }

        return CommandExecution(
            context=context,
            command_id=command_id,
            language=language,
            command=command_str,
            pyfunc=func,
            environ=environ
        )

    def make_python_function_command(
        self,
        job: Job,
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
        import yggdrasil.pickle.dill as pkl

        command_b64: str = base64.b64encode(pkl.dumps(self, byref=True)).decode("ascii")
        job_b64: str = self.encode_object(job)
        environ = environ or {}
        if not isinstance(environ, Mapping):
            environ = {
                k: os.getenv(k)
                for k in environ
                if k and os.getenv(k)
            }

        cmd = f"""\
{self.syspath_lines()}

import base64, os, traceback, json
import yggdrasil.pickle.dill as dill

_ctx = dill.loads(base64.b64decode({command_b64!r}.encode("ascii")))
_job = _ctx.decode_payload({job_b64!r})
_env = {environ!r}
for k, v in _env.items():
    os.environ[k] = v
_out = _ctx.encode_object(_job())
if not isinstance(_out, str):
    _out = _out.decode()

print({tag!r} + _out, flush=True)
"""
        return cmd

    def encode_object(
        self,
        obj: Any,
        byte_limit: int = 32 * 1024,
        byref: Any = None,
        recurse: Any = None,
        compression: Optional[str] = None
    ) -> str:
        """Serialize *obj* to a JSON envelope that ``decode_payload`` can reconstruct.

        Dispatch table (checked in order):

        +-----------------------+------------------------------------------+
        | Type                  | Strategy                                 |
        +=======================+==========================================+
        | ``pyarrow.Table``     | Parquet → DBFS temp file                 |
        +-----------------------+------------------------------------------+
        | ``polars.DataFrame``  | Parquet → DBFS temp file                 |
        +-----------------------+------------------------------------------+
        | ``pandas.DataFrame``  | Parquet → DBFS temp file                 |
        +-----------------------+------------------------------------------+
        | ``pandas.Series``     | Parquet via single-column DataFrame →    |
        |                       | DBFS temp file; index and name preserved |
        +-----------------------+------------------------------------------+
        | ``Generator`` /       | Each item encoded recursively            |
        | ``Iterator``          |                                          |
        +-----------------------+------------------------------------------+
        | Anything else         | ``dill`` → optional gzip → inline b64    |
        |                       | or DBFS temp file if > *byte_limit*      |
        +-----------------------+------------------------------------------+

        Args:
            obj: Object to encode.
            byte_limit: Maximum inline payload size in bytes before spilling to
                a DBFS temporary file (default 32 KiB).
            byref: Passed through to ``dill.dump`` — serialize by reference when
                ``True`` (useful for large closures that reference module-level
                objects).
            recurse: Passed through to ``dill.dump`` — recursively serialise
                referenced objects.
            compression: Force ``"gzip"`` compression regardless of size.
                Defaults to auto-compress when the raw payload exceeds
                *byte_limit*.

        Returns:
            A JSON string describing how to reconstruct the object on the remote
            side.  See ``decode_payload`` for the envelope schema.
        """
        buffer = io.BytesIO()

        from yggdrasil.arrow.lib import pyarrow
        from yggdrasil.polars.lib import polars
        from yggdrasil.pandas.lib import pandas
        import yggdrasil.pickle.dill as pkl
        import yggdrasil.pickle.json as json_mod

        if isinstance(obj, pyarrow.Table):
            import pyarrow.parquet as pq

            func = "pyarrow.parquet.read_table"
            extension = "parquet"
            pq.write_table(obj, buffer)

            buffer.seek(0)
            dbx_path = self.client.tmp_path(extension=extension)
            dbx_path.write_bytes(buffer)

            return json_mod.dumps({
                "func": func,
                "file": dbx_path.full_path()
            })

        if isinstance(obj, polars.DataFrame):
            func = "polars.read_parquet"
            extension = "parquet"
            obj.write_parquet(buffer)

            buffer.seek(0)
            dbx_path = self.client.tmp_path(extension=extension)
            dbx_path.write_bytes(buffer)

            return json_mod.dumps({
                "func": func,
                "file": dbx_path.full_path()
            })

        # --- pandas.Series ---
        # Encode as a single-column Parquet file.  The series name (or a
        # synthetic "__series__" fallback) becomes the column name, and the
        # index is written as a named reset column so it survives the
        # round-trip through Parquet without loss.
        if isinstance(obj, pandas.Series):
            func = "pandas.read_parquet_series"
            extension = "parquet"

            series_name = obj.name if obj.name is not None else "__series__"
            index_name  = obj.index.name if obj.index.name is not None else "__index__"

            # Reset the index into a regular column so Parquet preserves it.
            df = obj.rename(series_name).reset_index()
            df.columns = [index_name] + [series_name]
            df.to_parquet(path=buffer, index=False)

            buffer.seek(0)
            dbx_path = self.client.tmp_path(extension=extension)
            dbx_path.write_bytes(buffer)

            return json_mod.dumps({
                "func": func,
                "file": dbx_path.full_path(),
                "series_name": series_name,
                "index_name": index_name,
                # Preserve the original name exactly (None is meaningful).
                "original_name": obj.name,
            })

        elif isinstance(obj, pandas.DataFrame):
            func = "pandas.read_parquet"
            extension = "parquet"
            obj.to_parquet(path=buffer)

            buffer.seek(0)
            dbx_path = self.client.tmp_path(extension=extension)
            dbx_path.write_bytes(buffer)

            return json_mod.dumps({
                "func": func,
                "cpr": compression,
                "file": dbx_path.full_path()
            })

        elif isinstance(obj, (Generator, Iterator)):
            return json_mod.dumps({
                "func": "generator",
                "items": [
                    self.encode_object(_, byte_limit=byte_limit, byref=byref, recurse=recurse, compression=compression)
                    for _ in obj
                ]
            })

        pkl.dump(obj, buffer, byref=byref, protocol=5)

        raw = buffer.getvalue()

        if compression or len(raw) > byte_limit:
            compression = compression or "gzip"
            if compression != "gzip":
                raise ValueError(f"Unsupported compression: {compression}")
            raw = gzip.compress(raw)

        if len(raw) > byte_limit:
            dbx_path = self.client.tmp_path(extension="bin")
            dbx_path.write_bytes(raw)
            return json_mod.dumps({
                "func": "dill.load",
                "cpr": compression,
                "file": dbx_path.full_path(),
            })

        return json_mod.dumps({
            "func": "dill.load",
            "cpr": compression,
            "b64": base64.b64encode(raw).decode("ascii"),
        })

    def decode_payload(
        self,
        payload: Union[str, bytes, dict, list],
        temporary: bool = True
    ):
        """Reconstruct an object from a JSON envelope produced by ``encode_object``.

        Recognised ``func`` values:

        +---------------------------------+------------------------------------------+
        | ``func``                        | Action                                   |
        +=================================+==========================================+
        | ``"dill.load"``                 | dill-deserialise from inline b64 or file |
        +---------------------------------+------------------------------------------+
        | ``"pyarrow.parquet.read_table"``| Read Parquet file → ``pyarrow.Table``    |
        +---------------------------------+------------------------------------------+
        | ``"pandas.read_parquet"``       | Read Parquet file → ``pandas.DataFrame`` |
        +---------------------------------+------------------------------------------+
        | ``"pandas.read_parquet_series"``| Read Parquet file → ``pandas.Series``   |
        |                                 | (restores name and index)                |
        +---------------------------------+------------------------------------------+
        | ``"pandas.read_pickle"``        | Read pickle file → ``pandas.DataFrame`` |
        +---------------------------------+------------------------------------------+
        | ``"polars.read_parquet"``       | Read Parquet file → ``polars.DataFrame`` |
        +---------------------------------+------------------------------------------+
        | ``"generator"``                 | Lazy generator over recursively decoded  |
        |                                 | items                                    |
        +---------------------------------+------------------------------------------+

        Args:
            payload: JSON string, bytes, pre-parsed dict, or list.  Non-JSON
                strings are returned as-is.
            temporary: When ``True``, DBFS paths are treated as temporary files
                and may be cleaned up after reading.

        Returns:
            The reconstructed Python object, or *payload* unchanged if it does
            not match any known envelope schema.
        """
        import yggdrasil.pickle.dill as pkl
        import yggdrasil.pickle.json as json_mod

        if isinstance(payload, (str, bytes)):
            try:
                payload = json_mod.loads(payload)
            except Exception:
                return payload

        if isinstance(payload, dict):
            func, compression, b64, databricks_path = (
                payload.get("func"), payload.get("cpr"),
                payload.get("b64"), payload.get("file")
            )

            if isinstance(func, str) and func:
                if b64:
                    blob = base64.b64decode(b64.encode("ascii"))
                elif databricks_path:
                    blob = self.client.dbfs_path(databricks_path, temporary=temporary).read_bytes()
                else:
                    blob = None

                if func == "dill.load":
                    if compression == "gzip":
                        import gzip
                        blob = gzip.decompress(blob)

                    return pkl.loads(blob)

                elif func.startswith("pyarrow."):
                    import pyarrow.parquet as pq

                    buff = io.BytesIO(blob)
                    return pq.read_table(buff)

                elif func == "pandas.read_parquet_series":
                    # Reconstruct a pandas.Series from the single-column
                    # Parquet file written by encode_object.
                    import pandas

                    series_name   = payload.get("series_name", "__series__")
                    index_name    = payload.get("index_name",  "__index__")
                    original_name = payload.get("original_name")  # may be None

                    buff = io.BytesIO(blob)
                    df   = pandas.read_parquet(buff)

                    # Restore the index from its saved column, then extract
                    # the value column as a Series with the original name.
                    if index_name in df.columns:
                        df = df.set_index(index_name)
                        df.index.name = None if index_name == "__index__" else index_name

                    series = df[series_name].rename(original_name)
                    return series

                elif func.startswith("pandas."):
                    import pandas

                    buff = io.BytesIO(blob)

                    if func == "pandas.read_parquet":
                        return pandas.read_parquet(buff)
                    elif func == "pandas.read_pickle":
                        return pandas.read_pickle(buff, compression=compression)
                    else:
                        raise NotImplementedError

                elif func == "generator":
                    items = payload.get("items")

                    def gen(it: Iterator = items):
                        if it:
                            for item in it:
                                yield self.decode_payload(item)

                    return gen()

                elif func.startswith("polars."):
                    import polars

                    buff = io.BytesIO(blob)
                    return polars.read_parquet(buff)

                else:
                    raise NotImplementedError

        return payload

    def is_in_databricks_environment(self):
        """Return True when running on a Databricks runtime."""
        return self.cluster.client.is_in_databricks_environment()

    def check_with_env(
        self,
        env: PyEnv,
        wait: WaitingConfigArg = True,
        raise_error: bool = True
    ):
        local_reqs = env.requirements(with_system=False)
        remote_reqs = self.requirements
        diffs = diff_installed_libraries(local_reqs, remote_reqs)
        diffs = [
            name # "%s==%s" % (name, meta["current"])
            for name, meta in diffs.items()
            if meta and meta["current"] and _valid_install_package(name)
        ]

        if diffs:
            self.install_temporary_libraries(
                libraries=diffs,
                pip_install=False,
                wait=wait,
                raise_error=raise_error
            )
            self._requirements = None

        return self

def _valid_install_package(name: str):
    for prefix in (
        "pyspark", "pywin32", "ygg", "pip", "setuptools", "wheel",
        ""
    ):
        if name.startswith(prefix):
            return False
    return True

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
            raise ClientTerminatedSession(message)

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


def diff_installed_libraries(
    current: list[tuple[str, str]],
    target: list[tuple[str, str]],
) -> dict[str, dict[str, str | None]]:
    """
    Compare two package lists by name + major.minor version.
    Returns packages that differ, with exact full versions.
    """
    def to_major_minor(version: str) -> str:
        return ".".join(version.split(".")[:2])

    current_map = {name: ver for name, ver in current}
    target_map  = {name: ver for name, ver in target}

    all_names = current_map.keys() | target_map.keys()

    return {
        name: {
            "current": current_map.get(name),
            "target":  target_map.get(name),
        }
        for name in all_names
        if to_major_minor(current_map.get(name) or "0.0") != to_major_minor(target_map.get(name) or "0.0")
    }
