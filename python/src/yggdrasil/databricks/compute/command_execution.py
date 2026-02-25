import base64
import gzip
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import TYPE_CHECKING, Optional, Any, Callable, Dict, Iterable, Union, Generator, Iterator

import dill
import pyarrow
from databricks.sdk.errors import InternalError
from databricks.sdk.service.compute import (
    Language, CommandExecutionAPI, CommandStatusResponse, CommandStatus, ResultType
)

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from .exceptions import ClientTerminatedSession
from ...environ import PyEnv
from ...io.url import URL
from ...pyutils.exceptions import raise_parsed_traceback

DONE_STATES = {
    CommandStatus.FINISHED, CommandStatus.CANCELLED, CommandStatus.ERROR
}

PENDING_STATES = {
    CommandStatus.RUNNING, CommandStatus.QUEUED, CommandStatus.RUNNING
}

FAILED_STATES = {
    CommandStatus.ERROR, CommandStatus.CANCELLED
}


if TYPE_CHECKING:
    from .execution_context import ExecutionContext


__all__ = [
    "CommandExecution"
]


LOGGER = logging.getLogger(__name__)


@dataclass
class CommandExecution:
    context: "ExecutionContext"
    command_id: Optional[str] = None

    language: Optional[Language] = field(default=None, repr=False, compare=False, hash=False)
    command: Optional[str] = field(default=None, repr=False, compare=False, hash=False)
    environ: Optional[Dict[str, str]] = field(default=None, repr=False, compare=False, hash=False)

    _pyfunc: Optional[Callable] = field(default=None, repr=False, compare=False, hash=False)
    _details: Optional[CommandStatusResponse] = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self):
        if self.environ:
            if isinstance(self.environ, (list, tuple, set)):
                self.environ = {
                    k: os.getenv(k)
                    for k in self.environ
                }

    def __getstate__(self) -> dict:
        """Serialize for pickling.

        Two distinct pickle consumers exist:

        1. **Remote cluster execution** (via ``make_python_function_command``):
           ``dill.dumps(self)`` embeds this object in generated Python code
           that runs on the Databricks cluster.  Only ``decode_payload`` /
           ``encode_object`` are called there, so we need ``context.workspace``
           but not the live command state or SDK handles.

        2. **Local process serialization** (Spark UDFs, multiprocessing):
           Same rules as (1) — live handles don't survive cross-process.

        In both cases we drop:
        - ``_details``: live SDK response object, meaningless after transport
        - ``command``: can be 10s of KB; the remote side never calls ``.start()``
        - ``command_id``: local execution handle, invalid on the remote

        Returns:
            A lean, pickle-ready state dictionary.
        """
        state = self.__dict__.copy()

        if self.environ:
            environ = {
                k: os.getenv(v)
                for k, v in self.environ.items()
                if k and v
            }

            state["environ"] = environ

        return state

    def __setstate__(self, state: dict) -> None:
        """Restore command execution state after unpickling.

        Ensures all expected attributes are present even when the state was
        produced by a stripped ``__getstate__``.  The execution context is
        left as-is; callers must invoke ``connect(reset=True)`` if they need
        a live context after unpickling.

        Args:
            state: Serialized state dictionary.
        """
        # Guarantee attribute completeness for fields that may have been
        # pruned or were absent in older serialized states.
        for key in ("_details", "command", "command_id", "environ"):
            state[key] = state.get(key, None)

        self.__dict__.update(state)

    def __call__(self, *args, **kwargs):
        if self.context.is_in_databricks_environment() and self._pyfunc is not None:
            if self.environ:
                for k, v in self.environ:
                    os.environ[str(k)] = str(v)

            return self._pyfunc(*args, **kwargs)

        assert self.command, "Cannot call %s, missing command" % self

        args_blob = dill.dumps([self.encode_object(_) for _ in args])
        kwargs_blob = dill.dumps({k: self.encode_object(v) for k, v in kwargs.items()})

        if self.environ:
            env_blob = {
                k: os.getenv(k) or v
                for k, v in self.environ.items()
                if os.getenv(k) or v
            }
        else:
            env_blob = {}

        args_b64 = base64.b64encode(args_blob).decode("ascii")
        kwargs_b64 = base64.b64encode(kwargs_blob).decode("ascii")
        major, minor, _ = PyEnv.current().version_info
        pyversion = f"{major}.{minor}"

        command = (
            self.command
            .replace("__ARGS__", repr(args_b64))
            .replace("__KWARGS__", repr(kwargs_b64))
            .replace("__ENVIRON__", repr(env_blob))
            .replace("__CTX_KEY__", repr(self.context.context_key))
            .replace("__PYVERSION__", repr(pyversion))
        )

        run = (
            self.create(
                context=self.context,
                command=command,
                language=self.language
            )
            .start()
        )

        return run.wait(raise_error=True).result(raise_error=True)

    def __repr__(self):
        return "%s(url=%s)" % (
            self.__class__.__name__,
            self.url()
        )

    def __str__(self):
        return self.url().to_string()

    def url(self) -> URL:
        url = self.context.url()

        return url.with_query_items({
            **url.query_dict,
            **{"command_id": self.command_id or "unknown"}
        })

    def create(
        self,
        context: Optional["ExecutionContext"] = None,
        func: Optional[Callable] = None,
        command: Optional[str] = None,
        language: Optional[Language] = None,
        command_id: Optional[str] = None,
        environ: Optional[Union[Iterable[str], Dict[str, str]]] = None,
        packages: list[str] | None = None
    ):
        context = self.context if context is None else context
        command = self.command if command is None else command
        environ = self.environ if environ is None else environ

        if environ is not None:
            if not isinstance(environ, dict):
                environ = {
                    str(k): os.getenv(str(k))
                    for k in environ
                }
            else:
                environ = {
                    str(k): str(v)
                    for k, v in environ.items()
                }

        if not command:
            if callable(func):
                command = self.make_python_function_command(
                    func=func,
                    packages=packages
                )

        if language is None:
            language = context.language or Language.PYTHON

        assert context is not None, "Missing context to execute command"

        return CommandExecution(
            context=context,
            language=language,
            command=command,
            command_id=command_id,
            environ=environ
        )

    def start(self, reset: bool = False):
        if self.command_id:
            if not reset:
                return self

            if not self.done:
                self.cancel(wait=False)

            self._details = None
            self.command_id = None

        client = self.context.workspace_client().command_execution

        assert self.command, "Missing command arg in %s" % self

        try:
            details = client.execute(
                cluster_id=self.cluster_id,
                context_id=self.context_id,
                language=self.language,
                command=self.command,
            ).response
        except Exception as e:
            if "ontext" in str(e):  # context related
                self.context = self.context.connect(reset=True)

                details = client.execute(
                    cluster_id=self.cluster_id,
                    context_id=self.context_id,
                    language=self.language,
                    command=self.command,
                ).response
            else:
                raise e

        self.command_id = details.id
        self._details = None

        LOGGER.debug("Started %s", self)

        return self

    def cancel(
        self,
        wait: WaitingConfigArg | None = True
    ):
        if self.command_id:
            wait = WaitingConfig.check_arg(wait)
            client = self.context.workspace_client().command_execution

            response = client.cancel(
                cluster_id=self.cluster_id,
                context_id=self.context_id,
                command_id=self.command_id
            )

            if wait:
                response.result(timeout=wait.timeout_timedelta)

        return self

    @property
    def workspace(self):
        return self.context.workspace

    @property
    def cluster_id(self):
        return self.context.cluster.cluster_id

    @property
    def context_id(self):
        if not self.context.context_id:
            self.context = self.context.connect()
        return self.context.context_id

    @property
    def state(self):
        return self.details.status

    @property
    def running(self):
        return self.state in PENDING_STATES

    @property
    def done(self):
        return self.state in DONE_STATES

    def _command_status(self):
        try:
            return self.client().command_status(
                cluster_id=self.cluster_id,
                context_id=self.context_id,
                command_id=self.command_id
            )
        except InternalError:
            self.context.cluster.ensure_running()
            self.start(reset=True)

            return self.client().command_status(
                cluster_id=self.cluster_id,
                context_id=self.context_id,
                command_id=self.command_id
            )

    @property
    def details(self) -> CommandStatusResponse:
        if self._details is None:
            self._details = self._command_status()

        elif self._details.status not in DONE_STATES:
            self._details = self._command_status()

        return self._details

    @details.setter
    def details(self, value: Optional[CommandStatusResponse]):
        self._details = value

        if value is not None:
            assert isinstance(value, CommandStatusResponse), "%s.details must be CommandStatusResponse, got %s" %(
                self,
                type(value)
            )
            self.command_id = value.id

    def client(self) -> CommandExecutionAPI:
        return self.context.workspace_client().command_execution

    def raise_for_status(self):
        if self.state in FAILED_STATES:
            raise_error_from_response(
                response=self.details,
                language=self.language
            )

        return self

    def wait(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True
    ):
        if not self.command_id:
            return self.start().wait(
                wait=wait,
                raise_error=raise_error
            )

        wait = WaitingConfig.check_arg(wait)
        iteration, start = 0, time.time()

        if wait.timeout:
            while self.running:
                try:
                    wait.sleep(iteration=iteration, start=start)
                    iteration += 1
                except KeyboardInterrupt:
                    self.cancel(wait=False)
                    raise

        if raise_error:
            self.raise_for_status()

        return self

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

        from ...polars.lib import polars
        from ...pandas.lib import pandas

        if isinstance(obj, pyarrow.Table):
            import pyarrow.parquet as pq

            func = "pyarrow.parquet.read_table"
            extension = "parquet"
            pq.write_table(obj, buffer)

            buffer.seek(0)
            dbx_path = self.workspace.tmp_path(extension=extension)
            dbx_path.write_bytes(buffer)

            return json.dumps({
                "func": func,
                "file": dbx_path.full_path()
            })

        if isinstance(obj, polars.DataFrame):
            func = "polars.read_parquet"
            extension = "parquet"
            obj.write_parquet(buffer)

            buffer.seek(0)
            dbx_path = self.workspace.tmp_path(extension=extension)
            dbx_path.write_bytes(buffer)

            return json.dumps({
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
            dbx_path = self.workspace.tmp_path(extension=extension)
            dbx_path.write_bytes(buffer)

            return json.dumps({
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
            dbx_path = self.workspace.tmp_path(extension=extension)
            dbx_path.write_bytes(buffer)

            return json.dumps({
                "func": func,
                "cpr": compression,
                "file": dbx_path.full_path()
            })

        elif isinstance(obj, (Generator, Iterator)):
            return json.dumps({
                "func": "generator",
                "items": [
                    self.encode_object(_, byte_limit=byte_limit, byref=byref, recurse=recurse, compression=compression)
                    for _ in obj
                ]
            })

        dill.dump(obj, buffer, byref=byref, recurse=recurse)

        raw = buffer.getvalue()

        if compression or len(raw) > byte_limit:
            compression = compression or "gzip"
            if compression != "gzip":
                raise ValueError(f"Unsupported compression: {compression}")
            raw = gzip.compress(raw)

        if len(raw) > byte_limit:
            dbx_path = self.workspace.tmp_path(extension="bin")
            dbx_path.write_bytes(raw)
            return json.dumps({
                "func": "dill.load",
                "cpr": compression,
                "file": dbx_path.full_path(),
            })

        return json.dumps({
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
        if isinstance(payload, (str, bytes)):
            try:
                payload = json.loads(payload)
            except JSONDecodeError:
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
                    blob = self.workspace.dbfs_path(databricks_path, temporary=temporary).read_bytes()
                else:
                    blob = None

                if func == "dill.load":
                    if compression == "gzip":
                        import gzip
                        blob = gzip.decompress(blob)

                    return dill.loads(blob)

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

    def make_python_function_command(
        self,
        func: Callable,
        tag: str = "__CALL_RESULT__",
        byref: Any = None,
        recurse: Any = None,
        packages: list[str] | None = None,
    ) -> str:
        proxy = CommandExecution(
            context=self.context,
            language=self.language,
            environ=self.environ,
        )

        command_b64: str = base64.b64encode(dill.dumps(proxy)).decode("ascii")
        serialized_func: str = self.encode_object(
            func, byte_limit=64 * 1024, byref=byref, recurse=recurse
        )

        syspath_lines = self.context.syspath_lines()

        inner_code = "\n".join([
            "import base64, dill",
            f"command = dill.loads(base64.b64decode({command_b64!r}.encode('ascii')))",
            "args    = dill.loads(base64.b64decode(__ARGS__.encode('ascii')))",
            "kwargs  = dill.loads(base64.b64decode(__KWARGS__.encode('ascii')))",
            f"f  = command.decode_payload({serialized_func!r}, temporary=False)",
            "a  = [command.decode_payload(x) for x in args]",
            "kw = {k: command.decode_payload(v) for k, v in kwargs.items()}",
            "r  = f(*a, **kw)",
            f"print({tag!r} + command.encode_object(r))",
        ])

        cmd = f"""\
{syspath_lines}

import base64, dill, os

env = __ENVIRON__
if env:
    for k, v in env.items():
        os.environ[k] = v

{inner_code}
"""

        return cmd

    def decode_response(
        self,
        response: CommandStatusResponse,
        language: Language,
        raise_error: bool = True,
        tag: str = "__CALL_RESULT__",
        logger: bool = True,
        unpickle: bool = True
    ) -> Any:
        """Mirror the old Cluster.execute_command result handling.

        Args:
            response: Raw command execution response.
            language: Language executed
            raise_error: Raise error if response is failed
            tag: Result tag
            logger: Print logs
            unpickle: Unpickle

        Returns:
            The decoded output string.
        """
        raise_error_from_response(
            response=response,
            language=language,
            raise_error=raise_error
        )

        results = response.results

        # normal output
        if results.result_type == ResultType.TEXT:
            data = results.data or ""
        else:
            raise NotImplementedError(
                "Cannot decode result form %s" % response
            )

        raw_result = data

        if tag in raw_result:
            logs_text, raw_result = raw_result.split(tag, 1)

            try:
                if logger:
                    for line in logs_text.splitlines():
                        stripped_log = line.strip()

                        if stripped_log:
                            print(stripped_log)
            except Exception as e:
                LOGGER.warning(
                    "Cannot print logs from %s: %s",
                    logs_text,
                    e
                )

        if unpickle:
            return self.decode_payload(payload=raw_result)
        return raw_result

    def result(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        unpickle: bool = True
    ) -> Any:
        wait = WaitingConfig.check_arg(wait)
        installed_modules: set[str] = set()
        last_exc: Exception | None = None

        for attempt in range(wait.total_try_count):
            try:
                self.wait(wait=wait, raise_error=raise_error)

                return self.decode_response(
                    response=self.details,
                    language=self.language,
                    raise_error=raise_error,
                    unpickle=unpickle
                )

            except (InternalError, ClientTerminatedSession) as e:
                last_exc = e
                self.context = self.context.connect(reset=True)
                self.start(reset=True)

            except ModuleNotFoundError as e:
                last_exc = e
                module_name = e.name

                if not module_name or module_name in installed_modules:
                    raise

                self.context.install_temporary_libraries(
                    libraries=[module_name],
                    pip_install=False
                )
                self.context.check_with_env(
                    env=PyEnv.current(),
                    wait=WaitingConfig(timeout=60),
                    raise_error=False
                )
                installed_modules.add(module_name)
                self.start(reset=True)

        if last_exc is None:
            last_exc = RuntimeError(f"Failed to get result with {wait}")

        if raise_error:
            raise last_exc
        return None


def raise_error_from_response(
    response: CommandStatusResponse,
    language: Language,
    raise_error: bool = True
):
    if raise_error:
        results = response.results

        if results.result_type == ResultType.ERROR:
            message = results.cause or "Command execution failed"

            if "client terminated the session" in message:
                raise ClientTerminatedSession(message)

            if language == Language.PYTHON:
                raise_parsed_traceback(message)

            raise RuntimeError(str(response))