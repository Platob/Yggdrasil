import base64
import gzip
import io
import json
import logging
import os
import re
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

from .exceptions import ClientTerminatedSession
from ...environ import PyEnv
from ...io.url import URL
from ...pyutils.exceptions import raise_parsed_traceback
from ...pyutils.waiting_config import WaitingConfig, WaitingConfigArg

DONE_STATES = {
    CommandStatus.FINISHED, CommandStatus.CANCELLED, CommandStatus.ERROR
}

PENDING_STATES = {
    CommandStatus.RUNNING, CommandStatus.QUEUED, CommandStatus.RUNNING
}

FAILED_STATES = {
    CommandStatus.ERROR, CommandStatus.CANCELLED
}

# Module-level constants — computed once at import time
_CMD_RUNTIME_FIELDS = frozenset({"_details"})

# Fields dropped when serializing for remote cluster execution
# (command is reconstructed remotely; context_id / command_id are local)
_CMD_REMOTE_DROP = frozenset({"command", "command_id"})


if TYPE_CHECKING:
    from .execution_context import ExecutionContext


__all__ = [
    "CommandExecution"
]


LOGGER = logging.getLogger(__name__)

def _pkg_name(spec: str) -> str:
    """Strip PEP 508 version specifiers to get the bare package name."""
    return re.split(r"[=<>!~;@\[]", spec)[0].strip().lower()

DEFAULT_PACKAGES = ["ygg", "dill", "pandas", "mongoengine", "sqlalchemy"]

def _dedup_packages(defaults: list[str], extra: list[str] | None) -> list[str]:
    """Merge default + user packages, user versions win on conflict."""
    seen: dict[str, str] = {}
    for pkg in defaults + (extra or []):
        seen[_pkg_name(pkg)] = pkg  # later entry (user-supplied) overwrites
    return list(seen.values())


@dataclass
class CommandExecution:
    context: "ExecutionContext"
    command_id: Optional[str] = None

    language: Optional[Language] = field(default=None, repr=False, compare=False, hash=False)
    command: Optional[str] = field(default=None, repr=False, compare=False, hash=False)
    environ: Optional[Dict[str, str]] = field(default=None, repr=False, compare=False, hash=False)

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
        state = {}

        for key, value in self.__dict__.items():
            if key in _CMD_RUNTIME_FIELDS:
                continue
            if key in _CMD_REMOTE_DROP and value is not None:
                # Preserve None explicitly so __setstate__ can restore defaults
                # without an attribute error, but drop non-None payloads.
                state[key] = None
                continue
            state[key] = value

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
        state.setdefault("_details", None)
        state.setdefault("command", None)
        state.setdefault("command_id", None)
        state.setdefault("environ", None)

        self.__dict__.update(state)

    def __call__(self, *args, **kwargs):
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

        self.context.install_temporary_libraries(libraries=["yggdrasil"])

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

    def __bool__(self):
        return self.done

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
            **{"command": self.command_id or "unknown"}
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

    @property
    def details(self) -> CommandStatusResponse:
        if self._details is None:
            self._details = self.client().command_status(
                cluster_id=self.cluster_id,
                context_id=self.context_id,
                command_id=self.command_id
            )
        elif self._details.status not in DONE_STATES:
            self._details = self.client().command_status(
                cluster_id=self.cluster_id,
                context_id=self.context_id,
                command_id=self.command_id
            )
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

    @property
    def results_metadata(self):
        return self.details.results

    def client(self) -> CommandExecutionAPI:
        return self.context.workspace_client().command_execution

    def connect(self, reset: bool = False):
        self.context = self.context.connect(language=self.language)

        return self

    def cancel(self, raise_error: bool = False):
        if self.command_id:
            try:
                self.client().cancel_and_wait(
                    cluster_id=self.cluster_id,
                    command_id=self.command_id,
                    context_id=self.context_id
                )
            except Exception as e:
                if raise_error:
                    raise e
                LOGGER.exception(e)

    def raise_for_status(self):
        if self.state in FAILED_STATES:
            raise_error_from_response(
                response=self.details,
                language=self.language
            )

        return self

    def wait(
        self,
        wait: Optional[WaitingConfigArg] = True,
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
                wait.sleep(iteration=iteration, start=start)
                iteration += 1

        if raise_error:
            try:
                self.raise_for_status()
            except ModuleNotFoundError as e:
                module_name = e.name

                if module_name and not module_name.startswith("ygg"):
                    self.context.cluster.install_temporary_libraries(
                        libraries=[module_name]
                    )

                    return (
                        self
                        .start(reset=True)
                        .wait(wait=wait, raise_error=raise_error)
                    )
                else:
                    raise e
            except ClientTerminatedSession as e:
                LOGGER.error(
                    "%s aborted: %s",
                    self,
                    e
                )

                self.context = self.context.connect(reset=True)

                return (
                    self
                    .start(reset=True)
                    .wait(wait=wait, raise_error=raise_error)
                )

        return self

    def encode_object(
        self,
        obj: Any,
        byte_limit: int = 32 * 1024,
        byref: Any = None,
        recurse: Any = None,
        compression: Optional[str] = None
    ) -> str:
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

        # Inject ctx_libs into sys.path when the context has uploaded temporary libs
        syspath_lines: list[str] = []

        rm = self.context.remote_metadata

        if rm and rm.libs_path:
            syspath_lines = [
                "import os as _os, sys as _sys",
                f"_p = _os.path.normpath(_os.path.expandvars(_os.path.expanduser({rm.libs_path!r})))",
                "if _os.path.isdir(_p) and _p not in _sys.path:",
                "    _sys.path.insert(0, _p)",
                "del _p, _os, _sys",
            ]

        inner_code = "\n".join([
            *syspath_lines,  # ← injected before any imports
            "import base64, dill",
            f"command = dill.loads(base64.b64decode({command_b64!r}.encode('ascii')))",
            "args    = dill.loads(base64.b64decode(args_b64.encode('ascii')))",
            "kwargs  = dill.loads(base64.b64decode(kwargs_b64.encode('ascii')))",
            f"f  = command.decode_payload({serialized_func!r}, temporary=False)",
            "a  = [command.decode_payload(x) for x in args]",
            "kw = {k: command.decode_payload(v) for k, v in kwargs.items()}",
            "r  = f(*a, **kw)",
            f"print({tag!r} + command.encode_object(r))",
        ])

        resolved_packages = _dedup_packages(DEFAULT_PACKAGES, packages)

        inner_b64: str = base64.b64encode(inner_code.encode("utf-8")).decode("ascii")

        cmd = f"""\
import base64, dill, os
from yggdrasil.environ import PyEnv

_inner = base64.b64decode({inner_b64!r}.encode('ascii')).decode('utf-8')
_globs = {{
    "args_b64":   __ARGS__,
    "kwargs_b64": __KWARGS__,
}}

pyenv  = PyEnv.current().resolve_env(identifier=__CTX_KEY__, version=__PYVERSION__, packages={resolved_packages!r})
result = pyenv.run_python_code(code=_inner, env=__ENVIRON__, globs=_globs)

print(result.stdout or "")
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
        raise_error: bool = True,
        unpickle: bool = True
    ) -> Any:
        try:
            self.wait(raise_error=raise_error)

            obj = self.decode_response(
                response=self.details,
                language=self.language,
                raise_error=raise_error,
                unpickle=unpickle
            )
        except (InternalError, ClientTerminatedSession):
            self.context = self.context.connect(reset=True)

            return (
                self
                .start(reset=True)
                .result(raise_error=raise_error, unpickle=unpickle)
            )
        except ModuleNotFoundError as e:
            module_name = e.name

            if module_name:
                self.context.install_temporary_libraries(libraries=[module_name])

                return (
                    self
                    .start(reset=True)
                    .result(raise_error=raise_error, unpickle=unpickle)
                )
            else:
                raise e

        return obj


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
