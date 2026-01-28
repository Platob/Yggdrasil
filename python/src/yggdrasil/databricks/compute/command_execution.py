import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any, AnyStr, Callable

import dill
from databricks.sdk.errors import InternalError

from .exceptions import CommandExecutionAborted
from ...libs.databrickslib import databricks_sdk, DatabricksDummyClass
from ...libs.pandaslib import PandasDataFrame
from ...pyutils.exceptions import raise_parsed_traceback
from ...pyutils.waiting_config import WaitingConfig, WaitingConfigArg

if databricks_sdk is not None:
    from databricks.sdk.service.compute import (
        Language, CommandExecutionAPI, CommandStatusResponse, CommandStatus, ResultType
    )

    DONE_STATES = {
        CommandStatus.FINISHED, CommandStatus.CANCELLED, CommandStatus.ERROR
    }

    PENDING_STATES = {
        CommandStatus.RUNNING, CommandStatus.QUEUED, CommandStatus.RUNNING
    }

    FAILED_STATES = {
        CommandStatus.ERROR, CommandStatus.CANCELLED
    }
else:
    Language = DatabricksDummyClass
    CommandExecutionAPI = DatabricksDummyClass
    ResultType = DatabricksDummyClass

    DONE_STATES, PENDING_STATES, FAILED_STATES = set(), set(), set()


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

    _details: Optional[CommandStatusResponse] = field(default=None, repr=False, compare=False, hash=False)

    def __call__(self, *args, **kwargs):
        assert self.command, "Cannot call %s, missing command" % self

        args = dill.dumps([self.encode_object(_) for _ in args])
        kwargs = dill.dumps({
            k: self.encode_object(v)
            for k, v in kwargs.items()
        })

        command = (
            self.command
            .replace("__ARGS__", args.decode("latin-1"))
            .replace("__KWARGS__", kwargs.decode("latin-1"))
        )

        run = self.create(
            context=self.context,
            command=command,
            language=Language.PYTHON
        ).start()

        return run.wait(raise_error=True)

    def __bool__(self):
        return self.done

    def create(
        self,
        context: Optional["ExecutionContext"] = None,
        func: Optional[Callable] = None,
        command: Optional[str] = None,
        language: Optional[Language] = None,
        command_id: Optional[str] = None
    ):
        context = self.context if context is None else context
        command = self.command if command is None else command

        if not command:
            if callable(func):
                command = self.make_python_function_command(
                    func=func,
                )

        if language is None:
            language = context.language or Language.PYTHON

        assert context is not None, "Missing context to execute command"

        return CommandExecution(
            context=context,
            language=language,
            command=command,
            command_id=command_id
        )

    def start(self):
        if self.command_id:
            return self

        client = self.context.workspace_client().command_execution

        LOGGER.debug(
            "%s executing command:\m%s",
            self.context,
            self.command
        )

        try:
            details: CommandStatusResponse = client.execute(
                cluster_id=self.context.cluster_id,
                context_id=self.context.context_id,
                language=self.language,
                command=self.command,
            ).response
        except InternalError as e:
            if "ontext" in str(e):  # context related
                self.context = self.context.connect(reset=True)

                LOGGER.debug(
                    "%s executing command:\m%s",
                    self.context,
                    self.command
                )

                details: CommandStatusResponse = client.execute(
                    cluster_id=self.cluster_id,
                    context_id=self.context_id,
                    language=self.language,
                    command=self.command,
                ).response
            else:
                raise e

        LOGGER.info(
            "Started %s",
            self
        )

        self.details = details

        return self

    @property
    def workspace(self):
        return self.context.workspace

    @property
    def cluster_id(self):
        return self.context.cluster.cluster_id

    @property
    def context_id(self):
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
            self.command_id = value.id

    @property
    def results_metadata(self):
        return self.details.results

    def client(self) -> CommandExecutionAPI:
        return self.context.workspace_client().command_execution

    def connect(self, reset: bool = False):
        self.context = self.context.connect(language=self.language)

        if self.command_id:
            if not reset:
                return self

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
            return self

        wait = WaitingConfig.check_arg(wait)
        iteration, start = 0, time.time()

        while self.running:
            wait.sleep(
                iteration=iteration,
                start=start
            )
            iteration += 1

        if raise_error:
            try:
                self.raise_for_status()
            except CommandExecutionAborted as e:
                LOGGER.error(
                    "%s aborted: %s",
                    self,
                    e
                )

                if self.command:
                    # Retry with new context
                    new_execution = self.create(
                        context=self.context.connect(reset=True),
                        command=self.command,
                        language=self.language
                    )

                    self.context = new_execution.context
                    self.command_id = new_execution.command_id

                    self._details = new_execution._details
                else:
                    raise

        return self

    def shared_cache_path(
        self,
        suffix: str
    ):
        return self.context.shared_cache_path(
            suffix="/command/%s/%s" % (self.command_id, suffix.lstrip("/"))
        )

    def encode_object(
        self,
        obj: Any,
        byte_limit: int = 32 * 1024,
        byref: Any = None,
        recurse: Any = None,
    ) -> str:
        prefix = Prefixes.DILL
        compression: str = ""

        buffer = io.BytesIO()
        if isinstance(obj, PandasDataFrame):
            obj.to_pickle(path=buffer)
        else:
            dill.dump(
                obj,
                buffer,
                byref=byref,
                recurse=recurse
            )

        raw = buffer.getvalue()

        if len(raw) > byte_limit:
            import zlib
            compression = "zlib"
            raw = zlib.compress(raw)

        if len(raw) > byte_limit:
            dbx_path = self.shared_cache_path(suffix=f"{hash(obj)}.pkl")
            dbx_path.write_object(obj)
            prefix = Prefixes.DATABRICKS_PATH
            payload = dbx_path.full_path()
        else:
            payload = raw.decode("latin-1")

        return f"{prefix}:{compression}:{payload}"

    def decode_object(
        self,
        payload: str,
    ):
        try:
            prefix, compression, payload = payload.split(":", 2)
        except ValueError as e:
            raise ValueError(f"Malformed encoded object: {payload[:80]}...") from e

        def _maybe_decompress(blob: bytes) -> bytes:
            if not compression:
                return blob
            if compression == "zlib":
                import zlib
                return zlib.decompress(blob)
            raise ValueError(f"Unknown compression '{compression}'")

        if prefix == Prefixes.DILL:
            blob = payload.encode("latin-1")
            blob = _maybe_decompress(blob)
            return dill.loads(blob)

        if prefix == Prefixes.DATABRICKS_PATH:
            # payload is a full path string
            dbx_path = self.workspace.dbfs_path(payload)

            try:
                raw = dbx_path.read_bytes()
            finally:
                dbx_path.rmfile(allow_not_found=True)

            raw = _maybe_decompress(raw)
            return dill.loads(raw)

        raise ValueError(f"Unknown prefix '{prefix}'")

    def make_python_function_command(
        self,
        func: Callable,
        tag: str = "__CALL_RESULT__",
        byref: Any = None,
        recurse: Any = None,
    ):
        command_bytes = dill.dumps(self)
        serialized_func = self.encode_object(func)

        cmd = (
            """import dill
from yggdrasil.databricks.compute.command_execution import encode_object, decode_object

args, kwargs, func, tag, command = "__ARGS__", "__KWARGS__", "__FUNCTION__", "__TAG__", "__COMMAND__"

command = dill.loads(command.decode("latin-1"))

args = [command.decode_object(_) for _ in command.decode_object(args)]
kwargs = {k: command.decode_object(v) for k, v in command.decode_object(kwargs).items()}
func = command.decode_object(func)

result = func(*args, **kwargs)
encoded = command.encode_object(result)

print(tag + encoded)
"""
            .replace("__TAG__", tag)
            .replace("__FUNCTION__", serialized_func)
            .replace("__COMMAND__", command_bytes.decode("latin-1"))
        )

        return cmd

    def decode_response(
        self,
        response: CommandStatusResponse,
        language: Language,
        raise_error: bool = True,
        tag: str = "__CALL_RESULT__",
        logger: bool = True
    ) -> Any:
        """Mirror the old Cluster.execute_command result handling.

        Args:
            response: Raw command execution response.
            language: Language executed
            raise_error: Raise error if response is failed
            tag: Result tag
            logger: Print logs

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

            if logger:
                for line in logs_text.splitlines():
                    stripped_log = line.strip()

                    if stripped_log:
                        print(stripped_log)

            return self.decode_object(payload=raw_result)

        return raw_result

    def result(
        self,
        raise_error: bool = True
    ) -> Any:
        self.wait(raise_error=raise_error)

        obj = self.decode_response(
            response=self.details,
            language=self.language,
            raise_error=raise_error
        )

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
                raise CommandExecutionAborted(message)

            if language == Language.PYTHON:
                raise_parsed_traceback(message)
        else:
            message = str(response)

        raise RuntimeError(message)


class Prefixes:
    DILL = "EP000"
    DATABRICKS_PATH = "EP001"


