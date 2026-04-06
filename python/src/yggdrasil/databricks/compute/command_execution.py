from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Optional

from databricks.sdk.client_types import ClientType
from databricks.sdk.errors import InternalError, PermissionDenied, ResourceDoesNotExist
from databricks.sdk.service.compute import CommandStatus, CommandStatusResponse, Language, ResultType

from yggdrasil.dataclasses import (
    WaitingConfig,
    WaitingConfigArg,
    restore_dataclass_state,
    serialize_dataclass_state,
)
from yggdrasil.environ import PyEnv, shutdown as yg_shutdown
from yggdrasil.io.url import URL
from yggdrasil.pickle.ser import Serialized, loads, serialize
from yggdrasil.pyutils.exceptions import raise_parsed_traceback

from .exceptions import ClientTerminatedSession, CommandExecutionError
from .execution_context import ExecutionContext

DONE_STATES = {CommandStatus.FINISHED, CommandStatus.CANCELLED, CommandStatus.ERROR}
PENDING_STATES = {CommandStatus.RUNNING, CommandStatus.QUEUED}
FAILED_STATES = {CommandStatus.ERROR, CommandStatus.CANCELLED}

ALREADY_LIBS = {
    "Cython", "Deprecated", "GitPython", "PyGObject", "Send2Trash", "annotated-doc", "arro3-core",
    "async-lru", "autocommand", "azure-common", "azure-core", "azure-identity", "azure-mgmt-core",
    "azure-mgmt-web", "azure-storage-blob", "azure-storage-file-datalake", "backports.tarfile", "black",
    "databricks-agents", "databricks-sdk", "dbus-python", "deltalake", "distlib", "docstring-to-markdown",
    "facets-overview", "fastapi", "fqdn", "google-api-core", "google-cloud-core", "google-cloud-storage",
    "google-crc32c", "google-resumable-media", "googleapis-common-protos", "grpcio-status", "hf-xet",
    "httplib2", "huggingface_hub", "importlib_metadata", "inflect", "ipyflow-core", "ipython-genutils",
    "isodate", "isoduration", "jaraco.collections", "jaraco.context", "jaraco.functools", "jaraco.text",
    "jiter", "jsonpatch", "jupyter-events", "jupyter-lsp", "jupyter_server_terminals", "jupyterlab_server",
    "langchain-core", "langchain-openai", "langsmith", "launchpadlib", "lazr.restfulclient", "lazr.uri",
    "litellm", "markdown-it-py", "marshmallow", "mccabe", "mdurl", "mlflow-skinny", "mmh3", "msal",
    "msal-extensions", "mypy-extensions", "nodeenv", "oauthlib", "opentelemetry-api", "opentelemetry-proto",
    "opentelemetry-sdk", "opentelemetry-semantic-conventions", "orjson", "pathspec", "patsy", "plotly",
    "prometheus_client", "propcache", "proto-plus", "psycopg2", "pyasn1-modules", "pyccolo", "pyflakes",
    "pyiceberg", "pyodbc", "pyright", "pyroaring", "python-dotenv", "python-lsp-jsonrpc", "python-lsp-server",
    "pytoolconfig", "requests-toolbelt", "rich", "rope", "rpds-py", "s3transfer", "scikit-learn",
    "shellingham", "ssh-import-id", "strictyaml", "threadpoolctl", "tiktoken", "tokenize_rt", "tokenizers",
    "typeguard", "typer-slim", "typing-inspect", "unattended-upgrades", "uri-template", "uuid_utils",
    "wadllib", "whatthepatch", "whenever", "yggdrasil", "ygg",
}

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ModuleUploadCacheKey:
    cluster_id: str
    context_key: str
    remote_libs: str
    local_root: Path


@dataclass(slots=True)
class _ModuleUploadCacheEntry:
    tree_mtime: float
    remote_pkg_dir: str
    uploaded_at: float


_MODULE_UPLOAD_CACHE: dict[_ModuleUploadCacheKey, _ModuleUploadCacheEntry] = {}
_MODULE_UPLOAD_CACHE_LOCK = RLock()


@dataclass
class CommandExecution:
    context: ExecutionContext
    command_id: Optional[str] = None

    language: Optional[Language] = field(default=None, repr=False, compare=False, hash=False)
    command: Optional[str] = field(default=None, repr=False, compare=False, hash=False)

    pexit: bool = field(default=False, repr=False, compare=False, hash=False)
    pyfunc: Optional[Callable] = field(default=None, repr=False, compare=False, hash=False)
    environ: Optional[Mapping] = field(default=None, repr=False, compare=False, hash=False)

    _ser_pyfunc: Optional[Serialized] = field(default=None, repr=False, compare=False, hash=False)
    _details: Optional[CommandStatusResponse] = field(default=None, init=False, repr=False, compare=False, hash=False)
    _remote_payload_path: Optional[str] = field(default=None, init=False, repr=False, compare=False, hash=False)
    _shutdown_hook: Any = field(default=None, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self):
        if self.environ and not isinstance(self.environ, Mapping):
            try:
                self.environ = dict(self.environ)
            except Exception as e:
                raise ValueError(
                    f"environ must be a mapping or convertible to dict, got {type(self.environ)}"
                ) from e

        if self.language is None:
            self.language = (
                Language.PYTHON
                if self.pyfunc is not None
                else (self.context.language or Language.PYTHON)
            )

    def __getstate__(self):
        return serialize_dataclass_state(self)

    def __setstate__(self, state):
        restore_dataclass_state(self, state)
        self.__post_init__()

    @property
    def client(self):
        return self.context.cluster.client

    @property
    def cluster_id(self):
        return self.context.cluster.cluster_id

    def __repr__(self):
        return f"{self.__class__.__name__}<url={self.url()}>"

    def __str__(self):
        return self.url().to_string()

    def url(self) -> URL:
        url = self.context.url()
        return url.with_query_items(
            {
                **url.query_dict,
                "context_id": self.context.context_id or "unknown",
                "command_id": self.command_id or "unknown",
            }
        )

    def _check_py_versions(self):
        current_version = PyEnv.current().version_info
        target_version = self.context.cluster.python_version_info
        cur = (current_version.major, current_version.minor)
        tgt = (target_version.major, target_version.minor)
        if cur != tgt:
            raise RuntimeError(
                "Python version mismatch.\n"
                f"  Local interpreter:  Python {current_version.major}.{current_version.minor}.{getattr(current_version, 'micro', '?')}\n"
                f"  Cluster runtime:     Python {target_version.major}.{target_version.minor}.{getattr(target_version, 'micro', '?')}\n\n"
                "Fix: use a matching Python minor version locally (major.minor must match).\n\n"
                "Quick setup with uv:\n"
                f"  uv venv --python {target_version.major}.{target_version.minor} --seed\n"
                "  source .venv/bin/activate        # macOS/Linux\n"
                "  .venv\\Scripts\\activate         # Windows\n"
                "  uv pip install ygg\n"
            )

    def __call__(self, *args, **kwargs):
        install_modules = kwargs.pop("__install_modules", None) or []
        force_local = kwargs.pop("__force_local", False)
        environ = {k: v for k, v in (self.environ.items() if self.environ else {})}

        if self.pyfunc is not None:
            if force_local or self.context.is_in_databricks_environment():
                for k, v in environ.items():
                    if v is not None:
                        os.environ[str(k)] = str(v)
                return self.pyfunc(*args, **kwargs)

            if len(args) == 1 and not kwargs:
                to_decorate = args[0]
                if callable(to_decorate):
                    decorated = self.pyfunc(to_decorate)
                    return CommandExecution(
                        context=self.context,
                        pyfunc=decorated,
                        environ=self.environ,
                        language=Language.PYTHON,
                    )

            language_to_execute = Language.PYTHON
            command_to_execute, remote_payload_path = self.context.make_python_function_command(
                serfunc=self.serialized_pyfunc(),
                args=args,
                kwargs=kwargs,
                environ=environ,
                pyfunc=self.pyfunc,
            )
        else:
            language_to_execute = self.language
            command_to_execute = self.command
            remote_payload_path = None

        assert command_to_execute, f"Cannot call {self}, missing command"

        temp = CommandExecution(
            context=self.context,
            command=command_to_execute,
            language=language_to_execute,
            pyfunc=self.pyfunc,
            environ=self.environ,
        )
        temp._remote_payload_path = remote_payload_path

        if install_modules:
            for module in install_modules:
                temp.install_module(module, auto_pip=False)

        return temp.start().result(raise_error=True)

    @property
    def state(self):
        if not self.command_id:
            return None
        return self.details.status

    @property
    def running(self):
        return self.command_id is not None and self.state in PENDING_STATES

    @property
    def done(self):
        return self.command_id is not None and self.state in DONE_STATES

    def serialized_pyfunc(self):
        if self._ser_pyfunc is not None:
            return self._ser_pyfunc
        if self.pyfunc is None:
            raise ValueError(f"{self} does not contain python function to serialize")
        self._ser_pyfunc = serialize(self.pyfunc)
        return self._ser_pyfunc

    def _register_shutdown_cancel(self) -> None:
        if self._shutdown_hook is not None or not self.command_id:
            return
        try:
            self._shutdown_hook = yg_shutdown.register(self._unsafe_cancel)
        except Exception:
            LOGGER.debug(
                "Failed to register shutdown handler for command %s",
                self.command_id,
                exc_info=True,
            )

    def _unregister_shutdown_cancel(self) -> None:
        hook = self._shutdown_hook
        self._shutdown_hook = None
        if hook is None:
            return

        try:
            try:
                yg_shutdown.unregister(hook)
            except Exception:
                yg_shutdown.unregister(self._unsafe_cancel)
        except Exception:
            LOGGER.debug(
                "Failed to unregister shutdown handler for command %s",
                self.command_id,
                exc_info=True,
            )

    def _clear_active_command(self) -> None:
        self._details = None
        self.command_id = None
        self._unregister_shutdown_cancel()

    def _mark_done_if_terminal(self) -> None:
        if self._details is not None and self._details.status in DONE_STATES:
            self._unregister_shutdown_cancel()

    def start(self, reset: bool = False):
        if self.command_id:
            if not reset:
                return self
            if not self.done:
                self.cancel(wait=False, raise_error=False)
            else:
                self._clear_active_command()

        client = self.client.workspace_client().command_execution
        assert self.command, f"Missing command arg in {self}"

        self.context = self.context.connect(
            language=self.language or self.context.language or Language.PYTHON,
            reset=False,
        )
        self.context.touch()
        context_id = self.context.context_id

        def _execute(curr_context_id: str):
            return client.execute(
                cluster_id=self.cluster_id,
                context_id=curr_context_id,
                language=self.language,
                command=self.command,
            ).response

        try:
            details = _execute(context_id)
        except PermissionDenied as perm_denied:
            msg = str(perm_denied)
            match = re.search(r"but the single user of this cluster is\s*'([^']+)'", msg)
            single_user_name = match.group(1) if match else None
            if single_user_name and "@" not in single_user_name:
                groups = self.client.iam.groups
                group = next(
                    groups.list(
                        name=single_user_name,
                        limit=1,
                        client_type=ClientType.WORKSPACE,
                        raise_error=False,
                    ),
                    None,
                )
                if group is None:
                    raise ResourceDoesNotExist(
                        f"Cluster is single-user with user '{single_user_name}', but no matching IAM group found"
                    ) from perm_denied
                group.add_member(self.client.iam.users.current_user)
                details = _execute(context_id)
            else:
                raise
        except InternalError:
            self.context = self.context.connect(reset=True, language=self.language or Language.PYTHON)
            context_id = self.context.context_id
            details = _execute(context_id)
        except Exception as e:
            msg = str(e).lower()
            if "context" in msg or "context_id" in msg or "invalid context" in msg:
                self.context = self.context.connect(reset=True, language=self.language or Language.PYTHON)
                context_id = self.context.context_id
                details = _execute(context_id)
            else:
                raise

        self.command_id = details.id
        self._details = None
        self._register_shutdown_cancel()
        return self

    def cancel(self, wait: WaitingConfigArg = False, raise_error: bool = False):
        if not self.command_id or not self.context.context_id:
            self._unregister_shutdown_cancel()
            return self

        wait_cfg = WaitingConfig.check_arg(wait)
        client = self.client.workspace_client().command_execution
        command_id = self.command_id

        try:
            response = client.cancel(
                cluster_id=self.cluster_id,
                context_id=self.context.context_id,
                command_id=command_id,
            )
            if wait_cfg:
                response.result(timeout=wait_cfg.timeout_timedelta)
        except Exception as exc:
            if raise_error:
                raise
            LOGGER.debug("%s failed to cancel command %s: %s", self, command_id, exc)
        finally:
            self._clear_active_command()

        return self

    def _unsafe_cancel(self):
        return self.cancel(wait=False, raise_error=False)

    def _command_status(self) -> CommandStatusResponse | None:
        if not self.command_id or not self.context.context_id:
            return None

        client = self.client.workspace_client().command_execution
        try:
            return client.command_status(
                cluster_id=self.cluster_id,
                context_id=self.context.context_id,
                command_id=self.command_id,
            )
        except InternalError:
            self.context.cluster.ensure_running()
            self.context = self.context.connect(reset=True, language=self.language or Language.PYTHON)
            self.start(reset=True)
            return client.command_status(
                cluster_id=self.cluster_id,
                context_id=self.context.context_id,
                command_id=self.command_id,
            )

    @property
    def details(self) -> CommandStatusResponse:
        if self._details is None:
            self._details = self._command_status()
            if self._details is None:
                self._unregister_shutdown_cancel()
                return CommandStatusResponse(status=CommandStatus.FINISHED)

        elif self._details.status not in DONE_STATES:
            self._details = self._command_status()
            if self._details is None:
                self._unregister_shutdown_cancel()
                return CommandStatusResponse(status=CommandStatus.FINISHED)

        self._mark_done_if_terminal()
        return self._details

    @details.setter
    def details(self, value: Optional[CommandStatusResponse]):
        self._details = value
        if value is not None:
            assert isinstance(value, CommandStatusResponse), (
                f"{self}.details must be CommandStatusResponse, got {type(value)}"
            )
            self.command_id = value.id
            self._mark_done_if_terminal()
        else:
            self._unregister_shutdown_cancel()

    def raise_for_status(self):
        if self.state in FAILED_STATES:
            raise_error_from_response(response=self.details, language=self.language)
        return self

    def wait(self, wait: WaitingConfigArg = True, raise_error: bool = True):
        if not self.command_id:
            return self.start().wait(wait=wait, raise_error=raise_error)

        wait_cfg = WaitingConfig.check_arg(wait)
        iteration = 0
        start_time = time.time()

        if wait_cfg:
            while self.running:
                try:
                    wait_cfg.sleep(iteration=iteration, start=start_time)
                    iteration += 1
                except KeyboardInterrupt:
                    self.cancel(wait=False, raise_error=False)
                    raise

        if raise_error:
            self.raise_for_status()

        return self

    def decode_response(
        self,
        response: CommandStatusResponse,
        language: Language,
        raise_error: bool = True,
        tag: str = "__CALL_RESULT__",
    ) -> tuple[str | None, str]:
        raise_error_from_response(response=response, language=language, raise_error=raise_error)
        results = response.results
        if results.result_type != ResultType.TEXT:
            raise NotImplementedError(f"Cannot decode result from {response}")

        data = results.data or ""
        raw_result = data
        if tag in raw_result:
            logs_text, raw_result = raw_result.split(tag, 1)
            return logs_text, raw_result
        return None, raw_result

    def result(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        tag: str = "__CALL_RESULT__",
    ) -> Any:
        try:
            return self._result_inner(wait=wait, raise_error=raise_error, tag=tag)
        finally:
            self._cleanup_remote_payload()

    def _cleanup_remote_payload(self) -> None:
        path = self._remote_payload_path
        if path is None:
            return
        try:
            self.client.dbfs_path(path).remove()
        except Exception:
            LOGGER.debug("Failed to clean up remote payload path %s (non-fatal)", path, exc_info=True)
        finally:
            self._remote_payload_path = None

    def _result_inner(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        tag: str = "__CALL_RESULT__",
    ) -> Any:
        wait_cfg = WaitingConfig.check_arg(wait)
        installed_modules: set[str] = set()
        last_exc = None
        logs = None
        raw_result = None

        for _attempt in range(wait_cfg.total_try_count):
            try:
                self.wait(wait=wait_cfg, raise_error=raise_error)
                logs, raw_result = self.decode_response(
                    response=self.details,
                    language=self.language,
                    raise_error=raise_error,
                    tag=tag,
                )
                last_exc = None
                break

            except (InternalError, ClientTerminatedSession) as e:
                last_exc = e
                self.context = self.context.connect(reset=True, language=self.language or Language.PYTHON)
                self.start(reset=True)
                continue

            except ModuleNotFoundError as e:
                last_exc = e
                module_name = e.name
                root_module = module_name.split(".", 1)[0] if module_name else None
                if not root_module:
                    raise
                if root_module in installed_modules:
                    raise RuntimeError(
                        f"Remote module '{root_module}' is still missing after upload retry"
                    ) from e
                self.install_module(local_root=root_module, force=True)
                installed_modules.add(root_module)
                self.start(reset=True)
                continue

        if raise_error and last_exc is not None:
            raise last_exc

        if logs is not None:
            prefix = "DBXPATH:"
            if raw_result.startswith(prefix):
                result_path = raw_result[len(prefix):]
                try:
                    raw_result = self.client.dbfs_path(result_path).read_bytes().decode("ascii")
                finally:
                    self.client.dbfs_path(result_path).remove()

            if (
                (raw_result.startswith("[") and raw_result.endswith("]"))
                or (raw_result.startswith("{") and raw_result.endswith("}"))
            ):
                return json.loads(raw_result)

            return loads(raw_result)

        return raw_result

    @staticmethod
    def _get_local_module_path(obj: str | Path | Callable) -> Path:
        if isinstance(obj, Path):
            if not obj.exists():
                raise FileNotFoundError(f"Provided local module path does not exist: {obj}")
            return obj.resolve()

        module_name = obj if isinstance(obj, str) else getattr(obj, "__module__", None)
        if not module_name:
            raise ValueError(
                f"Module name must be a string or a callable with __module__ attribute, got {module_name}"
            )

        root_module = module_name.split(".", 1)[0]
        local_root = Path(PyEnv.get_root_module_directory(module_name=root_module)).resolve()
        if not local_root.exists():
            raise FileNotFoundError(
                f"Resolved local module path does not exist for '{root_module}': {local_root}"
            )
        return local_root

    @staticmethod
    def _zip_local_module(local_root: str | Path) -> tuple[str, bytes]:
        local_root = CommandExecution._get_local_module_path(local_root)
        buf = io.BytesIO()

        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            if local_root.is_dir():
                parent = local_root.parent
                for path in local_root.rglob("*"):
                    if path.is_dir():
                        continue
                    if path.name == ".DS_Store":
                        continue
                    if "__pycache__" in path.parts:
                        continue
                    if any(part.endswith(".egg-info") for part in path.parts):
                        continue
                    if any(part.endswith(".dist-info") for part in path.parts):
                        continue
                    zf.write(path, str(path.relative_to(parent)))
            else:
                zf.write(local_root, arcname=local_root.name)

        return local_root.name, buf.getvalue()

    def _module_upload_cache_key(self, local_root: Path) -> _ModuleUploadCacheKey:
        return _ModuleUploadCacheKey(
            cluster_id=str(self.cluster_id),
            context_key=str(self.context.context_key),
            remote_libs=self.context.remote_metadata.libs_path.rstrip("/"),
            local_root=local_root.resolve(),
        )

    def install_module(
        self,
        local_root: str | Path,
        *,
        auto_pip: bool = True,
        check_diffs: bool = False,
        force: bool = False,
    ) -> str | None:
        local_root = self._get_local_module_path(obj=local_root)
        module_name = local_root.name
        remote_libs = self.context.remote_metadata.libs_path.rstrip("/")
        remote_pkg_dir = f"{remote_libs}/{module_name}"

        if auto_pip and any(part == "site-packages" for part in local_root.parts):
            spec = _get_local_distribution_compatible_spec(
                module_name=module_name,
                raise_error=True,
                exclude_modules=ALREADY_LIBS,
            )
            if spec:
                self.context.cluster.install_libraries(
                    libraries=[spec],
                    raise_error=True,
                    remove_failed=True,
                )
            return spec

        current_mtime = _get_tree_mtime(local_root)
        cache_key = self._module_upload_cache_key(local_root)

        if not force:
            with _MODULE_UPLOAD_CACHE_LOCK:
                cached = _MODULE_UPLOAD_CACHE.get(cache_key)
                if check_diffs and cached is not None and cached.tree_mtime >= current_mtime:
                    return remote_libs

        root_module, zip_bytes = self._zip_local_module(local_root=local_root)
        payload_b64 = base64.b64encode(zip_bytes).decode("ascii")
        payload_json = json.dumps(payload_b64)

        bootstrap = f"""
import base64
import importlib
import io
import os
import shutil
import zipfile

root_module = {str(root_module)!r}
remote_libs = os.path.expanduser({str(self.context.remote_metadata.libs_path)!r})
remote_pkg_dir = os.path.expanduser({remote_pkg_dir!r})
payload_b64 = {payload_json}

os.makedirs(remote_libs, exist_ok=True)
if os.path.exists(remote_pkg_dir):
    shutil.rmtree(remote_pkg_dir)

buf = io.BytesIO(base64.b64decode(payload_b64.encode("ascii")))
with zipfile.ZipFile(buf, "r") as zf:
    zf.extractall(remote_libs)

_ = importlib.import_module(root_module)
print(remote_libs, flush=True)
"""
        res = self.context.command(language=Language.PYTHON, command=bootstrap).start().result()

        with _MODULE_UPLOAD_CACHE_LOCK:
            _MODULE_UPLOAD_CACHE[cache_key] = _ModuleUploadCacheEntry(
                tree_mtime=current_mtime,
                remote_pkg_dir=remote_pkg_dir,
                uploaded_at=time.time(),
            )

        return (res or "").strip() or remote_libs



def raise_error_from_response(
    response: CommandStatusResponse,
    language: Language,
    raise_error: bool = True,
):
    if raise_error:
        results = response.results
        if results.result_type == ResultType.ERROR:
            message = results.cause or "Command execution failed"
            if "client terminated the session" in message.lower():
                raise ClientTerminatedSession(message)
            if language == Language.PYTHON:
                raise_parsed_traceback(message)
            raise CommandExecutionError(str(response))


def _get_local_distribution_compatible_spec(
    module_name: str,
    raise_error: bool,
    exclude_modules: set[str] | None = None,
) -> str | None:
    import importlib.metadata as im

    exclude_modules = exclude_modules or set()
    root_module = module_name.split(".", 1)[0]
    if root_module in exclude_modules:
        return None if not raise_error else _raise_excluded(module_name)

    pkg_map = im.packages_distributions()
    dist_names = pkg_map.get(root_module) or [root_module]
    last_err: Exception | None = None

    for dist_name in dist_names:
        try:
            version = im.version(dist_name)
        except Exception as e:
            last_err = e
            continue

        parts = str(version).split(".")
        major = parts[0] if len(parts) > 0 else "0"
        minor = parts[1] if len(parts) > 1 else "0"
        return f"{dist_name}~={major}.{minor}"

    if not raise_error:
        return None

    if last_err is not None:
        raise RuntimeError(
            f"Cannot resolve installed distribution/version for module '{module_name}'"
        ) from last_err

    raise RuntimeError(f"Cannot resolve installed distribution/version for module '{module_name}'")


def _raise_excluded(module_name: str) -> None:
    raise RuntimeError(f"Module '{module_name}' is excluded from local distribution resolution")


def _get_tree_mtime(root: Path) -> float:
    root = root.resolve()
    if root.is_file():
        return root.stat().st_mtime

    latest = root.stat().st_mtime
    for p in root.rglob("*"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest