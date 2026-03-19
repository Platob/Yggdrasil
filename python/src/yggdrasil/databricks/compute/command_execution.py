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
from typing import Optional, Any, Callable, Mapping

from databricks.sdk.client_types import ClientType
from databricks.sdk.errors import InternalError, PermissionDenied, ResourceDoesNotExist
from databricks.sdk.service.compute import (
    Language, CommandStatusResponse, CommandStatus, ResultType
)

from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg, serialize_dataclass_state, restore_dataclass_state
from yggdrasil.environ import PyEnv
from yggdrasil.io.url import URL
from yggdrasil.pyutils.exceptions import raise_parsed_traceback
from .exceptions import ClientTerminatedSession, CommandExecutionError
from .execution_context import ExecutionContext

DONE_STATES = {
    CommandStatus.FINISHED, CommandStatus.CANCELLED, CommandStatus.ERROR
}

PENDING_STATES = {CommandStatus.RUNNING, CommandStatus.QUEUED}

FAILED_STATES = {
    CommandStatus.ERROR, CommandStatus.CANCELLED
}


__all__ = [
    "CommandExecution"
]


LOGGER = logging.getLogger(__name__)


@dataclass
class CommandExecution:
    context: ExecutionContext
    command_id: Optional[str] = None

    language: Optional[Language] = field(default=None, repr=False, compare=False, hash=False)
    command: Optional[str] = field(default=None, repr=False, compare=False, hash=False)
    ephemeral: bool = field(default=True, repr=False, compare=False, hash=False)

    pyfunc: Optional[Callable] = field(default=None, repr=False, compare=False, hash=False)
    environ: Optional[Mapping] = field(default=None, repr=False, compare=False, hash=False)

    _details: Optional[CommandStatusResponse] = field(default=None, init=False, repr=False, compare=False, hash=False)
    _local_checks: dict[Path, float] = field(default_factory=dict, init=False, repr=False, compare=False, hash=False)
    _remote_payload_path: Optional[str] = field(default=None, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self):
        if self.environ:
            if not isinstance(self.environ, Mapping):
                try:
                    self.environ = dict(self.environ)
                except Exception as e:
                    raise ValueError(
                        f"environ must be a mapping or convertible to dict, got {type(self.environ)}"
                    ) from e

        if self.language is None:
            self.language = Language.PYTHON if self.pyfunc is not None else self.context.language

        if self._local_checks is None:
            self._local_checks = {}

    def __getstate__(self):
        return serialize_dataclass_state(self)

    def __setstate__(self, state):
        restore_dataclass_state(self, state)
        self.__post_init__()

    def _check_py_versions(self):
        current_version = PyEnv.current().version_info
        target_version = self.context.cluster.python_version_info

        cur = (current_version.major, current_version.minor)
        tgt = (target_version.major, target_version.minor)

        if cur != tgt:
            raise RuntimeError(
                "Python version mismatch.\n"
                f"  Local interpreter:  Python {current_version.major}.{current_version.minor}.{getattr(current_version, 'micro', '?')}\n"
                f"  Cluster runtime:     Python {target_version.major}.{target_version.minor}.{getattr(target_version, 'micro', '?')}\n"
                "\n"
                "Fix: use a matching Python minor version locally (major.minor must match).\n"
                "\n"
                "Quick setup with uv:\n"
                f"  uv venv --python {target_version.major}.{target_version.minor} --seed\n"
                "  source .venv/bin/activate        # macOS/Linux\n"
                "  .venv\\Scripts\\activate           # Windows\n"
                "  uv pip install ygg\n"
            )

    def __call__(
        self,
        *args,
        **kwargs
    ):
        # self._check_py_versions()
        install_modules = kwargs.pop("__install_modules", None) or []
        force_local = kwargs.pop("__force_local", False)

        environ = {
            k: v for k, v in (self.environ.items() if self.environ else {})
        }

        if self.pyfunc is not None:
            if force_local or self.context.is_in_databricks_environment():
                for k, v in environ.items():
                    if v:
                        os.environ[str(k)] = str(v)
                return self.pyfunc(*args, **kwargs)

            if len(args) == 1 and not kwargs:
                to_decorate = args[0]

                if callable(to_decorate):
                    decorated = self.pyfunc(to_decorate)

                    # Add dependency in self._local_checks to trigger re-upload if the decorated function's module file changes on disk
                    root_module = self._get_local_module_path(to_decorate)
                    self._local_checks[root_module] = 0

                    built = CommandExecution(
                        context=self.context,
                        pyfunc=decorated,
                        environ=self.environ,
                        language=Language.PYTHON
                    )

                    built._local_checks.update(self._local_checks)
                    return built

            # Create command
            language_to_execute = Language.PYTHON
            command_to_execute, remote_payload_path = self.context.make_python_function_command(
                func=self.pyfunc,
                args=args,
                kwargs=kwargs,
                environ=environ
            )
        else:
            language_to_execute = self.language
            command_to_execute = self.command
            remote_payload_path = None

        assert command_to_execute, "Cannot call %s, missing command" % self

        temp = CommandExecution(
            context=self.context.create(language=language_to_execute),
            command=command_to_execute,
            language=language_to_execute,
            ephemeral=True
        )
        temp._local_checks = self._local_checks
        temp._remote_payload_path = remote_payload_path

        if install_modules:
            for m in install_modules:
                temp.install_module(m, auto_pip=False)

        # Check local dependencies
        for local_root in temp._local_checks.keys():
            if "site-packages" not in local_root.parts:
                temp.install_module(local_root=local_root, check_diffs=True)

        try:
            return (
                temp
                .start()
                .result(raise_error=True)
            )
        finally:
            temp.context.close(wait=False, raise_error=False)

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

    def start(self, reset: bool = False):
        if self.command_id:
            if not reset:
                return self

            if not self.done:
                self.cancel(wait=False)

            self._details = None
            self.command_id = None

        client = self.client.workspace_client().command_execution

        assert self.command, "Missing command arg in %s" % self

        try:
            details = client.execute(
                cluster_id=self.cluster_id,
                context_id=self.context_id,
                language=self.language,
                command=self.command,
            ).response
        except PermissionDenied as perm_denied:
            msg = str(perm_denied)
            m = re.search(r"but the single user of this cluster is\s*'([^']+)'", msg)
            single_user_name = m.group(1) if m else None

            if single_user_name and "@" not in single_user_name:  # likely a single-user cluster with a username instead of group name
                groups = self.client.iam.groups
                group = next(
                    groups.list(
                        name=single_user_name,
                        limit=1,
                        client_type=ClientType.WORKSPACE,
                        raise_error=False
                    ),
                    None
                )

                if group is None:
                    raise ResourceDoesNotExist(
                        "Cluster is single-user with user '%s', but no matching IAM group found" % single_user_name
                    ) from perm_denied
                try:
                    group.add_member(self.client.iam.users.current_user)
                except Exception as inner_e:
                    raise inner_e from perm_denied

                client = self.client.workspace_client().command_execution  # refresh client
                try:
                    details = client.execute(
                        cluster_id=self.cluster_id,
                        context_id=self.context_id,
                        language=self.language,
                        command=self.command,
                    ).response
                except Exception as inner_e:
                    raise inner_e from perm_denied
            else:
                raise
        except InternalError:
            self.context = self.context.connect(reset=True)
            client = self.client.workspace_client().command_execution  # refresh client
            details = client.execute(
                cluster_id=self.cluster_id,
                context_id=self.context_id,
                language=self.language,
                command=self.command,
            ).response
        except Exception as e:
            msg = str(e).lower()
            if "context" in msg or "context_id" in msg or "invalid context" in msg:
                self.context = self.context.connect(reset=True)
                client = self.client.workspace_client().command_execution  # refresh client
                details = client.execute(
                    cluster_id=self.cluster_id,
                    context_id=self.context_id,
                    language=self.language,
                    command=self.command,
                ).response
            else:
                raise

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
            client = self.client.workspace_client().command_execution

            response = client.cancel(
                cluster_id=self.cluster_id,
                context_id=self.context_id,
                command_id=self.command_id
            )

            if wait:
                response.result(timeout=wait.timeout_timedelta)

        return self

    @property
    def client(self):
        return self.context.cluster.client

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
        if not self.command_id:
            return None
        return self.details.status

    @property
    def running(self):
        return self.command_id is not None and self.state in PENDING_STATES

    @property
    def done(self):
        return self.command_id is not None and self.state in DONE_STATES

    def _command_status(self):
        client = self.client.workspace_client().command_execution

        try:
            return client.command_status(
                cluster_id=self.cluster_id,
                context_id=self.context_id,
                command_id=self.command_id
            )
        except InternalError:
            self.context.cluster.ensure_running()
            self.context = self.context.connect(reset=True)
            self.start(reset=True)

            client = self.client.workspace_client().command_execution

            return client.command_status(
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

        if wait:
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

    def decode_response(
        self,
        response: CommandStatusResponse,
        language: Language,
        raise_error: bool = True,
        tag: str = "__CALL_RESULT__",
    ) -> tuple[str | None, str]:
        """Mirror the old Cluster.execute_command result handling.

        Args:
            response: Raw command execution response.
            language: Language executed
            raise_error: Raise error if response is failed
            tag: Result tag

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
            return logs_text, raw_result

        return None, raw_result

    def result(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        tag: str = "__CALL_RESULT__",
    ) -> Any:
        """Wait for the command to finish and return the deserialized result.

        The remote snippet (see
        :meth:`ExecutionContext.make_python_function_command`) always
        prints a tagged payload — either the serialised return value or
        a serialised exception.  This method:

        1. Waits for the command to reach a terminal state.
        2. Decodes the tagged payload from stdout.
        3. Deserialises the payload.
        4. If the payload is a ``BaseException``, re-raises it.
        5. Otherwise returns the value.
        6. Cleans up the temporary remote payload file if one was used
           (see :attr:`_remote_payload_path`).

        Transient failures (``InternalError``, ``ClientTerminatedSession``,
        ``ModuleNotFoundError``) trigger automatic reconnection / module
        upload and retry up to ``wait.total_try_count`` times.

        Parameters
        ----------
        wait:
            Waiting / retry configuration.
        raise_error:
            Whether to raise on non-2xx command status.
        tag:
            Sentinel used to split logs from the return payload.

        Returns
        -------
        Any
            The deserialised return value of the remote function.

        Raises
        ------
        BaseException
            Re-raised from the remote side when the function raised.
        RuntimeError
            When a missing module persists after upload retry.
        """
        try:
            return self._result_inner(
                wait=wait,
                raise_error=raise_error,
                tag=tag,
            )
        finally:
            self._cleanup_remote_payload()

    def _cleanup_remote_payload(self) -> None:
        """Delete the temporary remote payload file, if one was used.

        Failures are logged but never propagated — the payload file
        lives inside the cluster's ephemeral scratch directory and will
        be garbage-collected eventually even if the explicit delete
        fails.
        """
        path = self._remote_payload_path
        if path is None:
            return

        self._remote_payload_path = None

        try:
            dbfs_path = self.client.dbfs_path(path)
            dbfs_path.rmfile(missing_ok=True)
        except Exception:
            LOGGER.debug(
                "Failed to clean up remote payload path %s (non-fatal)",
                path,
                exc_info=True,
            )

    def _result_inner(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        tag: str = "__CALL_RESULT__",
    ) -> Any:
        """Core result-retrieval logic (called by :meth:`result`)."""
        wait = WaitingConfig.check_arg(wait)
        installed_modules: set[str] = set()
        last_exc, logs, raw_result = None, None, None

        for attempt in range(wait.total_try_count):
            try:
                self.wait(wait=wait, raise_error=raise_error)
                logs, raw_result = self.decode_response(
                    response=self.details,
                    language=self.language,
                    raise_error=raise_error,
                    tag=tag
                )
                last_exc = None
                break
            except (InternalError, ClientTerminatedSession) as e:
                last_exc = e
                self.context = self.context.connect(reset=True)
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

                LOGGER.warning(
                    "Remote module '%s' not found; uploading zipped local module to remote libs path",
                    root_module,
                )

                self.install_module(local_root=root_module)
                installed_modules.add(root_module)

                self.start(reset=True)
                continue

        if raise_error and last_exc is not None:
            raise last_exc

        if logs is not None:
            from yggdrasil.pickle.ser import loads

            # If the result was too large to inline, the remote side
            # wrote it to a tmp path and printed DBXPATH:<path>.
            _DBXPATH_PREFIX = self.context._DBXPATH_PREFIX
            if raw_result.startswith(_DBXPATH_PREFIX):
                result_path = raw_result[len(_DBXPATH_PREFIX):]
                try:
                    raw_result = (
                        self.client
                        .dbfs_path(result_path)
                        .read_bytes()
                        .decode("ascii")
                    )
                finally:
                    try:
                        self.client.dbfs_path(result_path).remove(recursive=True)
                    except Exception:
                        LOGGER.debug(
                            "Failed to clean up remote result path %s (non-fatal)",
                            result_path,
                            exc_info=True,
                        )

            if (
                (raw_result.startswith("[") and raw_result.endswith("]"))
                or (raw_result.startswith("{") and raw_result.endswith("}"))
            ):
                return json.loads(raw_result)

            value = loads(raw_result)

            return value

        return raw_result


    @staticmethod
    def _get_local_module_path(
        obj: str | Path | Callable,
    ) -> Path:
        """
        Get the local filesystem path for the root local module/package.

        Returns:
            Path to the root module/package.
        """
        if isinstance(obj, Path):
            if not obj.exists():
                raise FileNotFoundError(f"Provided local module path does not exist: {obj}")
            return obj.resolve()

        if not isinstance(obj, str):
            module_name = getattr(obj, "__module__", None)
        else:
            module_name = obj

        if not module_name:
            raise ValueError(f"Module name must be a string or a callable with __module__ attribute, got {module_name}")

        root_module = module_name.split(".", 1)[0]
        local_root = Path(PyEnv.get_root_module_directory(module_name=root_module)).resolve()

        if not local_root.exists():
            raise FileNotFoundError(
                f"Resolved local module path does not exist for '{root_module}': {local_root}"
            )

        return local_root

    @staticmethod
    def _zip_local_module(
        local_root: str | Path
    ) -> tuple[str, bytes]:
        """
        Build a zip archive for the root local module/package.

        Returns:
            (root_module_name, zip_bytes)
        """
        local_root = CommandExecution._get_local_module_path(local_root)

        buf = io.BytesIO()

        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            if local_root.is_dir():
                parent = local_root.parent
                for path in local_root.rglob("*"):
                    if path.is_dir():
                        continue

                    # avoid junk
                    if path.name in {".DS_Store"}:
                        continue
                    if "__pycache__" in path.parts:
                        continue
                    if any(part.endswith(".egg-info") for part in path.parts):
                        continue
                    if any(part.endswith(".dist-info") for part in path.parts):
                        continue

                    arcname = str(path.relative_to(parent))
                    zf.write(path, arcname=arcname)
            else:
                zf.write(local_root, arcname=local_root.name)

        return local_root.name, buf.getvalue()

    def install_module(
        self,
        local_root: str | Path,
        *,
        auto_pip: bool = True,
        check_diffs: bool = False
    ) -> str:
        """
        Zip local root module/package, upload via command payload, unzip directly
        into the remote libs path already injected by syspath_lines().

        Returns:
            Remote libs path.
        """
        local_root: Path = self._get_local_module_path(obj=local_root)
        module_name = local_root.name

        # Check diffs
        if check_diffs:
            last_mtime = self._local_checks.get(local_root, 0)
            current_mtime = _get_tree_mtime(local_root)

            if current_mtime <= last_mtime:
                # update local check timestamp to avoid redundant checks in the future, but skip upload
                self._local_checks[local_root] = current_mtime

                return self.context.remote_metadata.libs_path.rstrip("/")

        if auto_pip and any(part == "site-packages" for part in local_root.parts):
            spec = _get_local_distribution_compatible_spec(
                module_name=module_name, raise_error=False
            )

            return self._remote_pip_install(
                spec,
                upgrade=True,
            )

        root_module, zip_bytes = self._zip_local_module(local_root=local_root)

        payload_b64 = base64.b64encode(zip_bytes).decode("ascii")
        payload_json = json.dumps(payload_b64)

        remote_libs = self.context.remote_metadata.libs_path.rstrip("/")
        remote_pkg_dir = f"{remote_libs}/{root_module}"

        bootstrap = f"""
import base64
import os
import io
import shutil
import zipfile
import importlib

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
        res = (
            self.context.command(
                language=Language.PYTHON,
                command=bootstrap
            )
            .start()
            .result()
        )

        LOGGER.info(
            "Installed local module '%s' to remote libs path '%s'",
            local_root,
            remote_libs
        )

        return (res or "").strip() or remote_libs

    def _remote_pip_install(
        self,
        *packages: str,
        upgrade: bool = False,
        editable: bool = False,
        quiet: bool = True,
    ) -> list[tuple[str, str]]:
        """
        Install packages into the remote context-local libs path.

        Packages are installed with:
            python -m pip install --target <remote_libs> ...

        Args:
            *packages: Package specs like "orjson", "pydantic==2.11.1", etc.
            upgrade: Add --upgrade
            editable: Add --editable (mainly useful for path/vcs specs)
            quiet: Add --quiet

        Returns:
            Parsed list of installed packages from `pip list --path <remote_libs>`.
        """
        pkgs = [str(p).strip() for p in packages if str(p).strip()]
        if not pkgs:
            return []

        remote_libs = self.context.remote_metadata.libs_path.rstrip("/")

        bootstrap = f"""
import json
import os
import shlex
import subprocess
import sys

remote_libs = os.path.expanduser({remote_libs!r})
packages = {pkgs!r}

os.makedirs(remote_libs, exist_ok=True)

cmd = [
    sys.executable,
    "-m",
    "pip",
    "install",
    *packages,
    "--target",
    remote_libs,
    "--disable-pip-version-check",
    "--no-input",
]

if {upgrade!r}:
    cmd.append("--upgrade")
if {editable!r}:
    cmd.append("--editable")
if {quiet!r}:
    cmd.append("--quiet")


proc = subprocess.run(
    cmd,
    text=True,
    capture_output=True,
)

if proc.stdout:
    print(proc.stdout, end="" if proc.stdout.endswith("\\n") else "\\n")
if proc.returncode != 0:
    raise RuntimeError(
        "Remote pip install failed\\n"
        f"command: {{shlex.join(cmd)}}\\n"
        f"stdout:\\n{{proc.stdout}}\\n"
        f"stderr:\\n{{proc.stderr}}"
    )

list_cmd = [
    sys.executable,
    "-m",
    "pip",
    "list",
    "--format=json",
    "--path",
    remote_libs,
    "--disable-pip-version-check",
]

listed = subprocess.run(
    list_cmd,
    text=True,
    capture_output=True,
    check=True,
)

print("__CALL_RESULT__" + listed.stdout, flush=True)
"""

        r = (
            self.context
            .command(language=Language.PYTHON, command=bootstrap)
            .start()
            .result()
        )

        LOGGER.info(
            "Installed packages '%s' to remote libs path '%s'",
            pkgs,
            remote_libs
        )

        return r


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

            raise CommandExecutionError(
                message=str(response)
            )


def _get_local_distribution_compatible_spec(
    module_name: str,
    raise_error: bool
) -> str:
    """
    Resolve a local import/module name to its installed distribution and
    return a compatible-release spec: '<dist>~=major.minor'.

    Examples:
        pydantic 2.11.1 -> 'pydantic~=2.11'
        pandas 2.2.3    -> 'pandas~=2.2'
    """
    import importlib.metadata as im

    root_module = module_name.split(".", 1)[0]

    # Build import-name -> distributions mapping
    pkg_map = im.packages_distributions()
    dist_names = pkg_map.get(root_module)

    if not dist_names:
        # fallback: assume import name == distribution name
        dist_names = [root_module]

    last_err: Exception | None = None

    for dist_name in dist_names:
        try:
            version = im.version(dist_name)
            parts = str(version).split(".")
            major = parts[0] if len(parts) > 0 else "0"
            minor = parts[1] if len(parts) > 1 else "0"
            return f"{dist_name}~={major}.{minor}"
        except Exception as e:
            last_err = e

    if not raise_error:
        return dist_names[0]

    if last_err is not None:
        raise RuntimeError(
            f"Cannot resolve installed distribution/version for module '{module_name}'"
        ) from last_err

    raise RuntimeError(
        f"Cannot resolve installed distribution/version for module '{module_name}'"
    )


def _get_tree_mtime(root: Path) -> float:
    root = root.resolve()

    if root.is_file():
        return root.stat().st_mtime

    latest = root.stat().st_mtime
    for p in root.rglob("*"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except FileNotFoundError:
            # file vanished during scan
            continue
    return latest
