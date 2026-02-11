"""Remote execution helpers for Databricks command contexts."""

import base64
import dataclasses as dc
import io
import logging
import os
import posixpath
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from types import ModuleType
from typing import TYPE_CHECKING, Optional, Any, Callable, List, Dict, Union, Iterable, Tuple

from databricks.sdk.service.compute import Language, ResultType, CommandStatusResponse

from .command_execution import CommandExecution
from .exceptions import ClientTerminatedSession
from ...pyutils.exceptions import raise_parsed_traceback
from ...pyutils.expiring_dict import ExpiringDict
from ...pyutils.modules import resolve_local_lib_path
from ...pyutils.waiting_config import WaitingConfigArg

if TYPE_CHECKING:
    from .cluster import Cluster


__all__ = [
    "ExecutionContext"
]

LOGGER = logging.getLogger(__name__)
UPLOADED_PACKAGE_ROOTS: Dict[str, ExpiringDict] = {}
BytesLike = Union[bytes, bytearray, memoryview]

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
    """Metadata describing the remote cluster execution environment."""
    site_packages_path: Optional[str] = dc.field(default=None)
    os_env: Dict[str, str] = dc.field(default_factory=dict)
    version_info: Tuple[int, int, int] = dc.field(default=(0, 0, 0))
    temp_path: str = ""

    def os_env_diff(
        self,
        current: Optional[Dict] = None
    ):
        """Return environment variables present locally but missing remotely."""
        if current is None:
            current = os.environ

        return {
            k: v
            for k, v in current.items()
            if k not in self.os_env.keys()
        }


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

    language: Optional[Language] = dc.field(default=None, repr=False, compare=False, hash=False)

    _was_connected: Optional[bool] = dc.field(default=None, repr=False, compare=False, hash=False)
    _remote_metadata: Optional[RemoteMetadata] = dc.field(default=None, repr=False, compare=False, hash=False)
    _uploaded_package_roots: Optional[ExpiringDict] = dc.field(default_factory=ExpiringDict, repr=False, compare=False, hash=False)
    _lock: threading.RLock = dc.field(default_factory=threading.RLock, init=False, repr=False, compare=False, hash=False)

    # --- Pickle / cloudpickle support (don’t serialize locks or cached remote metadata) ---
    def __getstate__(self):
        """Serialize context state, excluding locks and remote metadata."""
        state = self.__dict__.copy()

        # name-mangled field for _lock in instance dict:
        state.pop("_lock", None)

        return state

    def __setstate__(self, state):
        """Restore context state, rehydrating locks if needed."""
        state["_lock"] = state.get("_lock", threading.RLock())

        self.__dict__.update(state)

    def __enter__(self) -> "ExecutionContext":
        """Enter a context manager, opening a remote execution context."""
        self.cluster.__enter__()
        self._was_connected = self.context_id is not None
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close the remote context if created."""
        if not self._was_connected:
            self.close(wait=False)
        self.cluster.__exit__(exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def __repr__(self):
        return "%s(url=%s)" % (
            self.__class__.__name__,
            self.url()
        )

    def __str__(self):
        return self.url()

    def url(self) -> str:
        return "%s/context/%s" % (
            self.cluster.url(),
            self.context_id or "unknown"
        )

    @property
    def workspace(self):
        return self.cluster.workspace

    @property
    def cluster_id(self):
        return self.cluster.cluster_id

    @property
    def remote_metadata(self) -> RemoteMetadata:
        """Fetch and cache remote environment metadata for the cluster."""
        # fast path (no lock)
        rm = self._remote_metadata
        if rm is not None:
            return rm

        # slow path guarded
        with self._lock:
            # double-check after acquiring lock
            if self._remote_metadata is None:
                cmd = r"""import glob, json, os, tempfile
from yggdrasil.pyutils.python_env import PythonEnv

current_env = PythonEnv.get_current()
meta = {}

# temp dir (explicit + stable for downstream code)
tmp_dir = tempfile.mkdtemp(prefix="tmp_")
meta["temp_path"] = tmp_dir
os.environ["TMPDIR"] = tmp_dir  # many libs respect this

# find site-packages
for path in glob.glob('/local_**/.ephemeral_nfs/cluster_libraries/python/lib/python*/site-*', recursive=False):
    if path.endswith('site-packages'):
        meta["site_packages_path"] = path
        break

# env vars snapshot
os_env = meta["os_env"] = {}
for k, v in os.environ.items():
    os_env[k] = v

meta["version_info"] = current_env.version_info

print(json.dumps(meta))"""

                content = self.command(
                    command=cmd,
                    language=Language.PYTHON,
                ).start().wait().result(unpickle=True)

                self._remote_metadata = RemoteMetadata(**content)

            return self._remote_metadata

    # ------------ internal helpers ------------
    def workspace_client(self):
        """Return the Databricks SDK client for command execution.

        Returns:
            The underlying WorkspaceClient instance.
        """
        return self.cluster.workspace.sdk()

    def shared_cache_path(
        self,
        suffix: str
    ):
        assert suffix, "Missing suffix arg"

        return self.cluster.shared_cache_path(
            suffix="/context/%s" % suffix.lstrip("/")
        )

    def create(
        self,
        language: "Language",
        wait: Optional[WaitingConfigArg] = True,
    ) -> "ExecutionContext":
        """Create a command execution context, retrying if needed.

        Args:
            language: The Databricks command language to use.
            wait: Waiting config to update

        Returns:
            The created command execution context response.
        """
        LOGGER.debug(
            "Creating Databricks command execution context for %s",
            self.cluster
        )

        client = self.workspace_client().command_execution

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

        LOGGER.info(
            "Created %s",
            self
        )

        self.context_id = created.id

        return self

    def connect(
        self,
        language: Optional[Language] = None,
        wait: Optional[WaitingConfigArg] = True,
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
            wait=wait
        )

    def close(self, wait: bool = True) -> None:
        """Destroy the remote command execution context if it exists.

        Returns:
            None.
        """
        if not self.context_id:
            return

        client = self.workspace_client()

        try:
            if wait:
                client.command_execution.destroy(
                    cluster_id=self.cluster.cluster_id,
                    context_id=self.context_id,
                )
            else:
                Thread(
                    target=client.command_execution.destroy,
                    kwargs={
                        "cluster_id": self.cluster.cluster_id,
                        "context_id": self.context_id,
                    }
                ).start()
        except BaseException:
            # non-fatal: context cleanup best-effort
            pass
        finally:
            self.context_id = None

    # ------------ public API ------------
    def command(
        self,
        context: Optional["ExecutionContext"] = None,
        func: Optional[Callable] = None,
        command_id: Optional[str] = None,
        command: Optional[str] = None,
        language: Optional[Language] = None,
        environ: Optional[Dict[str, str]] = None,
    ):
        context = self if context is None else context

        return CommandExecution(
            context=context,
            command_id=command_id,
            language=language,
            command=command,
        ).create(
            context=context,
            language=language,
            command=command,
            func=func,
            environ=environ
        )

    def decorate(
        self,
        func: Optional[Callable] = None,
        command: Optional[str] = None,
        language: Optional[Language] = None,
        command_id: Optional[str] = None,
        environ: Optional[Union[Iterable[str], Dict[str, str]]] = None,
    ) -> Callable:
        language = Language.PYTHON if language is None else language

        def decorator(
            f: Callable,
            c: ExecutionContext = self,
            cmd: Optional[str] = command,
            l: Optional[Language] = language,
            cid: Optional[str] = command_id,
            env: Optional[Union[Iterable[str], Dict[str, str]]] = environ,
        ):
            if c.is_in_databricks_environment():
                return func

            return c.command(
                context=c,
                func=f,
                command_id=cid,
                command=cmd,
                language=l,
                environ=env
            )

        if func is not None and callable(func):
            return decorator(f=func)
        return decorator

    def is_in_databricks_environment(self):
        """Return True when running on a Databricks runtime."""
        return self.cluster.is_in_databricks_environment()

    # ------------------------------------------------------------------
    # generic local → remote uploader, via remote python
    # ------------------------------------------------------------------
    def upload_local_path(
        self,
        paths: Union[Iterable[Tuple[LocalSpec, str]], Dict[LocalSpec, str]],
        byte_limit: int = 64 * 1024,
        wait: WaitingConfigArg | None = None
    ) -> None:
        """
        One-shot uploader. Sends exactly ONE remote command.

        paths: dict[local_spec -> remote_target]

        local_spec can be:
          - str | PathLike: local file or directory
          - bytes/bytearray/memoryview: raw content (remote_target must be a file path)
          - BytesSource(name, data): raw content with a name
          - (name, bytes-like): raw content with a name

        remote_target:
          - if local_spec is file: full remote file path
          - if local_spec is dir:  remote directory root
          - if local_spec is bytes: full remote file path
        """
        if isinstance(paths, dict):
            paths = paths.items()

        def _to_bytes(x: BytesLike) -> bytes:
            if isinstance(x, bytes):
                return x
            if isinstance(x, bytearray):
                return bytes(x)
            if isinstance(x, memoryview):
                return x.tobytes()
            elif isinstance(x, io.BytesIO):
                return x.getvalue()
            raise TypeError(f"Unsupported bytes-like: {type(x)!r}")

        # normalize + validate + build a unified "work list"
        work: list[dict[str, Any]] = []
        for local_spec, remote in paths:
            if not isinstance(remote, str) or not remote:
                raise TypeError("remote_target must be a non-empty string")

            remote_posix = remote.replace("\\", "/")

            # --- bytes payloads ---
            if isinstance(local_spec, BytesSource):
                work.append({
                    "kind": "bytes",
                    "name": local_spec.name,
                    "data": local_spec.data,
                    "remote": remote_posix,
                })
                continue

            if isinstance(local_spec, tuple) and len(local_spec) == 2 and isinstance(local_spec[0], str):
                name, data = local_spec
                work.append({
                    "kind": "bytes",
                    "name": name,
                    "data": _to_bytes(data),
                    "remote": remote_posix,
                })
                continue

            if isinstance(local_spec, (bytes, bytearray, memoryview, io.BytesIO)):
                work.append({
                    "kind": "bytes",
                    "name": "blob",
                    "data": _to_bytes(local_spec),
                    "remote": remote_posix,
                })
                continue

            # --- filesystem payloads ---
            if isinstance(local_spec, os.PathLike):
                local_spec = os.fspath(local_spec)

            if isinstance(local_spec, str):
                local_abs = os.path.abspath(local_spec)
                if not os.path.exists(local_abs):
                    raise FileNotFoundError(f"Local path not found: {local_spec}")

                if os.path.isfile(local_abs):
                    work.append({
                        "kind": "file",
                        "local": local_abs,
                        "remote": remote_posix,
                    })
                else:
                    work.append({
                        "kind": "dir",
                        "local": local_abs,
                        "remote_root": remote_posix.rstrip("/"),
                        "top": os.path.basename(local_abs.rstrip(os.sep)) or "dir",
                    })
                continue

            raise TypeError(f"Unsupported local_spec type: {type(local_spec)!r}")

        # build one zip containing all content
        manifest: list[dict[str, Any]] = []
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for idx, item in enumerate(work):
                kind = item["kind"]

                if kind == "bytes":
                    zip_name = f"BYTES/{idx}"
                    zf.writestr(zip_name, item["data"])
                    manifest.append({
                        "kind": "bytes",
                        "zip": zip_name,
                        "remote": item["remote"],
                    })

                elif kind == "file":
                    zip_name = f"FILE/{idx}"
                    zf.write(item["local"], arcname=zip_name)
                    manifest.append({
                        "kind": "file",
                        "zip": zip_name,
                        "remote": item["remote"],
                    })

                elif kind == "dir":
                    local_root = item["local"]
                    top = item["top"]
                    prefix = f"DIR/{idx}/{top}"

                    for root, dirs, files in os.walk(local_root):
                        dirs[:] = [d for d in dirs if d != "__pycache__"]

                        rel_root = os.path.relpath(root, local_root)
                        rel_root = "" if rel_root == "." else rel_root

                        for name in files:
                            if name.endswith((".pyc", ".pyo")):
                                continue
                            full = os.path.join(root, name)
                            rel_path = os.path.join(rel_root, name) if rel_root else name
                            zip_name = f"{prefix}/{rel_path}".replace("\\", "/")
                            zf.write(full, arcname=zip_name)

                    manifest.append({
                        "kind": "dir",
                        "zip_prefix": f"{prefix}/",
                        "remote_root": item["remote_root"],
                    })

                else:
                    raise ValueError(f"Unknown kind in work list: {kind}")

        raw = buf.getvalue()

        # optional zlib on top of zip
        algo = "none"
        payload = raw
        if len(raw) > byte_limit:
            import zlib
            compressed = zlib.compress(raw, level=9)
            if len(compressed) < int(len(raw) * 0.95):
                algo = "zlib"
                payload = compressed

        packed = b"ALG:" + algo.encode("ascii") + b"\n" + payload
        data_b64 = base64.b64encode(packed).decode("ascii")

        cmd = f"""import base64, io, os, zipfile, zlib

packed_b64 = {data_b64!r}
manifest = {manifest!r}

packed = base64.b64decode(packed_b64)
nl = packed.find(b"\\n")
if nl == -1 or not packed.startswith(b"ALG:"):
    raise ValueError("Bad payload header")

algo = packed[4:nl].decode("ascii")
payload = packed[nl+1:]

if algo == "none":
    raw = payload
elif algo == "zlib":
    raw = zlib.decompress(payload)
else:
    raise ValueError(f"Unknown compression algo: {{algo}}")

buf = io.BytesIO(raw)
with zipfile.ZipFile(buf, "r") as zf:
    names = set(zf.namelist())

    for item in manifest:
        kind = item["kind"]

        if kind in ("file", "bytes"):
            zip_name = item["zip"]
            remote_file = item["remote"]
            if zip_name not in names:
                raise FileNotFoundError(f"Missing in zip: {{zip_name}}")

            parent = os.path.dirname(remote_file)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with zf.open(zip_name, "r") as src, open(remote_file, "wb") as dst:
                dst.write(src.read())

        elif kind == "dir":
            prefix = item["zip_prefix"]
            remote_root = item["remote_root"]
            os.makedirs(remote_root, exist_ok=True)

            for n in names:
                if not n.startswith(prefix):
                    continue
                rel = n[len(prefix):]
                if not rel or rel.endswith("/"):
                    continue

                target = os.path.join(remote_root, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(n, "r") as src, open(target, "wb") as dst:
                    dst.write(src.read())

        else:
            raise ValueError(f"Unknown manifest kind: {{kind}}")
"""
        self.command(command=cmd).start().wait(wait=wait)

    # ------------------------------------------------------------------
    # upload local lib into remote site-packages
    # ------------------------------------------------------------------
    def install_temporary_libraries(
        self,
        libraries: str | ModuleType | List[str | ModuleType],
    ) -> Union[str, ModuleType, List[str | ModuleType]]:
        """
        Upload a local Python lib/module into the remote cluster's
        site-packages.

        `local_lib` can be:
        - path to a folder  (e.g. "./ygg")
        - path to a file    (e.g. "./ygg/__init__.py")
        - module name       (e.g. "ygg")
        - module object     (e.g. import ygg; workspace.upload_local_lib(ygg))
        Args:
            libraries: Library path, name, module, or iterable of these.

        Returns:
            The resolved library or list of libraries uploaded.
        """
        if isinstance(libraries, (list, tuple, set)):
            return [
                self.install_temporary_libraries(l) for l in libraries
            ]

        resolved = resolve_local_lib_path(libraries)

        LOGGER.debug(
            "Installing temporary lib '%s' in %s",
            resolved,
            self
        )

        str_resolved = str(resolved)
        existing = self._uploaded_package_roots.get(str_resolved)

        if not existing:
            remote_site_packages_path = self.remote_metadata.site_packages_path

            if resolved.is_dir():
                # site-packages/<package_name>/
                remote_target = posixpath.join(remote_site_packages_path, resolved.name)
            else:
                # site-packages/<module_file>
                remote_target = posixpath.join(remote_site_packages_path, resolved.name)

            self.upload_local_path({
                str_resolved: remote_target
            })

            self._uploaded_package_roots[str_resolved] = remote_target

            LOGGER.info(
                "Installed temporary lib '%s' in %s",
                resolved,
                self
            )

        return libraries


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
