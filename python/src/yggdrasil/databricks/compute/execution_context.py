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
from types import ModuleType
from typing import TYPE_CHECKING, Optional, Any, Callable, List, Dict, Union, Iterable, Tuple, Literal

from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.compute import Language, ResultType, CommandStatusResponse

from yggdrasil.dataclasses.waiting import WaitingConfigArg
from .command_execution import CommandExecution
from .exceptions import ClientTerminatedSession
from ...concurrent.threading import Job
from ...dataclasses.expiring import ExpiringDict
from yggdrasil.environ import PyEnv, UserInfo
from yggdrasil.environ.modules import resolve_local_lib_path
from yggdrasil.io.url import URL
from yggdrasil.pyutils.exceptions import raise_parsed_traceback

if TYPE_CHECKING:
    from .cluster import Cluster


__all__ = [
    "ExecutionContext"
]

LOGGER = logging.getLogger(__name__)
UPLOADED_PACKAGE_ROOTS: Dict[str, ExpiringDict] = {}
BytesLike = Union[bytes, bytearray, memoryview]
# Module-level constants
_CTX_RUNTIME_FIELDS = frozenset({
    "_lock",           # threading.RLock — never picklable
})

_CTX_RESET_FIELDS = frozenset({
    "_remote_metadata",  # temp_path is remote-process-local; zero on restore
                         # caller re-fetches lazily on first access
})


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
        state.setdefault("_remote_metadata", None)
        state.setdefault("_uploaded_package_roots", ExpiringDict())

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
    def workspace(self):
        return self.cluster.workspace

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

        context_path = f"~/.ygg/dbx-ctx/{self.context_key}"
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
                command=command,
                language="shell",
                include_libs=True
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
    def workspace_client(self):
        """Return the Databricks SDK client for command execution.

        Returns:
            The underlying WorkspaceClient instance.
        """
        return self.cluster.workspace.sdk()

    def create(
        self,
        language: "Language",
        context_key: Optional[str] = None,
        *,
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

        instance = ExecutionContext(
            cluster=self.cluster,
            context_id=created.id,
            context_key=context_key or self.context_key or os.urandom(8).hex(),
            language=language
        )

        return instance

    def connect(
        self,
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

        client = self.workspace_client()

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
        command: Optional[str] = None,
        language: Optional[Language | Literal["python", "r", "sql", "scala", "shell"]] = None,
        *,
        context: Optional["ExecutionContext"] = None,
        func: Optional[Callable] = None,
        command_id: Optional[str] = None,
        environ: Optional[Dict[str, str]] = None,
        include_libs: bool = False
    ) -> "CommandExecution":
        context = self if context is None else context
        
        if command:
            if language == "shell":
                language = Language.PYTHON

                command = f"""
import subprocess, sys, shlex, pathlib

cmd = shlex.split({str(command)!r})
cmd = [str(pathlib.Path(arg).expanduser()) if arg.startswith("~/") else arg for arg in cmd]

p = subprocess.run(cmd, text=True, capture_output=True)

print(p.stdout)

if p.returncode != 0:
    raise RuntimeError(
        f"Command {{cmd}} failed with exit code {{p.returncode}}:\\n"
        f"stderr: {{p.stderr.strip()}}"
    )
"""
            if include_libs:
                command = self.syspath_lines() + "\n" + command
        else:
            if isinstance(language, str):
                language = Language[language]

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
            environ=environ,
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
                environ=env,
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

def _ep(p):
    \"\"\"Expand ~ and normalise path on the remote host.\"\"\"
    return os.path.expanduser(p)

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
            remote_file = _ep(item["remote"])          # ← expanded
            if zip_name not in names:
                raise FileNotFoundError(f"Missing in zip: {{zip_name}}")

            parent = os.path.dirname(remote_file)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with zf.open(zip_name, "r") as src, open(remote_file, "wb") as dst:
                dst.write(src.read())

        elif kind == "dir":
            prefix = item["zip_prefix"]
            remote_root = _ep(item["remote_root"])     # ← expanded

            os.makedirs(remote_root, exist_ok=True)

            for n in names:
                if not n.startswith(prefix):
                    continue
                rel = n[len(prefix):]
                if not rel or rel.endswith("/"):
                    continue

                target = os.path.join(remote_root, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                print(target)
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
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        pip_install: bool = False
    ) -> Union[str, ModuleType, List[str | ModuleType]]:
        connected = self.connect()

        is_collection = isinstance(libraries, (list, tuple, set))
        items = list(libraries) if is_collection else [libraries]
        libs_path = connected.remote_metadata.libs_path

        if pip_install:
            items = [str(x) for x in items]

            self.command(
                command=f"""import subprocess, pathlib, shlex, sys
tgt = pathlib.Path({str(self.remote_metadata.libs_path)!r}).expanduser()
tgt.mkdir(parents=True, exist_ok=True)

items = {items!r}

def run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True)

# 1) global attempt
cmd_all = ["uv","pip","install", *items, "--update", "--target", str(tgt)]
p = run(cmd_all)

if p.returncode == 0:
    print("[ok] installed all")
    sys.exit(0)
else:
    print("[warn] bulk install failed, falling back to per-package installs")
    print(p.stderr.strip() or p.stdout.strip())
    
    # 2) per-item fallback (ignore errors)
    failed = []
    for it in items:
        p2 = run(["uv","pip","install", it, "--update", "--target", str(tgt)])
        if p2.returncode != 0:
            failed.append(it)
            msg = (p2.stderr.strip() or p2.stdout.strip())
            print(f"[fail] {{it}} -> {{msg}}")
    print("[done] fallback complete. failed:", failed)
    sys.exit(0)
"""
            ).start().wait(wait=wait, raise_error=raise_error)
        else:
            upload_map = {
                str(resolved): posixpath.join(libs_path, resolved.name)
                for lib in items
                if not connected._uploaded_package_roots.get(str(resolved := resolve_local_lib_path(lib)))
            }

            if upload_map:
                connected.upload_local_path(upload_map)
                connected._uploaded_package_roots.update(upload_map)
                for str_resolved, remote_target in upload_map.items():
                    LOGGER.info("Installed temporary lib '%s' → %s on %s", str_resolved, remote_target, connected)

        return libraries

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
            "%s==%s" % (name, meta["current"])
            for name, meta in diffs.items()
            if meta and meta["current"] and _valid_install_package(name)
        ]

        if diffs:
            self.install_temporary_libraries(
                libraries=diffs,
                pip_install=True,
                wait=wait,
                raise_error=raise_error
            )
            self._requirements = None

        return self

def _valid_install_package(name: str):
    for prefix in ("pyspark", "pywin32"):
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
