import base64
import dataclasses
import io
import logging
import os
import posixpath
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Iterator,
    List,
    Optional,
    Union, Set, )

from ...libs.databrickslib import require_databricks_sdk, databricks_sdk

if databricks_sdk is not None:
    from databricks.sdk import WorkspaceClient, CredentialsStrategy
    from databricks.sdk.errors import ResourceDoesNotExist, NotFound
    from databricks.sdk.service.workspace import ImportFormat, ExportFormat, ObjectInfo
    from databricks.sdk.service import catalog as catalog_svc
    from databricks.sdk.dbutils import FileInfo
    from databricks.sdk.service.files import DirectoryEntry


__all__ = [
    "DBXWorkspace",
    "Workspace",
    "WorkspaceService",
]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT_MARKERS: Set[str] = {
    ".git",
    "pyproject.toml",
    "setup.cfg",
    "requirements.txt",
    ".idea",
}


def infer_project_root(start: Optional[Path] = None) -> Path:
    """
    Infer the 'project root' from the current working directory by walking
    upwards until we hit something that looks like a project boundary
    (git repo, Python project, or IDE project).

    If none of the markers are found, returns the starting directory.
    """
    path = (start or Path.cwd()).resolve()

    for candidate in [path, *path.parents]:
        if any((candidate / marker).exists() for marker in PROJECT_ROOT_MARKERS):
            return candidate

    return path


def infer_project_name(start: Optional[Path] = None) -> str:
    """
    Infer a project name purely from the filesystem, based on the
    inferred project root's directory name.
    """
    return infer_project_root(start).name


def _get_remote_size(sdk, target_path: str) -> Optional[int]:
    """
    Best-effort fetch remote file size for target_path across
    DBFS, Volumes, and Workspace. Returns None if not found.
    """
    try:
        if target_path.startswith("dbfs:/"):
            st = sdk.dbfs.get_status(target_path)
            return getattr(st, "file_size", None)

        if target_path.startswith("/Volumes"):
            st = sdk.files.get_status(file_path=target_path)
            return getattr(st, "file_size", None)

        # Workspace path
        st = sdk.workspace.get_status(target_path)
        return getattr(st, "size", None)

    except ResourceDoesNotExist:
        return None


@dataclass
class Workspace:
    # Databricks / generic
    host: Optional[str] = None
    account_id: Optional[str] = None
    token: Optional[str] = dataclasses.field(default=None, repr=False)
    client_id: Optional[str] = dataclasses.field(default=None, repr=False)
    client_secret: Optional[str] = dataclasses.field(default=None, repr=False)
    token_audience: Optional[str] = dataclasses.field(default=None, repr=False)
    credentials_strategy: Optional["CredentialsStrategy"] = dataclasses.field(default=None, repr=False)

    # Azure
    azure_workspace_resource_id: Optional[str] = dataclasses.field(default=None, repr=False)
    azure_use_msi: Optional[bool] = dataclasses.field(default=None, repr=False)
    azure_client_secret: Optional[str] = dataclasses.field(default=None, repr=False)
    azure_client_id: Optional[str] = dataclasses.field(default=None, repr=False)
    azure_tenant_id: Optional[str] = dataclasses.field(default=None, repr=False)
    azure_environment: Optional[str] = dataclasses.field(default=None, repr=False)

    # GCP
    google_credentials: Optional[str] = dataclasses.field(default=None, repr=False)
    google_service_account: Optional[str] = dataclasses.field(default=None, repr=False)

    # Config profile
    profile: Optional[str] = None
    config_file: Optional[str] = dataclasses.field(default=None, repr=False)

    # HTTP / client behavior
    auth_type: Optional[str] = None
    http_timeout_seconds: Optional[int] = dataclasses.field(default=None, repr=False)
    retry_timeout_seconds: Optional[int] = dataclasses.field(default=None, repr=False)
    debug_truncate_bytes: Optional[int] = dataclasses.field(default=None, repr=False)
    debug_headers: Optional[bool] = dataclasses.field(default=None, repr=False)
    rate_limit: Optional[int] = dataclasses.field(default=None, repr=False)

    # Extras
    product: Optional[str] = None
    product_version: Optional[str] = None

    # Runtime cache (never serialized)
    _sdk: Any = dataclasses.field(init=False, default=None, repr=False, compare=False, hash=False)
    _was_connected: bool = dataclasses.field(init=False, default=False, repr=False, compare=False)
    _cached_token: Optional[str] = dataclasses.field(init=False, default=None, repr=False, compare=False)

    # -------------------------
    # Pickle support
    # -------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_sdk", None)

        was_connected = self._sdk is not None
        self.connect()

        state["_was_connected"] = was_connected
        state["_cached_token"] = self.current_token()

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._sdk = None

        if self.is_in_databricks_environment():
            if self.auth_type == "external-browser":
                self.auth_type = None

        if self._was_connected:
            self.connect(reset=True)

    def __enter__(self) -> "Workspace":
        self._was_connected = self._sdk is not None
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._was_connected:
            self.close()

    # -------------------------
    # Clone
    # -------------------------
    def clone(self) -> "Workspace":
        return Workspace().__setstate__(self.__getstate__())

    # -------------------------
    # SDK connection
    # -------------------------
    def connect(self, reset: bool = False) -> "Workspace":
        if reset:
            logger.info("Resetting cached WorkspaceClient for host %s", self.host)
            self._sdk = None

        if self._sdk is None:
            require_databricks_sdk()
            logger.info("Connecting to Databricks workspace host=%s", self.host)

            # Build Config from config_dict if available, else from fields.
            kwargs = {
                "host": self.host,
                "account_id": self.account_id,
                "token": self.token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "token_audience": self.token_audience,
                "credentials_strategy": self.credentials_strategy,
                "azure_workspace_resource_id": self.azure_workspace_resource_id,
                "azure_use_msi": self.azure_use_msi,
                "azure_client_secret": self.azure_client_secret,
                "azure_client_id": self.azure_client_id,
                "azure_tenant_id": self.azure_tenant_id,
                "azure_environment": self.azure_environment,
                "google_credentials": self.google_credentials,
                "google_service_account": self.google_service_account,
                "profile": self.profile,
                "config_file": self.config_file,
                "auth_type": self.auth_type,
                "http_timeout_seconds": self.http_timeout_seconds,
                "retry_timeout_seconds": self.retry_timeout_seconds,
                "debug_truncate_bytes": self.debug_truncate_bytes,
                "debug_headers": self.debug_headers,
                "rate_limit": self.rate_limit,
                "product": self.product,
                "product_version": self.product_version,
            }

            build_kwargs = {k: v for k, v in kwargs.items() if v is not None}

            try:
                self._sdk = WorkspaceClient(**build_kwargs)
            except ValueError as e:
                if self.auth_type is None:
                    if self.is_in_databricks_environment() and self._cached_token:
                        build_kwargs["token"] = self._cached_token
                    elif "cannot configure default credentials" in str(e):
                        build_kwargs["auth_type"] = "external-browser"
                    else:
                        raise e

                    self._sdk = WorkspaceClient(**build_kwargs)
                else:
                    raise e

            # backfill resolved config values
            for key in list(kwargs.keys()):
                if getattr(self, key, None) is None:
                    v = getattr(self._sdk.config, key, None)
                    if v is not None:
                        setattr(self, key, v)

        return self

    # ------------------------------------------------------------------ #
    # Context manager + lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """
        Drop the cached WorkspaceClient (no actual close needed, but this
        avoids reusing stale config).
        """
        self._sdk = None
        self._was_connected = False

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def current_user(self):
        return self.sdk().current_user.me()

    def current_token(self) -> str:
        sdk = self.sdk()
        conf = sdk.config
        token = conf._credentials_strategy(conf)()["Authorization"].replace("Bearer ", "")

        return token

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def shared_cache_path(
        suffix: Optional[str] = None
    ) -> str:
        """
        Shared cache base under Volumes for the current user.
        """
        base = "/Workspace/Shared/.ygg/cache"

        if not suffix:
            return base

        suffix = suffix.lstrip("/")
        return f"{base}/{suffix}"

    def temp_volume_folder(
        self,
        suffix: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> str:
        """
        Temporary folder either under a UC Volume or dbfs:/FileStore/.ygg/tmp/<user>.
        """
        if catalog_name and schema_name and volume_name:
            base = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"
        else:
            base = f"dbfs:/FileStore/.ygg/tmp/{self.current_user.user_name}"

        if not suffix:
            return base

        suffix = suffix.lstrip("/")
        return f"{base}/{suffix}"

    # ------------------------------------------------------------------ #
    # SDK access / connection
    # ------------------------------------------------------------------ #

    def sdk(self) -> "WorkspaceClient":
        return self.connect()._sdk

    # ------------------------------------------------------------------ #
    # UC volume + directory management
    # ------------------------------------------------------------------ #

    def ensure_uc_volume_and_dir(
        self,
        target_path: str,
    ) -> None:
        """
        Ensure catalog, schema, volume exist for a UC volume path
        like /Volumes/<catalog>/<schema>/<volume>/...,
        then create the directory.
        """
        sdk = self.sdk()
        parts = target_path.split("/")

        # basic sanity check
        if len(parts) < 5 or parts[1] != "Volumes":
            raise ValueError(
                f"Unexpected UC volume path: {target_path!r}. "
                "Expected /Volumes/<catalog>/<schema>/<volume>/..."
            )

        # /Volumes/<catalog>/<schema>/<volume>/...
        _, _, catalog_name, schema_name, volume_name, *subpath = parts

        # 1) ensure catalog
        try:
            sdk.catalogs.get(name=catalog_name)
        except NotFound:
            sdk.catalogs.create(name=catalog_name)

        # 2) ensure schema
        schema_full_name = f"{catalog_name}.{schema_name}"
        try:
            sdk.schemas.get(full_name=schema_full_name)
        except NotFound:
            sdk.schemas.create(name=schema_name, catalog_name=catalog_name)

        # 3) ensure volume (managed volume is simplest)
        volume_full_name = f"{catalog_name}.{schema_name}.{volume_name}"
        try:
            sdk.volumes.read(name=volume_full_name)
        except NotFound:
            sdk.volumes.create(
                catalog_name=catalog_name,
                schema_name=schema_name,
                name=volume_name,
                volume_type=catalog_svc.VolumeType.MANAGED,
            )

        # 4) finally create the directory path itself
        sdk.files.create_directory(target_path)

    # ------------------------------------------------------------------ #
    # Upload helpers
    # ------------------------------------------------------------------ #
    def upload_file_content(
        self,
        content: Union[bytes, BinaryIO],
        target_path: str,
        makedirs: bool = True,
        overwrite: bool = True,
        only_if_size_diff: bool = False,
        parallel_pool: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Upload a single content blob into Databricks (Workspace / Volumes / DBFS).

        content:
            bytes or a binary file-like object.

        target_path:
            - "dbfs:/..."    → DBFS via dbfs.put
            - "/Volumes/..." → Unity Catalog Volumes via files.upload
            - anything else  → Workspace via workspace.upload

        If parallel_pool is provided, this schedules the upload on the pool
        and returns a Future. The underlying call is non-parallel (no nested pool).

        If only_if_size_diff=True, it will:
          - compute local content size (len(bytes))
          - fetch remote size (best-effort)
          - skip upload if sizes match.
        """
        # If we're doing this in a pool, normalize content to bytes *before*
        # submitting so we don't share a live file handle across threads.
        if parallel_pool is not None:
            if hasattr(content, "read"):
                data = content.read()
            else:
                data = content

            # use a cloned workspace so clients don't collide across threads
            return parallel_pool.submit(
                self.clone().upload_file_content,
                content=data,
                target_path=target_path,
                makedirs=makedirs,
                overwrite=overwrite,
                only_if_size_diff=only_if_size_diff,
                parallel_pool=None,
            )

        with self.connect() as connected:
            sdk = connected.sdk()

            # Normalize content to bytes once
            if hasattr(content, "read"):  # BinaryIO
                data = content.read()
            else:
                data = content

            if not isinstance(data, (bytes, bytearray)):
                raise TypeError(
                    f"content must be bytes or BinaryIO, got {type(content)!r}"
                )

            data_bytes = bytes(data)
            local_size = len(data_bytes)

            # Only-if-size-diff: check remote size and bail early if equal
            if only_if_size_diff:
                remote_size = _get_remote_size(sdk, target_path)
                if remote_size is not None and remote_size == local_size:
                    # Same size remotely -> skip upload
                    return None

            # Ensure parent directory if requested
            parent = os.path.dirname(target_path)

            if target_path.startswith("dbfs:/"):
                # --- DBFS path ---
                if makedirs and parent and parent != "dbfs:/":
                    sdk.dbfs.mkdirs(parent)

                data_str = base64.b64encode(data_bytes).decode("utf-8")
                sdk.dbfs.put(
                    path=target_path,
                    contents=data_str,
                    overwrite=overwrite,
                )

            elif target_path.startswith("/Volumes"):
                # --- Unity Catalog Volumes path ---
                if makedirs and parent and parent != "/":
                    try:
                        sdk.files.create_directory(parent)
                    except NotFound:
                        connected.ensure_uc_volume_and_dir(parent)

                sdk.files.upload(
                    file_path=target_path,
                    contents=io.BytesIO(data_bytes),
                    overwrite=overwrite,
                )

            else:
                # --- Workspace Files / Notebooks ---
                if makedirs and parent:
                    sdk.workspace.mkdirs(parent)

                sdk.workspace.upload(
                    path=target_path,
                    format=ImportFormat.RAW,
                    content=data_bytes,
                    overwrite=overwrite,
                )

    def upload_local_path(
        self,
        local_path: str,
        target_path: str,
        makedirs: bool = True,
        overwrite: bool = True,
        only_if_size_diff: bool = False,
        parallel_pool: Optional[ThreadPoolExecutor] = None,
    ):
        if os.path.isfile(local_path):
            return self.upload_local_file(
                local_path=local_path,
                target_path=target_path,
                makedirs=makedirs,
                overwrite=overwrite,
                only_if_size_diff=only_if_size_diff,
                parallel_pool=parallel_pool
            )
        else:
            return self.upload_local_folder(
                local_path=local_path,
                target_path=target_path,
                makedirs=makedirs,
                only_if_size_diff=only_if_size_diff,
                parallel_pool=parallel_pool
            )

    def upload_local_file(
        self,
        local_path: str,
        target_path: str,
        makedirs: bool = True,
        overwrite: bool = True,
        only_if_size_diff: bool = False,
        parallel_pool: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Upload a single local file into Databricks.

        If parallel_pool is provided, this schedules the upload on the pool
        and returns a Future.

        If only_if_size_diff=True, it will:
          - For large files (>4 MiB), check remote file status
          - Skip upload if remote size == local size
        """
        if parallel_pool is not None:
            # Submit a *non-parallel* variant into the pool
            return parallel_pool.submit(
                self.upload_local_file,
                local_path=local_path,
                target_path=target_path,
                makedirs=makedirs,
                overwrite=overwrite,
                only_if_size_diff=only_if_size_diff,
                parallel_pool=None,
            )

        sdk = self.sdk()

        local_size = os.path.getsize(local_path)
        large_threshold = 4 * 1024 * 1024  # 4 MiB

        if only_if_size_diff and local_size > large_threshold:
            try:
                info = sdk.workspace.get_status(path=target_path)
                remote_size = getattr(info, "size", None)

                if remote_size is not None and remote_size == local_size:
                    return
            except ResourceDoesNotExist:
                # Doesn't exist → upload below
                pass

        with open(local_path, "rb") as f:
            content = f.read()

        return self.upload_file_content(
            content=content,
            target_path=target_path,
            makedirs=makedirs,
            overwrite=overwrite,
            only_if_size_diff=False,
            parallel_pool=parallel_pool,
        )

    def upload_local_folder(
        self,
        local_path: str,
        target_path: str,
        makedirs: bool = True,
        only_if_size_diff: bool = True,
        exclude_dir_names: Optional[List[str]] = None,
        exclude_hidden: bool = True,
        parallel_pool: Optional[Union[ThreadPoolExecutor, int]] = None,
    ):
        """
        Recursively upload a local folder into Databricks Workspace Files.

        - Traverses subdirectories recursively.
        - Optionally skips files that match size/mtime of remote entries.
        - Can upload files in parallel using a ThreadPoolExecutor.

        Args:
            local_path: Local directory to upload from.
            target_path: Workspace path to upload into.
            makedirs: Create remote directories as needed.
            only_if_size_diff: Skip upload if remote file exists with same size and newer mtime.
            exclude_dir_names: Directory names to skip entirely.
            exclude_hidden: Skip dot-prefixed files/directories.
            parallel_pool: None | ThreadPoolExecutor | int (max_workers).
        """
        sdk = self.sdk()
        local_path = os.path.abspath(local_path)
        exclude_dirs_set = set(exclude_dir_names or [])

        try:
            existing_objs = list(sdk.workspace.list(target_path))
        except ResourceDoesNotExist:
            existing_objs = []

        # --- setup pool semantics ---
        created_pool: Optional[ThreadPoolExecutor] = None
        if isinstance(parallel_pool, int):
            created_pool = ThreadPoolExecutor(max_workers=parallel_pool)
            pool: Optional[ThreadPoolExecutor] = created_pool
        elif isinstance(parallel_pool, ThreadPoolExecutor):
            pool = parallel_pool
        else:
            pool = None

        futures = []

        def _upload_dir(local_root: str, remote_root: str, ensure_dir: bool):
            # Ensure remote directory exists if requested
            existing_remote_root_obj = [
                _ for _ in existing_objs
                if _.path.startswith(remote_root)
            ]

            if ensure_dir and not existing_remote_root_obj:
                sdk.workspace.mkdirs(remote_root)

            try:
                local_entries = list(os.scandir(local_root))
            except FileNotFoundError:
                return

            local_files = []
            local_dirs = []

            for local_entry in local_entries:
                # Skip hidden if requested
                if exclude_hidden and local_entry.name.startswith("."):
                    continue

                if local_entry.is_dir():
                    if local_entry.name in exclude_dirs_set:
                        continue
                    local_dirs.append(local_entry)
                elif existing_objs:
                    found_same_remote = None
                    for exiting_obj in existing_objs:
                        existing_obj_name = os.path.basename(exiting_obj.path)
                        if existing_obj_name == local_entry.name:
                            found_same_remote = exiting_obj
                            break

                    if found_same_remote:
                        found_same_remote_epoch = found_same_remote.modified_at / 1000
                        local_stats = local_entry.stat()

                        if (
                            only_if_size_diff
                            and found_same_remote.size
                            and found_same_remote.size != local_stats.st_size
                        ):
                            pass  # size diff -> upload
                        elif local_stats.st_mtime < found_same_remote_epoch:
                            # remote is newer -> skip
                            continue
                        else:
                            local_files.append(local_entry)
                    else:
                        local_files.append(local_entry)
                else:
                    local_files.append(local_entry)

            # ---- upload files in this directory ----
            for local_entry in local_files:
                remote_path = posixpath.join(remote_root, local_entry.name)

                entry_fut = self.upload_local_file(
                    local_path=local_entry.path,
                    target_path=remote_path,
                    makedirs=False,
                    overwrite=True,
                    only_if_size_diff=False,
                    parallel_pool=pool,
                )

                if pool is not None:
                    futures.append(entry_fut)

            # ---- recurse into subdirectories ----
            for local_entry in local_dirs:
                _upload_dir(
                    local_entry.path,
                    posixpath.join(remote_root, local_entry.name),
                    ensure_dir=makedirs,
                )

        try:
            _upload_dir(local_path, target_path, ensure_dir=makedirs)

            if pool is not None:
                for fut in as_completed(futures):
                    fut.result()
        finally:
            if created_pool is not None:
                created_pool.shutdown(wait=True)

    # ------------------------------------------------------------------ #
    # List / open / delete / SQL
    # ------------------------------------------------------------------ #

    def list_path(
        self,
        path: str,
        recursive: bool = False,
    ) -> Iterator[Union[FileInfo, ObjectInfo, DirectoryEntry]]:
        """
        List contents of a path across Databricks namespaces:

          - 'dbfs:/...'      -> DBFS (sdk.dbfs.list)
          - '/Volumes/...'   -> Unity Catalog Volumes (sdk.files.list_directory_contents)
          - other paths      -> Workspace paths (sdk.workspace.list)

        If recursive=True, yield all nested files/directories.
        """
        sdk = self.sdk()

        # DBFS
        if path.startswith("dbfs:/"):
            try:
                entries = list(sdk.dbfs.list(path, recursive=recursive))
            except ResourceDoesNotExist:
                return
            for info in entries:
                yield info
            return

        # UC Volumes
        if path.startswith("/Volumes"):
            try:
                entries = list(sdk.files.list_directory_contents(path))
            except ResourceDoesNotExist:
                return

            for entry in entries:
                yield entry

                if recursive and entry.is_directory:
                    child_path = posixpath.join(path, entry.path)
                    yield from self.list_path(child_path, recursive=True)
            return

        # Workspace files / notebooks
        try:
            entries = list(sdk.workspace.list(path, recursive=recursive))
        except ResourceDoesNotExist:
            return

        for obj in entries:
            yield obj

    def open_path(
        self,
        path: str,
        *,
        workspace_format: Optional[ExportFormat] = None,
    ) -> BinaryIO:
        """
        Open a remote path as BinaryIO.

        - If path starts with 'dbfs:/', it is treated as a DBFS path and
          opened for reading via DBFS download.
        - Otherwise it is treated as a Workspace file/notebook and returned
          via workspace.download(...).

        Returned object is a BinaryIO context manager.
        """
        sdk = self.sdk()

        # DBFS path
        if path.startswith("dbfs:/"):
            dbfs_path = path[len("dbfs:") :]
            return sdk.dbfs.download(dbfs_path)

        # Workspace path
        fmt = workspace_format or ExportFormat.AUTO
        return sdk.workspace.download(path=path, format=fmt)

    def delete_path(
        self,
        target_path: str,
        recursive: bool = True,
        ignore_missing: bool = True,
    ) -> None:
        """
        Delete a path in Databricks Workspace (file or directory).

        - If recursive=True and target_path is a directory, deletes entire tree.
        - If ignore_missing=True, missing paths won't raise.
        """
        sdk = self.sdk()

        try:
            sdk.workspace.delete(
                path=target_path,
                recursive=recursive,
            )
        except ResourceDoesNotExist:
            if ignore_missing:
                return
            raise

    @staticmethod
    def is_in_databricks_environment():
        return os.getenv("DATABRICKS_RUNTIME_VERSION") is not None

    def sql(self):
        from ..sql import SQLEngine

        return SQLEngine(workspace=self)
    
    def clusters(self):
        from ..compute.cluster import Cluster

        return Cluster(workspace=self)


# ---------------------------------------------------------------------------
# Workspace-bound base class
# ---------------------------------------------------------------------------

DBXWorkspace = Workspace


@dataclass
class WorkspaceService(ABC):
    workspace: Workspace = dataclasses.field(default_factory=Workspace)

    def __post_init__(self):
        if self.workspace is None:
            self.workspace = Workspace()

    def __enter__(self):
        self.workspace.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.workspace.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def is_in_databricks_environment(self):
        return self.workspace.is_in_databricks_environment()

    def connect(self):
        self.workspace = self.workspace.connect()
        return self

    def sdk(self):
        return self.workspace.sdk()

    @property
    def current_user(self):
        return self.workspace.current_user
