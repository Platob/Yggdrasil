import base64
import dataclasses
import io
import os
import platform
import posixpath
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    BinaryIO,
    Iterator,
    List,
    Optional,
    Union,
)

from databricks.sdk.dbutils import FileInfo
from databricks.sdk.service.files import DirectoryEntry

from ...libs import require_pyspark
from ...libs.databrickslib import require_databricks_sdk, databricks_sdk
from ...requests import MSALAuth

if databricks_sdk is not None:
    from databricks.sdk.errors import ResourceDoesNotExist, NotFound
    from databricks.sdk.service.workspace import ImportFormat, ExportFormat, ObjectInfo
    from databricks.sdk.service import catalog as catalog_svc

try:
    from pyspark.sql import SparkSession
except ImportError:  # pragma: no cover
    SparkSession = None

try:
    from delta.tables import DeltaTable
except ImportError:  # pragma: no cover
    DeltaTable = None


__all__ = [
    "DBXAuthType",
    "DBXWorkspace",
    "DBXWorkspaceObject",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Auth + Workspace
# ---------------------------------------------------------------------------


class DBXAuthType(Enum):
    external_browser = "external-browser"


@dataclass
class DBXWorkspace:
    """
    Thin wrapper around Databricks WorkspaceClient with helpers for:

    - connecting / disconnecting
    - temp + cache volume folders
    - uploading local files/folders
    - listing / opening / deleting paths across DBFS, UC Volumes, Workspace
    """

    # Full-ish mirror of WorkspaceClient config

    # raw Config object, if caller wants to pass it directly
    config: Any = None  # typically databricks.sdk.core.Config

    # Databricks / generic
    host: Optional[str] = None
    account_id: Optional[str] = None
    token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_audience: Optional[str] = None

    # Azure
    azure_workspace_resource_id: Optional[str] = None
    azure_use_msi: Optional[bool] = None
    azure_client_secret: Optional[str] = None
    azure_client_id: Optional[str] = None
    azure_tenant_id: Optional[str] = None
    azure_environment: Optional[str] = None

    # GCP
    google_credentials: Optional[str] = None
    google_service_account: Optional[str] = None

    # Config profile
    profile: Optional[str] = None
    config_file: Optional[str] = None

    # HTTP / client behavior
    auth_type: Optional[Union[str, DBXAuthType]] = None
    http_timeout_seconds: Optional[int] = None
    retry_timeout_seconds: Optional[int] = None
    debug_truncate_bytes: Optional[int] = None
    debug_headers: Optional[bool] = None
    rate_limit: Optional[int] = None

    # Extras
    msal_auth: Optional[MSALAuth] = None
    product: Optional[str] = None
    product_version: Optional[str] = None

    _sdk: "databricks_sdk.WorkspaceClient" = dataclasses.field(init=False, default=None)
    _was_connected: bool = dataclasses.field(init=False, default=False)

    # ------------------------------------------------------------------ #
    # Clone
    # ------------------------------------------------------------------ #

    def clone(self, *, with_client: bool = False) -> "DBXWorkspace":
        """
        Create a shallow clone of this workspace config.

        with_client=False:
            New DBXWorkspace with same config, no client yet.
        with_client=True:
            New DBXWorkspace with same config and a fresh WorkspaceClient.
        """
        clone = DBXWorkspace(
            config=self.config,
            host=self.host,
            account_id=self.account_id,
            token=self.token,
            client_id=self.client_id,
            client_secret=self.client_secret,
            token_audience=self.token_audience,
            azure_workspace_resource_id=self.azure_workspace_resource_id,
            azure_use_msi=self.azure_use_msi,
            azure_client_secret=self.azure_client_secret,
            azure_client_id=self.azure_client_id,
            azure_tenant_id=self.azure_tenant_id,
            azure_environment=self.azure_environment,
            google_credentials=self.google_credentials,
            google_service_account=self.google_service_account,
            profile=self.profile,
            config_file=self.config_file,
            auth_type=self.auth_type,
            http_timeout_seconds=self.http_timeout_seconds,
            retry_timeout_seconds=self.retry_timeout_seconds,
            debug_truncate_bytes=self.debug_truncate_bytes,
            debug_headers=self.debug_headers,
            rate_limit=self.rate_limit,
            msal_auth=self.msal_auth,
            product=self.product,
            product_version=self.product_version,
        )

        if with_client:
            clone.connect(reset=True, new_instance=False)

        return clone

    # ------------------------------------------------------------------ #
    # Context manager + lifecycle
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "DBXWorkspace":
        self._was_connected = self._sdk is not None
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._was_connected:
            self.close()

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

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #

    def cache_volume_folder(self, suffix: Optional[str] = None) -> str:
        """
        Shared cache base under Volumes for the current user.
        """
        base = f"/Shared/.ygg/cache/{self.current_user.name}"

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

    def sdk(
        self,
        *,
        new_instance: bool = False,
        reset: bool = False,
    ):
        """
        Return the underlying WorkspaceClient.

        Args:
            new_instance:
                If True, build a fresh DBXWorkspace (via connect(new_instance=True))
                and return *its* WorkspaceClient. The current instance is left
                untouched.
            reset:
                If True and new_instance is False, drop any cached client on this
                instance and create a new one in-place.
        """
        ws = self.connect(reset=reset, new_instance=new_instance)
        return ws._sdk

    def connect(
        self,
        reset: bool = False,
        new_instance: bool = False,
    ) -> "DBXWorkspace":
        """
        Ensure a WorkspaceClient is available.

        Args:
            reset:
                If True, always drop any existing client on *this* instance
                and re-create it in-place.
            new_instance:
                If True, build and return a completely new DBXWorkspace
                instance (with its own WorkspaceClient) based on the current
                config. The original instance is not modified.
        """
        if new_instance:
            clone = self.clone(with_client=False)
            clone.connect(reset=True, new_instance=False)
            return clone

        if reset:
            self._sdk = None

        if self._sdk is None:
            require_databricks_sdk()

            # Normalize auth_type once
            auth_type = self.auth_type
            if isinstance(auth_type, DBXAuthType):
                auth_type = auth_type.value
            elif self.token is None and auth_type is None:
                # default to external browser on Windows if nothing else is set
                if platform.system() == "Windows":
                    auth_type = DBXAuthType.external_browser.value

            # Prepare kwargs for WorkspaceClient, dropping None so SDK defaults apply
            kwargs = {
                "config": self.config,
                "host": self.host,
                "account_id": self.account_id,
                "token": self.token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "token_audience": self.token_audience,
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
                "auth_type": auth_type,
                "http_timeout_seconds": self.http_timeout_seconds,
                "retry_timeout_seconds": self.retry_timeout_seconds,
                "debug_truncate_bytes": self.debug_truncate_bytes,
                "debug_headers": self.debug_headers,
                "rate_limit": self.rate_limit,
                "product": self.product,
                "product_version": self.product_version or "0.0.0",
            }

            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            self._sdk = databricks_sdk.WorkspaceClient(**kwargs)

            # Fill in host/auth_type from resolved config if we didn't set them
            if not self.host:
                self.host = self._sdk.config.host

            if not self.auth_type:
                self.auth_type = self._sdk.config.auth_type

        return self

    # ------------------------------------------------------------------ #
    # Spark helpers
    # ------------------------------------------------------------------ #

    def spark_session(self):
        require_pyspark(active_session=True)
        return SparkSession.getActiveSession()

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

    def upload_content_file(
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
                self.clone(with_client=True).upload_content_file,
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
                    format=ImportFormat.AUTO,
                    content=data_bytes,
                    overwrite=overwrite,
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

        return self.upload_content_file(
            content=content,
            target_path=target_path,
            makedirs=makedirs,
            overwrite=overwrite,
            only_if_size_diff=False,
            parallel_pool=parallel_pool,
        )

    def upload_local_folder(
        self,
        local_dir: str,
        target_dir: str,
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
            local_dir: Local directory to upload from.
            target_dir: Workspace path to upload into.
            makedirs: Create remote directories as needed.
            only_if_size_diff: Skip upload if remote file exists with same size and newer mtime.
            exclude_dir_names: Directory names to skip entirely.
            exclude_hidden: Skip dot-prefixed files/directories.
            parallel_pool: None | ThreadPoolExecutor | int (max_workers).
        """
        sdk = self.sdk()
        local_dir = os.path.abspath(local_dir)
        exclude_dirs_set = set(exclude_dir_names or [])

        try:
            existing_objs = list(sdk.workspace.list(target_dir))
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
            if ensure_dir and not existing_objs:
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
                local_path = local_entry.path
                remote_path = posixpath.join(remote_root, local_entry.name)

                fut = self.upload_local_file(
                    local_path=local_path,
                    target_path=remote_path,
                    makedirs=False,
                    overwrite=True,
                    only_if_size_diff=False,
                    parallel_pool=pool,
                )

                if pool is not None:
                    futures.append(fut)

            # ---- recurse into subdirectories ----
            for local_entry in local_dirs:
                _upload_dir(
                    local_entry.path,
                    posixpath.join(remote_root, local_entry.name),
                    ensure_dir=makedirs,
                )

        try:
            _upload_dir(local_dir, target_dir, ensure_dir=makedirs)

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

    def sql(self):
        from ..sql import DBXSQL

        return DBXSQL(workspace=self)


# ---------------------------------------------------------------------------
# Workspace-bound base class
# ---------------------------------------------------------------------------


@dataclass
class DBXWorkspaceObject(ABC):
    workspace: DBXWorkspace = dataclasses.field(default_factory=DBXWorkspace)

    def __enter__(self):
        self.workspace.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.workspace.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def connect(self):
        self.workspace.connect()
        return self

    def sdk(self):
        return self.workspace.sdk()

    @property
    def current_user(self):
        return self.workspace.current_user
