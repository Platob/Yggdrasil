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
from typing import Mapping, Optional, Union, List, BinaryIO, Iterator

from databricks.sdk.dbutils import FileInfo
from databricks.sdk.service.files import DirectoryEntry

from ...libs import require_pyspark
from ...libs.databrickslib import require_databricks_sdk, databricks_sdk, databricks
from ...requests import MSALAuth

if databricks_sdk is not None:
    from databricks.sdk.errors import ResourceDoesNotExist, NotFound
    from databricks.sdk.service.workspace import ImportFormat, ExportFormat, ObjectInfo
    from databricks.sdk.service import catalog as catalog_svc

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None

try:
    from delta.tables import DeltaTable
except ImportError:
    DeltaTable = None


__all__ = [
    "DBXAuthType",
    "DBXWorkspace",
    "DBXWorkspaceObject"
]


def _get_remote_size(sdk, target_path: str) -> Optional[int]:
    """
    Best-effort fetch remote file size for target_path across
    dbfs, Volumes, and Workspace. Returns None if not found.
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


class DBXAuthType(Enum):
    external_browser = "external-browser"


@dataclass
class DBXWorkspace:
    host: str = None
    token: str = None
    msal_auth: Optional[MSALAuth] = None
    auth_type: Optional[Union[str, DBXAuthType]] = None
    product: Optional[str] = None
    product_version: Optional[str] = None

    _sdk: "databricks.sdk.WorkspaceClient" = dataclasses.field(init=False, default=None)
    _was_connected: bool = dataclasses.field(init=False, default=False)

    @classmethod
    def find_in_env(
        cls,
        env: Mapping = None,
        prefix: Optional[str] = None
    ):
        env = env or os.environ
        prefix = prefix or "DATABRICKS_"
        msal_auth = MSALAuth.find_in_env(env=env, prefix="AZURE_")
        options = {
            k: env.get(prefix + k.upper())
            for k in (
                "token", "host", "auth_type",
                "product", "product_version"
            )
            if env.get(prefix + k.upper())
        }

        return cls(
            msal_auth=msal_auth,
            **options,
        )

    def __enter__(self):
        self._was_connected = self._sdk is not None
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._was_connected:
            self.close()

    @property
    def current_user(self):
        return self.sdk().current_user.me()

    def cache_volume_folder(self, suffix: Optional[str] = None):
        base = f"/Shared/.ygg/cache/{self.current_user.name}"

        if not suffix:
            return base

        if suffix.startswith("/"):
            suffix = suffix[1:]

        return base + "/" + suffix

    def temp_volume_folder(
        self,
        suffix: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        volume_name: Optional[str] = None,
    ):
        if catalog_name and schema_name and volume_name:
            base = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"
        else:
            base = f"dbfs:/FileStore/.ygg/tmp/{self.current_user.user_name}"

        if not suffix:
            return base

        if suffix.startswith("/"):
            suffix = suffix[1:]

        return base + "/" + suffix


    def sdk(self):
        self.connect()
        return self._sdk

    def connect(
        self,
        reset: bool = False
    ):
        if reset or self._sdk is None:
            auth_type = self.auth_type
            if isinstance(auth_type, DBXAuthType):
                auth_type = auth_type.value
            elif self.token is None and auth_type is None:
                if platform.system() == "Windows":
                    # default to external browser on Windows
                    auth_type = DBXAuthType.external_browser.value

            require_databricks_sdk()

            self._sdk = databricks_sdk.WorkspaceClient(
                host=self.host,
                token=self.token,
                auth_type=auth_type,
                product=self.product,
                product_version=self.product_version or "0.0.0"
            )

            if not self.host:
                self.host = self._sdk.config.host

            if not self.auth_type:
                self.auth_type = self._sdk.config.auth_type

        return self

    def close(self):
        if self._sdk:
            self._sdk = None
            self._was_connected = False

    def spark_session(self):
        require_pyspark(active_session=True)
        return SparkSession.getActiveSession()

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
                # for EXTERNAL you'd also pass storage_location=...
            )

        # 4) finally create the directory path itself
        sdk.files.create_directory(target_path)

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
            - "dbfs:/..." → DBFS via dbfs.put
            - "/Volumes/..." → Unity Catalog Volumes via files.upload
            - anything else → Workspace via workspace.upload

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

            return parallel_pool.submit(
                self.upload_content_file,
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

                if isinstance(data_bytes, bytes):
                    data_str = base64.b64encode(data_bytes).decode("utf-8")
                else:
                    data_str = data_bytes

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
        Upload a single local file into Databricks Workspace Files.

        If parallel_pool is provided, this schedules the upload on the pool
        and returns a Future. The actual worker call runs synchronously
        (parallel_pool=None inside the job) to avoid infinite resubmission.

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

        if only_if_size_diff and local_size > 4 * 1024 * 1024:
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
            parallel_pool=parallel_pool
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
        - For each remote directory, can call workspace.list() once (when only_if_size_diff=True
          or exclude_present=True) to build {filename -> size} and skip unchanged/present files.
        - Can upload files in parallel using a ThreadPoolExecutor.

        Args:
            local_dir: Local directory to upload from.
            target_dir: Workspace path to upload into.
            makedirs: Create remote directories as needed.
            only_if_size_diff: Skip upload if remote file exists with same size.
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
            # Directory doesn't exist -> treat as empty, continue
            existing_objs = []

        # --- setup pool semantics ---
        created_pool: Optional[ThreadPoolExecutor] = None
        pool: Optional[ThreadPoolExecutor]

        if isinstance(parallel_pool, int):
            created_pool = ThreadPoolExecutor(max_workers=parallel_pool)
            pool = created_pool
        elif isinstance(parallel_pool, ThreadPoolExecutor):
            pool = parallel_pool
        else:
            pool = None

        futures = []

        def _upload_dir(local_root: str, remote_root: str, ensure_dir: bool):
            # Ensure remote directory exists if requested
            if ensure_dir and not existing_objs:
                sdk.workspace.mkdirs(remote_root)

            # List local entries
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

                        if only_if_size_diff and found_same_remote.size and found_same_remote.size != local_stats.st_size:
                            pass
                        elif local_stats.st_mtime < found_same_remote_epoch:
                            pass
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
                _upload_dir(local_entry.path, posixpath.join(remote_root, local_entry.name), ensure_dir=makedirs)

        try:
            # start recursion at root
            _upload_dir(local_dir, target_dir, ensure_dir=makedirs)

            # If we used a pool, wait for all uploads to finish
            if pool is not None:
                for fut in as_completed(futures):
                    fut.result()  # will re-raise any exceptions
        finally:
            if created_pool is not None:
                created_pool.shutdown(wait=True)

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

        # ---------------------------
        # DBFS RECURSIVE WALK
        # ---------------------------
        if path.startswith("dbfs:/"):
            try:
                entries = list(sdk.dbfs.list(path, recursive=recursive))
            except ResourceDoesNotExist:
                return

            for info in entries:
                yield info

            return

        # ---------------------------
        # UNITY CATALOG VOLUMES
        # ---------------------------
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

        # ---------------------------
        # WORKSPACE FILES / NOTEBOOKS
        # ---------------------------
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

        Returned object is a BinaryIO context manager, so use:

            with self.open_path(path) as f:
                data = f.read()
        """
        sdk = self.sdk()

        # DBFS path
        if path.startswith("dbfs:/"):
            # strip the dbfs: scheme -> `/mnt/...`
            dbfs_path = path[len("dbfs:"):]
            # dbfs.download returns BinaryIO
            return sdk.dbfs.download(dbfs_path)

        # Workspace path
        fmt = workspace_format or ExportFormat.AUTO
        return sdk.workspace.download(path=path, format=fmt)

    def delete_path(
        self,
        target_path: str,
        recursive: bool = True,
        ignore_missing: bool = True,
    ):
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

    def sql(
        self
    ):
        from ..sql import DBXSQL

        return DBXSQL(workspace=self)


@dataclass
class DBXWorkspaceObject(ABC):
    workspace: DBXWorkspace

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
