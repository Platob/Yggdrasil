import dataclasses
import io
import os
import platform
import posixpath
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Union, List, BinaryIO

from ...libs import require_pyspark
from ...libs.databrickslib import require_databricks_sdk, databricks_sdk, databricks
from ...requests import MSALAuth

if databricks_sdk is not None:
    from databricks.sdk.errors import ResourceDoesNotExist
    from databricks.sdk.service.workspace import ImportFormat, ObjectType, ExportFormat

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None

try:
    from delta.tables import DeltaTable
except ImportError:
    DeltaTable = None


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
            base = f"/Workspace/Shared/.ygg/tmp/{self.current_user.user_name}"

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
                self.upload_content_file,
                content=content,
                target_path=target_path,
                makedirs=makedirs,
                overwrite=overwrite,
                only_if_size_diff=only_if_size_diff,
                parallel_pool=None,
            )

        sdk = self.sdk()

        try:
            if target_path.startswith("/Volumes"):
                # Ensure parent directory exists in the volume
                parent_dir = os.path.dirname(target_path)
                if parent_dir and parent_dir != "/":
                    sdk.files.create_directory(parent_dir)

                sdk.files.upload(
                    file_path=target_path,
                    contents=content,
                    overwrite=overwrite,
                )
            else:
                sdk.workspace.upload(
                    path=target_path,
                    format=ImportFormat.AUTO,
                    content=content,
                    overwrite=overwrite,
                )
        except ResourceDoesNotExist:
            if not makedirs:
                raise

            parent = os.path.dirname(target_path)
            if parent:
                sdk.workspace.mkdirs(parent)

            sdk.workspace.upload(
                path=target_path,
                format=ImportFormat.AUTO,
                content=content,
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
        ignore_hidden: bool = True,
        only_if_size_diff: bool = False,
        exclude_dirs: Optional[List[str]] = None,
        parallel_pool: Optional[Union[ThreadPoolExecutor, int]] = None,
    ):
        """
        Recursively upload a local folder into Databricks Workspace Files.

        - Traverses subdirectories recursively.
        - For each remote directory, can call workspace.list() once (when only_if_size_diff=True)
          to build {filename -> size} and skip unchanged files.
        - Can upload files in parallel using a ThreadPoolExecutor.

        parallel_pool:
          - None: run uploads synchronously
          - ThreadPoolExecutor: reuse caller's pool
          - int: create a new ThreadPoolExecutor(max_workers=<value>)
        """
        sdk = self.sdk()
        local_dir = os.path.abspath(local_dir)
        exclude_dirs_set = set(exclude_dirs or [])

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
            if ensure_dir:
                sdk.workspace.mkdirs(remote_root)

            # List local entries
            try:
                entries = list(os.scandir(local_root))
            except FileNotFoundError:
                return

            files = []
            dirs = []

            for entry in entries:
                name = entry.name

                # Skip hidden if requested
                if ignore_hidden and name.startswith("."):
                    continue

                if entry.is_dir():
                    if name in exclude_dirs_set:
                        continue
                    dirs.append(entry)
                else:
                    files.append(entry)

            # ---- bulk fetch remote file info for this directory ----
            remote_sizes = {}
            if only_if_size_diff:
                try:
                    for obj in sdk.workspace.list(remote_root):
                        # we only care about files, not directories
                        if getattr(obj, "object_type", None) == ObjectType.DIRECTORY:
                            continue

                        name = posixpath.basename(obj.path)
                        size = getattr(obj, "size", None)
                        if size is not None:
                            remote_sizes[name] = size
                except ResourceDoesNotExist:
                    remote_sizes = {}

            # ---- upload files in this directory ----
            for entry in files:
                local_path = entry.path
                remote_path = posixpath.join(remote_root, entry.name)
                remote_size = None

                if only_if_size_diff:
                    local_size = os.path.getsize(local_path)
                    remote_size = remote_sizes.get(entry.name)

                    if remote_size is not None and remote_size == local_size:
                        # same size remotely -> skip
                        continue

                # If we don't know remote_size at dir-level, let upload_local_file
                # optionally do its own get_status() for big files
                file_only_if_size_diff = only_if_size_diff and (remote_size is None)

                fut = self.upload_local_file(
                    local_path=local_path,
                    target_path=remote_path,
                    makedirs=False,
                    overwrite=True,
                    only_if_size_diff=file_only_if_size_diff,
                    parallel_pool=pool,
                )

                if pool is not None:
                    futures.append(fut)

            # ---- recurse into subdirectories ----
            for entry in dirs:
                child_local = entry.path
                child_remote = posixpath.join(remote_root, entry.name)
                _upload_dir(child_local, child_remote, ensure_dir=makedirs)

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

    @property
    def current_user(self):
        return self.workspace.current_user
