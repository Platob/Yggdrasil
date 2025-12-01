import dataclasses
import os
import platform
import posixpath
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Union


from ...libs import require_pyspark
from ...libs.databrickslib import require_databricks_sdk, databricks_sdk, databricks
from ...requests import MSALAuth

if databricks_sdk is not None:
    from databricks.sdk.errors import ResourceDoesNotExist
    from databricks.sdk.service.workspace import ImportFormat, ObjectType

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

    def sdk(self):
        self.connect()
        return self._sdk

    @require_databricks_sdk
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

    def upload_local_file(
        self,
        local_path: str,
        target_path: str,
        makedirs: bool = True,
        overwrite: bool = True,
        only_if_size_diff: bool = False,
    ):
        """
        Upload a single local file into Databricks Workspace Files.

        If only_if_size_diff=True, it will:
        - Check remote file status
        - Skip upload if remote size == local size
        """
        sdk = self.sdk()

        local_size = os.path.getsize(local_path)

        if only_if_size_diff and local_size > 4 * 1024 * 1024:
            try:
                info = sdk.workspace.get_status(path=target_path)
                # Some safety: size may not exist for all object types
                remote_size = getattr(info, "size", None)

                if remote_size is not None and remote_size == local_size:
                    # Same size → assume no change, skip upload
                    # You can plug your logger here instead of print
                    print(f"[SKIP] Same size: {local_path} -> {target_path}")
                    return
            except ResourceDoesNotExist:
                # Doesn't exist → we will upload it below
                pass

        with open(local_path, "rb") as f:
            content = f.read()

        try:
            sdk.workspace.upload(
                path=target_path,
                format=ImportFormat.AUTO,
                content=content,
                overwrite=overwrite
            )
        except ResourceDoesNotExist:
            if not makedirs:
                raise

            # create parent dirs then retry
            parent = os.path.dirname(target_path)
            if parent:
                sdk.workspace.mkdirs(parent)

            sdk.workspace.upload(
                path=target_path,
                format=ImportFormat.AUTO,
                content=content,
                overwrite=overwrite
            )

    def upload_local_folder(
        self,
        local_dir: str,
        target_dir: str,
        makedirs: bool = True,
        ignore_hidden: bool = True,
        only_if_size_diff: bool = False,
    ):
        """
        Recursively upload a local folder into Databricks Workspace Files.

        If only_if_size_diff=True:
        - For each remote directory, call workspace.list() once
        - Build {filename -> size}
        - Skip uploading files where local_size == remote_size
        """
        sdk = self.sdk()
        local_dir = os.path.abspath(local_dir)

        if makedirs:
            sdk.workspace.mkdirs(target_dir)

        for root, dirs, files in os.walk(local_dir):
            rel_root = os.path.relpath(root, local_dir)

            if rel_root == ".":
                remote_root = target_dir
            else:
                rel_root_norm = rel_root.replace(os.sep, "/")
                remote_root = posixpath.join(target_dir, rel_root_norm)

            if makedirs:
                sdk.workspace.mkdirs(remote_root)

            if ignore_hidden:
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                files = [f for f in files if not f.startswith(".")]

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
                    # remote_root doesn't exist yet -> everything is new
                    remote_sizes = {}

            # ---- upload loop ----
            for name in files:
                local_path = os.path.join(root, name)
                remote_path = posixpath.join(remote_root, name)
                remote_size = None

                if only_if_size_diff:
                    local_size = os.path.getsize(local_path)
                    remote_size = remote_sizes.get(name)

                    if remote_size is not None and remote_size == local_size:
                        continue

                # we already handled dirs + size logic here
                self.upload_local_file(
                    local_path=local_path,
                    target_path=remote_path,
                    makedirs=False,
                    only_if_size_diff=only_if_size_diff and remote_size is None
                )

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

    @staticmethod
    def temp_volume_folder(suffix: Optional[str] = None):
        base = "/Shared/.ygg/tmp/"

        if not suffix:
            return base

        if suffix.startswith("/"):
            suffix = suffix[1:]

        return base + suffix
