import dataclasses
import logging
import os
import posixpath
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Iterator,
    Optional,
    Union, TYPE_CHECKING, List
)

if TYPE_CHECKING:
    from ..compute.cluster import Cluster

from .path import DatabricksPath, DatabricksPathKind
from ...version import __version__ as YGGDRASIL_VERSION
from ...libs.databrickslib import require_databricks_sdk, databricks_sdk

if databricks_sdk is not None:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import ResourceDoesNotExist
    from databricks.sdk.service.workspace import ExportFormat, ObjectInfo
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
def _get_env_product():
    v = os.getenv("DATABRICKS_PRODUCT")

    if not v:
        return "yggdrasil"
    return v.strip().lower()


def _get_env_product_version():
    v = os.getenv("DATABRICKS_PRODUCT_VERSION")

    if not v:
        return YGGDRASIL_VERSION
    return v.strip().lower()


def _get_env_product_tag():
    v = os.getenv("DATABRICKS_PRODUCT_TAG")

    if not v:
        return None
    return v.strip().lower()


@dataclass
class Workspace:
    # Databricks / generic
    host: Optional[str] = None
    account_id: Optional[str] = None
    token: Optional[str] = dataclasses.field(default=None, repr=False)
    client_id: Optional[str] = dataclasses.field(default=None, repr=False)
    client_secret: Optional[str] = dataclasses.field(default=None, repr=False)
    token_audience: Optional[str] = dataclasses.field(default=None, repr=False)

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
    profile: Optional[str] = dataclasses.field(default=None, repr=False)
    config_file: Optional[str] = dataclasses.field(default=None, repr=False)

    # HTTP / client behavior
    auth_type: Optional[str] = None
    http_timeout_seconds: Optional[int] = dataclasses.field(default=None, repr=False)
    retry_timeout_seconds: Optional[int] = dataclasses.field(default=None, repr=False)
    debug_truncate_bytes: Optional[int] = dataclasses.field(default=None, repr=False)
    debug_headers: Optional[bool] = dataclasses.field(default=None, repr=False)
    rate_limit: Optional[int] = dataclasses.field(default=None, repr=False)

    # Extras
    product: Optional[str] = dataclasses.field(default_factory=_get_env_product, repr=False)
    product_version: Optional[str] = dataclasses.field(default_factory=_get_env_product_version, repr=False)
    product_tag: Optional[str] = dataclasses.field(default_factory=_get_env_product_tag, repr=False)

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

        state["_was_connected"] = self._sdk is not None
        state["_cached_token"] = self.current_token()

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._sdk = None

        if self.auth_type in ["external-browser", "runtime"]:
            self.auth_type = None

        if self._was_connected:
            self.connect(reset=True)

    def __enter__(self) -> "Workspace":
        self._was_connected = self._sdk is not None
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._was_connected:
            self.close()

    def __del__(self):
        self.close()

    # -------------------------
    # Clone
    # -------------------------
    def clone_instance(
        self,
        **kwargs
    ) -> "Workspace":
        state = self.__getstate__()
        state.update(kwargs)
        return Workspace().__setstate__(state)

    # -------------------------
    # SDK connection
    # -------------------------
    @property
    def connected(self):
        return self._sdk is not None

    def connect(self, reset: bool = False, clone: bool = False) -> "Workspace":
        if reset:
            self._sdk = None

        if self._sdk is not None:
            return self

        instance = self.clone_instance() if clone else self

        require_databricks_sdk()
        logger.debug("Connecting %s", self)

        # Build Config from config_dict if available, else from fields.
        kwargs = {
            "host": instance.host,
            "account_id": instance.account_id,
            "token": instance.token,
            "client_id": instance.client_id,
            "client_secret": instance.client_secret,
            "token_audience": instance.token_audience,
            "azure_workspace_resource_id": instance.azure_workspace_resource_id,
            "azure_use_msi": instance.azure_use_msi,
            "azure_client_secret": instance.azure_client_secret,
            "azure_client_id": instance.azure_client_id,
            "azure_tenant_id": instance.azure_tenant_id,
            "azure_environment": instance.azure_environment,
            "google_credentials": instance.google_credentials,
            "google_service_account": instance.google_service_account,
            "profile": instance.profile,
            "config_file": instance.config_file,
            "auth_type": instance.auth_type,
            "http_timeout_seconds": instance.http_timeout_seconds,
            "retry_timeout_seconds": instance.retry_timeout_seconds,
            "debug_truncate_bytes": instance.debug_truncate_bytes,
            "debug_headers": instance.debug_headers,
            "rate_limit": instance.rate_limit,
            "product": instance.product,
            "product_version": instance.product_version,
        }

        build_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            instance._sdk = WorkspaceClient(**build_kwargs)
        except ValueError as e:
            if "cannot configure default credentials" in str(e) and instance.auth_type is None:
                last_error = e

                auth_types = ["runtime"] if instance.is_in_databricks_environment() else ["external-browser"]

                for auth_type in auth_types:
                    build_kwargs["auth_type"] = auth_type

                    try:
                        instance._sdk = WorkspaceClient(**build_kwargs)
                        break
                    except Exception as se:
                        last_error = se
                        build_kwargs.pop("auth_type")

                if instance._sdk is None:
                    if instance.is_in_databricks_environment() and instance._cached_token:
                        build_kwargs["token"] = instance._cached_token

                        try:
                            instance._sdk = WorkspaceClient(**build_kwargs)
                        except Exception as se:
                            last_error = se

                if instance._sdk is None:
                    raise last_error
            else:
                raise e

        # backfill resolved config values
        for key in list(kwargs.keys()):
            if getattr(instance, key, None) is None:
                v = getattr(instance._sdk.config, key, None)
                if v is not None:
                    setattr(instance, key, v)

        logger.info("Connected %s", instance)

        return instance

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
    @staticmethod
    def _local_cache_token_path():
        oauth_dir = Path.home() / ".config" / "databricks-sdk-py" / "oauth"
        if not oauth_dir.is_dir():
            return None

        # "first" = lexicographically first (stable)
        files = sorted(p for p in oauth_dir.iterdir() if p.is_file())
        return str(files[0]) if files else None

    def reset_local_cache(self):
        local_cache = self._local_cache_token_path()

        if local_cache:
            os.remove(local_cache)

    @property
    def current_user(self):
        try:
            return self.sdk().current_user.me()
        except:
            if self.auth_type == "external-browser":
                self.reset_local_cache()
            raise

    def current_token(self) -> str:
        if self.token:
            return self.token

        sdk = self.sdk()
        conf = sdk.config
        token = conf._credentials_strategy(conf)()["Authorization"].replace("Bearer ", "")

        return token

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #
    def filesytem(
        self,
        workspace: Optional["Workspace"] = None,
    ):
        from .filesytem import DatabricksFileSystem, DatabricksFileSystemHandler

        handler = DatabricksFileSystemHandler(
            workspace=self if workspace is None else workspace
        )

        return DatabricksFileSystem(
            handler=handler
        )

    def dbfs_path(
        self,
        parts: Union[List[str], str],
        kind: Optional[DatabricksPathKind] = None,
        workspace: Optional["Workspace"] = None
    ):
        workspace = self if workspace is None else workspace

        if kind is None or isinstance(parts, str):
            return DatabricksPath.parse(
                obj=parts,
                workspace=workspace
            )

        return DatabricksPath(
            kind=kind,
            parts=parts,
            _workspace=workspace
        )

    def shared_cache_path(
        self,
        suffix: Optional[str] = None
    ) -> DatabricksPath:
        """
        Shared cache base under Volumes for the current user.
        """
        base = "/Workspace/Shared/.ygg/cache"

        if not suffix:
            return base

        suffix = suffix.lstrip("/")
        return self.dbfs_path(f"{base}/{suffix}")

    # ------------------------------------------------------------------ #
    # SDK access / connection
    # ------------------------------------------------------------------ #

    def sdk(self) -> "WorkspaceClient":
        return self.connect()._sdk

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

    @staticmethod
    def is_in_databricks_environment():
        return os.getenv("DATABRICKS_RUNTIME_VERSION") is not None

    def default_tags(self):
        return {
            k: v
            for k, v in (
                ("Product", self.product),
                ("ProductVersion", self.product_version),
                ("ProductTag", self.product_tag),
            )
            if v
        }

    def merge_tags(self, existing: dict | None = None):
        if existing:
            return self.default_tags()

    def sql(
        self,
        workspace: Optional["Workspace"] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        **kwargs
    ):
        from ..sql import SQLEngine

        return SQLEngine(
            workspace=self if workspace is None else workspace,
            catalog_name=catalog_name,
            schema_name=schema_name,
            **kwargs
        )

    def clusters(
        self,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
        **kwargs
    ) -> "Cluster":
        from ..compute.cluster import Cluster

        return Cluster(workspace=self, cluster_id=cluster_id, cluster_name=cluster_name, **kwargs)

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

    def dbfs_path(
        self,
        parts: Union[List[str], str],
        kind: Optional[DatabricksPathKind] = None,
        workspace: Optional["Workspace"] = None
    ):
        return self.workspace.dbfs_path(
            kind=kind,
            parts=parts,
            workspace=workspace
        )

    def sdk(self):
        return self.workspace.sdk()

    @property
    def current_user(self):
        return self.workspace.current_user
