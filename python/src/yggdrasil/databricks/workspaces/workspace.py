"""Workspace configuration and Databricks SDK helpers."""

import dataclasses
import logging
import os
import posixpath
import time
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import (
    BinaryIO,
    Iterator,
    Optional,
    Union, TYPE_CHECKING, List, Set, Iterable
)

from databricks.sdk import WorkspaceClient
from databricks.sdk.dbutils import FileInfo
from databricks.sdk.errors import ResourceDoesNotExist, NotFound, InternalError
from databricks.sdk.service.files import DirectoryEntry
from databricks.sdk.service.iam import User, ComplexValue
from databricks.sdk.service.workspace import ExportFormat, ObjectInfo

from .path import DatabricksPath, DatabricksPathKind
from ...pyutils.expiring_dict import ExpiringDict
from ...pyutils.waiting_config import WaitingConfig, WaitingConfigArg
from ...version import __version__ as YGGDRASIL_VERSION

if TYPE_CHECKING:
    from ..sql.engine import SQLEngine
    from ..sql.warehouse import SQLWarehouse
    from ..compute.cluster import Cluster
    from ..secrets.secret import Secret


__all__ = [
    "DBXWorkspace",
    "Workspace",
    "WorkspaceService",
]


LOGGER = logging.getLogger(__name__)
CHECKED_TMP_WORKSPACES: ExpiringDict[str, Set[str]] = ExpiringDict()

def is_checked_tmp_path(
    host: str,
    base_path: str
):
    existing = CHECKED_TMP_WORKSPACES.get(host)

    if existing is None:
        CHECKED_TMP_WORKSPACES[host] = set(base_path)

        return False

    if base_path in existing:
        return True

    existing.add(base_path)

    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_env_product():
    return os.getenv("DATABRICKS_PRODUCT")


def _get_env_product_version():
    v = os.getenv("DATABRICKS_PRODUCT_VERSION")

    if not v:
        if _get_env_product() == "yggdrasil":
            return YGGDRASIL_VERSION
        return None
    return v


def _get_env_product_tag():
    return os.getenv("DATABRICKS_PRODUCT_TAG")


@dataclass
class Workspace:
    """Configuration wrapper for connecting to a Databricks workspace."""
    # Databricks / generic
    host: Optional[str] = None
    account_id: Optional[str] = dataclasses.field(default=None, repr=False)
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
    custom_tags: Optional[dict] = dataclasses.field(default=None, repr=False)

    # Runtime cache (never serialized)
    _sdk: Optional["WorkspaceClient"] = dataclasses.field(default=None, repr=False, compare=False, hash=False)
    _sql: Optional["SQLEngine"] = dataclasses.field(default=None, repr=False, compare=False, hash=False)
    _secrets: Optional["Secret"] = dataclasses.field(default=None, repr=False, compare=False, hash=False)
    _was_connected: bool = dataclasses.field(default=None, repr=False, compare=False, hash=False)
    _cached_token: Optional[str] = dataclasses.field(default=None, repr=False, compare=False, hash=False)

    # -------------------------
    # Python methods
    # -------------------------
    def __getstate__(self):
        """Serialize the workspace state for pickling.

        Returns:
            A pickle-ready state dictionary.
        """
        state = self.__dict__.copy()
        state.pop("_sdk", None)
        state.pop("_sql", None)

        state["_was_connected"] = self._sdk is not None
        state["_cached_token"] = self.current_token()

        return state

    def __setstate__(self, state):
        """Restore workspace state after unpickling.

        Args:
            state: Serialized state dictionary.
        """
        self.__dict__.update(state)
        self._sdk = None

        if self.auth_type in ["external-browser", "runtime"]:
            self.auth_type = None

        if self._was_connected:
            self.connect(reset=True)

    def __enter__(self) -> "Workspace":
        """Enter a context manager and connect to the workspace.

        Returns:
            The connected Workspace instance.
        """
        self._was_connected = self._sdk is not None
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and close if newly connected.

        Args:
            exc_type: Exception type, if raised.
            exc_val: Exception value, if raised.
            exc_tb: Exception traceback, if raised.

        Returns:
            None.
        """
        if not self._was_connected:
            self.close()

    def __del__(self):
        self.close()

    # -------------------------
    # Clone
    # -------------------------
    def clone_instance(
        self,
        host: Optional[str] = None,
        account_id: Optional[str] = None,
        token: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> "Workspace":
        """Clone the workspace config with overrides.

        Returns:
            A new Workspace instance with updated fields.
        """
        return Workspace(
            host = host or self.host,
            account_id = account_id or self.account_id,
            token = token or self.token,
            client_id = client_id or self.client_id,
            client_secret = client_secret or self.client_secret,
            token_audience = self.token_audience,
            azure_workspace_resource_id = self.azure_workspace_resource_id,
            azure_use_msi = self.azure_use_msi,
            azure_client_secret = self.azure_client_secret,
            azure_client_id = self.azure_client_id,
            azure_tenant_id = self.azure_tenant_id,
            azure_environment = self.azure_environment,
            google_credentials = self.google_credentials,
            google_service_account = self.google_service_account,
            profile = self.profile,
            config_file = self.config_file,
            auth_type = self.auth_type,
            http_timeout_seconds = self.http_timeout_seconds,
            retry_timeout_seconds = self.retry_timeout_seconds,
            debug_truncate_bytes = self.debug_truncate_bytes,
            debug_headers = self.debug_headers,
            rate_limit = self.rate_limit,
            product = self.product,
            product_version = self.product_version,
            product_tag = self.product_tag,
            custom_tags = self.custom_tags,
            _sdk = None,
            _was_connected = self._was_connected,
            _cached_token = self._cached_token,
        )

    # -------------------------
    # SDK connection
    # -------------------------
    @property
    def connected(self):
        """Return True when a WorkspaceClient is cached.

        Returns:
            True if connected, otherwise False.
        """
        return self._sdk is not None

    def connect(self, reset: bool = False, clone: bool = False) -> "Workspace":
        """Connect to the workspace and cache the SDK client.

        Args:
            reset: Whether to reset the cached client before connecting.
            clone: Whether to connect a cloned instance.

        Returns:
            The connected Workspace instance.
        """
        if reset:
            self._sdk = None

        if self._sdk is not None:
            return self

        instance = self.clone_instance() if clone else self

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

        if not self.product and self.auth_type == "external-browser":
            conf = instance._sdk.config
            conf._init_product(
                self.current_user.user_name,
                "0.0.0"
            )

            self.product, self.product_version = conf._product_info

        return instance

    # ------------------------------------------------------------------ #
    # Context manager + lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """
        Drop the cached WorkspaceClient (no actual close needed, but this
        avoids reusing stale config).
        """
        if self._sdk is not None:
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
        """Remove cached browser OAuth tokens.

        Returns:
            None.
        """
        local_cache = self._local_cache_token_path()

        if local_cache:
            os.remove(local_cache)

    @property
    def safe_host(self):
        return self.sdk().config.host

    @property
    def current_user(self):
        """Return the current Databricks user.

        Returns:
            The current user object from the SDK.
        """
        try:
            found = self.sdk().current_user.me()
        except:
            if self.auth_type == "runtime":
                found = User(
                    display_name="Databricks Runtime",
                    user_name="databricks-runtime",
                    name="Runtime",
                    groups=[]
                )
            else:
                raise

        if found is None:
            if self.auth_type == "external-browser":
                self.reset_local_cache()
            raise

        return found

    def current_user_groups(
        self,
        with_public: bool = True,
        raise_error: bool = True
    ) -> Iterable[ComplexValue]:
        try:
            user = self.current_user

            if user is not None:
                found = user.groups
            else:
                found = []
        except (NotFound, ResourceDoesNotExist, InternalError):
            if raise_error:
                raise
            found = []

        if not with_public:
            found = [
                group
                for group in found
                if group.display not in {"users"}
            ]

        return found

    def current_token(self) -> str:
        """Return the active API token for this workspace.

        Returns:
            The bearer token string.
        """
        if self.token:
            return self.token

        sdk = self.sdk()
        conf = sdk.config
        token = conf._credentials_strategy(conf)()["Authorization"].replace("Bearer ", "")

        return token

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #
    def arrow_filesystem(
        self,
        workspace: Optional["Workspace"] = None,
    ):
        """Return a PyArrow filesystem for Databricks paths.

        Args:
            workspace: Optional workspace override.

        Returns:
            A DatabricksFileSystem instance.
        """
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
        workspace: Optional["Workspace"] = None,
        temporary: bool = False
    ):
        """Create a DatabricksPath in this workspace.

        Args:
            parts: Path parts or string to parse.
            kind: Optional path kind override.
            workspace: Optional workspace override.
            temporary: Temporary path

        Returns:
            A DatabricksPath instance.
        """
        workspace = self if workspace is None else workspace

        if kind is None or isinstance(parts, str):
            return DatabricksPath.parse(
                obj=parts,
                workspace=workspace,
                temporary=temporary
            )

        return DatabricksPath(
            kind=kind,
            parts=parts,
            temporary=temporary,
            _workspace=workspace
        )

    @staticmethod
    def _base_tmp_path(
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> str:
        if catalog_name and schema_name:
            base_path = "/Volumes/%s/%s/%s" % (
                catalog_name, schema_name, volume_name or "tmp"
            )
        else:
            base_path = "/Workspace/Shared/.ygg/tmp"

        return base_path

    def tmp_path(
        self,
        suffix: Optional[str] = None,
        extension: Optional[str] = None,
        max_lifetime: Optional[float] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        volume_name: Optional[str] = None,
        base_path: Optional[str] = None,
    ) -> DatabricksPath:
        """
        Shared cache base under Volumes for the current user.

        Args:
            suffix: Optional suffix
            extension: Optional extension suffix to append.
            max_lifetime: Max lifetime of temporary path
            catalog_name: Unity catalog name for volume path
            schema_name: Unity schema name for volume path
            volume_name: Unity volume name for volume path
            base_path: Base temporary path

        Returns:
            A DatabricksPath pointing at the shared cache location.
        """
        start = int(time.time())
        max_lifetime = int(max_lifetime or 48 * 3600)
        end = max(0, int(start + max_lifetime))

        base_path = base_path or self._base_tmp_path(
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name
        )

        rnd = os.urandom(4).hex()
        temp_path = f"tmp-{start}-{end}-{rnd}"

        if suffix:
            temp_path += suffix

        if extension:
            temp_path += ".%s" % extension

        self.clean_tmp_folder(
            raise_error=False,
            wait=False,
            base_path=base_path
        )

        return self.dbfs_path(f"{base_path}/{temp_path}")

    def clean_tmp_folder(
        self,
        raise_error: bool = True,
        wait: Optional[WaitingConfigArg] = True,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        volume_name: Optional[str] = None,
        base_path: Optional[str] = None,
    ):
        wait = WaitingConfig.check_arg(wait)

        base_path = base_path or self._base_tmp_path(
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name
        )

        if is_checked_tmp_path(host=self.safe_host, base_path=base_path):
            return self

        if wait.timeout:
            base_path = self.dbfs_path(base_path)

            LOGGER.debug(
                "Cleaning temp path %s",
                base_path
            )

            for path in base_path.ls(recursive=False, allow_not_found=True):
                if path.name.startswith("tmp"):
                    parts = path.name.split("-")

                    if len(parts) > 2 and parts[0] == "tmp" and parts[1].isdigit() and parts[2].isdigit():
                        end = int(parts[2])

                        if end and time.time() > end:
                            path.remove(recursive=True)

            LOGGER.info(
                "Cleaned temp path %s",
                base_path
            )
        else:
            Thread(
                target=self.clean_tmp_folder,
                kwargs={
                    "raise_error": raise_error,
                    "base_path": base_path
                }
            ).start()

        return self

    def shared_cache_path(
        self,
        suffix: Optional[str] = None
    ) -> DatabricksPath:
        """
        Shared cache base under Volumes for the current user.

        Args:
            suffix: Optional path suffix to append.

        Returns:
            A DatabricksPath pointing at the shared cache location.
        """
        base = "/Workspace/Shared/.ygg/cache"

        if not suffix:
            return base

        suffix = suffix.lstrip("/")
        return self.dbfs_path(f"{base}/{suffix}")

    # ------------------------------------------------------------------ #
    # SDK access / connection
    # ------------------------------------------------------------------ #

    def sdk(self) -> WorkspaceClient:
        """Return the connected WorkspaceClient.

        Returns:
            The WorkspaceClient instance.
        """
        return self.connect(clone=False)._sdk

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

        Args:
            path: Path string to list.
            recursive: Whether to list recursively.

        Returns:
            An iterator of workspace/DBFS/volume entries.
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

        else:
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

        Args:
            path: Path to open.
            workspace_format: Optional export format for workspace paths.

        Returns:
            A BinaryIO stream for reading.
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
        """Return True when running on a Databricks runtime.

        Returns:
            True if running on Databricks, otherwise False.
        """
        return os.getenv("DATABRICKS_RUNTIME_VERSION") is not None

    def default_tags(self, update: bool = True):
        """Return default resource tags for Databricks assets.

        Returns:
            A dict of default tags.
        """
        if update:
            base = dict()
        else:
            base = {
                k: v
                for k, v in (
                    ("Product", self.product),
                    ("ProductVersion", self.product_version),
                    ("ProductTag", self.product_tag),
                )
                if v
            }

        if self.custom_tags:
            base.update(self.custom_tags)

        return base

    def sql(
        self,
        workspace: Optional["Workspace"] = None,
        warehouse: Optional["SQLWarehouse"] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ):
        """Return a SQLEngine configured for this workspace.

        Args:
            workspace: Optional workspace override.
            warehouse: Optional SQL warehouse.
            catalog_name: Optional catalog name.
            schema_name: Optional schema name.

        Returns:
            A SQLEngine instance.
        """
        from ..sql import SQLEngine

        if workspace is None and warehouse is None and catalog_name is None and schema_name is None:
            if self._sql is not None:
                return self._sql

            workspace = self if workspace is None else workspace

            self._sql = SQLEngine(
                workspace=workspace,
                catalog_name=catalog_name,
                schema_name=schema_name,
                _warehouse=warehouse,
            )

            return self._sql

        workspace = self if workspace is None else workspace

        if warehouse is not None:
            if isinstance(warehouse, str):
                warehouse = self.warehouses().find_warehouse(warehouse_name=warehouse)

        return SQLEngine(
            workspace=workspace,
            catalog_name=catalog_name,
            schema_name=schema_name,
            _warehouse=warehouse,
        )

    def warehouses(
        self,
        workspace: Optional["Workspace"] = None,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
    ):
        from ..sql.warehouse import SQLWarehouse

        return SQLWarehouse(
            workspace=self if workspace is None else workspace,
            warehouse_id=warehouse_id,
            warehouse_name=warehouse_name
        )

    def clusters(
        self,
        workspace: Optional["Workspace"] = None,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
    ) -> "Cluster":
        """Return a Cluster helper bound to this workspace.

        Args:
            workspace: Optional workspace override.
            cluster_id: Optional cluster id.
            cluster_name: Optional cluster name.

        Returns:
            A Cluster instance.
        """
        from ..compute.cluster import Cluster

        return Cluster(
            workspace=self if workspace is None else workspace,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
        )

    def secrets(
        self,
        workspace: Optional["Workspace"] = None,
        scope: Optional[str] = None,
        key: Optional[str] = None,
    ):
        from ..secrets.secret import Secret

        if workspace is None and scope is None and key is None:
            if self._secrets is not None:
                return self._secrets

            self._secrets = Secret(
                workspace=self if workspace is None else workspace,
                scope=scope,
                key=key,
            )

            return self._secrets

        return Secret(
            workspace=self if workspace is None else workspace,
            scope=scope,
            key=key,
        )


# ---------------------------------------------------------------------------
# Workspace-bound base class
# ---------------------------------------------------------------------------

DBXWorkspace = Workspace


@dataclass
class WorkspaceService(ABC):
    """Base class for helpers that depend on a Workspace."""
    workspace: Workspace = dataclasses.field(default_factory=Workspace)

    def __post_init__(self):
        if self.workspace is None:
            self.workspace = Workspace()

    def __enter__(self):
        """Enter a context manager and connect the workspace.

        Returns:
            The current WorkspaceService instance.
        """
        self.workspace.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close the workspace.

        Args:
            exc_type: Exception type, if raised.
            exc_val: Exception value, if raised.
            exc_tb: Exception traceback, if raised.

        Returns:
            None.
        """
        self.workspace.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def is_in_databricks_environment(self):
        """Return True when running on a Databricks runtime.

        Returns:
            True if running on Databricks, otherwise False.
        """
        return self.workspace.is_in_databricks_environment()

    def connect(self, clone: bool = False):
        """Connect the underlying workspace.

        Returns:
            The current WorkspaceService instance.
        """
        self.workspace = self.workspace.connect(clone=clone)
        return self

    def dbfs_path(
        self,
        parts: Union[List[str], str],
        kind: Optional[DatabricksPathKind] = None,
        workspace: Optional["Workspace"] = None
    ) -> "DatabricksPath":
        """Create a DatabricksPath in the underlying workspace.

        Args:
            parts: Path parts or string to parse.
            kind: Optional path kind override.
            workspace: Optional workspace override.

        Returns:
            A DatabricksPath instance.
        """
        return self.workspace.dbfs_path(
            kind=kind,
            parts=parts,
            workspace=workspace
        )

    def sdk(self):
        """Return the WorkspaceClient for the underlying workspace.

        Returns:
            The WorkspaceClient instance.
        """
        return self.workspace.sdk()

    @property
    def current_user(self):
        """Return the current Databricks user.

        Returns:
            The current user object from the SDK.
        """
        return self.workspace.current_user

    def current_user_groups(
        self,
        with_public: bool = True,
        raise_error: bool = True
    ):
        return self.workspace.current_user_groups(
            with_public=with_public,
            raise_error=raise_error
        )