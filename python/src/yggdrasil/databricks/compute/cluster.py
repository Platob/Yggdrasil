"""
Cluster management helpers for Databricks compute.

This module provides a lightweight ``Cluster`` helper that wraps the
Databricks SDK to simplify common CRUD operations and metadata handling
for clusters. Metadata is stored in custom tags prefixed with
``yggdrasil:``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import inspect
import logging
import os
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Iterator, Optional, Union, List, Dict, ClassVar

from databricks.sdk import ClustersAPI
from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import ResourceDoesNotExist
from databricks.sdk.service.compute import (
    ClusterDetails, Language, Kind, State, DataSecurityMode, Library, PythonPyPiLibrary, LibraryInstallStatus,
    ClusterAccessControlRequest, ClusterPermissionLevel
)
from databricks.sdk.service.compute import SparkVersion, RuntimeEngine

from .execution_context import ExecutionContext
from ..workspaces.workspace import WorkspaceService, Workspace
from ...pyutils.equality import dicts_equal
from ...pyutils.expiring_dict import ExpiringDict
from ...pyutils.modules import PipIndexSettings
from ...pyutils.python_env import PythonEnv
from ...pyutils.waiting_config import WaitingConfig, WaitingConfigArg

_CREATE_ARG_NAMES = {_ for _ in inspect.signature(ClustersAPI.create).parameters.keys()}
_EDIT_ARG_NAMES = {_ for _ in inspect.signature(ClustersAPI.edit).parameters.keys()}

__all__ = ["Cluster"]


LOGGER = logging.getLogger(__name__)
NAME_ID_CACHE: dict[str, ExpiringDict] = {}


def set_cached_cluster_name(
    host: str,
    cluster_name: str,
    cluster_id: str
) -> None:
    existing = NAME_ID_CACHE.get(host)

    if not existing:
        existing = NAME_ID_CACHE[host] = ExpiringDict(default_ttl=60)

    existing[cluster_name] = cluster_id


def get_cached_cluster_id(
    host: str,
    cluster_name: str,
) -> str:
    existing = NAME_ID_CACHE.get(host)

    return existing.get(cluster_name) if existing else None


# module-level mapping Databricks Runtime -> (major, minor) Python version
_PYTHON_BY_DBR: dict[str, tuple[int, int]] = {
    "10.4": (3, 8),
    "11.3": (3, 9),
    "12.2": (3, 9),
    "13.3": (3, 10),
    "14.3": (3, 10),
    "15.4": (3, 11),
    "16.4": (3, 12),
    "17.0": (3, 12),
    "17.1": (3, 12),
    "17.2": (3, 12),
    "17.3": (3, 12),
    "18.0": (3, 12),
}


@dataclass
class Cluster(WorkspaceService):
    """Helper for creating, retrieving, updating, and deleting clusters.

    Parameters
    ----------
    workspace:
        Optional :class:`Workspace` (or config-compatible object) used to
        build the underlying :class:`databricks.sdk.WorkspaceClient`.
        Defaults to a new :class:`Workspace`.
    cluster_id:
        Optional existing cluster identifier. Methods that operate on a
        cluster will use this value when ``cluster_id`` is omitted.
    """
    cluster_id: Optional[str] = None
    cluster_name: Optional[str] = None
    
    _details: Optional["ClusterDetails"] = dataclasses.field(default=None, repr=False)
    _details_refresh_time: float = dataclasses.field(default=0, repr=False)
    _system_context: Optional[ExecutionContext] = dataclasses.field(default=None, repr=False)

    # host → Cluster instance
    _env_clusters: ClassVar[Dict[str, "Cluster"]] = {}

    def __post_init__(self):
        if self.cluster_name and not self.cluster_id:
            self.cluster_id = self.find_cluster(cluster_name=self.cluster_name).cluster_id

    def __repr__(self):
        return "%s(url=%s)" % (
            self.__class__.__name__,
            self.url()
        )

    def __str__(self):
        return self.url()

    def url(self) -> str:
        return "%s/compute/clusters/%s" % (
            self.workspace.safe_host,
            self.cluster_id or "unknown"
        )

    @property
    def id(self):
        """Return the current cluster id."""
        return self.cluster_id

    @property
    def name(self) -> str:
        """Return the current cluster name."""
        return self.cluster_name

    @property
    def system_context(self):
        if self._system_context is None:
            self._system_context = self.context(language=Language.PYTHON)
        return self._system_context

    def is_in_databricks_environment(self):
        """Return True when running on a Databricks runtime."""
        return self.workspace.is_in_databricks_environment()

    @property
    def details(self):
        """Return cached cluster details, refreshing when needed."""
        if self._details is None and self.cluster_id is not None:
            self.details = self.clusters_client().get(cluster_id=self.cluster_id)
        return self._details

    def fresh_details(self, max_delay: float | None = None):
        """Refresh cluster details if older than ``max_delay`` seconds.

        Args:
            max_delay: Maximum age in seconds before refresh.

        Returns:
            The latest ClusterDetails object, if available.
        """
        max_delay = max_delay or 0
        delay = time.time() - self._details_refresh_time

        if self.cluster_id and delay > max_delay:
            self.details = self.clusters_client().get(cluster_id=self.cluster_id)
        return self._details

    def refresh(self, max_delay: float | None = None):
        self.details = self.fresh_details(max_delay=max_delay)

        return self

    @details.setter
    def details(self, value: "ClusterDetails"):
        """Cache cluster details and update identifiers."""
        self._details_refresh_time = time.time()
        self._details = value

        self.cluster_id = value.cluster_id
        self.cluster_name = value.cluster_name

    @property
    def state(self):
        """Return the current cluster state."""
        self.refresh()

        if self._details is not None:
            return self._details.state
        return State.UNKNOWN

    @property
    def is_running(self):
        """Return True when the cluster is running."""
        return self.state == State.RUNNING

    @property
    def is_pending(self):
        """Return True when the cluster is starting, resizing, or terminating."""
        return self.state in (
            State.PENDING, State.RESIZING, State.RESTARTING,
            State.TERMINATING
        )

    @property
    def is_error(self):
        """Return True when the cluster is in an error state."""
        return self.state == State.ERROR

    def raise_for_status(self):
        """Raise a DatabricksError if the cluster is in an error state."""
        if self.is_error:
            raise DatabricksError("Error in %s" % self)

        return self

    def wait_for_status(
        self,
        wait: Optional[WaitingConfigArg] = True,
        raise_error: bool = True
    ):
        """Wait for the cluster to exit pending states.

        Args:
            wait: Waiting config
            raise_error: Raise error if failed

        Returns:
            The current Cluster instance.
        """
        wait = WaitingConfig.check_arg(wait)
        iteration, start = 0, time.time()

        if wait.timeout:
            while self.is_pending:
                wait.sleep(iteration=iteration, start=start)
                iteration += 1

            self.wait_installed_libraries(
                raise_error=raise_error
            )

            if raise_error:
                self.raise_for_status()

        return self

    @property
    def spark_version(self) -> str:
        """Return the cluster Spark version string."""
        d = self.details
        if d is None:
            return None
        return d.spark_version

    @property
    def runtime_version(self):
        """Return the Databricks runtime major/minor version string."""
        # Extract "major.minor" from strings like "17.3.x-scala2.13-ml-gpu"
        v = self.spark_version

        if not v:
            return None

        parts = v.split(".")

        if len(parts) < 2:
            return None

        return ".".join(parts[:2])  # e.g. "17.3"

    @property
    def python_version(self) -> Optional[tuple[int, int]]:
        """Return the cluster Python version as (major, minor), if known.

        Uses the Databricks Runtime -> Python mapping in _PYTHON_BY_DBR.
        When the runtime can't be mapped, returns ``None``.
        """
        v = self.runtime_version

        if not v:
            return None

        return _PYTHON_BY_DBR.get(v)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def clusters_client(self) -> "ClustersAPI":
        """Return the Databricks clusters API client."""
        return self.workspace.sdk().clusters

    def shared_cache_path(
        self,
        suffix: str
    ):
        assert suffix, "Missing suffix arg"

        return self.workspace.shared_cache_path(
            suffix="/cluster/%s" % suffix.lstrip("/")
        )

    def spark_versions(
        self,
        photon: Optional[bool] = None,
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
    ):
        """List supported Spark runtimes filtered by photon and python version.

        Args:
            photon: If set, filter by Photon (True) or non-Photon (False).
            python_version: Optional Python version filter (string or tuple).

        Returns:
            A list of SparkVersion entries matching the filters.
        """
        all_versions = self.clusters_client().spark_versions().versions

        if not all_versions:
            raise ValueError("No databricks spark versions found")

        versions = all_versions

        # --- filter by Photon / non-Photon ---
        if photon is not None:
            if photon:
                versions = [v for v in versions if "photon" in v.key.lower()]
            else:
                versions = [v for v in versions if "photon" not in v.key.lower()]

        # --- filter by Python version (Databricks Runtime mapping) ---
        if python_version is not None:
            # normalize input python_version to (major, minor)
            if isinstance(python_version, str):
                parts = python_version.split(".")
                py_filter = tuple(int(p) for p in parts[:2])
            else:
                py_filter = tuple(python_version[:2])

            def dbr_from_key(key: str) -> Optional[str]:
                # "17.3.x-gpu-ml-scala2.13" -> "17.3"
                dbr_version_parts = key.split(".")
                if len(dbr_version_parts) < 2:
                    return None
                return ".".join(dbr_version_parts[:2])

            def py_for_key(key: str) -> Optional[tuple[int, int]]:
                dbr = dbr_from_key(key)
                if dbr is None:
                    return None
                return _PYTHON_BY_DBR.get(dbr)

            versions = [v for v in versions if py_for_key(v.key) == py_filter]

            # Handle superior pyton versions
            if not versions and py_filter[1] > 12:
                return self.spark_versions(photon=photon)

        return versions

    def latest_spark_version(
        self,
        photon: Optional[bool] = None,
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
    ):
        """Return the latest Spark version that matches requested filters.

        Args:
            photon: If set, filter by Photon (True) or non-Photon (False).
            python_version: Optional Python version filter (string or tuple).

        Returns:
            The latest SparkVersion matching the filters.
        """
        versions = self.spark_versions(photon=photon, python_version=python_version)

        max_version: SparkVersion = None

        for version in versions:
            if max_version is None or version.key > max_version.key:
                max_version = version

        if max_version is None:
            raise ValueError(f"No databricks runtime version found for photon={photon} and python_version={python_version}")

        return max_version

    # ------------------------------------------------------------------ #
    # CRUD operations
    # ------------------------------------------------------------------ #
    def _check_details(
        self,
        details: "ClusterDetails",
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
        **kwargs
    ):
        pip_settings = PipIndexSettings.default_settings()

        new_details = ClusterDetails(**{
            **details.as_shallow_dict(),
            **kwargs
        })

        default_tags = self.workspace.default_tags()

        if new_details.custom_tags is None:
            new_details.custom_tags = default_tags
        elif default_tags:
            new_tags = new_details.custom_tags.copy()
            new_tags.update(default_tags)

            new_details.custom_tags = new_tags

        if new_details.cluster_name is None:
            new_details.cluster_name = self.workspace.current_user.user_name

        if new_details.spark_version is None or python_version:
            new_details.spark_version = self.latest_spark_version(
                photon=False, python_version=python_version
            ).key

        if new_details.single_user_name:
            if not new_details.data_security_mode:
                new_details.data_security_mode = DataSecurityMode.DATA_SECURITY_MODE_DEDICATED

        if not new_details.node_type_id:
            new_details.node_type_id = "rd-fleet.xlarge"

        if getattr(new_details, "virtual_cluster_size", None) is None and new_details.num_workers is None and new_details.autoscale is None:
            if new_details.is_single_node is None:
                new_details.is_single_node = True

        if new_details.is_single_node is not None and new_details.kind is None:
            new_details.kind = Kind.CLASSIC_PREVIEW

        if pip_settings.extra_index_urls:
            if new_details.spark_env_vars is None:
                new_details.spark_env_vars = {}
            str_urls = " ".join(pip_settings.extra_index_urls)
            new_details.spark_env_vars["UV_EXTRA_INDEX_URL"] = new_details.spark_env_vars.get("UV_INDEX", str_urls)
            new_details.spark_env_vars["PIP_EXTRA_INDEX_URL"] = new_details.spark_env_vars.get("PIP_EXTRA_INDEX_URL", str_urls)

        return new_details

    def create_or_update(
        self,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        wait_update: Optional[WaitingConfigArg] = True,
        **cluster_spec: Any
    ):
        """Create a new cluster or update an existing one.

        Args:
            cluster_id: Optional cluster id to update.
            cluster_name: Optional cluster name to update or create.
            libraries: Optional libraries to install.
            wait_update: wait timeout, if None it will not wait completion
            **cluster_spec: Cluster specification overrides.

        Returns:
            A Cluster instance pointing at the created/updated cluster.
        """
        found = self.find_cluster(
            cluster_id=cluster_id or self.cluster_id,
            cluster_name=cluster_name or self.cluster_name,
            raise_error=False
        )

        if found is not None:
            return found.update(
                cluster_name=cluster_name,
                libraries=libraries,
                wait=wait_update,
                **cluster_spec
            )

        return self.create(
            cluster_name=cluster_name,
            libraries=libraries,
            wait=wait_update,
            **cluster_spec
        )

    def create(
        self,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        wait: Union[WaitingConfigArg] = True,
        **cluster_spec: Any
    ) -> str:
        """Create a new cluster and optionally install libraries.

        Args:
            libraries: Optional list of libraries to install after creation.
            wait: wait timeout, if None it will not wait completion
            **cluster_spec: Cluster specification overrides.

        Returns:
            The current Cluster instance.
        """
        cluster_spec["autotermination_minutes"] = int(cluster_spec.get("autotermination_minutes", 30))
        update_details = self._check_details(details=ClusterDetails(), **cluster_spec)
        update_details = {
            k: v
            for k, v in update_details.as_shallow_dict().items()
            if k in _CREATE_ARG_NAMES
        }

        LOGGER.debug(
            "Creating Databricks cluster %s with %s",
            update_details["cluster_name"],
            update_details,
        )

        self.details = self.clusters_client().create(**update_details)

        LOGGER.info(
            "Created %s",
            self
        )

        self.install_libraries(libraries=libraries, raise_error=False, wait_timeout=None)

        self.wait_for_status(wait=wait)

        return self

    def update(
        self,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        access_control_list: Optional[List["ClusterAccessControlRequest"] | bool] = True,
        wait: Optional[WaitingConfigArg] = True,
        **cluster_spec: Any
    ) -> "Cluster":
        """Update cluster configuration and optionally install libraries.

        Args:
            libraries: Optional libraries to install.
            access_control_list: List of permissions
            wait: waiting timeout until done, if None it does not wait
            **cluster_spec: Cluster specification overrides.

        Returns:
            The updated Cluster instance.
        """
        self.install_libraries(libraries=libraries, wait_timeout=None, raise_error=False)

        existing_details = {
            k: v
            for k, v in self.details.as_shallow_dict().items()
            if k in _EDIT_ARG_NAMES
        }

        update_details = {
            k: v
            for k, v in self._check_details(details=self.details, **cluster_spec).as_shallow_dict().items()
            if k in _EDIT_ARG_NAMES
        }

        same = dicts_equal(
            existing_details,
            update_details,
            keys=_EDIT_ARG_NAMES,
        )

        if not same:
            LOGGER.debug(
                "Updating %s with %s",
                self, update_details
            )

            self.wait_for_status(wait=wait)
            self.clusters_client().edit(**update_details)
            self.update_permissions(access_control_list=access_control_list)

            LOGGER.info(
                "Updated %s",
                self
            )

            self.wait_for_status(wait=wait)

        return self

    def update_permissions(
        self,
        access_control_list: Optional[List["ClusterAccessControlRequest"] | bool] = True,
    ):
        if not access_control_list:
            return self

        access_control_list = self._check_permission(access_control_list)

        self.clusters_client().update_permissions(
            cluster_id=self.cluster_id,
            access_control_list=access_control_list
        )

    def default_permissions(self):
        current_groups = self.current_user.groups or []

        return [
            ClusterAccessControlRequest(
                group_name=group.display,
                permission_level=ClusterPermissionLevel.CAN_MANAGE
            )
            for group in current_groups
            if group.display not in {"users"}
        ]

    def _check_permission(
        self,
        permission: Union[str, "ClusterAccessControlRequest", List[Union[str, "ClusterAccessControlRequest"]]],
    ):
        if isinstance(permission, ClusterAccessControlRequest):
            return permission

        if isinstance(permission, str):
            if "@" in permission:
                group_name, user_name = None, permission
            else:
                group_name, user_name = permission, None

            return ClusterAccessControlRequest(
                group_name=group_name,
                user_name=user_name,
                permission_level=ClusterPermissionLevel.CAN_MANAGE
            )

        defaults = self.default_permissions()

        if isinstance(permission, bool):
            return defaults if permission else []

        return defaults + [
            self._check_permission(_)
            for _ in permission
        ]

    def list_clusters(self) -> Iterator["Cluster"]:
        """Iterate clusters, yielding helpers annotated with metadata.

        Returns:
            An iterator of Cluster helpers for each cluster.
        """

        for details in self.clusters_client().list():
            details: ClusterDetails = details

            yield Cluster(
                workspace=self.workspace,
                cluster_id=details.cluster_id,
                cluster_name=details.cluster_name,
                _details=details
            )

    def find_cluster(
        self,
        cluster_id: Optional[str] = None,
        *,
        cluster_name: Optional[str] = None,
        raise_error: Optional[bool] = None
    ) -> Optional["Cluster"]:
        """Find a cluster by name or id and return a populated helper.

        Args:
            cluster_id: Optional cluster id to look up.
            cluster_name: Optional cluster name to look up.
            raise_error: Whether to raise if not found.

        Returns:
            A Cluster instance if found, otherwise None.
        """
        if not cluster_name and not cluster_id:
            raise ValueError("Either name or cluster_id must be provided")

        if not cluster_id:
            cluster_id = get_cached_cluster_id(
                host=self.workspace.safe_host,
                cluster_name=cluster_name
            )

        if cluster_id:
            try:
                details = self.clusters_client().get(cluster_id=cluster_id)
            except ResourceDoesNotExist:
                if raise_error:
                    raise ValueError(f"Cannot find databricks cluster {cluster_id!r}")
                return None

            return Cluster(
                workspace=self.workspace,
                cluster_id=details.cluster_id,
                cluster_name=details.cluster_name,
                _details=details
            )

        for cluster in self.list_clusters():
            if cluster_name == cluster.details.cluster_name:
                set_cached_cluster_name(
                    host=self.workspace.safe_host,
                    cluster_name=cluster.cluster_name,
                    cluster_id=cluster.cluster_id
                )
                return cluster

        if raise_error:
            raise ValueError(f"Cannot find databricks cluster {cluster_name!r}")
        return None

    def ensure_running(
        self,
        wait: Optional[WaitingConfigArg] = True
    ) -> "Cluster":
        """Ensure the cluster is running.

        Returns:
            The current Cluster instance.
        """
        return self.start(wait=wait)

    def start(
        self,
        wait: Optional[WaitingConfigArg] = True
    ) -> "Cluster":
        """Start the cluster if it is not already running.

        Returns:
            The current Cluster instance.
        """
        if self.is_running:
            return self

        client = self.clusters_client()
        wait = WaitingConfig.check_arg(wait)

        LOGGER.debug("Starting %s", self)

        try:
            client.start(cluster_id=self.cluster_id)
        except:
            self.wait_for_status(wait=wait)

            if self.is_running:
                return self

            client.start(cluster_id=self.cluster_id)

        LOGGER.info("Started %s", self)

        self.wait_for_status(wait=wait)

        return self

    def restart(
        self,
        wait: Optional[WaitingConfigArg] = True
    ):
        """Restart the cluster, waiting for libraries to install.

        Returns:
            The current Cluster instance.
        """
        self.wait_for_status()

        if self.is_running:
            self.details = self.clusters_client().restart_and_wait(cluster_id=self.cluster_id)
            return self

        return self.start(wait=wait)

    def delete(
        self
    ):
        """Delete the cluster.

        Returns:
            The SDK delete response.
        """
        if self.cluster_id:
            LOGGER.debug("Deleting %s", self)
            self.clusters_client().delete(cluster_id=self.cluster_id)
            LOGGER.info("Deleted %s", self)

    def context(
        self,
        language: Optional["Language"] = None,
        context_id: Optional[str] = None
    ) -> ExecutionContext:
        """Create a command execution context for this cluster.

        Args:
            language: Optional language for the execution context.
            context_id: Optional existing context id to reuse.

        Returns:
            An ExecutionContext instance.
        """
        return ExecutionContext(
            cluster=self,
            language=language,
            context_id=context_id
        )

    def install_libraries(
        self,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        wait_timeout: Optional[dt.timedelta] = dt.timedelta(minutes=20),
        pip_settings: Optional[PipIndexSettings] = None,
        raise_error: bool = True,
    ) -> "Cluster":
        """Install libraries on the cluster and optionally wait for completion.

        Args:
            libraries: Libraries or package names to install.
            wait_timeout: Optional timeout for installation.
            pip_settings: Optional pip index settings.
            raise_error: Whether to raise on install failure.

        Returns:
            The current Cluster instance.
        """
        if not libraries:
            return self

        wsdk = self.workspace.sdk()

        libraries = [
            self._check_library(_, pip_settings=pip_settings)
            for _ in libraries if _
        ]

        if libraries:
            existing = [
                _.library for _ in self.installed_library_statuses()
            ]

            libraries = [
                _
                for _ in libraries
                if _ not in existing
            ]

        if libraries:
            wsdk.libraries.install(
                cluster_id=self.cluster_id,
                libraries=[
                    self._check_library(_, pip_settings=pip_settings)
                    for _ in libraries if _
                ]
            )

            if wait_timeout is not None:
                self.wait_installed_libraries(
                    timeout=wait_timeout,
                    pip_settings=pip_settings,
                    raise_error=raise_error
                )

        return self

    def installed_library_statuses(self):
        """Return current library install statuses for the cluster.

        Returns:
            An iterator of library install status objects.
        """
        return self.workspace.sdk().libraries.cluster_status(cluster_id=self.cluster_id)

    def uninstall_libraries(
        self,
        pypi_packages: Optional[list[str]] = None,
        libraries: Optional[list["Library"]] = None,
        restart: bool = True
    ):
        """Uninstall libraries from the cluster and optionally restart.

        Args:
            pypi_packages: Optional list of PyPI package names to uninstall.
            libraries: Optional list of Library objects to uninstall.
            restart: Whether to restart the cluster afterward.

        Returns:
            The current Cluster instance.
        """
        if libraries is None:
            to_remove = [
                lib.library
                for lib in self.installed_library_statuses()
                if self._filter_lib(
                    lib,
                    pypi_packages=pypi_packages,
                    default_filter=False
                )
            ]
        else:
            to_remove = libraries

        if to_remove:
            self.workspace.sdk().libraries.uninstall(
                cluster_id=self.cluster_id,
                libraries=to_remove
            )

            if restart:
                self.restart()

        return self

    @staticmethod
    def _filter_lib(
        lib: Optional["Library"],
        pypi_packages: Optional[list[str]] = None,
        default_filter: bool = False
    ):
        if lib is None:
            return False

        if lib.pypi:
            if lib.pypi.package and pypi_packages:
                return lib.pypi.package in pypi_packages

        return default_filter

    def wait_installed_libraries(
        self,
        timeout: dt.timedelta = dt.timedelta(minutes=20),
        pip_settings: Optional[PipIndexSettings] = None,
        raise_error: bool = True,
    ):
        """Wait for library installations to finish on the cluster.

        Args:
            timeout: Maximum time to wait for installs.
            pip_settings: Optional pip index settings.
            raise_error: Whether to raise on failures.

        Returns:
            The current Cluster instance.
        """
        if not self.is_running:
            return self

        statuses = list(self.installed_library_statuses())

        max_time = time.time() + timeout.total_seconds()

        while True:
            failed = [
                _.library for _ in statuses
                if _.library and _.status == LibraryInstallStatus.FAILED
            ]

            if failed:
                if raise_error:
                    raise DatabricksError("Libraries %s in %s failed to install" % (failed, self))

                LOGGER.exception(
                    "Libraries %s in %s failed to install",
                    failed, self
                )

            running = [
                _ for _ in statuses if _.status in (
                    LibraryInstallStatus.INSTALLING, LibraryInstallStatus.PENDING,
                    LibraryInstallStatus.RESOLVING
                )
            ]

            if not running:
                break

            if time.time() > max_time:
                raise TimeoutError(
                    "Waiting %s to install libraries timed out" % self
                )

            time.sleep(5)
            statuses = list(self.installed_library_statuses())

        return self

    def install_temporary_libraries(
        self,
        libraries: str | ModuleType | List[str | ModuleType],
    ):
        """Upload local libraries to the cluster's site-packages.

        Args:
            libraries: Library path, name, module, or iterable of these.

        Returns:
            The uploaded library argument(s).
        """
        return self.system_context.install_temporary_libraries(libraries=libraries)

    def _check_library(
        self,
        value,
        pip_settings: Optional[PipIndexSettings] = None,
    ) -> "Library":
        if isinstance(value, Library):
            return value

        pip_settings = PipIndexSettings.default_settings() if pip_settings is None else pip_settings

        if isinstance(value, str):
            if os.path.exists(value):
                target_path = self.workspace.shared_cache_path(
                    suffix=f"/clusters/{self.cluster_id}/{os.path.basename(value)}"
                )

                with open(value, mode="rb") as f:
                    target_path.open().write_all_bytes(f.read())

                value = str(target_path)
            elif "." in value and not "/" in value:
                value = value.split(".")[0]

            # Now value is either a dbfs:/ path or plain package name
            if value.endswith(".jar"):
                return Library(jar=value)
            elif value.endswith("requirements.txt"):
                return Library(requirements=value)
            elif value.endswith(".whl"):
                return Library(whl=value)

            repo = None

            if pip_settings.extra_index_url and (
                value.startswith("datamanagement")
                or value.startswith("TSSecrets")
                or value.startswith("TSMails")
                or value.startswith("tgp_")
                or value.startswith("wma-data")
            ):
                repo = pip_settings.extra_index_url

            return Library(
                pypi=PythonPyPiLibrary(
                    package=value,
                    repo=repo,
                )
            )

        raise ValueError(f"Cannot build Library object from {type(value)}")
