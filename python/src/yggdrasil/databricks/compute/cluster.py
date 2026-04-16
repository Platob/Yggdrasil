"""
Cluster management helpers for Databricks compute.

This module exposes a lightweight ``Cluster`` wrapper around the Databricks
SDK to simplify common cluster lifecycle operations, permission updates,
execution context creation, and library management.

The class keeps a cached ``ClusterDetails`` object and provides convenience
helpers for:

- resolving clusters by name
- starting / restarting / deleting clusters
- waiting for cluster and library readiness
- updating cluster configuration
- managing execution contexts and commands
- installing / uninstalling libraries

Notes
-----
- Custom metadata handling is expected to be managed through Databricks tags
  by the surrounding service layer.
- This wrapper intentionally stays close to SDK behavior while providing
  cleaner ergonomics for application code.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, TYPE_CHECKING, Union

from databricks.sdk import ClustersAPI
from databricks.sdk.client_types import ClientType
from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import ResourceDoesNotExist
from databricks.sdk.service._internal import Wait
from databricks.sdk.service.compute import (
    ClusterAccessControlRequest,
    ClusterDetails,
    ClusterPermissionLevel,
    Language,
    Library,
    LibraryInstallStatus,
    PythonPyPiLibrary,
    State,
)

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.environ.pip_settings import PipIndexSettings
from yggdrasil.io.headers import DEFAULT_HOSTNAME
from yggdrasil.io.url import URL
from yggdrasil.pyutils.equality import dicts_equal
from yggdrasil.version import VersionInfo

from .execution_context import ExecutionContext
from .service import Clusters, PYTHON_BY_DBR
from ..client import DatabricksResource

if TYPE_CHECKING:
    from .command_execution import CommandExecution


__all__ = ["Cluster"]


LOGGER = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_EDIT_ARG_NAMES = set(inspect.signature(ClustersAPI.edit).parameters.keys())
_GROUPNAME_RE = re.compile(r"\bGroupName\((?P<group>[^)]*)\)")


def _library_sig(lib: Library) -> tuple[str, Any]:
    """
    Return a stable-ish signature for a library definition.

    This is used to deduplicate install requests against already installed
    cluster libraries.
    """
    if getattr(lib, "jar", None):
        return ("jar", lib.jar)
    if getattr(lib, "whl", None):
        return ("whl", lib.whl)
    if getattr(lib, "requirements", None):
        return ("requirements", lib.requirements)
    if getattr(lib, "pypi", None):
        return ("pypi", lib.pypi.package, lib.pypi.repo)
    return ("other", repr(lib))


# ---------------------------------------------------------------------------
# PyPI install guard
# ---------------------------------------------------------------------------

#: Packages that must never be overwritten by a pip install on a running
#: Databricks cluster.  Doing so risks destabilising the Spark/ML runtime.
PIP_INSTALL_BLACKLIST: frozenset[str] = frozenset({
    "daft",
    "flask",
    "fastapi",
    "dash",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "pyspark",
})


def _normalize_pip_pkg_name(spec: str) -> str:
    """
    Extract the bare, normalised package name from a pip requirement spec.

    Strips version constraints, normalises underscores to hyphens and
    lowercases so the result can be compared against :data:`PIP_INSTALL_BLACKLIST`.

    Examples::

        "polars~=0.19"  -> "polars"
        "PyArrow>=12.0" -> "pyarrow"
        "scikit_learn"  -> "scikit-learn"
        "my-pkg==1.2.3" -> "my-pkg"
    """
    spec = str(spec).strip().lower()
    # Match PEP 508 package name: starts with alnum, then alnum/._-
    match = re.match(r"^([a-z0-9]([a-z0-9._-]*[a-z0-9])?)", spec)
    name = match.group(1) if match else ""
    return name.replace("_", "-")


@dataclass
class Cluster(DatabricksResource):
    """
    High-level Databricks cluster helper.

    Parameters
    ----------
    service
        Cluster service wrapper used to resolve and validate cluster specs.
    cluster_id
        Existing Databricks cluster id.
    cluster_name
        Existing cluster name. If provided without ``cluster_id``, the cluster
        is resolved on initialization.

    Notes
    -----
    ``Cluster`` caches ``ClusterDetails`` and refreshes them lazily.
    """

    service: Clusters = dataclasses.field(
        default_factory=Clusters.current,
        repr=False,
        compare=False,
    )
    cluster_id: str | None = None
    cluster_name: str | None = None

    _details: Optional[ClusterDetails] = dataclasses.field(default=None, repr=False, hash=False, compare=False)
    _details_refresh_time: float = dataclasses.field(default=0.0, repr=False, hash=False, compare=False)
    _contexts: dict[str, ExecutionContext] = dataclasses.field(default_factory=dict, repr=False, hash=False, compare=False)

    # ------------------------------------------------------------------ #
    # Construction and identity
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        super().__post_init__()

        if self.cluster_name and not self.cluster_id:
            found = self.service.find_cluster(
                cluster_name=self.cluster_name,
                raise_error=True,
            )
            object.__setattr__(self, "cluster_id", found.cluster_id)
            object.__setattr__(self, "_details", found._details)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(url={self.url()})"

    def __str__(self) -> str:
        return self.url().to_string()

    def url(self) -> URL:
        """Return the Databricks workspace URL for this cluster."""
        return URL.parse_str(
            f"{self.client.base_url.to_string().rstrip('/')}/compute/clusters/{self.cluster_id or 'unknown'}"
        )

    # ------------------------------------------------------------------ #
    # SDK clients
    # ------------------------------------------------------------------ #
    def clusters_client(self) -> ClustersAPI:
        """Return the Databricks SDK clusters client."""
        return self.client.workspace_client().clusters

    def libraries_client(self):
        """Return the Databricks SDK libraries client."""
        return self.client.workspace_client().libraries

    # ------------------------------------------------------------------ #
    # Details caching and state
    # ------------------------------------------------------------------ #
    @property
    def details(self) -> Optional[ClusterDetails]:
        """
        Return cached cluster details.

        If no details are cached and the cluster id is known, details are
        fetched lazily from the Databricks API.
        """
        if self._details is None and self.cluster_id is not None:
            self.set_details(self.clusters_client().get(cluster_id=self.cluster_id))
        return self._details

    def fresh_details(self, max_delay: float | None = None) -> Optional[ClusterDetails]:
        """
        Return cluster details, refreshing cache when stale.

        Parameters
        ----------
        max_delay
            Maximum allowed age of cached details in seconds. ``None`` means
            refresh immediately.
        """
        max_delay = 0.0 if max_delay is None else float(max_delay)
        age = time.time() - float(self._details_refresh_time)

        if self.cluster_id and age > max_delay:
            self.set_details(self.clusters_client().get(cluster_id=self.cluster_id))

        return self._details

    def refresh(self, max_delay: float | None = None) -> "Cluster":
        """Refresh cached cluster details and return self."""
        self.set_details(self.fresh_details(max_delay=max_delay))
        return self

    def set_details(self, details: Optional[ClusterDetails | Wait]) -> "Cluster":
        """
        Update local cached details.

        Accepts either a ``ClusterDetails`` object, a Databricks ``Wait`` handle,
        or ``None``.
        """
        if isinstance(details, ClusterDetails):
            object.__setattr__(self, "_details_refresh_time", time.time())
            object.__setattr__(self, "_details", details)
            object.__setattr__(self, "cluster_id", details.cluster_id)
            object.__setattr__(self, "cluster_name", details.cluster_name)
            return self

        if isinstance(details, Wait):
            resolved = details.result()
            object.__setattr__(self, "_details_refresh_time", time.time())
            object.__setattr__(self, "_details", resolved)
            object.__setattr__(self, "cluster_id", details.cluster_id)

            if self.cluster_id and details.cluster_id != self.cluster_id:
                object.__setattr__(self, "cluster_name", None)

            return self

        object.__setattr__(self, "_details_refresh_time", 0.0)
        object.__setattr__(self, "_details", None)
        return self

    @property
    def state(self) -> State:
        """Return the latest cluster state."""
        self.refresh()
        return self._details.state if self._details is not None else State.UNKNOWN

    @property
    def is_running(self) -> bool:
        """Whether the cluster is currently running."""
        return self.state == State.RUNNING

    @property
    def is_pending(self) -> bool:
        """Whether the cluster is in a transitional state."""
        return self.state in {
            State.PENDING,
            State.RESIZING,
            State.RESTARTING,
            State.TERMINATING,
        }

    @property
    def is_error(self) -> bool:
        """Whether the cluster is in an error state."""
        return self.state == State.ERROR

    def raise_for_status(self) -> "Cluster":
        """Raise ``DatabricksError`` if cluster is in error state."""
        if self.is_error:
            raise DatabricksError(f"Error in {self}")
        return self

    def wait_for_status(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Cluster":
        """
        Wait until the cluster leaves its pending state.

        Also waits for library installation completion after the cluster becomes
        stable.
        """
        wait = WaitingConfig.check_arg(wait)
        if not wait:
            return self

        iteration = 0
        start = time.time()

        while self.is_pending:
            wait.sleep(iteration=iteration, start=start)
            iteration += 1

        self.wait_installed_libraries(wait=wait, raise_error=raise_error)

        if raise_error:
            self.raise_for_status()

        return self

    # ------------------------------------------------------------------ #
    # Runtime helpers
    # ------------------------------------------------------------------ #
    @property
    def spark_version(self) -> Optional[str]:
        """Return the configured Databricks runtime string."""
        details = self.details
        return None if details is None else details.spark_version

    @property
    def runtime_version(self) -> Optional[str]:
        """
        Return the major.minor Databricks runtime version.

        Example
        -------
        ``14.3.x-scala2.12`` -> ``14.3``
        """
        spark_version = self.spark_version
        if not spark_version:
            return None

        parts = spark_version.split(".")
        if len(parts) < 2:
            return None

        return ".".join(parts[:2])

    @property
    def python_version_info(self) -> Optional[VersionInfo]:
        """Return the Python version mapped from the Databricks runtime."""
        runtime_version = self.runtime_version
        if not runtime_version:
            return None
        return PYTHON_BY_DBR.get(runtime_version)

    # ------------------------------------------------------------------ #
    # Cluster updates and permissions
    # ------------------------------------------------------------------ #
    def update(
        self,
        *,
        single_user_name: str | None = None,
        libraries: Optional[list[Union[str, Library]]] = None,
        permissions: Optional[list[str | ClusterAccessControlRequest]] = None,
        wait: WaitingConfigArg = True,
        **cluster_spec: Any,
    ) -> "Cluster":
        """
        Update cluster configuration.

        Parameters
        ----------
        single_user_name
            Optional Databricks single-user assignment.
        libraries
            Libraries to install before / alongside update flow.
        permissions
            Cluster ACL entries. Strings are interpreted as:
            - email -> user permission
            - other string -> group permission
        wait
            Waiting behavior for cluster and library readiness.
        **cluster_spec
            Cluster edit spec supported by the service layer and Databricks SDK.
        """
        self.install_libraries(libraries=libraries, wait=False, raise_error=False)

        current = self._editable_details_from(self.details)
        desired = self._editable_details_from(
            self.service.check_details(
                update=True,
                details=self.details,
                single_user_name=single_user_name,
                **cluster_spec,
            )
        )

        if dicts_equal(current, desired, keys=_EDIT_ARG_NAMES):
            return self

        LOGGER.debug("Updating %s with %s", self, desired)

        self.wait_for_status(wait=wait)
        self.clusters_client().edit(**desired)
        self.update_permissions(permissions=permissions)

        LOGGER.info("Updated %s", self)
        self.wait_for_status(wait=wait)
        return self

    def update_permissions(
        self,
        permissions: Optional[list[str | ClusterAccessControlRequest]] = None,
    ) -> "Cluster":
        """
        Update cluster permissions.

        If a referenced group does not exist and Databricks reports it through
        a ``GroupName(...)`` error, the group is created and the operation is
        retried once.
        """
        if not permissions:
            return self

        checked_permissions = self._check_permission(permissions)

        try:
            self.clusters_client().update_permissions(
                cluster_id=self.cluster_id,
                access_control_list=checked_permissions,
            )
        except ResourceDoesNotExist as exc:
            match = _GROUPNAME_RE.search(str(exc))
            group_name = match.group("group") if match else None

            if not group_name:
                raise

            try:
                self.client.iam.groups.create(
                    name=group_name,
                    members=[self.client.iam.users.current_user],
                    client_type=ClientType.ACCOUNT,
                )
            except Exception as inner_exc:
                raise inner_exc from exc

            return self.update_permissions(permissions)

        return self

    def _check_permission(
        self,
        permission: Union[str, ClusterAccessControlRequest, list[Union[str, ClusterAccessControlRequest]]],
    ):
        """
        Normalize permission input into Databricks ACL request objects.
        """
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
                permission_level=ClusterPermissionLevel.CAN_MANAGE,
            )

        return [self._check_permission(item) for item in permission]

    @staticmethod
    def _editable_details_from(details: Optional[ClusterDetails]) -> dict[str, Any]:
        """
        Extract editable fields from cluster details for Databricks ``edit``.
        """
        if details is None:
            return {}
        return {
            key: value
            for key, value in details.as_shallow_dict().items()
            if key in _EDIT_ARG_NAMES
        }

    # ------------------------------------------------------------------ #
    # Lifecycle operations
    # ------------------------------------------------------------------ #
    def ensure_running(self, wait: WaitingConfigArg = True) -> "Cluster":
        """Ensure the cluster is running."""
        return self.start(wait=wait)

    def start(self, wait: WaitingConfigArg = True) -> "Cluster":
        """
        Start the cluster if needed.

        If the initial start call races with a transient state transition,
        the method waits once and retries.
        """
        if self.is_running:
            return self

        client = self.clusters_client()
        wait = WaitingConfig.check_arg(wait)

        LOGGER.debug("Starting %s", self)

        try:
            client.start(cluster_id=self.cluster_id)
        except DatabricksError:
            self.wait_for_status(wait=wait)
            if self.is_running:
                return self
            client.start(cluster_id=self.cluster_id)

        LOGGER.info("Started %s", self)
        self.wait_for_status(wait=wait)
        return self

    def restart(self, wait: WaitingConfigArg = True) -> "Cluster":
        """
        Restart the cluster if already running, otherwise start it.
        """
        self.wait_for_status(wait=wait)

        if self.is_running:
            self.set_details(self.clusters_client().restart_and_wait(cluster_id=self.cluster_id))
            return self

        return self.start(wait=wait)

    def delete(self) -> None:
        """Delete the cluster if it exists."""
        if not self.cluster_id:
            return

        LOGGER.debug("Deleting %s", self)
        self.clusters_client().delete(cluster_id=self.cluster_id)
        LOGGER.info("Deleted %s", self)

    # ------------------------------------------------------------------ #
    # Execution contexts and commands
    # ------------------------------------------------------------------ #
    def context(
        self,
        *,
        language: Optional[Language] = None,
        context_id: str | None = None,
        context_key: str | None = None,
    ) -> ExecutionContext:
        """
        Return an execution context for this cluster.

        When ``context_key`` is provided, the context is cached and reused.
        """
        if context_key:
            existing = self._contexts.get(context_key)
            if existing is None:
                existing = ExecutionContext(
                    cluster=self,
                    language=language,
                    context_id=context_id,
                    context_key=context_key,
                )
                self._contexts[context_key] = existing
            return existing

        return ExecutionContext(
            cluster=self,
            language=language,
            context_id=context_id,
            context_key=context_key,
        )

    def command(
        self,
        context: Optional[ExecutionContext | str] = None,
        command: Optional[str | Callable] = None,
        *,
        command_str: str | None = None,
        func: Optional[Callable] = None,
        language: Optional[Language] = None,
        command_id: str | None = None,
        environ: Optional[Union[Iterable[str], Dict[str, str]]] = None,
    ) -> "CommandExecution":
        """
        Create a command execution bound to a reusable execution context.
        """
        if context is None:
            context = self.context(language=language, context_key=DEFAULT_HOSTNAME)
        elif not isinstance(context, ExecutionContext):
            if isinstance(context, str):
                context = self.context(language=language, context_key=context)
            else:
                raise ValueError(
                    f"Invalid context type: {type(context)}"
                )

        return context.command(
            command=command,
            command_str=command_str,
            command_id=command_id,
            environ=environ,
            func=func,
            language=language,
            context=context,
        )

    # ------------------------------------------------------------------ #
    # Library management
    # ------------------------------------------------------------------ #
    def install_libraries(
        self,
        libraries: Optional[List[Union[str, Library]]] = None,
        wait: WaitingConfigArg = True,
        pip_settings: Optional[PipIndexSettings] = None,
        remove_failed: bool = True,
        raise_error: bool = True,
    ) -> "Cluster":
        """
        Install libraries on the cluster.

        String inputs are normalized as:
        - ``*.jar`` -> jar library
        - ``*.whl`` -> wheel library
        - ``*requirements.txt`` -> requirements file
        - otherwise -> PyPI package

        PyPI packages whose bare name appears in :data:`PIP_INSTALL_BLACKLIST`
        are silently skipped to protect the Databricks runtime from
        destabilising overwrites (e.g. ``pyspark``, ``tensorflow``).
        """
        if not libraries:
            return self

        pip_settings = PipIndexSettings.current() if pip_settings is None else pip_settings

        normalized = [
            self._check_library(item, pip_settings=pip_settings)
            for item in libraries
            if item
        ]

        # Drop PyPI packages that are on the runtime-protection blacklist.
        allowed: list[Library] = []
        skipped: list[str] = []
        for lib in normalized:
            pkg = getattr(lib.pypi, "package", None) if getattr(lib, "pypi", None) else None
            if pkg and _normalize_pip_pkg_name(pkg) in PIP_INSTALL_BLACKLIST:
                skipped.append(pkg)
            else:
                allowed.append(lib)

        if skipped:
            LOGGER.debug(
                "install_libraries: skipping blacklisted runtime package(s): %s",
                ", ".join(skipped),
            )

        normalized = self._dedupe_uninstalled_libraries(allowed)

        if not normalized:
            return self

        self.libraries_client().install(cluster_id=self.cluster_id, libraries=normalized)
        self.wait_installed_libraries(
            wait=wait,
            raise_error=raise_error,
            remove_failed=remove_failed,
        )
        return self

    def installed_library_statuses(self):
        """Return Databricks library installation statuses for this cluster."""
        return self.libraries_client().cluster_status(cluster_id=self.cluster_id)

    def _uninstall_libraries(self, libraries: list[Library]) -> None:
        """Uninstall the provided libraries from the cluster."""
        if not libraries:
            return

        self.libraries_client().uninstall(
            cluster_id=self.cluster_id,
            libraries=libraries,
        )

    def uninstall_libraries(
        self,
        pypi_packages: Optional[list[str]] = None,
        libraries: Optional[list[Library]] = None,
        restart: bool = True,
    ) -> "Cluster":
        """
        Uninstall libraries from the cluster.

        Parameters
        ----------
        pypi_packages
            Optional list of PyPI package names to remove.
        libraries
            Explicit library objects to uninstall. If provided, this takes
            precedence over ``pypi_packages`` filtering.
        restart
            Whether to restart the cluster after uninstalling.
        """
        if libraries is None:
            to_remove = [
                status.library
                for status in self.installed_library_statuses()
                if status.library is not None
                and self._filter_lib(
                    status.library,
                    pypi_packages=pypi_packages,
                    default_filter=False,
                )
            ]
        else:
            to_remove = libraries

        if to_remove:
            self._uninstall_libraries(to_remove)
            if restart:
                self.restart()

        return self

    @staticmethod
    def _filter_lib(
        lib: Optional[Library],
        pypi_packages: Optional[list[str]] = None,
        default_filter: bool = False,
    ) -> bool:
        """
        Return whether a library matches uninstall filter criteria.
        """
        if lib is None:
            return False

        if lib.pypi and lib.pypi.package and pypi_packages:
            return lib.pypi.package in pypi_packages

        return default_filter

    def wait_installed_libraries(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        remove_failed: bool = True,
    ) -> "Cluster":
        """
        Wait until all cluster libraries finish resolving / installing.

        Failed libraries can optionally be uninstalled automatically.
        """
        if not self.is_running:
            return self

        wait = WaitingConfig.check_arg(wait)
        if not wait:
            return self

        statuses = list(self.installed_library_statuses())
        start = time.time()
        iteration = 0

        while True:
            failed = [
                status.library
                for status in statuses
                if status.library and status.status == LibraryInstallStatus.FAILED
            ]

            if failed:
                if remove_failed:
                    try:
                        self._uninstall_libraries(failed)
                    except Exception:
                        LOGGER.exception(
                            "Failed to uninstall broken libraries %s from %s",
                            failed,
                            self,
                        )

                message = f"Libraries {failed} in {self} failed to install"

                if raise_error:
                    raise DatabricksError(message)

                LOGGER.error(message)
                return self

            running = [
                status
                for status in statuses
                if status.status in {
                    LibraryInstallStatus.INSTALLING,
                    LibraryInstallStatus.PENDING,
                    LibraryInstallStatus.RESOLVING,
                }
            ]

            if not running:
                return self

            wait.sleep(iteration=iteration, start=start)
            iteration += 1
            statuses = list(self.installed_library_statuses())

    def _dedupe_uninstalled_libraries(self, libraries: list[Library]) -> list[Library]:
        """
        Remove libraries that are already present on the cluster.
        """
        existing_sigs = {
            _library_sig(status.library)
            for status in self.installed_library_statuses()
            if getattr(status, "library", None) is not None
        }
        return [lib for lib in libraries if _library_sig(lib) not in existing_sigs]

    @staticmethod
    def _check_library(
        value: str | Library,
        pip_settings: Optional[PipIndexSettings] = None,
    ) -> Library:
        """
        Normalize a library input into a Databricks ``Library`` object.

        Private package prefixes are routed to the configured extra index URL
        when no explicit PyPI repo is already set.
        """
        if isinstance(value, Library):
            library = value
        elif isinstance(value, str):
            if value.endswith(".jar"):
                library = Library(jar=value)
            elif value.endswith(".whl"):
                library = Library(whl=value)
            elif value.endswith("requirements.txt"):
                library = Library(requirements=value)
            else:
                library = Library(pypi=PythonPyPiLibrary(package=value))
        else:
            raise ValueError(f"Cannot build Library object from {type(value)}")

        pip_settings = PipIndexSettings.current() if pip_settings is None else pip_settings
        package = getattr(library.pypi, "package", None)

        if (
            package
            and library.pypi.repo is None
            and pip_settings.extra_index_url
            and package.startswith(
                (
                    "datamanagement",
                    "TSSecrets",
                    "TSMails",
                    "tgp_",
                    "wma-data",
                )
            )
        ):
            library.pypi.repo = pip_settings.extra_index_url

        return library