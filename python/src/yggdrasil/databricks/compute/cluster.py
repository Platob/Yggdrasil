"""
Cluster management helpers for Databricks compute.

This module provides a lightweight ``Cluster`` helper that wraps the
Databricks SDK to simplify common CRUD operations and metadata handling
for clusters. Metadata is stored in custom tags prefixed with
``yggdrasil:``.
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


def _library_sig(lib: Library) -> tuple:
    # stable-ish signature for dedupe
    if getattr(lib, "jar", None):
        return "jar", lib.jar
    if getattr(lib, "whl", None):
        return "whl", lib.whl
    if getattr(lib, "requirements", None):
        return "req", lib.requirements
    if getattr(lib, "pypi", None) and lib.pypi:
        return "pypi", lib.pypi.package, lib.pypi.repo
    return "other", repr(lib)


@dataclass
class Cluster(DatabricksResource):
    service: Clusters = dataclasses.field(
        default_factory=Clusters.current,
        repr=False,
        compare=False,
    )
    cluster_id: Optional[str] = None
    cluster_name: Optional[str] = None

    _details: Optional[ClusterDetails] = dataclasses.field(default=None, repr=False, hash=False, compare=False)
    _details_refresh_time: float = dataclasses.field(default=0.0, repr=False, hash=False, compare=False)

    def __post_init__(self):
        super().__post_init__()

        if self.cluster_name and not self.cluster_id:
            found = self.service.find_cluster(
                cluster_name=self.cluster_name,
                raise_error=True,
            )

            object.__setattr__(self, "cluster_id", found.cluster_id)
            object.__setattr__(self, "_details", found._details)

    def __repr__(self):
        return "%s<url='%s'>" % (self.__class__.__name__, self.url())

    def __str__(self):
        return self.url().to_string()

    def url(self) -> URL:
        return URL.parse_str(
            "%s/compute/clusters/%s"
            % (
                self.client.base_url.to_string().rstrip("/"),
                self.cluster_id or "unknown",
            )
        )

    # ------------------------------------------------------------------ #
    # Details caching
    # ------------------------------------------------------------------ #
    @property
    def details(self) -> Optional[ClusterDetails]:
        if self._details is None and self.cluster_id is not None:
            self.set_details(self.clusters_client().get(cluster_id=self.cluster_id))
        return self._details

    def fresh_details(self, max_delay: float | None = None) -> Optional[ClusterDetails]:
        max_delay = 0.0 if max_delay is None else float(max_delay)
        delay = time.time() - float(self._details_refresh_time)

        if self.cluster_id and delay > max_delay:
            self.set_details(self.clusters_client().get(cluster_id=self.cluster_id))
        return self._details

    def refresh(self, max_delay: float | None = None):
        self.set_details(self.fresh_details(max_delay=max_delay))
        return self

    def set_details(self, details: Optional[ClusterDetails]):
        if isinstance(details, ClusterDetails):
            object.__setattr__(self, "_details_refresh_time", time.time())
            object.__setattr__(self, "_details", details)
            object.__setattr__(self, "cluster_id", details.cluster_id)
            object.__setattr__(self, "cluster_name", details.cluster_name)
        elif isinstance(details, Wait):
            # allow passing Wait from start/restart_and_wait without forcing an extra get call
            object.__setattr__(self, "_details_refresh_time", time.time())
            object.__setattr__(self, "_details", details.result())
            object.__setattr__(self, "cluster_id", details.cluster_id)

            if self.cluster_id and details.cluster_id != self.cluster_id:
                object.__setattr__(self, "cluster_name", None)
        else:
            object.__setattr__(self, "_details_refresh_time", 0.0)
            object.__setattr__(self, "_details", None)

        return self

    @property
    def state(self):
        self.refresh()
        if self._details is not None:
            return self._details.state
        return State.UNKNOWN

    @property
    def is_running(self):
        return self.state == State.RUNNING

    @property
    def is_pending(self):
        return self.state in (
            State.PENDING,
            State.RESIZING,
            State.RESTARTING,
            State.TERMINATING,
        )

    @property
    def is_error(self):
        return self.state == State.ERROR


    def raise_for_status(self):
        if self.is_error:
            raise DatabricksError("Error in %s" % self)
        return self

    def wait_for_status(self, wait: WaitingConfigArg = True, raise_error: bool = True):
        wait = WaitingConfig.check_arg(wait)
        if wait:
            iteration, start = 0, time.time()
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
        d = self.details
        return None if d is None else d.spark_version

    @property
    def runtime_version(self) -> Optional[str]:
        v = self.spark_version
        if not v:
            return None
        parts = v.split(".")
        if len(parts) < 2:
            return None
        return ".".join(parts[:2])

    @property
    def python_version_info(self) -> Optional[VersionInfo]:
        v = self.runtime_version
        if not v:
            return None
        return PYTHON_BY_DBR.get(v)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def clusters_client(self) -> ClustersAPI:
        return self.client.workspace_client().clusters

    def update(
        self,
        *,
        single_user_name: Optional[str] = None,
        libraries: Optional[list[Union[str, Library]]] = None,
        permissions: Optional[list[str | ClusterAccessControlRequest]] = None,
        wait: WaitingConfigArg = True,
        **cluster_spec: Any,
    ) -> "Cluster":
        self.install_libraries(libraries=libraries, wait=False, raise_error=False)

        existing_details = {
            k: v
            for k, v in self.details.as_shallow_dict().items()
            if k in _EDIT_ARG_NAMES
        }

        update_details = {
            k: v
            for k, v in self.service.check_details(
                update=True,
                details=self.details,
                single_user_name=single_user_name,
                **cluster_spec,
            ).as_shallow_dict().items()
            if k in _EDIT_ARG_NAMES
        }

        same = dicts_equal(existing_details, update_details, keys=_EDIT_ARG_NAMES)

        if not same:
            LOGGER.debug("Updating %s with %s", self, update_details)

            self.wait_for_status(wait=wait)
            self.clusters_client().edit(**update_details)
            self.update_permissions(permissions=permissions)

            LOGGER.info("Updated %s", self)
            self.wait_for_status(wait=wait)

        return self

    def update_permissions(
        self,
        permissions: Optional[list[str | ClusterAccessControlRequest]] = None,
    ):
        if not permissions:
            return self

        permissions = self._check_permission(permissions)

        try:
            self.clusters_client().update_permissions(
                cluster_id=self.cluster_id,
                access_control_list=permissions,
            )
        except ResourceDoesNotExist as e:
            m = _GROUPNAME_RE.search(str(e))
            group_name = m.group("group") if m else None

            if group_name:
                try:
                    self.client.iam.groups.create(
                        name=group_name,
                        members=[self.client.iam.users.current_user],
                        client_type=ClientType.ACCOUNT,
                    )
                except Exception as inner_e:
                    raise inner_e from e
                return self.update_permissions(permissions)
            raise

        return self

    def _check_permission(
        self,
        permission: Union[str, ClusterAccessControlRequest, list[Union[str, ClusterAccessControlRequest]]],
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
                permission_level=ClusterPermissionLevel.CAN_MANAGE,
            )

        return [self._check_permission(_) for _ in permission]

    def ensure_running(self, wait: WaitingConfigArg = True) -> "Cluster":
        return self.start(wait=wait)

    def start(self, wait: WaitingConfigArg = True) -> "Cluster":
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

    def restart(self, wait: WaitingConfigArg = True):
        self.wait_for_status()

        if self.is_running:
            self.set_details(self.clusters_client().restart_and_wait(cluster_id=self.cluster_id))
            return self

        return self.start(wait=wait)

    def delete(self):
        if self.cluster_id:
            LOGGER.debug("Deleting %s", self)
            self.clusters_client().delete(cluster_id=self.cluster_id)
            LOGGER.info("Deleted %s", self)

    # ------------------------------------------------------------------ #
    # Execution contexts
    # ------------------------------------------------------------------ #
    def context(
        self,
        *,
        language: Optional[Language] = None,
        context_key: Optional[str] = None,
        temporary: bool = False,
    ) -> "ExecutionContext":
        """Return a pooled execution context for this cluster."""
        return ExecutionContext.get_or_create(
            cluster=self,
            language=language or Language.PYTHON,
            context_key=context_key,
            temporary=temporary,
        )

    def command(
        self,
        command: Optional[str | Callable] = None,
        *,
        command_str: Optional[str] = None,
        func: Optional[Callable] = None,
        language: Optional[Language] = None,
        command_id: Optional[str] = None,
        environ: Optional[Union[Iterable[str], Dict[str, str]]] = None,
        context_key: Optional[str] = None,
        temporary: bool = False,
    ) -> "CommandExecution":
        ctx = self.context(
            language=language or Language.PYTHON,
            context_key=context_key,
            temporary=temporary,
        )
        return ctx.command(
            command=command,
            command_str=command_str,
            command_id=command_id,
            environ=environ,
            func=func,
            language=language,
            context=ctx,
        )

    def decorate(
        self,
        func: Optional[Callable] = None,
        *,
        environ: Optional[Union[Iterable[str], Dict[str, str]]] = None,
        language: Optional[Language] = None,
        context_key: Optional[str] = None,
    ) -> Callable:
        """
        Return a decorator that executes *func* on this cluster.

        Supports both bare and parameterised forms::

            @cluster.decorate
            def compute(x: int) -> int:
                return x * 2

            @cluster.decorate(environ=["MY_SECRET"])
            def fetch(key: str) -> dict:
                ...
        """
        import functools

        def decorator(fn: Callable) -> Callable:
            cmd = self.command(
                func=fn,
                environ=environ,
                language=language,
                context_key=context_key,
            )

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return cmd(*args, **kwargs)

            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    # ------------------------------------------------------------------ #
    # Libraries
    # ------------------------------------------------------------------ #
    def install_libraries(
        self,
        libraries: Optional[List[Union[str, Library]]] = None,
        wait: WaitingConfigArg = True,
        pip_settings: Optional[PipIndexSettings] = None,
        remove_failed: bool = True,
        raise_error: bool = True,
    ) -> "Cluster":
        if not libraries:
            return self

        wsdk = self.client.workspace_client()
        pip_settings = PipIndexSettings.current() if pip_settings is None else pip_settings

        normalized: list[Library] = [
            self._check_library(_, pip_settings=pip_settings)
            for _ in libraries
            if _
        ]

        if normalized:
            existing_sigs = {
                _library_sig(st.library)
                for st in self.installed_library_statuses()
                if getattr(st, "library", None) is not None
            }
            normalized = [lib for lib in normalized if _library_sig(lib) not in existing_sigs]

        if normalized:
            wsdk.libraries.install(cluster_id=self.cluster_id, libraries=normalized)
            self.wait_installed_libraries(wait=wait, raise_error=raise_error, remove_failed=remove_failed)

        return self

    def installed_library_statuses(self):
        return self.client.workspace_client().libraries.cluster_status(cluster_id=self.cluster_id)

    def _uninstall_libraries(self, libraries: list[Library]) -> None:
        if not libraries:
            return

        self.client.workspace_client().libraries.uninstall(
            cluster_id=self.cluster_id,
            libraries=libraries,
        )

    def uninstall_libraries(
        self,
        pypi_packages: Optional[list[str]] = None,
        libraries: Optional[list[Library]] = None,
        restart: bool = True,
    ):
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
    ):
        if lib is None:
            return False

        if lib.pypi:
            if lib.pypi.package and pypi_packages:
                return lib.pypi.package in pypi_packages

        return default_filter

    def wait_installed_libraries(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        remove_failed: bool = True
    ):
        if not self.is_running:
            return self

        wait = WaitingConfig.check_arg(wait)

        if wait:
            statuses = list(self.installed_library_statuses())
            start, iteration = time.time(), 0

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

                    msg = "Libraries %s in %s failed to install" % (failed, self)

                    if raise_error:
                        raise DatabricksError(msg)

                    LOGGER.error(msg)
                    return self

                running = [
                    status
                    for status in statuses
                    if status.status in (
                        LibraryInstallStatus.INSTALLING,
                        LibraryInstallStatus.PENDING,
                        LibraryInstallStatus.RESOLVING,
                    )
                ]

                if not running:
                    break

                wait.sleep(iteration=iteration, start=start)
                iteration += 1
                statuses = list(self.installed_library_statuses())

        return self

    @staticmethod
    def _check_library(value, pip_settings: Optional[PipIndexSettings] = None) -> Library:
        if isinstance(value, Library):
            library = value
        elif isinstance(value, str):
            if value.endswith(".jar"):
                library = Library(jar=value)
            elif value.endswith("requirements.txt"):
                library = Library(requirements=value)
            elif value.endswith(".whl"):
                library = Library(whl=value)
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