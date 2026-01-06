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
import functools
import inspect
import logging
import os
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Iterator, Optional, Union, List, Callable, Dict, ClassVar

from .execution_context import ExecutionContext
from ..workspaces.workspace import WorkspaceService, Workspace
from ... import retry, CallableSerde
from ...libs.databrickslib import databricks_sdk
from ...pyutils.modules import PipIndexSettings
from ...pyutils.python_env import PythonEnv

if databricks_sdk is None:  # pragma: no cover - import guard
    ResourceDoesNotExist = Exception  # type: ignore
else:  # pragma: no cover - runtime fallback when SDK is missing
    from databricks.sdk import ClustersAPI
    from databricks.sdk.errors import DatabricksError
    from databricks.sdk.errors.platform import ResourceDoesNotExist
    from databricks.sdk.service.compute import (
        ClusterDetails, Language, Kind, State, DataSecurityMode, Library, PythonPyPiLibrary, LibraryInstallStatus
    )
    from databricks.sdk.service.compute import SparkVersion, RuntimeEngine

    _CREATE_ARG_NAMES = {_ for _ in inspect.signature(ClustersAPI.create).parameters.keys()}
    _EDIT_ARG_NAMES = {_ for _ in inspect.signature(ClustersAPI.edit).parameters.keys()}


__all__ = ["Cluster"]


logger = logging.getLogger(__name__)


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

    # host → Cluster instance
    _env_clusters: ClassVar[Dict[str, "Cluster"]] = {}

    @property
    def id(self):
        return self.cluster_id

    @property
    def name(self) -> str:
        return self.cluster_name

    def __post_init__(self):
        if self._details is not None:
            self.details = self._details

    def is_in_databricks_environment(self):
        return self.workspace.is_in_databricks_environment()

    @classmethod
    def replicated_current_environment(
        cls,
        workspace: Optional["Workspace"] = None,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
        single_user_name: Optional[str] = None,
        runtime_engine: Optional["RuntimeEngine"] = None,
        libraries: Optional[list[str]] = None,
        **kwargs
    ) -> "Cluster":
        if workspace is None:
            workspace = Workspace()  # your default, whatever it is

        host = workspace.host

        # 🔥 return existing singleton for this host
        if host in cls._env_clusters:
            return cls._env_clusters[host]

        # 🔥 first time for this host → create
        inst = cls._env_clusters[host] = (
            cls(workspace=workspace, cluster_id=cluster_id, cluster_name=cluster_name)
            .push_python_environment(
                single_user_name=single_user_name,
                runtime_engine=runtime_engine,
                libraries=libraries,
                **kwargs
            )
        )

        return inst

    def push_python_environment(
        self,
        source: Optional[PythonEnv] = None,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
        single_user_name: Optional[str] = "current",
        runtime_engine: Optional["RuntimeEngine"] = None,
        libraries: Optional[list[str]] = None,
        **kwargs
    ) -> "Cluster":
        if source is None:
            source = PythonEnv.get_current()

        libraries = list(libraries) if libraries is not None else []
        libraries.extend([
            _ for _ in [
                "ygg",
                "uv",
            ] if _ not in libraries
        ])

        python_version = source.version_info

        if python_version[0] < 3:
            python_version = None
        elif python_version[1] < 11:
            python_version = None

        current_user_name = self.workspace.current_user.user_name

        if single_user_name == "current":
            single_user_name = current_user_name

        cluster_id = cluster_id or self.cluster_id
        cluster_name = cluster_name or self.cluster_name

        if not cluster_id and not cluster_name:
            cluster_name = current_user_name

        inst = self.create_or_update(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            python_version=python_version,
            single_user_name=single_user_name,
            runtime_engine=runtime_engine or RuntimeEngine.PHOTON,
            libraries=libraries,
            **kwargs
        )

        return inst

    def pull_python_environment(
        self,
        name: Optional[str] = None,
        target: PythonEnv | str | None = None,
    ):
        with self.context() as c:
            m = c.remote_metadata
            version_info = m.version_info

        python_version = ".".join(str(_) for _ in version_info)

        if target is None:
            target = PythonEnv.create(
                name=name or self.name,
                python=python_version
            )
        elif isinstance(target, str):
            if target.casefold() == "current":
                target = PythonEnv.get_current()
            else:
                target = PythonEnv.create(
                    name=target,
                    python=python_version
                )

        target.update(
            python=python_version,
        )

        return target
    
    @property
    def details(self):
        if self._details is None and self.cluster_id is not None:
            self.details = self.clusters_client().get(cluster_id=self.cluster_id)
        return self._details

    def fresh_details(self, max_delay: float | None = None):
        max_delay = max_delay or 0
        delay = time.time() - self._details_refresh_time

        if self.cluster_id and delay > max_delay:
            self.details = self.clusters_client().get(cluster_id=self.cluster_id)
        return self._details

    @details.setter
    def details(self, value: "ClusterDetails"):
        self._details_refresh_time = time.time()
        self._details = value

        self.cluster_id = value.cluster_id
        self.cluster_name = value.cluster_name

    @property
    def state(self):
        details = self.fresh_details(max_delay=10)

        if details is not None:
            return details.state
        return State.UNKNOWN

    def get_state(self, max_delay: float = None):
        details = self.fresh_details(max_delay=max_delay)

        if details is not None:
            return details.state
        return State.UNKNOWN

    @property
    def is_running(self):
        return self.state == State.RUNNING

    @property
    def is_pending(self):
        return self.state in  (State.PENDING, State.RESIZING, State.RESTARTING, State.TERMINATING)

    @property
    def is_error(self):
        return self.state == State.ERROR

    def raise_for_status(self):
        if self.is_error:
            raise DatabricksError("Error in %s" % self)

        return self

    def wait_for_status(
        self,
        tick: float = 0.5,
        timeout: float = 600,
        backoff: int = 2,
        max_sleep_time: float = 15
    ):
        start = time.time()
        sleep_time = tick

        while self.is_pending:
            time.sleep(sleep_time)

            if time.time() - start > timeout:
                raise TimeoutError("Waiting state for %s timed out")

            sleep_time = min(max_sleep_time, sleep_time * backoff)

        self.raise_for_status()

        return self

    @property
    def spark_version(self) -> str:
        d = self.details
        if d is None:
            return None
        return d.spark_version

    @property
    def runtime_version(self):
        # Extract "major.minor" from strings like "17.3.x-scala2.13-ml-gpu"
        v = self.spark_version

        if v is None:
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
        if v is None:
            return None
        return _PYTHON_BY_DBR.get(v)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def clusters_client(self) -> "ClustersAPI":
        return self.workspace.sdk().clusters

    def spark_versions(
        self,
        photon: Optional[bool] = None,
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
    ):
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

        if kwargs:
            details = ClusterDetails(**{
                **details.as_shallow_dict(),
                **kwargs
            })

        if details.custom_tags is None:
            details.custom_tags = self.workspace.default_tags()

        if details.cluster_name is None:
            details.cluster_name = self.workspace.current_user.user_name

        if details.spark_version is None or python_version:
            details.spark_version = self.latest_spark_version(
                photon=False, python_version=python_version
            ).key

        if details.single_user_name:
            if not details.data_security_mode:
                details.data_security_mode = DataSecurityMode.DATA_SECURITY_MODE_DEDICATED

        if not details.node_type_id:
            details.node_type_id = "rd-fleet.xlarge"

        if getattr(details, "virtual_cluster_size", None) is None and details.num_workers is None and details.autoscale is None:
            if details.is_single_node is None:
                details.is_single_node = True

        if details.is_single_node is not None and details.kind is None:
            details.kind = Kind.CLASSIC_PREVIEW

        if pip_settings.extra_index_urls:
            if details.spark_env_vars is None:
                details.spark_env_vars = {}
            str_urls = " ".join(pip_settings.extra_index_urls)
            details.spark_env_vars["UV_EXTRA_INDEX_URL"] = details.spark_env_vars.get("UV_INDEX", str_urls)
            details.spark_env_vars["PIP_EXTRA_INDEX_URL"] = details.spark_env_vars.get("PIP_EXTRA_INDEX_URL", str_urls)

        return details

    def create_or_update(
        self,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        **cluster_spec: Any
    ):
        found = self.find_cluster(
            cluster_id=cluster_id or self.cluster_id,
            cluster_name=cluster_name or self.cluster_name,
            raise_error=False
        )

        if found is not None:
            return found.update(
                cluster_name=cluster_name,
                libraries=libraries,
                **cluster_spec
            )

        return self.create(
            cluster_name=cluster_name,
            libraries=libraries,
            **cluster_spec
        )

    def create(
        self,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        **cluster_spec: Any
    ) -> str:
        cluster_spec["autotermination_minutes"] = int(cluster_spec.get("autotermination_minutes", 30))
        update_details = self._check_details(details=ClusterDetails(), **cluster_spec)
        update_details = {
            k: v
            for k, v in update_details.as_shallow_dict().items()
            if k in _CREATE_ARG_NAMES
        }

        logger.debug(
            "Creating Databricks cluster %s with %s",
            update_details["cluster_name"],
            update_details,
        )

        self.details = self.clusters_client().create_and_wait(**update_details)

        logger.info(
            "Created %s",
            self
        )

        self.install_libraries(libraries=libraries, raise_error=False)

        return self

    def update(
        self,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        **cluster_spec: Any
    ) -> "Cluster":
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

        if update_details != existing_details:
            logger.debug(
                "Updating %s with %s",
                self, update_details
            )

            self.details = self.clusters_client().edit_and_wait(**update_details)

            logger.info(
                "Updated %s",
                self
            )

        return self

    def list_clusters(self) -> Iterator["Cluster"]:
        """Iterate clusters, yielding helpers annotated with metadata."""

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
        """Find a cluster by name or id and return a populated helper."""
        if not cluster_name and not cluster_id:
            raise ValueError("Either name or cluster_id must be provided")

        if cluster_id:
            try:
                details = self.clusters_client().get(cluster_id=cluster_id)
            except ResourceDoesNotExist:
                if raise_error:
                    raise ValueError(f"Cannot find databricks cluster {cluster_id!r}")
                return None

            return Cluster(
                workspace=self.workspace, cluster_id=details.cluster_id, _details=details
            )

        cluster_name_cf = cluster_name.casefold()

        for cluster in self.list_clusters():
            if cluster_name_cf == cluster.details.cluster_name.casefold():
                return cluster

        if raise_error:
            raise ValueError(f"Cannot find databricks cluster {cluster_name!r}")
        return None

    def ensure_running(
        self,
    ) -> "Cluster":
        return self.start()

    def start(
        self,
    ) -> "Cluster":
        self.wait_for_status()

        if not self.is_running:
            logger.info("Starting %s", self)
            self.details = self.clusters_client().start_and_wait(cluster_id=self.cluster_id)
            return self.wait_installed_libraries()

        return self

    @retry(tries=4)
    def restart(
        self,
    ):
        self.wait_for_status()

        if self.is_running:
            logger.info("Restarting %s", self)
            self.details = self.clusters_client().restart_and_wait(cluster_id=self.cluster_id)
            return self.wait_installed_libraries()

        return self.start()

    def delete(
        self
    ):
        logger.info("Deleting %s", self)
        return self.clusters_client().delete(cluster_id=self.cluster_id)

    def context(
        self,
        language: Optional["Language"] = None,
        context_id: Optional[str] = None
    ) -> ExecutionContext:
        return ExecutionContext(
            cluster=self,
            language=language,
            context_id=context_id
        )

    def execute(
        self,
        obj: Union[str, Callable],
        *,
        language: Optional["Language"] = None,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        env_keys: Optional[List[str]] = None,
        timeout: Optional[dt.timedelta] = None,
        result_tag: Optional[str] = None,
    ):
        return self.context(language=language).execute(
            obj=obj,
            args=args,
            kwargs=kwargs,
            env_keys=env_keys,
            timeout=timeout,
            result_tag=result_tag
        )

    # ------------------------------------------------------------------
    # decorator that routes function calls via `execute`
    # ------------------------------------------------------------------
    def execution_decorator(
        self,
        _func: Optional[Callable] = None,
        *,
        language: Optional["Language"] = None,
        env_keys: Optional[List[str]] = None,
        env_variables: Optional[Dict[str, str]] = None,
        timeout: Optional[dt.timedelta] = None,
        result_tag: Optional[str] = None,
        **options
    ):
        """
        Decorator to run a function via Workspace.execute instead of locally.

        Usage:

            @ws.remote()
            def f(x, y): ...

            @ws.remote(timeout=dt.timedelta(seconds=5))
            def g(a): ...

        You can also use it without parentheses:

            @ws.remote
            def h(z): ...
        """
        def decorator(func: Callable):
            if os.getenv("DATABRICKS_RUNTIME_VERSION") is not None:
                return func

            context = self.context(language=language or Language.PYTHON)
            serialized = CallableSerde.from_callable(func)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if os.getenv("DATABRICKS_RUNTIME_VERSION") is not None:
                    return func(*args, **kwargs)

                return context.execute(
                    obj=serialized,
                    args=list(args),
                    kwargs=kwargs,
                    env_keys=env_keys,
                    env_variables=env_variables,
                    timeout=timeout,
                    result_tag=result_tag,
                    **options
                )

            return wrapper

        # Support both @ws.remote and @ws.remote(...)
        if _func is not None and callable(_func):
            return decorator(_func)

        return decorator

    def install_libraries(
        self,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        wait_timeout: Optional[dt.timedelta] = dt.timedelta(minutes=20),
        pip_settings: Optional[PipIndexSettings] = None,
        raise_error: bool = True,
        restart: bool = True,
    ) -> "Cluster":
        if not libraries:
            return self

        wsdk = self.workspace.sdk()

        libraries = [
            self._check_library(_, pip_settings=pip_settings)
            for _ in libraries if _
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
        return self.workspace.sdk().libraries.cluster_status(cluster_id=self.cluster_id)

    def uninstall_libraries(
        self,
        pypi_packages: Optional[list[str]] = None,
        libraries: Optional[list["Library"]] = None,
        restart: bool = True
    ):
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

                logger.warning(
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

            time.sleep(10)
            statuses = list(self.installed_library_statuses())

        return self

    def install_temporary_libraries(
        self,
        libraries: str | ModuleType | List[str | ModuleType],
    ):
        return self.context().install_temporary_libraries(libraries=libraries)

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
                    target_path.write_bytes(f.read())

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
                or value.startswith("tgp_")
            ):
                repo = pip_settings.extra_index_url

            return Library(
                pypi=PythonPyPiLibrary(
                    package=value,
                    repo=repo,
                )
            )

        raise ValueError(f"Cannot build Library object from {type(value)}")
