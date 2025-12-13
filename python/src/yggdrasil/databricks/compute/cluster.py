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
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Iterator, Optional, Union, List, Callable, Dict, ClassVar

from databricks.sdk.service.compute import SparkVersion, RuntimeEngine

from .execution_context import ExecutionContext
from ..workspaces.workspace import WorkspaceService, Workspace
from ...libs.databrickslib import databricks_sdk
from ...ser import SerializedFunction

if databricks_sdk is None:  # pragma: no cover - import guard
    ResourceDoesNotExist = Exception  # type: ignore
else:  # pragma: no cover - runtime fallback when SDK is missing
    from databricks.sdk import ClustersAPI
    from databricks.sdk.errors import ResourceDoesNotExist
    from databricks.sdk.service.compute import (
        ClusterDetails, Language, Kind, State, DataSecurityMode, Library, PythonPyPiLibrary
    )

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


CURRENT_ENV_CLUSTER: Optional["Cluster"] = None  # whatever the class is


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
    
    _details: Optional["ClusterDetails"] = None
    _details_refresh_time: float = dataclasses.field(default=0, init=False)

    # host → Cluster instance
    _env_clusters: ClassVar[Dict[str, "Cluster"]] = {}

    @property
    def id(self):
        return self.cluster_id

    @property
    def name(self) -> str:
        if not self.cluster_id:
            return "unknown"
        return self.details.cluster_name

    def __post_init__(self):
        if self._details:
            self._details_refresh_time = time.time()

    def is_in_databricks_environment(self):
        return self.workspace.is_in_databricks_environment()

    @classmethod
    def replicated_current_environment(
        cls,
        workspace: Optional["Workspace"] = None,
    ) -> "Cluster":
        if workspace is None:
            workspace = Workspace()  # your default, whatever it is

        host = workspace.host

        # 🔥 return existing singleton for this host
        if host in cls._env_clusters:
            return cls._env_clusters[host]

        # 🔥 first time for this host → create
        logger.info("Creating replicated cluster for host %s", host)
        inst = cls(workspace=workspace)

        inst = inst.create_or_update(
            cluster_name=inst.workspace.current_user.user_name,
            python_version=sys.version_info,
            single_user_name=inst.workspace.current_user.user_name,
            runtime_engine=RuntimeEngine.PHOTON,
            autotermination_minutes=30,
            libraries=["ygg"],
        )

        cls._env_clusters[host] = inst
        return inst

    @property
    def details(self):
        if self._details is None and self.cluster_id is not None:
            self._details = self.clusters_client().get(cluster_id=self.cluster_id)
        return self._details

    def fresh_details(self, max_delay: float):
        if self.cluster_id and time.time() - self._details_refresh_time > max_delay:
            self.details = self.clusters_client().get(cluster_id=self.cluster_id)
        return self._details

    @details.setter
    def details(self, value: "ClusterDetails"):
        self._details_refresh_time = time.time()
        self._details = value
        self.cluster_id = value.cluster_id

    @property
    def state(self):
        return self.fresh_details(max_delay=10).state

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
                if python_version.lower() == "current":
                    py_filter = (sys.version_info.major, sys.version_info.minor)
                else:
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

            if not versions and py_filter > 12:
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
        if kwargs:
            details = ClusterDetails(**{
                **details.as_shallow_dict(),
                **kwargs
            })

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

        return details

    def create_or_update(
        self,
        cluster_name: Optional[str] = None,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        **cluster_spec: Any
    ):
        logger.info(
            "Ensuring cluster exists (cluster_id=%s, cluster_name=%s)",
            self.cluster_id,
            cluster_name,
        )
        found = self.find_cluster(
            cluster_id=self.cluster_id,
            cluster_name=cluster_name,
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

        logger.info(
            "Creating Databricks cluster %s with spec keys=%s",
            update_details.cluster_name,
            sorted(update_details.as_shallow_dict().keys()),
        )

        self.details = self.clusters_client().create_and_wait(**{
            k: v
            for k, v in update_details.as_shallow_dict().items()
            if k not in _CREATE_ARG_NAMES
        })

        logger.info(
            "Created Databricks cluster %s (cluster_id=%s)",
            self.details.cluster_name,
            self.details.cluster_id,
        )

        self.install_libraries(libraries=libraries)

        return self

    def update(
        self,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        **cluster_spec: Any
    ) -> "Cluster":
        logger.info(
            "Updating Databricks cluster %s (cluster_id=%s)",
            cluster_spec.get("cluster_name", self.details.cluster_name if self.details else None),
            self.cluster_id,
        )
        self.install_libraries(libraries=libraries)

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
            self.details = self.clusters_client().edit_and_wait(**update_details)

        return self

    def list_clusters(self) -> Iterator["Cluster"]:
        """Iterate clusters, yielding helpers annotated with metadata."""

        for details in self.clusters_client().list():
            details: ClusterDetails = details

            yield Cluster(
                workspace=self.workspace,
                cluster_id=details.cluster_id,
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
            logger.debug("Looking up cluster by cluster_id=%s", cluster_id)
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

        logger.debug("Searching for cluster with name=%s", cluster_name)

        for cluster in self.list_clusters():
            if cluster_name_cf == cluster.details.cluster_name.casefold():
                return cluster

        if raise_error:
            raise ValueError(f"Cannot find databricks cluster {cluster_name!r}")
        return None

    def ensure_running(
        self,
    ) -> "Cluster":
        if self.state != State.RUNNING:
            logger.info("Starting cluster %s because state=%s", self.cluster_id, self.state)
            return self.start()

        return self

    def start(
        self,
    ) -> "Cluster":
        logger.info("Starting Databricks cluster %s", self.cluster_id)
        self.details = self.clusters_client().start_and_wait(cluster_id=self.cluster_id)
        return self

    def restart(
        self,
    ):
        logger.info("Restarting Databricks cluster %s", self.cluster_id)
        self.details = self.clusters_client().restart_and_wait(cluster_id=self.cluster_id)
        return self

    def delete(
        self
    ):
        logger.info("Deleting Databricks cluster %s", self.cluster_id)
        return self.clusters_client().delete(cluster_id=self.cluster_id)

    def execution_context(
        self,
        language: Optional["Language"] = None,
        context_id: Optional[str] = None
    ) -> ExecutionContext:
        logger.debug(
            "Creating execution context for cluster_id=%s with language=%s", self.cluster_id, language
        )
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
        if language is None:
            language = Language.PYTHON

        with self.execution_context(language=language) as ctx:
            return ctx.execute(
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
        timeout: Optional[dt.timedelta] = None,
        result_tag: Optional[str] = None,
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
            context = self.execution_context(language=language or Language.PYTHON)
            serialized = func if isinstance(func, SerializedFunction) else SerializedFunction.from_callable(func)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if os.getenv("DATABRICKS_RUNTIME_VERSION") is not None:
                    return func(*args, **kwargs)

                return context.execute(
                    obj=serialized,
                    args=list(args),
                    kwargs=kwargs,
                    env_keys=env_keys,
                    timeout=timeout,
                    result_tag=result_tag,
                )

            return wrapper

        # Support both @ws.remote and @ws.remote(...)
        if _func is not None and callable(_func):
            return decorator(_func)

        return decorator

    def install_libraries(
        self,
        libraries: Optional[List[Union[str, "Library"]]] = None,
    ) -> "Cluster":
        if not libraries:
            return self

        wsdk = self.workspace.sdk()

        wsdk.libraries.install(
            cluster_id=self.cluster_id,
            libraries=[
                self._check_library(_)
                for _ in libraries if _
            ]
        )

        return self

    def install_temporary_libraries(
        self,
        libraries: str | ModuleType | List[str | ModuleType],
    ):
        return self.execution_context().install_temporary_libraries(libraries=libraries)

    def _check_library(
        self,
        value,
    ) -> "Library":
        if isinstance(value, Library):
            return value

        if isinstance(value, str):
            if os.path.exists(value):
                target_path = self.workspace.shared_cache_path(
                    suffix=f"/clusters/{self.cluster_id}/{os.path.basename(value)}"
                )
                self.workspace.upload_local_path(local_path=value, target_path=target_path)
                value = target_path

            # Now value is either a dbfs:/ path or plain package name
            if value.endswith(".jar"):
                return Library(jar=value)
            elif value.endswith("requirements.txt"):
                return Library(requirements=value)
            elif value.endswith(".whl"):
                return Library(whl=value)

            # Fallback: treat as PyPI / private index package
            return Library(
                pypi=PythonPyPiLibrary(
                    package=value,
                    repo=os.getenv("PIP_EXTRA_INDEX_URL"),
                )
            )

        raise ValueError(f"Cannot build Library object from {type(value)}")
