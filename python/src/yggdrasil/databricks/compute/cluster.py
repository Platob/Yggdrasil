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
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Iterable, Iterator, List, Optional, Union

from databricks.sdk import ClustersAPI
from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import ResourceDoesNotExist
from databricks.sdk.service.compute import (
    ClusterAccessControlRequest,
    ClusterDetails,
    ClusterPermissionLevel,
    DataSecurityMode,
    Kind,
    Language,
    Library,
    LibraryInstallStatus,
    PythonPyPiLibrary,
    SparkVersion,
    State, RuntimeEngine,
)

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.environ import PyEnv, UserInfo
from yggdrasil.environ.pip_settings import PipIndexSettings
from yggdrasil.io.url import URL
from yggdrasil.version import VersionInfo, __version_info__ as YGG_VERSION_INFO

from .execution_context import ExecutionContext
from ..workspaces.workspace import WorkspaceService
from ...dataclasses.expiring import ExpiringDict
from ...pyutils.equality import dicts_equal

_CREATE_ARG_NAMES = set(inspect.signature(ClustersAPI.create).parameters.keys())
_EDIT_ARG_NAMES = set(inspect.signature(ClustersAPI.edit).parameters.keys())

__all__ = ["Cluster"]


LOGGER = logging.getLogger(__name__)

# host -> ExpiringDict(cluster_name -> cluster_id)
NAME_ID_CACHE: dict[str, ExpiringDict] = {}

# host -> ExpiringDict("versions" -> list[SparkVersion])
_SPARK_VERSIONS_CACHE: dict[str, ExpiringDict] = {}

_CLUSTER_RUNTIME_FIELDS = frozenset({"_system_context", "_contexts"})
_CLUSTER_SKIP_IF_NONE = frozenset({"_details", "cluster_name"})


def set_cached_cluster_name(host: str, cluster_name: str, cluster_id: str) -> None:
    existing = NAME_ID_CACHE.get(host)
    if not existing:
        existing = NAME_ID_CACHE[host] = ExpiringDict(default_ttl=60)
    existing[cluster_name] = cluster_id


def get_cached_cluster_id(host: str, cluster_name: str) -> str:
    existing = NAME_ID_CACHE.get(host)
    return existing.get(cluster_name) if existing else None


# module-level mapping Databricks Runtime -> Python version (major, minor, patch)
# Values reflect the "System environment -> Python:" line in DBR release notes.
_PYTHON_BY_DBR: dict[str, VersionInfo] = {
    "10.4": VersionInfo(3, 8, 10),   # Python 3.8.10
    "11.3": VersionInfo(3, 9, 21),   # Python 3.9.21
    "12.2": VersionInfo(3, 9, 21),   # Python 3.9.21
    "13.3": VersionInfo(3, 10, 12),  # Python 3.10.12
    "14.3": VersionInfo(3, 10, 12),  # Python 3.10.12
    "15.4": VersionInfo(3, 11, 11),  # Python 3.11.11
    "16.4": VersionInfo(3, 12, 3),   # Python 3.12.3
    "17.0": VersionInfo(3, 12, 3),   # Python 3.12.3
    "17.1": VersionInfo(3, 12, 3),   # Python 3.12.3
    "17.2": VersionInfo(3, 12, 3),   # Python 3.12.3
    "17.3": VersionInfo(3, 12, 3),   # Python 3.12.3
    "18.0": VersionInfo(3, 12, 3),   # Python 3.12.3
    "18.1": VersionInfo(3, 12, 3),   # Python 3.12.3
}

ALL_PURPOSE_CLUSTER: "Cluster | None" = None

_DBR_RE = re.compile(r"^(?P<maj>\d+)\.(?P<min>\d+)\.")


def _dbr_tuple_from_key(key: str) -> tuple[int, int]:
    # "17.3.x-gpu-ml-scala2.13" -> (17, 3)
    m = _DBR_RE.match(key)
    if not m:
        return -1, -1
    return int(m.group("maj")), int(m.group("min"))


def _dbr_str_from_key(key: str) -> Optional[str]:
    t = _dbr_tuple_from_key(key)
    if t == (-1, -1):
        return None
    return f"{t[0]}.{t[1]}"


def _is_photon_key(key: str) -> bool:
    return "photon" in key.lower()


def _py_filter_tuple(python_version: Union[str, tuple[int, ...]]) -> tuple[int, int]:
    if isinstance(python_version, str):
        parts = python_version.split(".")
        return int(parts[0]), int(parts[1])
    return int(python_version[0]), int(python_version[1])


def _py_tuple_for_key(key: str) -> Optional[tuple[int, int]]:
    dbr = _dbr_str_from_key(key)
    if not dbr:
        return None
    vi = _PYTHON_BY_DBR.get(dbr)
    return (vi.major, vi.minor) if vi else None


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

    _details: Optional[ClusterDetails] = dataclasses.field(default=None, repr=False, hash=False, compare=False)
    _details_refresh_time: float = dataclasses.field(default=0.0, repr=False, hash=False, compare=False)
    _contexts: dict[str, ExecutionContext] = dataclasses.field(default_factory=dict, repr=False, hash=False, compare=False)

    # host → Cluster instance
    _env_clusters: ClassVar[Dict[str, "Cluster"]] = {}

    def __post_init__(self):
        if self.cluster_name and not self.cluster_id:
            found = self.find_cluster(cluster_name=self.cluster_name, raise_error=True)
            self.cluster_id = found.cluster_id

    def __getstate__(self) -> dict:
        state = {}
        for key, value in self.__dict__.items():
            if key in _CLUSTER_RUNTIME_FIELDS:
                continue
            if key in _CLUSTER_SKIP_IF_NONE and value is None:
                continue
            state[key] = value

        # wall-clock timestamps don't survive process hops meaningfully
        state["_details_refresh_time"] = 0.0
        return state

    def __setstate__(self, state: dict) -> None:
        for field in _CLUSTER_SKIP_IF_NONE:
            state.setdefault(field, None)

        state["_system_context"] = None
        state["_contexts"] = {}
        self.__dict__.update(state)

        if self._details is not None:
            if self._details.cluster_id:
                self.cluster_id = self._details.cluster_id
            if self._details.cluster_name:
                self.cluster_name = self._details.cluster_name

    def __repr__(self):
        return "%s(url=%s)" % (self.__class__.__name__, self.url())

    def __str__(self):
        return self.url().to_string()

    def url(self) -> URL:
        return URL.parse_str("%s/compute/clusters/%s" % (self.workspace.safe_host, self.cluster_id or "unknown"))

    @property
    def id(self):
        return self.cluster_id

    @property
    def name(self) -> str:
        return self.cluster_name

    def is_in_databricks_environment(self):
        return self.workspace.is_in_databricks_environment()

    # ------------------------------------------------------------------ #
    # Details caching
    # ------------------------------------------------------------------ #
    @property
    def details(self) -> Optional[ClusterDetails]:
        if self._details is None and self.cluster_id is not None:
            self.details = self.clusters_client().get(cluster_id=self.cluster_id)
        return self._details

    def fresh_details(self, max_delay: float | None = None) -> Optional[ClusterDetails]:
        max_delay = 0.0 if max_delay is None else float(max_delay)
        delay = time.time() - float(self._details_refresh_time)

        if self.cluster_id and delay > max_delay:
            self.details = self.clusters_client().get(cluster_id=self.cluster_id)
        return self._details

    def refresh(self, max_delay: float | None = None):
        self.details = self.fresh_details(max_delay=max_delay)
        return self

    @details.setter
    def details(self, value: Optional[ClusterDetails]):
        if isinstance(value, ClusterDetails):
            self._details_refresh_time = time.time()
            self._details = value
            self.cluster_id = value.cluster_id
            self.cluster_name = value.cluster_name
        else:
            self._details_refresh_time = 0.0
            self._details = None
            self.cluster_id = getattr(value, "cluster_id", self.cluster_id)
            # keep cluster_id/cluster_name as-is unless you explicitly want to wipe them
            # (wiping tends to make URLs / logs less useful)

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
        return self.state in (State.PENDING, State.RESIZING, State.RESTARTING, State.TERMINATING)

    @property
    def is_error(self):
        return self.state == State.ERROR

    @property
    def requirements(self):
        return self.context(context_key="system").requirements

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
        return _PYTHON_BY_DBR.get(v)

    # ------------------------------------------------------------------ #
    # Singletons
    # ------------------------------------------------------------------ #
    def all_purpose_cluster(
        self,
        name: Optional[str] = None,
        python_version: Optional[str | tuple[int, ...]] = None,
    ):
        global ALL_PURPOSE_CLUSTER

        if ALL_PURPOSE_CLUSTER is not None:
            return ALL_PURPOSE_CLUSTER

        if not name:
            if not python_version:
                major, minor, _ = PyEnv.current().version_info
                python_version = f"{major}.{minor}"
            name = f"Yggdrasil {YGG_VERSION_INFO.major}.{YGG_VERSION_INFO.minor} All Purpose py{python_version}"

        ALL_PURPOSE_CLUSTER = self.find_cluster(cluster_name=name, raise_error=False)

        if ALL_PURPOSE_CLUSTER is None:
            ALL_PURPOSE_CLUSTER = self.create_or_update(
                cluster_name=name,
                python_version=python_version,
                libraries=[
                    f"ygg~={YGG_VERSION_INFO.major}.{YGG_VERSION_INFO.minor}",
                    "uv",
                    "dill",
                ],
                wait=False,
            )

        return ALL_PURPOSE_CLUSTER

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def clusters_client(self) -> ClustersAPI:
        return self.workspace.sdk().clusters

    def shared_cache_path(self, suffix: str):
        assert suffix, "Missing suffix arg"
        return self.workspace.shared_cache_path(suffix="/cluster/%s" % suffix.lstrip("/"))

    def _cached_spark_versions(self, ttl_seconds: int = 300) -> list[SparkVersion]:
        host = self.workspace.safe_host
        cache = _SPARK_VERSIONS_CACHE.get(host)
        if cache is None:
            cache = _SPARK_VERSIONS_CACHE[host] = ExpiringDict(default_ttl=ttl_seconds)

        versions = cache.get("versions")
        if versions is None:
            versions = self.clusters_client().spark_versions().versions or []
            cache["versions"] = versions

        return versions

    def spark_versions(
        self,
        photon: Optional[bool] = None,
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
        *,
        allow_ml: bool = False,
        allow_gpu: bool = True,
    ) -> list[SparkVersion]:
        versions = self._cached_spark_versions()
        if not versions:
            raise ValueError("No databricks spark versions found")

        # Filter ML/GPU early (cheaper, shrinks list)
        if not allow_ml:
            versions = [v for v in versions if "-ml-" not in v.key.lower()]
        if not allow_gpu:
            versions = [v for v in versions if "-gpu-" not in v.key.lower()]

        if photon is not None:
            versions = [v for v in versions if ("photon" in v.key.lower()) == photon]

        if python_version is not None:
            py_filter = _py_filter_tuple(python_version)
            versions = [v for v in versions if _py_tuple_for_key(v.key) == py_filter]

            if not versions and py_filter[1] > 12:
                # fallback: ignore python filter
                return self.spark_versions(photon=photon, allow_ml=allow_ml, allow_gpu=allow_gpu)

        return versions

    def latest_spark_version(
        self,
        photon: Optional[bool] = None,
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
        *,
        allow_ml: bool = False,
        allow_gpu: bool = True,
    ) -> SparkVersion:
        def pick(ph: Optional[bool], use_py: bool) -> Optional[SparkVersion]:
            versions = self.spark_versions(
                photon=ph,
                python_version=python_version if use_py else None,
                allow_ml=allow_ml,
                allow_gpu=allow_gpu,
            )
            return max(versions, key=lambda v: _dbr_tuple_from_key(v.key), default=None)

        chosen = (pick(True, True) or pick(False, True)) if photon is None else pick(photon, True)

        if chosen is None and python_version is not None:
            py_filter = _py_filter_tuple(python_version)
            if py_filter[1] > 12:
                chosen = (pick(True, False) or pick(False, False)) if photon is None else pick(photon, False)

        if chosen is None:
            raise ValueError(
                f"No databricks runtime version found for photon={photon} and python_version={python_version}"
            )
        return chosen

    # ------------------------------------------------------------------ #
    # CRUD operations
    # ------------------------------------------------------------------ #
    def _check_details(
        self,
        details: ClusterDetails,
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
        **kwargs,
    ) -> ClusterDetails:
        pip_settings = PipIndexSettings.current()

        new_details = ClusterDetails(
            **{
                **details.as_shallow_dict(),
                **kwargs,
            }
        )

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
                python_version=python_version,
                allow_ml=False,
            ).key

        is_photon = _is_photon_key(new_details.spark_version)

        if is_photon:
            new_details.spark_version = new_details.spark_version.replace("-photon-", "-")

        if new_details.single_user_name:
            if not new_details.data_security_mode:
                new_details.data_security_mode = DataSecurityMode.DATA_SECURITY_MODE_DEDICATED

        if not new_details.node_type_id:
            new_details.node_type_id = "rd-fleet.xlarge"

        if (
            getattr(new_details, "virtual_cluster_size", None) is None
            and new_details.num_workers is None
            and new_details.autoscale is None
        ):
            if new_details.is_single_node is None:
                new_details.is_single_node = True

        if new_details.runtime_engine is None and is_photon:
            new_details.runtime_engine = RuntimeEngine.PHOTON

        if new_details.kind is None:
            if new_details.is_single_node:
                new_details.kind = Kind.CLASSIC_PREVIEW

        if pip_settings.extra_index_urls:
            if new_details.spark_env_vars is None:
                new_details.spark_env_vars = {}
            str_urls = " ".join(pip_settings.extra_index_urls)
            # note: original code used UV_INDEX by mistake; keep behavior but avoid clobbering explicitly set vars
            new_details.spark_env_vars.setdefault("UV_EXTRA_INDEX_URL", str_urls)
            new_details.spark_env_vars.setdefault("PIP_EXTRA_INDEX_URL", str_urls)

        return new_details

    def create_or_update(
        self,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
        libraries: Optional[List[Union[str, Library]]] = None,
        wait: WaitingConfigArg = True,
        **cluster_spec: Any,
    ):
        found = self.find_cluster(
            cluster_id=cluster_id or self.cluster_id,
            cluster_name=cluster_name or self.cluster_name,
            raise_error=False,
        )

        if found is not None:
            return found.update(cluster_name=cluster_name, libraries=libraries, wait=wait, **cluster_spec)

        return self.create(cluster_name=cluster_name, libraries=libraries, wait=wait, **cluster_spec)

    def create(
        self,
        libraries: Optional[List[Union[str, Library]]] = None,
        wait: WaitingConfigArg = True,
        **cluster_spec: Any,
    ) -> "Cluster":
        cluster_spec["autotermination_minutes"] = int(cluster_spec.get("autotermination_minutes", 30))

        update_details = {
            k: v
            for k, v in self._check_details(details=ClusterDetails(), **cluster_spec).as_shallow_dict().items()
            if k in _CREATE_ARG_NAMES
        }

        LOGGER.debug("Creating Databricks cluster %s with %s", update_details.get("cluster_name"), update_details)

        self.details = self.clusters_client().create(**update_details)

        LOGGER.info("Created %s", self)

        self.install_libraries(libraries=libraries, raise_error=False, wait=False)
        self.wait_for_status(wait=wait)

        return self

    def update(
        self,
        libraries: Optional[List[Union[str, Library]]] = None,
        access_control_list: Optional[List[ClusterAccessControlRequest] | bool] = True,
        wait: WaitingConfigArg = True,
        **cluster_spec: Any,
    ) -> "Cluster":
        self.install_libraries(libraries=libraries, wait=False, raise_error=False)

        existing_details = {k: v for k, v in self.details.as_shallow_dict().items() if k in _EDIT_ARG_NAMES}

        update_details = {
            k: v for k, v in self._check_details(details=self.details, **cluster_spec).as_shallow_dict().items() if k in _EDIT_ARG_NAMES
        }

        same = dicts_equal(existing_details, update_details, keys=_EDIT_ARG_NAMES)

        if not same:
            LOGGER.debug("Updating %s with %s", self, update_details)

            self.wait_for_status(wait=wait)
            self.clusters_client().edit(**update_details)
            self.update_permissions(access_control_list=access_control_list)

            LOGGER.info("Updated %s", self)
            self.wait_for_status(wait=wait)

        return self

    def update_permissions(self, access_control_list: Optional[List[ClusterAccessControlRequest] | bool] = True):
        if not access_control_list:
            return self

        access_control_list = self._check_permission(access_control_list)

        self.clusters_client().update_permissions(cluster_id=self.cluster_id, access_control_list=access_control_list)
        return self

    def default_permissions(self):
        current_groups = self.current_user_groups() or []
        return [
            ClusterAccessControlRequest(group_name=group.display, permission_level=ClusterPermissionLevel.CAN_MANAGE)
            for group in current_groups
            if group.display not in {"users"}
        ]

    def _check_permission(
        self,
        permission: Union[str, ClusterAccessControlRequest, List[Union[str, ClusterAccessControlRequest]], bool],
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

        defaults = self.default_permissions()

        if isinstance(permission, bool):
            return defaults if permission else []

        return defaults + [self._check_permission(_) for _ in permission]

    def list_clusters(self) -> Iterator["Cluster"]:
        for details in self.clusters_client().list():
            details = details  # sdk model
            yield Cluster(
                workspace=self.workspace,
                cluster_id=details.cluster_id,
                cluster_name=details.cluster_name,
                _details=details,
            )

    def find_cluster(
        self,
        cluster_id: Optional[str] = None,
        *,
        cluster_name: Optional[str] = None,
        raise_error: Optional[bool] = None,
    ) -> Optional["Cluster"]:
        if not cluster_name and not cluster_id:
            raise ValueError("Either name or cluster_id must be provided")

        if not cluster_id and cluster_name:
            cluster_id = get_cached_cluster_id(host=self.workspace.safe_host, cluster_name=cluster_name)

        if cluster_id:
            try:
                details = self.clusters_client().get(cluster_id=cluster_id)
            except ResourceDoesNotExist:
                if raise_error:
                    raise ValueError(f"Cannot find databricks cluster {cluster_id!r}")
                return None

            # populate name cache for fast future lookups
            if details.cluster_name:
                set_cached_cluster_name(self.workspace.safe_host, details.cluster_name, details.cluster_id)

            return Cluster(
                workspace=self.workspace,
                cluster_id=details.cluster_id,
                cluster_name=details.cluster_name,
                _details=details,
            )

        # last resort: list scan (expensive)
        for cluster in self.list_clusters():
            if cluster_name == cluster.details.cluster_name:
                set_cached_cluster_name(
                    host=self.workspace.safe_host,
                    cluster_name=cluster.cluster_name,
                    cluster_id=cluster.cluster_id,
                )
                return cluster

        if raise_error:
            raise ValueError(f"Cannot find databricks cluster {cluster_name!r}")
        return None

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
            self.details = self.clusters_client().restart_and_wait(cluster_id=self.cluster_id)
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
        language: Optional[Language] = None,
        context_id: Optional[str] = None,
        context_key: Optional[str] = None,
    ) -> ExecutionContext:
        if context_key:
            existing = self._contexts.get(context_key)
            if existing is None:
                existing = self._contexts[context_key] = ExecutionContext(
                    cluster=self,
                    language=language,
                    context_id=context_id,
                    context_key=context_key,
                )
            return existing

        return ExecutionContext(
            cluster=self,
            language=language,
            context_id=context_id,
            context_key=context_key,
        )

    def decorate(
        self,
        func: Optional[Callable] = None,
        *,
        command: Optional[str] = None,
        language: Optional[Language] = None,
        command_id: Optional[str] = None,
        environ: Optional[Union[Iterable[str], Dict[str, str]]] = None,
        context_key: Optional[str] = None,
    ) -> Callable:
        language = Language.PYTHON if language is None else language

        if not context_key:
            usr, env = UserInfo.current(), PyEnv.current()
            major, minor, _ = env.version_info
            context_key = f"{usr.hostname}-py{major}.{minor}"

        context = self.context(language=language, context_key=context_key)

        return context.decorate(
            func=func,
            command=command,
            language=language,
            command_id=command_id,
            environ=environ,
        )

    # ------------------------------------------------------------------ #
    # Libraries
    # ------------------------------------------------------------------ #
    def install_libraries(
        self,
        libraries: Optional[List[Union[str, Library]]] = None,
        wait: WaitingConfigArg = True,
        pip_settings: Optional[PipIndexSettings] = None,
        raise_error: bool = True,
    ) -> "Cluster":
        if not libraries:
            return self

        wsdk = self.workspace.sdk()

        pip_settings = PipIndexSettings.current() if pip_settings is None else pip_settings

        normalized: list[Library] = [self._check_library(_, pip_settings=pip_settings) for _ in libraries if _]

        if normalized:
            existing_sigs = {
                _library_sig(st.library)
                for st in self.installed_library_statuses()
                if getattr(st, "library", None) is not None
            }
            normalized = [lib for lib in normalized if _library_sig(lib) not in existing_sigs]

        if normalized:
            wsdk.libraries.install(cluster_id=self.cluster_id, libraries=normalized)
            self.wait_installed_libraries(wait=wait, raise_error=raise_error)

        return self

    def installed_library_statuses(self):
        return self.workspace.sdk().libraries.cluster_status(cluster_id=self.cluster_id)

    def uninstall_libraries(
        self,
        pypi_packages: Optional[list[str]] = None,
        libraries: Optional[list[Library]] = None,
        restart: bool = True,
    ):
        if libraries is None:
            to_remove = [
                lib.library
                for lib in self.installed_library_statuses()
                if self._filter_lib(lib, pypi_packages=pypi_packages, default_filter=False)
            ]
        else:
            to_remove = libraries

        if to_remove:
            self.workspace.sdk().libraries.uninstall(cluster_id=self.cluster_id, libraries=to_remove)

            if restart:
                self.restart()

        return self

    @staticmethod
    def _filter_lib(lib: Optional[Library], pypi_packages: Optional[list[str]] = None, default_filter: bool = False):
        if lib is None:
            return False

        if lib.pypi:
            if lib.pypi.package and pypi_packages:
                return lib.pypi.package in pypi_packages

        return default_filter

    def wait_installed_libraries(
        self,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True,
    ):
        if not self.is_running:
            return self

        wait = WaitingConfig.check_arg(wait)

        if wait:
            statuses = list(self.installed_library_statuses())
            start, iteration = time.time(), 0

            while True:
                failed = [_.library for _ in statuses if _.library and _.status == LibraryInstallStatus.FAILED]

                if failed:
                    if raise_error:
                        raise DatabricksError("Libraries %s in %s failed to install" % (failed, self))
                    LOGGER.exception("Libraries %s in %s failed to install", failed, self)

                running = [
                    _
                    for _ in statuses
                    if _.status
                    in (LibraryInstallStatus.INSTALLING, LibraryInstallStatus.PENDING, LibraryInstallStatus.RESOLVING)
                ]

                if not running:
                    break

                wait.sleep(iteration=iteration, start=start)
                iteration += 1
                statuses = list(self.installed_library_statuses())

        return self

    def _check_library(self, value, pip_settings: Optional[PipIndexSettings] = None) -> Library:
        if isinstance(value, Library):
            return value

        pip_settings = PipIndexSettings.current() if pip_settings is None else pip_settings

        if isinstance(value, str):
            # local path -> copy to shared cache
            if os.path.exists(value):
                target_path = self.workspace.shared_cache_path(
                    suffix=f"/clusters/{self.cluster_id}/{os.path.basename(value)}"
                )

                # NOTE: if URL.open() supports streaming writes, use it here.
                with open(value, mode="rb") as f:
                    target_path.open().write_all_bytes(f.read())

                value = str(target_path)

            # Now value is either a dbfs:/ path or plain package name
            if value.endswith(".jar"):
                return Library(jar=value)
            if value.endswith("requirements.txt"):
                return Library(requirements=value)
            if value.endswith(".whl"):
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