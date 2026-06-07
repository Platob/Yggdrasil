import inspect
import logging
import re
from typing import Optional, MutableMapping, Sequence, Union, Any, TYPE_CHECKING, Iterator

from databricks.sdk.errors import ResourceDoesNotExist, PermissionDenied
from databricks.sdk.service.compute import ClusterAccessControlRequest, Library, ClustersAPI, ClusterDetails, \
    DataSecurityMode, RuntimeEngine, Kind, SparkVersion, ListClustersFilterBy, ClusterSource

from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.version import VersionInfo
from ..client import DatabricksClient
from ..client import DatabricksService
from ...dataclasses.expiring import ExpiringDict
from ...environ.pip_settings import PipIndexSettings

if TYPE_CHECKING:
    from .cluster import Cluster

__all__ = [
    "Clusters",
    "PYTHON_BY_DBR",
]

LOGGER = logging.getLogger(__name__)
_CREATE_ARG_NAMES = set(inspect.signature(ClustersAPI.create).parameters.keys())

_CLUSTER_ID_RE = re.compile(r"^\d{4}-\d{6}-[a-z0-9]{8}$")
# host -> ExpiringDict(cluster_name -> cluster_id)
NAME_ID_CACHE: dict[str, ExpiringDict] = {}
NAMED_CLUSTERS: ExpiringDict[str, "Cluster"] = ExpiringDict(default_ttl=7200.0)
# host -> ExpiringDict("versions" -> list[SparkVersion])
_SPARK_VERSIONS_CACHE: dict[str, ExpiringDict] = {}
_DBR_RE = re.compile(r"^(?P<maj>\d+)\.(?P<min>\d+)\.")

# module-level mapping Databricks Runtime -> Python version (major, minor, patch)
# Values reflect the "System environment -> Python:" line in DBR release notes.
PYTHON_BY_DBR: dict[str, VersionInfo] = {
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


def _py_tuple_for_key(key: str) -> Optional[tuple[int, int]]:
    dbr = _dbr_str_from_key(key)
    if not dbr:
        return None
    vi = PYTHON_BY_DBR.get(dbr)
    return (vi.major, vi.minor) if vi else None


def set_cached_cluster_name(client: DatabricksClient, cluster_name: str, cluster_id: str) -> None:
    host = client.base_url.to_string()
    existing = NAME_ID_CACHE.get(host)
    if not existing:
        existing = NAME_ID_CACHE[host] = ExpiringDict(default_ttl=60)
    existing[cluster_name] = cluster_id


def get_cached_cluster_id(client: DatabricksClient, cluster_name: str) -> str:
    host = client.base_url.to_string()
    existing = NAME_ID_CACHE.get(host)
    return existing.get(cluster_name) if existing else None


def _is_photon_key(key: str) -> bool:
    return "photon" in key.lower()


def _py_filter_tuple(python_version: Union[str, tuple[int, ...]]) -> tuple[int, int]:
    if isinstance(python_version, str):
        parts = python_version.split(".")
        return int(parts[0]), int(parts[1])
    return int(python_version[0]), int(python_version[1])


class Clusters(DatabricksService):

    # ------------------------------------------------------------------ #
    # Singletons
    # ------------------------------------------------------------------ #
    def _default_ygg_layer(self, python: str | None = None) -> str:
        """The default ygg library for a cluster: the **base environment**'s
        classic-cluster ``requirements.txt`` (zero-PyPI) for the ygg image
        matching the cluster's **Python**, installed via ``Library(requirements=…)``
        — resolved (and get-or-created on a miss) through ``dbc.environments``.

        *python* is the cluster's ``"3.X"`` runtime Python (derived from its DBR
        in :meth:`all_purpose_cluster`); ``None`` falls back to the local
        interpreter. An explicit ``environment`` / ``libraries`` argument
        overrides it (and may be a PyPI spec to opt back into a pip resolve)."""
        return self.client.environments.find("ygg", python=python).cluster

    def _cluster_python(self, cluster_spec: MutableMapping[str, Any]) -> str | None:
        """The ``"3.X"`` Python the cluster-to-be will run, so its ygg layer can
        pick the matching ``requirements.txt``.

        Resolved from an explicit ``python_version``, else the cluster's
        ``spark_version`` (DBR → Python via :data:`PYTHON_BY_DBR`), else the
        latest runtime that :meth:`create` would default to. Best-effort —
        returns ``None`` (the caller falls back to the local interpreter) when
        the runtime list can't be reached or the DBR is unmapped."""
        py = cluster_spec.get("python_version")
        if py is not None:
            t = _py_filter_tuple(py)
            return f"{t[0]}.{t[1]}"
        spark_version = cluster_spec.get("spark_version")
        if spark_version is None:
            try:
                spark_version = self.latest_spark_version(allow_ml=False).key
            except Exception:  # noqa: BLE001 — no runtime list → fall back to local
                return None
        t = _py_tuple_for_key(spark_version)
        return f"{t[0]}.{t[1]}" if t else None

    def all_purpose_cluster(
        self,
        name: str | None = None,
        *,
        key: str | None = None,
        custom_tags: Optional[MutableMapping[str, str]] = None,
        single_user_name: str | None = None,
        permissions: Optional[list[str | ClusterAccessControlRequest]] = None,
        libraries: Optional[Sequence[str]] = None,
        environment: str | None = None,
        wait: WaitingConfigArg = True,
        **cluster_spec
    ):
        global NAMED_CLUSTERS

        if not name:
            if key:
                key = key.strip()

            # Default to a per-user name (e.g. ``"All Purpose-alice"``) so
            # multiple developers in the same workspace don't fight over a
            # single ``"All Purpose"`` cluster.
            name = key or self.client.user_scoped_name("All Purpose")

        existing = NAMED_CLUSTERS.get(name, None)

        if existing is not None:
            return existing

        existing = next(
            self.list(
                name=name,
                sources=[ClusterSource.API, ClusterSource.UI],
                limit=1
            ),
            None
        )

        # By default the cluster installs the seeded **generic environment**
        # (zero-PyPI workspace wheels via :meth:`_default_ygg_layer`) rather than
        # ``pip install``-ing ygg from PyPI. An explicit ``environment`` overrides
        # it — a ``…requirements.txt`` / ``…whl`` path, or a PyPI spec to opt back
        # into a pip resolve. ``uv`` / ``dill`` (small public runtime helpers the
        # env doesn't bundle) ride along either way.
        ygg_layer = (
            environment if environment
            else self._default_ygg_layer(python=self._cluster_python(cluster_spec))
        )
        libraries = (libraries or []) + [ygg_layer, "uv", "dill"]

        if existing is None:
            existing = self.create(
                cluster_name=name,
                single_user_name=single_user_name,
                libraries=libraries,
                permissions=permissions,
                wait=wait,
                custom_tags=custom_tags,
                **cluster_spec
            )

        NAMED_CLUSTERS[existing.cluster_name] = existing
        return existing

    # ------------------------------------------------------------------ #
    # CRUD operations
    # ------------------------------------------------------------------ #
    def _cached_spark_versions(self, ttl_seconds: int = 300) -> list[SparkVersion]:
        host = self.client.base_url.to_string()
        cache = _SPARK_VERSIONS_CACHE.get(host)
        if cache is None:
            cache = _SPARK_VERSIONS_CACHE[host] = ExpiringDict(default_ttl=ttl_seconds)

        versions = cache.get("versions")
        if versions is None:
            client = self.client.workspace_client().clusters
            versions = client.spark_versions().versions or []
            cache["versions"] = versions

        return versions

    def spark_versions(
        self,
        photon: bool | None = None,
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
        photon: bool | None = None,
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
        *,
        allow_ml: bool = False,
        allow_gpu: bool = True,
    ) -> SparkVersion:
        def pick(ph: bool | None, use_py: bool) -> Optional[SparkVersion]:
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

    def check_details(
        self,
        update: bool,
        details: ClusterDetails,
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
        single_user_name: str | None = None,
        **kwargs,
    ) -> ClusterDetails:
        pip_settings = PipIndexSettings.current()

        new_details = ClusterDetails(
            **{
                **details.as_shallow_dict(),
                **kwargs,
            }
        )

        default_tags = self.default_tags(update=update)

        if new_details.custom_tags is None:
            new_details.custom_tags = default_tags
        elif default_tags:
            new_tags = new_details.custom_tags.copy()
            new_tags.update(default_tags)
            new_details.custom_tags = new_tags

        if new_details.spark_version is None or python_version:
            new_details.spark_version = self.latest_spark_version(
                python_version=python_version,
                allow_ml=False,
            ).key

        is_photon = _is_photon_key(new_details.spark_version)

        if is_photon:
            new_details.spark_version = new_details.spark_version.replace("-photon-", "-")

        if single_user_name and not new_details.single_user_name:
            new_details.single_user_name = single_user_name

        if new_details.single_user_name:
            if not new_details.data_security_mode:
                new_details.data_security_mode = DataSecurityMode.DATA_SECURITY_MODE_DEDICATED

        if new_details.instance_pool_id:
            new_details.node_type_id = None
            new_details.driver_node_type_id = None
            new_details.enable_elastic_disk = None
            new_details.aws_attributes = None
            new_details.gcp_attributes = None
        elif not new_details.node_type_id:
            from yggdrasil.enums import NodeType
            new_details.node_type_id = NodeType.DEFAULT.value

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
            new_details.spark_env_vars.setdefault("UV_EXTRA_INDEX_URL", str_urls)
            new_details.spark_env_vars.setdefault("PIP_EXTRA_INDEX_URL", str_urls)

        return new_details

    def get_or_create(
        self,
        obj: Any = None,
        *,
        cluster_id: str | None = None,
        cluster_name: str | None = None,
        single_user_name: str | None = None,
        permissions: Optional[list[str | ClusterAccessControlRequest]] = None,
        libraries: Optional[list[Union[str, Library]]] = None,
        wait: WaitingConfigArg = True,
        **cluster_spec: Any,
    ):
        # Positional shortcut: accept a Cluster, an ID, or a name.
        if obj is not None:
            from .cluster import Cluster

            if isinstance(obj, Cluster):
                return obj
            if isinstance(obj, str):
                if _CLUSTER_ID_RE.match(obj):
                    cluster_id = cluster_id or obj
                else:
                    cluster_name = cluster_name or obj
            else:
                raise TypeError(
                    f"obj must be Cluster | str | None, got {type(obj).__name__}"
                )

        found = self.find_cluster(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            sources=[ClusterSource.API, ClusterSource.UI],
            raise_error=False,
        )

        if found is not None:
            return found

        # Default to the seeded generic environment (zero-PyPI workspace wheels)
        # for the cluster's Python instead of ``pip install``-ing ygg from PyPI.
        ygg_layer = self._default_ygg_layer(python=self._cluster_python(cluster_spec))
        libraries = (list(libraries) if libraries else []) + [ygg_layer]

        return self.create(
            cluster_name=cluster_name,
            libraries=libraries,
            wait=wait,
            permissions=permissions,
            single_user_name=single_user_name,
            **cluster_spec
        )

    def create_or_update(
        self,
        obj: Any = None,
        *,
        cluster_id: str | None = None,
        cluster_name: str | None = None,
        single_user_name: str | None = None,
        permissions: Optional[list[str | ClusterAccessControlRequest]] = None,
        libraries: Optional[list[Union[str, Library]]] = None,
        wait: WaitingConfigArg = True,
        **cluster_spec: Any,
    ):
        # Positional shortcut: accept a Cluster, an ID, or a name.
        if obj is not None:
            from .cluster import Cluster

            if isinstance(obj, Cluster):
                return obj
            if isinstance(obj, str):
                if _CLUSTER_ID_RE.match(obj):
                    cluster_id = cluster_id or obj
                else:
                    cluster_name = cluster_name or obj
            else:
                raise TypeError(
                    f"obj must be Cluster | str | None, got {type(obj).__name__}"
                )

        found = self.find_cluster(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            raise_error=False,
        )

        if found is not None:
            return found.update(
                cluster_name=cluster_name,
                libraries=libraries,
                wait=wait,
                permissions=permissions,
                single_user_name=single_user_name,
                **cluster_spec
            )

        # Default to the seeded generic environment (zero-PyPI workspace wheels)
        # for the cluster's Python instead of ``pip install``-ing ygg from PyPI.
        ygg_layer = self._default_ygg_layer(python=self._cluster_python(cluster_spec))
        libraries = (list(libraries) if libraries else []) + [ygg_layer]

        return self.create(
            cluster_name=cluster_name,
            libraries=libraries,
            wait=wait,
            permissions=permissions,
            single_user_name=single_user_name,
            **cluster_spec
        )

    def create(
        self,
        *,
        libraries: Optional[list[Union[str, Library]]] = None,
        permissions: Optional[list[str | ClusterAccessControlRequest]] = None,
        single_user_name: str | None = None,
        wait: WaitingConfigArg = True,
        **cluster_spec: Any,
    ) -> "Cluster":
        from .cluster import Cluster

        cluster_spec["autotermination_minutes"] = int(cluster_spec.get("autotermination_minutes", 30))

        update_details = {
            k: v
            for k, v in self.check_details(
                update=False, details=ClusterDetails(),
                single_user_name=single_user_name,
                **cluster_spec
            ).as_shallow_dict().items()
            if k in _CREATE_ARG_NAMES
        }

        LOGGER.debug(
            "Creating cluster %r with spec %s",
            update_details.get("cluster_name"), update_details,
        )

        client = self.client.workspace_client().clusters

        try:
            details = client.create(**update_details)
        except PermissionDenied as e:
            raise PermissionDenied(
                f"Permission denied when creating cluster {cluster_spec.get('cluster_name')!r}. "
                "Ensure that you have the rights to do it."
            ) from e

        instance = Cluster(service=self, cluster_id=details.cluster_id).set_details(details=details)

        LOGGER.info("Created cluster %r", instance)

        instance.install_libraries(libraries=libraries, raise_error=False, wait=False)

        instance.update_permissions(permissions)

        instance.wait_for_status(wait=wait)

        return instance

    def default(self, *, raise_error: bool = False) -> Optional["Cluster"]:
        """The project's **default all-purpose cluster** — the single-user cluster
        named for the running client project (its capitalized display name,
        :attr:`DatabricksClient.product_name`), as provisioned by
        ``ygg databricks deploy``. Resolves the existing cluster (no creation);
        ``None`` when it isn't deployed unless *raise_error*. The cluster sibling
        of ``dbc.environments.default()`` / ``warehouses.default()``."""
        name = self.client.product_name or "Yggdrasil"
        return self.find_cluster(cluster_name=name, raise_error=raise_error)

    def find_cluster(
        self,
        obj: Any = None,
        *,
        cluster_id: str | None = None,
        cluster_name: str | None = None,
        sources: Optional[list[ClusterSource]] = ...,
        raise_error: bool | None = None,
    ) -> Optional["Cluster"]:
        if obj is not None:
            from .cluster import Cluster

            if isinstance(obj, Cluster):
                return obj
            if isinstance(obj, str):
                if _CLUSTER_ID_RE.match(obj):
                    cluster_id = cluster_id or obj
                else:
                    cluster_name = cluster_name or obj
            else:
                raise TypeError(
                    f"obj must be Cluster | str | None, got {type(obj).__name__}"
                )

        if not cluster_name and not cluster_id:
            raise ValueError("Either name or cluster_id must be provided")

        from .cluster import Cluster

        if not cluster_id and cluster_name:
            cluster_id = get_cached_cluster_id(
                client=self.client,
                cluster_name=cluster_name
            )

        if cluster_id:
            try:
                client = self.client.workspace_client().clusters
                details = client.get(cluster_id=cluster_id)
            except ResourceDoesNotExist:
                if raise_error:
                    raise ValueError(f"Cannot find databricks cluster {cluster_id!r}")
                return None

            # populate name cache for fast future lookups
            if details.cluster_name:
                set_cached_cluster_name(self.client, details.cluster_name, details.cluster_id)

            return Cluster(
                service=self,
                cluster_id=details.cluster_id,
                cluster_name=details.cluster_name,
                details=details,
            )

        # last resort: list scan (expensive)
        for cluster in self.list(name=cluster_name, limit=1, sources=sources):
            if cluster_name == cluster.details.cluster_name:
                set_cached_cluster_name(
                    client=self.client,
                    cluster_name=cluster.cluster_name,
                    cluster_id=cluster.cluster_id,
                )
                return cluster

        if raise_error:
            raise ValueError(f"Cannot find databricks cluster {cluster_name!r}")
        return None

    def list(
        self,
        *,
        name: str | None = None,
        sources: Optional[list[ClusterSource]] = ...,
        limit: int | None = None,
    ) -> Iterator["Cluster"]:
        from .cluster import Cluster

        client = self.client.workspace_client().clusters
        cnt, limit = 0, limit or float("inf")

        if sources is ...:
            sources = [ClusterSource.API, ClusterSource.UI]

        filter_by = ListClustersFilterBy(cluster_sources=sources)

        for details in client.list(filter_by=filter_by):
            if name:
                if name == details.cluster_name:
                    cluster = Cluster(service=self, cluster_id=details.cluster_id).set_details(details=details)
                    yield cluster
                    cnt += 1
            else:
                cluster = Cluster(service=self, cluster_id=details.cluster_id).set_details(details=details)

                yield cluster
                cnt += 1

            if cnt >= limit:
                break
