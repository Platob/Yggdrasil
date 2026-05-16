"""
Instance-pool management helpers for Databricks compute.

Exposes two coordinated surfaces:

- :class:`InstancePools` — collection-level service mirroring the shape of
  :class:`yggdrasil.databricks.cluster.Clusters`. Handles CRUD,
  permissions and find-or-create singleton helpers.
- :class:`InstancePool` — resource wrapper around a single instance pool with
  state helpers, lifecycle operations, and convenience entry points for
  executing Python code on a pool-backed cluster.

The "simplest way to execute Python code" goal is covered by
:meth:`InstancePool.run` / :meth:`InstancePool.decorate` / the module-level
:func:`databricks_pool_remote_compute` decorator. They transparently no-op
when called from inside a Databricks runtime (the function runs locally on the
driver) and otherwise route through a pool-backed cluster.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

from databricks.sdk.client_types import ClientType
from databricks.sdk.errors import PermissionDenied, ResourceDoesNotExist
from databricks.sdk.service.compute import (
    GetInstancePool,
    InstancePoolAccessControlRequest,
    InstancePoolAndStats,
    InstancePoolPermissionLevel,
    InstancePoolsAPI,
    InstancePoolState,
    Library,
)

from yggdrasil.data.cast.registry import identity
from yggdrasil.data.enums import NodeType
from yggdrasil.dataclasses import ExpiringDict
from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io.url import URL
from yggdrasil.pyutils.equality import dicts_equal

from ..client import DatabricksClient, DatabricksResource, DatabricksService

if TYPE_CHECKING:
    from ..cluster.cluster import Cluster


__all__ = [
    "InstancePools",
    "InstancePool",
    "InstancePoolDefaults",
    "DEFAULT_POOL_NAME_PREFIX",
    "databricks_pool_remote_compute",
]


LOGGER = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
ReturnType = TypeVar("ReturnType")

_CREATE_ARG_NAMES = set(inspect.signature(InstancePoolsAPI.create).parameters.keys())
_EDIT_ARG_NAMES = set(inspect.signature(InstancePoolsAPI.edit).parameters.keys())
_GROUPNAME_RE = re.compile(r"\bGroupName\((?P<group>[^)]*)\)")

# host -> ExpiringDict(pool_name -> pool_id)
_NAME_ID_CACHE: dict[str, ExpiringDict] = {}
_NAMED_POOLS: ExpiringDict[str, "InstancePool"] = ExpiringDict(default_ttl=7200.0)

# Default node type used when callers do not supply one explicitly. Sourced
# from the centralized :class:`NodeType` enum so the cluster service, pool
# service, and any downstream caller see the same value.
_DEFAULT_NODE_TYPE_ID = NodeType.DEFAULT.value

# Prefix used by :meth:`InstancePools.default_pool` when no explicit
# pool name is configured on the defaults. The effective name appends a
# per-user slug (email local part / whoami / hostname) at resolution time so
# shared workspaces do not collide on a single ``"Yggdrasil"`` pool.
DEFAULT_POOL_NAME_PREFIX = "Yggdrasil"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstancePoolDefaults:
    """Default configuration applied to instance pools created via :class:`InstancePools`.

    Anywhere the caller leaves a field unset (``None``), the corresponding
    value from this object is injected at create / edit time. Tweak per
    workspace with::

        from dataclasses import replace
        client.compute.instance_pools.defaults = replace(
            client.compute.instance_pools.defaults,
            max_capacity=50,
        )

    or by passing ``defaults=InstancePoolDefaults(...)`` to
    :class:`InstancePools` directly.

    Attributes
    ----------
    pool_name
        Explicit pool name. When ``None`` (the default), the effective name is
        derived at resolution time from :attr:`pool_name_prefix` plus a stable
        per-user slug (e.g. ``"Yggdrasil-alice"``) via
        :meth:`DatabricksClient.user_scoped_name`, so shared workspaces do not
        collide on a single pool.
    pool_name_prefix
        Prefix used when :attr:`pool_name` is ``None``. Set this to namespace
        team-wide pools (e.g. ``"DataPlatform"`` → ``"DataPlatform-alice"``).
    node_type_id
        Default node SKU. Matches the cluster-service default so a pool-backed
        cluster lines up without extra configuration.
    idle_instance_autotermination_minutes
        How long an idle pool node sits before Databricks releases it. ``30``
        keeps cost predictable while still smoothing the second attach.
    min_idle_instances
        Minimum pre-warmed nodes kept alive. ``0`` means fully lazy — the
        cheapest default that still lets the pool be the canonical attach point.
    max_capacity
        Cap on the total pool size (idle + in-use). ``10`` is a safe ceiling
        for shared workspaces; set to ``None`` for unlimited.
    enable_elastic_disk
        Auto-grow attached disk under load. Recommended for ad-hoc workloads.
    preload_local_python_runtime
        When ``True``, :meth:`InstancePools.create` preloads the latest DBR
        runtime whose Python minor matches the local interpreter. Speeds up
        first attach when the caller decorates a local function with
        :func:`databricks_pool_remote_compute`.
    """

    pool_name: Optional[str] = None
    pool_name_prefix: str = DEFAULT_POOL_NAME_PREFIX
    node_type_id: str = _DEFAULT_NODE_TYPE_ID
    idle_instance_autotermination_minutes: int = 30
    min_idle_instances: int = 0
    max_capacity: Optional[int] = 10
    enable_elastic_disk: bool = True
    preload_local_python_runtime: bool = True

    #: Pool-spec fields that :meth:`InstancePools._normalize_pool_spec` injects
    #: when the caller leaves them unset. Restricted to keys accepted by both
    #: ``create`` and ``edit`` so the same defaults survive a pool update.
    _EDITABLE_FIELDS: ClassVar[tuple[str, ...]] = (
        "idle_instance_autotermination_minutes",
        "min_idle_instances",
        "max_capacity",
    )

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow dict view, dropping ``None`` values."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        }


# Module-level singleton; cheap to share since the dataclass is frozen.
DEFAULTS = InstancePoolDefaults()


def _set_cached_pool_id(client: DatabricksClient, name: str, pool_id: str) -> None:
    host = client.base_url.to_string()
    existing = _NAME_ID_CACHE.get(host)
    if not existing:
        existing = _NAME_ID_CACHE[host] = ExpiringDict(default_ttl=60)
    existing[name] = pool_id


def _get_cached_pool_id(client: DatabricksClient, name: str) -> Optional[str]:
    host = client.base_url.to_string()
    existing = _NAME_ID_CACHE.get(host)
    return existing.get(name) if existing else None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class InstancePools(DatabricksService):
    """Collection-level Databricks instance-pool service.

    Mirrors the shape of :class:`yggdrasil.databricks.cluster.Clusters`
    so callers can switch between cluster and pool flows with the same vocabulary.

    The :attr:`defaults` attribute controls auto-configuration of pools created
    through this service. Override at construction time or replace it in place::

        pools = client.compute.instance_pools
        pools.defaults = replace(pools.defaults, max_capacity=50)
    """

    def __init__(
        self,
        client=None,
        defaults: Optional[InstancePoolDefaults] = None,
    ):
        super().__init__(client=client)
        self.defaults = defaults if defaults is not None else InstancePoolDefaults()

    # ------------------------------------------------------------------ #
    # Singletons
    # ------------------------------------------------------------------ #
    def default_pool_name(self) -> str:
        """Resolve the effective default pool name for this workspace.

        Uses :attr:`InstancePoolDefaults.pool_name` verbatim when set,
        otherwise falls back to ``"{pool_name_prefix}-{user_slug}"`` via
        :meth:`DatabricksClient.user_scoped_name`. When no user slug can be
        derived, returns the bare prefix.
        """
        if self.defaults.pool_name:
            return self.defaults.pool_name
        return self.client.user_scoped_name(self.defaults.pool_name_prefix)

    def pool(
        self,
        name: str | None = None,
        *,
        key: str | None = None,
        node_type_id: str | None = None,
        min_idle_instances: int | None = None,
        max_capacity: int | None = None,
        idle_instance_autotermination_minutes: int | None = None,
        custom_tags: Optional[Mapping[str, str]] = None,
        preloaded_spark_versions: Optional[Sequence[str]] = None,
        permissions: Optional[list[str | InstancePoolAccessControlRequest]] = None,
        **pool_spec: Any,
    ) -> "InstancePool":
        """Return a named instance pool, creating it if necessary.

        Cached per-host for ``7200s`` so repeated calls in the same process
        skip the SDK round trip. Any argument left as ``None`` falls back to
        the value stored on :attr:`defaults`; an absent pool name defers to
        :meth:`default_pool_name` (user-scoped by default).
        """
        if not name:
            name = (key.strip() if key else None) or self.default_pool_name()

        existing = _NAMED_POOLS.get(name)
        if existing is not None:
            return existing

        existing = self.find(name=name)
        if existing is None:
            existing = self.create(
                instance_pool_name=name,
                node_type_id=node_type_id,
                min_idle_instances=min_idle_instances,
                max_capacity=max_capacity,
                idle_instance_autotermination_minutes=idle_instance_autotermination_minutes,
                custom_tags=custom_tags,
                preloaded_spark_versions=preloaded_spark_versions,
                permissions=permissions,
                **pool_spec,
            )

        _NAMED_POOLS[name] = existing
        return existing

    def default_pool(self, **overrides: Any) -> "InstancePool":
        """Return the shared default pool used by :meth:`InstancePool.run`.

        The pool name is resolved by :meth:`default_pool_name`, so by default
        every workspace gets its own per-user pool. ``overrides`` win over the
        defaults for this one call.
        """
        return self.pool(self.default_pool_name(), **overrides)

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def _pools_client(self) -> InstancePoolsAPI:
        return self.client.workspace_client().instance_pools

    def _normalize_pool_spec(
        self,
        *,
        update: bool,
        instance_pool_name: Optional[str],
        node_type_id: Optional[str],
        custom_tags: Optional[Mapping[str, str]],
        **pool_spec: Any,
    ) -> dict[str, Any]:
        """Apply :attr:`defaults`, merge tags, and drop unset keys.

        Defaults from :attr:`InstancePools.defaults` are filled in only for
        keys the caller left as ``None``. Explicit values always win.
        """
        spec: dict[str, Any] = {}
        if instance_pool_name is not None:
            spec["instance_pool_name"] = instance_pool_name
        spec["node_type_id"] = NodeType.to_id(
            node_type_id, default=self.defaults.node_type_id,
        )

        # Inject edit-safe defaults for any field the caller left unset.
        for key in InstancePoolDefaults._EDITABLE_FIELDS:
            value = pool_spec.get(key)
            if value is None:
                value = getattr(self.defaults, key, None)
            if value is not None:
                spec[key] = value
            pool_spec.pop(key, None)

        for key, value in pool_spec.items():
            if value is not None:
                spec[key] = value

        default_tags = self.default_tags(update=update)
        if custom_tags:
            merged: dict[str, str] = dict(default_tags) if default_tags else {}
            merged.update({str(k): str(v) for k, v in custom_tags.items()})
            spec["custom_tags"] = merged
        elif default_tags:
            spec["custom_tags"] = dict(default_tags)

        return spec

    def _local_python_preloaded_versions(self) -> Optional[list[str]]:
        """Resolve the latest DBR runtime matching the local Python minor.

        Returns ``None`` on any lookup failure — preloading is a smoothing
        optimization, not a correctness requirement.
        """
        try:
            local = PyEnv.current().version_info
            version = self.client.compute.clusters.latest_spark_version(
                python_version=(local.major, local.minor),
                allow_ml=False,
                allow_gpu=False,
            )
        except Exception:  # noqa: BLE001 - best-effort preload
            LOGGER.debug(
                "Could not resolve local-Python-matching DBR runtime for instance pool preload",
                exc_info=True,
            )
            return None
        return [version.key] if version and version.key else None

    def create(
        self,
        *,
        instance_pool_name: str,
        node_type_id: str | None = None,
        permissions: Optional[list[str | InstancePoolAccessControlRequest]] = None,
        **pool_spec: Any,
    ) -> "InstancePool":
        """Create a new instance pool and return its :class:`InstancePool` wrapper.

        Create-only defaults from :attr:`defaults` are applied here:

        - ``enable_elastic_disk`` is injected when the caller did not supply one.
        - ``preloaded_spark_versions`` is auto-populated with the latest DBR
          whose Python minor matches the local interpreter when
          :attr:`InstancePoolDefaults.preload_local_python_runtime` is set
          and the caller passed nothing. Lookup failures are swallowed.
        """
        if pool_spec.get("enable_elastic_disk") is None and self.defaults.enable_elastic_disk is not None:
            pool_spec["enable_elastic_disk"] = self.defaults.enable_elastic_disk

        if (
            pool_spec.get("preloaded_spark_versions") is None
            and self.defaults.preload_local_python_runtime
        ):
            preloaded = self._local_python_preloaded_versions()
            if preloaded:
                pool_spec["preloaded_spark_versions"] = preloaded

        spec = self._normalize_pool_spec(
            update=False,
            instance_pool_name=instance_pool_name,
            node_type_id=node_type_id,
            custom_tags=pool_spec.pop("custom_tags", None),
            **pool_spec,
        )

        create_kwargs = {k: v for k, v in spec.items() if k in _CREATE_ARG_NAMES}
        LOGGER.debug(
            "Creating instance pool %r with %s", instance_pool_name, create_kwargs
        )

        try:
            response = self._pools_client().create(**create_kwargs)
        except PermissionDenied as exc:
            raise PermissionDenied(
                f"Permission denied when creating instance pool {instance_pool_name!r}. "
                "Ensure that you have 'Can Manage' on instance pools in this workspace."
            ) from exc

        pool_id = response.instance_pool_id
        _set_cached_pool_id(self.client, instance_pool_name, pool_id)

        instance = InstancePool(
            service=self,
            instance_pool_id=pool_id,
            instance_pool_name=instance_pool_name,
        ).refresh()

        LOGGER.info("Created instance pool %r", instance)

        instance.update_permissions(permissions)
        return instance

    def create_or_update(
        self,
        *,
        instance_pool_id: str | None = None,
        instance_pool_name: str | None = None,
        node_type_id: str | None = None,
        permissions: Optional[list[str | InstancePoolAccessControlRequest]] = None,
        **pool_spec: Any,
    ) -> "InstancePool":
        """Update an existing pool by id/name, or create one if missing."""
        found = self.find(pool_id=instance_pool_id, name=instance_pool_name)

        if found is not None:
            return found.update(
                instance_pool_name=instance_pool_name,
                node_type_id=node_type_id,
                permissions=permissions,
                **pool_spec,
            )

        if not instance_pool_name:
            raise ValueError(
                "Cannot create a new instance pool without instance_pool_name; "
                f"pass instance_pool_name=... or an existing instance_pool_id "
                f"(received instance_pool_id={instance_pool_id!r})."
            )

        return self.create(
            instance_pool_name=instance_pool_name,
            node_type_id=node_type_id,
            permissions=permissions,
            **pool_spec,
        )

    def list(
        self,
        *,
        name: str | None = None,
        limit: int | None = None,
    ) -> Iterator["InstancePool"]:
        """Iterate over workspace instance pools, optionally filtered by name."""
        cnt, limit = 0, limit or float("inf")

        for entry in self._pools_client().list():
            if name and entry.instance_pool_name != name:
                continue

            pool = InstancePool(
                service=self,
                instance_pool_id=entry.instance_pool_id,
                instance_pool_name=entry.instance_pool_name,
                details=entry,
            )

            if entry.instance_pool_name and entry.instance_pool_id:
                _set_cached_pool_id(
                    self.client, entry.instance_pool_name, entry.instance_pool_id
                )

            yield pool
            cnt += 1
            if cnt >= limit:
                break

    def find(
        self,
        pool_id: str | None = None,
        *,
        name: str | None = None,
        raise_error: bool | None = None,
    ) -> Optional["InstancePool"]:
        """Look up an instance pool by id or name. Returns ``None`` if absent."""
        if not pool_id and not name:
            raise ValueError("Either pool_id or name must be provided")

        if not pool_id and name:
            pool_id = _get_cached_pool_id(self.client, name)

        if pool_id:
            try:
                details = self._pools_client().get(instance_pool_id=pool_id)
            except ResourceDoesNotExist:
                if raise_error:
                    raise ValueError(f"Cannot find databricks instance pool {pool_id!r}")
                return None

            if details.instance_pool_name:
                _set_cached_pool_id(self.client, details.instance_pool_name, pool_id)

            return InstancePool(
                service=self,
                instance_pool_id=pool_id,
                instance_pool_name=details.instance_pool_name,
                details=details,
            )

        # last resort: list scan
        for pool in self.list(name=name, limit=1):
            return pool

        if raise_error:
            raise ValueError(f"Cannot find databricks instance pool {name!r}")
        return None

    def get(
        self,
        pool_id: str | None = None,
        *,
        name: str | None = None,
    ) -> "InstancePool":
        """Like :meth:`find` but raises if the pool does not exist."""
        return self.find(pool_id=pool_id, name=name, raise_error=True)

    def delete(self, pool_id: str | None = None, *, name: str | None = None) -> None:
        """Delete an instance pool by id or name (no-op if it does not exist)."""
        found = self.find(pool_id=pool_id, name=name)
        if found is not None:
            found.delete()


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------


class InstancePool(Singleton, DatabricksResource):
    """High-level wrapper around a single Databricks instance pool.

    Holds a cached :class:`GetInstancePool` and exposes:

    - lifecycle (:meth:`refresh`, :meth:`update`, :meth:`delete`)
    - state introspection (:attr:`state`, :attr:`is_active`, :attr:`is_pending`)
    - permission management (:meth:`update_permissions`)
    - a :meth:`cluster` helper that returns a cluster attached to this pool
    - :meth:`run` / :meth:`decorate` for the simplest local-or-remote Python
      execution flow built on top of the existing cluster command stack

    Inherits :class:`Singleton` (``_SINGLETON_TTL = None``) so two
    callers asking for the same pool under the same service share the
    cached :class:`GetInstancePool` snapshot and the per-pool name
    resolution.
    """

    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        service: "InstancePools | None" = None,
        instance_pool_id: str | None = None,
        instance_pool_name: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        return (cls, service, instance_pool_id, instance_pool_name)

    def __init__(
        self,
        service: InstancePools | None = None,
        instance_pool_id: str | None = None,
        instance_pool_name: str | None = None,
        *,
        details: Optional[GetInstancePool | InstancePoolAndStats] = None,
        singleton_ttl: Any = ...,
    ):
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return

        super().__init__()
        self.service = service or InstancePools.current()
        self.instance_pool_id = instance_pool_id
        self.instance_pool_name = instance_pool_name
        self._details = details

        if self.instance_pool_name and not self.instance_pool_id:
            found = self.service.find(name=self.instance_pool_name, raise_error=True)
            self.instance_pool_id = found.instance_pool_id
            self._details = found._details

        self._initialized = True

    # ------------------------------------------------------------------ #
    # Display / identity
    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        return self.explore_url.to_string()

    def __hash__(self) -> int:
        return hash(self.explore_url)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, InstancePool) and self.explore_url == other.explore_url

    @property
    def explore_url(self) -> URL:
        """Workspace UI URL pointing at this instance pool's page."""
        return self.client.base_url.with_path(
            f"/compute/instance-pools/{self.instance_pool_id or 'unknown'}"
        )

    def url(self) -> URL:
        """Deprecated alias for :attr:`explore_url` (method form)."""
        return self.explore_url

    # ------------------------------------------------------------------ #
    # SDK client
    # ------------------------------------------------------------------ #
    def _pools_client(self) -> InstancePoolsAPI:
        return self.client.workspace_client().instance_pools

    # ------------------------------------------------------------------ #
    # Details
    # ------------------------------------------------------------------ #
    @property
    def details(self) -> Optional[GetInstancePool | InstancePoolAndStats]:
        """Return cached pool details, fetching lazily when not yet loaded."""
        if self._details is None and self.instance_pool_id:
            self._details = self._pools_client().get(
                instance_pool_id=self.instance_pool_id,
            )
            if self._details.instance_pool_name:
                self.instance_pool_name = self._details.instance_pool_name
        return self._details

    def refresh(self) -> "InstancePool":
        """Force a refresh of the cached pool details and return self."""
        if self.instance_pool_id:
            self._details = self._pools_client().get(
                instance_pool_id=self.instance_pool_id,
            )
            if self._details.instance_pool_name:
                self.instance_pool_name = self._details.instance_pool_name
        return self

    @property
    def state(self) -> Optional[InstancePoolState]:
        """Return the latest pool state."""
        details = self.details
        return getattr(details, "state", None) if details else None

    @property
    def is_active(self) -> bool:
        """Whether the pool is currently ``ACTIVE``."""
        return self.state == InstancePoolState.ACTIVE

    @property
    def is_pending(self) -> bool:
        """Pools have no transitional state today, but expose the helper for
        symmetry with :class:`Cluster` so callers can write the same wait loop.
        """
        return False

    @property
    def node_type_id(self) -> Optional[str]:
        details = self.details
        return getattr(details, "node_type_id", None) if details else None

    # ------------------------------------------------------------------ #
    # Update / lifecycle
    # ------------------------------------------------------------------ #
    def update(
        self,
        *,
        instance_pool_name: str | None = None,
        node_type_id: str | None = None,
        permissions: Optional[list[str | InstancePoolAccessControlRequest]] = None,
        **pool_spec: Any,
    ) -> "InstancePool":
        """Update the pool's editable fields. ``edit`` requires name + node type."""
        details = self.details
        if details is None:
            raise ValueError(f"Cannot update {self}: pool details are not available")

        current_name = instance_pool_name or self.instance_pool_name or details.instance_pool_name
        current_node = node_type_id or details.node_type_id or _DEFAULT_NODE_TYPE_ID

        desired = self.service._normalize_pool_spec(
            update=True,
            instance_pool_name=current_name,
            node_type_id=current_node,
            custom_tags=pool_spec.pop("custom_tags", None),
            **pool_spec,
        )
        desired_edit = {k: v for k, v in desired.items() if k in _EDIT_ARG_NAMES}
        desired_edit["instance_pool_id"] = self.instance_pool_id

        current_edit = {
            k: getattr(details, k, None)
            for k in _EDIT_ARG_NAMES
            if k != "instance_pool_id"
        }

        if dicts_equal(current_edit, desired_edit, keys=_EDIT_ARG_NAMES - {"instance_pool_id"}):
            self.update_permissions(permissions)
            return self

        LOGGER.debug("Updating instance pool %r with %s", self, desired_edit)
        self._pools_client().edit(**desired_edit)

        self.refresh()
        self.update_permissions(permissions)
        LOGGER.info("Updated instance pool %r", self)
        return self

    def delete(self) -> None:
        """Delete the pool if it exists. Also drops the named-pool cache entry."""
        if not self.instance_pool_id:
            return

        LOGGER.debug("Deleting instance pool %r", self)
        self._pools_client().delete(instance_pool_id=self.instance_pool_id)
        if self.instance_pool_name:
            _NAMED_POOLS.pop(self.instance_pool_name, None)
        LOGGER.info("Deleted instance pool %r", self)

    # ------------------------------------------------------------------ #
    # Permissions
    # ------------------------------------------------------------------ #
    def update_permissions(
        self,
        permissions: Optional[list[str | InstancePoolAccessControlRequest]] = None,
    ) -> "InstancePool":
        """Apply ACL entries to this pool, creating missing groups on demand."""
        if not permissions:
            return self

        normalized = self._check_permission(permissions)

        try:
            self._pools_client().update_permissions(
                instance_pool_id=self.instance_pool_id,
                access_control_list=normalized,
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
        permission: Union[
            str,
            InstancePoolAccessControlRequest,
            list[Union[str, InstancePoolAccessControlRequest]],
        ],
    ):
        if isinstance(permission, InstancePoolAccessControlRequest):
            return permission

        if isinstance(permission, str):
            if "@" in permission:
                group_name, user_name = None, permission
            else:
                group_name, user_name = permission, None

            return InstancePoolAccessControlRequest(
                group_name=group_name,
                user_name=user_name,
                permission_level=InstancePoolPermissionLevel.CAN_MANAGE,
            )

        return [self._check_permission(item) for item in permission]

    # ------------------------------------------------------------------ #
    # Cluster bound to this pool
    # ------------------------------------------------------------------ #
    def cluster(
        self,
        name: str | None = None,
        *,
        key: str | None = None,
        single_user_name: str | None = None,
        libraries: Optional[Sequence[str | Library]] = None,
        permissions: Optional[list[str]] = None,
        custom_tags: Optional[Mapping[str, str]] = None,
        wait: WaitingConfigArg = True,
        **cluster_spec: Any,
    ) -> "Cluster":
        """Return a cluster attached to this pool, creating one if needed.

        Routes through :meth:`Clusters.all_purpose_cluster` so the existing
        named-cluster cache, library install path, and permission flow apply
        unchanged. ``instance_pool_id`` is force-injected and any explicit
        ``node_type_id`` is dropped — node type comes from the pool itself.
        """
        # Pool-backed clusters MUST omit node_type_id.
        cluster_spec.pop("node_type_id", None)
        cluster_spec["instance_pool_id"] = self.instance_pool_id

        if not name:
            name = (key.strip() if key else None) or (
                f"{self.instance_pool_name or 'pool'}-cluster"
            )

        return self.client.compute.clusters.all_purpose_cluster(
            name=name,
            key=key,
            single_user_name=single_user_name,
            libraries=list(libraries) if libraries else None,
            permissions=permissions,
            custom_tags=dict(custom_tags) if custom_tags else None,
            wait=wait,
            **cluster_spec,
        )

    # ------------------------------------------------------------------ #
    # Execute Python code in the simplest way
    # ------------------------------------------------------------------ #
    def run(
        self,
        func: Callable[..., ReturnType],
        /,
        *args: Any,
        env_keys: Optional[List[str]] = None,
        cluster_name: str | None = None,
        cluster_kwargs: Optional[Mapping[str, Any]] = None,
        force_local: bool = False,
        **kwargs: Any,
    ) -> ReturnType:
        """Execute ``func(*args, **kwargs)`` either locally or on a pool cluster.

        When the caller is already inside a Databricks runtime (driver node) —
        or ``force_local`` is set — ``func`` runs locally and the return value
        is returned as-is. Otherwise, a pool-backed cluster is resolved
        (creating it if needed) and the call is routed through the existing
        :class:`yggdrasil.databricks.compute.command_execution.CommandExecution`
        pipeline.

        ``env_keys`` forwards the listed local environment variables to the
        remote command. ``cluster_kwargs`` is passed through to
        :meth:`cluster` so callers can customise the spec (libraries,
        single_user_name, …) on first creation.
        """
        if force_local or DatabricksClient.is_in_databricks_environment():
            return func(*args, **kwargs)

        cluster = self.cluster(name=cluster_name, **(cluster_kwargs or {}))
        decorated = cluster.command(func=func, environ=env_keys)
        return decorated(*args, **kwargs)

    def decorate(
        self,
        _func: Optional[Callable[..., ReturnType]] = None,
        *,
        env_keys: Optional[List[str]] = None,
        cluster_name: str | None = None,
        cluster_kwargs: Optional[Mapping[str, Any]] = None,
        force_local: bool = False,
    ) -> Callable[..., ReturnType]:
        """Decorator form of :meth:`run`. Supports ``@pool.decorate`` and
        ``@pool.decorate(env_keys=[...], cluster_name=...)`` shapes.

        When invoked from inside Databricks (or with ``force_local=True``), the
        decorator collapses to :func:`yggdrasil.data.cast.registry.identity` so
        the wrapped function runs locally with no proxying overhead.
        """
        if force_local or DatabricksClient.is_in_databricks_environment():
            return identity if _func is None else _func

        def _wrap(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
            def _runner(*args: Any, **kwargs: Any) -> ReturnType:
                return self.run(
                    func,
                    *args,
                    env_keys=env_keys,
                    cluster_name=cluster_name,
                    cluster_kwargs=cluster_kwargs,
                    **kwargs,
                )

            _runner.__wrapped__ = func  # type: ignore[attr-defined]
            _runner.__name__ = getattr(func, "__name__", "pool_runner")
            _runner.__doc__ = getattr(func, "__doc__", None)
            return _runner

        return _wrap if _func is None else _wrap(_func)


# ---------------------------------------------------------------------------
# Top-level convenience decorator (mirrors ``databricks_remote_compute``)
# ---------------------------------------------------------------------------


def databricks_pool_remote_compute(
    _func: Optional[Callable[..., ReturnType]] = None,
    *,
    pool_id: str | None = None,
    pool_name: str | None = None,
    workspace: Optional[Union[DatabricksClient, str]] = None,
    pool: Optional[InstancePool] = None,
    env_keys: Optional[List[str]] = None,
    cluster_name: str | None = None,
    cluster_kwargs: Optional[Mapping[str, Any]] = None,
    force_local: bool = False,
) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    """Decorate a function so it runs on a Databricks instance-pool cluster.

    Mirrors :func:`yggdrasil.databricks.compute.remote.databricks_remote_compute`
    but resolves the target compute via :class:`InstancePool` rather than a
    standalone cluster. When called from inside a Databricks runtime — or when
    no workspace is configured — the function executes locally untouched.

    Resolution order for the pool: ``pool`` argument → ``pool_id`` /
    ``pool_name`` lookup on the resolved workspace → the workspace's default
    pool (creates ``"ygg-default-pool"`` if missing).
    """
    import os

    if force_local or DatabricksClient.is_in_databricks_environment():
        return identity if _func is None else _func

    if workspace is None:
        workspace = os.getenv("DATABRICKS_HOST")

    if workspace is None:
        return identity if _func is None else _func

    workspace = DatabricksClient.parse(workspace)
    pools = workspace.compute.instance_pools

    if pool is None:
        if pool_id or pool_name:
            pool = pools.find(pool_id=pool_id, name=pool_name, raise_error=True)
        else:
            pool = pools.default_pool()

    return pool.decorate(
        _func,
        env_keys=env_keys,
        cluster_name=cluster_name,
        cluster_kwargs=cluster_kwargs,
    )
