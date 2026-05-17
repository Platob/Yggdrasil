"""Per-volume resource: lifecycle, metadata caching, credential vending.

:class:`Volume` wraps a single Unity Catalog volume and is **the**
home for everything that used to live as ad-hoc state on
:class:`VolumePath`:

- :class:`VolumeInfo` lookup with a 5-minute TTL.
- ``READ_VOLUME`` / ``WRITE_VOLUME`` temporary-credential vending,
  routed through the process-wide
  :class:`AWSDatabricksVolumeCredentials` refresher.
- Storage-location → :class:`Path` resolution (S3 today; other
  backends fall back to the generic Path registry).
- Lifecycle (``create`` / ``ensure_created`` / ``delete``) and
  the bottom-up ``_ensure_volume`` recovery used by
  :class:`VolumePath` when a write hits a missing catalog / schema
  / volume.

Identity & singleton caching
----------------------------

Instances are **singletons per ``(host, catalog, schema, name)``**.
Two callers asking for the same UC volume — directly via
``client.volumes["cat.sch.vol"]`` or transitively via
``volume_path.volume`` — collapse to the same :class:`Volume`,
share one cached :class:`VolumeInfo`, and reuse the same
credentials refresher.

The cached metadata refreshes on first access past the 5-minute
TTL window; pass ``refresh=True`` to any of the info-reading
methods to force a fresh ``volumes.read`` immediately.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, ClassVar, Iterator, Mapping, Optional, TYPE_CHECKING, Tuple

from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import (
    PermissionsChange,
    Privilege,
    PrivilegeAssignment,
    SecurableType,
    VolumeInfo,
    VolumeOperation,
)

from yggdrasil.concurrent.threading import Job
from yggdrasil.data.enums import Mode, ModeLike, Scheme
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.aws import AWSDatabricksVolumeCredentials
from yggdrasil.databricks.client import DatabricksResource
from yggdrasil.dataclasses import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io.url import URL

if TYPE_CHECKING:
    from yggdrasil.aws.client import AWSClient
    from yggdrasil.databricks.catalog.catalog import Catalog
    from yggdrasil.databricks.fs.volume_path import VolumePath
    from yggdrasil.databricks.schema.schema import Schema as UCSchema
    from yggdrasil.databricks.volume.volumes import Volumes

__all__ = ["Volume"]

logger = logging.getLogger(__name__)


class Volume(DatabricksResource, Singleton):
    """A single Unity Catalog volume — metadata, credentials, storage path.

    Construct via :meth:`Volumes.volume` /
    :meth:`Volumes.__getitem__` for the cache-aware path, or
    directly when callers already have the coordinates. Either way
    the instance is interned per ``(client, catalog, schema, name)``
    so subsequent calls collapse to the same live cache — same
    convention :class:`Catalog` / :class:`Schema` / :class:`Table`
    use.
    """

    DEFAULT_INFO_TTL: ClassVar[float] = 300.0  # 5 minutes

    # Per-class singleton cache so this surface stays separate from
    # the rest of the project's :class:`Singleton` users.
    _INSTANCES: ClassVar = Singleton._INSTANCES.__class__(default_ttl=None)
    # Cache every Volume under the singleton convention; the cached
    # ``VolumeInfo`` and credentials refresher are worth keeping for
    # the process lifetime so navigation / repeated reads don't keep
    # refetching.
    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        service: "Volumes | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        # Key on the bound :class:`DatabricksClient` *instance* + the
        # three-part name. Same convention as :class:`Catalog` /
        # :class:`Schema` / :class:`Table`.
        client = None
        try:
            client = service.client if service is not None else None
        except Exception:
            client = None
        return (cls, client, catalog_name, schema_name, volume_name)

    def __new__(
        cls,
        service: "Volumes | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
        *,
        singleton_ttl: "int | None" = ...,
        **_kwargs: Any,
    ) -> "Volume":
        if not (catalog_name and schema_name and volume_name):
            raise ValueError(
                f"Volume requires catalog_name + schema_name + volume_name "
                f"(got {catalog_name!r}, {schema_name!r}, {volume_name!r})."
            )
        if singleton_ttl is ...:
            singleton_ttl = cls._SINGLETON_TTL

        def _allocate() -> "Volume":
            return object.__new__(cls)

        if singleton_ttl is ...:
            return _allocate()

        key = cls._singleton_key(
            service,
            catalog_name=str(catalog_name),
            schema_name=str(schema_name),
            volume_name=str(volume_name),
        )
        with cls._INSTANCES_LOCK:
            existing = cls._INSTANCES.get(key)
            if existing is not None:
                return existing
            instance = _allocate()
            try:
                object.__setattr__(instance, "_singleton_key_", key)
            except AttributeError:
                pass
            ttl_arg = (
                float(singleton_ttl)
                if isinstance(singleton_ttl, int) and not isinstance(singleton_ttl, bool)
                else singleton_ttl
            )
            cls._INSTANCES.set(key, instance, ttl=ttl_arg)
            return instance

    def __init__(
        self,
        service: "Volumes | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
        *,
        infos: VolumeInfo | None = None,
        infos_fetched_at: float | None = None,
        infos_ttl: float | None = None,
        singleton_ttl: "int | None" = ...,
    ) -> None:
        # ``singleton_ttl`` is consumed by ``__new__``; accept here so
        # the auto-init pass after ``__new__`` doesn't trip on it.
        del singleton_ttl
        # Singleton-cached instances are re-entered on every constructor
        # call (Python always invokes __init__ after __new__). Skip the
        # second pass so the live cache survives — but rebind the service
        # so the latest workspace handle wins, and accept newer ``infos``
        # so callers can refresh the cached entry through the constructor.
        if getattr(self, "_initialized", False):
            if service is not None:
                self.service = service
            if infos is not None:
                self._store_infos(infos, fetched_at=infos_fetched_at)
            return
        if service is None:
            from .volumes import Volumes
            service = Volumes.current()
        super().__init__(service=service)
        self.catalog_name = str(catalog_name)
        self.schema_name = str(schema_name)
        self.volume_name = str(volume_name)
        self._infos_ttl: float = (
            self.DEFAULT_INFO_TTL if infos_ttl is None else float(infos_ttl)
        )
        self._infos: Optional[VolumeInfo] = infos
        self._infos_fetched_at: Optional[float] = (
            infos_fetched_at if (infos is not None and infos_fetched_at is not None)
            else (time.time() if infos is not None else None)
        )
        self._storage_path: Any = None
        self._catalog: Any = None
        self._schema: Any = None
        self._initialized = True

    # ── identity ──────────────────────────────────────────────────────────────

    def full_name(self, safe: str | bool | None = None) -> str:
        """Return the three-part volume name, optionally backtick-quoted."""
        if safe:
            q = safe if isinstance(safe, str) else "`"
            return f"{q}{self.catalog_name}{q}.{q}{self.schema_name}{q}.{q}{self.volume_name}{q}"
        return f"{self.catalog_name}.{self.schema_name}.{self.volume_name}"

    @property
    def name(self) -> str:
        return self.volume_name

    def __str__(self) -> str:
        return self.full_name()

    def __hash__(self) -> int:
        return hash((type(self), self.catalog_name, self.schema_name, self.volume_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Volume):
            return NotImplemented
        return (
            self.catalog_name == other.catalog_name
            and self.schema_name == other.schema_name
            and self.volume_name == other.volume_name
        )

    # ── URL ───────────────────────────────────────────────────────────────────

    @property
    def url(self) -> URL:
        """Workspace UI URL pointing at this volume's Catalog Explorer page."""
        return self.client.base_url.with_path(
            f"/explore/data/volumes/{self.catalog_name}/{self.schema_name}/{self.volume_name}"
        )

    @property
    def explore_url(self) -> URL:
        return self.url

    @classmethod
    def from_url(cls, url: "URL | str", **kwargs: Any) -> "Volume":
        """Build a :class:`Volume` from a ``dbfs+volume:///cat/sch/vol`` URL.

        Used by the :class:`DatabricksPath` dispatcher when a caller
        passes a POSIX volume path that resolves exactly to volume
        depth (``DatabricksPath("/Volumes/main/sales/staging")`` →
        ``Volume("main", "sales", "staging")``); deeper paths resolve
        to :class:`VolumePath` instead.
        """
        from .volumes import Volumes

        u = URL.from_(url)
        parts = [p for p in (u.path or "/").lstrip("/").split("/") if p]
        if len(parts) < 3:
            raise ValueError(
                f"Cannot derive volume name from URL {u!r} — expected "
                f"three path segments "
                f"(e.g. ``dbfs+volume:///main/sales/staging``)."
            )
        catalog_name, schema_name, volume_name = parts[0], parts[1], parts[2]
        client = kwargs.pop("client", None)
        if client is None:
            client = (
                DatabricksClient(host=f"https://{u.host}/")
                if u.host else DatabricksClient.current()
            )
        service = kwargs.pop("service", None) or Volumes(client=client)
        return cls(
            service=service,
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name,
            **kwargs,
        )

    # ── pickle ────────────────────────────────────────────────────────────────

    def __getnewargs_ex__(self):
        # Pickle uses these to invoke ``__new__`` on the receiver. Carrying
        # the live ``service`` (rather than ``None``) does two things:
        # (1) ``__new__`` computes the singleton key against the *source*
        # host instead of falling back to ``Volumes.current()`` in the
        # receiver — so the cache key doesn't drift; (2) the underlying
        # :class:`DatabricksClient` rides along inside ``Volumes``'
        # picklable state (its ``__getstate__`` carries the client, host,
        # auth fields, and session-token snapshot), so the unpickled
        # Volume reuses the original client config instead of re-resolving
        # auth against whatever ``DatabricksClient.current()`` returns on
        # the receiving side.
        return (), {
            "service": self.service,
            "catalog_name": self.catalog_name,
            "schema_name": self.schema_name,
            "volume_name": self.volume_name,
        }

    def __getstate__(self):
        # ``service`` is also carried by ``__getnewargs_ex__``; pickle
        # memoization collapses the two references to a single payload.
        # Restating it here means ``__setstate__`` doesn't have to reach
        # for ``Volumes.current()`` on the receiver — the source client
        # wins.
        return {
            "service": self.service,
            "catalog_name": self.catalog_name,
            "schema_name": self.schema_name,
            "volume_name": self.volume_name,
            "_infos_ttl": self._infos_ttl,
            "_infos": self._infos,
            "_infos_fetched_at": self._infos_fetched_at,
        }

    def __setstate__(self, state):
        carried_service = state["service"]
        if getattr(self, "_initialized", False):
            # Singleton cache hit — keep the live instance but rebind to
            # the carried service (mirrors ``__init__``'s rebind-on-cache-
            # hit behavior) so the freshly-pickled client wins.
            self.service = carried_service
            return
        self.service = carried_service
        self.catalog_name = state["catalog_name"]
        self.schema_name = state["schema_name"]
        self.volume_name = state["volume_name"]
        self._infos_ttl = state.get("_infos_ttl", self.DEFAULT_INFO_TTL)
        self._infos = state.get("_infos")
        self._infos_fetched_at = state.get("_infos_fetched_at")
        self._storage_path = None
        self._catalog = None
        self._schema = None
        self._initialized = True

    # ── cache management ──────────────────────────────────────────────────────

    def _is_fresh(self) -> bool:
        if self._infos is None or self._infos_fetched_at is None:
            return False
        if self._infos_ttl is None:
            return True
        return (time.time() - self._infos_fetched_at) < self._infos_ttl

    def _store_infos(
        self, info: VolumeInfo, *, fetched_at: float | None = None,
    ) -> VolumeInfo:
        self._infos = info
        self._infos_fetched_at = (
            float(fetched_at) if fetched_at is not None else time.time()
        )
        # Storage location may have shifted (external volume rebind);
        # drop the cached Path so the next call resolves against the
        # fresh URL.
        self._storage_path = None
        return info

    def _reset_cache(self) -> None:
        self._infos = None
        self._infos_fetched_at = None
        self._storage_path = None
        self._catalog = None
        self._schema = None

    def clear(self) -> "Volume":
        """Public alias for :meth:`_reset_cache`; returns ``self``."""
        self._reset_cache()
        return self

    # ── metadata ──────────────────────────────────────────────────────────────

    @property
    def infos(self) -> VolumeInfo:
        """Alias for :attr:`info` — matches the ``Schema`` / ``Table`` shape."""
        return self.info

    @property
    def info(self) -> VolumeInfo:
        """Cached :class:`VolumeInfo` (5-minute TTL by default)."""
        return self.read_info()

    def read_info(self, *, refresh: bool = False) -> VolumeInfo:
        """Return the SDK's :class:`VolumeInfo` for this volume.

        Refreshes whenever the cached entry is past
        :attr:`DEFAULT_INFO_TTL` (5 minutes by default), or when
        ``refresh=True`` forces it. If the underlying ``volumes.read``
        raises :class:`NotFound`, the missing pieces of
        catalog / schema / volume are created on demand via
        :meth:`_ensure_volume` and the read is retried exactly once.
        """
        if not refresh and self._is_fresh():
            return self._infos  # type: ignore[return-value]

        try:
            info = self.client.workspace_client().volumes.read(self.full_name())
        except Exception as exc:
            if not _looks_like_not_found(exc):
                raise
            if not self._ensure_volume():
                raise
            info = self.client.workspace_client().volumes.read(self.full_name())
        return self._store_infos(info)

    @property
    def exists(self) -> bool:
        """``True`` if this volume is reachable via the Unity Catalog API."""
        try:
            _ = self.read_info()
            return True
        except NotFound:
            return False
        except DatabricksError as exc:
            if _looks_like_not_found(exc):
                return False
            raise

    @property
    def volume_id(self) -> str:
        vid = getattr(self.info, "volume_id", None)
        if not vid:
            raise ValueError(
                f"{self!r}: volume_info has no volume_id; Unity Catalog "
                f"hasn't finished registering this volume yet."
            )
        return str(vid)

    @property
    def volume_type(self) -> Optional[str]:
        vt = getattr(self.info, "volume_type", None)
        return getattr(vt, "value", None) or (str(vt) if vt is not None else None)

    @property
    def comment(self) -> Optional[str]:
        return getattr(self.info, "comment", None)

    @property
    def owner(self) -> Optional[str]:
        return getattr(self.info, "owner", None)

    # ── storage location / temporary credentials ──────────────────────────────

    def storage_location(self, *, refresh: bool = False) -> str:
        """Volume's backing storage URL string (e.g. ``s3://bucket/...``).

        Pure read from :meth:`read_info` — no AWS auth resolution, no
        Path construction. Use this when you only need the URL
        rendering for logging / config plumbing; reach for
        :meth:`storage_path` when you'll do actual I/O against the
        location.
        """
        info = self.read_info(refresh=refresh)
        raw = getattr(info, "storage_location", None)
        if not raw:
            raise ValueError(
                f"{self!r}: volume has no storage_location yet. "
                f"VolumeInfo: {info!r}."
            )
        return str(raw)

    def storage_path(
        self,
        *,
        mode: ModeLike = Mode.AUTO,
        region: Optional[str] = None,
        refresh: bool = False,
    ) -> Any:
        """Return the volume's root storage :class:`Path`.

        Dispatches to the right :class:`URLBased` subclass —
        :class:`S3Path` for ``s3://``, generic Path registry for
        anything else. The returned Path is cached on the instance
        and (for S3) carries the auto-refreshing :class:`AWSClient`
        session minted via :meth:`credentials_refresher`.
        """
        if self._storage_path is not None and not refresh:
            return self._storage_path

        from yggdrasil.io.path import Path

        raw = self.storage_location(refresh=refresh)
        scheme = URL.from_str(raw).scheme or ""
        if scheme.startswith("s3"):
            storage_path = self.aws(mode=mode, region=region).s3.path(raw)
        else:
            storage_path = Path.from_url(raw)
        self._storage_path = storage_path
        return storage_path

    def temporary_credentials(self, *, mode: ModeLike = Mode.AUTO) -> Any:
        """Vend temporary cloud credentials for this volume.

        Wraps ``temporary_volume_credentials.generate_temporary_volume_credentials``.
        Read-only modes map to ``READ_VOLUME``; everything else to
        ``WRITE_VOLUME``.
        """
        resolved = Mode.from_(mode, default=Mode.READ_ONLY)
        operation = (
            VolumeOperation.READ_VOLUME
            if resolved is Mode.READ_ONLY
            else VolumeOperation.WRITE_VOLUME
        )
        return (
            self.client.workspace_client()
            .temporary_volume_credentials
            .generate_temporary_volume_credentials(
                volume_id=self.volume_id,
                operation=operation,
            )
        )

    def credentials_refresher(self) -> AWSDatabricksVolumeCredentials:
        """Return the process-wide singleton credentials provider for
        this volume.

        Keyed by ``volume_id`` — every :class:`Volume` / :class:`VolumePath`
        pointing at the same UC volume collapses to one provider.
        """
        return AWSDatabricksVolumeCredentials(
            volume_id=self.volume_id,
            client=self.client,
        )

    def aws(
        self,
        *,
        mode: ModeLike = None,
        region: Optional[str] = None,
    ) -> "AWSClient":
        """Return an :class:`AWSClient` whose credentials self-refresh
        from :meth:`temporary_credentials`."""
        return self.credentials_refresher().aws_client(mode=mode, region=region)

    def arrow_filesystem(
        self,
        *,
        mode: ModeLike = None,
        region: Optional[str] = None,
    ) -> Any:
        """Build a :class:`pyarrow.fs.S3FileSystem` for this volume."""
        # Ensure storage path is resolved so the AWSClient session is live.
        self.storage_path(mode=mode, region=region)
        return self.aws(mode=mode, region=region).s3.arrow_filesystem(region=region)

    # ── navigation ────────────────────────────────────────────────────────────

    @property
    def catalog(self) -> "Catalog":
        """Navigate up to the parent :class:`Catalog`.

        Returns the singleton-cached :class:`Catalog` for this
        client + catalog name — repeated calls hand back the same
        instance with shared :class:`CatalogInfo` cache. The
        per-instance ``_catalog`` slot is kept as a one-attribute
        shortcut so the navigation path skips the
        ``client.catalogs.catalog(...)`` round trip on hot loops.
        """
        if self._catalog is None:
            self._catalog = self.client.catalogs.catalog(self.catalog_name)
        return self._catalog

    @property
    def schema(self) -> "UCSchema":
        """Navigate up to the parent :class:`Schema`.

        Returns the singleton-cached :class:`Schema` for this
        client + (catalog, schema) — repeated calls hand back the
        same instance with shared :class:`SchemaInfo` cache. The
        per-instance ``_schema`` slot is kept as a one-attribute
        shortcut so the navigation path skips the
        ``client.schemas.schema(...)`` round trip on hot loops.
        """
        if self._schema is None:
            self._schema = self.client.schemas.schema(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
            )
        return self._schema

    def path(
        self,
        sub: str = "",
        *,
        url: URL | None = None,
        volume: "Volume | None" = None,
        client: "DatabricksClient | None" = None,
        temporary: bool = False,
        **kwargs
    ) -> "VolumePath":
        """Return a :class:`VolumePath` rooted at this volume.

        ``sub`` is appended under the volume root (``""`` → the volume
        itself, ``"sub/x.parquet"`` → ``/Volumes/<cat>/<sch>/<vol>/sub/x.parquet``).
        """
        from yggdrasil.databricks.fs.volume_path import VolumePath

        leaf = "/" + sub.lstrip("/") if sub else ""

        return VolumePath(
            url=url or URL(
                scheme=Scheme.DATABRICKS_VOLUME,
                path=f"/{self.catalog_name}/{self.schema_name}/{self.volume_name}{leaf}",
            ),
            volume=volume or self,
            client=client or self.client,
            temporary=temporary,
            **kwargs
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def create(
        self,
        *,
        comment: str | None = None,
        storage_location: str | None = None,
        volume_type: Any = None,
        if_not_exists: bool = True,
    ) -> "Volume":
        """Create this volume in Unity Catalog.

        Defaults to a managed volume. Pass ``storage_location`` +
        ``volume_type="EXTERNAL"`` for an external volume.
        """
        uc = self.client.workspace_client().volumes
        vt = volume_type if volume_type is not None else _managed_volume_type()
        kwargs: dict[str, Any] = {
            "catalog_name": self.catalog_name,
            "schema_name": self.schema_name,
            "name": self.volume_name,
            "volume_type": vt,
        }
        if comment is not None:
            kwargs["comment"] = comment
        if storage_location is not None:
            kwargs["storage_location"] = storage_location

        try:
            info = uc.create(**kwargs)
            self._store_infos(info)
        except DatabricksError as exc:
            if if_not_exists and "already exists" in str(exc).lower():
                self._reset_cache()
            else:
                raise
        return self

    def ensure_created(
        self,
        *,
        comment: str | None = None,
        storage_location: str | None = None,
        volume_type: Any = None,
    ) -> "Volume":
        """Create this volume if it does not already exist, then return ``self``."""
        if not self.exists:
            self.create(
                comment=comment,
                storage_location=storage_location,
                volume_type=volume_type,
                if_not_exists=True,
            )
        return self

    def delete(
        self,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Volume":
        """Delete this volume from Unity Catalog."""
        uc = self.client.workspace_client().volumes
        if wait:
            try:
                uc.delete(name=self.full_name())
            except DatabricksError:
                if raise_error:
                    raise
        else:
            Job.make(uc.delete, self.full_name()).fire_and_forget()
        self._reset_cache()
        return self

    def _ensure_schema_and_catalog(self) -> None:
        """Best-effort create of the parent schema (and catalog if needed).

        Mirrors :meth:`_ensure_volume`'s middle/last legs without
        touching the volume itself — so :meth:`Volumes.create` can
        recover from a "schema missing" NotFound and then retry the
        volume create with the *caller's* args (comment,
        storage_location, volume_type) intact, rather than letting
        :meth:`_ensure_volume` materialise a default-managed volume
        on the way up.
        """
        ws = self.client.workspace_client()
        try:
            ws.schemas.create(name=self.schema_name, catalog_name=self.catalog_name)
        except Exception as exc:
            if _looks_like_already_exists(exc):
                return
            if _looks_like_not_found(exc):
                # Catalog also missing — create catalog, then schema.
                _safe_create(lambda: ws.catalogs.create(name=self.catalog_name))
                _safe_create(
                    lambda: ws.schemas.create(
                        name=self.schema_name, catalog_name=self.catalog_name,
                    ),
                )
                return
            raise

    def _ensure_volume(self) -> bool:
        """Bottom-up create of any missing catalog / schema / volume.

        Used by :meth:`read_info` when ``volumes.read`` returns
        ``NotFound``, and by :class:`VolumePath` when a write hits a
        missing target. Routes the volume create through
        :meth:`create` (which warms ``_store_infos``) and delegates
        parent creation to :meth:`_ensure_schema_and_catalog`;
        ``AlreadyExists`` is swallowed so concurrent creators don't
        fight. Returns ``True`` if at least one create landed.

        ``self.create`` rather than ``self.service.create`` here on
        purpose: :meth:`Volumes.create` calls ``_ensure_volume`` for
        its own bottom-up recovery, so routing back through the
        service would recurse.
        """
        logger.debug("Ensuring volume %r exists", self)

        # 1) Try volume only — common case where catalog + schema exist.
        # ``if_not_exists=False`` so AlreadyExists surfaces here and
        # we can distinguish "already there" (return False) from
        # "newly created" (return True).
        try:
            self.create(if_not_exists=False)
            return True
        except Exception as exc:
            if _looks_like_already_exists(exc):
                self._reset_cache()
                return False
            if not _looks_like_not_found(exc):
                raise
            # Fall through: a parent (schema or catalog) is missing.

        # 2) Schema (and maybe catalog) missing — create them, then
        #    retry the volume.
        self._ensure_schema_and_catalog()
        _safe_create(lambda: self.create(if_not_exists=False))
        return True

    # ── grants ────────────────────────────────────────────────────────────────

    def _grants_securable_type(self) -> SecurableType:
        return SecurableType.VOLUME

    def _grants_full_name(self) -> str:
        return self.full_name()

    def permissions(
        self,
        *,
        principal: str | None = None,
    ) -> tuple[PrivilegeAssignment, ...]:
        """Direct grants on this volume (no inherited privileges)."""
        kwargs: dict[str, Any] = {}
        if principal is not None:
            kwargs["principal"] = principal
        response = self.client.workspace_client().grants.get(
            securable_type=SecurableType.VOLUME.value,
            full_name=self.full_name(),
            **kwargs,
        )
        return tuple(response.privilege_assignments or ())

    def grant(
        self,
        principal: str,
        privileges: "str | Privilege | list[str | Privilege]",
    ) -> "Volume":
        """Add one or more privileges for *principal* on this volume."""
        return self.update_permissions(
            changes=[PermissionsChange(
                principal=principal,
                add=list(_normalize_privileges(privileges)),
            )]
        )

    def revoke(
        self,
        principal: str,
        privileges: "str | Privilege | list[str | Privilege]",
    ) -> "Volume":
        """Remove one or more privileges for *principal* on this volume."""
        return self.update_permissions(
            changes=[PermissionsChange(
                principal=principal,
                remove=list(_normalize_privileges(privileges)),
            )]
        )

    def update_permissions(
        self,
        changes: "list[PermissionsChange | Mapping[str, Any]]",
    ) -> "Volume":
        """Apply a batch of ``PermissionsChange`` to this volume."""
        normalized: list[PermissionsChange] = []
        for change in changes or ():
            if isinstance(change, PermissionsChange):
                pc = change
            elif isinstance(change, Mapping):
                pc = PermissionsChange(
                    principal=change.get("principal"),
                    add=list(_normalize_privileges(change.get("add"))) or None,
                    remove=list(_normalize_privileges(change.get("remove"))) or None,
                )
            else:
                raise TypeError(
                    f"Volume.update_permissions: each change must be a "
                    f"PermissionsChange or mapping, got {type(change).__name__}: "
                    f"{change!r}."
                )
            if not pc.principal:
                raise ValueError(
                    f"Volume.update_permissions: change is missing 'principal': {pc!r}."
                )
            if not pc.add and not pc.remove:
                continue
            normalized.append(pc)

        if not normalized:
            return self

        self.client.workspace_client().grants.update(
            securable_type=SecurableType.VOLUME.value,
            full_name=self.full_name(),
            changes=normalized,
        )
        return self


# ---------------------------------------------------------------------------
# Helpers — shared with VolumePath via re-export from
# yggdrasil.databricks.fs.volume_path for backwards compatibility.
# ---------------------------------------------------------------------------


def _normalize_privileges(
    privileges: Any,
) -> Iterator[Privilege]:
    """Yield :class:`Privilege` enums for any caller-facing privilege spec."""
    if privileges is None:
        return
    if isinstance(privileges, (str, Privilege)):
        items: Any = (privileges,)
    else:
        items = privileges
    seen: set[Privilege] = set()
    for item in items:
        if item is None:
            continue
        if isinstance(item, Privilege):
            normalized = item
        else:
            token = str(item).strip()
            if not token:
                continue
            key = token.upper().replace("-", "_").replace(" ", "_")
            key = "_".join(p for p in key.split("_") if p)
            try:
                normalized = Privilege(key)
            except ValueError as exc:
                valid = ", ".join(p.value for p in Privilege)
                raise ValueError(
                    f"Unknown Unity Catalog privilege {token!r}. "
                    f"Pass a Privilege enum or one of: {valid}."
                ) from exc
        if normalized in seen:
            continue
        seen.add(normalized)
        yield normalized


def _looks_like_not_found(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("NotFound", "ResourceDoesNotExist", "FileNotFoundError"):
        return True
    if isinstance(exc, FileNotFoundError):
        return True
    return "does not exist" in str(exc).lower()


def _looks_like_already_exists(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("AlreadyExists", "ResourceAlreadyExists", "FileExistsError"):
        return True
    return "already exists" in str(exc).lower()


def _safe_create(create: Any) -> bool:
    """Run *create()*; treat ``AlreadyExists`` as success (idempotent)."""
    try:
        create()
    except Exception as exc:
        if _looks_like_already_exists(exc):
            return False
        raise
    return True


def _managed_volume_type() -> Any:
    """Resolve ``VolumeType.MANAGED`` (falls back to the string for older SDKs)."""
    try:
        from databricks.sdk.service.catalog import VolumeType
        return VolumeType.MANAGED
    except Exception:
        return "MANAGED"
