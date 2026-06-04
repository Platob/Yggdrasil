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
- Lifecycle (``create`` / ``get_or_create`` / ``delete``)
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
import time
from typing import Any, ClassVar, Iterator, Mapping, Optional, TYPE_CHECKING

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
from yggdrasil.databricks.aws import AWSDatabricksVolumeCredentials
from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.path import DatabricksPath
from yggdrasil.dataclasses import Singleton, WaitingConfig
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.enums import Mode, ModeLike, Scheme, IOKind, MediaTypes
from yggdrasil.io import IOStats
from yggdrasil.url import URL

if TYPE_CHECKING:
    from databricks.sdk.service.catalog import VolumeType
    from yggdrasil.aws.client import AWSClient
    from yggdrasil.databricks.catalog.catalog import UCCatalog
    from yggdrasil.databricks.fs.volume_path import VolumePath
    from yggdrasil.databricks.schema.schema import UCSchema
    from yggdrasil.databricks.volume.volumes import Volumes

__all__ = ["Volume"]

logger = logging.getLogger(__name__)


class Volume(DatabricksPath):
    """A single Unity Catalog volume — metadata, credentials, storage path.

    Construct via :meth:`Volumes.volume` /
    :meth:`Volumes.__getitem__` for the cache-aware path, or
    directly when callers already have the coordinates. Either way
    the instance is interned per ``(client, catalog, schema, name)``
    so subsequent calls collapse to the same live cache — same
    convention :class:`Catalog` / :class:`Schema` / :class:`Table`
    use.
    """

    DEFAULT_INFO_TTL: ClassVar[float] = 1800.0  # 30 minutes
    NAMESPACE_PREFIX: ClassVar[str] = "/Volumes/"
    _INSTANCES: ClassVar = Singleton._INSTANCES.__class__(default_ttl=None)
    _SINGLETON_TTL: ClassVar[Any] = 300.0

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
        ttl_arg = (
            float(singleton_ttl)
            if isinstance(singleton_ttl, int) and not isinstance(singleton_ttl, bool)
            else singleton_ttl
        )

        def _build() -> "Volume":
            inst = _allocate()
            try:
                object.__setattr__(inst, "_singleton_key_", key)
            except AttributeError:
                pass
            return inst

        return cls._INSTANCES.get_or_set(key, _build, ttl=ttl_arg)

    def __init__(
        self,
        service: "Volumes | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
        *,
        client: DatabricksClient | None = None,
        infos: VolumeInfo | None = None,
        infos_fetched_at: float | None = None,
        infos_ttl: float | None = None,
        singleton_ttl: "int | None" = ...,
        **kwargs
    ) -> None:
        del singleton_ttl

        if service is None:
            from .volumes import Volumes
            service = Volumes.current() if client is None else Volumes(client=client)

        if getattr(self, "_initialized", False):
            if service is not None:
                self.service = service
            if infos is not None:
                self._store_infos(infos, fetched_at=infos_fetched_at)
            return

        self.catalog_name = str(catalog_name)
        self.schema_name = str(schema_name)
        self.volume_name = str(volume_name)

        super().__init__(service=service, url=URL(
            scheme=Scheme.DATABRICKS_VOLUME.value,
            path=f"{self.NAMESPACE_PREFIX}{self.catalog_name}/{self.schema_name}/{self.volume_name}"
        ), **kwargs)
        self._infos_ttl: float = self.DEFAULT_INFO_TTL if infos_ttl is None else float(infos_ttl)
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

    def _from_url(self, url: URL) -> "DatabricksPath":
        # ``url.parts`` is 0-indexed (leading ``/`` stripped), so this
        # volume's own URL (``/<cat>/<sch>/<vol>``) is three parts.
        # Path-join navigation follows the volume-family depth model —
        # catalog (1) → schema (2) → volume (3) → :class:`VolumePath`
        # (4+).
        # Drop empty components (trailing / duplicate slashes) so the
        # depth count reflects real segments however the URL was built,
        # and a leading ``Volumes`` namespace token (this volume's own URL
        # carries the ``/Volumes/`` prefix) so the depth model lines up
        # whether or not the joined URL kept the namespace.
        parts = [p for p in url.parts if p]
        if parts and parts[0] == "Volumes":
            parts = parts[1:]
        n = len(parts)

        if n <= 1:
            # ``/<catalog>`` — up to the parent catalog.
            return self.catalog
        if n == 2:
            # ``/<catalog>/<schema>`` — up to the parent schema.
            return self.schema
        if n == 3:
            # ``/<catalog>/<schema>/<volume>`` — this volume itself.
            return self
        # Depth ≥ 4 — a file or directory under this volume.
        return self.path("/".join(parts[3:]))

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
    def explore_url(self) -> URL:
        return self.client.base_url.with_path(
            f"/explore/data/volumes/{self.catalog_name}/{self.schema_name}/{self.volume_name}"
        )

    @classmethod
    def from_url(cls, url: "URL | str", **kwargs: Any) -> "Volume":
        """Build a :class:`Volume` from a ``dbfs+volume:///cat/sch/vol`` URL.

        Used by the :class:`DatabricksPath` dispatcher when a caller
        passes a POSIX volume path that resolves exactly to volume
        depth (``DatabricksPath("/Volumes/main/sales/staging")`` →
        ``Volume("main", "sales", "staging")``); deeper paths resolve
        to :class:`VolumePath` instead.
        """

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
                DatabricksClient.from_url(u)
                if u.host else DatabricksClient.current()
            )
        service = kwargs.pop("service", None) or client.volumes

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
        # A successful info fetch proves the volume exists — seed the stat
        # cache (a volume is directory-like) in the same beat so a follow-up
        # ``exists`` / ``stat`` / ``is_dir`` reuses it instead of re-probing.
        # Build the IOStats inline (not via ``_stat_uncached``, which routes
        # back through ``read_info`` → here and would recurse).
        self._persist_stat_cache(
            IOStats(kind=IOKind.DIRECTORY, media_type=MediaTypes.DIRECTORY)
        )
        return info

    def _reset_cache(self) -> None:
        self._infos = None
        self._infos_fetched_at = None
        self._storage_path = None
        self._catalog = None
        self._schema = None
        # The info and the stat snapshot describe the same object — drop
        # them together so a re-probe after a delete / rebind is honest.
        self._stat_cached = None
        self._stat_cached_at = 0.0

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

    def read_info(self, *, refresh: bool = False, default: Any = ...) -> VolumeInfo:
        """Return the SDK's :class:`VolumeInfo` for this volume.

        Refreshes whenever the cached entry is past
        :attr:`DEFAULT_INFO_TTL` (5 minutes by default), or when
        ``refresh=True`` forces it. If the underlying ``volumes.read``
        raises :class:`NotFound`, auto-creates the volume (and any
        missing catalog / schema parents) and retries the read once.
        """
        if not refresh and self._is_fresh():
            return self._infos  # type: ignore[return-value]

        try:
            info = self.client.workspace_client().volumes.read(self.full_name())
        except Exception as e:
            if default is ...:
                raise
            logger.warning(f"Volume {self.full_name(safe=True)} not found: {e}")
            return default

        return self._store_infos(info)

    def exists(self) -> bool:
        """``True`` if this volume is reachable via the Unity Catalog API.

        Always hits the API (``refresh=True``): ``exists`` is a liveness
        probe, and a cached :class:`VolumeInfo` (5-minute TTL) would keep
        reporting ``True`` for minutes after the volume was dropped — e.g.
        right after a ``delete`` / a peer's cascade teardown. On a clean
        not-found the stale cache is dropped so a following :attr:`info`
        doesn't resurrect the deleted volume's metadata.
        """
        try:
            _ = self.read_info(refresh=True)
            return True
        except NotFound:
            self._reset_cache()
            return False
        except DatabricksError as exc:
            if _looks_like_not_found(exc):
                self._reset_cache()
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

        from yggdrasil.path import Path

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

    def credentials_refresher(
        self,
        *,
        secret_cache: bool = False,
    ) -> AWSDatabricksVolumeCredentials:
        """Return the process-wide singleton credentials provider for
        this volume.

        Keyed by ``volume_id`` — every :class:`Volume` / :class:`VolumePath`
        pointing at the same UC volume collapses to one provider.

        ``secret_cache=True`` opts the provider into persisting its vended
        AWS credentials in a per-volume Databricks secret scope (off by
        default); the opt-in is sticky across the shared singleton.
        """
        return AWSDatabricksVolumeCredentials(
            volume_id=self.volume_id,
            client=self.client,
            resource_url=self.full_name(),
            secret_cache=secret_cache,
        )

    def aws(
        self,
        *,
        mode: ModeLike = None,
        region: Optional[str] = None,
        secret_cache: bool = False,
    ) -> "AWSClient":
        """Return an :class:`AWSClient` whose credentials self-refresh
        from :meth:`temporary_credentials`.

        ``secret_cache=True`` backs the vended credentials with a per-volume
        Databricks secret scope (off by default)."""
        return self.credentials_refresher(
            secret_cache=secret_cache,
        ).aws_client(mode=mode, region=region)

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
    def catalog(self) -> "UCCatalog":
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

    @schema.setter
    def schema(self, value: "str | UCSchema") -> None:
        if isinstance(value, str):
            value = self.client.schemas.schema(
                catalog_name=self.catalog_name,
                schema_name=value,
            )
        elif not isinstance(value, UCSchema):
            raise ValueError(f"Expected schema name or UCSchema instance; got {value!r}.")
        if value.catalog_name != self.catalog_name:
            raise ValueError(
                f"Cannot set {self!r}'s schema to {value!r} — "
                f"catalog mismatch (expected {self.catalog_name!r}, got {value.catalog_name!r})."
            )
        if value.schema_name != self.schema_name:
            raise ValueError(
                f"Cannot set {self!r}'s schema to {value!r} — "
                f"schema name mismatch (expected {self.schema_name!r}, got {value.schema_name!r})."
            )
        self._schema = value

    @property
    def parent(self) -> "DatabricksPath | None":
        return self.schema

    @property
    def parents(self) -> "Iterator[DatabricksPath]":
        yield self.schema
        yield self.catalog

    def path(
        self,
        sub: str = "",
        *,
        url: URL | None = None,
        volume: "Volume | None" = None,
        service: Any = None,
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
            service=service or self.service,
            temporary=temporary,
            **kwargs
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def create(
        self,
        *,
        refresh: bool = False,
        comment: str | None = None,
        storage_location: str | None = None,
        volume_type: "VolumeType | str" = None,
        missing_ok: bool = True,
    ) -> "Volume":
        """Create this volume in Unity Catalog.

        Idempotent — a successful read means it already exists (reads never
        auto-create). On a not-found create error the missing parent schema
        (and, through it, the catalog) is created and the create retried once.

        Defaults to a managed volume. Pass ``storage_location`` +
        ``volume_type="EXTERNAL"`` for an external volume.
        """
        if self.read_info(refresh=refresh, default=None) is not None:
            self._persist_stat_cache(self._stat_uncached())
            return self

        uc = self.client.workspace_client().volumes

        try:
            from databricks.sdk.service.catalog import VolumeType

            if volume_type is None:
                volume_type = VolumeType.EXTERNAL if storage_location else VolumeType.MANAGED
            elif not isinstance(volume_type, VolumeType):
                volume_type = VolumeType[str(volume_type)]

            is_external = volume_type == VolumeType.EXTERNAL
        except Exception:
            if volume_type is None:
                volume_type = "EXTERNAL" if storage_location else "MANAGED"
            else:
                volume_type = str(volume_type).upper()

            is_external = volume_type == "EXTERNAL"

        if not is_external:
            storage_location = None
        # An external volume requires an explicit ``storage_location``; when the
        # caller doesn't pin one the Files/UC API rejects the create — no
        # location is fabricated on their behalf.

        kwargs: dict[str, Any] = {
            "catalog_name": self.catalog_name,
            "schema_name": self.schema_name,
            "name": self.volume_name,
            "volume_type": volume_type,
        }
        if comment is not None:
            kwargs["comment"] = comment
        if storage_location is not None:
            kwargs["storage_location"] = storage_location

        try:
            info = uc.create(**kwargs)
            self._store_infos(info)
        except Exception as exc:
            low = str(exc).lower()
            if missing_ok and "already exists" in low:
                self._reset_cache()
            elif "not exist" in low or "not found" in low:
                # Parent schema (and, through it, the catalog) missing —
                # create the parents and retry the volume create once.
                logger.info("Volume %r create failed (%s); ensuring parents", self, exc)
                self.schema.get_or_create()
                self._store_infos(uc.create(**kwargs))
            else:
                raise
        # Keep the path stat cache in lock-step with the now-current info so a
        # follow-up exists() / is_dir() / stat() doesn't observe a stale MISSING.
        self._persist_stat_cache(self._stat_uncached())
        return self

    def get_or_create(
        self,
        *,
        comment: str | None = None,
        storage_location: str | None = None,
        volume_type: Any = None,
    ) -> "Volume":
        """Create this volume (and any missing parents) if it doesn't exist,
        then return ``self``. :meth:`create` is itself idempotent and ensures
        the parents on a not-found, so this is just a named alias."""
        return self.create(
            comment=comment,
            storage_location=storage_location,
            volume_type=volume_type,
            missing_ok=True,
        )

    def delete(
        self,
        predicate: str = None,
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

    def _stat_uncached(self) -> IOStats:
        infos = self.read_info(default=None)

        return IOStats(
            kind=IOKind.MISSING if infos is None else IOKind.DIRECTORY,
            media_type=MediaTypes.DIRECTORY
        )

    def _ls(self, recursive: bool = False, *, singleton_ttl: Any = False) -> Iterator["VolumePath"]:
        pass

    def _read_mv(self, n: int, pos: int) -> memoryview:
        raise NotImplementedError(
            f"{type(self).__name__} is a Unity Catalog volume, not a positional "
            f"byte buffer. Navigate to a file via ``volume.path('<sub/path>')`` "
            f"and read that instead."
        )

    def full_path(self) -> str:
        return f"/Volumes/{self.catalog_name}/{self.schema_name}/{self.volume_name}"

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        self.create()
        return None

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        self.delete(wait=wait)
        return None

    def _remove_dir(self, recursive: bool, missing_ok: bool, wait: WaitingConfig) -> None:
        self.delete(wait=wait)
        return None

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

