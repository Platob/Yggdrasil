"""Per-external-location resource: metadata, storage path, lifecycle.

An :class:`ExternalLocation` wraps one Unity Catalog external location — a
named binding of a cloud storage URL (``s3://`` / ``abfss://`` / ``gs://``) to a
storage credential. Collection operations (list / create) live on
:class:`~yggdrasil.databricks.external.location.locations.ExternalLocations`.

    el = client.external_locations["raw_zone"]
    el.url                 # 's3://my-bucket/raw/'
    el.credential_name     # 'prod-storage-cred'
    el.path                # S3Path('s3://my-bucket/raw/') — browse the storage
    el.explore_url         # clickable Catalog Explorer link
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional

from databricks.sdk.service.catalog import ExternalLocationInfo

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.databricks.resource import DatabricksResource
from yggdrasil.url import URL

if TYPE_CHECKING:
    from yggdrasil.databricks.external.location.locations import ExternalLocations
    from yggdrasil.path import Path

__all__ = ["ExternalLocation"]

#: External locations are near-static config — cache the handle (+ fetched
#: info) for 30 min before re-resolving.
_RESOURCE_TTL: float = 30 * 60.0


class ExternalLocation(DatabricksResource, Singleton):
    """A single Unity Catalog external location.

    Cached as a singleton per ``(service, name)`` for 30 min
    (``_SINGLETON_TTL``) — repeated ``client.external_locations[name]`` share
    one handle (and its fetched info) without re-resolving.
    """

    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(default_ttl=_RESOURCE_TTL, max_size=4096)
    _SINGLETON_TTL: ClassVar[Any] = _RESOURCE_TTL

    @classmethod
    def _singleton_key(cls, name: Any = None, *, service: Any = None, **kwargs: Any) -> Any:
        return (cls, service, name)

    def __init__(
        self,
        name: str,
        *,
        service: "Optional[ExternalLocations]" = None,
        info: Optional[ExternalLocationInfo] = None,
        singleton_ttl: Any = ...,
    ) -> None:
        del singleton_ttl  # consumed by Singleton.__new__
        if getattr(self, "_initialized", False):
            if info is not None:
                self._info = info  # refresh cache on an eager fetch
            return
        if service is None:
            from yggdrasil.databricks.external.location.locations import ExternalLocations

            service = ExternalLocations.current()
        super().__init__(service=service)
        self.name = name
        self._info = info
        self._initialized = True

    def __getstate__(self) -> dict:
        return {"service": self.service, "name": self.name, "info": self._info}

    def __setstate__(self, state: dict) -> None:
        self.service = state["service"]
        self.name = state["name"]
        self._info = state.get("info")
        self._initialized = True

    # -- metadata (lazy fetch + cache) ---------------------------------
    @property
    def info(self) -> ExternalLocationInfo:
        if self._info is None:
            self._info = self.service.get_info(self.name)
        return self._info

    def refresh(self) -> "ExternalLocation":
        """Drop the cached :class:`ExternalLocationInfo` and re-fetch."""
        self._info = self.service.get_info(self.name)
        return self

    @property
    def url(self) -> Optional[str]:
        return self.info.url

    @property
    def credential_name(self) -> Optional[str]:
        return self.info.credential_name

    @property
    def read_only(self) -> bool:
        return bool(self.info.read_only)

    @property
    def comment(self) -> Optional[str]:
        return self.info.comment

    @property
    def owner(self) -> Optional[str]:
        return self.info.owner

    @property
    def metastore_id(self) -> Optional[str]:
        return self.info.metastore_id

    @property
    def credential_id(self) -> Optional[str]:
        return self.info.credential_id

    @property
    def isolation_mode(self) -> Any:
        return self.info.isolation_mode

    @property
    def browse_only(self) -> bool:
        return bool(self.info.browse_only)

    # -- storage path ---------------------------------------------------
    @property
    def path(self) -> "Path":
        """A :class:`~yggdrasil.path.Path` over this location's storage URL —
        e.g. an :class:`S3Path` for an ``s3://`` location, so you can ``ls`` /
        read the backing storage directly."""
        from yggdrasil.path import Path

        if not self.url:
            raise ValueError(f"external location {self.name!r} has no url yet")
        return Path.from_(self.url)

    # -- lifecycle (delegate to the service) ---------------------------
    def update(self, **changes: Any) -> "ExternalLocation":
        updated = self.service.update(self.name, **changes)
        self._info = updated._info
        return self

    def delete(self, *, force: bool = False) -> None:
        self.service.delete(self.name, force=force)

    # -- explore --------------------------------------------------------
    @property
    def explore_url(self) -> URL:
        """Catalog Explorer deep-link to this external location."""
        return self.client.base_url.with_path(f"/explore/location/{self.name}")
