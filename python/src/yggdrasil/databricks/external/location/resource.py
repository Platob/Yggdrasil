"""Per-external-location resource: metadata + a credential-backed storage path.

An :class:`ExternalLocation` wraps one Unity Catalog external location — a named
binding of a cloud storage URL to a storage credential — and **behaves like the
storage path itself**. It holds a single inner ``_external_path`` (an
:class:`~yggdrasil.aws.fs.path.S3Path` built from the location's URL + a
credentials-vended AWS client, so the path lives on its own) and mirrors every
filesystem operation straight to it; ``parent`` / children navigation returns
that inner path, leaving the UC wrapper behind.

    el = client.external_locations["raw_zone"]
    el.url                 # 's3://my-bucket/raw/'
    el.credential_name     # 'prod-storage-cred'
    el.ls() / el.read_bytes() / el / "sub/file.parquet"   # → inner S3Path I/O
    el.explore_url         # clickable Catalog Explorer link

Only AWS S3 (``s3://`` / ``s3a://`` / ``s3n://``) is supported for now; any other
scheme raises.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional

from databricks.sdk.service.catalog import ExternalLocationInfo

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.databricks.resource import DatabricksResource
from yggdrasil.path import ProxyPathMixin
from yggdrasil.url import URL

if TYPE_CHECKING:
    from yggdrasil.databricks.credentials import Credential
    from yggdrasil.path import Path

__all__ = ["ExternalLocation"]

#: External locations are near-static config — cache the handle (+ fetched
#: info) for 30 min before re-resolving.
_RESOURCE_TTL: float = 30 * 60.0

_S3_SCHEMES = ("s3", "s3a", "s3n")


class ExternalLocation(DatabricksResource, ProxyPathMixin, Singleton):
    """A single Unity Catalog external location, usable as its storage path.

    Cached as a singleton per ``(service, name)`` for 30 min
    (``_SINGLETON_TTL``). Via :class:`~yggdrasil.path.ProxyPathMixin` the
    whole filesystem surface is mirrored to the inner credential-backed
    :attr:`path` (its :meth:`_internal_path` hook); only AWS S3 is handled
    for now. UC metadata / lifecycle (``url`` / ``credential_name`` /
    ``update`` / ``delete`` / ``explore_url`` / …) stays on the wrapper and
    shadows the delegated surface.
    """

    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(default_ttl=_RESOURCE_TTL, max_size=4096)
    _SINGLETON_TTL: ClassVar[Any] = _RESOURCE_TTL

    @classmethod
    def _singleton_key(cls, name: Any = None, *, service: Any = None, **kwargs: Any) -> Any:
        return (cls, service, name)

    @classmethod
    def from_url(
        cls,
        url: Any,
        *,
        service: "Optional[Any]" = None,
        client: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Build a location from a ``dbfs+extloc:///<name>[/<sub>]`` URL.

        The first path segment is the external-location *name*; any deeper
        path is joined onto the resolved storage path, so a file address
        (``/ExternalLocations/raw_zone/sub/f.parquet``) returns the inner
        storage :class:`~yggdrasil.path.Path` for that file while a bare
        name returns the :class:`ExternalLocation` itself. ``client=`` /
        ``service=`` bind it to a workspace the same way the rest of the
        :class:`~yggdrasil.databricks.path.DatabricksPath` family does.
        """
        segments = [s for s in (URL.from_(url).path or "").split("/") if s]
        if not segments:
            raise ValueError(f"external-location URL carries no name: {url!r}")
        if service is None:
            from yggdrasil.databricks.external.location.service import ExternalLocations

            service = (
                ExternalLocations(client=client)
                if client is not None
                else ExternalLocations.current()
            )
        location = cls(segments[0], service=service)
        if len(segments) > 1:
            return location.joinpath(*segments[1:])
        return location

    def __init__(
        self,
        name: str,
        *,
        service: "Optional[Any]" = None,
        info: Optional[ExternalLocationInfo] = None,
        singleton_ttl: Any = ...,
    ) -> None:
        del singleton_ttl  # consumed by Singleton.__new__
        if getattr(self, "_initialized", False):
            if info is not None:
                self._info = info  # refresh cache on an eager fetch
            return
        if service is None:
            from yggdrasil.databricks.external.location.service import ExternalLocations

            service = ExternalLocations.current()
        super().__init__(service=service)
        self.name = name
        self._info = info
        self._external_path: "Optional[Path]" = None
        self._initialized = True

    def __getstate__(self) -> dict:
        return {"service": self.service, "name": self.name, "info": self._info}

    def __setstate__(self, state: dict) -> None:
        self.service = state["service"]
        self.name = state["name"]
        self._info = state.get("info")
        self._external_path = None
        self._initialized = True

    # -- metadata (lazy fetch + cache) ---------------------------------
    @property
    def info(self) -> ExternalLocationInfo:
        if self._info is None:
            self._info = self.service.get_info(self.name)
        return self._info

    def refresh(self) -> "ExternalLocation":
        """Drop the cached info + inner path and re-fetch."""
        self._info = self.service.get_info(self.name)
        self._external_path = None
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

    # -- the inner credential-backed storage path ----------------------
    @property
    def credential(self) -> "Credential":
        """The storage :class:`Credential` backing this location."""
        return self.client.credentials.credential(self.credential_name)

    def _internal_path(self) -> "Path":
        """:class:`~yggdrasil.path.ProxyPathMixin` hook — the inner
        :class:`~yggdrasil.path.Path` every filesystem op delegates to.

        Built once with credentials vended by this location's storage
        credential (so the path lives on its own), then cached. Only AWS
        S3 for now; other schemes raise."""
        if self._external_path is None:
            self._external_path = self._build_path()
        return self._external_path

    #: The inner storage path (alias for :meth:`_internal_path`).
    path = property(_internal_path)

    def _build_path(self) -> "Path":
        url = self.url
        if not url:
            raise ValueError(f"external location {self.name!r} has no url yet")
        scheme = (URL.from_(url).scheme or "").lower()
        if scheme not in _S3_SCHEMES:
            raise NotImplementedError(
                f"external location {self.name!r}: only AWS S3 is supported for now "
                f"(got {scheme!r} in {url!r})"
            )
        from yggdrasil.enums.aws import AWSRegion

        region = AWSRegion.from_text(url)
        # The path carries its own credentials — an AWS client vended by this
        # location's storage credential — so it can read/write standalone.
        aws = self.credential.aws_client(region=str(region) if region else None)
        return aws.s3.path(url)

    # The whole filesystem surface (ls / read_* / write_* / stat / exists /
    # parent / iterdir / size / unlink / ``/`` / ``with`` / …) is mirrored to
    # :meth:`_internal_path` by :class:`~yggdrasil.path.ProxyPathMixin`.

    # -- lifecycle (UC ops — stay on the wrapper) ----------------------
    def update(self, **changes: Any) -> "ExternalLocation":
        updated = self.service.update(self.name, **changes)
        self._info = updated._info
        self._external_path = None  # url/credential may have changed
        return self

    def delete(self, *, force: bool = False) -> None:
        self.service.delete(self.name, force=force)

    # -- explore --------------------------------------------------------
    @property
    def explore_url(self) -> URL:
        """Catalog Explorer deep-link to this external location."""
        return self.client.base_url.with_path(f"/explore/locations/{self.name}")
