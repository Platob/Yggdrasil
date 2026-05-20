"""S3 service object.

Thin :class:`AWSService` subclass: owns no behavior beyond what
:class:`AWSService` provides — its single job is to be the
S3-flavored entry point. :class:`S3Path` holds an :class:`S3Service`
in its ``service`` field; the path reaches the boto S3 client via
``path.service.boto_client``.

A "fat" filesystem service (`list_buckets`, `head_object`,
`copy_object`, `sync`, …) is intentionally *not* here — those
operations are addressable on :class:`S3Path` itself (you walk to
the path you care about and call ``stat()`` / ``ls()`` /
``copy_to()`` / etc., which all go through the boto client this
service exposes).

What :class:`S3Service` *does* expose:

- :attr:`boto_client` — the cached boto S3 client (inherited from
  :class:`AWSService`). :class:`S3Path` reaches this via its own
  ``client`` property (= ``self.service.boto_client``).
- :attr:`s3_client` shorthand — same thing, explicit about which
  service the boto client is for.
- :meth:`path` — convenience constructor for :class:`S3Path`
  bound to this service.
- :attr:`ls_cache` — short-lived ``ListObjectsV2`` cache shared
  across every :class:`S3Path` bound to this service. Stat caching
  lives on :class:`yggdrasil.io.fs.RemotePath`, not here.

Construction
------------

    >>> AWSClient.current().s3                    # default singleton
    >>> S3Service(client=AWSClient(config=...))   # explicit client

The ``s3`` property on :class:`AWSClient` lazily caches a single
:class:`S3Service` per client, so reaching for ``client.s3`` in
hot code is free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional

from yggdrasil.aws.client import AWSClient, AWSService
from yggdrasil.dataclasses.expiring import ExpiringDict

if TYPE_CHECKING:
    from botocore.client import BaseClient  # type: ignore[import-untyped]

    from yggdrasil.aws.fs.path import S3Path


__all__ = ["S3Service"]

#: Default TTL for listing cache entries (seconds). Listings are more
#: expensive than stats (paginated ListObjectsV2); callers that need
#: a fresh listing can bypass via :meth:`S3Service.invalidate_ls_cache`.
_LS_CACHE_TTL: float = 15.0

#: Maximum cached listing results per service.
_LS_CACHE_MAX: int = 256


class S3Service(AWSService):
    """Thin S3 service object — owns the boto S3 client.

    Inherits everything from :class:`AWSService`; the only thing
    that's S3-specific here is :attr:`service_name` (= ``"s3"``)
    and the convenience :meth:`path` factory.

    The :attr:`ls_cache` is keyed on ``(bucket, prefix)`` tuples
    with a short TTL, so a DeltaIO replay that hits the same prefix
    5–10 times pays one ListObjectsV2 instead of N.
    """

    # ``_ls_cache`` is a per-instance live ExpiringDict — excluded from
    # pickling so a snapshot doesn't drag a TTL-managed dict across
    # process boundaries. The receiver re-builds on first access.
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = (
        AWSService._TRANSIENT_STATE_ATTRS | frozenset({"_ls_cache"})
    )

    def __init__(self, client: Optional[AWSClient] = None) -> None:
        if getattr(self, "_initialized", False):
            return
        super().__init__(client=client)
        self._ls_cache: Optional[ExpiringDict] = None

    def __setstate__(self, state):
        # On cross-process unpickle the transient _ls_cache slot is
        # absent from the payload — re-create the attribute so the
        # property's lazy build still works. The live-singleton path
        # short-circuits via the inherited _initialized guard.
        if getattr(self, "_initialized", False):
            return
        super().__setstate__(state)
        self._ls_cache = None

    @classmethod
    def service_name(cls) -> str:
        return "s3"

    # ------------------------------------------------------------------
    # ls cache — lazy init, one per service instance
    # ------------------------------------------------------------------

    @property
    def ls_cache(self) -> ExpiringDict:
        """Per-prefix listing cache. Keys are ``"bucket/prefix"`` strings.

        Values are tuples of child key strings — lightweight and
        serializable. The cache stores the *result* of one
        ListObjectsV2 paginator run so repeated ``iterdir()`` /
        ``ls()`` on the same prefix within the TTL window reuses it.
        """
        if self._ls_cache is None:
            self._ls_cache = ExpiringDict(
                default_ttl=_LS_CACHE_TTL,
                max_size=_LS_CACHE_MAX,
            )
        return self._ls_cache

    def invalidate_ls_cache(
        self,
        bucket: str | None = None,
        key: str | None = None,
    ) -> None:
        """Drop cached listings whose prefix could contain ``bucket/key``.

        With no args, drops every listing. Stat caching is mutualized
        on :class:`RemotePath`; call
        :meth:`RemotePath.invalidate_singleton` for that.
        """
        if self._ls_cache is None:
            return

        if bucket is None:
            self._ls_cache.clear()
            return

        full = f"{bucket}/{key}" if key else bucket
        to_drop = [k for k in self._ls_cache if full.startswith(str(k))]
        for k in to_drop:
            self._ls_cache.pop(k, None)

    # ------------------------------------------------------------------
    # Ergonomic alias — `service.s3_client` reads better than
    # `service.boto_client` from outside S3Path.
    # ------------------------------------------------------------------

    @property
    def s3_client(self) -> "BaseClient":
        """Alias for :attr:`boto_client` — explicit about what's being
        returned when reading the call site."""
        return self.boto_client

    # ------------------------------------------------------------------
    # Convenience: build an S3Path bound to this service
    # ------------------------------------------------------------------

    def path(self, obj: Any, *, temporary: bool = False) -> "S3Path":
        """Build an :class:`S3Path` bound to this service.

        Saves callers from importing :class:`S3Path` directly when
        they already have an :class:`S3Service` in hand. Mirrors
        :meth:`DatabricksClient.dbfs_path` from the Databricks side.
        """
        # Local import to dodge the module-level cycle:
        # path.py imports service.py for typing.
        from yggdrasil.aws.fs.path import S3Path

        return S3Path(obj, service=self, temporary=temporary)

    # ------------------------------------------------------------------
    # PyArrow filesystem — centralized credential snapshot
    # ------------------------------------------------------------------

    def arrow_filesystem(
        self,
        *,
        region: Optional[str] = None,
        **overrides: Any,
    ) -> Any:
        """Build a :class:`pyarrow.fs.S3FileSystem` from this service's creds.

        Central credential point for every caller that wants pyarrow's
        native S3 filesystem (parquet streaming, IPC, CSV
        scanners, …). The boto :class:`Session` underneath holds the
        live :class:`RefreshableCredentials` — we snapshot the current
        access key / secret / session token here and hand them to
        pyarrow. The snapshot is valid for the lifetime of the STS
        token; long-running consumers should rebuild the filesystem
        once per refresh window (call this method again after a token
        rotation).

        ``region`` overrides the client's configured region (useful
        when a bucket sits in a different region than the default).
        ``**overrides`` flow straight to :class:`pyarrow.fs.S3FileSystem`
        so callers can tune ``request_timeout`` / ``proxy_options`` /
        ``retry_strategy`` / ``endpoint_override`` without subclassing.
        """
        from pyarrow.fs import S3FileSystem

        # Snapshot the live credentials from the boto session. With a
        # ``RefreshableCredentials``-backed session this returns the
        # currently-vended STS token; botocore would refresh it
        # near-expiry inside ``get_frozen_credentials()``.
        creds = self.client.session.get_credentials()
        if creds is None:
            # No credentials configured — let pyarrow walk its own
            # default chain (env / config / instance metadata).
            return S3FileSystem(region=region or self.client.region, **overrides)
        frozen = creds.get_frozen_credentials()
        return S3FileSystem(
            access_key=frozen.access_key,
            secret_key=frozen.secret_key,
            session_token=frozen.token,
            region=region or self.client.region,
            **overrides,
        )