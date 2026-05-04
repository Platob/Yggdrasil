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
service exposes). When something genuinely cross-cutting comes up
(e.g. a parallel sync between two prefixes), it can land here.

What :class:`S3Service` *does* expose:

- :meth:`boto_client` — the cached boto S3 client (inherited from
  :class:`AWSService`).
- :attr:`client` shorthand — same thing, named for ergonomics so
  ``service.client.head_object(...)`` reads well from S3Path.
- :meth:`path` — convenience constructor for :class:`S3Path`
  bound to this service.
- :attr:`stat_cache` / :attr:`ls_cache` — short-lived caches for
  ``HeadObject`` and ``ListObjectsV2`` results. Shared across all
  S3Paths bound to this service, so a DeltaIO replay that hits
  the same keys repeatedly pays one round-trip instead of N.

Construction
------------

    >>> AWSClient.current().s3                    # default singleton
    >>> S3Service(client=AWSClient(config=...))   # explicit client

The ``s3`` property on :class:`AWSClient` lazily caches a single
:class:`S3Service` per client, so reaching for ``client.s3`` in
hot code is free.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.aws.client import AWSClient, AWSService
from yggdrasil.dataclasses.expiring import ExpiringDict

if TYPE_CHECKING:
    from botocore.client import BaseClient  # type: ignore[import-untyped]

    from yggdrasil.aws.fs.path import S3Path


__all__ = ["S3Service"]

#: Default TTL for stat cache entries (seconds). Short enough to not
#: cause stale reads in typical workflows, long enough to collapse the
#: 10+ HeadObject calls a single DeltaIO replay makes on the same keys.
_STAT_CACHE_TTL: float = 30.0

#: Default TTL for listing cache entries (seconds). Listings are more
#: expensive (paginated ListObjectsV2) so the TTL is the same; callers
#: that need a fresh listing can bypass via ``invalidate_cache``.
_LS_CACHE_TTL: float = 15.0

#: Maximum number of cached stat entries per service. Keeps memory
#: bounded even when walking very large prefixes.
_STAT_CACHE_MAX: int = 4096

#: Maximum cached listing results per service.
_LS_CACHE_MAX: int = 256


@dataclasses.dataclass
class S3Service(AWSService):
    """Thin S3 service object — owns the boto S3 client.

    Inherits everything from :class:`AWSService`; the only thing
    that's S3-specific here is :attr:`service_name` (= ``"s3"``)
    and the convenience :meth:`path` factory.

    The stat and listing caches are keyed on ``(bucket, key)`` tuples.
    Both use short TTLs so callers see fresh data within seconds, but
    hot loops (like DeltaIO replay) that hit the same keys 5–10 times
    within one operation pay only one S3 round-trip per key.
    """

    # ExpiringDict fields — not part of the dataclass __init__; lazily
    # created on first access so default S3Service() is cheap.
    _stat_cache: ExpiringDict | None = dataclasses.field(
        default=None, init=False, repr=False,
    )
    _ls_cache: ExpiringDict | None = dataclasses.field(
        default=None, init=False, repr=False,
    )

    @classmethod
    def service_name(cls) -> str:
        return "s3"

    # ------------------------------------------------------------------
    # Caches — lazy init, one per service instance
    # ------------------------------------------------------------------

    @property
    def stat_cache(self) -> ExpiringDict:
        """Per-key ``PathStats`` cache. Keys are ``"bucket/key"`` strings."""
        if self._stat_cache is None:
            self._stat_cache = ExpiringDict(
                default_ttl=_STAT_CACHE_TTL,
                max_size=_STAT_CACHE_MAX,
            )
        return self._stat_cache

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

    def invalidate_cache(
        self,
        bucket: str | None = None,
        key: str | None = None,
    ) -> None:
        """Drop cached entries. Call after writes / deletes.

        With no args, drops everything. With ``bucket`` + ``key``,
        drops the exact stat entry and any listing whose prefix is
        a parent of the key.
        """
        if bucket is None:
            if self._stat_cache is not None:
                self._stat_cache.clear()
            if self._ls_cache is not None:
                self._ls_cache.clear()
            return

        full = f"{bucket}/{key}" if key else bucket
        if self._stat_cache is not None:
            self._stat_cache.pop(full, None)

        # Invalidate any listing prefix that could contain this key.
        if self._ls_cache is not None:
            to_drop = [
                k for k in self._ls_cache
                if full.startswith(str(k))
            ]
            for k in to_drop:
                self._ls_cache.pop(k, None)

    # ------------------------------------------------------------------
    # Ergonomic alias — `service.client` reads better than
    # `service.boto_client` from inside S3Path.
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