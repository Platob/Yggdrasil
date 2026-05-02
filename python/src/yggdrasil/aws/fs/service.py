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

if TYPE_CHECKING:
    from botocore.client import BaseClient  # type: ignore[import-untyped]

    from yggdrasil.aws.fs.path import S3Path


__all__ = ["S3Service"]


@dataclasses.dataclass
class S3Service(AWSService):
    """Thin S3 service object — owns the boto S3 client.

    Inherits everything from :class:`AWSService`; the only thing
    that's S3-specific here is :attr:`service_name` (= ``"s3"``)
    and the convenience :meth:`path` factory.
    """

    @classmethod
    def service_name(cls) -> str:
        return "s3"

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