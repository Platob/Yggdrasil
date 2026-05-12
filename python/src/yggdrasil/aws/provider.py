"""Abstract refreshable AWS credentials provider.

A provider vends fresh :class:`AwsCredentials` on demand and is
directly callable, so it slots straight into
:attr:`AWSConfig.refresher` (which botocore re-invokes ~5 min before
each STS token expires).

Identity & singleton caching
----------------------------

Providers are cached process-wide per ``(cls, key)`` — two callers
asking for the same key collapse to one provider instance, which in
turn caches one :class:`AWSClient` per region. The credential vend,
the boto session, the connection pool, and the
:class:`RefreshableCredentials` cycle are all shared across every
caller on the same scope.

Subclasses
----------

Subclasses implement :meth:`get_credentials` and pass a string
``key`` to ``super().__init__`` that uniquely identifies the
credential scope (volume id + operation, table id + operation, STS
role ARN, …). Everything else — the singleton dance, the per-region
:class:`AWSClient` cache, pickle hooks — is inherited.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Tuple

from .config import AwsCredentials

if TYPE_CHECKING:
    from .client import AWSClient


__all__ = ["AwsCredentialsProvider"]


class AwsCredentialsProvider(ABC):
    """Abstract refreshable AWS credentials provider.

    Construct with a string ``key`` that uniquely identifies the
    credential scope. Two providers built with the same ``(cls, key)``
    collapse to the same instance.
    """

    # Per-(cls, key) singleton cache. Subclasses inherit the slot —
    # the ``(cls, key)`` tuple disambiguates AWSDatabricksVolumeCredentials
    # from AWSDatabricksTableCredentials against the same string key.
    _INSTANCES: ClassVar[dict[Tuple[type, str], "AwsCredentialsProvider"]] = {}
    _INSTANCES_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, key: str, *args, **kwargs) -> "AwsCredentialsProvider":
        cache_key = (cls, str(key))
        with cls._INSTANCES_LOCK:
            existing = cls._INSTANCES.get(cache_key)
            if existing is not None:
                return existing
            instance = super().__new__(cls)
            cls._INSTANCES[cache_key] = instance
            return instance

    def __init__(self, key: str) -> None:
        # Singleton-cached instances are re-entered on every constructor
        # call (Python always invokes __init__ after __new__); skip the
        # second pass so the live per-region AWSClient cache survives.
        if getattr(self, "_initialized", False):
            return
        self.key: str = str(key)
        # Cache key is provider-defined — region for the plain base,
        # ``(mode, region)`` for the Databricks subclasses that vend
        # different creds per read/write scope.
        self._client_cache: "dict[Any, AWSClient]" = {}
        self._client_cache_lock: threading.Lock = threading.Lock()
        self._initialized = True

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def get_credentials(self) -> AwsCredentials:
        """Return a fresh :class:`AwsCredentials`. Called by botocore
        ~5 min before token expiry."""

    def __call__(self) -> AwsCredentials:
        """Delegate to :meth:`get_credentials` so the provider doubles
        as an :attr:`AWSConfig.refresher`."""
        return self.get_credentials()

    # ------------------------------------------------------------------
    # AWSClient binding — one client per (provider, region)
    # ------------------------------------------------------------------

    def aws_client(self, *, region: Optional[str] = None) -> "AWSClient":
        """Return the cached :class:`AWSClient` for this provider / region.

        First call seeds a botocore :class:`RefreshableCredentials`
        backed session by invoking ``self()`` once; subsequent calls
        with the same *region* return the same live client (and
        therefore share the connection pool, boto-client cache, and
        in-flight refresh state). Different *region* values mint
        different clients — boto region is a per-client concern.
        """
        with self._client_cache_lock:
            existing = self._client_cache.get(region)
            if existing is not None:
                return existing
            from .config import AWSConfig
            client = AWSConfig.from_refresher(self, region=region).to_client()
            self._client_cache[region] = client
            return client

    # ------------------------------------------------------------------
    # Pickling — collapse to the live singleton in the same process,
    # rebuild transients cross-process.
    # ------------------------------------------------------------------

    def __getnewargs__(self) -> Tuple[str]:
        return (self.key,)

    def __getstate__(self) -> dict:
        # The per-region AWSClient cache is transient — receivers in a
        # different process rebuild lazily on first ``aws_client`` call.
        return {"key": self.key}

    def __setstate__(self, state: dict) -> None:
        # ``__new__`` may have returned a live singleton — leave its
        # cache untouched.
        if getattr(self, "_initialized", False):
            return
        self.key = state["key"]
        self._client_cache = {}
        self._client_cache_lock = threading.Lock()
        self._initialized = True

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        return hash((type(self), self.key))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AwsCredentialsProvider):
            return NotImplemented
        return type(self) is type(other) and self.key == other.key

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key={self.key!r})"
