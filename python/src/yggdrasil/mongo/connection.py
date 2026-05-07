"""MongoDB connection wrapper.

A :class:`MongoConnection` owns one lazily-opened :class:`pymongo.MongoClient`
plus the resolved default database name. The connection is a
:class:`Disposable` so context-manager use cascades through the
:class:`MongoEngine` that owns it.

Construction
------------
Pass a connection URI directly, an existing :class:`pymongo.MongoClient`
to wrap, or rely on the ``MONGO_URI`` / ``MONGODB_URI`` environment
variable. The default database can be overridden per-call but must
exist (or be reachable) on the cluster.

URI handling
------------
:class:`yggdrasil.io.URL` parses the URI for log-safe rendering — we
log the URL with credentials stripped via :meth:`URL.with_user_password`
so leaks don't end up in pipeline traces.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse

from yggdrasil.disposable import Disposable
from yggdrasil.io import URL

from yggdrasil.lazy_imports import has_pymongoarrow, pymongo_module

if TYPE_CHECKING:
    from pymongo import MongoClient
    from pymongo.database import Database

logger = logging.getLogger(__name__)

__all__ = [
    "MongoConnection",
    "normalize_mongo_uri",
    "DEFAULT_URI_ENVS",
]


#: Environment variables (in order) consulted by :meth:`MongoConnection.from_`
#: when no explicit URI is passed. ``MONGO_URI`` first matches the rest of
#: yggdrasil's ``<NAME>_URI`` pattern (``POSTGRES_URI``, ``KAFKA_URI``);
#: ``MONGODB_URI`` is the upstream MongoDB convention.
DEFAULT_URI_ENVS: tuple[str, ...] = ("MONGO_URI", "MONGODB_URI")


def normalize_mongo_uri(uri: str) -> str:
    """Strip extraneous whitespace and validate the scheme.

    pymongo accepts both ``mongodb://`` and ``mongodb+srv://``;
    anything else is rejected. A bare ``host[:port][/db]`` (no
    ``://`` separator) is treated as a hostname and prefixed with
    ``mongodb://`` — saves callers a few characters of boilerplate
    on the common local-dev case.
    """
    if not uri:
        raise ValueError("Mongo URI cannot be empty")
    cleaned = uri.strip()
    if "://" not in cleaned:
        return f"mongodb://{cleaned}"
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"mongodb", "mongodb+srv"}:
        raise ValueError(
            f"Unsupported Mongo URI scheme {parsed.scheme!r}; "
            "expected mongodb:// or mongodb+srv://."
        )
    return cleaned


def _safe_uri(uri: str) -> str:
    """Render a URI without credentials for log lines / repr."""
    parsed = urlparse(uri)
    if parsed.password:
        netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
        return urlunparse(parsed._replace(netloc=netloc))
    return uri


class MongoConnection(Disposable):
    """Lazily-opened :class:`pymongo.MongoClient` bound to a single URI.

    The client is opened once on first access and reused for the
    lifetime of this :class:`MongoConnection`. pymongo internally
    pools connections per ``(host, port)`` pair and is thread-safe,
    so a single :class:`MongoConnection` is enough even for high-
    concurrency workloads — we don't shard handles by use-case the
    way the Postgres backend does (psycopg + ADBC), because pymongo's
    Arrow path is layered *on top* of the same client.

    Default database
    ----------------
    The URI may carry a default database (``mongodb://host/db``); the
    constructor's ``default_database=`` overrides it. When neither is
    set, calls that need a database (``database()`` without args)
    raise — explicit beats implicit.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        *,
        default_database: Optional[str] = None,
        client_kwargs: Optional[Mapping[str, Any]] = None,
        client: Optional["MongoClient"] = None,
    ):
        super().__init__()
        if client is not None:
            self.uri = self._uri_from_client(client)
            self._client: Any = client
            self._owns_client = False
        else:
            if uri is None:
                uri = self._uri_from_environment()
            if not uri:
                raise ValueError(
                    "MongoConnection requires a URI; pass uri=, an existing "
                    f"client=, or set one of: {', '.join(DEFAULT_URI_ENVS)}."
                )
            self.uri = normalize_mongo_uri(uri)
            self._client = None
            self._owns_client = True

        self.client_kwargs = dict(client_kwargs or {})
        self._lock = threading.Lock()
        self.default_database = (
            default_database
            or self._database_from_uri(self.uri)
            or None
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_(
        cls,
        value: "MongoConnection | str | Mapping[str, Any] | Any | None" = None,
        **overrides: Any,
    ) -> "MongoConnection":
        """Coerce a URI / mapping / existing connection into one of these.

        Recognised inputs:

        - :class:`MongoConnection` — passed through (overrides ignored).
        - ``str`` — treated as a URI.
        - :class:`pymongo.MongoClient` — wrapped without taking ownership.
        - :class:`Mapping` — forwarded to the constructor.
        - ``None`` — built from the environment.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            return cls(**overrides)
        if isinstance(value, str):
            return cls(uri=value, **overrides)
        if isinstance(value, Mapping):
            kwargs = dict(value)
            kwargs.update(overrides)
            return cls(**kwargs)
        # Duck-type: pymongo.MongoClient (avoid the import on the hot path).
        if type(value).__name__ == "MongoClient":
            return cls(client=value, **overrides)
        raise TypeError(
            f"Cannot build MongoConnection from {type(value).__name__}: {value!r}"
        )

    @staticmethod
    def _uri_from_environment() -> Optional[str]:
        for env in DEFAULT_URI_ENVS:
            value = os.environ.get(env)
            if value:
                return value
        return None

    @staticmethod
    def _uri_from_client(client: Any) -> str:
        # pymongo's MongoClient exposes the original URI on
        # ``client.HOST`` / ``client._MongoClient__init_kwargs`` but
        # those are private. ``client.address`` is host:port — close
        # enough for repr / equality. Fall back to a placeholder when
        # the client isn't yet connected.
        addr = getattr(client, "address", None)
        if addr:
            host, port = addr
            return f"mongodb://{host}:{port}"
        return "mongodb://wrapped-client"

    @staticmethod
    def _database_from_uri(uri: str) -> Optional[str]:
        try:
            url = URL.from_str(uri, default_scheme="mongodb")
        except Exception:
            return None
        path = url.path or ""
        path = path.lstrip("/").split("/", 1)[0]
        return path or None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"MongoConnection({_safe_uri(self.uri)!r}, db={self.default_database!r})"

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    @property
    def client(self) -> "MongoClient":
        """Lazily-opened pymongo client.

        First access establishes the TCP / TLS handshake. Reused for
        the lifetime of the connection; pymongo handles its own pool
        internally.
        """
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            pymongo = pymongo_module()
            kwargs = dict(self.client_kwargs)
            logger.debug("Opening MongoClient to %r", self)
            self._client = pymongo.MongoClient(self.uri, **kwargs)
        return self._client

    def database(self, name: Optional[str] = None) -> "Database":
        """Resolve a pymongo :class:`Database`.

        With no name we use the URI's default database (if any) or
        :attr:`default_database` (if set); otherwise we raise — there
        is no global default in pymongo and silently picking
        ``"test"`` is exactly the kind of footgun yggdrasil avoids.
        """
        target = name or self.default_database
        if not target:
            raise ValueError(
                "MongoConnection.database() requires an explicit name when "
                "the URI doesn't carry a default database."
            )
        return self.client[target]

    @property
    def has_pymongoarrow(self) -> bool:
        """Whether the Arrow-native path is available (probe-only)."""
        return has_pymongoarrow()

    @property
    def address(self) -> Optional[tuple[str, int]]:
        """Currently-connected primary host (probe-only — may be None when not yet connected)."""
        return getattr(self._client, "address", None)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the wrapped pymongo client (idempotent).

        We only close it when we own the handle; an externally-passed
        client (``MongoConnection(client=...)``) is the caller's to
        close.
        """
        client = self._client
        self._client = None
        if client is None or not self._owns_client:
            return
        try:
            client.close()
        except Exception:
            logger.exception("MongoClient close failed; continuing.")

    def _release(self, committed: bool = False) -> None:
        """Disposable hook — close the client on context-manager exit."""
        self.close()

    # MongoDB has no transactional close hook in the open/close sense;
    # commit / rollback are session-scoped (``with client.start_session()``)
    # and managed by callers that need them. We deliberately don't expose
    # commit/rollback on the connection itself.
