"""Postgres connection wrapper with paired psycopg + ADBC handles.

A :class:`PostgresConnection` carries the connection URI plus two
lazy handles:

* ``psycopg_conn`` — a psycopg 3 connection used for DDL, lifecycle
  management, and the metadata queries (``information_schema``,
  ``pg_catalog``) that the resource hierarchy relies on.
* ``adbc_conn`` — the ADBC DBAPI connection used for Arrow-native
  reads (``cursor.fetch_arrow_table``) and bulk writes
  (``cursor.adbc_ingest``).

Both are opened on first use and closed together via :meth:`close`.
A :class:`PostgresConnection` is a :class:`Disposable`, so context-
manager use is supported and cleanup cascades through the
:class:`PostgresEngine` that owns it.

URI handling
------------
ADBC and psycopg both accept the standard ``postgresql://`` URI
form — same string for both. We normalise the prefix once in
:meth:`from_` so callers can pass either ``postgresql://`` or
``postgres://``.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse

from yggdrasil.disposable import Disposable

from yggdrasil.lazy_imports import adbc_dbapi_module, has_adbc, psycopg_module

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "PostgresConnection",
    "normalize_postgres_uri",
]


_DEFAULT_URI_ENV = "POSTGRES_URI"


def normalize_postgres_uri(uri: str) -> str:
    """Coerce ``postgres://...`` to ``postgresql://...``.

    Both schemes are interchangeably accepted by psycopg; ADBC is
    stricter and requires ``postgresql://``. Normalising once at
    construction lets every downstream user assume the canonical
    form.
    """
    if not uri:
        raise ValueError("Postgres URI cannot be empty")
    parsed = urlparse(uri)
    if parsed.scheme == "postgres":
        return urlunparse(parsed._replace(scheme="postgresql"))
    return uri


class PostgresConnection(Disposable):
    """Paired psycopg + ADBC connection bound to a single Postgres URI.

    The two handles are opened lazily and only when actually needed:

    * Pure-DDL workflows (``CREATE TABLE``, ``DROP TABLE``) only ever
      open the psycopg side.
    * Arrow-native reads / writes go through ADBC, which is only
      imported when an Arrow-shaped call fires.

    Locking is fine-grained: each driver has its own lock, so a
    metadata query on the psycopg side doesn't block a parallel
    ``fetch_arrow_table`` on the ADBC side. Postgres itself permits
    concurrent statements on different connections — there's no
    cross-driver coordination needed.

    Construction
    ------------
    Pass a URI directly, a mapping of ``connect()``-style kwargs
    (passed verbatim to ``psycopg.connect``), or rely on the
    ``POSTGRES_URI`` environment variable.
    """

    uri: str

    def __init__(
        self,
        uri: Optional[str] = None,
        *,
        psycopg_kwargs: Optional[Mapping[str, Any]] = None,
        adbc_kwargs: Optional[Mapping[str, Any]] = None,
        autocommit: bool = True,
    ):
        super().__init__()
        if uri is None:
            uri = os.environ.get(_DEFAULT_URI_ENV)
        if not uri:
            raise ValueError(
                "PostgresConnection requires a URI; pass uri= or set "
                f"the {_DEFAULT_URI_ENV} environment variable."
            )
        self.uri = normalize_postgres_uri(uri)
        self.psycopg_kwargs = dict(psycopg_kwargs or {})
        self.adbc_kwargs = dict(adbc_kwargs or {})
        self.autocommit = autocommit

        self._psycopg_conn: Any = None
        self._adbc_conn: Any = None
        self._psycopg_lock = threading.Lock()
        self._adbc_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_(
        cls,
        value: "PostgresConnection | str | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "PostgresConnection":
        """Coerce a URI / mapping / existing connection into one of these.

        - ``None`` → built from the ``POSTGRES_URI`` env var.
        - ``str`` → treated as a URI.
        - existing :class:`PostgresConnection` → passed through.
        - ``Mapping`` → forwarded to the constructor.
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
        raise TypeError(
            f"Cannot build PostgresConnection from {type(value).__name__}: {value!r}"
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        # Strip credentials from the URI for log-safe rendering.
        parsed = urlparse(self.uri)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
            safe = urlunparse(parsed._replace(netloc=netloc))
        else:
            safe = self.uri
        return f"PostgresConnection({safe!r})"

    # ------------------------------------------------------------------
    # psycopg
    # ------------------------------------------------------------------

    @property
    def psycopg_conn(self) -> Any:
        """Lazily-opened psycopg 3 connection.

        Reused across DDL / metadata calls — Postgres connections are
        cheap to keep alive and expensive to re-establish (TLS
        handshake, ``pg_authenticate`` round-trips). The first miss
        opens the connection under a lock; subsequent calls hit the
        cached handle directly.
        """
        if self._psycopg_conn is not None:
            return self._psycopg_conn
        with self._psycopg_lock:
            if self._psycopg_conn is not None:
                return self._psycopg_conn
            psycopg = psycopg_module()
            kwargs = dict(self.psycopg_kwargs)
            kwargs.setdefault("autocommit", self.autocommit)
            logger.debug("Opening psycopg connection to %r", self)
            self._psycopg_conn = psycopg.connect(self.uri, **kwargs)
        return self._psycopg_conn

    def psycopg_cursor(self) -> Any:
        """Return a fresh psycopg cursor.

        The cursor is the caller's responsibility — close via context
        manager or explicit ``cursor.close()`` to release the server-
        side resource. The underlying connection is shared.
        """
        return self.psycopg_conn.cursor()

    # ------------------------------------------------------------------
    # ADBC
    # ------------------------------------------------------------------

    @property
    def adbc_conn(self) -> Any:
        """Lazily-opened ADBC DBAPI connection.

        Imports :mod:`adbc_driver_postgresql` on first use; raises a
        clear ImportError when the driver isn't installed. Use
        :meth:`has_adbc` upstream to fall back gracefully when the
        Arrow-fast path isn't available.
        """
        if self._adbc_conn is not None:
            return self._adbc_conn
        with self._adbc_lock:
            if self._adbc_conn is not None:
                return self._adbc_conn
            adbc = adbc_dbapi_module()
            logger.debug("Opening ADBC connection to %r", self)
            self._adbc_conn = adbc.connect(self.uri, **self.adbc_kwargs)
            try:
                self._adbc_conn.autocommit = self.autocommit
            except Exception:
                # Older adbc-driver-postgresql versions don't expose
                # an autocommit setter; the default (commit on close)
                # is fine for our DDL-via-psycopg / DML-via-ADBC
                # split.
                logger.debug("ADBC connection does not expose autocommit; ignoring.")
        return self._adbc_conn

    def adbc_cursor(self) -> Any:
        """Return a fresh ADBC DBAPI cursor for an Arrow-shaped call."""
        return self.adbc_conn.cursor()

    @property
    def has_adbc(self) -> bool:
        """Whether the ADBC driver is importable (probe-only, never raises)."""
        return has_adbc()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """Commit the active transaction on both handles, if any."""
        if self._psycopg_conn is not None and not getattr(self._psycopg_conn, "autocommit", True):
            try:
                self._psycopg_conn.commit()
            except Exception:
                logger.exception("psycopg commit failed; continuing.")
        if self._adbc_conn is not None and not getattr(self._adbc_conn, "autocommit", True):
            try:
                self._adbc_conn.commit()
            except Exception:
                logger.exception("ADBC commit failed; continuing.")

    def rollback(self) -> None:
        """Roll back any active transaction on both handles."""
        for label, conn in (("psycopg", self._psycopg_conn), ("ADBC", self._adbc_conn)):
            if conn is None:
                continue
            try:
                conn.rollback()
            except Exception:
                logger.exception("%s rollback failed; continuing.", label)

    def close(self) -> None:
        """Close both connections idempotently."""
        for label, attr in (("psycopg", "_psycopg_conn"), ("ADBC", "_adbc_conn")):
            conn = getattr(self, attr, None)
            if conn is None:
                continue
            try:
                conn.close()
            except Exception:
                logger.exception("%s close failed; continuing.", label)
            setattr(self, attr, None)

    def _release(self, committed: bool = False) -> None:
        """Disposable hook — close on context-manager exit."""
        if committed:
            self.commit()
        self.close()
