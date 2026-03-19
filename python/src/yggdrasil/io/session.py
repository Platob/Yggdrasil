"""HTTP session abstraction with transparent Delta-table caching.

The :class:`Session` base class provides:

* A **thread-safe connection pool** backed by :class:`~yggdrasil.concurrent.threading.JobPoolExecutor`.
* **Single-request** dispatch via :meth:`~Session.send` (implemented by subclasses).
* **Concurrent batch dispatch** via :meth:`~Session.send_many`, with optional Delta
  table caching that de-duplicates live requests against previously stored responses.
* **Convenience HTTP-verb shortcuts**: :meth:`~Session.get`, :meth:`~Session.post`,
  :meth:`~Session.put`, :meth:`~Session.patch`, :meth:`~Session.delete`,
  :meth:`~Session.head`, :meth:`~Session.options`.
* **Spark-native scatter** via :meth:`~Session.spark_send` — distributes requests
  across Spark workers using a broadcast-session UDF.

Public method summary
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - :meth:`~Session.send`
     - Send one :class:`~yggdrasil.io.request.PreparedRequest`; must be
       implemented by subclasses.
   * - :meth:`~Session.send_many`
     - Send many requests in parallel, with optional Delta-table caching.
   * - :meth:`~Session.spark_send`
     - Distribute requests across a Spark cluster via a broadcast-session UDF.
   * - :meth:`~Session.request`
     - Build + dispatch a request from raw components (method, url, …).
   * - :meth:`~Session.prepare_request`
     - Resolve the URL and build a :class:`~yggdrasil.io.request.PreparedRequest`.
   * - ``get`` / ``post`` / ``put`` / ``patch`` / ``delete`` / ``head`` / ``options``
     - HTTP-verb shortcuts that delegate to :meth:`~Session.request`.

Configuration dataclasses
-------------------------
:class:`~yggdrasil.io.send_config.SendConfig`
    Controls a single :meth:`~Session.send` call (wait strategy, error handling,
    streaming, optional Delta cache).
:class:`~yggdrasil.io.send_config.SendManyConfig`
    Extends :class:`~yggdrasil.io.send_config.SendConfig` with batching and
    concurrency options for :meth:`~Session.send_many` and :meth:`~Session.spark_send`.

Config precedence rule
~~~~~~~~~~~~~~~~~~~~~~
When both a *config* object and explicit keyword arguments are supplied to any
send method, **explicit kwargs always win**.  This lets you share a base config
and override individual fields per-call without rebuilding the object::

    base = SendConfig(raise_error=False, wait=30)
    # raise_error is overridden to True for this one call:
    resp = session.get("/critical", config=base, raise_error=True)

Typical usage
-------------
::

    from yggdrasil.io.session import Session
    from yggdrasil.io.send_config import SendConfig, SendManyConfig

    # Single request with a context manager (ensures pool shutdown)
    with Session.from_url("https://api.example.com") as session:
        resp = session.get("/data", config=SendConfig(stream=False))

    # Batch with caching
    many_cfg = SendManyConfig(batch_size=50, ordered=True, cache=my_table)
    for resp in session.send_many(requests, config=many_cfg):
        process(resp)

    # Spark scatter
    spark_df = session.spark_send(requests, config=many_cfg, spark_session=spark)
"""
from __future__ import annotations

import base64
import datetime as dt
import itertools
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Union,
)

import pyarrow as pa
import yggdrasil.pickle.ser as pickle
from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.data import CastOptions, CastOptionsArg, any_to_datetime
from yggdrasil.dataclasses import restore_dataclass_state, serialize_dataclass_state
from yggdrasil.dataclasses.waiting import DEFAULT_WAITING_CONFIG, WaitingConfig, WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io import SaveMode

from .buffer import BytesIO
from .request import PreparedRequest
from .response import RESPONSE_ARROW_SCHEMA, Response
from .send_config import SendConfig, SendManyConfig
from .url import URL

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from yggdrasil.databricks.sql.table import Table

__all__ = ["Session", "SendConfig", "SendManyConfig"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_utc_epoch_us(x: dt.datetime | dt.date | str) -> int:
    """Convert a date, datetime, or ISO-8601 string to UTC microseconds since epoch.

    Parameters
    ----------
    x:
        A :class:`datetime.date`, :class:`datetime.datetime`, or ISO-8601
        string.  Naive datetimes are assumed to be in UTC.

    Returns
    -------
    int
        Microseconds since the Unix epoch in UTC.
    """
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        v = dt.datetime(x.year, x.month, x.day, tzinfo=dt.timezone.utc)
    else:
        v = any_to_datetime(x)

    if v.tzinfo is None:
        v = v.replace(tzinfo=dt.timezone.utc)

    return int(v.astimezone(dt.timezone.utc).timestamp() * 1_000_000)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Session(ABC):
    """Abstract base class for HTTP sessions.

    Subclasses must implement :meth:`send`.  Everything else — pooling,
    caching, batching, verb shortcuts, and Spark scatter — is provided here.

    Parameters
    ----------
    base_url:
        Optional base URL prepended to every relative path passed to
        :meth:`request` and the HTTP-verb shortcuts.  Parsed to a
        :class:`~yggdrasil.io.url.URL` on first use.
    verify:
        Whether to verify TLS certificates.  Passed through to the
        transport layer unchanged.
    pool_maxsize:
        Maximum number of concurrent worker threads in the job pool.
        Values ≤ 0 are silently coerced to 8.
    send_headers:
        Default headers merged into every outgoing request.  Useful for
        API keys, ``User-Agent``, or ``Content-Type`` applied globally.
        See also :attr:`x_api_key` for a typed ``X-API-Key`` shortcut.
    waiting:
        Default :class:`~yggdrasil.dataclasses.waiting.WaitingConfig`
        used when ``wait=None`` is passed to :meth:`send`.  Controls
        timeout, retry interval, back-off factor and maximum retries.

    Notes
    -----
    **Config precedence** — when both a *config* object and explicit keyword
    arguments are provided to any send method, the explicit kwargs always win.
    The config is only consulted for fields whose kwargs equal the method
    default (i.e. the caller did not override them).

    **Thread safety** — the connection pool and job pool are lazily
    initialised under a :class:`threading.RLock`, so the session is safe to
    share across threads.  The ``send_headers`` dict is *not* protected by a
    lock; if you mutate it from multiple threads, add your own synchronisation.

    **Pickle / Spark** — :meth:`__getstate__` / :meth:`__setstate__` are
    implemented via :func:`~yggdrasil.dataclasses.serialize_dataclass_state`,
    so the session can be pickled and broadcast to Spark executors.  The
    internal connection pool and lock are re-created on unpickling.

    Examples
    --------
    Open a session with a context manager (ensures pool shutdown on exit)::

        with Session.from_url("https://api.example.com") as s:
            resp = s.get("/v1/items")

    Share a base :class:`~yggdrasil.io.send_config.SendConfig` and override
    individual fields per-call::

        cfg = SendConfig(raise_error=False, stream=False)
        resp = s.get("/v1/items", config=cfg)
        if not resp.ok:
            log_error(resp)

    Batch dispatch with Delta-table caching::

        many_cfg = SendManyConfig(
            cache=my_delta_table,
            cache_by=["request_url_path", "request_body_hash"],
            batch_size=100,
            ordered=False,
        )
        for resp in s.send_many(iter(prepared_requests), config=many_cfg):
            handle(resp)
    """

    base_url: Optional[URL] = None
    verify: bool = True
    pool_maxsize: int = 10
    send_headers: Optional[dict[str, str]] = field(default=None, repr=False)
    waiting: WaitingConfig = field(
        default_factory=lambda: DEFAULT_WAITING_CONFIG,
        repr=False,
        compare=False,
        hash=False,
    )

    _lock: threading.RLock = field(default=None, init=False, repr=False, compare=False)
    _job_pool: JobPoolExecutor = field(default=None, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Normalise fields and initialise internal state after construction.

        * Parses :attr:`base_url` to a :class:`~yggdrasil.io.url.URL` if a
          plain string was supplied.
        * Creates the :class:`threading.RLock` used to guard lazy
          initialisation of the connection and job pools.
        * Coerces :attr:`pool_maxsize` ≤ 0 to 8.
        """
        if self.base_url:
            self.base_url = URL.parse(self.base_url)
        if self._lock is None:
            self._lock = threading.RLock()
        if self.pool_maxsize <= 0:
            self.pool_maxsize = 8

    def __getstate__(self) -> dict:
        """Return a picklable state dict (excludes lock and pools).

        Uses :func:`~yggdrasil.dataclasses.serialize_dataclass_state` so that
        non-picklable fields (``_lock``, ``_job_pool``) are excluded, enabling
        the session to be broadcast to Spark executors.
        """
        return serialize_dataclass_state(self)

    def __setstate__(self, state: dict) -> None:
        """Restore the session from a pickled state dict.

        Calls :meth:`__post_init__` after restoring fields so that the lock
        and pools are re-created fresh on the receiving process / executor.
        """
        restore_dataclass_state(self, state)
        self.__post_init__()

    def __enter__(self) -> "Session":
        """Support use as a context manager; returns *self*."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Shut down the job pool and release worker threads on context exit."""
        if self._job_pool:
            self._job_pool.shutdown(wait=True)
            self._job_pool = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def job_pool(self) -> JobPoolExecutor:
        """Lazily-initialised thread pool (double-checked locking)."""
        if self._job_pool is None:
            with self._lock:
                if self._job_pool is None:
                    self._job_pool = JobPoolExecutor(max_workers=self.pool_maxsize)
        return self._job_pool

    @property
    def x_api_key(self) -> Optional[str]:
        """Read the ``X-API-Key`` header from :attr:`send_headers`, if present."""
        if self.send_headers:
            return self.send_headers.get("X-API-Key")
        return None

    @x_api_key.setter
    def x_api_key(self, value: Optional[str]) -> None:
        """Set the ``X-API-Key`` header in :attr:`send_headers`."""
        if value:
            if not self.send_headers:
                self.send_headers = {}
            self.send_headers["X-API-Key"] = value

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_url(
        cls,
        url: Union[URL, str],
        *,
        verify: bool = True,
        normalize: bool = True,
        waiting: WaitingConfigArg = True,
    ) -> "Session":
        """Construct the appropriate :class:`Session` subclass for *url*.

        Currently supports ``http://`` and ``https://`` schemes, returning an
        :class:`~yggdrasil.io.http_.HTTPSession`.

        Parameters
        ----------
        url:
            Target base URL.
        verify:
            Whether to verify TLS certificates.
        normalize:
            Whether to normalize the URL (e.g. lower-case scheme/host).
        waiting:
            Default retry/wait configuration for the session.

        Returns
        -------
        Session
            A concrete session instance ready to send requests.

        Raises
        ------
        ValueError
            If the URL scheme is not supported.
        """
        url = URL.parse(url, normalize=normalize)

        if url.scheme.startswith("http"):
            from .http_ import HTTPSession

            return HTTPSession(
                base_url=url,
                verify=verify,
                waiting=WaitingConfig.check_arg(waiting) if waiting is not None else None,
            )

        raise ValueError(f"Cannot build session from scheme: {url.scheme!r}")

    # ------------------------------------------------------------------
    # Abstract transport
    # ------------------------------------------------------------------

    @abstractmethod
    def send(
        self,
        request: PreparedRequest,
        *,
        config: Optional[SendConfig] = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        cache: Optional["Table"] = None,
        cache_by: Optional[list[str]] = None,
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
        anonymize: Literal["remove", "redact"] = "remove",
        wait_cache: WaitingConfigArg = False,
    ) -> Response:
        """Send a single prepared request and return its response.

        This is the only method subclasses **must** implement.

        Parameters
        ----------
        request:
            The fully prepared request to dispatch.
        config:
            Optional :class:`SendConfig` providing defaults for every other
            keyword argument.  Explicit kwargs override the config.
        wait:
            Retry / waiting strategy.  ``None`` defers to ``config.wait``
            or the session's :attr:`waiting` default.
        raise_error:
            Raise on non-2xx responses when ``True`` (default).
        stream:
            Stream the response body lazily when ``True`` (default).
        cache:
            Delta table used to cache responses.  A cache hit returns the
            stored response without hitting the network.
        cache_by:
            Column names that form the cache lookup key.  Defaults to the
            standard request-fingerprint columns when ``cache`` is set.
        received_from:
            Earliest acceptable cached response timestamp.
        received_to:
            Latest acceptable cached response timestamp.
        anonymize:
            How to strip sensitive fields before the cache lookup
            (``"remove"`` or ``"redact"``).
        wait_cache:
            Waiting config for the background cache-write.  ``False``
            means fire-and-forget.

        Returns
        -------
        Response
            The HTTP response.

        Raises
        ------
        NotImplementedError
            Always — subclasses must override this method.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Cache helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_by_keys(arg: Optional[list[str]] = None) -> list[str]:
        """Resolve and validate cache-key column names.

        Parameters
        ----------
        arg:
            User-supplied column list.  ``None`` or empty falls back to the
            standard request fingerprint columns.

        Returns
        -------
        list[str]
            Validated list of column names present in
            :data:`~yggdrasil.io.response.RESPONSE_ARROW_SCHEMA`.

        Raises
        ------
        ValueError
            If any name is not in the response Arrow schema.
        """
        if not arg:
            arg = [
                "request_method",
                "request_url_host",
                "request_url_path",
                "request_url_query",
                "request_content_length",
                "request_body_hash",
            ]

        invalid = [k for k in arg if k not in RESPONSE_ARROW_SCHEMA.names]
        if invalid:
            raise ValueError(
                f"Invalid cache_by key(s): {invalid!r}. "
                f"Must be within: {RESPONSE_ARROW_SCHEMA.names}"
            )
        return arg

    @staticmethod
    def _cache_value_from_request(request: PreparedRequest, key: str) -> Any:
        """Extract the value of a single cache-key field from a prepared request.

        A fixed dispatch table handles the standard request-fingerprint fields.
        Any field not in the table falls back to ``getattr(request, key)``,
        allowing future :class:`~yggdrasil.io.request.PreparedRequest`
        extensions without changing this method.

        Parameters
        ----------
        request:
            The prepared request to inspect.
        key:
            One of the recognised cache-key field names (e.g.
            ``"request_method"``, ``"request_url_host"``,
            ``"request_body_hash"``).

        Returns
        -------
        Any
            The extracted value, or ``None`` if the field is absent / empty.

        Raises
        ------
        ValueError
            If *key* is not in the dispatch table and not an attribute of
            *request*.
        """
        _EXTRACTORS: dict[str, Callable[[PreparedRequest], Any]] = {
            "request_method":         lambda r: r.method,
            "request_url":            lambda r: r.url.to_string(),
            "request_url_scheme":     lambda r: r.url.scheme,
            "request_url_host":       lambda r: r.url.host,
            "request_url_port":       lambda r: r.url.port,
            "request_url_path":       lambda r: r.url.path,
            "request_url_query":      lambda r: r.url.query,
            "request_body_hash":      lambda r: r.body.xxh3_int64() if r.body else None,
            "request_content_length": lambda r: r.content_length,
        }
        extractor = _EXTRACTORS.get(key)
        if extractor is not None:
            return extractor(request)
        # Generic fallback for future PreparedRequest attributes
        if hasattr(request, key):
            return getattr(request, key)
        raise ValueError(f"Unsupported request cache_by key: {key!r}")

    @classmethod
    def _cache_values_from_request(
        cls,
        request: PreparedRequest,
        keys: list[str],
    ) -> dict[str, Any]:
        """Extract multiple cache-key values from *request* into a mapping.

        Parameters
        ----------
        request:
            The prepared request to inspect.
        keys:
            Ordered list of cache-key field names.

        Returns
        -------
        dict[str, Any]
            ``{key: value}`` for every key in *keys*, preserving order.
        """
        return {k: cls._cache_value_from_request(request, k) for k in keys}

    @classmethod
    def _cache_value_from_response(cls, response: Response, key: str) -> Any:
        """Extract a single cache-key value from a *response* or its embedded request.

        Parameters
        ----------
        response:
            The response to inspect.  If it carries an embedded
            :class:`~yggdrasil.io.request.PreparedRequest`, request-prefixed
            keys (``"request_*"``) are resolved from that object.
        key:
            Cache-key field name.

        Returns
        -------
        Any

        Raises
        ------
        ValueError
            If *key* cannot be resolved from the response or its request.
        """
        if hasattr(response, key):
            return getattr(response, key)
        if key.startswith("request_"):
            return cls._cache_value_from_request(response.request, key)
        raise ValueError(f"Unsupported response cache_by key: {key!r}")

    @classmethod
    def _cache_tuple_from_request(
        cls,
        request: PreparedRequest,
        keys: list[str],
    ) -> tuple:
        """Build a hashable cache-key tuple from a request.

        Parameters
        ----------
        request:
            The prepared request.
        keys:
            Ordered list of cache-key field names.  The tuple preserves
            this order so it can be compared against tuples built from
            :meth:`_cache_tuple_from_response` using the same *keys*.

        Returns
        -------
        tuple
            ``(value_for_keys[0], value_for_keys[1], …)``
        """
        values = cls._cache_values_from_request(request, keys)
        return tuple(values[k] for k in keys)

    @classmethod
    def _cache_tuple_from_response(
        cls,
        response: Response,
        keys: list[str],
    ) -> tuple:
        """Build a hashable cache-key tuple from a response.

        Parameters
        ----------
        response:
            The response (may embed a request for ``"request_*"`` keys).
        keys:
            Ordered list of cache-key field names.

        Returns
        -------
        tuple
            ``(value_for_keys[0], value_for_keys[1], …)``
        """
        return tuple(cls._cache_value_from_response(response, k) for k in keys)

    @staticmethod
    def _sql_literal(value: Any) -> str:
        """Format *value* as a SQL literal safe for embedding in a WHERE clause.

        Handles the types that appear in the response cache schema:

        * ``None`` → ``null``
        * ``bool`` → ``true`` / ``false``  (checked before ``int`` because
          :class:`bool` is a subclass of :class:`int`)
        * ``int`` / ``float`` → bare numeric string
        * :class:`datetime.datetime` → ``timestamp 'YYYY-MM-DD HH:MM:SS.ffffff'``
        * ``bytes`` → base-64 encoded, single-quoted string
        * anything else → ``str(value)``, single-quoted with ``'`` escaped as ``''``

        Parameters
        ----------
        value:
            The Python value to serialise.

        Returns
        -------
        str
            A SQL literal fragment (no trailing semicolon).
        """
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, dt.datetime):
            return f"timestamp '{value.isoformat(sep=' ', timespec='microseconds')}'"
        if isinstance(value, bytes):
            value = base64.b64encode(value).decode("ascii")
        else:
            value = str(value)
        return f"'{value.replace(chr(39), chr(39) * 2)}'"

    @classmethod
    def _sql_match_clause(
        cls,
        request: Optional[PreparedRequest],
        keys: list[str],
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
    ) -> str:
        """Build a SQL WHERE clause that matches *request* against the cache table.

        Parameters
        ----------
        request:
            The request whose field values become equality predicates.
            Pass ``None`` to skip request predicates (time-range only).
        keys:
            Cache-key column names to include in the WHERE clause.
        received_from:
            Lower bound on ``response_received_at_epoch`` (inclusive).
        received_to:
            Upper bound on ``response_received_at_epoch`` (inclusive).

        Returns
        -------
        str
            A SQL fragment suitable for use after ``WHERE``.  Returns
            ``"1=1"`` when no predicates are required.
        """
        clauses: list[str] = []

        if request is not None and keys:
            for key, value in cls._cache_values_from_request(request, keys).items():
                if value is None:
                    clauses.append(f"{key} IS NULL")
                else:
                    clauses.append(f"{key} = {cls._sql_literal(value)}")

        if received_from not in (None, ""):
            clauses.append(
                f"response_received_at_epoch >= {to_utc_epoch_us(received_from)}"
            )
        if received_to not in (None, ""):
            clauses.append(
                f"response_received_at_epoch <= {to_utc_epoch_us(received_to)}"
            )

        return " AND ".join(clauses) if clauses else "1=1"

    # ------------------------------------------------------------------
    # Config resolution helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_send_config(
        config: Optional[SendConfig],
        *,
        wait: WaitingConfigArg,
        raise_error: bool,
        stream: bool,
        cache: Optional["Table"],
        cache_by: Optional[list[str]],
        anonymize: Literal["remove", "redact"],
        received_from: Optional[dt.datetime | dt.date | str],
        received_to: Optional[dt.datetime | dt.date | str],
        wait_cache: WaitingConfigArg,
    ) -> SendConfig:
        """Merge *config* with explicit keyword arguments into a resolved :class:`SendConfig`.

        Explicit kwargs always win over the config object.  A ``None`` config
        is equivalent to :meth:`SendConfig.default`.

        Parameters
        ----------
        config:
            Base config or ``None``.
        wait, raise_error, stream, cache, cache_by, anonymize,
        received_from, received_to, wait_cache:
            Call-site keyword arguments that take precedence over the config.

        Returns
        -------
        SendConfig
            Fully resolved configuration with no ``None`` ambiguity.
        """
        base = config or SendConfig.default()
        return SendConfig(
            wait=wait if wait is not None else base.wait,
            raise_error=raise_error if raise_error is not True else base.raise_error,
            stream=stream if stream is not True else base.stream,
            cache=cache if cache is not None else base.cache,
            cache_by=cache_by if cache_by is not None else base.cache_by,
            anonymize=anonymize if anonymize != "remove" else base.anonymize,
            received_from=received_from if received_from is not None else base.received_from,
            received_to=received_to if received_to is not None else base.received_to,
            wait_cache=wait_cache if wait_cache is not False else base.wait_cache,
        )

    @staticmethod
    def _resolve_send_many_config(
        config: Optional[SendManyConfig],
        *,
        wait: WaitingConfigArg,
        raise_error: bool,
        normalize: Optional[bool],
        stream: bool,
        cache: Optional["Table"],
        cache_by: Optional[list[str]],
        cache_anonymize: Literal["remove", "redact"],
        received_from: Optional[dt.datetime | dt.date | str],
        received_to: Optional[dt.datetime | dt.date | str],
        wait_cache: WaitingConfigArg,
        batch_size: Optional[int],
        ordered: bool,
        max_in_flight: Optional[int],
    ) -> SendManyConfig:
        """Merge *config* with explicit keyword arguments into a resolved :class:`SendManyConfig`.

        Explicit kwargs always win over the config object.  A ``None`` config
        is equivalent to :meth:`SendManyConfig.default`.

        Parameters
        ----------
        config:
            Base config or ``None``.
        wait, raise_error, normalize, stream, cache, cache_by, cache_anonymize,
        received_from, received_to, wait_cache, batch_size, ordered, max_in_flight:
            Call-site keyword arguments that take precedence over the config.

        Returns
        -------
        SendManyConfig
            Fully resolved configuration with no ``None`` ambiguity.
        """
        base = config or SendManyConfig.default()
        return SendManyConfig(
            wait=wait if wait is not None else base.wait,
            raise_error=raise_error if raise_error is not True else base.raise_error,
            normalize=normalize if normalize is not None else base.normalize,
            stream=stream if stream is not True else base.stream,
            cache=cache if cache is not None else base.cache,
            cache_by=cache_by if cache_by is not None else base.cache_by,
            cache_anonymize=cache_anonymize if cache_anonymize != "remove" else base.cache_anonymize,
            received_from=received_from if received_from is not None else base.received_from,
            received_to=received_to if received_to is not None else base.received_to,
            wait_cache=wait_cache if wait_cache is not False else base.wait_cache,
            batch_size=batch_size if batch_size is not None else base.batch_size,
            ordered=ordered if ordered is not False else base.ordered,
            max_in_flight=max_in_flight if max_in_flight is not None else base.max_in_flight,
        )

    # ------------------------------------------------------------------
    # Batch dispatch with optional Delta-table caching
    # ------------------------------------------------------------------

    def send_many(
        self,
        requests: Iterator[PreparedRequest],
        *,
        config: Optional[SendManyConfig] = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: Optional[bool] = None,
        stream: bool = True,
        cache: Optional["Table"] = None,
        cache_by: Optional[list[str]] = None,
        cache_anonymize: Literal["remove", "redact"] = "remove",
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
        wait_cache: WaitingConfigArg = False,
        batch_size: Optional[int] = None,
        ordered: bool = False,
        max_in_flight: Optional[int] = None,
    ) -> Iterator[Response]:
        """Send multiple requests concurrently, with optional Delta-table caching.

        Requests are dispatched in parallel up to :attr:`pool_maxsize`.  When
        a *cache* table is supplied, each batch is first looked up in the
        table; only cache misses hit the network.  Successful live responses
        are written back to the cache asynchronously.

        Parameters
        ----------
        requests:
            Iterator of :class:`PreparedRequest` objects to dispatch.
        config:
            Optional :class:`SendManyConfig` providing defaults for every
            other keyword argument.  Explicit kwargs override the config.
        wait:
            Retry / waiting strategy forwarded to each :meth:`send` call.
            ``None`` defers to ``config.wait`` or the session default.
        raise_error:
            When ``True`` (default), any failed response raises after all
            in-flight requests in the current batch complete.
        normalize:
            Whether to normalise request URLs.  ``None`` (default) enables
            normalisation automatically when ``cache`` is set.
        stream:
            Whether to stream response bodies lazily.
        cache:
            Delta table used as a response cache.  Cache hits are returned
            immediately; misses are fetched and written back.
        cache_by:
            Column names forming the cache key.  Defaults to the standard
            request fingerprint when ``cache`` is set.
        cache_anonymize:
            How to strip sensitive fields before cache reads/writes
            (``"remove"`` or ``"redact"``).
        received_from:
            Only return cached responses received on or after this timestamp.
        received_to:
            Only return cached responses received on or before this timestamp.
        wait_cache:
            Waiting config for background cache writes.  ``False`` is
            fire-and-forget.
        batch_size:
            Requests per cache-lookup batch.  ``None`` defaults to
            ``pool_maxsize × 100``.
        ordered:
            When ``True``, yield responses in input order.  ``False``
            (default) yields in completion order for higher throughput.
        max_in_flight:
            Maximum concurrent network requests.  ``None`` lets the pool
            decide.

        Yields
        ------
        Response
            Successful responses (or all responses when ``raise_error=False``).

        Raises
        ------
        Exception
            Re-raises the last failure in a batch when ``raise_error=True``
            and at least one request fails.
        """
        # Merge config + explicit kwargs → single resolved config
        cfg = self._resolve_send_many_config(
            config,
            wait=wait,
            raise_error=raise_error,
            normalize=normalize,
            stream=stream,
            cache=cache,
            cache_by=cache_by,
            cache_anonymize=cache_anonymize,
            received_from=received_from,
            received_to=received_to,
            wait_cache=wait_cache,
            batch_size=batch_size,
            ordered=ordered,
            max_in_flight=max_in_flight,
        )
        wait          = cfg.wait
        raise_error   = cfg.raise_error
        normalize     = cfg.normalize
        stream        = cfg.stream
        cache         = cfg.cache
        cache_by      = cfg.cache_by
        cache_anonymize = cfg.cache_anonymize
        received_from = cfg.received_from
        received_to   = cfg.received_to
        wait_cache    = cfg.wait_cache
        batch_size    = cfg.batch_size
        ordered       = cfg.ordered
        max_in_flight = cfg.max_in_flight

        if normalize is None:
            normalize = cache is not None

        # Resolve & validate cache keys up front
        if cache is not None:
            cache_by = self._cache_by_keys(cache_by)
            cache_request_by = [k for k in cache_by if k.startswith("request")]
        else:
            cache_request_by = []

        pool = self.job_pool
        if not batch_size:
            batch_size = pool.max_workers * 100

        # ----------------------------------------------------------
        # Fast path: no caching — just fan out all requests
        # ----------------------------------------------------------
        if cache is None:
            def _uncached_jobs() -> Iterator[Job]:
                for req in requests:
                    yield Job.make(
                        self.send,
                        req,
                        wait=wait,
                        raise_error=raise_error,
                        stream=stream,
                    )

            for result in pool.as_completed(
                _uncached_jobs(),
                ordered=ordered,
                max_in_flight=self.pool_maxsize,
                cancel_on_exit=True,
                shutdown_on_exit=True,
                raise_error=True,
            ):
                resp: Response = result.result
                if raise_error:
                    resp.raise_for_status()
                yield resp
            return

        # ----------------------------------------------------------
        # Cached path: batch → lookup → miss → fetch → write back
        # ----------------------------------------------------------
        def _batched(it: Iterator, n: int) -> Iterator[list]:
            it = iter(it)
            while True:
                batch = list(itertools.islice(it, n))
                if not batch:
                    break
                yield batch

        for batch in _batched(requests, batch_size):
            # Anonymise the batch for the cache lookup
            anon_batch = [
                req.anonymize(mode=cache_anonymize) if cache_anonymize else req
                for req in batch
            ]

            # Build a compound WHERE clause: (req1 OR req2 OR …) AND time_range
            request_clauses = " OR ".join(
                f"({self._sql_match_clause(req, keys=cache_request_by)})"
                for req in anon_batch
            )
            time_clause = self._sql_match_clause(
                None,
                keys=[],
                received_from=received_from,
                received_to=received_to,
            )
            where_parts: list[str] = []
            if request_clauses:
                where_parts.append(f"({request_clauses})")
            if time_clause and time_clause != "1=1":
                where_parts.append(f"({time_clause})")

            base_query = f"SELECT * FROM {cache.full_name(safe=True)}"
            if where_parts:
                base_query += " WHERE " + " AND ".join(where_parts)

            # Deduplicate: keep only the most-recent response per cache key
            if cache_request_by:
                partition_by = ", ".join(cache_request_by)
                query = (
                    f"SELECT * FROM ("
                    f"  SELECT t.*, row_number() OVER ("
                    f"    PARTITION BY {partition_by}"
                    f"    ORDER BY response_received_at_epoch DESC"
                    f"  ) AS __rn"
                    f"  FROM ({base_query}) t"
                    f") ranked WHERE __rn = 1"
                )
            else:
                query = base_query

            # Query cache (auto-create table on first run if missing)
            try:
                cache_result = cache.sql.execute(query)
            except Exception as exc:
                if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                    cache.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                    cache_result = cache.sql.execute(query)
                else:
                    raise

            # Parse cached responses into a lookup map
            cached_arrow = cache_result.to_arrow_table()
            cached_responses = list(Response.from_arrow(cached_arrow))
            cache_map: dict[tuple, Response] = {
                self._cache_tuple_from_response(r, cache_request_by): r
                for r in cached_responses
            }

            # Partition the batch into hits and misses
            hits: list[Response] = []
            misses: list[PreparedRequest] = []
            for req in batch:
                anon_req = req.anonymize(mode="remove")
                key = self._cache_tuple_from_request(anon_req, cache_request_by)
                if key in cache_map:
                    hits.append(cache_map[key])
                else:
                    misses.append(req)

            # Yield cache hits immediately
            yield from hits

            if not misses:
                continue

            # Fetch misses from the network
            def _miss_jobs() -> Iterator[Job]:
                for req in misses:
                    yield Job.make(
                        self.send,
                        req,
                        wait=wait,
                        raise_error=False,
                        stream=stream,
                        cache=None,
                    )

            to_insert: list[Response] = []
            failed: list[Response] = []

            for result in pool.as_completed(
                _miss_jobs(),
                ordered=ordered,
                max_in_flight=max_in_flight,
                cancel_on_exit=False,
                shutdown_on_exit=False,
                raise_error=True,
            ):
                resp = result.result
                if resp.ok:
                    to_insert.append(resp)
                    yield resp
                elif raise_error:
                    failed.append(resp)

            # Write successful responses back to the cache
            if to_insert:
                batches = [
                    r.anonymize(mode="remove").to_arrow_batch(parse=False)
                    for r in to_insert
                ]
                combined = pa.Table.from_batches(batches).combine_chunks()
                cache.insert(
                    combined,
                    mode=SaveMode.APPEND,
                    match_by=cache_by,
                    wait=wait_cache,
                )

            if raise_error and failed:
                failed[-1].raise_for_status()

    # ------------------------------------------------------------------
    # Convenience HTTP-verb methods
    # ------------------------------------------------------------------

    def get(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        config: Optional[SendConfig] = None,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs: Any,
    ) -> Response:
        """Send a GET request.

        Parameters
        ----------
        url:
            Relative or absolute URL.  Relative URLs are resolved against
            :attr:`base_url`.
        config:
            Optional :class:`SendConfig` for this request.
        params:
            Query-string parameters merged into *url*.
        headers:
            Additional headers merged with :attr:`send_headers`.
        body:
            Optional request body.
        tags:
            Arbitrary string tags attached to the prepared request.
        stream:
            Stream the response body lazily (default ``True``).
        wait:
            Retry / waiting strategy.
        normalize:
            Normalise the URL before sending.
        cache:
            Delta table for response caching.
        **kwargs:
            Extra keyword arguments forwarded to :meth:`request`.
        """
        return self.request(
            "GET", url,
            config=config, params=params, headers=headers, body=body,
            tags=tags, stream=stream, wait=wait, normalize=normalize,
            cache=cache, **kwargs,
        )

    def post(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        config: Optional[SendConfig] = None,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs: Any,
    ) -> Response:
        """Send a POST request.  See :meth:`get` for parameter documentation."""
        return self.request(
            "POST", url,
            config=config, params=params, headers=headers, body=body,
            tags=tags, json=json, stream=stream, wait=wait,
            normalize=normalize, cache=cache, **kwargs,
        )

    def put(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        config: Optional[SendConfig] = None,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs: Any,
    ) -> Response:
        """Send a PUT request.  See :meth:`get` for parameter documentation."""
        return self.request(
            "PUT", url,
            config=config, params=params, headers=headers, body=body,
            tags=tags, json=json, stream=stream, wait=wait,
            normalize=normalize, cache=cache, **kwargs,
        )

    def patch(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        config: Optional[SendConfig] = None,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs: Any,
    ) -> Response:
        """Send a PATCH request.  See :meth:`get` for parameter documentation."""
        return self.request(
            "PATCH", url,
            config=config, params=params, headers=headers, body=body,
            tags=tags, json=json, stream=stream, wait=wait,
            normalize=normalize, cache=cache, **kwargs,
        )

    def delete(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        config: Optional[SendConfig] = None,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs: Any,
    ) -> Response:
        """Send a DELETE request.  See :meth:`get` for parameter documentation."""
        return self.request(
            "DELETE", url,
            config=config, params=params, headers=headers, body=body,
            tags=tags, json=json, stream=stream, wait=wait,
            normalize=normalize, cache=cache, **kwargs,
        )

    def head(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        config: Optional[SendConfig] = None,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        stream: bool = False,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs: Any,
    ) -> Response:
        """Send a HEAD request (``stream=False`` by default).

        See :meth:`get` for parameter documentation.
        """
        return self.request(
            "HEAD", url,
            config=config, params=params, headers=headers, body=body,
            tags=tags, stream=stream, wait=wait, normalize=normalize,
            cache=cache, **kwargs,
        )

    def options(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        config: Optional[SendConfig] = None,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs: Any,
    ) -> Response:
        """Send an OPTIONS request.  See :meth:`get` for parameter documentation."""
        return self.request(
            "OPTIONS", url,
            config=config, params=params, headers=headers, body=body,
            tags=tags, json=json, stream=stream, wait=wait,
            normalize=normalize, cache=cache, **kwargs,
        )

    # ------------------------------------------------------------------
    # Request orchestration
    # ------------------------------------------------------------------

    def request(
        self,
        method: str,
        url: Optional[Union[URL, str]] = None,
        *,
        config: Optional[SendConfig] = None,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        before_send: Optional[Callable[[PreparedRequest], PreparedRequest]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
    ) -> Response:
        """Build and dispatch an HTTP request.

        Combines URL resolution, header merging, body serialisation, and
        :meth:`send` dispatch into a single call.  The HTTP-verb shortcuts
        (:meth:`get`, :meth:`post`, …) delegate here.

        Parameters
        ----------
        method:
            HTTP method string (e.g. ``"GET"``, ``"POST"``).
        url:
            Relative or absolute URL.
        config:
            Optional :class:`SendConfig` providing defaults.
        params:
            Query-string parameters merged into *url*.
        headers:
            Additional request headers.
        body:
            Raw request body.
        tags:
            Arbitrary string tags attached to the prepared request.
        before_send:
            Optional hook called on the :class:`PreparedRequest` immediately
            before dispatch.  Useful for signing or adding per-request headers.
        json:
            JSON-serialisable body.  Mutually exclusive with *body*.
        stream:
            Stream the response body lazily.
        wait:
            Retry / waiting strategy.
        normalize:
            Normalise the URL.  ``None`` enables normalisation when
            *cache* is set.
        cache:
            Delta table for response caching.

        Returns
        -------
        Response
        """
        if normalize is None:
            normalize = cache is not None

        prepared = self.prepare_request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            before_send=before_send,
        )
        return self.send(
            prepared,
            config=config,
            stream=stream,
            wait=wait,
            cache=cache,
        )

    def prepare_request(
        self,
        method: str,
        url: Optional[Union[URL, str]] = None,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        before_send: Optional[Callable[[PreparedRequest], PreparedRequest]] = None,
        after_received: Optional[Callable[[Response], Response]] = None,
        *,
        json: Optional[Any] = None,
        normalize: bool = True,
    ) -> PreparedRequest:
        """Resolve the URL and construct a :class:`PreparedRequest`.

        Parameters
        ----------
        method:
            HTTP method.
        url:
            Relative or absolute target.  Resolved against :attr:`base_url`
            when set.
        params:
            Query-string parameters merged into *url*.
        headers:
            Additional headers.
        body:
            Raw request body.
        tags:
            Arbitrary metadata attached to the request.
        before_send:
            Hook applied after preparation, before dispatch.
        after_received:
            Hook applied to the raw :class:`Response` after receipt.
        json:
            JSON body (mutually exclusive with *body*).
        normalize:
            Normalise the URL (default ``True``).

        Returns
        -------
        PreparedRequest

        Raises
        ------
        ValueError
            If both *url* and :attr:`base_url` are ``None``.
        """
        full_url: Union[URL, str, None] = url
        if self.base_url:
            full_url = self.base_url.join(url) if url else self.base_url
        elif url is None:
            raise ValueError("url is required when base_url is not set on the session.")

        if params:
            u = URL.parse(full_url, normalize=normalize)
            items = list(u.query_items(keep_blank_values=True))
            items.extend((k, v) for k, v in params.items())
            full_url = u.with_query_items(tuple(items))

        return PreparedRequest.prepare(
            method=method,
            url=full_url,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            before_send=before_send,
            after_received=after_received,
        )

    # ------------------------------------------------------------------
    # Spark scatter
    # ------------------------------------------------------------------

    def spark_send(
        self,
        requests: Iterator[PreparedRequest],
        *,
        parse: Union[CastOptionsArg, bool] = None,
        apply: Optional[Callable[[Response], Any]] = None,
        config: Optional[SendManyConfig] = None,
        spark_session: Optional["SparkSession"] = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: Optional[bool] = None,
        stream: bool = True,
        cache: Optional["Table"] = None,
        cache_by: Optional[list[str]] = None,
        cache_anonymize: Literal["remove", "redact"] = "remove",
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
        wait_cache: WaitingConfigArg = False,
        batch_size: Optional[int] = None,
        ordered: bool = False,
        max_in_flight: Optional[int] = None,
    ):
        """Distribute requests across Spark workers via ``mapInArrow``.

        Requests are serialised as native Arrow rows conforming to
        :data:`~yggdrasil.io.request.REQUEST_ARROW_SCHEMA` — no pickling is
        involved on the input side.  Each executor deserialises its partition
        back into :class:`PreparedRequest` objects via
        :meth:`PreparedRequest.from_arrow`, sends them through
        :meth:`send_many`, and yields Arrow batches for the responses.

        The output schema is controlled by *parse*:

        ``parse=None`` / ``parse=False``
            Each response is serialised with
            ``resp.to_arrow_batch(parse=False)`` yielding the full
            :data:`~yggdrasil.io.response.RESPONSE_ARROW_SCHEMA` (request
            metadata + response metadata + raw body bytes).

        ``parse=True``
            The **first** request is sent eagerly on the driver to discover
            the parsed body Arrow schema.  Executors yield
            ``resp.to_arrow_batch(parse=True)`` — only the parsed body
            columns.

        ``parse=<CastOptionsArg>``
            A :class:`pa.Schema`, :class:`~yggdrasil.data.CastOptions`,
            etc.  The schema is extracted without a probe.  If it resolves
            to ``None``, falls back to ``RESPONSE_ARROW_SCHEMA`` (like
            ``parse=None``).

        Parameters
        ----------
        requests:
            Iterator of :class:`~yggdrasil.io.request.PreparedRequest`
            objects.  Consumed eagerly to build the input Arrow DataFrame.
        parse:
            Controls how responses are returned:

            * ``None`` / ``False`` — full response schema
              (:data:`RESPONSE_ARROW_SCHEMA`).
            * ``True`` — parsed body columns; schema discovered from a
              probe request.
            * Any :data:`~yggdrasil.data.CastOptionsArg` value — parsed
              body with an explicit schema.  Falls back to
              ``RESPONSE_ARROW_SCHEMA`` when no schema can be derived.
        apply:
            Optional transform applied to each :class:`Response` on the
            executor **before** Arrow serialisation.
        config:
            Optional :class:`~yggdrasil.io.send_config.SendManyConfig`.
            Explicit kwargs override the config.
        spark_session:
            Active :class:`pyspark.sql.SparkSession`.  When ``None`` one
            is created (or retrieved) automatically.
        wait:
            Retry / waiting strategy for each executor.
        raise_error:
            When ``True`` a non-2xx response raises inside the executor.
        normalize:
            Whether to normalise request URLs.
        stream:
            Whether to stream response bodies lazily on executors.
        cache:
            Delta table used as a response cache on each executor.
        cache_by:
            Column names forming the cache key.
        cache_anonymize:
            How sensitive fields are stripped before cache reads/writes.
        received_from:
            Earliest acceptable cached-response timestamp.
        received_to:
            Latest acceptable cached-response timestamp.
        wait_cache:
            Waiting config for background cache writes on executors.
        batch_size:
            Requests per cache-lookup batch on each executor.
        ordered:
            When ``True`` responses are returned in input order.
        max_in_flight:
            Maximum concurrent network requests per executor.

        Returns
        -------
        pyspark.sql.DataFrame
            When ``parse`` is ``None`` / ``False`` (or schema falls back):
            columns from :data:`RESPONSE_ARROW_SCHEMA`.

            When ``parse`` resolves to a body schema: columns from the
            parsed response body.

        Notes
        -----
        Because the session is broadcast, **all executors share the same
        session state**.  Mutable fields are copied at broadcast time.

        Examples
        --------
        Full response metadata (default)::

            df = session.spark_send(reqs)

        Parsed body — auto-discover schema::

            df = session.spark_send(reqs, parse=True)

        Parsed body — explicit schema::

            schema = pa.schema([("id", pa.int64()), ("name", pa.utf8())])
            df = session.spark_send(reqs, parse=schema)

        With a per-response transform::

            df = session.spark_send(reqs, parse=True, apply=my_transform)
        """
        from yggdrasil.spark.cast import arrow_schema_to_spark_schema
        from .request import REQUEST_ARROW_SCHEMA

        serialized_session = pickle.dumps(self)

        if spark_session is None:
            spark_session = PyEnv.spark_session(
                create=True, import_error=True, install_spark=True,
            )

        # --- Materialise requests into Arrow ----------------------------
        request_list = list(requests)
        request_batches = [req.to_arrow_batch() for req in request_list]

        if request_batches:
            request_table = pa.Table.from_batches(request_batches)
        else:
            request_table = pa.table(
                {f.name: pa.array([], type=f.type) for f in REQUEST_ARROW_SCHEMA},
            )

        request_spark_schema = arrow_schema_to_spark_schema(REQUEST_ARROW_SCHEMA)
        request_df = spark_session.createDataFrame(
            request_table.to_pandas(),
            schema=request_spark_schema,
        )

        # --- Broadcast session + config ---------------------------------
        bc_session = spark_session.sparkContext.broadcast(serialized_session)
        bc_config = spark_session.sparkContext.broadcast(
            pickle.dumps(
                config or SendManyConfig(
                    wait=wait,
                    raise_error=raise_error,
                    normalize=normalize,
                    stream=stream,
                    cache=cache,
                    cache_by=cache_by,
                    cache_anonymize=cache_anonymize,
                    received_from=received_from,
                    received_to=received_to,
                    wait_cache=wait_cache,
                    batch_size=batch_size,
                    ordered=ordered,
                    max_in_flight=max_in_flight,
                )
            )
        )

        # --- Resolve output Arrow schema --------------------------------
        parse_body: bool

        if parse is None or parse is False:
            arrow_schema = RESPONSE_ARROW_SCHEMA
            parse_body = False
        elif parse is True:
            if not request_list:
                raise ValueError(
                    "spark_send(parse=True) requires at least one request "
                    "to discover the output schema, but the request "
                    "iterator was empty."
                )
            probe_resp = self.send(
                request_list[0],
                wait=wait,
                raise_error=raise_error,
                stream=stream,
            )
            if apply is not None:
                probe_resp = apply(probe_resp)
            arrow_schema = probe_resp.to_arrow_table(parse=True).schema
            parse_body = True
        else:
            # parse is a CastOptionsArg
            if isinstance(parse, pa.Schema):
                arrow_schema = parse
            else:
                opts = CastOptions.check_arg(parse)
                arrow_schema = opts.target_arrow_schema

            if arrow_schema is None:
                arrow_schema = RESPONSE_ARROW_SCHEMA
                parse_body = False
            else:
                parse_body = True

        spark_schema = arrow_schema_to_spark_schema(arrow_schema)

        # --- Broadcast helpers ------------------------------------------
        bc_apply = (
            spark_session.sparkContext.broadcast(pickle.dumps(apply))
            if apply is not None else None
        )
        bc_parse_body = spark_session.sparkContext.broadcast(parse_body)

        # --- mapInArrow -------------------------------------------------
        def _map_arrow(batch_iter: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
            """Reconstruct requests from Arrow, send, yield response batches."""
            import yggdrasil.pickle.ser as _pickle

            _session: Session = _pickle.loads(bc_session.value)
            _cfg: SendManyConfig = _pickle.loads(bc_config.value)
            _apply_fn: Optional[Callable] = (
                _pickle.loads(bc_apply.value) if bc_apply is not None else None
            )
            _parse = bc_parse_body.value

            for batch in batch_iter:
                for _req in PreparedRequest.from_arrow(batch):
                    _resp: Response = next(iter(
                        _session.send_many([_req], config=_cfg)
                    ))
                    if _apply_fn is not None:
                        _resp = _apply_fn(_resp)
                    if _parse:
                        yield _resp.to_arrow_table(parse=True).to_batches()[0]
                    else:
                        yield _resp.to_arrow_batch(parse=False)

        return request_df.mapInArrow(_map_arrow, schema=spark_schema)


