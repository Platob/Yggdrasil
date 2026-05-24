"""Concrete HTTP/HTTPS session — the single public entry point of :mod:`yggdrasil.http_`.

Construct one :class:`HTTPSession` per host (singleton-cached by config), drive
verb methods (``get`` / ``post`` / ``put`` / ``patch`` / ``delete`` / ``head``
/ ``options`` / ``request``), and read the returned :class:`HTTPResponse`. All
the HTTP machinery lives on this class:

* connection cache + per-host keep-alive sockets,
* status-aware retry (:class:`_TieredRetry`) + redirect handling,
* the prepare → cache-lookup → wire-send → cache-writeback pipeline,
* :meth:`send_many` with Spark fan-out,
* verb sugar and cookie-header coercion.

Wire calls go straight to :mod:`http.client` — no intermediate transport
class wraps the send. The supporting types (:class:`Retry`,
:class:`Timeout`, :class:`HTTPResponse`, :class:`HTTPHeaderDict`,
:mod:`exceptions`) live in the small side modules (:mod:`yggdrasil.http_.retry`,
:mod:`yggdrasil.http_.timeout`, :mod:`yggdrasil.http_.exceptions`,
:mod:`yggdrasil.http_.headers`); feature code
should not import them directly.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime as dt
import http.client
import itertools
import logging
import os
import pickle
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import takewhile
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
)
from urllib.parse import urlsplit, urlunsplit

import pyarrow as pa

from yggdrasil.arrow.cast import rechunk_arrow_batches
from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.dataclasses.waiting import (
    DEFAULT_WAITING_CONFIG,
    WaitingConfig,
    WaitingConfigArg,
)
from yggdrasil.data.enums import MediaTypes, Mode
from yggdrasil.io.authorization.base import Authorization
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.headers import Headers
from yggdrasil.io.memory import Memory
from yggdrasil.io.path import Path
from yggdrasil.io.primitive import ArrowIPCFile
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, Response, RESPONSE_SCHEMA
from yggdrasil.http_.response_batch import HTTPResponseBatch
from yggdrasil.io.send_config import CacheConfig, DEFAULT_MAX_BATCH_TTL, SendConfig, _request_column_sql_name
from yggdrasil.io.session import Session
from yggdrasil.io.url import URL

from .exceptions import (
    LocationParseError,
    LocationValueError,
    MaxRetryError,
    NewConnectionError,
    ReadTimeoutError,
    SSLError,
)
from .response import HTTPResponse
from .retry import Retry
from .timeout import _resolve_timeout

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from yggdrasil.io.tabular import Tabular

__all__ = ["HTTPSession"]


LOGGER = logging.getLogger(__name__)


# Cap on per-batch byte size when emitting responses from a Spark
# `mapInArrow` worker. 128 MiB matches Spark's default Arrow batch
# preference and keeps a single oversized response from inflating the
# whole partition's output. A response that is itself larger than the
# cap is sliced row-wise by the shared rechunker, which never splits a
# single row across batches.
_SPARK_RESPONSE_BATCH_BYTE_LIMIT: int = 128 * 1024 * 1024

# Rechunk byte target for paginated responses assembled by
# ``_combine_paginated_pages``.  Keeps the IPC file's record batches
# at a predictable size instead of flushing the whole concatenation as
# a single oversized batch.
_PAGINATED_RECHUNK_BYTE_SIZE: int = 128 * 1024 * 1024


# Local cache is a partitioned tabular tree backed by
# :class:`yggdrasil.io.nested.folder_path.FolderPath`:
# ``<root>/partition_key=<int>/part-{epoch_ms}-{seed}.<ext>``.
# Same Hive-style partition shape the remote :class:`Tabular` cache
# uses, so the same lookup primitives — :meth:`CacheConfig.make_lookup_predicate`
# / :meth:`CacheConfig.make_batch_lookup_predicate` — prune both
# backends identically. The predicate's ``partition_key IN (...)``
# clause flows through :meth:`FolderPath.iter_children`'s candidate
# probe, so a batch lookup ``stat``s only the partition directories
# its requests touch instead of walking the whole tree.



def _synthetic_not_found(request: PreparedRequest) -> Response:
    """Build a synthetic 404 response for a cache-only miss."""
    return Response(
        request=request,
        status_code=404,
        headers={"Content-Type": "application/json"},
        tags={"synthetic": "cache_only_miss"},
        buffer=b'{"error": "not found in cache"}',
        received_at=dt.datetime.now(dt.timezone.utc),
    )


def _insert_cache(
    tabular: Any,
    cache_cfg: CacheConfig,
    data: "pa.RecordBatch | pa.Table | SparkDataFrame",
    *,
    mode: "Mode | None" = None,
    spark_session: Optional["SparkSession"] = None,
    prune_values: "Mapping[str, Any] | None" = None,
    raise_error: bool = False,
) -> None:
    """Write *data* to any cache backend through the unified surface.

    Both local :class:`FolderPath` and remote
    :class:`~yggdrasil.databricks.table.Table` implement the
    :class:`Tabular` write protocol, so the Session never has to
    branch on which backend it's talking to — same call shape for
    the single-response store, the bulk backfill, the bulk
    persist, and the Spark persist.

    Dispatches on input type:

    * :class:`pa.RecordBatch` / :class:`pa.Table` →
      :meth:`Tabular.write_arrow_batches` (the Arrow-native path
      both backends implement).
    * :class:`pyspark.sql.DataFrame` →
      :meth:`Tabular.write_spark_frame` (Databricks Table routes
      this through its Spark-native MERGE / append plan; the local
      :class:`FolderPath` falls through to ``toArrow()`` then the
      Arrow path on the rare local-+-Spark mix).

    ``cache_cfg.match_by_columns`` rides through
    :attr:`CastOptions.match_by` so MERGE-mode writes dedup on the
    right keys. ``prune_values`` rides through
    :attr:`CastOptions.prune_values` and is **caller-supplied** —
    the remote MERGE turns it into a narrow-target predicate
    (partition pruning + IN-set narrowing); the local
    :class:`FolderPath` ignores it on writes. Default ``None``
    keeps the local hot path free of the dict-build the local
    backend wouldn't use anyway. Spark inserts also skip the
    ``prune_values`` knob — :meth:`_spark_persist_remote`
    pre-dedups via a ``left_anti`` join before reaching here.

    Errors are caught and logged by default — a failed cache write
    must not poison the request flow that just produced the
    response. ``raise_error=True`` flips that for the bulk-persist
    paths where the caller wants to surface failures.
    """
    if data is None:
        return
    from yggdrasil.data.options import CastOptions
    opts = CastOptions(
        mode=mode if mode is not None else cache_cfg.mode,
        match_by=cache_cfg.match_by_columns or None,
        wait=cache_cfg.wait,
        spark_session=spark_session,
        prune_values=prune_values,
    )
    try:
        # Spark frames go through ``write_spark_frame``; everything
        # else (RecordBatch, Table) through ``write_arrow_batches``.
        # The Spark detection is duck-typed via ``toArrow`` /
        # ``toPandas`` so we don't import pyspark at module load
        # for the common arrow-only path.
        if not isinstance(data, (pa.RecordBatch, pa.Table)) and (
            hasattr(data, "toArrow") or hasattr(data, "toPandas")
        ):
            tabular.write_spark_frame(data, options=opts)
            return
        if isinstance(data, pa.RecordBatch):
            if data.num_rows == 0:
                return
            batches: "Iterable[pa.RecordBatch]" = (data,)
        elif isinstance(data, pa.Table):
            if data.num_rows == 0:
                return
            batches = data.to_batches()
        else:
            # Unknown shape — let the backend decide / fail loudly.
            batches = data
        tabular.write_arrow_batches(batches, options=opts)
    except Exception as exc:
        if raise_error:
            raise
        LOGGER.debug(
            "Cache write failed for %r: %s", tabular, exc,
        )


def _cache_prune_values_for(batch: "pa.RecordBatch | pa.Table") -> "dict[str, Any]":
    """Return the MERGE narrow-target prune set for *batch*.

    ``{"partition_key": <column>, "public_hash": <column>}`` — both
    int64 so the IN-set literal stays compact. The columns are read
    straight off the batch (zero-copy reference to the data we're
    about to insert), so the caller doesn't materialise anything new
    — they're naming the columns the remote MERGE should narrow on.
    """
    return {
        "partition_key": batch["partition_key"],
        "public_hash":   batch["public_hash"],
    }


def _encode_request_data(
    data: Any,
) -> tuple[bytes, "str | None"]:
    """Encode a ``requests``-style ``data=`` payload into request bytes.

    * ``bytes`` / ``bytearray`` / ``memoryview`` → raw bytes, no header
      change (caller's ``Content-Type`` wins).
    * ``str`` → UTF-8 bytes, no header change.
    * ``Mapping`` / ``Iterable[tuple]`` → ``urlencode(doseq=True)`` with
      ``application/x-www-form-urlencoded`` as the suggested header,
      matching :meth:`requests.Session.post`.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data), None
    if isinstance(data, str):
        return data.encode("utf-8"), None
    from urllib.parse import urlencode

    return urlencode(data, doseq=True).encode("utf-8"), "application/x-www-form-urlencoded"


def _format_cookie_header(cookies: Any) -> str:
    """Serialize a ``requests``-style ``cookies=`` arg into one ``Cookie`` header value."""
    if hasattr(cookies, "to_header"):
        return cookies.to_header()
    if isinstance(cookies, Mapping):
        return "; ".join(f"{k}={v}" for k, v in cookies.items())
    raise TypeError(
        f"cookies must be a Mapping or expose to_header(); got "
        f"{type(cookies).__name__}."
    )


# ---------------------------------------------------------------------------
# Retry tuning — backoff schedule for HTTPSession's connection pool.
# ---------------------------------------------------------------------------

# 429s get a longer schedule than 5xx because rate limits need wall-clock
# time to clear; both schedules are tight (we'd rather surface an error
# fast than mask a real outage with a minute-long retry storm).
# Server-supplied Retry-After always wins over these when present.
_RETRY_TOTAL = 3
_RETRY_CONNECT = 2
_RETRY_READ = 2

# 5xx schedule: 0.5, 1, 2 (capped at backoff_max). Worst-case ~3.5s.
_BACKOFF_5XX_FACTOR = 0.5
_BACKOFF_5XX_MAX = 5.0

# 429 schedule: 1, 2, 4 (capped at backoff_max). Worst-case ~7s.
_BACKOFF_429_FACTOR = 1.0
_BACKOFF_429_MAX = 5.0

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class _TieredRetry(Retry):
    """:class:`Retry` variant with status-aware backoff.

    Standard ``Retry`` exposes a single ``backoff_factor`` shared by every
    retry, so 429 (rate limit) and 503 (transient outage) get the same
    schedule. This subclass branches on the most recent response status:

    * **429** uses a longer, gentler exponential schedule, since rate-limit
      windows are typically wall-clock bound and respond poorly to tight
      retries.
    * **Everything else** (5xx, transport errors) uses a shorter schedule.
    * The server's ``Retry-After`` header — when present and respected via
      ``respect_retry_after_header=True`` — always overrides this, because
      the pool checks ``get_retry_after`` before ``get_backoff_time``.
    """

    BACKOFF_MAX = _BACKOFF_429_MAX

    def get_backoff_time(self) -> float:  # type: ignore[override]
        # Mirror urllib3's own short-circuit: no backoff before the second
        # consecutive error. ``history`` is a tuple of RequestHistory entries.
        consecutive_errors = list(
            takewhile(lambda x: x.redirect_location is None, reversed(self.history))
        )
        if len(consecutive_errors) <= 1:
            return 0.0

        last_status = consecutive_errors[0].status

        if last_status == 429:
            # Count *consecutive* 429s only — if the last attempt was a 503,
            # we want the 5xx schedule, not a 429 schedule inflated by older
            # rate-limit hits.
            n = 0
            for h in consecutive_errors:
                if h.status == 429:
                    n += 1
                else:
                    break
            backoff = _BACKOFF_429_FACTOR * (2 ** (n - 1))
            return float(min(_BACKOFF_429_MAX, backoff))

        # Default 5xx / transport-error schedule, mirroring urllib3's formula
        # but with our own factor and cap.
        backoff = _BACKOFF_5XX_FACTOR * (2 ** (len(consecutive_errors) - 1))
        return float(min(_BACKOFF_5XX_MAX, backoff))


class HTTPSession(Session):
    """HTTP/HTTPS session — singleton-keyed by ``(base_url, verify, pool_maxsize, headers, waiting, auth)``.

    Inherits the singleton + pickle + ``job_pool`` plumbing from
    :class:`~yggdrasil.io.session.Session` and adds every HTTP-specific
    concern: a stdlib-backed connection pool (HTTPSession owns the per-host
    socket cache directly — no separate ``PoolManager`` indirection)
    connection pool, the prepare → cache-lookup → wire-send → cache-writeback
    pipeline (both single-request :meth:`send` and bulk :meth:`send_many`),
    Spark fan-out via ``mapInArrow``, the verb sugar
    (:meth:`get` / :meth:`post` / :meth:`put` / :meth:`patch` /
    :meth:`delete` / :meth:`head` / :meth:`options` / :meth:`request`),
    cookie-header coercion, and the 403 auth-refresh retry loop.

    No User-Agent generator, cookie jar, or browser-emulation layering is
    built in — per-vendor integrations subclass this for their own auth /
    pagination / rate-limit policy (see :class:`yggdrasil.fxrate.FxRate` for
    a worked example).
    """

    _PREPARED_CLASS: ClassVar[type] = PreparedRequest
    _RESPONSE_CLASS: ClassVar[type] = Response
    _BATCH_CLASS: ClassVar[type] = HTTPResponseBatch

    _TRANSIENT_STATE_ATTRS = Session._TRANSIENT_STATE_ATTRS | {"_connections", "_retry"}

    # Status codes that trigger an automatic redirect when ``redirect=True``.
    # 303 always falls back to GET (per RFC 7231); 307/308 preserve method.
    _REDIRECT_STATUSES: ClassVar[frozenset[int]] = frozenset({301, 302, 303, 307, 308})
    _MAX_REDIRECTS: ClassVar[int] = 10

    def __init__(
        self,
        base_url: Optional[URL | str] = None,
        verify: bool = True,
        pool_maxsize: int = 10,
        headers: "Headers | Mapping[str, str] | None" = None,
        waiting: WaitingConfig = DEFAULT_WAITING_CONFIG,
        *,
        auth: Optional[Authorization] = None,
    ) -> None:
        # Singleton-cached instances are re-entered on every constructor call
        # (Python always invokes ``__init__`` after ``__new__``); skip the
        # second pass so we don't drop the live connection pool / cookies.
        if getattr(self, "_initialized", False):
            return
        if auth is not None and not isinstance(auth, Authorization):
            raise TypeError(
                f"auth must be an Authorization instance or None; got "
                f"{type(auth).__name__}."
            )
        # The pool caps idle sockets per host; 8 is plenty for our typical
        # workloads. Clamping here means the singleton key (built from
        # ``pool_maxsize``) collapses ``HTTPSession(pool_maxsize=20)`` and
        # ``HTTPSession()`` to one instance the way they always did.
        pool_maxsize = min(8, int(pool_maxsize)) if pool_maxsize else 8
        self.base_url = URL.from_(base_url) if base_url else None
        self.verify = verify
        self.headers: Headers = Headers.from_(headers)
        self.waiting = waiting
        self.auth: Authorization | None = auth

        # Singleton-key probe path bails here — :class:`Session._singleton_key`
        # reads ``probe.__dict__`` and keeps only the keys whose names
        # appear in ``__init__``'s parameter list (``base_url`` / ``verify``
        # / ``pool_maxsize`` / ``headers`` / ``waiting`` / ``auth``), so
        # the lock + connection cache + retry policy + ``_job_pool`` build
        # below contribute nothing to identity. Skipping them on the probe
        # halves the singleton-hit cost (the bench was paying for an RLock
        # alloc + a ``_TieredRetry()`` per ``HTTPSession(base_url=…)`` call).
        if getattr(self, "_in_probe", False):
            self.pool_maxsize = pool_maxsize
            return

        super().__init__(pool_maxsize=pool_maxsize)
        # When a session-wide auth handler is bound at construction
        # time, pre-stamp ``self.headers["Authorization"]`` so anyone
        # inspecting the session sees the current credential without
        # going through a request first. :meth:`refresh_auth` keeps
        # the session header in sync on subsequent refreshes.
        if auth is not None:
            self.headers["Authorization"] = auth.authorization
        # Per-host idle-connection cache keyed by ``(scheme, host, port)``.
        # Sockets are recycled across requests so warm calls reuse the
        # existing TCP / TLS handshake instead of paying for a new one;
        # capped at ``pool_maxsize`` entries per host.
        self._connections: dict[tuple[str, str, int], "collections.deque[http.client.HTTPConnection]"] = {}
        # Retry policy is built once at init — same policy applies to
        # every wire send and the singleton key already pins
        # ``pool_maxsize`` / ``waiting`` / ``verify`` so two sessions
        # with the same identity share the same policy by construction.
        self._retry: Retry = self._build_retry()

    def __getnewargs_ex__(self):
        # Promote ``base_url`` to the positional arg slot so the receiver
        # can reproduce the canonical HTTPSession signature even when the
        # subclass introduces extra named knobs. Everything else rides as
        # kwargs through the inherited identity machinery.
        param_names = self._init_param_names()
        excluded = self._identity_excluded_attrs()
        state = {
            k: v for k, v in self.__dict__.items()
            if k in param_names and k not in excluded and k != "base_url"
        }
        return (self.base_url,), state

    def __setstate__(self, state):
        # Defer to :meth:`Session.__setstate__` for the live-singleton
        # short-circuit + lock rebuild; then promote ``_connections`` and
        # ``_retry`` from ``None`` (the transient default) to a usable
        # value. ``_connections`` starts empty — the receiver opens its
        # own sockets — and ``_retry`` is rebuilt fresh from the same
        # policy the sender used (it's stateless config-shaped).
        if getattr(self, "_initialized", False):
            return
        super().__setstate__(state)
        self._connections = {}
        self._retry = self._build_retry()

    # ------------------------------------------------------------------
    # Retry policy + connection cache
    # ------------------------------------------------------------------

    def _build_retry(self) -> Retry:
        """Build the :class:`Retry` policy applied to every wire send.

        Subclasses can override to swap the policy entirely, or call
        ``super()._build_retry().new(...)`` to tweak a single field.
        """
        return _TieredRetry(
            total=_RETRY_TOTAL,
            connect=_RETRY_CONNECT,
            read=_RETRY_READ,
            status=_RETRY_TOTAL,
            other=2,
            status_forcelist=_RETRY_STATUSES,
            allowed_methods=None,  # retry every method, incl. POST/PATCH
            respect_retry_after_header=True,
            raise_on_status=False,
            raise_on_redirect=False,
            # backoff_factor/backoff_max are unused — _TieredRetry overrides
            # get_backoff_time entirely — but we set sane defaults so any
            # fallback path (e.g. .new() that drops back to base behavior) is
            # still well-behaved.
            backoff_factor=_BACKOFF_5XX_FACTOR,
            backoff_max=_BACKOFF_429_MAX,
        )

    def _build_connection(
        self,
        scheme: str,
        host: str,
        port: int,
        connect_timeout: Optional[float],
    ) -> http.client.HTTPConnection:
        """Open a fresh :class:`http.client.HTTPConnection` to *(scheme, host, port)*.

        Honours ``self.verify``: when False the HTTPS context turns off
        certificate verification and hostname checking, matching the
        ``cert_reqs="CERT_NONE"`` shape urllib3 callers rely on (Databricks
        external links, some private-link deployments).
        """
        if scheme == "https":
            if self.verify:
                ssl_ctx: ssl.SSLContext = ssl.create_default_context()
            else:
                ssl_ctx = ssl._create_unverified_context()  # type: ignore[attr-defined]
                ssl_ctx.check_hostname = False
            return http.client.HTTPSConnection(
                host, port=port, timeout=connect_timeout, context=ssl_ctx,
            )
        return http.client.HTTPConnection(host, port=port, timeout=connect_timeout)

    def _get_connection(
        self,
        scheme: str,
        host: str,
        port: int,
        connect_timeout: Optional[float],
    ) -> http.client.HTTPConnection:
        """Pop an idle connection for *(scheme, host, port)* or build one."""
        key = (scheme, host, port)
        with self._lock:
            cached = self._connections.get(key)
            if cached:
                return cached.popleft()
        return self._build_connection(scheme, host, port, connect_timeout)

    def _release_connection(
        self,
        key: tuple[str, str, int],
        conn: http.client.HTTPConnection,
    ) -> None:
        """Return ``conn`` to the per-host idle cache or close it.

        Called by :meth:`HTTPResponse.release_conn` after a response is
        fully drained. Connections beyond ``pool_maxsize`` get closed
        instead of cached so a runaway caller can't leak sockets.
        """
        with self._lock:
            cached = self._connections.setdefault(key, collections.deque())
            if len(cached) < self.pool_maxsize:
                cached.append(conn)
                return
        try:
            conn.close()
        except Exception:
            pass

    def clear_connections(self) -> None:
        """Close every cached idle connection.

        Lifecycle convenience — closes the per-host sockets the session
        accumulated. Not called automatically; explicit cleanup is the
        caller's responsibility (or rely on process exit).
        """
        with self._lock:
            cached, self._connections = self._connections, {}
        for queue in cached.values():
            while queue:
                try:
                    queue.popleft().close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Direct HTTP send — no PoolManager indirection
    # ------------------------------------------------------------------

    def fetch(
        self,
        method: str,
        url: URL | str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        body: Any = None,
        timeout: Any = None,
        preload_content: bool = False,
        decode_content: bool = True,
        redirect: bool = True,
        tags: Optional[Mapping[str, str]] = None,
    ) -> HTTPResponse:
        """Low-level wire fetch — bypasses the cache / auth-refresh pipeline.

        Build a synthetic :class:`PreparedRequest` from ``method`` /
        ``url`` / ``headers`` / ``body`` and run it through the same
        retry + redirect machinery :meth:`_send_http` does for the
        regular :meth:`send` path. Useful when the caller just wants a
        raw byte stream off a URL — Databricks external-link readers
        feeding :func:`pa.input_stream` are the canonical case.
        """
        request = PreparedRequest.prepare(
            method=method,
            url=str(url) if isinstance(url, URL) else url,
            headers=dict(headers) if headers is not None else None,
            body=body,
        )
        return self._send_http(
            request,
            timeout=timeout,
            preload_content=preload_content,
            decode_content=decode_content,
            redirect=redirect,
            tags=tags,
        )

    def _send_http(
        self,
        request: PreparedRequest,
        *,
        timeout: Any = None,
        preload_content: bool = True,
        decode_content: bool = True,
        redirect: bool = True,
        tags: Optional[Mapping[str, str]] = None,
    ) -> HTTPResponse:
        """Drive one full HTTP request → response, retries + redirects included.

        :class:`HTTPSession` IS the connection pool — each retry attempt
        acquires a socket from :meth:`_get_connection`, fires
        ``conn.request`` + ``conn.getresponse`` once, and (on success)
        wraps the raw response in a high-level :class:`HTTPResponse`
        whose :meth:`release_conn` routes straight back into
        :meth:`_release_connection`. No intermediate transport class.
        """
        retries: Retry = self._retry.new()  # fresh history per call
        current_url = request.url.to_string()
        current_method = request.method
        current_body = request.buffer.to_bytes() if request.buffer is not None else None
        current_headers = dict(request.headers or {})
        current_request = request
        visited_redirects = 0

        while True:
            try:
                response = self._send_once(
                    request=current_request,
                    method=current_method,
                    url=current_url,
                    body=current_body,
                    headers=current_headers,
                    timeout=timeout,
                    preload_content=preload_content,
                    decode_content=decode_content,
                    tags=tags,
                )
            except (socket.timeout, TimeoutError) as exc:
                wrapped: Exception = ReadTimeoutError(self, current_url, str(exc))
                retries = retries.increment(
                    method=current_method, url=current_url, error=wrapped, _pool=self,
                )
                retries.sleep()
                continue
            except ssl.SSLError as exc:
                raise SSLError(str(exc)) from exc
            except (OSError, http.client.HTTPException) as exc:
                wrapped = NewConnectionError(self, str(exc))
                retries = retries.increment(
                    method=current_method, url=current_url, error=wrapped, _pool=self,
                )
                retries.sleep()
                continue

            # Redirect handling — drains the body, releases the socket,
            # rewrites method/body for 301/302/303 per RFC 7231.
            if redirect and response.status in self._REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if location and visited_redirects < self._MAX_REDIRECTS:
                    response.drain_conn()
                    response.release_conn()
                    visited_redirects += 1
                    current_url = self._resolve_redirect(current_url, location)
                    if response.status in (301, 302, 303) and current_method.upper() != "HEAD":
                        current_method = "GET"
                        current_body = None
                        current_headers.pop("Content-Length", None)
                        current_headers.pop("Content-Type", None)
                    # Rebuild a synthetic request for the redirected hop.
                    current_request = current_request.copy(url=current_url)
                    continue

            # Retry on status_forcelist (5xx / 429 by default).
            if retries.is_retry(
                current_method,
                response.status,
                response.headers.get("Retry-After") is not None,
            ):
                try:
                    next_retries = retries.increment(
                        method=current_method, url=current_url, response=response, _pool=self,
                    )
                except MaxRetryError:
                    if retries.raise_on_status:
                        raise
                    return response
                response.drain_conn()
                response.release_conn()
                next_retries.sleep(response=response)
                retries = next_retries
                continue

            return response

    def _send_once(
        self,
        *,
        request: PreparedRequest,
        method: str,
        url: str,
        body: Any,
        headers: Mapping[str, str],
        timeout: Any,
        preload_content: bool,
        decode_content: bool,
        tags: Optional[Mapping[str, str]] = None,
    ) -> HTTPResponse:
        """Single wire send — one connection, one ``conn.getresponse``."""
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            raise LocationParseError(url)
        scheme = parts.scheme.lower()
        if scheme not in ("http", "https"):
            raise LocationValueError(f"Unsupported scheme: {scheme!r}")
        host = parts.hostname or ""
        port = parts.port or (443 if scheme == "https" else 80)
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"

        connect_timeout, read_timeout = _resolve_timeout(timeout)
        key = (scheme, host, port)
        conn = self._get_connection(scheme, host, port, connect_timeout)
        try:
            if read_timeout is not None:
                conn.timeout = read_timeout
            send_headers = {k: str(v) for k, v in headers.items()}
            send_headers.setdefault(
                "Host", f"{host}:{port}" if port not in (80, 443) else host,
            )
            if (
                body is not None
                and "Content-Length" not in send_headers
                and isinstance(body, (bytes, bytearray))
            ):
                send_headers["Content-Length"] = str(len(body))
            conn.request(method, path, body=body, headers=send_headers)
            raw = conn.getresponse()
        except socket.timeout as exc:
            try:
                conn.close()
            except Exception:
                pass
            raise ReadTimeoutError(self, url, str(exc)) from exc
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

        # Build the high-level :class:`HTTPResponse` directly off the
        # raw socket. ``release_conn`` on the returned response calls
        # back into ``self._release_connection`` so the keep-alive
        # socket goes back to the per-host cache without HTTPSession
        # having to thread the connection through every caller.
        return HTTPResponse.from_wire(
            request=request,
            raw=raw,
            session=self,
            connection=conn,
            pool_key=key,
            decode_content=decode_content,
            preload_content=preload_content,
            tags=tags,
        )

    @staticmethod
    def _resolve_redirect(current_url: str, location: str) -> str:
        """Resolve a redirect ``Location`` against the current URL."""
        if "://" in location:
            return location
        parts = urlsplit(current_url)
        if location.startswith("/"):
            return urlunsplit((parts.scheme, parts.netloc, location, "", ""))
        # Relative path — drop the trailing segment of the current path.
        base = parts.path.rsplit("/", 1)[0] + "/"
        return urlunsplit((parts.scheme, parts.netloc, base + location, "", ""))

    @property
    def x_api_key(self) -> Optional[str]:
        if self.headers:
            return self.headers.get("X-API-Key")
        return None

    @x_api_key.setter
    def x_api_key(self, value: Optional[str]) -> None:
        if self.headers is None:
            self.headers = Headers()
        if value is None:
            self.headers.pop("X-API-Key", None)
        else:
            self.headers["X-API-Key"] = value

    def _request_log_id(self, request: PreparedRequest) -> str:
        try:
            return request.xxh3_b64(url_safe=True)
        except Exception:
            return request.url.to_string()

    def _cache_tabular_for_source(
        self,
        cache_cfg: CacheConfig,
        source: str,
    ) -> Any:
        """Resolve the :class:`Tabular` backend for a cache *source*.

        ``"local"`` materialises the disk-backed :class:`FolderPath`
        via :meth:`CacheConfig.cache_tabular` (memoised so repeat
        calls hand back the same instance); ``"remote"`` reads
        :attr:`CacheConfig.tabular` directly, which the
        config-builder has already populated. Returns ``None`` when
        the requested backend isn't configured — both the loader
        and the storer treat that as "cache miss, do nothing".
        """
        if source == "local":
            return cache_cfg.cache_tabular(session=self)
        return cache_cfg.tabular

    def _read_cache_batches(
        self,
        tabular: Any,
        options: Any,
    ) -> list:
        """Read with optional ``TABLE_OR_VIEW_NOT_FOUND`` recovery.

        :class:`FolderPath` yields an empty stream when the cache
        folder hasn't been written to yet (``Path.iterdir``'s
        missing-dir → empty contract). Remote Databricks tables
        raise ``[TABLE_OR_VIEW_NOT_FOUND]`` instead — catch that one
        case and lazily create the table before retrying. Any other
        backend exception propagates.
        """
        try:
            return list(tabular.read_arrow_batches(options=options))
        except Exception as exc:
            if "TABLE_OR_VIEW_NOT_FOUND" in str(exc) and hasattr(tabular, "create"):
                tabular.create(RESPONSE_ARROW_SCHEMA, missing_ok=True)
                return list(tabular.read_arrow_batches(options=options))
            raise

    def _load_cached_response(
        self,
        request: PreparedRequest,
        cache_cfg: CacheConfig,
        *,
        source: str,
    ) -> Optional[Response]:
        """Resolve one request against a cache backend (local or remote).

        Same call shape for both sides: build
        :meth:`CacheConfig.make_lookup_predicate`, push it through
        :meth:`Tabular.read_arrow_batches`, client-side dedup by
        ``received_at`` (APPEND-mode caches can hold multiple rows
        per identity), filter on
        :meth:`CacheConfig.filter_response`, and stamp the matching
        ``local_cached`` / ``remote_cached`` flag for downstream
        provenance.

        Skips the per-request :meth:`PreparedRequest.anonymize` pass
        when ``cache_cfg.request_by_is_public`` holds — the
        predicate and the row's match keys collapse to the same
        ``public_*`` columns either way.
        """
        if source == "remote" and not cache_cfg.remote_cache_enabled:
            return None
        tabular = self._cache_tabular_for_source(cache_cfg, source)
        if tabular is None:
            return None

        from yggdrasil.data.options import CastOptions

        # Local cache stores responses as-is (the writer never
        # anonymises before persist), so the lookup tuple matches
        # the row's original ``request_*`` columns straight from
        # the request — no per-call URL parse + header normalise.
        # Remote stores remove user-info too, but only paths keyed
        # on private ``request_by`` columns need to anonymise the
        # lookup; ``request_by_is_public`` collapses to the same
        # ``public_*`` hash on both projections, so anonymisation
        # is a no-op there either way.
        lookup_request = (
            request
            if source == "local" or cache_cfg.request_by_is_public
            else request.anonymize(mode=cache_cfg.anonymize)
        )
        predicate = cache_cfg.make_lookup_predicate(request=lookup_request)
        opts = CastOptions(predicate=predicate, spark_session=request.send_config_or_default.spark_session)
        batches = self._read_cache_batches(tabular, opts)

        best: Optional[Response] = None
        for response in Response.from_arrow_tabular(iter(batches)):
            if not cache_cfg.filter_response(response, request=request):
                continue
            if best is None or response.received_at >= best.received_at:
                best = response
        if best is not None:
            LOGGER.info(
                "Cache hit %s %s %s in %r (status=%d, received_at=%s) "
                "— skipping network",
                source, request.method, request.url, tabular,
                best.status_code, best.received_at,
            )
            best.local_cached = (source == "local")
            best.remote_cached = (source == "remote")
        return best

    def _store_cached_response(
        self,
        response: Response,
        cache_cfg: CacheConfig,
        *,
        source: str,
        tabular: Any = None,
        mode: Optional[Mode] = None,
        async_write: "bool | None" = None,
    ) -> None:
        """Persist one response to a cache backend (local or remote).

        Both backends go through :func:`_insert_cache` — the
        canonical :meth:`Tabular.write_arrow_batches` adapter that
        also carries ``prune_values`` for the MERGE narrow-target
        path. The Session never branches on backend type for the
        write itself.

        ``async_write`` controls the dispatch policy: ``True`` queues
        the write through the :class:`Job` fire-and-forget pool so
        the caller doesn't block on disk / network IO; ``False`` runs
        inline. Default ``None`` picks per-source — local cache
        writes are fire-and-forget (the response buffer is still
        live, so we drain it inline but ship the actual write off
        the request hot path), remote writes run synchronously (the
        bulk persist path :meth:`_persist_remote` already
        parallelises across write groups).
        """
        if not response.ok:
            return
        if source == "local" and response.request is None:
            return

        tabular = (
            tabular if tabular is not None
            else self._cache_tabular_for_source(cache_cfg, source)
        )
        if tabular is None:
            return
        batch = response.to_arrow_batch(parse=False)
        # Prune values matter for the remote MERGE narrow-target
        # path; the local FolderPath ignores them on writes (it
        # splits by partition automatically from the batch's
        # ``t:partition_by`` schema metadata). Skip the build for
        # local so the cache hot path doesn't pay the dict-build
        # the local backend wouldn't use.
        prune_values = (
            _cache_prune_values_for(batch) if source == "remote" else None
        )
        if async_write is None:
            async_write = (source == "local")
        req = response.request
        spark = req.send_config_or_default.spark_session if req is not None else None
        if async_write:
            Job.make(
                _insert_cache,
                tabular,
                cache_cfg,
                batch,
                mode=mode,
                spark_session=spark,
                prune_values=prune_values,
            ).fire_and_forget()
        else:
            _insert_cache(
                tabular,
                cache_cfg,
                batch,
                mode=mode,
                spark_session=spark,
                prune_values=prune_values,
            )

    @classmethod
    def from_url(
        cls,
        url: URL | str,
        *,
        verify: bool = True,
        normalize: bool = True,
        waiting: WaitingConfigArg = True,
    ) -> "Session":
        parsed = URL.from_(url, normalize=normalize)

        if parsed.scheme.startswith("http"):
            from yggdrasil.http_ import HTTPSession

            return HTTPSession(
                base_url=parsed,
                verify=verify,
                waiting=WaitingConfig.from_(waiting) if waiting is not None else None,
            )

        raise ValueError(f"Cannot build session from scheme: {parsed.scheme!r}")

    def send(
        self,
        request: PreparedRequest,
        config: SendConfig | Mapping[str, Any] | None = None,
        *,
        wait: WaitingConfigArg = ...,
        raise_error: bool = ...,
        stream: bool = ...,
        remote_cache: CacheConfig | Mapping[str, Any] | None = ...,
        local_cache: CacheConfig | Mapping[str, Any] | None = ...,
        cache_only: bool = ...,
        spark_session: Optional["SparkSession"] = ...,
        start: bool = True,
        **options,
    ) -> Response:
        """Prepare, dispatch, and (optionally) await the response.

        Single-request entry point — always returns a :class:`Response`.
        :class:`SendConfig.as_tabular` is accepted for API symmetry with
        :meth:`send_many` but ignored here: a one-row tabular is what
        :meth:`send_many` already produces, so wrapping a single send
        adds no value.

        When ``spark_session`` is bound (or ``True`` / ``...`` resolved
        via :meth:`PyEnv.spark_session`), the wire send is fanned out
        to an executor through the same ``mapInArrow`` path
        :meth:`send_many` uses — the driver does not cross the wire.
        Otherwise the call runs synchronously on the local pool.

        ``start=True`` (default) fires the wire call. ``start=False``
        builds the prepared request + response shell without crossing
        the wire — the :class:`StatementExecutor` override uses the
        same knob to return an idled :class:`StatementResult` whose
        backend submission is deferred until
        :meth:`StatementResult.start` fires. Plain HTTP sessions don't
        need an idle :class:`Response` (the network call is
        synchronous), so the base raises a clean
        ``NotImplementedError`` via :meth:`_build_idle_response`.

        Per-request :attr:`PreparedRequest.send_config` is used as the
        base when no explicit *config* is passed — explicit kwargs
        still override individual fields.
        """
        overrides: dict[str, Any] = {**options}
        if wait is not ...:
            overrides["wait"] = wait
        if raise_error is not ...:
            overrides["raise_error"] = raise_error
        if stream is not ...:
            overrides["stream"] = stream
        if remote_cache is not ...:
            overrides["remote_cache"] = remote_cache
        if local_cache is not ...:
            overrides["local_cache"] = local_cache
        if cache_only is not ...:
            overrides["cache_only"] = cache_only
        if spark_session is not ...:
            overrides["spark_session"] = spark_session
        base = config if config is not None else request.send_config
        cfg = SendConfig.from_(base, **overrides)
        # Per-request cache configs always win over session-level
        # fallbacks — stamp the request's own caches back on top.
        req_sc = request.send_config
        if req_sc is not None:
            merge_back: dict[str, Any] = {}
            if req_sc.local_cache is not None:
                merge_back["local_cache"] = req_sc.local_cache
            if req_sc.remote_cache is not None:
                merge_back["remote_cache"] = req_sc.remote_cache
            if merge_back:
                cfg = dataclasses.replace(cfg, **merge_back)
        request.send_config = cfg
        if not start:
            return self._build_idle_response(request, cfg)
        if cfg.spark_session is not None:
            for response in self._send_many(iter([request]), cfg):
                return response
        return self._send(request)

    def _build_idle_response(
        self,
        request: PreparedRequest,
        config: SendConfig,
    ) -> Response:
        """Return a not-yet-sent response shell for *request*.

        Concrete HTTP sessions don't need an idle :class:`Response`
        (the network call is synchronous), so the default raises.
        :class:`StatementExecutor` overrides this to return the
        standard ``start=False`` :class:`StatementResult` so callers
        get a uniform "build now, dispatch later" knob across HTTP
        and SQL.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.send(start=False) is not implemented; "
            "use the synchronous send path (start=True) for HTTP sessions, "
            "or override _build_idle_response on a custom subclass."
        )

    def refresh_auth(
        self,
        request: PreparedRequest,
        force: bool = True,
    ) -> "tuple[Session, bool]":
        """Resolve the auth handler and stamp the Authorization header.

        Per-request ``request.auth`` wins over the session-wide
        ``self.auth``. When a handler is bound, ``refresh_auth`` calls
        its ``refresh(force=force)`` if it exposes one — MSAL-style
        handlers use ``force`` to bypass the in-memory token cache
        and mint a fresh credential — then reads its ``authorization``
        property, writes the value to ``request.headers["Authorization"]``,
        and (when the resolved handler is the session-wide one) keeps
        ``self.headers["Authorization"]`` in sync so the session-level
        view stays current too.

        Returns ``(self, refreshed)`` where ``refreshed`` is ``True``
        when a handler ran and the header was stamped, ``False`` when
        the silent no-op branch was taken (force=False + no handler).
        Returning ``self`` lets the caller chain
        (``session.refresh_auth(req)[0].send(req)``); the bool surfaces
        whether the request now carries an Authorization header so a
        retry loop can short-circuit when nothing changed.

        When **no** handler is bound:

        - ``force=True`` (default) → raises
          :class:`~yggdrasil.exceptions.AuthRequiredError`. The caller
          explicitly asked to force-refresh credentials and there is
          nothing to refresh — failing fast catches misconfigured
          integrations at the right line instead of letting an
          un-authenticated send go to the wire.
        - ``force=False`` → returns ``(self, False)``. This is the
          steady-state path used by
          :meth:`prepare_request_before_send`: a request to a public
          endpoint shouldn't fail just because the session doesn't
          have a token to refresh.

        Two regular call sites:

        - :meth:`prepare_request_before_send` calls this with
          ``force=False`` so steady-state requests reuse the cached
          token (the handler's own ``is_expired`` / refresh-skew
          logic still mints a new one when the cache is empty or
          stale).
        - The HTTP send path calls this with the default ``force=True``
          after a 403, to mint a fresh token before the single retry —
          some vendors (Salesforce, M365, …) return 403 instead of
          401 when a previously-valid token has been silently rotated
          upstream.

        Subclasses with vendor-specific auth (HMAC signing, SigV4,
        challenge-response) override this to do whatever their API
        needs while keeping the same contract.
        """
        handler = request.auth or self.auth
        if handler is None:
            if force:
                from yggdrasil.exceptions import AuthRequiredError
                raise AuthRequiredError(
                    f"refresh_auth(force=True) requested but no Authorization "
                    f"handler is bound to the request or to {type(self).__name__}. "
                    "Bind one via Session(auth=handler), request.auth=handler, "
                    "or call refresh_auth(request, force=False) if a missing "
                    "handler should be tolerated.",
                    request=request,
                )
            return self, False
        refresh = getattr(handler, "refresh", None)
        if callable(refresh):
            try:
                refresh(force=force)
            except TypeError:
                # Handler's refresh() doesn't accept ``force`` — fall
                # back to a no-arg call so legacy handlers still work.
                refresh()
        authorization = handler.authorization
        if request.headers is None:
            request.headers = Headers()
        request.headers["Authorization"] = authorization
        # Mirror the refresh on the session-level header when the
        # session-wide handler is what we just ran — a per-request
        # override (request.auth) deliberately doesn't pollute the
        # session view.
        if handler is self.auth:
            self.headers["Authorization"] = authorization
        return self, True

    def prepare_request_before_send(self, request: PreparedRequest) -> PreparedRequest:
        """Session-wide request hook fired once per outbound request.

        Default returns *request* unchanged. Subclasses override to inject
        session-level concerns — auth, signing, correlation IDs, mandatory
        headers — that should apply to every request leaving this session.
        Runs in :meth:`_send` just before :meth:`_local_send`, so cache hits
        bypass it. Travels with the session into Spark workers via
        ``__getstate__`` / ``__setstate__``.
        """
        request.attach_session(self)
        request.sent_at = dt.datetime.now(dt.timezone.utc)
        if self.headers:
            if request.headers is None:
                request.headers = {}
            request.headers.update(self.headers)
        # Steady-state requests reuse the handler's cached token; the
        # 403 retry path in HTTPSession._local_send re-calls
        # ``refresh_auth`` with ``force=True`` to bypass that cache
        # when the upstream rotated the credential.
        self.refresh_auth(request, force=False)
        return request

    def prepare_response_after_received(self, response: Response) -> Response:
        """Session-wide response hook fired once per completed network send.

        Default returns *response* unchanged. Subclasses override to log,
        redact, enrich, or wrap responses returned from the wire. Runs in
        :meth:`_send` after :meth:`_local_send` and before cache writeback,
        so the persisted response reflects any post-processing. Cache hits
        bypass it. Travels with the session into Spark workers via
        ``__getstate__`` / ``__setstate__``.
        """
        return response

    def _send(
        self,
        request: PreparedRequest,
    ) -> Response:
        """Core send pipeline: local cache → remote cache → network → writeback.

        Reads the fully-resolved config from
        :attr:`PreparedRequest.send_config_or_default` — callers
        (``send``, ``_send_many``, ``_fetch_misses``) are responsible
        for stamping the effective :class:`SendConfig` on each request
        *before* this method is reached.
        """
        config = request.send_config_or_default
        local_cfg = config.local_cache
        remote_cfg = config.remote_cache

        local_cache_tabular: Any = None
        if local_cfg is not None and local_cfg.local_cache_enabled:
            local_cache_tabular = local_cfg.cache_tabular(session=self)
            if local_cfg.mode != Mode.UPSERT:
                local_response = self._load_cached_response(
                    request, local_cfg, source="local",
                )
                if local_response is not None:
                    if config.raise_error:
                        local_response.raise_for_status()
                    return local_response

        if (
            remote_cfg is not None
            and remote_cfg.remote_cache_enabled
            and remote_cfg.mode != Mode.UPSERT
        ):
            remote_response = self._load_cached_response(
                request, remote_cfg, source="remote",
            )
            if remote_response is not None:
                if local_cache_tabular is not None:
                    self._store_cached_response(
                        remote_response,
                        local_cfg,
                        source="local",
                        tabular=local_cache_tabular,
                    )
                if config.raise_error:
                    remote_response.raise_for_status()
                return remote_response

        if config.cache_only:
            response = _synthetic_not_found(request)
            if config.raise_error:
                response.raise_for_status()
            return response

        request = self.prepare_request_before_send(request)
        LOGGER.debug("Sending %s %s", request.method, request.url)
        response = self._local_send(request)
        response = self.prepare_response_after_received(response)
        LOGGER.info("Sent %s %s", request.method, request.url)

        if local_cache_tabular is not None:
            self._store_cached_response(
                response,
                local_cfg,
                source="local",
                tabular=local_cache_tabular,
            )

        if remote_cfg is not None and remote_cfg.remote_cache_enabled:
            self._store_cached_response(
                response, remote_cfg, source="remote",
            )

        if config.raise_error:
            response.raise_for_status()

        return response

    # ------------------------------------------------------------------
    # Wire transport
    # ------------------------------------------------------------------

    def _local_send(
        self,
        request: PreparedRequest,
    ) -> HTTPResponse:
        config = request.send_config_or_default
        wait_cfg = self.waiting if config.wait is None else config.wait

        result = self._wire_send(request, wait_cfg)

        # 403 → refresh auth and retry once. The pool's status_forcelist
        # covers 5xx / 429 transients; 403 is a deliberate auth signal
        # some vendors (Salesforce, M365 SharePoint, …) emit instead
        # of 401 when a previously-valid token has been silently
        # rotated upstream. Only worth retrying when an auth handler
        # is actually bound — otherwise the second attempt would
        # carry the same headers and 403 again.
        if result.status_code == 403 and (request.auth or self.auth) is not None:
            LOGGER.warning(
                "Refreshing auth after 403 for %s %s — retrying once",
                request.method, request.url,
            )
            _, refreshed = self.refresh_auth(request)  # force=True default
            if refreshed:
                result = self._wire_send(request, wait_cfg)

        x_current_page = result.headers.get("X-Current-Page")
        x_total_pages = result.headers.get("X-Last-Page")

        if x_current_page and x_total_pages:
            result = self._combine_paginated_pages(
                result=result,
                request=request,
                current_page=int(x_current_page),
                total_pages=int(x_total_pages),
                wait_cfg=wait_cfg,
                stream=config.stream,
                raise_error=config.raise_error,
            )

        if config.raise_error:
            result.raise_for_status()

        return result

    def _wire_send(
        self,
        request: PreparedRequest,
        wait_cfg: WaitingConfig,
    ) -> HTTPResponse:
        """Single wire-level send.

        :class:`HTTPSession` IS the pool now: :meth:`_send_http`
        returns the drained :class:`HTTPResponse` directly, so callers
        read ``X-Current-Page`` / ``X-Last-Page`` straight off
        ``response.headers`` without a parallel raw-response object.
        """
        result = self._send_http(
            request,
            timeout=wait_cfg.timeout_pool,
            preload_content=True,
            decode_content=False,
            redirect=True,
        )
        return result

    def _fetch_paginated_page(
        self,
        *,
        request: PreparedRequest,
        page_num: int,
        body_seed: bytes | None,
        wait_cfg: WaitingConfig,
        stream: bool,
        raise_error: bool,
    ) -> tuple[int, HTTPResponse]:
        page_url = request.url.add_param("page", str(page_num), replace=True)

        page_request = request.copy(
            url=page_url,
            buffer=Memory(binary=body_seed) if body_seed is not None else None,
        )

        page_result = self._send_http(
            page_request,
            timeout=wait_cfg.timeout_pool,
            preload_content=True,
            decode_content=False,
            redirect=True,
        )

        if raise_error:
            page_result.raise_for_status()

        return page_num, page_result

    def _combine_paginated_pages(
        self,
        *,
        result: HTTPResponse,
        request: PreparedRequest,
        current_page: int,
        total_pages: int,
        wait_cfg: WaitingConfig,
        stream: bool,
        raise_error: bool,
        pool: Optional[JobPoolExecutor | int] = None,
    ) -> HTTPResponse:
        if not isinstance(pool, JobPoolExecutor):
            with JobPoolExecutor.from_(pool) as parsed_pool:
                return self._combine_paginated_pages(
                    result=result,
                    request=request,
                    current_page=current_page,
                    total_pages=total_pages,
                    wait_cfg=wait_cfg,
                    stream=stream,
                    raise_error=raise_error,
                    pool=parsed_pool,
                )

        from yggdrasil.lazy_imports import polars as pl

        init_df = result.to_polars(parse=True, lazy=False)
        if total_pages <= current_page:
            return result

        remaining_pages = list(range(current_page + 1, total_pages + 1))
        body_seed = request.buffer.to_bytes() if request.buffer else None

        def jobs():
            for pn in remaining_pages:
                yield Job.make(
                    self._fetch_paginated_page,
                    request=request,
                    page_num=pn,
                    body_seed=body_seed,
                    wait_cfg=wait_cfg,
                    stream=stream,
                    raise_error=raise_error,
                )

        frames = [init_df]
        for job_result in pool.as_completed(
            jobs(),
            ordered=False,
            max_in_flight=len(remaining_pages),
            cancel_on_exit=False,
            shutdown_on_exit=False,
            raise_error=True,
        ):
            _, page_resp = job_result.result
            frames.append(page_resp.to_polars(parse=True, lazy=False))

        final_df = pl.concat(frames, how="diagonal_relaxed", rechunk=False)
        combined_table = final_df.to_arrow(compat_level=pl.CompatLevel.newest())

        new_holder = Memory()
        new_holder.media_type = MediaTypes.ARROW_IPC
        with ArrowIPCFile(holder=new_holder, owns_holder=False, mode="wb") as new_buffer:
            new_buffer.write_arrow_batches(
                rechunk_arrow_batches(
                    combined_table.to_batches(),
                    byte_size=_PAGINATED_RECHUNK_BYTE_SIZE,
                ),
                compression="zstd",
            )

        result.buffer.close()
        result.buffer = new_holder
        result.set_media_type(MediaTypes.ARROW_IPC)

        result.update_tags({
            "page_start": str(current_page),
            "page_total": str(total_pages),
        })

        return result

    def send_many(
        self,
        requests: Iterator[PreparedRequest],
        config: SendConfig | Mapping[str, Any] | None = None,
        *,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        cache_only: bool = False,
        as_tabular: bool = False,
        spark_session: Optional["SparkSession"] = None,
        batch_size: int | None = None,
        ordered: bool = False,
        max_in_flight: int | None = None,
        max_batch_ttl: float | None = None,
        **options,
    ) -> "Iterator[Response] | Tabular":
        """Stream responses for a batch of requests.

        Yields :class:`Response` objects one at a time by default.
        ``as_tabular=True`` concatenates into a single :class:`Tabular`.

        Batch orchestration kwargs (``batch_size``, ``ordered``,
        ``max_in_flight``, ``max_batch_ttl``) control chunking and
        concurrency; everything else is folded into a :class:`SendConfig`
        that gets stamped on each request.
        """
        cfg = SendConfig.from_(
            config,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
            cache_only=cache_only,
            as_tabular=as_tabular,
            spark_session=spark_session,
            **options,
        )
        batch_kw: dict[str, Any] = {}
        if batch_size is not None:
            batch_kw["batch_size"] = batch_size
        if ordered:
            batch_kw["ordered"] = ordered
        if max_in_flight is not None:
            batch_kw["max_in_flight"] = max_in_flight
        if max_batch_ttl is not None:
            batch_kw["max_batch_ttl"] = max_batch_ttl
        if cfg.as_tabular:
            return self._send_many_as_tabular(requests, cfg, **batch_kw)
        return self._send_many(requests, cfg, **batch_kw)

    def _send_many_as_tabular(
        self,
        requests: Iterator[PreparedRequest],
        config: SendConfig,
        **batch_kw: Any,
    ) -> "Tabular":
        """Drain :meth:`_send_many_batches` and concat into one :class:`Tabular`.

        Spark mode unions per-chunk :class:`SparkDataFrame` lazily via
        :meth:`HTTPResponseBatch.extend` then wraps the result through
        :meth:`HTTPResponseBatch.to_tabular`, so no executor job fires
        until the caller triggers an action. Python mode concatenates
        Arrow record batches at the end of the stream.
        """
        spark = config.spark_session
        accumulator: HTTPResponseBatch | None = None
        for batch in self._send_many_batches(requests, config, **batch_kw):
            if accumulator is None:
                accumulator = batch
            else:
                accumulator.extend(batch)
        if accumulator is None:
            if spark is not None:
                from yggdrasil.http_.response_batch import spark_to_tabular
                return spark_to_tabular(self._cached_empty_spark_frame(spark))
            from yggdrasil.io.tabular import ArrowTabular
            return ArrowTabular(
                RESPONSE_ARROW_SCHEMA.empty_table(),
                schema=RESPONSE_ARROW_SCHEMA,
            )
        return accumulator.to_tabular(spark)

    # ------------------------------------------------------------------ #
    # send_many — staged pipeline                                         #
    #                                                                    #
    # The flow per batch is:                                             #
    #   1. Local cache: yield hits, evict UPSERT entries, collect misses #
    #   2. Remote cache: group misses by effective table, run one SQL    #
    #      lookup per table, yield hits, collect misses                  #
    #   3. Network: fan out misses through the job pool                  #
    #   4. Bulk remote writeback: group successful responses by          #
    #      (table, mode, match_by, wait, anonymize) so per-request       #
    #      cache configs are honoured exactly                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _remote_write_group_key(cfg: CacheConfig) -> tuple:
        """Identity used to group responses for a single bulk remote insert.

        Two responses can share an insert iff every config dimension that
        affects the write call is identical: target table, mode, match-by
        columns, wait flag, and anonymize mode. Without all five, distinct
        per-request configs get silently collapsed onto whichever config
        landed in the bucket first.
        """
        return (
            cfg.tabular.full_name(safe=True),
            cfg.mode,
            tuple(cfg.match_by) if cfg.match_by else (),
            bool(cfg.wait),
            cfg.anonymize,
        )

    def _split_local_cache(
        self,
        batch: list[PreparedRequest],
    ) -> tuple["Tabular | None", list[PreparedRequest]]:
        """Stage 1: scan the local cache.

        Returns ``(hits_tabular, misses)``. Hits are wrapped in an
        :class:`ArrowTabular`; ``None`` when everything misses.
        UPSERT entries bypass the read entirely (always miss).
        """
        misses: list[PreparedRequest] = []

        if not any(r.local_cache_config for r in batch):
            return None, list(batch)

        path_to_cfg: dict[Path, CacheConfig] = {}
        path_to_reqs: dict[Path, list[PreparedRequest]] = {}

        for req in batch:
            eff = req.local_cache_config
            if eff is not None:
                eff = eff.prebuild(session=self)
            if eff is None or not eff.local_cache_enabled or eff.mode == Mode.UPSERT:
                misses.append(req)
                continue
            root = eff.local_cache_folder(session=self)
            bucket = path_to_reqs.get(root)
            if bucket is None:
                path_to_cfg[root] = eff
                path_to_reqs[root] = bucket = []
            bucket.append(req)

        all_hits: list[Response] = []
        for root, reqs in path_to_reqs.items():
            eff = path_to_cfg[root]
            r_hits, r_misses = self._lookup_cached(eff, reqs, source="local")
            all_hits.extend(r_hits)
            misses.extend(r_misses)

        if all_hits:
            LOGGER.info(
                "Batch local cache: %d/%d hit(s)",
                len(all_hits), len(batch),
            )
            from yggdrasil.http_.response_batch import responses_to_tabular
            return responses_to_tabular(all_hits), misses
        return None, misses

    def _split_remote_cache(
        self,
        requests: list[PreparedRequest],
    ) -> tuple[dict[str, list[Response]], list[PreparedRequest]]:
        """Stage 2: scan the remote cache.

        UPSERT requests bypass the read entirely (always misses, refetch).
        Non-UPSERT requests are grouped by their effective cache table so we
        execute exactly one batch SQL lookup per table.
        """
        hits: dict[str, list[Response]] = {}
        misses: list[PreparedRequest] = []
        table_to_cfg: dict[str, CacheConfig] = {}
        table_to_reqs: dict[str, list[PreparedRequest]] = {}

        for req in requests:
            t_cfg = req.remote_cache_config
            if t_cfg is None or t_cfg.mode == Mode.UPSERT:
                misses.append(req)
                continue
            if not t_cfg.remote_cache_enabled or t_cfg.mode != Mode.APPEND:
                misses.append(req)
                continue
            tkey = t_cfg.tabular.full_name(safe=True)
            bucket = table_to_reqs.get(tkey)
            if bucket is None:
                table_to_cfg[tkey] = t_cfg
                table_to_reqs[tkey] = bucket = []
            bucket.append(req)

        total_hits = 0
        for tkey, t_reqs in table_to_reqs.items():
            t_cfg = table_to_cfg[tkey]
            t_hits, t_misses = self._lookup_cached(
                t_cfg, t_reqs, source="remote",
            )
            if t_hits:
                hits[tkey] = t_hits
                total_hits += len(t_hits)
            misses.extend(t_misses)

        if total_hits:
            LOGGER.info(
                "Batch remote cache: %d/%d hit(s) across %d table(s) — "
                "skipping network for %d request(s); breakdown: %s",
                total_hits, len(requests), len(table_to_cfg), total_hits,
                ", ".join(f"{tkey}={len(v)}" for tkey, v in hits.items()),
            )
        return hits, misses

    def _lookup_cached(
        self,
        cfg: CacheConfig,
        requests: list[PreparedRequest],
        *,
        source: str,
    ) -> tuple[list[Response], list[PreparedRequest]]:
        """Batch-lookup *requests* against any cache backend.

        Same call shape for local :class:`FolderPath` and remote
        :class:`~yggdrasil.databricks.table.Table`: build
        :meth:`CacheConfig.make_batch_lookup_predicate`, push it
        through :meth:`Tabular.read_arrow_batches`, client-side
        dedup by ``received_at`` (APPEND-mode caches can hold
        multiple rows per identity), and stamp the matching
        ``local_cached`` / ``remote_cached`` flag per hit.

        Returns ``(hits, misses)`` paired with the input
        ``requests`` order — ``hits`` carry the cached response,
        ``misses`` carry the original :class:`PreparedRequest` for
        the next stage. Skips the per-request
        :meth:`PreparedRequest.anonymize` pass when
        ``cfg.request_by_is_public`` holds — ``public_*`` match
        keys collapse to the same value on the original and
        anonymised request, so the lookup tuple and the predicate
        match clause come out identical without paying for one URL
        parse + header normalise per request.
        """
        tabular = self._cache_tabular_for_source(cfg, source)
        if tabular is None:
            return [], list(requests)

        from yggdrasil.data.options import CastOptions

        # Same anonymise-or-skip rule as :meth:`_load_cached_response`:
        # local cache stores rows as-is so the lookup uses originals;
        # public ``request_by`` keys collapse on both projections so
        # anonymising is a no-op there. Only remote + private
        # ``request_by`` pays for the per-request rebuild.
        if source == "local" or cfg.request_by_is_public:
            lookup_batch: list[PreparedRequest] = list(requests)
        else:
            lookup_batch = [r.anonymize(mode=cfg.anonymize) for r in requests]

        predicate = cfg.make_batch_lookup_predicate(requests=lookup_batch)
        spark_session = requests[0].send_config_or_default.spark_session if requests else None
        opts = CastOptions(predicate=predicate, spark_session=spark_session)
        batches = self._read_cache_batches(tabular, opts)

        # Pre-compute the lookup tuple once per input request. The match
        # loop below would otherwise re-call ``cfg.request_tuple(lookup)``
        # for every input — at ~2 us / call that's 130 us / 64 reqs of
        # pure book-keeping on the batched cache hit path.
        request_tuple = cfg.request_tuple
        lookup_keys: list[tuple] = [request_tuple(r) for r in lookup_batch]

        # Client-side dedup: keep the latest ``received_at`` per
        # request-tuple. APPEND-mode caches (both backends) may hold
        # multiple rows per identity.
        result_map: dict[tuple, Response] = {}
        for response in Response.from_arrow_tabular(iter(batches)):
            request = response.request
            if request is None:
                continue
            key = request_tuple(request)
            existing = result_map.get(key)
            if existing is None or response.received_at >= existing.received_at:
                result_map[key] = response

        hits: list[Response] = []
        misses: list[PreparedRequest] = []
        is_local = (source == "local")
        filter_response = cfg.filter_response
        for req, lookup_key in zip(requests, lookup_keys):
            candidate = result_map.get(lookup_key)
            if candidate is not None and filter_response(candidate, request=req):
                candidate.local_cached = is_local
                candidate.remote_cached = not is_local
                hits.append(candidate)
            else:
                misses.append(req)
        return hits, misses

    # Per-SparkSession cache of the empty :class:`SparkDataFrame` keyed
    # to :data:`RESPONSE_SCHEMA`. The bare ``createDataFrame([],
    # schema=...)`` path costs ~30 ms per call (most of which is JVM
    # round-trip overhead on the empty payload), and the cache layer
    # hits it once per bucket pass + once per empty-table branch in
    # :meth:`_lookup_remote_table_spark`. Caching by ``id(spark)`` keeps
    # this safe across multiple concurrent SparkSessions in the same
    # process; ``WeakValueDictionary`` would be cleaner but Spark
    # DataFrames hold a strong reference to their session so the
    # entries don't outlive the session anyway.
    _EMPTY_SPARK_FRAMES: "ClassVar[dict[int, SparkDataFrame]]" = {}

    @classmethod
    def _cached_empty_spark_frame(
        cls, spark: "SparkSession",
    ) -> "SparkDataFrame":
        """Return a process-cached empty :class:`SparkDataFrame` with
        :data:`RESPONSE_SCHEMA`.

        One DataFrame per :class:`SparkSession` — same identity for
        repeat callers, so downstream ``unionByName`` paths can keep
        their plan cached. The cache key is ``id(spark)``: each
        SparkSession owns its own DataFrames anyway, and the entry
        is dropped when the session is.
        """
        key = id(spark)
        cached = cls._EMPTY_SPARK_FRAMES.get(key)
        if cached is not None:
            return cached
        df = spark.createDataFrame(
            [], schema=RESPONSE_SCHEMA.to_spark_schema(),
        )
        cls._EMPTY_SPARK_FRAMES[key] = df
        return df

    @classmethod
    def _responses_to_spark(
        cls,
        responses: list[Response],
        spark: "SparkSession",
    ) -> "SparkDataFrame":
        """Lift a list of :class:`Response` to a schema-bearing Spark frame.

        Used on the spark path to keep every bucket frame-resident.
        Empty input yields a cached empty DataFrame keyed to
        :data:`RESPONSE_SCHEMA` (one per :class:`SparkSession`) so the
        ``createDataFrame([], schema=...)`` JVM round-trip — about 30 ms
        on a warm cluster — only pays once per session instead of per
        bucket pass.
        """
        if not responses:
            return cls._cached_empty_spark_frame(spark)
        # Single bulk struct-array build — same ~30x speedup over the
        # per-row ``[r.to_arrow_batch(...) for r]`` loop documented on
        # :meth:`Response.values_to_arrow_batch`.
        table = pa.Table.from_batches(
            [Response.values_to_arrow_batch(responses)]
        )
        return spark.createDataFrame(table)

    @staticmethod
    def _flatten_remote_hits(
        remote_hits_by_table: (
            "dict[str, list[Response]] | dict[str, SparkDataFrame]"
        ),
    ) -> "list[Response] | SparkDataFrame | None":
        """Collapse a per-table remote-hits dict into one bucket.

        Empty dict → ``None``. Python lists chain into a single list
        (or ``None`` when every list is empty). Spark frames union via
        ``unionByName(allowMissingColumns=True)`` so the response
        schema is enforced across tables; a missing column would mean
        a real schema drift, and silent column-fill is worse than a
        loud failure.
        """
        if not remote_hits_by_table:
            return None
        values = list(remote_hits_by_table.values())
        first = values[0]
        if isinstance(first, list):
            flat: list[Response] = [
                r for v in remote_hits_by_table.values() for r in v  # type: ignore[union-attr]
            ]
            return flat or None
        result = first
        for part in values[1:]:
            result = result.unionByName(part, allowMissingColumns=True)
        return result

    def _split_remote_cache_spark(
        self,
        requests: list[PreparedRequest],
        *,
        spark: "SparkSession",
    ) -> tuple[dict[str, "SparkDataFrame"], list[PreparedRequest]]:
        """Spark variant of :meth:`_split_remote_cache`.

        Returns ``(hits_by_table, misses)`` — hits stay as Spark
        DataFrames keyed by ``CacheConfig.table.full_name(safe=True)``
        so the caller can hand them straight to :class:`HTTPResponseBatch`
        without ever materialising rows on the driver and without
        losing the per-table provenance to a premature
        ``unionByName``. Misses still come back as a Python list
        because the driver needs concrete request objects to scatter
        through stage 3.
        """
        misses: list[PreparedRequest] = []
        table_to_cfg: dict[str, CacheConfig] = {}
        table_to_reqs: dict[str, list[PreparedRequest]] = {}
        for req in requests:
            t_cfg = req.remote_cache_config
            if t_cfg is None or t_cfg.mode == Mode.UPSERT:
                misses.append(req)
                continue
            if not t_cfg.remote_cache_enabled or t_cfg.mode != Mode.APPEND:
                misses.append(req)
                continue
            tkey = t_cfg.tabular.full_name(safe=True)
            bucket = table_to_reqs.get(tkey)
            if bucket is None:
                table_to_cfg[tkey] = t_cfg
                table_to_reqs[tkey] = bucket = []
            bucket.append(req)

        hits_by_table: dict[str, "SparkDataFrame"] = {}
        for tkey, t_reqs in table_to_reqs.items():
            t_cfg = table_to_cfg[tkey]
            t_hits_df, t_misses = self._lookup_remote_table_spark(
                t_cfg, t_reqs, spark=spark,
            )
            if t_hits_df is not None:
                hits_by_table[tkey] = t_hits_df
            misses.extend(t_misses)

        if any(table_to_reqs.values()):
            LOGGER.debug(
                "Batch remote cache (spark): scanned %s table(s) for %s request(s)",
                len(table_to_cfg), len(requests),
            )
        return hits_by_table, misses

    def _lookup_remote_table_spark(
        self,
        cfg: CacheConfig,
        requests: list[PreparedRequest],
        *,
        spark: "SparkSession",
    ) -> tuple[Optional["SparkDataFrame"], list[PreparedRequest]]:
        """Spark variant of :meth:`_lookup_cached` (source=``remote``).

        Runs the same batch lookup SQL, but keeps the result as a Spark
        DataFrame instead of materialising :class:`Response` objects on
        the driver. Misses are computed by collecting the distinct
        ``request_by`` key tuples back to the driver — bounded by the
        number of cached rows that match this batch, not by total cache
        size — and diffing against the input requests.

        :meth:`CacheConfig.filter_response`'s per-row branch is skipped
        on the spark path: ``received_from`` / ``received_to`` are
        already encoded in the predicate the backend pushed down,
        and the request-key check is what the ``request_tuple`` diff
        already enforces.
        """
        from yggdrasil.data.options import CastOptions

        if cfg.request_by_is_public:
            lookup_batch: list[PreparedRequest] = list(requests)
        else:
            lookup_batch = [r.anonymize(mode=cfg.anonymize) for r in requests]
        predicate = cfg.make_batch_lookup_predicate(requests=lookup_batch)
        opts = CastOptions(predicate=predicate, spark_session=spark)
        try:
            hits_df = cfg.tabular.read_spark_frame(options=opts)
        except Exception as exc:
            if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                cfg.tabular.create(RESPONSE_ARROW_SCHEMA, missing_ok=True)
                hits_df = cfg.tabular.read_spark_frame(options=opts)
            else:
                raise
        # Stamp the origin flags on the read side. Stored values may be
        # stale (``mirror_local_to_remote`` pushes local hits into the
        # remote table with ``local_cached=True``) — overwrite both so
        # the downstream consumer always sees the layer that answered.
        from pyspark.sql import functions as F
        hits_df = (
            hits_df
            .withColumn("local_cached", F.lit(False))
            .withColumn("remote_cached", F.lit(True))
        )

        key_cols = list(cfg.request_by or [])
        if not key_cols:
            # No request-key columns means the SQL can't disambiguate
            # rows per request; mirror the Python path's behaviour by
            # treating every input request as a hit when any row came
            # back, otherwise everything is a miss. Pin the snapshot
            # (see the keyed branch below for the rationale) so the
            # caller's later ``.count()`` doesn't re-read the cache
            # table after stage 4 inserted the freshly-fetched misses.
            try:
                any_row = hits_df.head(1)
            except Exception:
                any_row = None
            if any_row:
                return self._pin_spark_snapshot(hits_df), []
            return None, list(requests)

        # Request-side ``request_by`` keys (``public_url_hash``,
        # ``method`` …) are stored on the response cache table under
        # the flattened ``request_<col>`` form, so the bare keys can't
        # be referenced as Spark column names — select via
        # ``_request_column_sql_name`` and read rows back through the
        # same prefixed names.
        sql_cols = [_request_column_sql_name(c) for c in key_cols]
        matched_rows = hits_df.select(*sql_cols).distinct().toLocalIterator()
        matched: set[tuple] = {
            tuple(row[c] for c in sql_cols) for row in matched_rows
        }

        misses: list[PreparedRequest] = []
        for req, lookup in zip(requests, lookup_batch):
            if cfg.request_tuple(lookup) not in matched:
                misses.append(req)
        if not matched:
            # Cold-cache short-circuit: returning a still-bound
            # SparkDataFrame for the empty match would let any later
            # action — e.g. :attr:`HTTPResponseBatch.counts` — re-execute
            # the SELECT after stage 4 has inserted ``misses`` into the
            # same cache table, double-counting those rows as remote
            # hits. The bucket really is empty; let the consumer drop it.
            return None, misses
        # Pin the matched-row snapshot. The lazy ``hits_df`` reads the
        # cache table, which stage 4 mutates in place — without an
        # eagerly-materialised cache snapshot a later ``.count()`` would
        # re-issue the SELECT and pick up the freshly-inserted miss rows.
        return self._pin_spark_snapshot(hits_df), misses

    @staticmethod
    def _pin_spark_snapshot(df: "SparkDataFrame") -> "SparkDataFrame":
        """Cache ``df`` and force one action so the snapshot is stable.

        Spark's ``DataFrame.cache`` is lazy — the partitions only
        materialise on the first action against the frame. When the
        downstream caller's first action runs after a sibling write to
        the same source table, the cached snapshot ends up containing
        rows that landed *after* the logical read. Forcing a single
        ``.count()`` here pins the partitions to the pre-mutation state.
        ``.cache`` itself can fail on Spark Connect logical plans the
        backend won't materialise — log and return the original frame
        so a best-effort pin doesn't crash the caller.
        """
        try:
            df = df.cache()
            df.count()
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "Failed to pin Spark snapshot for %r; downstream counts "
                "may re-execute the plan",
                df, exc_info=True,
            )
        return df

    def _fetch_misses(
        self,
        misses: list[PreparedRequest],
        config: SendConfig,
        *,
        ordered: bool = False,
        max_in_flight: int | None = None,
    ) -> Iterator[Response]:
        """Stage 3: send misses through the job pool."""
        miss_send_config = dataclasses.replace(
            config,
            remote_cache=None,
            local_cache=None,
            spark_session=None,
            raise_error=False,
        )

        pool = self.job_pool
        LOGGER.info(
            "Fetching %d send_many miss(es) through job pool "
            "(max_in_flight=%d, ordered=%s)",
            len(misses),
            max_in_flight or pool.max_workers,
            ordered,
        )
        for result in pool.as_completed(
            (
                Job.make(
                    self._send,
                    r.copy(send_config=miss_send_config),
                )
                for r in misses
            ),
            ordered=ordered,
            max_in_flight=max_in_flight or self.pool_maxsize,
            cancel_on_exit=False,
            shutdown_on_exit=False,
            raise_error=True,
        ):
            yield result.result

    @staticmethod
    def _enable_fair_spark_scheduler(spark: "SparkSession") -> None:
        """Best-effort switch the Spark session into FAIR scheduling.

        Stage 4 in Spark mode submits one Spark job per remote-cache
        insert; concurrent inserts (e.g. when several requests carry
        per-table remote configs, or when the local-cache writeback
        runs alongside the remote insert) all want share-the-cluster
        semantics rather than the default FIFO queue, where a slow
        insert blocks every other job behind it.

        ``spark.scheduler.mode`` is normally a SparkContext-level
        setting; some Spark builds will accept the runtime
        ``conf.set`` and apply it to subsequent jobs, others reject
        it. Either way this is non-fatal — we don't want a managed
        cluster's scheduler policy to break the request flow.
        """
        if spark is None:
            return
        try:
            spark.conf.set("spark.scheduler.mode", "FAIR")
        except Exception as exc:
            LOGGER.debug(
                "Could not switch Spark scheduler to FAIR: %s", exc,
            )

    @staticmethod
    def _run_concurrently(
        tasks: "Sequence[Callable[[], Any]]",
        *,
        max_workers: "int | None" = None,
        thread_name_prefix: str = "ygg-session",
    ) -> None:
        """Run *tasks* in parallel, re-raising the first failure.

        Used by the cache-write path (``_backfill_local_cache``,
        ``_persist_remote``, stage 4) to fan out independent inserts
        across threads so a batch that targets several remote tables
        or local cache roots doesn't pay for a head-to-tail
        serialization. ``ThreadPoolExecutor`` is used unconditionally
        because the work is I/O-bound (SQL statements, disk writes,
        Spark job submissions) — GIL contention is not the
        bottleneck.

        - 0 tasks → no-op.
        - 1 task  → run inline (skip pool / thread-spawn overhead).
        - N tasks → spawn ``min(N, max_workers or os.cpu_count())``
          workers, wait for all, then propagate the first captured
          exception so the caller sees the same failure shape it
          would have under the old sequential loop.
        """
        if not tasks:
            return
        if len(tasks) == 1:
            tasks[0]()
            return

        workers = min(len(tasks), max_workers or (os.cpu_count() or 4))
        first_exc: "BaseException | None" = None
        with ThreadPoolExecutor(
            max_workers=max(workers, 1),
            thread_name_prefix=thread_name_prefix,
        ) as pool:
            futures = [pool.submit(task) for task in tasks]
            for fut in futures:
                try:
                    fut.result()
                except BaseException as exc:
                    if first_exc is None:
                        first_exc = exc
        if first_exc is not None:
            raise first_exc

    def _backfill_local_cache(
        self,
        responses: list[Response],
    ) -> None:
        """Write remote-cache hits back to the partitioned local cache.

        Each response's local cache config is read from
        ``response.request.local_cache_config``. Responses are bucketed
        by their effective cache folder; each bucket fans out one Arrow
        batch routed through :meth:`FolderPath._write_arrow_batches`.
        """
        from yggdrasil.io.response import Response as _Response

        groups: dict[Path, tuple[CacheConfig, list[Response]]] = {}
        for response in responses:
            req = response.request
            if req is None:
                continue
            eff = req.local_cache_config
            if eff is not None:
                eff = eff.prebuild(session=self)
            if eff is None or not eff.local_cache_enabled:
                continue
            root = eff.local_cache_folder(session=self)
            bucket = groups.get(root)
            if bucket is None:
                groups[root] = (eff, [response])
            else:
                bucket[1].append(response)

        for root, (eff, group_responses) in groups.items():
            tabular = eff.cache_tabular(session=self)
            # One C++ struct walk per bucket beats N per-response
            # writes — same shape :meth:`_persist_remote` uses on
            # the SQL side. The folder's ``_write_arrow_batches``
            # splits the batch back out by ``partition_key`` so each
            # response still lands under its own
            # ``partition_key=<v>/`` directory.
            batch = _Response.values_to_arrow_batch(group_responses)
            Job.make(
                _insert_cache, tabular, eff, batch,
            ).fire_and_forget()

    def _persist_remote(
        self,
        responses: list[Response],
    ) -> None:
        """Stage 4: bulk-insert successful responses into the remote cache.

        Each response's remote cache config is read from
        ``response.request.remote_cache_config``. Responses are bucketed
        by the full write-group key (table, mode, match_by, wait,
        anonymize) so distinct per-request configs don't get collapsed.
        """
        groups: dict[tuple, tuple[CacheConfig, list[Response]]] = {}
        for response in responses:
            req = response.request
            if req is None:
                continue
            eff = req.remote_cache_config
            if eff is None or not eff.remote_cache_enabled:
                continue
            gkey = self._remote_write_group_key(eff)
            if gkey not in groups:
                groups[gkey] = (eff, [])
            groups[gkey][1].append(response)

        if not groups:
            LOGGER.info(
                "Stage 4: no remote-cache groups to persist "
                "(all configs disabled or missing)",
            )
            return

        def _insert_one(
            mode: "Mode",
            cfg: "CacheConfig",
            group_responses: "list[Response]",
        ) -> None:
            LOGGER.info(
                "%s %d response(s) in remote cache %r",
                "Upserting" if mode == Mode.UPSERT else "Persisting",
                len(group_responses),
                cfg.tabular,
            )
            # One C++ struct walk over the whole group beats N
            # per-row builds + an outer ``combine_chunks`` concat —
            # see :meth:`Response.values_to_arrow_batch`. The result
            # is a single chunked-on-construction batch wrapped in a
            # one-element table so the downstream
            # ``batches["partition_key"]`` slot lookup keeps working.
            batches = pa.Table.from_batches(
                [Response.values_to_arrow_batch(group_responses)]
            )
            _insert_cache(
                cfg.tabular, cfg, batches,
                mode=mode,
                prune_values=_cache_prune_values_for(batches),
                raise_error=True,
            )

        self._run_concurrently(
            [
                lambda m=gkey[1], c=cfg, r=group_responses: _insert_one(m, c, r)
                for gkey, (cfg, group_responses) in groups.items()
            ],
            thread_name_prefix="ygg-remote-cache-insert",
        )

    def _mirror_local_hits_to_remote(
        self,
        local_hits: "list[Response]",
    ) -> None:
        """Bulk-upsert local-cache hits into the remote cache.

        Activation is gated on the *remote* config — local-only
        sessions skip it entirely. Each response's remote cache config
        is read from ``response.request.remote_cache_config``.
        """
        if not local_hits:
            return

        keep: list[Response] = []
        for response in local_hits:
            req = response.request
            if req is None:
                continue
            eff = req.remote_cache_config
            if eff is None or not eff.remote_cache_enabled:
                continue
            if not eff.mirror_local_to_remote:
                continue
            keep.append(response)

        if not keep:
            return

        LOGGER.info(
            "Mirroring %d local-cache hit(s) to remote cache",
            len(keep),
        )
        self._persist_remote(keep)

    def _send_many(
        self,
        requests: Iterator[PreparedRequest],
        config: SendConfig,
        **batch_kw: Any,
    ) -> Iterator[Response]:
        """Stream responses, flattening the per-chunk :class:`HTTPResponseBatch`.

        In Spark mode ``raise_error`` is applied at the driver-iteration
        boundary so a single failure doesn't poison a whole partition.
        """
        is_spark = config.spark_session is not None
        for batch in self._send_many_batches(requests, config, **batch_kw):
            for holder in batch.parts():
                for response in Response.from_records(holder.read_records()):
                    if is_spark and config.raise_error and not response.ok:
                        response.raise_for_status()
                    yield response

    def send_many_batches(
        self,
        requests: Iterator[PreparedRequest],
        config: SendConfig | Mapping[str, Any] | None = None,
        *,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        cache_only: bool = False,
        as_tabular: bool = False,
        spark_session: Optional["SparkSession"] = None,
        batch_size: int | None = None,
        ordered: bool = False,
        max_in_flight: int | None = None,
        max_batch_ttl: float | None = None,
        **options,
    ) -> Iterator[HTTPResponseBatch]:
        """Yield one :class:`HTTPResponseBatch` per processed chunk."""
        cfg = SendConfig.from_(
            config,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
            cache_only=cache_only,
            as_tabular=as_tabular,
            spark_session=spark_session,
            **options,
        )
        yield from self._send_many_batches(
            requests, cfg,
            batch_size=batch_size,
            ordered=ordered,
            max_in_flight=max_in_flight,
            max_batch_ttl=max_batch_ttl,
        )

    def _send_many_batches(
        self,
        requests: Iterator[PreparedRequest],
        config: SendConfig,
        *,
        batch_size: int | None = None,
        max_batch_size: int | None = None,
        max_in_flight: int | None = None,
        ordered: bool = False,
        max_batch_ttl: float | None = DEFAULT_MAX_BATCH_TTL,
    ) -> Iterator[HTTPResponseBatch]:
        """Yield one :class:`HTTPResponseBatch` per processed chunk."""
        spark = config.spark_session
        is_spark = spark is not None

        if is_spark:
            self._enable_fair_spark_scheduler(spark)

        pool = self.job_pool
        if batch_size:
            eff_batch_size = batch_size
        elif is_spark:
            eff_batch_size = max_batch_size or 1024
        else:
            eff_batch_size = min(max_batch_size or 1024, pool.max_workers * 10)

        ttl = max_batch_ttl

        _lc = config.local_cache
        _rc = config.remote_cache
        LOGGER.debug(
            "Starting send_many pipeline (mode=%s, batch_size=%d, "
            "max_in_flight=%s, ttl=%s, ordered=%s, "
            "local_cache=%s, remote_cache=%s)",
            "spark" if is_spark else "python",
            eff_batch_size,
            max_in_flight,
            ttl,
            ordered,
            _lc.local_cache_enabled if _lc is not None else False,
            _rc.remote_cache_enabled if _rc is not None else False,
        )

        chunk_index = 0
        total_local = 0
        total_remote = 0
        total_network = 0
        total_failed = 0

        def _batched(
            it: Iterator[PreparedRequest],
            n: int,
            ttl_seconds: float | None,
        ) -> Iterator[list[PreparedRequest]]:
            # When no TTL is set, fall back to the cheap islice path —
            # avoids the per-request monotonic() probe.
            iterator = iter(it)
            if ttl_seconds is None or ttl_seconds <= 0:
                while True:
                    b = list(itertools.islice(iterator, n))
                    if not b:
                        break
                    yield b
                return

            # Time-bounded path: pull one item at a time and flush
            # when either the size cap or the wall-clock deadline is
            # reached. The deadline is reset per chunk so a slow
            # upstream gets a fresh window after each flush.
            buf: list[PreparedRequest] = []
            deadline: float | None = None
            for item in iterator:
                if not buf:
                    deadline = time.monotonic() + ttl_seconds
                buf.append(item)
                if len(buf) >= n or (
                    deadline is not None and time.monotonic() >= deadline
                ):
                    yield buf
                    buf = []
                    deadline = None
            if buf:
                yield buf

        chunks = _batched(requests, eff_batch_size, ttl)

        for chunk in chunks:
            if not chunk:
                continue

            chunk_index += 1

            # Enrich: stamp the session-level send_config on every
            # request that doesn't carry a per-request override.
            # Per-request cache configs are preserved (same rule as
            # single send()).
            for r in chunk:
                req_sc = r.send_config
                if req_sc is None:
                    r.send_config = config
                else:
                    merge_back: dict[str, Any] = {}
                    if req_sc.local_cache is not None:
                        merge_back["local_cache"] = req_sc.local_cache
                    if req_sc.remote_cache is not None:
                        merge_back["remote_cache"] = req_sc.remote_cache
                    base = dataclasses.replace(config, **merge_back) if merge_back else config
                    if base is not req_sc:
                        r.send_config = base

            # Split: requests with cache config (tabular) vs bare network.
            uncached: list[PreparedRequest] = []
            cached: list[PreparedRequest] = []
            for r in chunk:
                if r.local_cache_config or r.remote_cache_config:
                    cached.append(r)
                else:
                    uncached.append(r)

            LOGGER.debug(
                "Processing send_many chunk #%d (requests=%d, "
                "cached=%d, uncached=%d, mode=%s)",
                chunk_index, len(chunk), len(cached), len(uncached),
                "spark" if is_spark else "python",
            )

            # --- Cached path: local → remote → network for misses ---
            local_hits: "Tabular | None" = None
            remote_hits: "Tabular | list[Response] | None" = None
            cache_misses: list[PreparedRequest] = list(cached)

            if cached:
                local_hits, cache_misses = self._split_local_cache(cached)

                if cache_misses:
                    remote_hits_by_table, cache_misses = self._split_remote_cache(
                        cache_misses,
                    )
                    remote_flat = [
                        r
                        for table_hits in remote_hits_by_table.values()
                        for r in table_hits
                    ]
                    self._backfill_local_cache(remote_flat)
                    remote_hits = self._flatten_remote_hits(remote_hits_by_table)

                if local_hits is not None:
                    local_responses = list(
                        Response.from_arrow_tabular(local_hits.read_arrow_batches())
                    )
                    self._mirror_local_hits_to_remote(local_responses)

            # --- Network: uncached requests + cache misses ---
            to_fetch = uncached + cache_misses
            new_hits: "list[Response] | SparkDataFrame | None" = None
            failed: list[Response] = []

            if to_fetch and config.cache_only:
                new_hits = [_synthetic_not_found(r) for r in to_fetch]
                LOGGER.info(
                    "send_many chunk #%d: cache_only=True, synthesising "
                    "%d 404 response(s)",
                    chunk_index, len(to_fetch),
                )
            elif to_fetch:
                spark_misses: list[PreparedRequest] = []
                local_misses: list[PreparedRequest] = []
                for r in to_fetch:
                    if r.send_config_or_default.spark_session is not None:
                        spark_misses.append(r)
                    else:
                        local_misses.append(r)

                spark_new_hits: "SparkDataFrame | None" = None
                if spark_misses:
                    spark_new_hits = self._spark_fetch_misses(
                        spark_misses, config,
                        spark_misses[0].send_config_or_default.spark_session,
                    )

                new_list: list[Response] = []
                if local_misses:
                    for response in self._fetch_misses(
                        local_misses, config,
                        ordered=ordered, max_in_flight=max_in_flight,
                    ):
                        if response.ok:
                            new_list.append(response)
                        elif config.raise_error:
                            failed.append(response)

                # --- Writeback: persist network results to caches ---
                stage4: "list[Callable[[], Any]]" = []
                if spark_new_hits is not None:
                    _spark_hits = spark_new_hits
                    _spark_remote_cfg = {
                        r.public_url_hash: r.remote_cache_config
                        for r in spark_misses
                    }
                    stage4.append(
                        lambda: self._spark_persist_remote(
                            _spark_hits,
                            _spark_remote_cfg,
                            spark=spark_misses[0].send_config_or_default.spark_session,
                        )
                    )
                if new_list:
                    _local_hits = new_list
                    stage4.append(lambda: self._persist_remote(_local_hits))
                    stage4.append(lambda: self._backfill_local_cache(_local_hits))
                if stage4:
                    self._run_concurrently(
                        stage4, thread_name_prefix="ygg-stage4",
                    )

                if spark_new_hits is not None and new_list:
                    from yggdrasil.io.response import Response as _Resp
                    for r in _Resp.from_records(spark_new_hits.toArrow()):
                        new_list.append(r)
                    new_hits = new_list
                elif spark_new_hits is not None:
                    new_hits = spark_new_hits
                else:
                    new_hits = new_list or None

            local_count = local_hits.num_rows if local_hits is not None else 0
            remote_count = len(remote_hits) if isinstance(remote_hits, list) else 0
            net_count = len(new_hits) if isinstance(new_hits, list) else 0
            total_local += local_count
            total_remote += remote_count
            total_network += net_count
            total_failed += len(failed)
            LOGGER.info(
                "Completed send_many chunk #%d (requests=%d, "
                "local_hits=%d, remote_hits=%d, network=%d, failed=%d)",
                chunk_index, len(chunk), local_count,
                remote_count, net_count, len(failed),
            )

            yield HTTPResponseBatch(
                local_hits=local_hits,
                remote_hits=remote_hits,
                new_hits=new_hits,
                spark=spark,
            )

            if config.raise_error and failed:
                failed[-1].raise_for_status()

        LOGGER.info(
            "Finished send_many pipeline (chunks=%d, local_hits=%d, "
            "remote_hits=%d, network=%d, failed=%d)",
            chunk_index, total_local, total_remote,
            total_network, total_failed,
        )

    # ------------------------------------------------------------------ #
    # Spark stage 3 / 4 helpers                                           #
    # ------------------------------------------------------------------ #

    def _spark_persist_remote(
        self,
        new_responses_df: "SparkDataFrame",
        key_to_remote_cfg: Mapping[int, CacheConfig],
        *,
        spark: "SparkSession",
    ) -> None:
        """Stage 4 on Spark: per-request bulk-insert into the remote cache.

        Mirrors :meth:`_persist_remote`: each ``public_url_hash`` resolves
        to its effective :class:`CacheConfig` via ``key_to_remote_cfg``,
        groups bucket by :meth:`_remote_write_group_key`, and each group's
        insert runs concurrently. The Spark frame is persisted once when
        more than one group fires so the network fetch behind it doesn't
        re-execute per group; single-group inserts keep the legacy
        zero-persist plan.

        Before inserting, APPEND-mode writes are de-duplicated against the
        existing remote rows via a ``left_anti`` join on the response
        ``(partition_key, public_hash)`` keys — the remote table stores
        anonymized requests (cf. ``_persist_remote``), so a row whose
        hash already lives in the cache is suppressed rather than
        re-inserted. UPSERT mode keeps its read-free fast path and
        relies on ``match_by`` to collapse duplicates server-side.
        """
        from pyspark.sql import functions as F

        # Bucket request hashes by their effective remote-cache config's
        # write group. Disabled configs drop out here so the persist
        # path never fires for them.
        groups: dict[tuple, tuple[CacheConfig, list[int]]] = {}
        for cfg_key, eff in key_to_remote_cfg.items():
            if eff is None or not eff.remote_cache_enabled:
                continue
            gkey = self._remote_write_group_key(eff)
            if gkey not in groups:
                groups[gkey] = (eff, [])
            groups[gkey][1].append(cfg_key)

        if not groups:
            LOGGER.info(
                "Spark stage 4: no remote-cache groups to persist "
                "(all configs disabled or missing)",
            )
            return

        LOGGER.info(
            "Spark stage 4: persisting into %d remote-cache group(s) — %s",
            len(groups),
            ", ".join(
                f"{cfg.tabular.full_name(safe=True)} "
                f"(mode={cfg.mode.name}, hashes=%d)" % len(hashes)
                for (_gkey, (cfg, hashes)) in groups.items()
            ),
        )

        ok_df = new_responses_df.where(
            (F.col("status_code") >= 200)
            & (F.col("status_code") < 300)
        )

        # Persist only when more than one group will read ``ok_df`` —
        # otherwise the single insert action is the one and only
        # evaluation and we save the storage round trip. Route the
        # cache through :class:`yggdrasil.spark.tabular.Dataset`
        # so backends that reject ``persist`` (Databricks Connect
        # serverless raises ``[NOT_SUPPORTED_WITH_SERVERLESS] PERSIST
        # TABLE``) fall through to the un-cached frame instead of
        # crashing stage 4 — pass two then runs twice, but the
        # alternative is a hard failure on serverless compute.
        multi_group = len(groups) > 1
        ok_dataset: "Dataset | None" = None
        if multi_group:
            from yggdrasil.spark.tabular import Dataset

            ok_dataset = Dataset(frame=ok_df).cache()
            ok_df = ok_dataset.frame

        # When every key in the chunk's cfg map collapses onto one
        # group, the inserted frame IS the whole ``ok_df``. With any
        # disabled / split-out keys present we have to filter by
        # ``request_public_url_hash`` so dropped requests don't leak
        # into the surviving group's insert.
        covers_chunk = (
            len(groups) == 1
            and sum(len(hs) for _, (_, hs) in groups.items())
            == len(key_to_remote_cfg)
        )

        def _insert_one(cfg: CacheConfig, hashes: list[int]) -> None:
            table_name = cfg.tabular.full_name(safe=True)
            if covers_chunk:
                df = ok_df
            else:
                df = ok_df.where(
                    F.col("request_public_url_hash").isin(hashes)
                )

            if cfg.mode != Mode.UPSERT:
                try:
                    wanted_partitions = [
                        row["partition_key"]
                        for row in df.select("partition_key").distinct().collect()
                    ]
                except Exception:
                    wanted_partitions = []

                LOGGER.info(
                    "Spark dedup for %s: checking %d partition(s) "
                    "against existing rows",
                    table_name, len(wanted_partitions),
                )

                try:
                    if wanted_partitions:
                        literals = ", ".join(str(int(v)) for v in wanted_partitions)
                        existing_df = spark.sql(
                            "SELECT DISTINCT partition_key, public_hash "
                            f"FROM {table_name} WHERE partition_key IN ({literals})"
                        )
                    else:
                        existing_df = spark.sql(
                            "SELECT DISTINCT partition_key, public_hash "
                            f"FROM {table_name}"
                        )
                except Exception as exc:
                    if "TABLE_OR_VIEW_NOT_FOUND" not in str(exc):
                        raise
                    LOGGER.info(
                        "Spark dedup for %s: table not found — "
                        "skipping dedup (first insert)",
                        table_name,
                    )
                    existing_df = None

                if existing_df is not None:
                    df = df.join(
                        existing_df,
                        on=["partition_key", "public_hash"],
                        how="left_anti",
                    )

            LOGGER.info(
                "%s ok response(s) into remote cache %s "
                "(spark insert, mode=%s, hashes=%d)",
                "Upserting" if cfg.mode == Mode.UPSERT else "Persisting",
                table_name, cfg.mode.name, len(hashes),
            )
            _insert_cache(
                cfg.tabular, cfg, df,
                spark_session=spark,
                raise_error=True,
            )

        try:
            self._run_concurrently(
                [
                    lambda c=cfg, hs=hashes: _insert_one(c, hs)
                    for (_gkey, (cfg, hashes)) in groups.items()
                ],
                thread_name_prefix="ygg-spark-remote-cache-insert",
            )
            LOGGER.info(
                "Spark stage 4 completed: persisted into %d "
                "remote-cache group(s)",
                len(groups),
            )
        finally:
            if ok_dataset is not None:
                ok_dataset.unpersist()

    def _spark_fetch_misses(
        self,
        misses: list[PreparedRequest],
        config: SendConfig,
        spark: "SparkSession",
    ) -> "SparkDataFrame":
        """Stage 3 on Spark: scatter misses to workers via mapInArrow.

        Requests cross the wire as an Arrow table with the canonical
        :data:`~yggdrasil.io.request.REQUEST_SCHEMA` — deterministic
        columns, no pickled Python objects in the row payload.

        The driver-side :class:`Session` itself is pickled once with
        :func:`pickle.dumps` and that bytes blob travels with the
        ``mapInArrow`` closure; each partition rehydrates the same
        subclass via :func:`pickle.loads`, so user overrides (custom
        auth, header injection, request hooks) execute on the worker.
        ``Session.__getstate__`` / ``__setstate__`` strip the
        threading.RLock and JobPoolExecutor before serialising and
        re-init them on the way back in, and ``Singleton.__new__``
        collapses the loads onto the executor's per-``(cls, config)``
        cached instance so all partitions share one connection pool.
        Spark Connect / Databricks Connect callers need the user's
        subclass module on the cluster's ``sys.path`` for the
        cloudpickle round-trip to resolve — that's the price of running
        a driver-side subclass on the worker.

        Each Spark partition becomes one :meth:`Session.send_many` call,
        fanning out via the rehydrated session's thread pool. Both local
        and remote cache configs are forwarded: workers consult the same
        caches as the driver, so a request the driver fan-out missed but
        a peer worker has already cached can still short-circuit before
        hitting the network.
        """
        if not misses:
            return self._cached_empty_spark_frame(spark)

        # Driver-side prepare before the mapInArrow scatter. Subclasses
        # use :meth:`prepare_request_before_send` for URL normalisation,
        # session-wide header injection, auth refresh, request signing
        # — applying it here means the wire payload, the Arrow row,
        # and any signed Authorization header all reflect the driver's
        # view before crossing executors. Workers re-prepare inside
        # :meth:`send_many` → :meth:`_send` (double-prepare) so
        # per-executor concerns — stale token refresh on a long-running
        # partition, executor-local correlation IDs — still apply.
        # The default hook (session-header merge + cached-token auth
        # refresh) is idempotent under double-call; subclasses that
        # mutate non-idempotently should guard on their own state.
        for req in misses:
            self.prepare_request_before_send(req)
        LOGGER.debug(
            "Driver-prepared %d send_many miss(es) before mapInArrow "
            "scatter (workers re-prepare via send_many)",
            len(misses),
        )

        # One C++-side struct walk turns the request list into a single
        # Arrow table that matches REQUEST_SCHEMA column-for-column.
        # No per-row pickle, no closure capture of ``self``.
        request_table = pa.Table.from_batches(
            [PreparedRequest.values_to_arrow_batch(misses)]
        )

        # Spread requests across many partitions so mapInArrow scatters
        # across the whole cluster instead of piling them onto a handful
        # of executors. ``createDataFrame`` defaults to a single partition
        # for small Python lists, which serialises stage 3. Target one
        # request per partition, capped at ``defaultParallelism * 8`` so
        # huge request lists don't explode into thousands of micro-tasks
        # whose scheduler overhead dominates the actual fetch.
        #
        # ``sparkContext`` isn't reachable from a Spark Connect proxy
        # (``PySparkAttributeError(JVM_ATTRIBUTE_NOT_SUPPORTED)``); fall
        # back to a sensible default for the partition fan-out.
        try:
            default_par = max(spark.sparkContext.defaultParallelism, 1)
        except Exception:
            default_par = 8
        n_parts = max(1, min(len(misses), default_par * 8))
        LOGGER.info(
            "Scattering %d send_many miss(es) across %d Spark partition(s) "
            "via mapInArrow (default_parallelism=%d)",
            len(misses), n_parts, default_par,
        )
        request_df = spark.createDataFrame(request_table).repartition(n_parts)

        # Worker-side send config: local cache with 15-min TTL keeps
        # the executor's disk cache bounded; no remote cache — the
        # driver handles remote persistence in stage 4.
        worker_send_config = dataclasses.replace(
            config,
            remote_cache=None,
            spark_session=None,
            raise_error=False,
        )
        lc = worker_send_config.local_cache
        if lc is not None:
            worker_send_config = dataclasses.replace(
                worker_send_config,
                local_cache=lc.copy(received_ttl=dt.timedelta(minutes=15)),
            )

        self_serialized = pickle.dumps(self)
        response_spark_schema = RESPONSE_SCHEMA.to_spark_schema()

        # Broadcast session bytes and send_config so Spark distributes
        # them once via the broadcast protocol instead of embedding the
        # blobs in every task closure.
        try:
            broadcast_session = spark.sparkContext.broadcast(self_serialized)
            broadcast_config = spark.sparkContext.broadcast(worker_send_config)
            _use_broadcast = True
        except Exception:
            _use_broadcast = False

        def _send_partition(
            batches: Iterator[pa.RecordBatch],
        ) -> Iterator[pa.RecordBatch]:
            if _use_broadcast:
                session = pickle.loads(broadcast_session.value)
                partition_config = broadcast_config.value
            else:
                session = pickle.loads(self_serialized)
                partition_config = worker_send_config
            for batch in batches:
                partition_requests = list(PreparedRequest.from_arrow(batch))
                if not partition_requests:
                    continue

                def _row_batches() -> Iterator[pa.RecordBatch]:
                    for resp in session.send_many(
                        iter(partition_requests), partition_config,
                    ):
                        yield resp.to_arrow_batch(parse=False)

                yield from rechunk_arrow_batches(
                    _row_batches(),
                    byte_size=_SPARK_RESPONSE_BATCH_BYTE_LIMIT,
                )

        result_df = request_df.mapInArrow(
            _send_partition, schema=response_spark_schema,
        )
        # Cache so stage 4's insert (the first action on this frame)
        # both materialises and caches it. Without the cache, every
        # later action — including :attr:`HTTPResponseBatch.counts` — would
        # re-execute the ``mapInArrow``, re-issuing per-partition
        # network calls AND letting workers' ``send_many`` short-circuit
        # on the very rows stage 4 has just persisted to the remote
        # cache table (double-counting them as fresh hits).
        try:
            result_df = result_df.cache()
            LOGGER.info(
                "Spark stage 3 completed: cached mapInArrow result "
                "for %d miss(es) across %d partition(s)",
                len(misses), n_parts,
            )
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "Failed to cache stage-3 mapInArrow result; downstream "
                "counts may re-execute the per-partition fetch",
                exc_info=True,
            )
        return result_df
    
    def get(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> Response | PreparedRequest:
        return self.request(
            "GET",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def post(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> Response | PreparedRequest:
        return self.request(
            "POST",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            json=json,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def put(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> Response | PreparedRequest:
        return self.request(
            "PUT",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            json=json,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def patch(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> Response | PreparedRequest:
        return self.request(
            "PATCH",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            json=json,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def delete(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> Response | PreparedRequest:
        return self.request(
            "DELETE",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            json=json,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def head(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = False,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> Response | PreparedRequest:
        return self.request(
            "HEAD",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def options(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> Response | PreparedRequest:
        return self.request(
            "OPTIONS",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            json=json,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def request(
        self,
        method: str,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> Response | PreparedRequest:
        # ``requests``-style aliases: ``data=`` becomes ``body=`` (with
        # form-urlencoding for mappings/sequences), ``timeout=`` becomes
        # ``wait=``, and ``cookies=`` joins ``headers={'Cookie': ...}``.
        # Conflicting pairs raise — picking a side silently would mask a
        # caller bug.
        if data is not None:
            if body is not None:
                raise ValueError(
                    "Pass only one of body= or data= (got both). Use data= "
                    "for requests-style form/raw bodies, body= for the "
                    "native BytesIO/bytes path."
                )
            body, form_content_type = _encode_request_data(data)
            if form_content_type is not None:
                merged = Headers(headers) if headers else Headers()
                if "Content-Type" not in merged:
                    merged["Content-Type"] = form_content_type
                headers = merged

        if timeout is not None:
            if wait is not None:
                raise ValueError(
                    "Pass only one of wait= or timeout= (got both). They "
                    "map to the same underlying WaitingConfig; pick one."
                )
            wait = timeout

        if cookies:
            cookie_header = _format_cookie_header(cookies)
            if cookie_header:
                merged = Headers(headers) if headers else Headers()
                if "Cookie" not in merged:
                    merged["Cookie"] = cookie_header
                headers = merged

        prepared = self.prepare_request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
        )

        if not send:
            return prepared

        return self.send(
            prepared,
            config=config,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
            **options,
        )

    def prepare_request(
        self,
        method: str,
        url: URL | str | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        local_cache_config: Optional[CacheConfig] = None,
        remote_cache_config: Optional[CacheConfig] = None,
        *,
        json: Any | None = None,
        normalize: bool = True,
        send_config: SendConfig | None = None,
        **send_kwargs: Any,
    ) -> PreparedRequest:
        full_url: URL | str | None = url

        if self.base_url:
            full_url = self.base_url.join(url) if url else self.base_url
        elif url is None:
            raise ValueError("url is required when base_url is not set on the session.")

        if params:
            parsed = URL.from_(full_url, normalize=normalize)
            full_url = parsed.with_query_items(params)

        if send_kwargs:
            send_config = SendConfig.from_(send_config, **send_kwargs)

        request = PreparedRequest.prepare(
            method=method,
            url=full_url,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            local_cache_config=local_cache_config,
            remote_cache_config=remote_cache_config,
            send_config=send_config,
        )
        request.attach_session(self)
        return request
