"""Concrete HTTP/HTTPS session backed by ``urllib3``.

This module provides :class:`HTTPSession`, the only built-in subclass of
:class:`~yggdrasil.io.session.Session`.  It uses a
:class:`urllib3.PoolManager` for connection reuse and exposes the full
:meth:`~Session.send` contract — including optional Delta-table caching and
transparent pagination assembly — without requiring the caller to know about
the underlying transport.

Typical usage
-------------
::

    from yggdrasil.io.session import Session
    from yggdrasil.io.send_config import SendConfig

    with Session.from_url("https://api.example.com") as s:
        resp = s.get("/v1/items", config=SendConfig(wait=30))

Or construct :class:`HTTPSession` directly for full control::

    from yggdrasil.io.http_ import HTTPSession

    s = HTTPSession(
        base_url="https://api.example.com",
        pool_maxsize=20,
        verify=True,
    )
    s.x_api_key = "secret"
    resp = s.post("/upload", json={"key": "value"})
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Literal

import urllib3

from yggdrasil.concurrent.threading import JobPoolExecutor, Job
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg
from yggdrasil.io import MediaType, MimeType, BytesIO
from .response import HTTPResponse
from ..enums import SaveMode
from ..request import PreparedRequest
from ..response import RESPONSE_ARROW_SCHEMA
from ..send_config import SendConfig
from ..session import Session

if TYPE_CHECKING:
    from yggdrasil.databricks.sql.table import Table

__all__ = ["HTTPSession"]


@dataclass
class HTTPSession(Session):
    """Concrete HTTP/HTTPS session backed by a ``urllib3`` connection pool.

    Inherits all batching, caching, Spark-scatter, and verb-shortcut
    behaviour from :class:`~yggdrasil.io.session.Session`.  This class adds:

    * A :class:`urllib3.PoolManager` with configurable retry logic
      (exponential back-off, status-code allow-list).
    * Transparent **pagination assembly**: if the server returns
      ``X-Current-Page`` / ``X-Last-Page`` headers, all remaining pages are
      fetched concurrently and merged into a single
      :class:`~yggdrasil.io.http_.response.HTTPResponse` before returning.
    * Full Delta-table **response caching** in :meth:`send`: cache hits skip
      the network entirely; cache misses are written back after a successful
      live response.

    Parameters
    ----------
    base_url:
        See :class:`~yggdrasil.io.session.Session`.
    verify:
        Whether to verify TLS certificates.  Maps to urllib3's
        ``cert_reqs="CERT_REQUIRED"`` / ``"CERT_NONE"``.
    pool_maxsize:
        Maximum number of open connections.  Capped at 8 because
        urllib3 does not handle larger pools well on most platforms.
        Values supplied above 8 are silently clamped.
    send_headers:
        See :class:`~yggdrasil.io.session.Session`.
    waiting:
        See :class:`~yggdrasil.io.session.Session`.

    Notes
    -----
    The ``urllib3`` pool is lazily initialised on the first :meth:`send`
    call.  The pool itself is **not** pickled (it is excluded from
    :meth:`~yggdrasil.io.session.Session.__getstate__`) so the session can
    be safely broadcast to Spark executors; each executor creates its own
    pool on first use.

    Examples
    --------
    ::

        s = HTTPSession(base_url="https://api.example.com", pool_maxsize=4)
        with s:
            pages = list(s.send_many(requests, ordered=True))
    """

    _http_pool: urllib3.PoolManager = field(default=None, init=False, repr=False, compare=False)

    def _build_http_pool(self) -> urllib3.PoolManager:
        """Create and return a new :class:`urllib3.PoolManager`.

        Retry policy:

        * Up to 6 total retries (2 on connect, 4 on read).
        * Exponential back-off with a factor of 10 s.
        * Retries on status codes 429, 500, 502, 503, 504 (but does not
          raise; the raw response is returned so the caller can inspect it).

        Returns
        -------
        urllib3.PoolManager
        """
        retries = urllib3.Retry(
            total=6,
            connect=2,
            read=4,
            backoff_factor=10,
            status_forcelist=(429, 500, 502, 503, 504),
            raise_on_status=False,
        )

        return urllib3.PoolManager(
            num_pools=self.pool_maxsize,
            maxsize=self.pool_maxsize,
            block=True,
            retries=retries,
            cert_reqs="CERT_REQUIRED" if self.verify else "CERT_NONE",
            ca_certs=None,
        )

    def __post_init__(self):
        """Clamp :attr:`pool_maxsize` to 8, then delegate to the parent.

        urllib3 does not handle pools larger than 8 connections reliably on
        most platforms, so the value is clamped here before being forwarded
        to :meth:`~yggdrasil.io.session.Session.__post_init__` which
        initialises the threading lock and job pool.
        """
        if self.pool_maxsize:
            self.pool_maxsize = min(8, int(self.pool_maxsize))
        else:
            self.pool_maxsize = 8

        super().__post_init__()

        if self._http_pool is None:
            self._build_http_pool()

    @property
    def http_pool(self):
        """Return the lazily-initialised :class:`urllib3.PoolManager` (double-checked locking)."""
        if self._http_pool is None:
            with self._lock:
                if self._http_pool is None:
                    self._http_pool = self._build_http_pool()
        return self._http_pool

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
    ) -> HTTPResponse:
        """Send a single HTTP request and return its response.

        Implements :meth:`~yggdrasil.io.session.Session.send` for HTTP/HTTPS
        targets using a ``urllib3`` connection pool.

        Cache behaviour (when *cache* is supplied):

        1. The request is anonymised and looked up in the Delta table.
        2. If a cached response is found it is returned immediately — no
           network call is made.
        3. On a cache miss the request is dispatched, and the successful
           response is written back to the table asynchronously.

        Pagination (``X-Current-Page`` / ``X-Last-Page`` headers):

        If the response carries pagination headers, all remaining pages are
        fetched concurrently and merged into a single response before
        returning.

        Parameters
        ----------
        request:
            Fully prepared request to dispatch.
        config:
            Optional :class:`~yggdrasil.io.send_config.SendConfig` providing
            defaults for every other keyword argument.  Explicit kwargs
            override the config.
        wait:
            Retry / waiting strategy.  ``None`` defers to ``config.wait``
            or the session's :attr:`~yggdrasil.io.session.Session.waiting`
            default.
        raise_error:
            Raise :exc:`~yggdrasil.io.response.ResponseError` on non-2xx
            responses when ``True`` (default).
        stream:
            Stream the response body lazily when ``True`` (default).
        cache:
            Delta table used to cache responses.  ``None`` disables caching.
        cache_by:
            Column names forming the cache key.  Defaults to the standard
            request-fingerprint columns when *cache* is set.
        received_from:
            Earliest acceptable cached-response timestamp.
        received_to:
            Latest acceptable cached-response timestamp.
        anonymize:
            How to strip sensitive fields before the cache lookup and write
            (``"remove"`` or ``"redact"``).
        wait_cache:
            Waiting config for the background cache write.  ``False`` means
            fire-and-forget.

        Returns
        -------
        HTTPResponse
            The HTTP response (possibly assembled from multiple pages).
        """
        # Merge config + explicit kwargs into a single resolved config
        cfg = self._resolve_send_config(
            config,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            cache=cache,
            cache_by=cache_by,
            anonymize=anonymize,
            received_from=received_from,
            received_to=received_to,
            wait_cache=wait_cache,
        )
        wait         = cfg.wait
        raise_error  = cfg.raise_error
        stream       = cfg.stream
        cache        = cfg.cache
        cache_by     = cfg.cache_by
        anonymize    = cfg.anonymize
        received_from = cfg.received_from
        received_to  = cfg.received_to
        wait_cache   = cfg.wait_cache

        if cache is not None:
            cache_by = self._cache_by_keys(cache_by)
            cache_request_by = [_ for _ in cache_by if _.startswith("request")]

            anon = request.anonymize(mode=anonymize)
            query = (
                f"SELECT * FROM {cache.full_name(safe=True)}"
                f" WHERE {self._sql_match_clause(anon, keys=cache_request_by, received_from=received_from, received_to=received_to)}"
                f" ORDER BY response_received_at_epoch DESC"
                f" LIMIT 1"
            )

            try:
                sql_cache_statement = cache.sql.execute(query)
            except Exception as e:
                if "TABLE_OR_VIEW_NOT_FOUND" in str(e):
                    cache.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                    sql_cache_statement = cache.sql.execute(query)
                else:
                    raise

            for response in HTTPResponse.from_arrow(sql_cache_statement.to_arrow_table()):
                if raise_error:
                    response.raise_for_status()
                return response

        request = request.prepare_to_send(
            sent_at_timestamp=time.time_ns() // 1000,
            headers=self.send_headers,
        )
        http_pool = self.http_pool
        wait_cfg = self.waiting if wait is None else WaitingConfig.check_arg(wait)

        first_resp = http_pool.request(
            method=request.method,
            url=request.url.to_string(),
            body=request.buffer,
            headers=request.headers,
            timeout=wait_cfg.timeout_urllib3,
            preload_content=False,
            decode_content=False,
            redirect=True,
        )

        received_at_timestamp = time.time_ns() // 1000

        result = HTTPResponse.from_urllib3(
            request=request,
            response=first_resp,
            tags=None,
            received_at_timestamp=received_at_timestamp,
        )

        result.drain_urllib3(first_resp, stream=True, release_conn=True)

        x_current_page, x_total_pages = (
            first_resp.headers.get("X-Current-Page"),
            first_resp.headers.get("X-Last-Page"),
        )

        if x_current_page and x_total_pages:
            result = self._combine_paginated_pages(
                result=result,
                request=request,
                current_page=int(x_current_page),
                total_pages=int(x_total_pages),
                wait_cfg=wait_cfg,
                stream=stream,
                raise_error=raise_error,
            )

        if raise_error:
            result.raise_for_status()

        if cache is not None and result.ok:
            batch = result.anonymize(mode=anonymize).to_arrow_batch(parse=False)
            cache.insert(
                batch,
                mode=SaveMode.APPEND,
                match_by=cache_by,
                wait=wait_cache,
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
        """Fetch a single pagination page and return ``(page_num, response)``.

        Called concurrently by :meth:`_combine_paginated_pages` for every
        page after the first.  The page number is injected as a ``page``
        query-string parameter, overriding any existing value.

        Parameters
        ----------
        request:
            The original (first-page) prepared request.  Used as a template;
            the URL is updated with the new page number and the body is
            re-seeded from *body_seed*.
        page_num:
            1-based page index to fetch.
        body_seed:
            Raw bytes of the original request body, or ``None`` if there was
            no body.  Re-wrapped in a :class:`~yggdrasil.io.buffer.BytesIO`
            for each page request so that the body can be sent multiple times.
        wait_cfg:
            Resolved :class:`~yggdrasil.dataclasses.waiting.WaitingConfig`
            used for the urllib3 timeout.
        stream:
            Whether to stream the page response body lazily.
        raise_error:
            Whether to raise on a non-2xx page response.

        Returns
        -------
        tuple[int, HTTPResponse]
            ``(page_num, response)`` — the page number is included so that
            :meth:`_combine_paginated_pages` can sort frames if needed.

        Raises
        ------
        ResponseError
            If *raise_error* is ``True`` and the page response is not 2xx.
        """
        page_url = request.url.add_query_item("page", str(page_num), replace=True)

        page_request = request.copy(
            url=page_url,
            buffer=BytesIO(body_seed) if body_seed is not None else None,
        )

        raw_resp = self.http_pool.request(
            method=page_request.method,
            url=page_url.to_string(),
            body=page_request.buffer,
            headers=page_request.headers,
            timeout=wait_cfg.timeout_urllib3,
            preload_content=not stream,
            decode_content=False,
            redirect=True,
        )

        received_at_timestamp = time.time_ns() // 1000

        page_result = HTTPResponse.from_urllib3(
            request=page_request,
            response=raw_resp,
            tags=None,
            received_at_timestamp=received_at_timestamp,
        )
        page_result.drain_urllib3(raw_resp, stream=stream, release_conn=True)

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
        """Fetch all remaining pages concurrently and merge them into *result*.

        Called by :meth:`send` when the server returns ``X-Current-Page`` and
        ``X-Last-Page`` headers.  Pages after *current_page* are fetched in
        parallel using the job pool, then concatenated (via Polars
        ``diagonal_relaxed`` join) and written back into *result*'s buffer as
        Arrow IPC.

        Parameters
        ----------
        result:
            The first-page :class:`HTTPResponse` that triggered pagination
            detection.  Its buffer is **replaced in-place** with the merged
            Arrow IPC payload.
        request:
            The original prepared request, used as a template for subsequent
            page requests.
        current_page:
            The page number of *result* (typically 1).
        total_pages:
            Total number of pages reported by the server.
        wait_cfg:
            Resolved :class:`~yggdrasil.dataclasses.waiting.WaitingConfig`
            forwarded to each page fetch.
        stream:
            Passed through to :meth:`_fetch_paginated_page`.
        raise_error:
            Passed through to :meth:`_fetch_paginated_page`.
        pool:
            A :class:`~yggdrasil.concurrent.threading.JobPoolExecutor` or an
            integer max-worker count.  ``None`` / ``0`` creates a fresh pool
            scoped to this call.

        Returns
        -------
        HTTPResponse
            *result*, mutated in-place with the merged multi-page body.
            Tags are updated with ``start_page`` and ``total_pages`` metadata.
        """
        if not isinstance(pool, JobPoolExecutor):
            with JobPoolExecutor.parse(pool) as pool:
                return self._combine_paginated_pages(
                    result=result,
                    request=request,
                    current_page=current_page,
                    total_pages=total_pages,
                    wait_cfg=wait_cfg,
                    stream=stream,
                    raise_error=raise_error,
                    pool=pool,
                )
        else:
            from yggdrasil.polars.lib import polars as pl

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
                page_num, page_resp = job_result.result
                frames.append(page_resp.to_polars(parse=True, lazy=False))

            final_df = pl.concat(frames, how="diagonal_relaxed", rechunk=True)

            result.buffer.truncate(size=0)

            arr = final_df.to_arrow(compat_level=pl.CompatLevel.newest())
            mt = MediaType(MimeType.ARROW_IPC)
            mio = result.buffer.media_io(mt)
            mio.write_arrow_table(arr)
            result.set_media_type(mt, safe=False)

            result.update_tags({
                "start_page": str(current_page),
                "total_pages": str(total_pages),
            })

            return result
