"""Concrete HTTP/HTTPS session — the single public entry point of :mod:`yggdrasil.http_`.

Construct one :class:`HTTPSession` per host (singleton-cached by config), drive
verb methods (``get`` / ``post`` / ``put`` / ``patch`` / ``delete`` / ``head``
/ ``options`` / ``request``) inherited from
:class:`yggdrasil.io.session.Session`, and read the returned
:class:`HTTPResponse`. The pool, retry, and timeout primitives live in
:mod:`yggdrasil.http_._pool` (stdlib-backed, urllib3-shaped) — feature code
should not import them directly.
"""
from __future__ import annotations

import datetime as dt
import logging
from itertools import takewhile
from typing import Any, Optional

from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.dataclasses.waiting import DEFAULT_WAITING_CONFIG
from yggdrasil.data.enums import MediaTypes
from yggdrasil.io.authorization.base import Authorization
from yggdrasil.io.memory import Memory
from yggdrasil.io.primitive import ArrowIPCFile
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.send_config import SendConfig
from yggdrasil.io.session import Session
from yggdrasil.io.url import URL

from ._pool import PoolManager, Retry
from .response import HTTPResponse

__all__ = ["HTTPSession"]


LOGGER = logging.getLogger(__name__)

# Backoff tuning. 429s still get a longer schedule than 5xx because rate
# limits need wall-clock time to clear, but both schedules are tight:
# we'd rather surface an error fast than mask a real outage with a
# minute-long retry storm. Server-supplied Retry-After always wins over
# the 429 schedule when present.
_RETRY_TOTAL = 3
_RETRY_CONNECT = 2
_RETRY_READ = 2

# 5xx schedule: 0.5, 1, 2 (capped at backoff_max). Worst-case ~3.5s.
_BACKOFF_5XX_FACTOR = 0.5
_BACKOFF_5XX_MAX = 5.0

# 429 schedule: 1, 2, 4 (capped at backoff_max). Worst-case ~7s.
# Server-supplied Retry-After always wins over this when present.
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
    """HTTP/HTTPS session backed by :class:`yggdrasil.http_._pool.PoolManager`.

    Inherits the verb methods (``get`` / ``post`` / ``put`` / ``patch`` /
    ``delete`` / ``head`` / ``options`` / ``request``) from
    :class:`~yggdrasil.io.session.Session`; URL resolution and query-param
    handling live on :class:`~yggdrasil.io.url.URL` and
    :meth:`Session.prepare_request`. Per-request headers are
    :attr:`headers` (session default) merged with the per-call ``headers=``
    kwarg. No User-Agent generator, cookie jar, or browser-emulation
    layering is built in.
    """

    def __init__(
        self,
        base_url: Optional[URL | str] = None,
        verify: bool = True,
        pool_maxsize: int = 10,
        headers: Optional[dict[str, str]] = None,
        waiting: WaitingConfig = DEFAULT_WAITING_CONFIG,
        *,
        auth: Optional[Authorization] = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        # The pool caps idle sockets per host; 8 is plenty for our typical
        # workloads. Clamping here means the singleton key (built from
        # ``pool_maxsize``) collapses ``HTTPSession(pool_maxsize=20)`` and
        # ``HTTPSession()`` to one instance the way they always did.
        pool_maxsize = min(8, int(pool_maxsize)) if pool_maxsize else 8
        super().__init__(
            base_url=base_url,
            verify=verify,
            pool_maxsize=pool_maxsize,
            headers=headers,
            waiting=waiting,
            auth=auth,
        )
        # Connection pool is built lazily on first :attr:`http_pool`
        # access. Keeping ``__init__`` side-effect-free lets the
        # singleton-key probe (see :meth:`Session._singleton_key`) run
        # the constructor without opening sockets.
        self._http_pool: Optional[PoolManager] = None

    _TRANSIENT_STATE_ATTRS = Session._TRANSIENT_STATE_ATTRS | {"_http_pool"}


    def _build_retry(self) -> Retry:
        """Build the :class:`Retry` policy used by the connection pool.

        Subclasses can override to swap the policy entirely, or call
        ``super()._build_retry().new(...)`` to tweak a single field.
        """
        kwargs: dict = dict(
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
        return _TieredRetry(**kwargs)

    def _build_http_pool(self) -> PoolManager:
        return PoolManager(
            num_pools=self.pool_maxsize,
            maxsize=self.pool_maxsize,
            block=True,
            retries=self._build_retry(),
            cert_reqs="CERT_REQUIRED" if self.verify else "CERT_NONE",
            ca_certs=None,
        )

    @property
    def http_pool(self):
        if self._http_pool is None:
            with self._lock:
                if self._http_pool is None:
                    self._http_pool = self._build_http_pool()
        return self._http_pool

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _local_send(
        self,
        request: PreparedRequest,
        config: SendConfig,
    ) -> HTTPResponse:
        wait_cfg = self.waiting if config.wait is None else config.wait

        raw_resp, result = self._wire_send(request, wait_cfg)

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
                raw_resp, result = self._wire_send(request, wait_cfg)

        x_current_page = raw_resp.headers.get("X-Current-Page")
        x_total_pages = raw_resp.headers.get("X-Last-Page")

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
    ) -> tuple[Any, HTTPResponse]:
        """Single wire-level send.

        Returns the raw pool response (kept around so the caller can read
        pagination headers like ``X-Current-Page`` without a second round
        trip) alongside the drained :class:`HTTPResponse`. Extracted from
        :meth:`_local_send` so the 403-retry branch re-uses the exact same
        transport call.
        """
        raw_resp = self.http_pool.request(
            method=request.method,
            url=request.url.to_string(),
            body=request.buffer.to_bytes() if request.buffer is not None else None,
            headers=request.headers,
            timeout=wait_cfg.timeout_pool,
            preload_content=False,
            decode_content=False,
            redirect=True,
        )
        result = HTTPResponse.from_pool(
            request=request,
            response=raw_resp,
            tags=None,
            received_at=dt.datetime.now(dt.timezone.utc),
            stream=True,
            release_conn=True,
        )
        return raw_resp, result

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

        raw_resp = self.http_pool.request(
            method=page_request.method,
            url=page_url.to_string(),
            body=page_request.buffer.to_bytes() if page_request.buffer is not None else None,
            headers=page_request.headers,
            timeout=wait_cfg.timeout_pool,
            preload_content=not stream,
            decode_content=False,
            redirect=True,
        )

        page_result = HTTPResponse.from_pool(
            request=page_request,
            response=raw_resp,
            tags=None,
            received_at=dt.datetime.now(tz=dt.timezone.utc),
            stream=stream,
            release_conn=True,
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
            with JobPoolExecutor.parse(pool) as parsed_pool:
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

        final_df = pl.concat(frames, how="diagonal_relaxed", rechunk=True)

        new_holder = Memory()
        new_holder.media_type = MediaTypes.ARROW_IPC
        with ArrowIPCFile(holder=new_holder, owns_holder=False, mode="wb") as new_buffer:
            new_buffer.write_arrow_table(
                final_df.to_arrow(compat_level=pl.CompatLevel.newest()),
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
