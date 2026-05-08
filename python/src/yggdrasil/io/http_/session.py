"""Concrete HTTP/HTTPS session backed by ``urllib3``."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import urllib3
from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.io import BytesIO
from yggdrasil.data.enums import MediaTypes

from .response import HTTPResponse
from yggdrasil.io.primitive import ArrowIPCFile
from ..request import PreparedRequest
from ..send_config import SendConfig
from ..session import Session

if TYPE_CHECKING:
    from .browser import BrowserHTTPSession

__all__ = ["HTTPSession"]


# Backoff tuning. 429s get a longer, gentler schedule than 5xx because rate
# limits often need real wall-clock time to clear; transient server errors
# usually resolve faster.
_RETRY_TOTAL = 6
_RETRY_CONNECT = 3
_RETRY_READ = 3

# 5xx schedule: 1, 2, 4, 8, 16, 32 (capped at backoff_max)
_BACKOFF_5XX_FACTOR = 1.0
_BACKOFF_5XX_MAX = 60.0

# 429 schedule: 4, 8, 16, 32, 64, 128 (capped at backoff_max)
# Server-supplied Retry-After always wins over this when present.
_BACKOFF_429_FACTOR = 4.0
_BACKOFF_429_MAX = 300.0

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class _TieredRetry(urllib3.Retry):
    """``urllib3.Retry`` variant with status-aware backoff.

    Standard ``Retry`` exposes a single ``backoff_factor`` shared by every
    retry, so 429 (rate limit) and 503 (transient outage) get the same
    schedule. This subclass branches on the most recent response status:

    * **429** uses a longer, gentler exponential schedule, since rate-limit
      windows are typically wall-clock bound and respond poorly to tight
      retries.
    * **Everything else** (5xx, transport errors) uses a shorter schedule.
    * The server's ``Retry-After`` header — when present and respected via
      ``respect_retry_after_header=True`` — always overrides this, because
      ``urllib3`` checks ``get_retry_after`` before ``get_backoff_time``.
    """

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


# Imported lazily-style at module level to keep _TieredRetry readable above.
from itertools import takewhile  # noqa: E402


@dataclass
class HTTPSession(Session):
    """Concrete HTTP/HTTPS session backed by a ``urllib3`` connection pool."""

    _http_pool: urllib3.PoolManager = field(default=None, init=False, repr=False, compare=False)

    def __getstate__(self):
        state = super().__getstate__()
        state["_http_pool"] = True

        return state

    def __setstate__(self, state):
        super().__setstate__(state)

        self._http_pool = state.get("_http_pool", None)
        if self._http_pool:
            self._http_pool = self._build_http_pool()


    def _build_retry(self) -> urllib3.Retry:
        """Build the :class:`urllib3.Retry` policy used by the connection pool.

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

    def _build_http_pool(self) -> urllib3.PoolManager:
        return urllib3.PoolManager(
            num_pools=self.pool_maxsize,
            maxsize=self.pool_maxsize,
            block=True,
            retries=self._build_retry(),
            cert_reqs="CERT_REQUIRED" if self.verify else "CERT_NONE",
            ca_certs=None,
        )

    def __post_init__(self):
        if self.pool_maxsize:
            self.pool_maxsize = min(8, int(self.pool_maxsize))
        else:
            self.pool_maxsize = 8

        super().__post_init__()

        if self._http_pool is None:
            self._http_pool = self._build_http_pool()

    @property
    def http_pool(self):
        if self._http_pool is None:
            with self._lock:
                if self._http_pool is None:
                    self._http_pool = self._build_http_pool()
        return self._http_pool

    # ------------------------------------------------------------------
    # Extensibility hooks
    # ------------------------------------------------------------------

    def _build_request_headers(
        self,
        request: PreparedRequest,
    ) -> Optional[dict[str, str]]:
        """Return the headers dict to merge into *request* before sending.

        Subclasses may override this to inject per-request headers without
        replacing the entire :attr:`send_headers` mapping.  The default
        implementation returns :attr:`send_headers` unchanged.
        """
        return self.send_headers

    def to_browser(
        self,
        *,
        user_agent: str | None = None,
        accept_language: str = "en-US,en;q=0.9",
        ua_seed: int | None = None,
    ) -> "BrowserHTTPSession":
        """Return a :class:`~yggdrasil.io.http_.browser.BrowserHTTPSession`
        that inherits this session's connection settings.

        Parameters
        ----------
        user_agent:
            Explicit ``User-Agent`` string.  ``None`` (default) = auto-generate.
        accept_language:
            ``Accept-Language`` header value for browser requests.
        ua_seed:
            Optional integer seed for deterministic user-agent generation.

        Returns
        -------
        BrowserHTTPSession
        """
        from .browser import BrowserHTTPSession

        return BrowserHTTPSession(
            base_url=self.base_url,
            verify=self.verify,
            pool_maxsize=self.pool_maxsize,
            send_headers=dict(self.send_headers) if self.send_headers else None,
            waiting=self.waiting,
            user_agent=user_agent,
            accept_language=accept_language,
            ua_seed=ua_seed,
        )

    def _local_send(
        self,
        request: PreparedRequest,
        config: SendConfig,
    ) -> HTTPResponse:
        wait_cfg = self.waiting if config.wait is None else config.wait

        request = request.prepare_to_send(
            sent_at=None,
            headers=self._build_request_headers(request),
        )

        raw_resp = self.http_pool.request(
            method=request.method,
            url=request.url.to_string(),
            body=request.buffer,
            headers=request.headers,
            timeout=wait_cfg.timeout_urllib3,
            preload_content=False,
            decode_content=False,
            redirect=True,
        )

        result = HTTPResponse.from_urllib3(
            request=request,
            response=raw_resp,
            tags=None,
            received_at=dt.datetime.now(dt.timezone.utc),
        )
        result.drain_urllib3(raw_resp, stream=True, release_conn=True)

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

        page_result = HTTPResponse.from_urllib3(
            request=page_request,
            response=raw_resp,
            tags=None,
            received_at=dt.datetime.now(tz=dt.timezone.utc),
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

        new_buffer = ArrowIPCFile(media_type=MediaTypes.ARROW_IPC)
        new_buffer.write_arrow_table(
            final_df.to_arrow(compat_level=pl.CompatLevel.newest()),
            compression="zstd",
        )
        new_buffer.seek(0)

        result.buffer.close()
        result.buffer = new_buffer
        result.set_media_type(MediaTypes.ARROW_IPC)

        result.update_tags({
            "page_start": str(current_page),
            "page_total": str(total_pages),
        })

        return result