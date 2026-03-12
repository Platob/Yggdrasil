from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Literal

import urllib3

from yggdrasil.concurrent.threading import JobPoolExecutor, Job
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.io import MediaType, MimeType, BytesIO
from .response import HTTPResponse
from ..enums import SaveMode
from ..request import PreparedRequest
from ..response import RESPONSE_ARROW_SCHEMA
from ..session import Session

if TYPE_CHECKING:
    from yggdrasil.databricks.sql.table import Table

__all__ = ["HTTPSession"]


@dataclass
class HTTPSession(Session):
    pool_connections: int = 8
    pool_maxsize: int = 8
    pool_block: bool = False

    _http_pool: urllib3.PoolManager = field(default=None, init=False, repr=False, compare=False)

    def _build_pool(self) -> urllib3.PoolManager:
        num_tries = max(self.waiting.retries, 0) + 1
        retries = urllib3.Retry(
            total=num_tries * 2,
            connect=num_tries,
            read=num_tries,
            backoff_factor=self.waiting.backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            raise_on_status=False,
        )
        return urllib3.PoolManager(
            num_pools=self.pool_connections,
            maxsize=self.pool_maxsize,
            block=self.pool_block,
            retries=retries,
            cert_reqs="CERT_REQUIRED" if self.verify else "CERT_NONE",
            ca_certs=None,
        )

    def __post_init__(self):
        if self._http_pool is None:
            self._http_pool = self._build_pool()

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.pop("_http", None)
        state.pop("_http_pool", None)
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self._http_pool = self._build_pool()

    @property
    def http_pool(self):
        if self._http_pool is None:
            with self._lock:
                if self._http_pool is None:
                    self._http_pool = self._build_pool()
        return self._http_pool

    def send(
        self,
        request: PreparedRequest,
        *,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        cache: Optional["Table"] = None,
        cache_by: Optional[list[str]] = None,
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
        anonymize: Literal["remove", "redact"] = "remove",
        wait_cache: WaitingConfigArg = False,
        pool: Optional[JobPoolExecutor | int] = None,
    ) -> HTTPResponse:
        if cache is not None:
            cache_by = self._cache_by_keys(cache_by)
            cache_request_by = [_ for _ in cache_by if _.startswith("request")]

            anon = request.anonymize(mode=anonymize)
            query = f"""select * from {cache.full_name(safe=True)}
where {self._sql_match_clause(
    anon,
    keys=cache_request_by,
    received_from=received_from,
    received_to=received_to,
)}
order by response_received_at_epoch desc
limit 1"""

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
            headers=self.send_headers
        )
        http_pool = self.http_pool
        wait_cfg = self.waiting if wait is None else WaitingConfig.check_arg(wait)

        first_resp = http_pool.request(
            method=request.method,
            url=request.url.to_string(),
            body=request.buffer,
            headers=request.headers,
            timeout=wait_cfg.timeout_urllib3,
            preload_content=not stream,
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

        result.drain_urllib3(first_resp, stream=stream, release_conn=True)

        x_current_page, x_total_pages = (
            first_resp.headers.get("X-Current-Page"),
            first_resp.headers.get("X-Last-Page")
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
                pool=pool,
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

            mt = MediaType(MimeType.ARROW_IPC)
            mio = result.buffer.media_io(mt)
            mio.write_arrow_table(
                final_df.to_arrow(compat_level=pl.CompatLevel.newest())
            )

            result.update_tags({
                "start_page": str(current_page),
                "total_pages": str(total_pages),
            })

            return result
