from __future__ import annotations

import base64
import datetime as dt
import itertools
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Literal, Iterator, Any

import urllib3
from urllib3 import BaseHTTPResponse
from urllib3.exceptions import (
    HTTPError,
    MaxRetryError,
    TimeoutError,
    ConnectTimeoutError,
    ReadTimeoutError,
    NewConnectionError,
    ProtocolError,
    SSLError,
)

import yggdrasil.arrow as pa
from yggdrasil.concurrent.threading import JobPoolExecutor, Job
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from .response import HTTPResponse
from ..enums import SaveMode
from ..request import PreparedRequest
from ..response import RESPONSE_ARROW_SCHEMA
from ..session import Session
from ...data import any_to_datetime

if TYPE_CHECKING:
    from ...databricks.sql.table import Table

__all__ = ["HTTPSession"]

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
RETRYABLE_EXC = (
    TimeoutError,
    ConnectTimeoutError,
    ReadTimeoutError,
    NewConnectionError,
    ProtocolError,
    SSLError,
    MaxRetryError,
    HTTPError,
)


@dataclass
class HTTPSession(Session):
    pool_connections: int = 8  # number of hosts to keep pools for
    pool_maxsize: int = 8  # max connections per host
    pool_block: bool = False  # block when pool is full instead of discarding

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
        state.pop("_http_pool", None)   # ← also drop the pool; it holds sockets
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

    @staticmethod
    def _sql_match_clause(
        request: PreparedRequest,
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
    ):
        def fmt(key: str, op: str, value: Any) -> str:
            if value is None:
                return f"{key} is null"

            if isinstance(value, bytes):
                value = base64.b64encode(value).decode("ascii")

            return f"{key} {op} '{value}'"

        def to_utc_epoch_us(x: dt.datetime | dt.date | str) -> int:
            """
            Convert input to UTC epoch microseconds.

            - dt.datetime with tzinfo: converted to UTC
            - dt.datetime naive: treated as UTC (no implicit local shift)
            - dt.date: promoted to midnight UTC
            - str: parsed by any_to_datetime (string parser may attach CURRENT_TZINFO),
                   then converted to UTC
            """
            if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
                v = dt.datetime(x.year, x.month, x.day, tzinfo=dt.timezone.utc)
            else:
                v = any_to_datetime(x)

            if v.tzinfo is None:
                v = v.replace(tzinfo=dt.timezone.utc)

            v = v.astimezone(dt.timezone.utc)
            # epoch microseconds
            return int(v.timestamp() * 1_000_000)

        clauses: list[str] = [
            fmt("request_url.host", "=", request.url.host),
            fmt("request_url.path", "=", request.url.path),
            fmt("request_url.query", "=", request.url.query),
            fmt("request_body_hash", "=", request.body.blake3().digest() if request.body else None),
        ]

        # ✅ filter on epoch micros field
        if received_from is not None and received_from != "":
            clauses.append(f"response_received_at_epoch >= {to_utc_epoch_us(received_from)}")
        if received_to is not None and received_to != "":
            clauses.append(f"response_received_at_epoch <= {to_utc_epoch_us(received_to)}")

        return " AND ".join(clauses)

    def send(
        self,
        request: PreparedRequest,
        *,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        sniff: Optional[bool] = None,
        stream: bool = True,
        cache: Optional["Table"] = None,
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
        anonymize: Literal["remove", "redact", "hash"] = "remove",
    ) -> HTTPResponse:
        """
        Implementation of the abstract send method.
        Handles the actual urllib3 network call and custom retry logic.
        """
        if sniff is None:
            sniff = cache is not None

        if cache is not None:
            anon = request.anonymize(mode=anonymize)
            query = f"""select * from {cache.full_name(safe=True)}
where {self._sql_match_clause(anon, received_from=received_from, received_to=received_to)}"""

            try:
                sql_cache_statement = cache.sql(query)
            except Exception as e:
                if "TABLE_OR_VIEW_NOT_FOUND" in str(e):
                    cache.create(
                        RESPONSE_ARROW_SCHEMA,
                        if_not_exists=True
                    )
                    sql_cache_statement = cache.sql(query)
                else:
                    raise

            responses = list(HTTPResponse.from_arrow(sql_cache_statement.to_arrow_table()))

            if responses:
                return responses[-1]

        http_pool = self.http_pool
        wait_cfg = self.waiting if wait is None else WaitingConfig.check_arg(wait)

        start = time.time()
        last_exc: Exception | None = None
        num_tries = max(wait_cfg.retries, 0) + 1
        result: HTTPResponse = None

        for iteration in range(num_tries):
            resp: BaseHTTPResponse | None = None
            request = request.prepare_to_send(sniff=sniff)

            try:
                resp = http_pool.request(
                    method=request.method,
                    url=request.url.to_string(),
                    body=request.buffer,
                    headers=request.headers,
                    timeout=wait_cfg.timeout_urllib3,
                    preload_content=not stream,
                    decode_content=False,
                    redirect=True,
                )

                received_at_timestamp = time.time_ns() // 1000 if sniff else 0

                # Custom handling for retryable status codes
                if resp.status in RETRYABLE_STATUS:
                    resp.release_conn()
                    wait_cfg.sleep(iteration=iteration, start=start)
                    continue

                result = HTTPResponse.from_urllib3(
                    request=request,
                    response=resp,
                    stream=stream,
                    tags=None,
                    received_at_timestamp=received_at_timestamp
                )
                break

            except RETRYABLE_EXC as e:
                last_exc = e
                if resp is not None:
                    resp.release_conn()

                # If this was our last attempt, raise the error
                if iteration >= (num_tries - 1):
                    raise

                wait_cfg.sleep(iteration=iteration, start=start)
                continue

        if result is None:
            if raise_error:
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("Retry loop exited unexpectedly")
            return None

        if raise_error:
            result.raise_for_status()

        if cache is not None and result.ok:
            # insert
            batch = result.anonymize(mode=anonymize).to_arrow_batch(parse=False)
            cache.insert(
                batch,
                mode=SaveMode.AUTO,
                match_by=["request_url.host", "request_url.path", "request_url.query", "request_body_hash"]
            )
        return result

    def send_many(
        self,
        requests: Iterator[PreparedRequest],
        *,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        sniff: Optional[bool] = None,
        stream: bool = True,
        cache: Optional["Table"] = None,
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
        wait_cache: WaitingConfigArg = False,
        pool: Optional[JobPoolExecutor, int] = None,
        batch_size: Optional[int] = None,
        ordered: bool = False,
        max_in_flight: Optional[int] = None,
        cancel_on_exit: bool = False,
        shutdown_on_exit: bool = False,
        shutdown_wait: bool = False,
    ) -> Iterator[HTTPResponse]:
        if sniff is None:
            sniff = cache is not None

        # Default worker count to pool_maxsize so the thread pool and the
        # connection pool are naturally matched — no thread ever waits on a
        # connection slot and no connection slot goes unused.
        if pool is None:
            pool = self.pool_maxsize

        with JobPoolExecutor.parse(pool) as pool:
            if not batch_size:
                batch_size: int = pool.max_workers * 100

            if cache is None:
                def jobs():
                    for req in requests:
                        yield Job.make(
                            self.send,
                            req,
                            wait=wait,
                            raise_error=raise_error,
                            sniff=sniff,
                            stream=stream
                        )

                for result in pool.as_completed(
                    jobs(),
                    ordered=ordered,
                    max_in_flight=max_in_flight,
                    cancel_on_exit=True,
                    shutdown_on_exit=True,
                    raise_error=True
                ):
                    resp = result.result

                    if raise_error:
                        resp.raise_for_status()
                        yield resp
                    elif resp.ok:
                        yield resp
            else:
                def _batched(it: Iterator, n: int) -> Iterator[list]:
                    it = iter(it)
                    while True:
                        batch = list(itertools.islice(it, n))
                        if not batch:
                            break
                        yield batch

                for batch in _batched(requests, batch_size):
                    # ── Cache lookup for the entire batch ────────────────────────
                    anon_batch = [req.anonymize(mode="remove") for req in batch]

                    clauses = " OR ".join(
                        f"({self._sql_match_clause(a, received_from=received_from, received_to=received_to)})" for a in anon_batch
                    )
                    query = f"SELECT * FROM {cache.full_name(safe=True)} WHERE {clauses}"

                    try:
                        sql_cache_statement = cache.sql(query)
                    except Exception as e:
                        if "TABLE_OR_VIEW_NOT_FOUND" in str(e):
                            cache.create(
                                RESPONSE_ARROW_SCHEMA,
                                if_not_exists=True
                            )
                            sql_cache_statement = cache.sql(query)
                        else:
                            raise

                    arrow_batch = sql_cache_statement.to_arrow_table()
                    cached_responses = list(HTTPResponse.from_arrow(arrow_batch))

                    # Build a lookup: (host, path, query, body_hash) → latest response
                    def _cache_key(r: HTTPResponse) -> tuple:
                        return (
                            r.request.url.host,
                            r.request.url.path,
                            r.request.url.query,
                            r.request.body.blake3().digest() if r.request.body else None,
                        )

                    def _request_key(req: PreparedRequest) -> tuple:
                        anon = req.anonymize(mode="remove")
                        return (
                            anon.url.host,
                            anon.url.path,
                            anon.url.query,
                            req.body.blake3().digest() if req.body else None,
                        )

                    cache_map: dict[tuple, HTTPResponse] = {}
                    for r in cached_responses:
                        cache_map[_cache_key(r)] = r  # last-write-wins → most recent row

                    # ── Partition: hits vs misses ─────────────────────────────────
                    hits: list[HTTPResponse] = []
                    misses: list[PreparedRequest] = []

                    for req in batch:
                        key = _request_key(req)
                        if key in cache_map:
                            hits.append(cache_map[key])
                        else:
                            misses.append(req)

                    # ── Yield cached hits immediately ─────────────────────────────
                    # (respect ordered=False; hits come back as-is)
                    for resp in hits:
                        yield resp

                    if not misses:
                        continue

                    # ── Fire network requests for misses ─────────────────────────
                    def miss_jobs():
                        for req in misses:
                            yield Job.make(
                                self.send,
                                req,
                                wait=wait,
                                raise_error=raise_error,
                                sniff=sniff,
                                stream=stream,
                                cache=None,  # avoid double-caching inside send()
                            )

                    to_insert: list[HTTPResponse] = []
                    failed: list[HTTPResponse] = []

                    for result in pool.as_completed(
                        miss_jobs(),
                        ordered=ordered,
                        max_in_flight=max_in_flight,
                        cancel_on_exit=True,
                        shutdown_on_exit=False,  # pool is shared across batches
                        raise_error=True,
                    ):
                        resp = result.result

                        if resp.ok:
                            to_insert.append(resp)
                            yield resp
                        else:
                            failed.append(resp)

                    # ── Bulk insert successful responses ─────────────────────────
                    if to_insert:
                        batches = [
                            r.anonymize(mode="remove").to_arrow_batch(parse=False)
                            for r in to_insert
                        ]
                        combined = pa.Table.from_batches(batches).combine_chunks()
                        cache.insert(
                            combined,
                            mode=SaveMode.APPEND,
                            match_by=[
                                "request_url_host",
                                "request_url_path",
                                "request_url_query",
                                "request_body_hash",
                            ],
                            wait=wait_cache
                        )

                    if raise_error and failed:
                        failed[-1].raise_for_status()
