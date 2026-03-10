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


def to_utc_epoch_us(x: dt.datetime | dt.date | str) -> int:
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        v = dt.datetime(x.year, x.month, x.day, tzinfo=dt.timezone.utc)
    else:
        v = any_to_datetime(x)

    if v.tzinfo is None:
        v = v.replace(tzinfo=dt.timezone.utc)

    v = v.astimezone(dt.timezone.utc)
    return int(v.timestamp() * 1_000_000)


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

    @staticmethod
    def _cache_by_keys(arg: Optional[list[str]] = None) -> list[str]:
        if not arg:
            arg = [
                "request_url_host",
                "request_url_path",
                "request_url_query",
                "request_body_hash",
                "response_body_hash"
            ]

        invalid = [key for key in arg if key not in RESPONSE_ARROW_SCHEMA.names]
        if invalid:
            raise ValueError(
                f"Invalid cache_by key(s): {invalid}, must be within {RESPONSE_ARROW_SCHEMA.names}"
            )

        return arg

    @staticmethod
    def _cache_value_from_request(request: PreparedRequest, key: str) -> Any:
        if key == "request_method":
            return request.method
        if key == "request_url":
            return request.url.to_string()
        if key == "request_url_scheme":
            return request.url.scheme
        if key == "request_url_host":
            return request.url.host
        if key == "request_url_port":
            return request.url.port
        if key == "request_url_path":
            return request.url.path
        if key == "request_url_query":
            return request.url.query
        if key == "request_body_hash":
            return request.body.blake3().digest() if request.body else None

        # fallback for future extension if PreparedRequest exposes same-name attrs
        if hasattr(request, key):
            return getattr(request, key)

        raise ValueError(f"Unsupported request cache_by key: {key}")

    @classmethod
    def _cache_values_from_request(
        cls,
        request: PreparedRequest,
        keys: list[str],
    ) -> dict[str, Any]:
        return {key: cls._cache_value_from_request(request, key) for key in keys}

    @staticmethod
    def _cache_value_from_response(response: HTTPResponse, key: str) -> Any:
        if hasattr(response, key):
            return getattr(response, key)

        raise ValueError(f"Unsupported response cache_by key: {key}")

    @classmethod
    def _cache_tuple_from_request(
        cls,
        request: PreparedRequest,
        keys: list[str],
    ) -> tuple:
        values = cls._cache_values_from_request(request, keys)
        return tuple(values[key] for key in keys)

    @classmethod
    def _cache_tuple_from_response(
        cls,
        response: HTTPResponse,
        keys: list[str],
    ) -> tuple:
        return tuple(cls._cache_value_from_response(response, key) for key in keys)

    @staticmethod
    def _sql_literal(value: Any) -> str:
        if value is None:
            return "null"

        if isinstance(value, bytes):
            value = base64.b64encode(value).decode("ascii")
        else:
            value = str(value)

        value = value.replace("'", "''")
        return f"'{value}'"

    @classmethod
    def _sql_match_clause(
        cls,
        request: PreparedRequest,
        keys: list[str],
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
    ) -> str:
        values = cls._cache_values_from_request(request, keys)
        clauses: list[str] = []

        for key in keys:
            value = values[key]
            if value is None:
                clauses.append(f"{key} IS NULL")
            else:
                clauses.append(f"{key} = {cls._sql_literal(value)}")

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
        normalize: Optional[bool] = None,
        stream: bool = True,
        cache: Optional["Table"] = None,
        cache_by: Optional[list[str]] = None,
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
        anonymize: Literal["remove", "redact", "hash"] = "remove",
        wait_cache: WaitingConfigArg = False,
    ) -> HTTPResponse:
        if normalize is None:
            normalize = cache is not None

        if cache is not None:
            cache_by = self._cache_by_keys(cache_by)

            anon = request.anonymize(mode=anonymize)
            query = f"""select * from {cache.full_name(safe=True)}
where {self._sql_match_clause(
    anon,
    keys=[_ for _ in cache_by if _.startswith("request")],
    received_from=received_from,
    received_to=received_to,
)}"""

            try:
                sql_cache_statement = cache.sql.execute(query)
            except Exception as e:
                if "TABLE_OR_VIEW_NOT_FOUND" in str(e):
                    cache.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                    sql_cache_statement = cache.sql.execute(query)
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
            request = request.prepare_to_send(normalize=normalize)

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

                received_at_timestamp = time.time_ns() // 1000 if normalize else 0

                if resp.status in RETRYABLE_STATUS:
                    resp.release_conn()
                    wait_cfg.sleep(iteration=iteration, start=start)
                    continue

                result = HTTPResponse.from_urllib3(
                    request=request,
                    response=resp,
                    stream=stream,
                    tags=None,
                    received_at_timestamp=received_at_timestamp,
                )
                break

            except RETRYABLE_EXC as e:
                last_exc = e
                if resp is not None:
                    resp.release_conn()

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
            batch = result.anonymize(mode=anonymize).to_arrow_batch(parse=False)
            cache.insert(
                batch,
                mode=SaveMode.APPEND,
                match_by=cache_by,
                wait=wait_cache,
            )

        return result

    def send_many(
        self,
        requests: Iterator[PreparedRequest],
        *,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: Optional[bool] = None,
        stream: bool = True,
        cache: Optional["Table"] = None,
        cache_by: Optional[list[str]] = None,
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
        anonymize: Literal["remove", "redact", "hash"] = "remove",
        wait_cache: WaitingConfigArg = False,
        # Pooling options
        pool: Optional[JobPoolExecutor | int] = None,
        batch_size: Optional[int] = None,
        ordered: bool = False,
        max_in_flight: Optional[int] = None,
        cancel_on_exit: bool = False,
        shutdown_on_exit: bool = False,
        shutdown_wait: bool = False,
    ) -> Iterator[HTTPResponse]:
        if normalize is None:
            normalize = cache is not None

        if pool is None:
            pool = self.pool_maxsize

        if cache is not None:
            cache_by = self._cache_by_keys(cache_by)
            cache_request_by = [_ for _ in cache_by if _.startswith("request")]
        else:
            cache_request_by = []

        with JobPoolExecutor.parse(pool) as pool:
            if not batch_size:
                batch_size = pool.max_workers * 100

            if cache is None:
                def jobs():
                    for req in requests:
                        yield Job.make(
                            self.send,
                            req,
                            wait=wait,
                            raise_error=raise_error,
                            normalize=normalize,
                            stream=stream,
                        )

                for result in pool.as_completed(
                    jobs(),
                    ordered=ordered,
                    max_in_flight=max_in_flight,
                    cancel_on_exit=True,
                    shutdown_on_exit=True,
                    raise_error=True,
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
                        b = list(itertools.islice(it, n))
                        if not b:
                            break
                        yield b

                for batch in _batched(requests, batch_size):
                    anon_batch = [req.anonymize(mode="remove") for req in batch]

                    clauses = " OR ".join(
                        f"({self._sql_match_clause(
                            req,
                            keys=cache_request_by,
                            received_from=received_from,
                            received_to=received_to,
                        )})"
                        for req in anon_batch
                    )
                    query = f"SELECT * FROM {cache.full_name(safe=True)} WHERE {clauses}"

                    try:
                        sql_cache_statement = cache.sql.execute(query)
                    except Exception as e:
                        if "TABLE_OR_VIEW_NOT_FOUND" in str(e):
                            cache.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                            sql_cache_statement = cache.sql.execute(query)
                        else:
                            raise

                    arrow_batch = sql_cache_statement.to_arrow_table()
                    cached_responses = list(HTTPResponse.from_arrow(arrow_batch))

                    cache_map: dict[tuple, HTTPResponse] = {}
                    for resp in cached_responses:
                        cache_map[self._cache_tuple_from_response(resp, cache_request_by)] = resp

                    hits: list[HTTPResponse] = []
                    misses: list[PreparedRequest] = []

                    for req in batch:
                        anon_req = req.anonymize(mode="remove")
                        key = self._cache_tuple_from_request(anon_req, cache_request_by)
                        if key in cache_map:
                            hits.append(cache_map[key])
                        else:
                            misses.append(req)

                    for resp in hits:
                        yield resp

                    if not misses:
                        continue

                    def miss_jobs():
                        for req in misses:
                            yield Job.make(
                                self.send,
                                req,
                                wait=wait,
                                raise_error=raise_error,
                                normalize=normalize,
                                stream=stream,
                                cache=None,
                            )

                    to_insert: list[HTTPResponse] = []
                    failed: list[HTTPResponse] = []

                    for result in pool.as_completed(
                        miss_jobs(),
                        ordered=ordered,
                        max_in_flight=max_in_flight,
                        cancel_on_exit=True,
                        shutdown_on_exit=False,
                        raise_error=True,
                    ):
                        resp = result.result

                        if resp.ok:
                            to_insert.append(resp)
                            yield resp
                        elif raise_error:
                            failed.append(resp)

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
