from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Mapping, Union, Any, Literal, Iterator

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
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.io.response import FULL_ARROW_SCHEMA

from .response import HTTPResponse
from ..buffer import BytesIO
from ..enums import SaveMode
from ..request import PreparedRequest
from ..session import Session
from ..url import URL
from yggdrasil.concurrent.threading import JobPoolExecutor

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
    _http_pool: urllib3.PoolManager = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self._http_pool is None:
            # Use waiting config to define retry policy for the pool
            num_tries = max(self.waiting.retries, 0) + 1

            retries = urllib3.Retry(
                total=num_tries * 2,
                connect=num_tries,
                read=num_tries,
                backoff_factor=self.waiting.backoff,
                status_forcelist=(429, 500, 502, 503, 504),
                raise_on_status=False,
            )

            self._http_pool = urllib3.PoolManager(
                retries=retries,
                cert_reqs="CERT_REQUIRED" if self.verify else "CERT_NONE",
                ca_certs=None,
            )

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        # Ensure thread locks and pool managers aren't pickled
        state.pop("_http", None)
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)

        # Re-initialize the lock; _http will be lazily created in pool_manager()
        # Use waiting config to define retry policy for the pool
        num_tries = max(self.waiting.retries, 0) + 1

        retries = urllib3.Retry(
            total=num_tries * 2,
            connect=num_tries,
            read=num_tries,
            backoff_factor=self.waiting.backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            raise_on_status=False,
        )

        self._http_pool = urllib3.PoolManager(
            retries=retries,
            cert_reqs="CERT_REQUIRED" if self.verify else "CERT_NONE",
            ca_certs=None,
        )

    def request(
        self,
        method: str,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        add_statistics: Optional[bool] = None,
        wait: Optional[WaitingConfigArg] = None,
        normalize: bool = True,
        cache: Optional["Table"] = None
    ) -> HTTPResponse:
        if add_statistics is None:
            add_statistics = cache is not None

        request = self.prepare_request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            body=body,
            json=json,
            normalize=normalize,
        )

        return self.send(
            request=request,
            add_statistics=add_statistics,
            stream=stream,
            wait=wait,
            cache=cache
        )

    @staticmethod
    def _sql_match_clause(request: PreparedRequest):
        def fmt(key, value):
            if value is None:
                return f"{key} is null"
            if isinstance(value, bytes):
                value = base64.b64encode(value).decode("ascii")
            return f"{key} = '{value}'"

        return " AND ".join(
            fmt(k, v)
            for k, v in (
                ("request_url_host", request.url.host),
                ("request_url_path", request.url.path),
                ("request_url_query", request.url.query),
                ("request_body_hash", request.body.blake3().digest() if request.body else None),
            )
        )

    def send(
        self,
        request: PreparedRequest,
        *,
        add_statistics: Optional[bool] = None,
        stream: bool = True,
        wait: Optional[WaitingConfigArg] = None,
        cache: Optional[Union["Table", bool]] = None,
        anonymize: Literal["remove", "redact", "hash"] = "remove",
    ) -> HTTPResponse:
        """
        Implementation of the abstract send method.
        Handles the actual urllib3 network call and custom retry logic.
        """
        if add_statistics is None:
            add_statistics = cache is not None

        if cache is not None:
            anon = request.anonymize(mode=anonymize)
            query = f"""select * from {cache.full_name(safe=True)}
where {self._sql_match_clause(anon)}"""

            batch = cache.sql(query).to_arrow_table()
            responses = list(HTTPResponse.from_arrow_batch(batch))

            if responses:
                return responses[-1]

        http_pool = self._http_pool
        wait_cfg = self.waiting if wait is None else WaitingConfig.check_arg(wait)

        start = time.time()
        last_exc: Exception | None = None
        num_tries = max(wait_cfg.retries, 0) + 1
        result: HTTPResponse = None

        for iteration in range(num_tries):
            resp: BaseHTTPResponse | None = None
            request.sent_at_timestamp = time.time_ns() // 1000 if add_statistics else 0

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

                received_at_timestamp = time.time_ns() // 1000 if add_statistics else 0

                # Custom handling for retryable status codes
                if resp.status in RETRYABLE_STATUS:
                    resp.release_conn()
                    wait_cfg.sleep(iteration=iteration, start=start)
                    continue

                result = HTTPResponse.from_urllib3(
                    request=request,
                    response=resp,
                    stream=stream,
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
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("Retry loop exited unexpectedly")

        if cache is not None and result.ok:
            # insert
            batch = result.anonymize(mode=anonymize).to_arrow_batch(parse=False)
            cache.insert(
                batch,
                mode=SaveMode.AUTO,
                match_by=["request_url_host", "request_url_path", "request_url_query", "request_body_hash"]
            )
        return result

    def send_many(
        self,
        requests: Iterator[PreparedRequest],
        *,
        add_statistics: Optional[bool] = None,
        stream: bool = True,
        wait: Optional[WaitingConfigArg] = None,
        cache: Optional["Table"] = None,
        ordered: bool = False,
        pool: Optional[JobPoolExecutor | int] = None
    ):
        raise NotImplementedError
