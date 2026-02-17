from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

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

from .response import HTTPResponse
from ..request import PreparedRequest
from ..session import Session
from ...pyutils.waiting_config import WaitingConfig, WaitingConfigArg

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

    def send(
        self,
        request: PreparedRequest,
        *,
        add_statistics: bool = False,
        stream: bool = True,
        wait: Optional[WaitingConfigArg] = None
    ) -> HTTPResponse:
        """
        Implementation of the abstract send method.
        Handles the actual urllib3 network call and custom retry logic.
        """
        http_pool = self._http_pool
        wait_cfg = self.waiting if wait is None else WaitingConfig.check_arg(wait)

        start = time.time()
        last_exc: Exception | None = None
        num_tries = max(wait_cfg.retries, 0) + 1

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

                return HTTPResponse.from_urllib3(
                    request=request,
                    response=resp,
                    stream=stream,
                    received_at_timestamp=received_at_timestamp
                )

            except RETRYABLE_EXC as e:
                last_exc = e
                if resp is not None:
                    resp.release_conn()

                # If this was our last attempt, raise the error
                if iteration >= (num_tries - 1):
                    raise

                wait_cfg.sleep(iteration=iteration, start=start)
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Retry loop exited unexpectedly")