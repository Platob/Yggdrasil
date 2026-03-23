"""HTTP session abstraction with transparent local and remote cache support."""

from __future__ import annotations

import itertools
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Optional

import pyarrow as pa

import yggdrasil.pickle.ser as pickle
from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.dataclasses import restore_dataclass_state, serialize_dataclass_state
from yggdrasil.dataclasses.waiting import (
    DEFAULT_WAITING_CONFIG,
    WaitingConfig,
    WaitingConfigArg,
)
from .buffer import BytesIO
from .request import PreparedRequest
from .response import RESPONSE_ARROW_SCHEMA, Response
from .send_config import CacheConfig, SendConfig, SendManyConfig
from .url import URL

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame

__all__ = ["Session", "CacheConfig", "SendConfig", "SendManyConfig"]


LOGGER = logging.getLogger(__name__)


@dataclass
class Session(ABC):
    base_url: Optional[URL] = None
    verify: bool = True
    pool_maxsize: int = 10
    send_headers: Optional[dict[str, str]] = field(default=None, repr=False)
    waiting: WaitingConfig = field(
        default=DEFAULT_WAITING_CONFIG,
        repr=False,
        compare=False,
        hash=False,
    )

    _lock: threading.RLock = field(default=None, init=False, repr=False, compare=False)
    _job_pool: JobPoolExecutor = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.base_url:
            self.base_url = URL.parse(self.base_url)
        if self._lock is None:
            self._lock = threading.RLock()
        if self.pool_maxsize <= 0:
            self.pool_maxsize = 8

    def __getstate__(self) -> dict[str, Any]:
        return serialize_dataclass_state(self)

    def __setstate__(self, state: dict[str, Any]) -> None:
        restore_dataclass_state(self, state)
        self.__post_init__()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._job_pool:
            self._job_pool.shutdown(wait=True)
            self._job_pool = None

    @property
    def job_pool(self) -> JobPoolExecutor:
        if self._job_pool is None:
            with self._lock:
                if self._job_pool is None:
                    self._job_pool = JobPoolExecutor(max_workers=self.pool_maxsize)
                    LOGGER.debug("Created job pool with max_workers=%s", self.pool_maxsize)
        return self._job_pool

    @property
    def x_api_key(self) -> Optional[str]:
        if self.send_headers:
            return self.send_headers.get("X-API-Key")
        return None

    @x_api_key.setter
    def x_api_key(self, value: Optional[str]) -> None:
        if self.send_headers is None:
            self.send_headers = {}
        if value is None:
            self.send_headers.pop("X-API-Key", None)
        else:
            self.send_headers["X-API-Key"] = value

    def _request_log_id(self, request: PreparedRequest) -> str:
        try:
            return request.xxh3_b64(url_safe=True)
        except Exception:
            return request.url.to_string()

    def _load_local_cached_response(
        self,
        request: PreparedRequest,
        cache_cfg: CacheConfig,
    ) -> tuple[Optional[Response], Optional[Any]]:
        filepath = cache_cfg.local_cache_file(
            request=request,
            suffix=".ypkl",
        )
        if filepath is None:
            return None, filepath

        if not filepath.exists():
            return None, filepath

        loaded = pickle.load(
            filepath,
            unpickle=True,
            clean_corrupted=True,
            default=None,
        )

        if not isinstance(loaded, Response):
            try:
                filepath.unlink(missing_ok=True)
            except:
                pass
            return None, filepath

        if not cache_cfg.filter_response(loaded, request=request):
            return None, filepath

        return loaded, filepath

    def _store_local_cached_response(
        self,
        request: PreparedRequest,
        response: Response,
        cache_cfg: CacheConfig,
        *,
        filepath=None,
    ) -> None:
        if not cache_cfg.local_cache_enabled or not response.ok:
            return

        target = filepath if filepath else cache_cfg.local_cache_file(
            request=request,
            suffix=".ypkl",
            force=True
        )

        Job.make(pickle.dump, response, target).fire_and_forget()

    def _load_remote_cached_response(
        self,
        request: PreparedRequest,
        cache_cfg: CacheConfig,
        *,
        spark_session: Optional["SparkSession"] = None,
    ) -> Optional[Response]:
        if not cache_cfg.remote_cache_enabled:
            return None

        query = cache_cfg.make_batch_lookup_sql(
            table_name=cache_cfg.table.full_name(safe=True),
            requests=[request.anonymize(mode=cache_cfg.anonymize)],
        )

        try:
            cache_result = cache_cfg.table.sql.execute(
                query,
                spark_session=spark_session,
            )
        except Exception as exc:
            if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                cache_cfg.table.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                cache_result = cache_cfg.table.sql.execute(
                    query,
                    spark_session=spark_session,
                )
            else:
                raise

        for response in Response.from_arrow_tabular(cache_result.to_arrow_batches()):
            if cache_cfg.filter_response(response, request=request):
                LOGGER.debug(
                    "Found match of %s in %s",
                    request,
                    cache_cfg.table
                )
                return response

        return None

    def _store_remote_cached_response(
        self,
        response: Response,
        cache_cfg: CacheConfig,
        *,
        spark_session: Optional["SparkSession"] = None,
    ) -> None:
        if not cache_cfg.remote_cache_enabled or not response.ok:
            return

        batch = response.anonymize(mode=cache_cfg.anonymize).to_arrow_batch(parse=False)
        cache_cfg.table.insert(
            batch,
            mode=cache_cfg.mode,
            match_by=cache_cfg.by or None,
            wait=cache_cfg.wait,
            spark_session=spark_session,
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
        parsed = URL.parse(url, normalize=normalize)

        if parsed.scheme.startswith("http"):
            from .http_ import HTTPSession

            return HTTPSession(
                base_url=parsed,
                verify=verify,
                waiting=WaitingConfig.check_arg(waiting) if waiting is not None else None,
            )

        raise ValueError(f"Cannot build session from scheme: {parsed.scheme!r}")

    def send(
        self,
        request: PreparedRequest,
        config: SendConfig | Mapping[str, Any] | None = None,
        *,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        spark_session: Optional["SparkSession"] = None,
        **options,
    ) -> Response:
        cfg = SendConfig.check_arg(
            config,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
            spark_session=spark_session,
            **options,
        )

        remote_cfg = cfg.remote_cache
        local_cfg = cfg.local_cache

        final_response: Optional[Response] = None
        local_filepath = None

        in_local = False
        in_remote = False

        if remote_cfg.remote_cache_enabled:
            final_response = self._load_remote_cached_response(
                request,
                remote_cfg,
                spark_session=cfg.spark_session,
            )
            in_remote = final_response is not None

        if local_cfg.local_cache_enabled:
            local_response, local_filepath = self._load_local_cached_response(
                request,
                local_cfg,
            )
            in_local = local_response is not None
            if final_response is None and local_response is not None:
                final_response = local_response

        if final_response is not None:
            if in_remote and not in_local and local_cfg.local_cache_enabled:
                self._store_local_cached_response(
                    request,
                    final_response,
                    local_cfg,
                    filepath=local_filepath,
                )
            elif in_local and not in_remote and remote_cfg.remote_cache_enabled and remote_cfg.table is not None:
                self._store_remote_cached_response(
                    final_response,
                    remote_cfg,
                    spark_session=cfg.spark_session,
                )

            if cfg.raise_error:
                final_response.raise_for_status()
            return final_response

        response = self._local_send(request, config=cfg)

        if response.ok:
            if local_cfg.local_cache_enabled:
                self._store_local_cached_response(
                    request,
                    response,
                    local_cfg,
                    filepath=local_filepath,
                )

            if remote_cfg.remote_cache_enabled and remote_cfg.table is not None:
                self._store_remote_cached_response(
                    response,
                    remote_cfg,
                    spark_session=cfg.spark_session,
                )

        if cfg.raise_error:
            response.raise_for_status()

        return response

    @abstractmethod
    def _local_send(
        self,
        request: PreparedRequest,
        config: SendConfig,
    ) -> Response:
        raise NotImplementedError

    def send_many(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig | SendConfig | Mapping[str, Any] | None = None,
        *,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool | None = None,
        stream: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        batch_size: int | None = None,
        ordered: bool = False,
        max_in_flight: int | None = None,
        spark_session: Optional["SparkSession"] = None,
        **options,
    ) -> "Iterator[Response] | SparkDataFrame":
        cfg = SendManyConfig.check_arg(
            config,
            wait=wait,
            raise_error=raise_error,
            normalize=normalize,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
            batch_size=batch_size,
            ordered=ordered,
            max_in_flight=max_in_flight,
            spark_session=spark_session,
            **options,
        )

        return self._send_many(requests, config=cfg)

    def _send_many(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig,
    ) -> "Iterator[Response] | SparkDataFrame":
        pool = self.job_pool
        remote_cfg = config.remote_cache
        batch_size = config.batch_size or pool.max_workers * 100

        if not remote_cfg.remote_cache_enabled:
            yield from self._send_many_local(requests, config)
        else:
            def _batched(it: Iterator[PreparedRequest], n: int) -> Iterator[list[PreparedRequest]]:
                iterator = iter(it)
                while True:
                    b = list(itertools.islice(iterator, n))
                    if not b:
                        break
                    yield b

            for batch in _batched(requests, batch_size):
                anonymized_batch = [
                    req.anonymize(mode=remote_cfg.anonymize)
                    for req in batch
                ]

                query = remote_cfg.make_batch_lookup_sql(
                    table_name=remote_cfg.table.full_name(safe=True),
                    requests=anonymized_batch,
                )

                try:
                    cache_result = remote_cfg.table.sql.execute(query)
                except Exception as exc:
                    if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                        remote_cfg.table.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                        cache_result = remote_cfg.table.sql.execute(query)
                    else:
                        raise

                cached_responses = list(Response.from_arrow_tabular(cache_result.to_arrow_table()))

                cache_map: dict[tuple[Any, ...], list[Response]] = {}
                for response in cached_responses:
                    key = remote_cfg.request_tuple(response.request)
                    cache_map.setdefault(key, []).append(response)

                hits: list[Response] = []
                misses: list[PreparedRequest] = []

                for req in batch:
                    key = remote_cfg.request_tuple(
                        req.anonymize(mode=remote_cfg.anonymize)
                    )
                    candidates = cache_map.get(key)
                    if candidates:
                        hits.append(candidates[0])
                    else:
                        misses.append(req)

                yield from hits

                if not misses:
                    continue

                miss_send_config = config.to_send_config(
                    with_remote_cache=False,
                    with_local_cache=True,
                    with_spark=False,
                )

                def _miss_jobs() -> Iterator[Job]:
                    for req in misses:
                        yield Job.make(self.send, req, miss_send_config)

                to_insert: list[Response] = []
                failed: list[Response] = []

                for result in pool.as_completed(
                    _miss_jobs(),
                    ordered=config.ordered,
                    max_in_flight=config.max_in_flight,
                    cancel_on_exit=False,
                    shutdown_on_exit=False,
                    raise_error=True,
                ):
                    response: Response = result.result
                    if response.ok:
                        to_insert.append(response)
                        yield response
                    elif config.raise_error:
                        failed.append(response)

                if to_insert:
                    LOGGER.debug(
                        "Persisting %s send_many responses to remote cache table=%s",
                        len(to_insert),
                        remote_cfg.table.full_name(safe=True),
                    )
                    batches = [
                        response.anonymize(mode=remote_cfg.anonymize).to_arrow_batch(parse=False)
                        for response in to_insert
                    ]
                    combined = pa.Table.from_batches(batches).combine_chunks()

                    remote_cfg.table.insert(
                        combined,
                        mode=remote_cfg.mode,
                        match_by=remote_cfg.by or None,
                        wait=remote_cfg.wait,
                    )

                if config.raise_error and failed:
                    failed[-1].raise_for_status()

    def _send_many_local(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig,
    ) -> Iterator[Response]:
        pool = self.job_pool
        send_config = config.to_send_config(
            with_remote_cache=False,
            with_local_cache=True,
            with_spark=False,
        )

        def _uncached_jobs() -> Iterator[Job]:
            for req in requests:
                yield Job.make(self.send, req, send_config)

        for result in pool.as_completed(
            _uncached_jobs(),
            ordered=config.ordered,
            max_in_flight=config.max_in_flight or self.pool_maxsize,
            cancel_on_exit=False,
            shutdown_on_exit=False,
            raise_error=True,
        ):
            response: Response = result.result
            if config.raise_error:
                response.raise_for_status()
            yield response

    def get(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        normalize: bool = True,
        **send_kwargs: Any,
    ) -> Response:
        return self.request(
            "GET",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            normalize=normalize,
            **send_kwargs,
        )

    def post(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        normalize: bool = True,
        **send_kwargs: Any,
    ) -> Response:
        return self.request(
            "POST",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            **send_kwargs,
        )

    def put(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        normalize: bool = True,
        **send_kwargs: Any,
    ) -> Response:
        return self.request(
            "PUT",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            **send_kwargs,
        )

    def patch(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        normalize: bool = True,
        **send_kwargs: Any,
    ) -> Response:
        return self.request(
            "PATCH",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            **send_kwargs,
        )

    def delete(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        normalize: bool = True,
        **send_kwargs: Any,
    ) -> Response:
        return self.request(
            "DELETE",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            **send_kwargs,
        )

    def head(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        normalize: bool = True,
        **send_kwargs: Any,
    ) -> Response:
        send_kwargs.setdefault("stream", False)
        return self.request(
            "HEAD",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            normalize=normalize,
            **send_kwargs,
        )

    def options(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        normalize: bool = True,
        **send_kwargs: Any,
    ) -> Response:
        return self.request(
            "OPTIONS",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            **send_kwargs,
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
        tags: Mapping[str, str] | None = None,
        before_send: Callable[[PreparedRequest], PreparedRequest] | None = None,
        json: Any | None = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
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
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
        )

    def prepare_request(
        self,
        method: str,
        url: URL | str | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        before_send: Callable[[PreparedRequest], PreparedRequest] | None = None,
        after_received: Callable[[Response], Response] | None = None,
        *,
        json: Any | None = None,
        normalize: bool = True,
    ) -> PreparedRequest:
        full_url: URL | str | None = url

        if self.base_url:
            full_url = self.base_url.join(url) if url else self.base_url
        elif url is None:
            raise ValueError("url is required when base_url is not set on the session.")

        if params:
            parsed = URL.parse(full_url, normalize=normalize)
            items = list(parsed.query_items(keep_blank_values=True))
            items.extend((key, value) for key, value in params.items())
            full_url = parsed.with_query_items(tuple(items))

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