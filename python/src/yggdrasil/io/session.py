import base64
import datetime as dt
import itertools
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Mapping, Any, Union, TYPE_CHECKING, Iterator, Callable, Literal

from yggdrasil.concurrent.threading import JobPoolExecutor, Job
from yggdrasil.data import any_to_datetime
from yggdrasil.dataclasses import serialize_dataclass_state, restore_dataclass_state
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg, DEFAULT_WAITING_CONFIG
from yggdrasil.io import SaveMode
from .buffer import BytesIO
from .request import PreparedRequest
from .response import Response, RESPONSE_ARROW_SCHEMA
from .url import URL

if TYPE_CHECKING:
    from ..databricks.sql.table import Table


__all__ = ["Session"]


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
class Session(ABC):
    base_url: Optional[URL] = None
    verify: bool = True
    pool_maxsize: int = 10

    send_headers: Optional[dict[str, str]] = field(default=None, repr=False)
    waiting: WaitingConfig = field(default_factory=lambda: DEFAULT_WAITING_CONFIG, repr=False, compare=False, hash=False)

    _lock: threading.RLock = field(default=None, init=False, repr=False, compare=False)
    _job_pool: JobPoolExecutor = field(default=None, init=False, repr=False, compare=False)

    def _build_job_pool(self) -> JobPoolExecutor:
        return JobPoolExecutor(max_workers=self.pool_maxsize)

    def __post_init__(self) -> None:
        if self.base_url:
            self.base_url = URL.parse(self.base_url)

        if self._lock is None:
            self._lock = threading.RLock()

        if self.pool_maxsize <= 0:
            self.pool_maxsize = 8  # default pool size

    def __getstate__(self) -> dict:
        return serialize_dataclass_state(self)

    def __setstate__(self, state: dict) -> None:
        restore_dataclass_state(self, state)
        self.__post_init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._job_pool:
            self._job_pool.shutdown(wait=True)
            self._job_pool = None

    @property
    def job_pool(self) -> JobPoolExecutor:
        if self._job_pool is None:
            with self._lock:
                if self._job_pool is None:
                    self._job_pool = self._build_job_pool()
        return self._job_pool

    @classmethod
    def from_url(
        cls,
        url: Union[URL, str],
        *,
        verify: bool = True,
        normalize: bool = True,
        waiting: WaitingConfigArg = True,
    ):
        url = URL.parse(url, normalize=normalize)

        if url.scheme.startswith("http"):
            from .http_ import HTTPSession

            return HTTPSession(
                base_url=url,
                verify=verify,
                waiting=WaitingConfig.check_arg(waiting) if waiting is not None else None,
            )

        raise ValueError(f"Cannot build session from scheme: {url.scheme}")

    @abstractmethod
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
    ) -> Response:
        raise NotImplementedError


    @staticmethod
    def _cache_by_keys(arg: Optional[list[str]] = None) -> list[str]:
        if not arg:
            arg = [
                "request_method",
                "request_url_host",
                "request_url_path",
                "request_url_query",
                "request_content_length",
                "request_body_hash",
                "response_content_length",
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
            return request.body.xxh3_int64() if request.body else None
        if key == "request_content_length":
            return request.content_length

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
    def _cache_value_from_response(response: Response, key: str) -> Any:
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
        response: Response,
        keys: list[str],
    ) -> tuple:
        return tuple(cls._cache_value_from_response(response, key) for key in keys)

    @staticmethod
    def _sql_literal(value: Any) -> str:
        if value is None:
            return "null"

        if isinstance(value, bytes):
            value = base64.b64encode(value).decode("ascii")
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, dt.datetime):
            return f"timestamp '{value.isoformat(sep=' ', timespec='microseconds')}'"
        else:
            value = str(value)

        value = value.replace("'", "''")
        return f"'{value}'"

    @classmethod
    def _sql_match_clause(
        cls,
        request: PreparedRequest | None,
        keys: list[str],
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
    ) -> str:
        clauses: list[str] = []

        if request is not None and keys:
            values = cls._cache_values_from_request(request, keys)

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
        cache_anonymize: Literal["remove", "redact"] = "remove",
        received_from: Optional[dt.datetime | dt.date | str] = None,
        received_to: Optional[dt.datetime | dt.date | str] = None,
        wait_cache: WaitingConfigArg = False,
        # Pooling options
        batch_size: Optional[int] = None,
        ordered: bool = False,
        max_in_flight: Optional[int] = None,
    ) -> Iterator[Response]:
        if normalize is None:
            normalize = cache is not None

        if cache is not None:
            cache_by = self._cache_by_keys(cache_by)
            cache_request_by = [_ for _ in cache_by if _.startswith("request")]
        else:
            cache_request_by = []

        pool = self.job_pool
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
                max_in_flight=self.pool_maxsize,
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
                anon_batch = [
                    req.anonymize(mode=cache_anonymize) if cache_anonymize else req
                    for req in batch
                ]

                time_filter = self._sql_match_clause(
                    None, keys=[],
                    received_from=received_from,
                    received_to=received_to,
                )
                clauses = " OR ".join(
                    "(%s)" % self._sql_match_clause(req, keys=cache_request_by)
                    for req in anon_batch
                )
                query = f"SELECT * FROM {cache.full_name(safe=True)}"
                if clauses and time_filter:
                    query += f" WHERE ({clauses}) AND ({time_filter})"
                elif clauses:
                    query += f" WHERE {clauses}"
                elif time_filter:
                    query += f" WHERE {time_filter}"

                try:
                    sql_cache_statement = cache.sql.execute(query)
                except Exception as e:
                    if "TABLE_OR_VIEW_NOT_FOUND" in str(e):
                        cache.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                        sql_cache_statement = cache.sql.execute(query)
                    else:
                        raise

                arrow_batch = sql_cache_statement.to_arrow_table()
                cached_responses = list(Response.from_arrow(arrow_batch))

                cache_map: dict[tuple, Response] = {}
                for resp in cached_responses:
                    cache_map[self._cache_tuple_from_response(resp, cache_request_by)] = resp

                hits: list[Response] = []
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
                            raise_error=False,
                            stream=stream,
                            cache=None,
                        )

                to_insert: list[Response] = []
                failed: list[Response] = []

                for result in pool.as_completed(
                    miss_jobs(),
                    ordered=ordered,
                    max_in_flight=max_in_flight,
                    cancel_on_exit=False,
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
                    import pyarrow as pa
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

    # --- Convenience HTTP Methods ---

    def get(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs
    ) -> Response:
        return self.request(
            "GET",
            url,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
            **kwargs
        )

    def post(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs
    ) -> Response:
        return self.request(
            "POST",
            url,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
            **kwargs
        )

    def put(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs
    ) -> Response:
        return self.request(
            "PUT",
            url,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
            **kwargs
        )

    def patch(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs
    ) -> Response:
        return self.request(
            "PATCH",
            url,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
            **kwargs
        )

    def delete(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs
    ) -> Response:
        return self.request(
            "DELETE",
            url,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
            **kwargs
        )

    def head(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        stream: bool = False,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs
    ) -> Response:
        return self.request(
            "HEAD",
            url,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
            **kwargs
        )

    def options(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
        **kwargs
    ) -> Response:
        return self.request(
            "OPTIONS",
            url,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
            **kwargs
        )

    # --- Request Orchestration ---

    def request(
        self,
        method: str,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        before_send: Optional[Callable[["PreparedRequest"], "PreparedRequest"]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: WaitingConfigArg = None,
        normalize: bool = True,
        cache: Optional["Table"] = None,
    ) -> Response:
        if normalize is None:
            normalize = cache is not None

        request = self.prepare_request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            before_send=before_send
        )

        return self.send(
            request=request,
            stream=stream,
            wait=wait,
            cache=cache
        )

    def prepare_request(
        self,
        method: str,
        url: Optional[Union[URL, str]] = None,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[BytesIO, bytes]] = None,
        tags: Optional[Mapping[str, str]] = None,
        before_send: Optional[Callable[["PreparedRequest"], "PreparedRequest"]] = None,
        after_received: Optional[Callable[["Response"], "Response"]] = None,
        *,
        json: Optional[Any] = None,
        normalize: bool = True
    ) -> PreparedRequest:
        full_url = url
        if self.base_url:
            full_url = self.base_url.join(url) if url else self.base_url
        elif url is None:
            raise ValueError("URL is required if base_url is not set.")

        # Apply params (merge with existing query)
        if params:
            u = URL.parse(full_url, normalize=normalize)
            items = list(u.query_items(keep_blank_values=True))
            items.extend((k, v) for k, v in params.items())
            full_url = u.with_query_items(tuple(items))

        return PreparedRequest.prepare(
            method=method,
            url=full_url,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            before_send=before_send,
            after_received=after_received
        )
