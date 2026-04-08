"""HTTP session abstraction with transparent local and remote cache support."""

from __future__ import annotations

import itertools
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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
from yggdrasil.io import SaveMode

from .buffer import BytesIO
from .request import PreparedRequest
from .response import RESPONSE_ARROW_SCHEMA, RESPONSE_SCHEMA, Response
from .send_config import CacheConfig, SendConfig, SendManyConfig
from .url import URL
from ..data import Schema

if TYPE_CHECKING:
    from yggdrasil.spark.frame import DynamicFrame
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
    ) -> tuple[Optional[Response], Optional[Path]]:
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

        LOGGER.debug(
            "Found local %s %s from %s",
            request.method, request.url, filepath
        )
        return loaded, filepath

    def _store_local_cached_response(
        self,
        response: Response,
        cache_cfg: CacheConfig,
        *,
        filepath=None,
    ) -> None:
        if not response.ok:
            return

        anonymized = response.anonymize(mode=cache_cfg.anonymize)
        target = filepath if filepath else cache_cfg.local_cache_file(
            request=anonymized.request,
            suffix=".ypkl",
            force=True
        )

        Job.make(pickle.dump, anonymized, target).fire_and_forget()

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
                    "Found remote %s %s in %s",
                    request.method,
                    request.url,
                    cache_cfg.table,
                )
                return response

        return None

    def _store_remote_cached_response(
        self,
        response: Response,
        cache_cfg: CacheConfig,
        *,
        spark_session: Optional["SparkSession"] = None,
        mode: Optional[SaveMode] = None,
    ) -> None:
        if not response.ok:
            return

        batch = response.anonymize(mode=cache_cfg.anonymize).to_arrow_batch(parse=False)

        cache_cfg.table.insert(
            batch,
            mode=mode if mode is not None else cache_cfg.mode,
            match_by=cache_cfg.match_by or None,
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

        # Per-request configs take precedence over the session-level ones.
        effective_local_cfg = request.local_cache_config or local_cfg
        effective_remote_cfg = request.remote_cache_config or remote_cfg

        # --- 1. Check local cache first (fast, disk-based) ---
        local_filepath = None
        if effective_local_cfg.local_cache_enabled:
            if effective_local_cfg.mode == SaveMode.UPSERT:
                # Force-evict any stale local entry so the fresh response
                # can be written in its place after the actual fetch.
                local_filepath = effective_local_cfg.local_cache_file(
                    request=request, suffix=".ypkl", force=True
                )
                if local_filepath is not None and local_filepath.exists():
                    local_filepath.unlink(missing_ok=True)
                    LOGGER.debug(
                        "UPSERT: evicted local cache for %s %s",
                        request.method, request.url,
                    )
            else:
                local_response, local_filepath = self._load_local_cached_response(
                    request, effective_local_cfg
                )
                if local_response is not None:
                    if cfg.raise_error:
                        local_response.raise_for_status()
                    return local_response

        # --- 2. Check remote cache (slower, SQL-based) ---
        # Skip when the effective config demands a forced refresh (UPSERT).
        if (
            effective_remote_cfg.remote_cache_enabled
            and effective_remote_cfg.mode == SaveMode.APPEND
        ):
            remote_response = self._load_remote_cached_response(
                request,
                effective_remote_cfg,
                spark_session=cfg.spark_session,
            )
            if remote_response is not None:
                # Backfill local cache with the remote hit
                if effective_local_cfg.local_cache_enabled:
                    self._store_local_cached_response(
                        remote_response,
                        effective_local_cfg,
                        filepath=local_filepath,
                    )
                if cfg.raise_error:
                    remote_response.raise_for_status()
                return remote_response

        # --- 3. No cache hit — perform actual request ---
        LOGGER.debug("Sending %s %s", request.method, request.url)
        response = self._local_send(request, config=cfg)
        LOGGER.info("Sent %s %s", request.method, request.url)

        if effective_local_cfg.local_cache_enabled:
            self._store_local_cached_response(
                response,
                effective_local_cfg,
                filepath=local_filepath,
            )

        if effective_remote_cfg.remote_cache_enabled:
            # Pass the effective config so that its mode (UPSERT or APPEND)
            # is used directly by _store_remote_cached_response.
            self._store_remote_cached_response(
                response,
                effective_remote_cfg,
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
    ) -> Iterator[Response]:
        pool = self.job_pool
        remote_cfg = config.remote_cache
        local_cfg = config.local_cache
        batch_size = config.batch_size or pool.max_workers * 100

        def _batched(it: Iterator[PreparedRequest], n: int) -> Iterator[list[PreparedRequest]]:
            iterator = iter(it)
            while True:
                b = list(itertools.islice(iterator, n))
                if not b:
                    break
                yield b

        for batch in _batched(requests, batch_size):
            # --- 1. Check local cache first for each request in the batch ---
            after_local: list[PreparedRequest] = []
            if local_cfg.local_cache_enabled:
                for req in batch:
                    effective_local = req.local_cache_config or local_cfg
                    if effective_local.mode == SaveMode.UPSERT:
                        # Evict any stale local entry; the fresh response will
                        # be stored after the fetch.
                        evict_path = effective_local.local_cache_file(
                            req, suffix=".ypkl", force=True
                        )
                        if evict_path is not None and evict_path.exists():
                            evict_path.unlink(missing_ok=True)
                            LOGGER.debug(
                                "UPSERT: evicted local cache for %s %s",
                                req.method, req.url,
                            )
                        after_local.append(req)
                    else:
                        local_response, _ = self._load_local_cached_response(req, effective_local)
                        if local_response is not None:
                            yield local_response
                        else:
                            after_local.append(req)
                local_hits = len(batch) - len(after_local)
                if local_hits:
                    LOGGER.debug(
                        "Batch local cache: %s/%s hits",
                        local_hits,
                        len(batch),
                    )
            else:
                after_local = batch

            if not after_local:
                continue

            # UPSERT requests must bypass the remote cache read entirely —
            # split them out before building any SQL lookup.
            upsert_reqs = [
                r for r in after_local
                if (r.remote_cache_config or remote_cfg).mode == SaveMode.UPSERT
            ]
            lookup_reqs = [
                r for r in after_local
                if (r.remote_cache_config or remote_cfg).mode != SaveMode.UPSERT
            ]

            # Group non-UPSERT requests by their effective remote cache table so
            # we execute exactly one batch SQL query per table.
            table_to_cfg: dict[str, CacheConfig] = {}
            table_to_reqs: dict[str, list[PreparedRequest]] = {}
            remote_hits: list[Response] = []
            # UPSERT requests are unconditional misses — always refetch.
            remote_misses: list[PreparedRequest] = list(upsert_reqs)

            for req in lookup_reqs:
                t_cfg = req.remote_cache_config or remote_cfg
                if not t_cfg.remote_cache_enabled or t_cfg.mode != SaveMode.APPEND:
                    remote_misses.append(req)
                    continue
                tkey = t_cfg.table.full_name(safe=True)
                if tkey not in table_to_cfg:
                    table_to_cfg[tkey] = t_cfg
                    table_to_reqs[tkey] = []
                table_to_reqs[tkey].append(req)

            for tkey, t_reqs in table_to_reqs.items():
                t_cfg = table_to_cfg[tkey]
                anonymized_batch = [req.anonymize(mode=t_cfg.anonymize) for req in t_reqs]
                query = t_cfg.make_batch_lookup_sql(
                    table_name=tkey, requests=anonymized_batch
                )
                try:
                    cache_result = t_cfg.table.sql.execute(query)
                except Exception as exc:
                    if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                        t_cfg.table.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                        cache_result = t_cfg.table.sql.execute(query)
                    else:
                        raise

                result_map: dict[tuple, Response] = {}
                for response in Response.from_arrow_tabular(cache_result.to_arrow_batches()):
                    result_map[t_cfg.request_tuple(response.request)] = response

                for req in t_reqs:
                    rkey = t_cfg.request_tuple(req.anonymize(mode=t_cfg.anonymize))
                    candidate = result_map.get(rkey)
                    if candidate is not None:
                        remote_hits.append(candidate)
                    else:
                        remote_misses.append(req)

            # --- 3. Yield remote hits and backfill local cache ---
            if remote_hits:
                LOGGER.debug(
                    "Batch remote cache: %s/%s hits (table=%s)",
                    len(remote_hits),
                    len(after_local),
                    remote_cfg.table,
                )
            for remote_hit in remote_hits:
                if local_cfg.local_cache_enabled:
                    self._store_local_cached_response(remote_hit, local_cfg)
                yield remote_hit

            if not remote_misses:
                continue

            # --- 4. Send misses (local cache only; remote written in batch below) ---
            miss_send_config = config.to_send_config(
                with_remote_cache=False,
                with_local_cache=True,
                with_spark=False,
                raise_error=False
            )

            # Snapshot each miss's effective remote config BEFORE we strip
            # remote_cache_config from the copies sent to the thread pool.
            # Keyed by URL string — the natural cache identity for a request.
            # Without this, r.request.remote_cache_config is always None on
            # the response side and every miss falls back to remote_cfg.
            miss_url_to_remote_cfg: dict[str, CacheConfig] = {
                str(r.url): (r.remote_cache_config or remote_cfg)
                for r in remote_misses
                if (r.remote_cache_config or remote_cfg).remote_cache_enabled
            }

            to_insert: list[Response] = []
            failed: list[Response] = []

            for result in pool.as_completed(
                (
                    # Disable request-level remote cache while sending misses;
                    # remote writes are mutualized in the bulk insert block below.
                    Job.make(self.send, r.copy(remote_cache_config=None), miss_send_config)
                    for r in remote_misses
                ),
                ordered=config.ordered,
                max_in_flight=config.max_in_flight or self.pool_maxsize,
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
                # Group responses by (effective table full name, effective mode)
                # so each unique (table, mode) combination gets one bulk insert.
                # Use the pre-snapshotted URL → cfg map to recover the correct
                # per-request config now that the copies no longer carry it.
                table_write_groups: dict[
                    tuple[str, SaveMode], tuple[CacheConfig, list[Response]]
                ] = {}
                for r in to_insert:
                    req = r.request
                    url_key = str(req.url) if req else None
                    t_cfg = (
                        miss_url_to_remote_cfg.get(url_key)
                        if url_key else None
                    ) or remote_cfg
                    if not t_cfg.remote_cache_enabled:
                        continue
                    gkey = (t_cfg.table.full_name(safe=True), t_cfg.mode)
                    if gkey not in table_write_groups:
                        table_write_groups[gkey] = (t_cfg, [])
                    table_write_groups[gkey][1].append(r)

                for (_, mode), (t_cfg, t_responses) in table_write_groups.items():
                    LOGGER.debug(
                        "%s %s responses in remote cache %s",
                        "Upserting" if mode == SaveMode.UPSERT else "Persisting",
                        len(t_responses),
                        t_cfg.table,
                    )
                    batches = [
                        r.anonymize(mode=t_cfg.anonymize).to_arrow_batch(parse=False)
                        for r in t_responses
                    ]
                    t_cfg.table.insert(
                        pa.Table.from_batches(batches).combine_chunks(),
                        mode=mode,
                        match_by=t_cfg.match_by or None,
                        wait=t_cfg.wait,
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
            raise_error=False
        )

        for result in pool.as_completed(
            (
                # Keep local-only path local-only even when requests carry
                # per-request remote cache config.
                Job.make(self.send, r.copy(remote_cache_config=None), send_config)
                for r in requests
            ),
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
    
    def spark_send(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig | SendConfig | Mapping[str, Any] | None = None,
        *,
        schema: Schema | None = None,
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
    ) -> "SparkDataFrame":
        from yggdrasil.spark.frame import DynamicFrame, PICKLE_COLUMN_NAME
        from pyspark.sql.types import BinaryType, StructField, StructType

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

        # Resolve Spark session — prefer cfg.spark_session, fall back to auto-create.
        effective_spark = cfg.spark_session
        if effective_spark is None:
            from yggdrasil.environ import PyEnv
            effective_spark = PyEnv.spark_session(
                create=True,
                install_spark=False,
                import_error=True,
            )

        # Per-executor config: remote cache (Delta SQL) is a driver-level concern
        # and must not execute on Spark workers; local disk cache passes through.
        send_config = cfg.to_send_config(
            with_remote_cache=False,
            with_local_cache=True,
            with_spark=False,
        )

        # Number of requests grouped into each Spark task.
        # Larger batches → fewer tasks but more concurrency per executor via send_many.
        effective_batch_size = cfg.batch_size or 100

        # Capture the session for serialisation to Spark executors.
        # Session.__getstate__ / __setstate__ make it fully pickle-safe.
        session = self

        # Spark schema for the DynamicFrame binary pickle column.
        _pickle_spark_schema = StructType([StructField(PICKLE_COLUMN_NAME, BinaryType(), nullable=False)])

        # Python 3.10-compatible batching (itertools.batched requires 3.12).
        def _batched(it: Iterator, n: int) -> Iterator[list]:
            it = iter(it)
            while True:
                chunk = list(itertools.islice(it, n))
                if not chunk:
                    break
                yield chunk

        # Each Spark task receives a batch and fans out via send_many's thread pool.
        def _send_batch(batch: list[PreparedRequest]) -> list[Response]:
            # Remote cache is a driver concern in spark_send(); enforce that
            # request-level remote cache overrides are disabled on workers.
            local_only_batch = (r.copy(remote_cache_config=None) for r in batch)
            return list(session.send_many(local_only_batch, send_config))

        remote_cfg = cfg.remote_cache

        # ------------------------------------------------------------------ #
        # Remote cache prefetch — only when a remote cache table is present.  #
        # ------------------------------------------------------------------ #
        if remote_cfg.remote_cache_enabled and remote_cfg.mode == SaveMode.APPEND:
            # Materialise the full request list so we can run one SQL batch lookup
            # and compare request keys on the driver side.
            all_requests: list[PreparedRequest] = list(requests)
            LOGGER.info(
                "spark_send: %s requests, remote cache enabled (table=%s, batch_size=%s)",
                len(all_requests),
                remote_cfg.table,
                effective_batch_size,
            )

            anonymized = [r.anonymize(mode=remote_cfg.anonymize) for r in all_requests]

            # UPSERT requests bypass the cache read entirely — split them up
            # front so they never enter any SQL lookup.
            def _is_remote_upsert(req: PreparedRequest) -> bool:
                return (req.remote_cache_config or remote_cfg).mode == SaveMode.UPSERT

            upsert_requests = [r for r in all_requests if _is_remote_upsert(r)]
            lookup_pairs = [
                (orig, anon)
                for orig, anon in zip(all_requests, anonymized)
                if not _is_remote_upsert(orig)
            ]

            # UPSERT requests are unconditional misses; others are resolved
            # per-table below.
            miss_requests: list[PreparedRequest] = list(upsert_requests)

            # Group lookup requests by their effective remote cache table so we
            # execute exactly one SQL query and one to_spark() call per table.
            table_to_cfg: dict[str, CacheConfig] = {}
            table_to_pairs: dict[str, list[tuple[PreparedRequest, PreparedRequest]]] = {}
            for orig, anon in lookup_pairs:
                t_cfg = orig.remote_cache_config or remote_cfg
                if not t_cfg.remote_cache_enabled or t_cfg.mode != SaveMode.APPEND:
                    miss_requests.append(orig)
                    continue
                tkey = t_cfg.table.full_name(safe=True)
                if tkey not in table_to_cfg:
                    table_to_cfg[tkey] = t_cfg
                    table_to_pairs[tkey] = []
                table_to_pairs[tkey].append((orig, anon))

            LOGGER.debug(
                "spark_send: executing remote cache batch lookup "
                "(%s tables, lookup=%s, upsert=%s)",
                len(table_to_cfg),
                sum(len(p) for p in table_to_pairs.values()),
                len(upsert_requests),
            )

            # _cache_to_responses: executor-safe mapInArrow function — all
            # imports are local so nothing from the driver is serialised.
            def _cache_to_responses(
                batches: Iterator[pa.RecordBatch],
            ) -> Iterator[pa.RecordBatch]:
                from yggdrasil.io.response import Response as _Resp
                from yggdrasil.pickle.ser.serde import dumps as _dumps
                from yggdrasil.spark.frame import PICKLE_COLUMN_NAME as _COL
                import pyarrow as _pa

                _schema = _pa.schema([_pa.field(_COL, _pa.binary(), nullable=False)])
                byte_limit = 128 * 1024 * 1024
                out: list[dict[str, bytes]] = []
                out_bytes = 0
                for batch in batches:
                    for response in _Resp.from_arrow_tabular(batch):
                        ser = _dumps(response)
                        if out and out_bytes + len(ser) > byte_limit:
                            yield _pa.RecordBatch.from_pylist(out, schema=_schema)  # noqa
                            out = []
                            out_bytes = 0
                        out.append({_COL: ser})
                        out_bytes += len(ser)
                if out:
                    yield _pa.RecordBatch.from_pylist(out, schema=_schema)  # noqa

            # Per-table: SQL lookup → to_spark() → collect hit keys → hits DF.
            hits_spark_dfs: list = []
            for tkey, t_pairs in table_to_pairs.items():
                t_cfg = table_to_cfg[tkey]
                request_by_cols = list(t_cfg.request_by or [])

                query = t_cfg.make_batch_lookup_sql(
                    table_name=tkey,
                    requests=[anon for _, anon in t_pairs],
                )
                try:
                    t_cache_result = t_cfg.table.sql.execute(query)
                except Exception as exc:
                    if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                        LOGGER.debug(
                            "spark_send: cache table not found, creating (table=%s)",
                            t_cfg.table,
                        )
                        t_cfg.table.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                        t_cache_result = t_cfg.table.sql.execute(query)
                    else:
                        raise

                LOGGER.debug(
                    "spark_send: converting cache result to Spark DataFrame (table=%s)",
                    t_cfg.table,
                )
                t_spark_df = t_cache_result.to_spark(spark=effective_spark)
                hits_spark_dfs.append(t_spark_df)

                # Collect only the key columns to identify hits (cheap: strings/ints).
                LOGGER.debug(
                    "spark_send: collecting hit keys (table=%s, key_cols=%s)",
                    t_cfg.table, request_by_cols,
                )
                if request_by_cols:
                    t_hit_keys: set[tuple] = {
                        tuple(row[col] for col in request_by_cols)
                        for row in t_spark_df.select(*request_by_cols).collect()
                    }
                else:
                    t_hit_keys = set()

                for orig, anon in t_pairs:
                    key = tuple(anon.match_value(col) for col in request_by_cols)
                    if key not in t_hit_keys:
                        miss_requests.append(orig)

            n_hits = len(all_requests) - len(miss_requests)
            LOGGER.info(
                "spark_send: cache prefetch complete — %s hits, %s misses (%s tables)",
                n_hits,
                len(miss_requests),
                len(table_to_cfg),
            )

            # Build hits DynamicFrame: union all per-table Spark DFs then
            # deserialise each row into a pickled Response on the executors.
            LOGGER.debug(
                "spark_send: building hits DynamicFrame via mapInArrow "
                "(n=%s, tables=%s)",
                n_hits, len(hits_spark_dfs),
            )
            if hits_spark_dfs:
                combined_hits_df = hits_spark_dfs[0]
                for df in hits_spark_dfs[1:]:
                    combined_hits_df = combined_hits_df.union(df)
                hits_dyn = DynamicFrame(
                    df=combined_hits_df.mapInArrow(
                        _cache_to_responses, schema=_pickle_spark_schema
                    )
                )
            else:
                hits_dyn = None

            # --- Build misses DynamicFrame (scatter → fetch → gather) ---
            if miss_requests:
                LOGGER.debug(
                    "spark_send: scattering %s misses across Spark (batch_size=%s)",
                    len(miss_requests),
                    effective_batch_size,
                )
                miss_dyn: DynamicFrame = DynamicFrame.parallelize(
                    _send_batch,
                    _batched(iter(miss_requests), effective_batch_size),
                    spark_session=effective_spark,
                ).explode()
                if hits_dyn is not None:
                    LOGGER.debug(
                        "spark_send: unioning %s cache hits with miss results", n_hits
                    )
                    result = DynamicFrame(df=hits_dyn.df.union(miss_dyn.df))
                else:
                    result = miss_dyn
            else:
                LOGGER.debug("spark_send: all requests served from cache, skipping scatter")
                result = hits_dyn

        else:
            # ---------------------------------------------------------------- #
            # No remote cache — pure scatter / gather path.                    #
            # ---------------------------------------------------------------- #
            LOGGER.debug(
                "spark_send: no remote cache, scattering requests (batch_size=%s)",
                effective_batch_size,
            )
            result = DynamicFrame.parallelize(
                _send_batch,
                _batched(requests, effective_batch_size),
                spark_session=effective_spark,
            ).explode()

        target_schema = Schema.from_any(schema) if schema is not None else RESPONSE_SCHEMA
        LOGGER.debug("spark_send: casting result to schema (%s fields)", len(target_schema))
        return result.cast(target_schema)
    
    def get(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        before_send: Callable[[PreparedRequest], PreparedRequest] | None = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
        return self.request(
            "GET",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            before_send=before_send,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
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
        before_send: Callable[[PreparedRequest], PreparedRequest] | None = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
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
            before_send=before_send,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
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
        before_send: Callable[[PreparedRequest], PreparedRequest] | None = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
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
            before_send=before_send,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
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
        before_send: Callable[[PreparedRequest], PreparedRequest] | None = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
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
            before_send=before_send,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
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
        before_send: Callable[[PreparedRequest], PreparedRequest] | None = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
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
            before_send=before_send,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
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
        before_send: Callable[[PreparedRequest], PreparedRequest] | None = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = False,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
        return self.request(
            "HEAD",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            before_send=before_send,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
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
        before_send: Callable[[PreparedRequest], PreparedRequest] | None = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
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
            before_send=before_send,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
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
        local_cache_config: Optional[CacheConfig] = None,
        remote_cache_config: Optional[CacheConfig] = None,
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
            local_cache_config=local_cache_config,
            remote_cache_config=remote_cache_config
        )
