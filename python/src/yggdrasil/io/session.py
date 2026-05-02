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
from yggdrasil.dataclasses.waiting import (
    DEFAULT_WAITING_CONFIG,
    WaitingConfig,
    WaitingConfigArg,
)
from yggdrasil.io.enums import Mode
from .buffer import BytesIO
from .request import PreparedRequest
from .response import RESPONSE_ARROW_SCHEMA, Response, RESPONSE_SCHEMA
from .send_config import CacheConfig, SendConfig, SendManyConfig
from .url import URL
from ..environ import PyEnv

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame

__all__ = ["Session", "CacheConfig", "SendConfig", "SendManyConfig"]


LOGGER = logging.getLogger(__name__)


# Cap on per-batch byte size when emitting responses from a Spark
# `mapInArrow` worker. 128 MiB matches Spark's default Arrow batch
# preference and keeps a single oversized response from inflating the
# whole partition's output. A response that is itself larger than the
# cap is emitted alone in its own batch — chunking the row would
# require splitting binary columns, which is not meaningful here.
_SPARK_RESPONSE_BATCH_BYTE_LIMIT: int = 128 * 1024 * 1024


def _rechunk_to_byte_limit(
    batches: "Iterator[pa.RecordBatch]",
    *,
    byte_limit: int = _SPARK_RESPONSE_BATCH_BYTE_LIMIT,
) -> "Iterator[pa.RecordBatch]":
    """Group single-row record batches so each emitted batch stays under
    *byte_limit* bytes.

    A row whose own size exceeds *byte_limit* is emitted alone — the
    rechunker preserves rows, never splits them across batches.
    Empty input yields nothing.
    """
    pending: list[pa.RecordBatch] = []
    pending_bytes = 0

    def _flush() -> "pa.RecordBatch":
        # `combine_chunks` collapses the buffered rows into a single
        # contiguous batch; `to_batches()[0]` is the resulting batch.
        return pa.Table.from_batches(pending).combine_chunks().to_batches()[0]

    for rb in batches:
        size = int(rb.nbytes)
        if size > byte_limit:
            # Oversized single row — flush whatever's queued first so
            # ordering is preserved, then emit the giant alone.
            if pending:
                yield _flush()
                pending = []
                pending_bytes = 0
            yield rb
            continue

        if pending and pending_bytes + size > byte_limit:
            yield _flush()
            pending = [rb]
            pending_bytes = size
        else:
            pending.append(rb)
            pending_bytes += size

    if pending:
        yield _flush()


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
            self.base_url = URL.from_(self.base_url)
        if self._lock is None:
            self._lock = threading.RLock()
        if self.pool_maxsize <= 0:
            self.pool_maxsize = 8

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._job_pool:
            self._job_pool.shutdown(wait=True)
            self._job_pool = None

    def __getstate__(self):
        state = {
            "base_url": self.base_url,
            "verify": self.verify,
            "pool_maxsize": self.pool_maxsize,
            "send_headers": self.send_headers,
            "waiting": self.waiting,
        }

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()
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
        mode: Optional[Mode] = None,
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
        parsed = URL.from_(url, normalize=normalize)

        if parsed.scheme.startswith("http"):
            from .http_ import HTTPSession

            return HTTPSession(
                base_url=parsed,
                verify=verify,
                waiting=WaitingConfig.from_(waiting) if waiting is not None else None,
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
        return self._send(request, cfg)

    def _send(
        self,
        request: PreparedRequest,
        config: SendConfig,
    ) -> Response:
        """Core send pipeline: local cache → remote cache → network → writeback.

        Assumes `config` is already a fully-resolved `SendConfig` (no kwargs
        merging, no `check_arg`). Intended to be called by `send`, `_send_many`,
        and any other path that has already built its effective config.
        """
        remote_cfg = config.remote_cache
        local_cfg = config.local_cache

        # Per-request configs take precedence over the session-level ones.
        effective_local_cfg = request.local_cache_config or local_cfg
        effective_remote_cfg = request.remote_cache_config or remote_cfg

        # --- 1. Check local cache first (fast, disk-based) ---
        # Pin the filepath BEFORE any session-level mutation of the request
        # (`_local_send` calls `prepare_to_send` which adds outgoing headers
        # and changes the cache key). Without this, load and store hash
        # different shapes of the same request and the cache never hits.
        local_filepath = None
        if effective_local_cfg.local_cache_enabled:
            local_filepath = effective_local_cfg.local_cache_file(
                request=request, suffix=".ypkl", force=True
            )
            if effective_local_cfg.mode == Mode.UPSERT:
                # Force-evict any stale local entry so the fresh response
                # can be written in its place after the actual fetch.
                if local_filepath is not None and local_filepath.exists():
                    local_filepath.unlink(missing_ok=True)
                    LOGGER.debug(
                        "UPSERT: evicted local cache for %s %s",
                        request.method, request.url,
                    )
            else:
                local_response, _ = self._load_local_cached_response(
                    request, effective_local_cfg
                )
                if local_response is not None:
                    if config.raise_error:
                        local_response.raise_for_status()
                    return local_response

        # --- 2. Check remote cache (slower, SQL-based) ---
        # Skip when the effective config demands a forced refresh (UPSERT).
        if (
            effective_remote_cfg.remote_cache_enabled
            and effective_remote_cfg.mode == Mode.APPEND
        ):
            remote_response = self._load_remote_cached_response(
                request,
                effective_remote_cfg,
                spark_session=config.spark_session,
            )
            if remote_response is not None:
                # Backfill local cache with the remote hit
                if effective_local_cfg.local_cache_enabled:
                    self._store_local_cached_response(
                        remote_response,
                        effective_local_cfg,
                        filepath=local_filepath,
                    )
                if config.raise_error:
                    remote_response.raise_for_status()
                return remote_response

        # --- 3. No cache hit — perform actual request ---
        LOGGER.debug("Sending %s %s", request.method, request.url)
        response = self._local_send(request, config=config)
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
                spark_session=config.spark_session,
            )

        if config.raise_error:
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

        if cfg.spark_session is not None:
            return self._spark_send_many(requests, config=cfg)

        return self._send_many(requests, config=cfg)

    # ------------------------------------------------------------------ #
    # send_many — staged pipeline                                         #
    #                                                                    #
    # The flow per batch is:                                             #
    #   1. Local cache: yield hits, evict UPSERT entries, collect misses #
    #   2. Remote cache: group misses by effective table, run one SQL    #
    #      lookup per table, yield hits, collect misses                  #
    #   3. Network: fan out misses through the job pool                  #
    #   4. Bulk remote writeback: group successful responses by          #
    #      (table, mode, match_by, wait, anonymize) so per-request       #
    #      cache configs are honoured exactly                            #
    # ------------------------------------------------------------------ #

    def _effective_local_cfg(
        self,
        request: PreparedRequest,
        session_cfg: CacheConfig,
    ) -> CacheConfig:
        return request.local_cache_config or session_cfg

    def _effective_remote_cfg(
        self,
        request: PreparedRequest,
        session_cfg: CacheConfig,
    ) -> CacheConfig:
        return request.remote_cache_config or session_cfg

    @staticmethod
    def _remote_write_group_key(cfg: CacheConfig) -> tuple:
        """Identity used to group responses for a single bulk remote insert.

        Two responses can share an insert iff every config dimension that
        affects the write call is identical: target table, mode, match-by
        columns, wait flag, and anonymize mode. Without all five, distinct
        per-request configs get silently collapsed onto whichever config
        landed in the bucket first.
        """
        return (
            cfg.table.full_name(safe=True),
            cfg.mode,
            tuple(cfg.match_by) if cfg.match_by else (),
            bool(cfg.wait),
            cfg.anonymize,
        )

    def _split_local_cache(
        self,
        batch: list[PreparedRequest],
        session_local_cfg: CacheConfig,
    ) -> tuple[list[Response], list[PreparedRequest]]:
        """Stage 1: scan the local cache.

        Returns (hits, misses). UPSERT entries are evicted on the way through
        so the eventual fresh response can be written in their place.
        Each request is evaluated against its own effective local cache
        config (per-request override or session-level fallback).
        """
        hits: list[Response] = []
        misses: list[PreparedRequest] = []

        if not session_local_cfg.local_cache_enabled and not any(
            r.local_cache_config for r in batch
        ):
            # Cheap path: no local cache anywhere in this batch.
            return hits, list(batch)

        for req in batch:
            eff = self._effective_local_cfg(req, session_local_cfg)
            if not eff.local_cache_enabled:
                misses.append(req)
                continue

            if eff.mode == Mode.UPSERT:
                evict_path = eff.local_cache_file(req, suffix=".ypkl", force=True)
                if evict_path is not None and evict_path.exists():
                    evict_path.unlink(missing_ok=True)
                    LOGGER.debug(
                        "UPSERT: evicted local cache for %s %s",
                        req.method, req.url,
                    )
                misses.append(req)
                continue

            local_response, _ = self._load_local_cached_response(req, eff)
            if local_response is not None:
                hits.append(local_response)
            else:
                misses.append(req)

        if hits:
            LOGGER.debug(
                "Batch local cache: %s/%s hits", len(hits), len(batch),
            )
        return hits, misses

    def _split_remote_cache(
        self,
        requests: list[PreparedRequest],
        session_remote_cfg: CacheConfig,
        *,
        spark_session: Optional["SparkSession"] = None,
    ) -> tuple[list[Response], list[PreparedRequest]]:
        """Stage 2: scan the remote cache.

        UPSERT requests bypass the read entirely (always misses, refetch).
        Non-UPSERT requests are grouped by their effective cache table so we
        execute exactly one batch SQL lookup per table.
        """
        hits: list[Response] = []
        # UPSERT is unconditional miss.
        upsert_reqs = [
            r for r in requests
            if self._effective_remote_cfg(r, session_remote_cfg).mode == Mode.UPSERT
        ]
        misses: list[PreparedRequest] = list(upsert_reqs)

        # Group APPEND-mode requests by effective table.
        table_to_cfg: dict[str, CacheConfig] = {}
        table_to_reqs: dict[str, list[PreparedRequest]] = {}
        for req in requests:
            if req in upsert_reqs:
                continue
            t_cfg = self._effective_remote_cfg(req, session_remote_cfg)
            if not t_cfg.remote_cache_enabled or t_cfg.mode != Mode.APPEND:
                misses.append(req)
                continue
            tkey = t_cfg.table.full_name(safe=True)
            if tkey not in table_to_cfg:
                table_to_cfg[tkey] = t_cfg
                table_to_reqs[tkey] = []
            table_to_reqs[tkey].append(req)

        for tkey, t_reqs in table_to_reqs.items():
            t_cfg = table_to_cfg[tkey]
            t_hits, t_misses = self._lookup_remote_table(
                t_cfg, t_reqs, spark_session=spark_session,
            )
            hits.extend(t_hits)
            misses.extend(t_misses)

        if hits:
            LOGGER.debug(
                "Batch remote cache: %s/%s hits across %s table(s)",
                len(hits), len(requests), len(table_to_cfg),
            )
        return hits, misses

    def _lookup_remote_table(
        self,
        cfg: CacheConfig,
        requests: list[PreparedRequest],
        *,
        spark_session: Optional["SparkSession"] = None,
    ) -> tuple[list[Response], list[PreparedRequest]]:
        """Execute one batch SQL lookup against a single cache table."""
        anonymized_batch = [r.anonymize(mode=cfg.anonymize) for r in requests]
        query = cfg.make_batch_lookup_sql(
            table_name=cfg.table.full_name(safe=True),
            requests=anonymized_batch,
        )
        try:
            cache_result = cfg.table.sql.execute(query, spark_session=spark_session)
        except Exception as exc:
            if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                cfg.table.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                cache_result = cfg.table.sql.execute(query, spark_session=spark_session)
            else:
                raise

        result_map: dict[tuple, Response] = {}
        for response in Response.from_arrow_tabular(cache_result.to_arrow_batches()):
            result_map[cfg.request_tuple(response.request)] = response

        hits: list[Response] = []
        misses: list[PreparedRequest] = []
        for req, anon in zip(requests, anonymized_batch):
            candidate = result_map.get(cfg.request_tuple(anon))
            if candidate is not None and cfg.filter_response(candidate, request=req):
                hits.append(candidate)
            else:
                misses.append(req)
        return hits, misses

    def _fetch_misses(
        self,
        misses: list[PreparedRequest],
        config: SendManyConfig,
    ) -> Iterator[Response]:
        """Stage 3: send misses through the job pool.

        Returns the raw `Response` stream — caller decides what to do with
        ok/error responses (yield them, persist them, raise).
        """
        # Local cache writes happen here; remote writes are mutualised in
        # `_persist_remote` so we strip per-request remote configs from the
        # copies handed to the workers.
        miss_send_config = config.to_send_config(
            with_remote_cache=False,
            with_local_cache=True,
            with_spark=False,
            raise_error=False,
        )

        pool = self.job_pool
        for result in pool.as_completed(
            (
                Job.make(
                    self._send,
                    r.copy(remote_cache_config=None),
                    miss_send_config,
                )
                for r in misses
            ),
            ordered=config.ordered,
            max_in_flight=config.max_in_flight or self.pool_maxsize,
            cancel_on_exit=False,
            shutdown_on_exit=False,
            raise_error=True,
        ):
            yield result.result

    def _backfill_local_cache(
        self,
        responses: list[Response],
        url_to_local_cfg: Mapping[str, CacheConfig],
        session_local_cfg: CacheConfig,
    ) -> None:
        """Write remote-cache hits back to the local cache.

        Each response is stored against its originating request's effective
        local config (looked up by URL) — using the session-level config for
        every response would be wrong whenever a request carries a custom
        per-request local cache.
        """
        for response in responses:
            url_key = str(response.request.url) if response.request else None
            eff = url_to_local_cfg.get(url_key) if url_key else None
            if eff is None:
                eff = session_local_cfg
            if eff.local_cache_enabled:
                self._store_local_cached_response(response, eff)

    def _persist_remote(
        self,
        responses: list[Response],
        url_to_remote_cfg: Mapping[str, CacheConfig],
        session_remote_cfg: CacheConfig,
    ) -> None:
        """Stage 4: bulk-insert successful responses into the remote cache.

        Responses are bucketed by the full write-group key
        (table, mode, match_by, wait, anonymize) so that distinct per-request
        configs targeting the same table never get collapsed onto a single
        insert with the wrong parameters.
        """
        groups: dict[tuple, tuple[CacheConfig, list[Response]]] = {}
        for response in responses:
            anonymized = response.anonymize(mode="remove")
            url_key = str(anonymized.request.url) if anonymized.request else None
            eff = url_to_remote_cfg.get(url_key) if url_key else None
            if eff is None:
                eff = session_remote_cfg
            if not eff.remote_cache_enabled:
                continue
            gkey = self._remote_write_group_key(eff)
            if gkey not in groups:
                groups[gkey] = (eff, [])
            groups[gkey][1].append(anonymized)

        for (_, mode, _, _, _), (cfg, group_responses) in groups.items():
            LOGGER.debug(
                "%s %s response(s) in remote cache %s",
                "Upserting" if mode == Mode.UPSERT else "Persisting",
                len(group_responses),
                cfg.table,
            )
            batches = pa.Table.from_batches([
                r.to_arrow_batch(parse=False)
                for r in group_responses
            ]).combine_chunks()

            cfg.table.insert(
                batches,
                mode=mode,
                match_by=cfg.match_by or None,
                wait=cfg.wait,
                prune_values={"request_url_path": batches["request_url_path"]},
            )

    def _send_many(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig,
    ) -> Iterator[Response]:
        pool = self.job_pool
        session_remote_cfg = config.remote_cache
        session_local_cfg = config.local_cache
        batch_size = config.batch_size or min(
            config.max_batch_size or 1024, pool.max_workers * 100
        )

        def _batched(it: Iterator[PreparedRequest], n: int) -> Iterator[list[PreparedRequest]]:
            iterator = iter(it)
            while True:
                b = list(itertools.islice(iterator, n))
                if not b:
                    break
                yield b

        for batch in _batched(requests, batch_size):
            # --- Stage 1: local cache ---
            local_hits, after_local = self._split_local_cache(batch, session_local_cfg)
            for hit in local_hits:
                yield hit

            if not after_local:
                continue

            # Snapshot per-request effective configs BEFORE we mutate copies
            # for the worker pool. Keyed by URL string — the natural identity
            # for matching a response back to its originating request.
            url_to_remote_cfg: dict[str, CacheConfig] = {
                str(r.anonymize(mode="remove").url): self._effective_remote_cfg(r, session_remote_cfg)
                for r in after_local
            }
            url_to_local_cfg: dict[str, CacheConfig] = {
                str(r.anonymize(mode="remove").url): self._effective_local_cfg(r, session_local_cfg)
                for r in after_local
            }

            # --- Stage 2: remote cache ---
            remote_hits, after_remote = self._split_remote_cache(
                after_local,
                session_remote_cfg,
                spark_session=config.spark_session,
            )
            # Backfill local cache with remote hits using each request's
            # effective local config — not the session-level fallback.
            self._backfill_local_cache(
                remote_hits, url_to_local_cfg, session_local_cfg,
            )
            for hit in remote_hits:
                yield hit

            if not after_remote:
                continue

            # --- Stage 3: fetch misses ---
            to_insert: list[Response] = []
            failed: list[Response] = []
            for response in self._fetch_misses(after_remote, config):
                if response.ok:
                    to_insert.append(response)
                    yield response
                elif config.raise_error:
                    failed.append(response)

            # --- Stage 4: bulk remote writeback ---
            if to_insert:
                self._persist_remote(
                    to_insert, url_to_remote_cfg, session_remote_cfg,
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

    # ================================================================== #
    # spark_send_many — Spark-native equivalent of _send_many.            #
    #                                                                    #
    # Mirrors the four-stage pipeline of `_send_many`:                   #
    #   1. Local cache  — driver-side, per-request effective config      #
    #   2. Remote cache — driver-side batch SQL lookup, grouped by table #
    #   3. Network      — Spark mapInArrow over batched request DF       #
    #   4. Writeback    — driver-side bulk insert, grouped by write key  #
    #                                                                    #
    # Returns a Spark DataFrame typed as RESPONSE_ARROW_SCHEMA:           #
    # cache hits and network results are unioned into a single output.    #
    #                                                                    #
    # Trade-offs vs `_send_many` (local):                                 #
    # - Per-request local/remote cache overrides ARE honoured in stages  #
    #   1, 2, and 4 (driver-side), but DROPPED in stage 3 — workers see  #
    #   only the session-level local cache config. Same trade-off as the #
    #   existing `spark_send`.                                           #
    # - `raise_error=True` does not short-circuit a partial batch; on    #
    #   workers we always run with raise_error=False and the caller is   #
    #   expected to filter the result DF on `response_status_code`.       #
    #   (The same Arrow round-trip that gives us a typed result also     #
    #   means error short-circuit would require an eager collect.)        #
    # ================================================================== #
    def _spark_send_many(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig,
    ) -> "SparkDataFrame":
        from pyspark.sql import functions as F  # noqa: F401

        # Resolve Spark session — prefer cfg.spark_session, else auto-create.
        spark = config.spark_session
        if spark is None:
            spark = PyEnv.spark_session(
                create=True, install_spark=False, import_error=True,
            )

        session_remote_cfg = config.remote_cache
        session_local_cfg = config.local_cache

        # Materialise — we need the full list to run driver-side stages 1, 2, 4.
        all_requests: list[PreparedRequest] = list(requests)
        if not all_requests:
            return spark.createDataFrame([], schema=RESPONSE_SCHEMA.to_spark_schema())

        LOGGER.info(
            "spark_send_many: %s requests (batch_size=%s, remote=%s, local=%s)",
            len(all_requests),
            config.batch_size,
            session_remote_cfg.remote_cache_enabled,
            session_local_cfg.local_cache_enabled,
        )

        # --- Stage 1: local cache (driver-side) ---
        local_hits, after_local = self._split_local_cache(
            all_requests, session_local_cfg,
        )
        LOGGER.debug(
            "spark_send_many stage 1 (local): %s hits, %s misses",
            len(local_hits), len(after_local),
        )

        # Snapshot per-request effective configs BEFORE any further work.
        # Keyed by the anonymized URL string, same identity rule as `_send_many`.
        url_to_remote_cfg: dict[str, CacheConfig] = {
            str(r.anonymize(mode="remove").url):
                self._effective_remote_cfg(r, session_remote_cfg)
            for r in after_local
        }
        url_to_local_cfg: dict[str, CacheConfig] = {
            str(r.anonymize(mode="remove").url):
                self._effective_local_cfg(r, session_local_cfg)
            for r in after_local
        }

        # --- Stage 2: remote cache (driver-side, batched per table) ---
        if after_local:
            remote_hits, after_remote = self._split_remote_cache(
                after_local,
                session_remote_cfg,
                spark_session=spark,
            )
        else:
            remote_hits, after_remote = [], []

        LOGGER.debug(
            "spark_send_many stage 2 (remote): %s hits, %s misses",
            len(remote_hits), len(after_remote),
        )

        # Backfill local cache with remote hits — must use each request's
        # effective local config, not the session-level fallback.
        if remote_hits:
            self._backfill_local_cache(
                remote_hits, url_to_local_cfg, session_local_cfg,
            )

        # Cache hits → Arrow → Spark DF (driver-side, cheap: already in memory).
        hit_df = self._responses_to_spark_df(
            local_hits + remote_hits, spark,
        )

        # --- Stage 3: network fetch via mapInArrow ---
        if after_remote:
            miss_df = self._spark_fetch_misses(
                after_remote, config, spark,
            )
        else:
            miss_df = None

        # Combine cache hits with network results.
        if hit_df is not None and miss_df is not None:
            result_df = hit_df.unionByName(miss_df, allowMissingColumns=False)
        elif miss_df is not None:
            result_df = miss_df
        elif hit_df is not None:
            result_df = hit_df
        else:
            result_df = spark.createDataFrame([], schema=RESPONSE_SCHEMA.to_spark_schema())

        # --- Stage 4: bulk remote writeback (driver-side) ---
        # We write back ONLY network-fetched responses, matching `_send_many`.
        # Cache hits are by definition already persisted.
        # Bulk writeback requires Response objects (for anonymize + match_by);
        # if no per-request remote configs and no enabled remote cache, skip.
        any_remote = session_remote_cfg.remote_cache_enabled or any(
            (r.remote_cache_config and r.remote_cache_config.remote_cache_enabled)
            for r in after_remote
        )
        if miss_df is not None and any_remote:
            # Pull the network-fetched responses back to the driver as Arrow.
            # This is the unavoidable cost of doing per-request-config-aware
            # bulk writeback: we need Response objects, not just rows.
            # Filter to ok responses to match `_send_many` semantics.
            miss_responses: list[Response] = []
            failed: list[Response] = []
            arrow_table = miss_df.toArrow()
            for response in Response.from_arrow_tabular(arrow_table):
                if response.ok:
                    miss_responses.append(response)
                elif config.raise_error:
                    failed.append(response)

            if miss_responses:
                self._persist_remote(
                    miss_responses, url_to_remote_cfg, session_remote_cfg,
                )

            if config.raise_error and failed:
                failed[-1].raise_for_status()

        return result_df

    # ------------------------------------------------------------------ #
    # Helpers for spark_send_many                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _responses_to_spark_df(
        responses: list["Response"],
        spark: "SparkSession",
    ) -> Optional["SparkDataFrame"]:
        """Convert a list of Response objects to a Spark DataFrame.

        Used to lift driver-side cache hits into the same shape as the
        `mapInArrow` network output so the two can be unioned.
        """
        if not responses:
            return None
        batches = [r.to_arrow_batch(parse=False) for r in responses]
        table = pa.Table.from_batches(batches).combine_chunks()
        # PySpark 3.4+ accepts a pyarrow.Table directly. Fall back to a
        # schema-typed pandas conversion for older releases — the bare
        # ``to_pandas()`` path drops type info and Spark fails to infer
        # nested map / binary columns.
        try:
            return spark.createDataFrame(table)
        except (TypeError, AttributeError):
            return spark.createDataFrame(
                table.to_pandas(),
                schema=RESPONSE_SCHEMA.to_spark_schema(),
            )

    def _spark_fetch_misses(
        self,
        misses: list[PreparedRequest],
        config: SendManyConfig,
        spark: "SparkSession",
    ) -> "SparkDataFrame":
        """Stage 3 on Spark: scatter misses to workers via mapInArrow.

        Each Spark partition becomes one `send_many` call on the executor,
        fanning out via the session's local thread pool. Per-request remote
        cache configs are stripped (driver concern); per-request local cache
        configs are dropped on workers (they see only the session config).
        """
        # Build the request DataFrame on the driver. Each request becomes
        # one row in REQUEST_ARROW_SCHEMA shape. We control partitioning by
        # chunking before handing to createDataFrame.
        request_batches: list[pa.RecordBatch] = [
            r.to_arrow_batch(parse=False) for r in misses
        ]
        request_table = pa.Table.from_batches(request_batches)
        # PySpark 3.4+ accepts a pyarrow.Table directly and preserves the
        # full schema (maps, nullable bytes). The pandas fallback drops
        # type information for object-typed columns and fails on empty
        # maps, which `request_headers` and `request_tags` produce
        # whenever the request carries no extras.
        try:
            request_df = spark.createDataFrame(request_table)
        except (TypeError, AttributeError):
            from .request import REQUEST_SCHEMA

            request_df = spark.createDataFrame(
                request_table.to_pandas(),
                schema=REQUEST_SCHEMA.to_spark_schema(),
            )

        # Per-executor send config: remote cache disabled (driver-only),
        # local cache passthrough, no spark session, raise_error=False so
        # individual failures don't blow up the whole partition.
        send_config = config.to_send_config(
            with_remote_cache=False,
            with_local_cache=True,
            with_spark=False,
            raise_error=False,
        )

        # Capture the session for executor serialisation. Session.__getstate__
        # / __setstate__ are required to make this pickle-safe — otherwise
        # the threading.RLock / JobPoolExecutor will choke at scatter time.
        session = self
        response_spark_schema = RESPONSE_SCHEMA.to_spark_schema()

        def _send_partition(
            batches: Iterator[pa.RecordBatch],
        ) -> Iterator[pa.RecordBatch]:
            # All imports local — keeps the closure self-contained on workers.
            from yggdrasil.io.request import PreparedRequest as _Req

            for batch in batches:
                # Decode this partition's slice of requests.
                partition_requests = list(_Req.from_arrow(batch, normalize=False))
                if not partition_requests:
                    continue

                # Strip per-request remote cache (driver concern) and run the
                # full `send_many` pipeline locally — local cache + thread
                # pool fanout — yielding Response objects.
                local_only_iter = (
                    r.copy(remote_cache_config=None) for r in partition_requests
                )

                # Stream each Response → 1-row Arrow batch → rechunker.
                # The rechunker groups consecutive small responses so each
                # emitted batch stays below the byte cap, and emits any
                # single oversized response on its own.
                def _row_batches() -> Iterator[pa.RecordBatch]:
                    for resp in session.send_many(local_only_iter, send_config):
                        yield resp.to_arrow_batch(parse=False)

                yield from _rechunk_to_byte_limit(_row_batches())

        return request_df.mapInArrow(_send_partition, schema=response_spark_schema)
    
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
            parsed = URL.from_(full_url, normalize=normalize)
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
