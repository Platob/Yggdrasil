"""HTTP session abstraction with transparent local and remote cache support."""

from __future__ import annotations

import itertools
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Optional

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.arrow.cast import rechunk_arrow_batches_by_byte_size
from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.dataclasses.waiting import (
    DEFAULT_WAITING_CONFIG,
    WaitingConfig,
    WaitingConfigArg,
)
from yggdrasil.io.enums import Mode
from .buffer import BytesIO
from .buffer.nested.folder_io import FolderIO, FolderOptions
from .request import PreparedRequest
from .response import RESPONSE_ARROW_SCHEMA, Response, RESPONSE_SCHEMA
from .response_batch import ResponseBatch
from .send_config import CacheConfig, SendConfig, SendManyConfig
from .url import URL

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame

__all__ = [
    "Session",
    "CacheConfig",
    "SendConfig",
    "SendManyConfig",
    "ResponseBatch",
]


LOGGER = logging.getLogger(__name__)


# Cap on per-batch byte size when emitting responses from a Spark
# `mapInArrow` worker. 128 MiB matches Spark's default Arrow batch
# preference and keeps a single oversized response from inflating the
# whole partition's output. A response that is itself larger than the
# cap is sliced row-wise by the shared rechunker, which never splits a
# single row across batches.
_SPARK_RESPONSE_BATCH_BYTE_LIMIT: int = 128 * 1024 * 1024


# Local cache lives in a partitioned :class:`FolderIO` rooted at
# ``<CacheConfig.path>/cache``. Partition columns come from
# ``RESPONSE_SCHEMA``'s ``partition_by``-tagged fields and the schema
# itself is dropped to ``<root>/.schema`` on first write so future
# reads pick up the layout without inferring. Reads filter by
# request-key tuple + received-window; writes are append-only with
# UUID-named leaves so concurrent fire-and-forget workers can't race
# on a final filename.


def _store_local_arrow_batch(
    cache: "FolderIO",
    batch: pa.RecordBatch,
) -> None:
    """Append a single Arrow batch to ``cache`` from a worker thread.

    Module-level so the fire-and-forget Job pickles cleanly without
    dragging Session state along. Routes the batch through the
    folder's ``write_arrow_batches`` with ``Mode.APPEND`` —
    :class:`FolderIO` resolves the partition tuple from the batch's
    own columns and drops a UUID-named leaf into the matching
    directory.
    """
    cache.write_arrow_batches(
        [batch], options=FolderOptions(mode=Mode.APPEND),
    )


def _request_partition_filter(
    cache: "FolderIO",
    requests: "list[PreparedRequest]",
) -> "pc.Expression | None":
    """Build a pyarrow predicate pruning to the partitions in ``requests``.

    Walks every request, projects its values for the cache's
    partition columns, and AND-s an ``isin`` filter per column.
    Used by the lookup path to skip partitions no incoming request
    actually touches before the row-level match-by filter runs.
    Returns ``None`` when the cache isn't partitioned or the
    request list yields no usable values.
    """
    parts = cache._resolve_partition_columns()
    if not parts:
        return None
    expr: "pc.Expression | None" = None
    for f in parts:
        try:
            values = sorted(
                {r.match_value(f.name) for r in requests},
                key=lambda v: (v is None, v),
            )
        except Exception:
            continue
        if not values:
            continue
        term = pc.is_in(pc.field(f.name), value_set=pa.array(values))
        expr = term if expr is None else (expr & term)
    return expr


def _lookup_local_responses(
    cache: "FolderIO",
    requests: "list[PreparedRequest]",
    *,
    match_by: "tuple[str, ...]",
    received_from: "Any | None" = None,
    received_to: "Any | None" = None,
) -> dict[tuple, Response]:
    """Bulk lookup by request-key tuple against a folder cache.

    One folder read per call — partition pruning shrinks the set of
    leaves visited, then we filter rows by request-key tuple + the
    optional received-window. When several rows share a key the
    latest one (max ``response_received_at``) wins, giving
    UPSERT-on-write semantics for free.
    """
    if not requests:
        return {}
    if not cache.path.exists():
        return {}

    try:
        with cache:
            table = cache.read_arrow_table()
    except Exception:
        # Corrupt leaf, race with a concurrent write, schema drift
        # — fall through to "nothing matches" so callers progress
        # to the next pipeline stage cleanly.
        LOGGER.debug(
            "Local response cache read failed under %s; treating as miss",
            cache.path, exc_info=True,
        )
        return {}

    if table.num_rows == 0:
        return {}

    if "received_at" in table.column_names:
        if received_from is not None:
            table = table.filter(
                pc.greater_equal(
                    table["received_at"],
                    pa.scalar(received_from),
                )
            )
        if received_to is not None:
            table = table.filter(
                pc.less(
                    table["received_at"],
                    pa.scalar(received_to),
                )
            )

    wanted: set[tuple] = {
        tuple(r.match_value(c) for c in match_by) for r in requests
    }
    out: dict[tuple, Response] = {}
    latest_at: dict[tuple, Any] = {}
    for response in Response.from_arrow_tabular(table.to_batches()):
        try:
            key = tuple(response.match_value(c) for c in match_by)
        except Exception:
            continue
        if key not in wanted:
            continue
        ts = response.received_at
        existing = latest_at.get(key)
        if existing is None or (ts is not None and ts >= existing):
            out[key] = response
            if ts is not None:
                latest_at[key] = ts
    return out


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
    ) -> Optional[Response]:
        """Resolve a single request against the partitioned local cache.

        Routes through :func:`_lookup_local_responses` — one folder
        read, partition-pruned by the request's own values, then
        row-filtered by the ``request_by`` tuple and the configured
        received-window.
        """
        match_by = tuple(cache_cfg.request_by or ()) or ("public_url_hash",)
        cache = cache_cfg.local_cache()
        looked = _lookup_local_responses(
            cache, [request],
            match_by=match_by,
            received_from=cache_cfg.received_from,
            received_to=cache_cfg.received_to,
        )
        try:
            key = tuple(request.match_value(c) for c in match_by)
        except Exception:
            return None
        loaded = looked.get(key)
        if loaded is None:
            return None
        if not cache_cfg.filter_response(loaded, request=request):
            return None
        LOGGER.debug(
            "Found local %s %s in %s",
            request.method, request.url, cache.path,
        )
        return loaded

    def _store_local_cached_response(
        self,
        response: Response,
        cache_cfg: CacheConfig,
        *,
        cache: "FolderIO | None" = None,
    ) -> None:
        """Persist one response to the partitioned local cache.

        The response is anonymized first (matches the on-disk
        identity used at lookup time) and the actual write is fired
        off through the job pool — :class:`FolderIO` resolves the
        partition tuple from the batch's own columns and drops a
        UUID-named leaf into the matching directory, so concurrent
        fire-and-forget workers can't race on a final filename.

        ``cache`` lets a hot-loop caller reuse a single
        :class:`FolderIO` instance across many writes instead of
        paying for the per-call construction (cheap but not free).
        """
        if not response.ok:
            return

        anonymized = response.anonymize(mode=cache_cfg.anonymize)
        cache = cache or cache_cfg.local_cache()
        # Build the Arrow batch on the caller's thread while the
        # buffer is still live; the partitioned write goes through
        # fire-and-forget so the caller doesn't block on disk IO.
        batch = anonymized.to_arrow_batch(parse=False)
        Job.make(_store_local_arrow_batch, cache, batch).fire_and_forget()

    def _load_remote_cached_response(
        self,
        request: PreparedRequest,
        cache_cfg: CacheConfig,
        *,
        spark_session: Optional["SparkSession"] = None,
    ) -> Optional[Response]:
        if not cache_cfg.remote_cache_enabled:
            return None

        # Skip the per-request ``anonymize()`` when the match keys are
        # all ``public_*`` — the SQL clause and the response-side
        # join key both come out identical without it.
        lookup_request = (
            request
            if cache_cfg.request_by_is_public
            else request.anonymize(mode=cache_cfg.anonymize)
        )
        query = cache_cfg.make_batch_lookup_sql(
            table_name=cache_cfg.table.full_name(safe=True),
            requests=[lookup_request],
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
            prune_values={"public_hash": batch["public_hash"]},
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

    def _prepare_request(self, request: PreparedRequest) -> PreparedRequest:
        """Session-wide request hook fired once per outbound request.

        Default returns *request* unchanged. Subclasses override to inject
        session-level concerns — auth, signing, correlation IDs, mandatory
        headers — that should apply to every request leaving this session.
        Runs in :meth:`_send` just before :meth:`_local_send`, so cache hits
        bypass it. Travels with the session into Spark workers via
        ``__getstate__`` / ``__setstate__``.
        """
        return request

    def _prepare_response(self, response: Response) -> Response:
        """Session-wide response hook fired once per completed network send.

        Default returns *response* unchanged. Subclasses override to log,
        redact, enrich, or wrap responses returned from the wire. Runs in
        :meth:`_send` after :meth:`_local_send` and before cache writeback,
        so the persisted response reflects any post-processing. Cache hits
        bypass it. Travels with the session into Spark workers via
        ``__getstate__`` / ``__setstate__``.
        """
        return response

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
        # UPSERT mode skips the lookup outright — the lookup
        # tie-break already returns the latest matching row, so a
        # stale entry can sit on disk indefinitely without
        # affecting correctness; the fresh fetch below will append
        # a newer row that wins on the next read.
        local_cache: "FolderIO | None" = None
        if effective_local_cfg.local_cache_enabled:
            local_cache = effective_local_cfg.local_cache()
            if effective_local_cfg.mode != Mode.UPSERT:
                local_response = self._load_local_cached_response(
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
                if local_cache is not None:
                    self._store_local_cached_response(
                        remote_response,
                        effective_local_cfg,
                        cache=local_cache,
                    )
                if config.raise_error:
                    remote_response.raise_for_status()
                return remote_response

        # --- 3. No cache hit — perform actual request ---
        request = self._prepare_request(request)
        LOGGER.debug("Sending %s %s", request.method, request.url)
        response = self._local_send(request, config=config)
        response = self._prepare_response(response)
        LOGGER.info("Sent %s %s", request.method, request.url)

        if local_cache is not None:
            self._store_local_cached_response(
                response,
                effective_local_cfg,
                cache=local_cache,
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
        max_batch_ttl: float | None = None,
        spark_session: Optional["SparkSession"] = None,
        **options,
    ) -> Iterator[Response]:
        """Stream responses one at a time, in both Python and Spark modes.

        Spark-backed buckets are drained via the holder's
        :meth:`TabularIO.read_records`, which for :class:`MemorySparkIO`
        uses ``df.toLocalIterator()`` — rows stream from the executors
        one at a time, so the driver memory footprint stays bounded
        even for large network-fetch batches. Callers that want a
        :class:`SparkDataFrame` (or the per-bucket origin breakdown)
        should consume :meth:`send_many_batches` and call
        ``ResponseBatch.to_dataframe()`` themselves.

        ``max_batch_ttl`` (default :data:`DEFAULT_MAX_BATCH_TTL`,
        300 s) caps how long the batcher will wait for ``requests`` to
        produce a full chunk before flushing what it has — bounds tail
        latency when the upstream iterator is slow. ``None`` disables
        the time cap; the batch only closes when ``batch_size`` is
        reached or the iterator is exhausted.
        """
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
            max_batch_ttl=max_batch_ttl,
            spark_session=spark_session,
            **options,
        )
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
    ) -> tuple[dict[str, list[Response]], list[PreparedRequest]]:
        """Stage 1: scan the local cache.

        Returns ``(hits_by_path, misses)``. UPSERT entries are evicted
        on the way through so the eventual fresh response can be
        written in their place. Each request is evaluated against its
        own effective local cache config (per-request override or
        session-level fallback).

        Hits are grouped by the effective config's local-cache folder
        — keyed via :meth:`CacheConfig.local_cache_folder`, which
        auto-fills the default ``~/.yggdrasil/cache/response`` root on any
        config that didn't carry a ``path`` — so the per-config split
        survives all the way to :class:`ResponseBatch.local_hits`.
        Collapsing them back into one bucket would lose that
        provenance.
        """
        hits: dict[str, list[Response]] = {}
        misses: list[PreparedRequest] = []

        if not session_local_cfg.local_cache_enabled and not any(
            r.local_cache_config for r in batch
        ):
            # Cheap path: no local cache anywhere in this batch.
            return hits, list(batch)

        # Group active requests by their effective config so each
        # cache root takes one folder read instead of one per
        # request — that's the whole point of the partitioned
        # layout. UPSERT-mode and disabled requests fall straight
        # to misses without touching the cache.
        cfg_groups: dict[str, tuple[CacheConfig, list[PreparedRequest]]] = {}
        for req in batch:
            eff = self._effective_local_cfg(req, session_local_cfg)
            if not eff.local_cache_enabled or eff.mode == Mode.UPSERT:
                misses.append(req)
                continue
            pkey = str(eff.local_cache_folder())
            slot = cfg_groups.get(pkey)
            if slot is None:
                cfg_groups[pkey] = (eff, [req])
            else:
                slot[1].append(req)

        for pkey, (eff, group_reqs) in cfg_groups.items():
            cache = eff.local_cache()
            match_by = tuple(eff.request_by or ()) or ("public_url_hash",)
            looked_up = _lookup_local_responses(
                cache, group_reqs,
                match_by=match_by,
                received_from=eff.received_from,
                received_to=eff.received_to,
            )
            for req in group_reqs:
                try:
                    key = tuple(req.match_value(c) for c in match_by)
                except Exception:
                    misses.append(req)
                    continue
                cached = looked_up.get(key)
                if cached is None:
                    misses.append(req)
                    continue
                if not eff.filter_response(cached, request=req):
                    misses.append(req)
                    continue
                hits.setdefault(pkey, []).append(cached)

        if hits:
            total = sum(len(v) for v in hits.values())
            LOGGER.debug(
                "Batch local cache: %s/%s hits across %s path(s)",
                total, len(batch), len(hits),
            )
        return hits, misses

    def _split_remote_cache(
        self,
        requests: list[PreparedRequest],
        session_remote_cfg: CacheConfig,
        *,
        spark_session: Optional["SparkSession"] = None,
    ) -> tuple[dict[str, list[Response]], list[PreparedRequest]]:
        """Stage 2: scan the remote cache.

        UPSERT requests bypass the read entirely (always misses, refetch).
        Non-UPSERT requests are grouped by their effective cache table so we
        execute exactly one batch SQL lookup per table.

        Returns hits as a per-table mapping keyed by
        ``CacheConfig.table.full_name(safe=True)`` so the downstream
        :class:`ResponseBatch` can preserve which table answered which
        subset of the batch — collapsing them back into one bucket
        would lose that provenance.
        """
        hits: dict[str, list[Response]] = {}
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

        total_hits = 0
        for tkey, t_reqs in table_to_reqs.items():
            t_cfg = table_to_cfg[tkey]
            t_hits, t_misses = self._lookup_remote_table(
                t_cfg, t_reqs, spark_session=spark_session,
            )
            if t_hits:
                hits[tkey] = t_hits
                total_hits += len(t_hits)
            misses.extend(t_misses)

        if total_hits:
            LOGGER.debug(
                "Batch remote cache: %s/%s hits across %s table(s)",
                total_hits, len(requests), len(table_to_cfg),
            )
        return hits, misses

    def _lookup_remote_table(
        self,
        cfg: CacheConfig,
        requests: list[PreparedRequest],
        *,
        spark_session: Optional["SparkSession"] = None,
    ) -> tuple[list[Response], list[PreparedRequest]]:
        """Execute one batch SQL lookup against a single cache table.

        When ``cfg.request_by_is_public`` holds, the per-request
        ``anonymize()`` pass is skipped — ``public_*`` match keys hash
        to the same value on the original and the anonymized request,
        so the lookup tuple and SQL clause both come out identical
        without paying for one URL parse + header normalize per
        request.
        """
        if cfg.request_by_is_public:
            lookup_batch: list[PreparedRequest] = list(requests)
        else:
            lookup_batch = [r.anonymize(mode=cfg.anonymize) for r in requests]

        query = cfg.make_batch_lookup_sql(
            table_name=cfg.table.full_name(safe=True),
            requests=lookup_batch,
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
        for req, lookup in zip(requests, lookup_batch):
            candidate = result_map.get(cfg.request_tuple(lookup))
            if candidate is not None and cfg.filter_response(candidate, request=req):
                hits.append(candidate)
            else:
                misses.append(req)
        return hits, misses

    @staticmethod
    def _responses_to_spark(
        responses: list[Response],
        spark: "SparkSession",
    ) -> "SparkDataFrame":
        """Lift a list of :class:`Response` to a schema-bearing Spark frame.

        Used on the spark path to keep every bucket frame-resident.
        Empty input yields an empty DataFrame keyed to
        :data:`RESPONSE_SCHEMA` so downstream ``unionByName`` calls never
        trip on a column-list mismatch.
        """
        if not responses:
            return spark.createDataFrame(
                [], schema=RESPONSE_SCHEMA.to_spark_schema(),
            )
        table = pa.Table.from_batches(
            [r.to_arrow_batch(parse=False) for r in responses]
        )
        return spark.createDataFrame(table)

    def _split_remote_cache_spark(
        self,
        requests: list[PreparedRequest],
        session_remote_cfg: CacheConfig,
        *,
        spark: "SparkSession",
    ) -> tuple[dict[str, "SparkDataFrame"], list[PreparedRequest]]:
        """Spark variant of :meth:`_split_remote_cache`.

        Returns ``(hits_by_table, misses)`` — hits stay as Spark
        DataFrames keyed by ``CacheConfig.table.full_name(safe=True)``
        so the caller can hand them straight to :class:`ResponseBatch`
        without ever materialising rows on the driver and without
        losing the per-table provenance to a premature
        ``unionByName``. Misses still come back as a Python list
        because the driver needs concrete request objects to scatter
        through stage 3.
        """
        upsert_reqs = [
            r for r in requests
            if self._effective_remote_cfg(r, session_remote_cfg).mode == Mode.UPSERT
        ]
        misses: list[PreparedRequest] = list(upsert_reqs)

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

        hits_by_table: dict[str, "SparkDataFrame"] = {}
        for tkey, t_reqs in table_to_reqs.items():
            t_cfg = table_to_cfg[tkey]
            t_hits_df, t_misses = self._lookup_remote_table_spark(
                t_cfg, t_reqs, spark=spark,
            )
            if t_hits_df is not None:
                hits_by_table[tkey] = t_hits_df
            misses.extend(t_misses)

        if any(table_to_reqs.values()):
            LOGGER.debug(
                "Batch remote cache (spark): scanned %s table(s) for %s request(s)",
                len(table_to_cfg), len(requests),
            )
        return hits_by_table, misses

    def _lookup_remote_table_spark(
        self,
        cfg: CacheConfig,
        requests: list[PreparedRequest],
        *,
        spark: "SparkSession",
    ) -> tuple[Optional["SparkDataFrame"], list[PreparedRequest]]:
        """Spark variant of :meth:`_lookup_remote_table`.

        Runs the same batch lookup SQL, but keeps the result as a Spark
        DataFrame instead of materialising :class:`Response` objects on
        the driver. Misses are computed by collecting the distinct
        ``request_by`` key tuples back to the driver — bounded by the
        number of cached rows that match this batch, not by total cache
        size — and diffing against the input requests.

        :meth:`CacheConfig.filter_response`'s per-row branch is skipped
        on the spark path: ``received_from`` / ``received_to`` are
        already encoded in :meth:`CacheConfig.make_batch_lookup_sql`'s
        ``WHERE`` clause, and the request-key check is what the
        ``request_tuple`` diff already enforces.
        """
        if cfg.request_by_is_public:
            lookup_batch: list[PreparedRequest] = list(requests)
        else:
            lookup_batch = [r.anonymize(mode=cfg.anonymize) for r in requests]
        query = cfg.make_batch_lookup_sql(
            table_name=cfg.table.full_name(safe=True),
            requests=lookup_batch,
        )
        try:
            cache_result = cfg.table.sql.execute(query, spark_session=spark)
        except Exception as exc:
            if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                cfg.table.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                cache_result = cfg.table.sql.execute(query, spark_session=spark)
            else:
                raise

        hits_df = cache_result.read_spark_frame()

        key_cols = list(cfg.request_by or [])
        if not key_cols:
            # No request-key columns means the SQL can't disambiguate
            # rows per request; mirror the Python path's behaviour by
            # treating every input request as a hit when any row came
            # back, otherwise everything is a miss.
            try:
                any_row = hits_df.head(1)
            except Exception:
                any_row = None
            if any_row:
                return hits_df, []
            return None, list(requests)

        matched_rows = hits_df.select(*key_cols).distinct().toLocalIterator()
        matched: set[tuple] = {
            tuple(row[c] for c in key_cols) for row in matched_rows
        }

        misses: list[PreparedRequest] = []
        for req, lookup in zip(requests, lookup_batch):
            if cfg.request_tuple(lookup) not in matched:
                misses.append(req)
        return hits_df, misses

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

        Each response is stored against its originating request's
        effective local config (looked up by URL) — using the
        session-level config for every response would be wrong
        whenever a request carries a custom per-request local
        cache. Responses are anonymized then grouped by the
        effective cache root so each :class:`FolderIO` takes a
        single bulk write instead of one fire-and-forget per
        response — collapsing N small writes into one
        partition-routed write per root, with FolderIO's auto-prune
        path handling the merge when the config is in UPSERT mode.
        """
        groups: dict[str, tuple["FolderIO", "CacheConfig", list[Response]]] = {}
        for response in responses:
            url_key = str(response.request.url) if response.request else None
            eff = url_to_local_cfg.get(url_key) if url_key else None
            if eff is None:
                eff = session_local_cfg
            if not eff.local_cache_enabled:
                continue
            anonymized = response.anonymize(mode=eff.anonymize)
            pkey = str(eff.local_cache_folder())
            slot = groups.get(pkey)
            if slot is None:
                groups[pkey] = (eff.local_cache(), eff, [anonymized])
            else:
                slot[2].append(anonymized)

        for cache, eff, group_responses in groups.values():
            ok_responses = [r for r in group_responses if r.ok]
            if not ok_responses:
                continue
            table = pa.Table.from_batches([
                r.to_arrow_batch(parse=False) for r in ok_responses
            ])
            mode = Mode.UPSERT if eff.mode == Mode.UPSERT else Mode.APPEND
            options = FolderOptions(
                mode=mode,
                match_by_names=list(eff.request_by or ()) or None,
            )
            with cache:
                cache.write_arrow_table(table, options=options)

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
                prune_values={"public_hash": batches["public_hash"]},
            )

    def _send_many(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig,
    ) -> Iterator[Response]:
        """Stream responses, flattening the per-chunk :class:`ResponseBatch`.

        Iteration order matches :class:`ResponseBatch.parts`: local hits
        first, then remote hits, then network fetches. Callers that need
        the origin breakdown should use :meth:`send_many_batches`
        instead.

        Works in both Python and Spark modes. Spark-backed buckets are
        drained via the holder's :meth:`TabularIO.read_records`, which
        for :class:`MemorySparkIO` uses ``df.toLocalIterator()`` — rows
        stream from the executors one at a time, so the driver memory
        footprint stays bounded even for large network-fetch batches.
        :class:`ResponseBatch.__iter__` rejects Spark mode (it would
        force a ``df.toArrow()`` collect); going through the holders
        sidesteps that guard.
        """
        for batch in self._send_many_batches(requests, config):
            for holder in batch.parts():
                yield from Response.from_records(holder.read_records())

    def send_many_batches(
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
        max_batch_ttl: float | None = None,
        spark_session: Optional["SparkSession"] = None,
        **options,
    ) -> Iterator[ResponseBatch]:
        """Yield one :class:`ResponseBatch` per processed chunk.

        Public entry point: both Python and Spark modes yield the same
        ``Iterator[ResponseBatch]`` shape, chunked the same way, so
        downstream consumers can stream partial results uniformly. Each
        yielded batch carries schema-bearing holders even when a stage
        produced no rows — the schema is preserved for empty results.

        ``max_batch_ttl`` (default :data:`DEFAULT_MAX_BATCH_TTL`,
        300 s) caps how long the batcher waits for ``requests`` to
        fill one chunk before flushing what's accumulated — keeps
        downstream stages moving when the upstream iterator is slow.
        ``None`` disables the time cap.
        """
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
            max_batch_ttl=max_batch_ttl,
            spark_session=spark_session,
            **options,
        )
        yield from self._send_many_batches(requests, cfg)

    def _send_many_batches(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig,
    ) -> Iterator[ResponseBatch]:
        """Yield one :class:`ResponseBatch` per processed chunk.

        Single pipeline for both Python and Spark modes — the only
        differences are stage 3 (fetch misses through the local job
        pool vs. ``mapInArrow`` over executors) and stage 4 (per-row
        Arrow insert vs. lazy Spark insert). Mode is picked from
        ``config.spark_session``.

        Both modes chunk requests by ``batch_size`` and yield one
        :class:`ResponseBatch` per chunk so callers see the same
        streaming shape regardless of engine. In Spark mode each chunk
        produces its own ``mapInArrow`` job — pass a larger
        ``batch_size`` (or ``max_batch_size``) when you'd rather
        amortise scheduler overhead across a single bulk fetch. Empty
        buckets are returned as schema-bearing holders so a chunk that
        fully short-circuited on local cache still advertises the
        response schema for ``remote_hits`` / ``new_hits``.
        """
        spark = config.spark_session
        is_spark = spark is not None
        session_remote_cfg = config.remote_cache
        session_local_cfg = config.local_cache

        if is_spark:
            # Spark mode has no driver-side thread pool to scale the
            # default against — fall back to ``max_batch_size`` (or
            # 1024) so each chunk maps to one ``mapInArrow`` scatter
            # of bounded width. Callers who want a single mega-chunk
            # (preserving the original bulk-fetch optimisation) can
            # pass an explicit ``batch_size`` larger than their
            # request count.
            batch_size = config.batch_size or config.max_batch_size or 1024
        else:
            pool = self.job_pool
            batch_size = config.batch_size or min(
                config.max_batch_size or 1024, pool.max_workers * 100
            )

        ttl = config.max_batch_ttl

        def _batched(
            it: Iterator[PreparedRequest],
            n: int,
            ttl_seconds: float | None,
        ) -> Iterator[list[PreparedRequest]]:
            # When no TTL is set, fall back to the cheap islice path —
            # avoids the per-request monotonic() probe.
            iterator = iter(it)
            if ttl_seconds is None or ttl_seconds <= 0:
                while True:
                    b = list(itertools.islice(iterator, n))
                    if not b:
                        break
                    yield b
                return

            # Time-bounded path: pull one item at a time and flush
            # when either the size cap or the wall-clock deadline is
            # reached. The deadline is reset per chunk so a slow
            # upstream gets a fresh window after each flush.
            buf: list[PreparedRequest] = []
            deadline: float | None = None
            for item in iterator:
                if not buf:
                    deadline = time.monotonic() + ttl_seconds
                buf.append(item)
                if len(buf) >= n or (
                    deadline is not None and time.monotonic() >= deadline
                ):
                    yield buf
                    buf = []
                    deadline = None
            if buf:
                yield buf

        chunks = _batched(requests, batch_size, ttl)

        for chunk in chunks:
            if not chunk:
                continue

            # --- Stage 1: local cache ---
            local_hits_by_path, after_local = self._split_local_cache(
                chunk, session_local_cfg,
            )
            # On the spark path, lift each path's responses to its own
            # Spark frame so every bucket downstream is frame-resident
            # — matches stage 2/3 and lets the caller union holders
            # without a per-bucket type switch. Empty dict (no local
            # hits) is left as-is; ResponseBatch installs a
            # schema-bearing default placeholder, Spark or Arrow as
            # appropriate.
            local_hits: "dict[str, list[Response]] | dict[str, SparkDataFrame]"
            if is_spark:
                local_hits = {
                    pkey: self._responses_to_spark(rs, spark)
                    for pkey, rs in local_hits_by_path.items()
                }
            else:
                local_hits = local_hits_by_path
            # Remote hits are split per cache table — keyed by
            # ``CacheConfig.table.full_name(safe=True)`` — so the
            # downstream :class:`ResponseBatch` preserves which table
            # answered which subset. An empty dict tells
            # ``ResponseBatch`` to install a schema-bearing default.
            remote_hits: "dict[str, list[Response]] | dict[str, SparkDataFrame]" = {}
            # Default new_hits to None so ResponseBatch coerces it to a
            # schema-bearing empty holder (Spark or Arrow depending on
            # mode) — no special-case for "stage skipped".
            new_hits: "list[Response] | SparkDataFrame | None" = None

            if not after_local:
                yield ResponseBatch(
                    local_hits=local_hits,
                    remote_hits=remote_hits,
                    new_hits=new_hits,
                    spark=spark,
                )
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
            # Python path drains the StatementResult into Response
            # objects. Spark path keeps the result as a Spark DataFrame
            # (via :meth:`_split_remote_cache_spark`) so ``remote_hits``
            # never collects to the driver and downstream callers can
            # union it with stage 3's Spark output.
            if is_spark:
                remote_hits, after_remote = self._split_remote_cache_spark(
                    after_local,
                    session_remote_cfg,
                    spark=spark,
                )
                # Local-cache backfill from a Spark frame would force a
                # toLocalIterator on the driver — skip it on the spark
                # path, matching how stage 3/4 keep network results
                # frame-resident. Drivers that want a hot local cache
                # should use the Python path explicitly.
            else:
                remote_hits, after_remote = self._split_remote_cache(
                    after_local,
                    session_remote_cfg,
                    spark_session=spark,
                )
                # Backfill local cache with remote hits using each request's
                # effective local config — not the session-level fallback.
                # Flatten the per-table split here: the local cache
                # doesn't care which remote table sourced a row, only
                # the response-to-request mapping by URL.
                self._backfill_local_cache(
                    [r for table_hits in remote_hits.values() for r in table_hits],
                    url_to_local_cfg,
                    session_local_cfg,
                )

            if not after_remote:
                yield ResponseBatch(
                    local_hits=local_hits,
                    remote_hits=remote_hits,
                    new_hits=new_hits,
                    spark=spark,
                )
                continue

            # --- Stage 3: fetch misses ---
            failed: list[Response] = []
            if is_spark:
                # Network results stay in Spark — never collected to
                # the driver. raise_error doesn't short-circuit a
                # partial mapInArrow batch; callers filter on
                # response_status_code if they care.
                new_hits = self._spark_fetch_misses(after_remote, config, spark)
            else:
                new_list: list[Response] = []
                for response in self._fetch_misses(after_remote, config):
                    if response.ok:
                        new_list.append(response)
                    elif config.raise_error:
                        failed.append(response)
                new_hits = new_list

            # --- Stage 4: bulk remote writeback ---
            if is_spark:
                # `cfg.table.insert` accepts the Spark DataFrame
                # directly, so we hand off the lazy DF without
                # materialising on the driver.
                if (
                    new_hits is not None
                    and session_remote_cfg.remote_cache_enabled
                ):
                    self._spark_persist_remote(
                        new_hits, session_remote_cfg, spark=spark,
                    )
            else:
                if new_hits:
                    self._persist_remote(
                        new_hits, url_to_remote_cfg, session_remote_cfg,
                    )

            yield ResponseBatch(
                local_hits=local_hits,
                remote_hits=remote_hits,
                new_hits=new_hits,
                spark=spark,
            )

            if not is_spark and config.raise_error and failed:
                failed[-1].raise_for_status()

    # ------------------------------------------------------------------ #
    # Spark stage 3 / 4 helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _spark_persist_remote(
        new_responses_df: "SparkDataFrame",
        cfg: CacheConfig,
        *,
        spark: "SparkSession",
    ) -> None:
        """Stage 4 on Spark: bulk-insert successful responses into the remote cache.

        Honours the session-level remote config only — per-request overrides
        collapse onto it on the spark path, mirroring stage 3 where workers
        see only the session-level local cache config. ``cfg.table.insert``
        accepts the Spark DataFrame directly via ``spark_insert_into``, so
        no driver-side collect is needed.

        Before inserting, APPEND-mode writes are de-duplicated against the
        existing remote rows via a ``left_anti`` join on the response
        ``hash`` column — the remote table stores anonymized requests
        (cf. ``_persist_remote``), so a row whose hash already lives in
        the cache is suppressed rather than re-inserted. UPSERT mode
        keeps its read-free fast path and relies on ``match_by`` to
        collapse duplicates server-side.
        """
        from pyspark.sql import functions as F

        ok_df = new_responses_df.where(
            (F.col("status_code") >= 200)
            & (F.col("status_code") < 300)
        )

        if cfg.mode != Mode.UPSERT:
            table_name = cfg.table.full_name(safe=True)
            try:
                existing_df = spark.sql(
                    "SELECT DISTINCT public_hash "
                    f"FROM {table_name}"
                )
            except Exception as exc:
                # Table doesn't exist yet — nothing to dedup against; the
                # downstream `cfg.table.insert` handles creation. Match the
                # error-string sniff used by `_lookup_remote_table`.
                if "TABLE_OR_VIEW_NOT_FOUND" not in str(exc):
                    raise
                existing_df = None

            if existing_df is not None:
                ok_df = ok_df.join(
                    existing_df,
                    on=["public_hash"],
                    how="left_anti",
                )

        LOGGER.debug(
            "%s ok response(s) into remote cache %s (spark insert)",
            "Upserting" if cfg.mode == Mode.UPSERT else "Persisting",
            cfg.table,
        )
        cfg.table.insert(
            ok_df,
            mode=cfg.mode,
            match_by=cfg.match_by or None,
            wait=cfg.wait,
            prune_by=["public_hash"],
            spark_session=spark,
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

        The session is shipped to executors via ``sparkContext.broadcast``
        — once per executor instead of once per task closure — and
        re-attached to each request on the worker via
        :meth:`PreparedRequest.attach_session`. Requests cross the wire
        as a single pickled-bytes column; pickle preserves the full
        :class:`PreparedRequest` (closures, buffer, cache configs)
        without the per-engine schema dance the Arrow round-trip needed.
        """
        import pickle

        from pyspark.sql.types import BinaryType, StructField, StructType

        req_schema = StructType([StructField("request", BinaryType(), nullable=False)])
        req_rows = [
            (pickle.dumps(r.copy(remote_cache_config=None), protocol=pickle.HIGHEST_PROTOCOL),)
            for r in misses
        ]

        # Spread requests across many partitions so mapInArrow scatters
        # across the whole cluster instead of piling them onto a handful
        # of executors. ``createDataFrame`` defaults to a single partition
        # for small Python lists, which serialises stage 3. Target one
        # request per partition, capped at ``defaultParallelism * 8`` so
        # huge request lists don't explode into thousands of micro-tasks
        # whose scheduler overhead dominates the actual fetch.
        default_par = max(spark.sparkContext.defaultParallelism, 1)
        n_parts = max(1, min(len(req_rows), default_par * 8))
        request_df = spark.createDataFrame(req_rows, req_schema).repartition(n_parts)

        # Per-executor send config: remote cache disabled (driver-only),
        # local cache passthrough, no spark session, raise_error=False so
        # individual failures don't blow up the whole partition.
        send_config = config.to_send_config(
            with_remote_cache=False,
            with_local_cache=True,
            with_spark=False,
            raise_error=False,
        )

        # Broadcast the session so every executor receives the
        # (pickle-safe) session state once and reuses it across tasks,
        # rather than re-shipping a closure-captured copy per partition.
        # Session.__getstate__ / __setstate__ make this pickle-safe by
        # dropping the threading.RLock and JobPoolExecutor.
        session_bc = spark.sparkContext.broadcast(self)
        response_spark_schema = RESPONSE_SCHEMA.to_spark_schema()

        def _send_partition(
            batches: Iterator[pa.RecordBatch],
        ) -> Iterator[pa.RecordBatch]:
            import pickle as _pickle

            session = session_bc.value
            for batch in batches:
                partition_requests = [
                    _pickle.loads(buf).attach_session(session)
                    for buf in batch.column("request").to_pylist()
                ]
                if not partition_requests:
                    continue

                def _row_batches() -> Iterator[pa.RecordBatch]:
                    for resp in session.send_many(iter(partition_requests), send_config):
                        yield resp.to_arrow_batch(parse=False)

                yield from rechunk_arrow_batches_by_byte_size(
                    _row_batches(),
                    byte_size=_SPARK_RESPONSE_BATCH_BYTE_LIMIT,
                )

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
            full_url = parsed.with_query_items(params)

        return PreparedRequest.prepare(
            method=method,
            url=full_url,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            local_cache_config=local_cache_config,
            remote_cache_config=remote_cache_config
        )
