from __future__ import annotations

import logging
from typing import Any, Mapping, TYPE_CHECKING

from yggdrasil.dataclasses.waiting import WaitingConfig
from yggdrasil.environ import PyEnv
from yggdrasil.http_.schemas import RESPONSE_SCHEMA
from yggdrasil.io.request import PreparedRequest

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from yggdrasil.io.response import Response
    from yggdrasil.io.tabular.base import Tabular


__all__ = ["CacheConfig", "SendConfig"]


# Module-level cached paths — avoids repeated syscalls in hot paths
# (``local_cache_folder`` is called per-request in the batch pipeline).
from yggdrasil.http_.cache_config import (
    CacheConfig,
    MATCH_COLUMN,
    MATCH_KEY,
)



DEFAULT_MAX_BATCH_TTL: float = 300.0

#: The only columns a cache read needs to rebuild a response — the response
#: payload plus the request join key (``request_public_hash``) and ``request_url``
#: (so the row's throwaway request still deserializes; the *real* request is
#: reattached from memory in ``HTTPResponseBatch._read_cache_hits``). ``received_at``
#: is deliberately *not* read — the received-window is already enforced upstream
#: (the remote probe is window-aware; the local cache filters on its own stored
#: value), so the retrieved response is stamped with the current UTC time instead.
#: The heavy ``request_headers`` / ``request_body`` / ``request_params`` /
#: ``receiver`` / ``*_hash`` / ``_pkl`` columns are never pulled back over the
#: wire — a remote (Databricks/Delta) hit read scans far fewer bytes.
RESPONSE_REBUILD_COLUMNS: "tuple[str, ...]" = (
    MATCH_COLUMN, "request_url", "status_code", "headers", "tags", "body",
)



class SendConfig:

    __slots__ = (
        "raise_error", "wait", "remote_cache", "local_cache",
        "cache_only", "spark_session", "stream",
    )

    def __init__(
        self,
        raise_error: bool = True,
        wait: "WaitingConfig | None" = None,
        remote_cache: "CacheConfig | None" = None,
        local_cache: "CacheConfig | None" = None,
        cache_only: bool = False,
        spark_session: "bool | None" = None,
        stream: bool = False,
    ):
        self.raise_error = raise_error
        self.wait = WaitingConfig.from_(wait) if wait is not None else None
        self.remote_cache = CacheConfig.from_(remote_cache) if remote_cache is not None else None
        self.local_cache = CacheConfig.from_(local_cache) if local_cache is not None else None
        self.cache_only = cache_only
        # When ``stream`` is set the wire send leaves the body un-preloaded
        # (``preload_content=False``): ``from_wire`` keeps the live socket
        # as the buffer's source instead of draining it eagerly, so a
        # consumer reading via ``.stream()`` / ``.iter_content`` pulls the
        # body incrementally and a multi-GB download never fully resides in
        # memory. The consumer owns draining + connection release (every
        # body accessor — ``.content`` / ``.data`` / ``.text`` / ``.json``
        # / ``.stream`` — does this). Incompatible with caching, which must
        # read the whole body to persist it; ``_can_fast_path`` already
        # routes cached requests away from the streaming wire path.
        self.stream = stream
        if spark_session is not None and spark_session is not False:
            self.spark_session = True
        else:
            self.spark_session = False

    def get_spark_session(self) -> "SparkSession | None":
        if not self.spark_session:
            return None
        try:
            return PyEnv.spark_session()
        except Exception:
            return None

    def split_requests(
        self,
        requests: list[PreparedRequest],
        *,
        session: "Any" = None,
    ) -> "tuple[set[int], set[int], list[PreparedRequest]]":
        """Split requests into local hit hashes, remote hit hashes, and misses.

        Performs lightweight hash-only lookups against local then remote
        cache. Returns ``(local_hashes, remote_hashes, misses)`` —
        full response rows are read on demand, not here.
        """
        from yggdrasil.execution.expr import col

        local_hashes: set[int] = set()
        remote_hashes: set[int] = set()
        misses = requests

        if not requests:
            return set(), set(), []

        request_hash_set = {r.match_value(MATCH_KEY) for r in requests}
        partition_keys = sorted({r.partition_key for r in requests})
        predicate = col("partition_key").is_in(partition_keys)

        def _probe(cache: CacheConfig) -> set[int]:
            holder = cache.tabular or cache.cache_tabular(session=session)
            if holder is None:
                return set()
            from yggdrasil.http_.response_cache import HttpResponseCache
            if isinstance(holder, HttpResponseCache):
                # Specialized local cache: O(1) per-key file presence check.
                return holder.probe_hashes(requests)
            # Constrain the presence probe to *this* cache's received window so a
            # row outside it is never counted as a hit. Otherwise — when the
            # caller asks for fresh data only (``received_from`` past the cached
            # row's ``received_at``) — the batch would read the full remote
            # response row only for ``filter_response`` to drop it and re-fetch
            # from the network: a wasted remote read of stale cache data. The
            # window pushes into the same predicate the backend already filters
            # on (SQL ``WHERE`` for a Table, Arrow predicate for a Folder), so
            # "only new data" skips the remote read entirely.
            probe_predicate = predicate
            window = []
            if cache.received_from is not None:
                window.append(col("received_at") >= cache.received_from)
            if cache.received_to is not None:
                window.append(col("received_at") < cache.received_to)
            if window:
                from yggdrasil.execution.expr import all_of
                probe_predicate = all_of(predicate, *window)
            try:
                table = holder.read_arrow_table(
                    predicate=probe_predicate, columns=[MATCH_COLUMN],
                )
                if table is None or table.num_rows == 0:
                    LOGGER.debug("Cache probe: 0 hits in %r", holder)
                    return set()
                hits = set(table.column(MATCH_COLUMN).to_pylist())
                LOGGER.debug("Cache probe: %d hit(s) in %r", len(hits), holder)
                return hits
            except Exception as e:
                if "not found" in str(e):
                    holder.create(RESPONSE_SCHEMA)
                    return _probe(cache)

                LOGGER.warning("Cache probe failed for %r", holder, exc_info=True)
                return set()

        if self.local_cache is not None:
            local_hashes = _probe(self.local_cache) & request_hash_set
            if local_hashes:
                matched = [r for r in misses if r.match_value(MATCH_KEY) in local_hashes]
                misses = [r for r in misses if r.match_value(MATCH_KEY) not in local_hashes]
                if matched and LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        "Local cache hit: %s",
                        ", ".join(f"{r.method} {r.url}" for r in matched),
                    )

        if self.remote_cache is not None and misses:
            remaining = {r.match_value(MATCH_KEY) for r in misses}
            remote_hashes = _probe(self.remote_cache) & remaining
            if remote_hashes:
                matched = [r for r in misses if r.match_value(MATCH_KEY) in remote_hashes]
                misses = [r for r in misses if r.match_value(MATCH_KEY) not in remote_hashes]
                if matched and LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        "Remote cache hit: %s",
                        ", ".join(f"{r.method} {r.url}" for r in matched),
                    )

        if misses and LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "Cache miss: %s",
                ", ".join(f"{r.method} {r.url}" for r in misses),
            )

        return local_hashes, remote_hashes, misses

    def read_hits(
        self,
        cache: "CacheConfig",
        requests: "list[PreparedRequest]",
        *,
        session: "Any" = None,
    ) -> "Tabular | None":
        """Read full response rows for hit requests from a cache."""
        from yggdrasil.data.options import CastOptions
        from yggdrasil.io.response import RESPONSE_SCHEMA

        holder = cache.tabular or cache.cache_tabular(session=session)
        if holder is None or not requests:
            return None
        from yggdrasil.http_.response_cache import HttpResponseCache
        if isinstance(holder, HttpResponseCache):
            # Specialized local cache: read the per-key files directly, wrap the
            # hits as an in-memory tabular (the generic predicate read doesn't
            # apply to a content-addressed blob store).
            from yggdrasil.http_.response_batch import responses_to_tabular
            hits, _misses = holder.read_responses(requests, config=cache)
            return responses_to_tabular(hits) if hits else None
        batch_predicate = cache.make_batch_lookup_predicate(requests)
        # Project to just the rebuild columns: the response payload + the request
        # join key. The full request is reattached from memory in
        # ``HTTPResponseBatch._read_cache_hits``, so a remote read never pulls the
        # heavy request_* / receiver / *_hash / _pkl columns it would discard.
        light = RESPONSE_SCHEMA.select(RESPONSE_REBUILD_COLUMNS)
        opts = CastOptions(predicate=batch_predicate, target=light)
        return holder.read_table(options=opts)

    def write_responses(
        self,
        responses: "list[Response]",
        *,
        session: "Any" = None,
    ) -> None:
        """Write responses to both local and remote caches asynchronously."""
        if not responses:
            return
        import pyarrow as pa
        from yggdrasil.io.response import Response

        table = pa.Table.from_batches(
            [Response.values_to_arrow_batch(responses)]
        )
        self.write_responses_tabular(table, session=session)

    def write_responses_tabular(
        self,
        data: "Any",
        *,
        session: "Any" = None,
    ) -> None:
        """Write Arrow/Spark response data to local and remote caches.

        Both writes run concurrently via threads when both caches
        are configured.
        """
        if data is None:
            return
        spark = self.get_spark_session()

        tasks: list = []
        if self.local_cache is not None:
            tasks.append(lambda: self.local_cache.write_responses_tabular(
                data, spark_session=spark, session=session,
            ))
        if self.remote_cache is not None:
            tasks.append(lambda: self.remote_cache.write_responses_tabular(
                data, spark_session=spark, session=session,
            ))

        if not tasks:
            return
        if len(tasks) == 1:
            tasks[0]()
            return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ygg-wb") as pool:
            futures = [pool.submit(t) for t in tasks]
            for f in futures:
                try:
                    f.result()
                except Exception:
                    LOGGER.warning("Cache writeback failed", exc_info=True)

    def __repr__(self):
        parts = []
        if not self.raise_error:
            parts.append("raise_error=False")
        if self.wait is not None:
            parts.append(f"wait={self.wait!r}")
        if self.remote_cache is not None:
            parts.append(f"remote_cache={self.remote_cache!r}")
        if self.local_cache is not None:
            parts.append(f"local_cache={self.local_cache!r}")
        if self.cache_only:
            parts.append("cache_only=True")
        if self.spark_session:
            parts.append("spark_session=True")
        if self.stream:
            parts.append("stream=True")
        return f"SendConfig({', '.join(parts)})"

    def __eq__(self, other):
        if not isinstance(other, SendConfig):
            return NotImplemented
        return (
            self.raise_error == other.raise_error
            and self.wait == other.wait
            and self.remote_cache == other.remote_cache
            and self.local_cache == other.local_cache
            and self.cache_only == other.cache_only
            and self.stream == other.stream
        )

    def __hash__(self):
        return hash((
            self.raise_error, self.wait,
            self.remote_cache, self.local_cache,
            self.cache_only, self.stream,
        ))

    def __getstate__(self):
        return {
            "raise_error": self.raise_error,
            "wait": self.wait,
            "remote_cache": self.remote_cache,
            "local_cache": self.local_cache,
            "cache_only": self.cache_only,
            "spark_session": self.spark_session,
            "stream": self.stream,
        }

    def __setstate__(self, state):
        self.raise_error = state.get("raise_error", True)
        self.wait = state.get("wait")
        self.remote_cache = state.get("remote_cache")
        self.local_cache = state.get("local_cache")
        self.cache_only = state.get("cache_only", False)
        self.spark_session = bool(state.get("spark_session", False))
        self.stream = bool(state.get("stream", False))

    def copy(self, **overrides):
        clean = {k: v for k, v in overrides.items() if v is not ...}
        if not clean:
            return self
        state = self.__getstate__()
        state.update(clean)
        return type(self)(**state)

    @classmethod
    def default(cls):
        inst = cls.__dict__.get("_DEFAULT_INSTANCE")
        if inst is None:
            inst = cls()
            type.__setattr__(cls, "_DEFAULT_INSTANCE", inst)
        return inst

    @classmethod
    def parse_mapping(cls, options: Mapping[str, Any], **overrides: Any):
        values = {k: v for k, v in options.items() if k in set(cls.__slots__)}
        values.update(overrides)
        if cls._matches_default(values):
            return cls.default()
        return cls(**{k: v for k, v in values.items() if v is not None})

    @classmethod
    def _matches_default(cls, values: Mapping[str, Any]) -> bool:
        if not values:
            return True
        default = cls.default()
        for k, v in values.items():
            if v is None:
                continue
            default_v = getattr(default, k, ...)
            if default_v is v:
                continue
            if type(v) in (bool, int, float, str, bytes) and default_v == v:
                continue
            return False
        return True

    def merge(self, **overrides: Any):
        unknown = set(overrides) - set(self.__slots__)
        if unknown:
            raise TypeError(
                f"{type(self).__name__}.merge got unexpected field(s): {sorted(unknown)!r}"
            )
        return self.copy(**overrides)

    @classmethod
    def from_(
        cls,
        arg: "SendConfig | Mapping[str, Any] | None",
        *,
        default: Any = ...,
        **overrides: Any,
    ) -> "SendConfig":
        try:
            if arg is None:
                if not overrides:
                    return default if default is not ... else cls.default()
                if cls._matches_default(overrides):
                    return cls.default()
                return cls.parse_mapping(overrides)
            if isinstance(arg, cls):
                return arg.merge(**overrides) if overrides else arg
            if isinstance(arg, Mapping):
                return cls.parse_mapping(arg, **overrides)
            raise TypeError(
                f"{cls.__name__}.from_ expects a {cls.__name__}, Mapping, or None; "
                f"got {type(arg).__name__!r}"
            )
        except (TypeError, ValueError):
            if default is ...:
                raise
            return default


