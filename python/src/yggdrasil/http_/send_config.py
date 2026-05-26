from __future__ import annotations

import datetime as dt
import logging
import pathlib
import time
from typing import Any, ClassVar, Iterable, Literal, Mapping, MutableMapping, Optional, TYPE_CHECKING

from yggdrasil.data.cast import any_to_datetime, any_to_timedelta
from yggdrasil.data.cast.datetime import truncate_datetime
from yggdrasil.enums import Mode
from yggdrasil.dataclasses.waiting import WaitingConfig
from yggdrasil.environ import PyEnv
from yggdrasil.io.holder import Holder
from yggdrasil.io.path import Path
from yggdrasil.io.request import PreparedRequest

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from yggdrasil.io.response import Response
    from yggdrasil.io.session import Session
    from yggdrasil.io.tabular.base import Tabular


__all__ = ["CacheConfig", "SendConfig"]


# Module-level cached paths — avoids repeated syscalls in hot paths
# (``local_cache_folder`` is called per-request in the batch pipeline).
from yggdrasil.http_.cache_config import (
    CacheConfig,
    DEFAULT_CACHE_CONFIG,
    MATCH_COLUMN,
    MATCH_KEY,
    _CACHE_CONFIG_FIELDS,
    _DEFAULT_CACHE_ROOT,
)


_SEND_CONFIG_FIELDS: frozenset[str] = frozenset(
    list(_CACHE_CONFIG_FIELDS) + [
        "remote_cache", "local_cache", "max_batch_ttl",
        "batch_predicate", "cache_only",
    ]
)

DEFAULT_MAX_BATCH_TTL: float = 300.0


class SendConfig:
    _FIELD_NAMES: ClassVar[frozenset[str]] = _SEND_CONFIG_FIELDS

    __slots__ = (
        "raise_error", "wait", "remote_cache", "local_cache",
        "cache_only", "spark_session",
    )

    def __init__(
        self,
        raise_error: bool = True,
        wait: "WaitingConfig | None" = None,
        remote_cache: "CacheConfig | None" = None,
        local_cache: "CacheConfig | None" = None,
        cache_only: bool = False,
        spark_session: "bool | None" = None,
    ):
        self.raise_error = raise_error
        self.wait = WaitingConfig.from_(wait) if wait is not None else None
        self.remote_cache = CacheConfig.from_(remote_cache) if remote_cache is not None else None
        self.local_cache = CacheConfig.from_(local_cache) if local_cache is not None else None
        self.cache_only = cache_only
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
            try:
                table = holder.read_arrow_table(
                    predicate=predicate, columns=[MATCH_COLUMN],
                )
                if table is None or table.num_rows == 0:
                    return set()
                return set(table.column(MATCH_COLUMN).to_pylist())
            except Exception:
                return set()

        if self.local_cache is not None:
            local_hashes = _probe(self.local_cache) & request_hash_set
            if local_hashes:
                misses = [r for r in misses if r.match_value(MATCH_KEY) not in local_hashes]

        if self.remote_cache is not None and misses:
            remaining = {r.match_value(MATCH_KEY) for r in misses}
            remote_hashes = _probe(self.remote_cache) & remaining
            if remote_hashes:
                misses = [r for r in misses if r.match_value(MATCH_KEY) not in remote_hashes]

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
        batch_predicate = cache.make_batch_lookup_predicate(requests)
        opts = CastOptions(predicate=batch_predicate, target=RESPONSE_SCHEMA)
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
        )

    def __hash__(self):
        return hash((
            self.raise_error, self.wait,
            self.remote_cache, self.local_cache,
            self.cache_only,
        ))

    def __getstate__(self):
        return {
            "raise_error": self.raise_error,
            "wait": self.wait,
            "remote_cache": self.remote_cache,
            "local_cache": self.local_cache,
            "cache_only": self.cache_only,
            "spark_session": self.spark_session,
        }

    def __setstate__(self, state):
        self.raise_error = state.get("raise_error", True)
        self.wait = state.get("wait")
        self.remote_cache = state.get("remote_cache")
        self.local_cache = state.get("local_cache")
        self.cache_only = state.get("cache_only", False)
        self.spark_session = bool(state.get("spark_session", False))

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
        values = {k: v for k, v in options.items() if k in cls._FIELD_NAMES}
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
        unknown = set(overrides) - self._FIELD_NAMES
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


