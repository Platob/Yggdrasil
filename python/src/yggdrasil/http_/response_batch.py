"""Carrier for batched ``HTTPSession.send_many`` results.

:class:`HTTPResponseBatch` holds the send config and requests, lazily
resolving cache hits (local/remote) and network responses on access.
"""

from __future__ import annotations

import datetime as dt
import logging
import pickle
from typing import Any, TYPE_CHECKING, Iterator, Optional

import pyarrow as pa
from yggdrasil.arrow.cast import rechunk_arrow_batches
from yggdrasil.data import Mode
from yggdrasil.environ import PyEnv
from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse
from yggdrasil.http_.schemas import REQUEST_SCHEMA, RESPONSE_SCHEMA
from yggdrasil.http_.cache_config import CacheConfig, MATCH_KEY
from yggdrasil.http_.send_config import SendConfig
from yggdrasil.io.tabular import ArrowTabular
from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.io.tabular.base import Tabular

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
    from yggdrasil.spark.frame import Dataset as SparkDataset

LOGGER = logging.getLogger(__name__)

_SPARK_RESPONSE_BATCH_BYTE_LIMIT: int = 128 * 1024 * 1024


__all__ = [
    "HTTPResponseBatch",
    "responses_to_tabular",
]


def _synthetic_not_found(request: "HTTPRequest") -> HTTPResponse:
    return HTTPResponse(
        request=request,
        status_code=404,
        headers={"Content-Type": "application/json"},
        tags={"synthetic": "cache_only_miss"},
        buffer=b'{"error": "not found in cache"}',
        received_at=dt.datetime.now(dt.timezone.utc),
    )


def responses_to_tabular(responses: list[HTTPResponse]) -> ArrowTabular:
    return ArrowTabular(
        [HTTPResponse.values_to_arrow_batch(responses)],
        schema=RESPONSE_SCHEMA.to_arrow_schema(),
    )


def _to_tabular(data) -> "Tabular | None":
    if data is None:
        return None
    if isinstance(data, list):
        return responses_to_tabular(data) if data else None
    if isinstance(data, pa.Table):
        return ArrowTabular(data.to_batches(), schema=data.schema) if data.num_rows > 0 else None
    if isinstance(data, pa.RecordBatch):
        return ArrowTabular([data], schema=data.schema) if data.num_rows > 0 else None
    if isinstance(data, Tabular):
        return data
    if hasattr(data, "toArrow") or hasattr(data, "toPandas"):
        from yggdrasil.spark.tabular import SparkDataset
        return SparkDataset(data)
    return None


class HTTPResponseBatch(Tabular):
    """Lazy batch of responses with send config context.

    Holds the send config and original requests. Cache hits (local/remote)
    are resolved lazily via ``send_config.split_requests`` on first access.
    Network responses are set eagerly after fetch.
    """

    __slots__ = (
        "_send_config", "_requests", "_session",
        "_local_hashes", "_remote_hashes",
        "_local_tabular", "_remote_tabular", "_new_tabular", "_failed",
        "_split_done", "_misses", "_ignored_count",
        "_new_responses",
    )

    def __init__(
        self,
        send_config: "SendConfig",
        requests: "list[HTTPRequest]",
        new_responses: "list[HTTPResponse] | None" = None,
        new_responses_tabular: "Tabular | pa.Table | SparkDataFrame | None" = None,
        *,
        misses: "list[HTTPRequest] | None" = None,
        failed: "list[HTTPResponse] | None" = None,
        session: "Any" = None,
    ) -> None:
        super().__init__()
        self._send_config = send_config
        self._requests = requests
        self._local_hashes: set[int] = set()
        self._remote_hashes: set[int] = set()
        self._local_tabular: Optional[Tabular] = ...
        self._remote_tabular: Optional[Tabular] = ...
        self._split_done = False
        self._session = session

        self._new_responses: list[HTTPResponse] | None = new_responses
        if new_responses is not None:
            self._new_tabular: Optional[Tabular] = ...
        else:
            self._new_tabular = _to_tabular(new_responses_tabular)

        self._misses: list = misses if misses is not None else list(requests)
        self._failed: Optional[Tabular] = (
            responses_to_tabular(failed) if failed else None
        )
        self._ignored_count: int = 0

    def _ensure_split(self) -> None:
        if self._split_done:
            return
        self._split_done = True
        if self._send_config is None or not self._requests:
            return
        local_hashes, remote_hashes, remaining = self._send_config.split_requests(
            self._requests, session=self._session,
        )
        self._local_hashes = local_hashes
        self._remote_hashes = remote_hashes
        self._misses = remaining

    def _read_cache_hits(
        self, cache: "CacheConfig | None", hashes: set[int],
    ) -> "Tabular | None":
        """Read full responses for hit hashes from a cache, filtering stale."""
        if not hashes or cache is None:
            return None
        hit_reqs = [r for r in self._requests if r.match_value(MATCH_KEY) in hashes]
        if not hit_reqs:
            return None
        tab = self._send_config.read_hits(cache, hit_reqs, session=self._session)
        if tab is None:
            return None
        request_map = {r.match_value(MATCH_KEY): r for r in hit_reqs}
        kept: list[HTTPResponse] = []
        for resp in HTTPResponse.from_arrow_tabular(tab.read_arrow_batches()):
            req = request_map.get(
                resp.match_value(MATCH_KEY) if hasattr(resp, "match_value") else None
            )
            if req is not None and cache.filter_response(resp, request=req):
                kept.append(resp)
        return responses_to_tabular(kept) if kept else None

    def _fetch(
        self,
        *,
        ordered: bool = False,
        max_in_flight: int | None = None,
    ) -> "HTTPResponseBatch":
        """Resolve cache hits, fetch misses, write back.

        Only the **local** cache is materialised here, and only to reject stale
        hits: its probe (``HttpResponseCache.probe_hashes``) is presence-only, so
        a file can be reported as a hit yet fail the received-window filter on
        read — those keys must fall back to misses or they'd silently vanish from
        the result. That read is O(1) per key (and memoised in-memory after).

        The **remote** cache is deliberately *not* read here. Its probe is
        window-aware (``SendConfig.split_requests`` pushes ``received_from`` /
        ``received_to`` into the predicate), so every remote hit is already valid
        — there's nothing to reject. Touching ``remote_tabular`` would force a
        full (potentially Databricks) read eagerly; leaving it lazy means the
        remote rows are fetched only when the consumer iterates the batch.
        """
        cfg = self._send_config

        misses = list(self.misses)
        local_tab = self.local_tabular

        served = set()
        if local_tab is not None:
            for resp in HTTPResponse.from_arrow_tabular(local_tab.read_arrow_batches()):
                req = resp.request
                if req is not None:
                    served.add(req.match_value(MATCH_KEY))

        rejected = self._local_hashes - served
        if rejected:
            misses = misses + [
                r for r in self._requests
                if r.match_value(MATCH_KEY) in rejected and r not in misses
            ]
            self._misses = misses

        LOGGER.debug(
            "Cache split: %d local hit(s), %d remote hit(s), %d miss(es)",
            len(self._local_hashes), len(self._remote_hashes), len(misses),
        )

        if not misses or cfg.cache_only:
            if misses:
                LOGGER.warning(
                    "cache_only=True: %d request(s) not in cache, returning synthetic 404",
                    len(misses),
                )
                self.new_tabular = [_synthetic_not_found(r) for r in misses]
            return self

        spark = cfg.get_spark_session()
        if spark is not None:
            self._fetch_spark(misses, spark)
        else:
            self._fetch_local(misses, ordered=ordered, max_in_flight=max_in_flight)

        return self

    def _fetch_local(
        self,
        misses: "list[HTTPRequest]",
        *,
        ordered: bool = False,
        max_in_flight: int | None = None,
    ) -> None:
        """Fetch misses via the session's thread pool."""
        cfg = self._send_config
        ok_list: list[HTTPResponse] = []
        all_list: list[HTTPResponse] = []
        err_list: list[HTTPResponse] = []
        ignored: list[HTTPResponse] = []
        for response in self._session._fetch_misses(
            misses, ordered=ordered, max_in_flight=max_in_flight,
        ):
            all_list.append(response)
            if response.ok:
                ok_list.append(response)
            elif cfg.raise_error:
                err_list.append(response)
            else:
                ignored.append(response)
        if err_list:
            self.failed = err_list
        for r in err_list:
            LOGGER.warning(
                "%s %s failed %d (%d bytes)",
                r.request.method, r.request.url, r.status_code, r.body_size,
            )
        for r in ignored:
            LOGGER.warning(
                "%s %s ignored %d (%d bytes, raise_error=False)",
                r.request.method, r.request.url, r.status_code, r.body_size,
            )

        self._ignored_count = len(ignored)
        LOGGER.info(
            "Fetched %d miss(es): ok=%d, failed=%d, ignored=%d",
            len(misses), len(ok_list), len(err_list), len(ignored),
        )

        if all_list:
            self._new_responses = all_list
            self._new_tabular = ...
        if ok_list:
            write_data = pa.Table.from_batches(
                [HTTPResponse.values_to_arrow_batch(ok_list)]
            )
            if "received_at" in write_data.column_names:
                import datetime as _dt
                now_us = int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1_000_000)
                write_data = write_data.set_column(
                    write_data.column_names.index("received_at"),
                    "received_at",
                    pa.array([now_us] * write_data.num_rows, type=pa.int64()),
                )
            cfg.write_responses_tabular(write_data, session=self._session)

    def _fetch_spark(
        self,
        misses: "list[HTTPRequest]",
        spark: "SparkSession",
    ) -> None:
        """Fetch misses via Spark mapInArrow."""
        cfg = self._send_config
        result_df = self._spark_scatter(misses, spark)
        write_data = result_df

        if cfg.raise_error:
            from pyspark.sql import functions as F

            ok_df = result_df.where(
                (F.col("status_code") >= 200) & (F.col("status_code") < 400)
            )
            err_df = result_df.where(
                (F.col("status_code") < 200) | (F.col("status_code") >= 400)
            )
            if err_df.count() > 0:
                self._failed = err_df
            write_data = ok_df

        self.new_tabular = result_df
        cfg.write_responses_tabular(write_data, session=self._session)

    @staticmethod
    def _scatter_partition_count(
        n_misses: int,
        cluster_cores: int,
        n_executors: int,
        max_executors: int = 0,
        executor_cores: int = 0,
    ) -> int:
        """Number of partitions to scatter *n_misses* HTTP requests into.

        Each Spark task already multiplexes its slice of requests across the
        session's own thread pool (``_send_partition`` calls ``send_many``), so
        the per-socket I/O concurrency is handled *inside* a partition — we do
        NOT need extra partitions to keep the wire busy. We therefore target
        roughly one partition per task slot rather than oversubscribing.

        Autoscaling is the wrinkle: ``cluster_cores`` (``defaultParallelism``)
        only reflects the executors *currently registered*, which on an
        autoscaling cluster lags the configured maximum — and is 0 right after
        launch before any worker has joined. When the cluster advertises a
        ``max_executors`` and per-``executor_cores`` count, we size against
        that ceiling (``max_executors * executor_cores``) so a scatter that
        triggers a scale-up still lands enough partitions to use the eventual
        capacity instead of collapsing onto the few warm executors.

        Node topology then picks the factor: a single-node cluster (no live
        executors *and* no configured workers — local mode on the driver) gets
        exactly one partition per core; a multi-node cluster gets a light ×2 so
        the scheduler can rebalance stragglers across machines. A cluster with
        ``max_executors > 0`` counts as multi-node even before its workers
        register, so an autoscaling-from-zero scatter isn't mis-sized as
        single-node. The result is clamped to ``n_misses`` (never more
        partitions than requests) and floored at 1.
        """
        target_cores = cluster_cores
        if max_executors > 0 and executor_cores > 0:
            target_cores = max(target_cores, max_executors * executor_cores)
        single_node = n_executors == 0 and max_executors == 0
        oversubscribe = 1 if single_node else 2
        return max(1, min(n_misses, max(target_cores, 1) * oversubscribe))

    def _spark_scatter(
        self,
        misses: "list[HTTPRequest]",
        spark: "SparkSession",
    ) -> "Tabular":
        """Scatter misses to Spark workers via mapInArrow."""
        session = self._session
        cfg = self._send_config

        if not misses:
            schema = RESPONSE_SCHEMA.to_spark_schema()
            return spark.createDataFrame([], schema=schema)

        for req in misses:
            session.prepare_request_before_send(req)

        request_table = pa.Table.from_batches(
            [HTTPRequest.values_to_arrow_batch(misses)]
        )

        # Probe the cluster's real CPU + node topology to size the scatter.
        # ``getExecutorInfos()`` returns one entry per live executor PLUS the
        # driver, so the executor count is ``len(infos) - 1``; a single-node /
        # local cluster runs everything on the driver in local mode and reports
        # no separate executors. ``spark.sparkContext`` *raises*
        # JVM_ATTRIBUTE_NOT_SUPPORTED on Spark Connect / Databricks Connect (it
        # never just returns None), and the status tracker is unavailable there
        # too — so the whole probe is guarded and falls back to a sane
        # single-node default.
        try:
            sc = spark.sparkContext
            n_executors = max(len(sc.statusTracker().getExecutorInfos()) - 1, 0)
            cluster_cores = max(sc.defaultParallelism, 1)
        except Exception:
            n_executors = 0
            cluster_cores = 8

        # Autoscaling ceiling — read from Spark conf (available on Spark
        # Connect, unlike sparkContext): Databricks autoscaling tags first,
        # then generic dynamic-allocation, then per-executor core count. These
        # let ``_scatter_partition_count`` size against the cluster's *max*
        # capacity instead of the executors that happen to be warm right now.
        def _conf_int(*keys: str) -> int:
            for key in keys:
                try:
                    raw = spark.conf.get(key, None)
                except Exception:
                    raw = None
                if raw:
                    try:
                        return int(raw)
                    except (TypeError, ValueError):
                        continue
            return 0

        max_executors = _conf_int(
            "spark.databricks.clusterUsageTags.clusterMaxWorkers",
            "spark.dynamicAllocation.maxExecutors",
        )
        executor_cores = _conf_int("spark.executor.cores")
        n_parts = self._scatter_partition_count(
            len(misses), cluster_cores, n_executors, max_executors, executor_cores,
        )
        LOGGER.info(
            "Scattering %d miss(es) across %d Spark partition(s) "
            "(%s cluster, cores=%d, executors=%d, max_workers=%d, exec_cores=%d)",
            len(misses), n_parts,
            "single-node" if n_executors == 0 and max_executors == 0 else "multi-node",
            cluster_cores, n_executors, max_executors, executor_cores,
        )
        request_df = spark.createDataFrame(
            request_table,
            schema=REQUEST_SCHEMA.to_spark_schema()
        ).repartition(n_parts)

        worker_config = cfg.copy(
            remote_cache=None,
            spark_session=False,
            raise_error=False,
        )
        lc = worker_config.local_cache
        if lc is not None:
            now = dt.datetime.now(dt.timezone.utc)
            worker_config = worker_config.copy(
                raise_error=False,
                local_cache=lc.copy(
                    received_to=now,
                    received_from=now - dt.timedelta(minutes=15),
                ),
            )

        session_bytes = pickle.dumps(session)
        response_spark_schema = RESPONSE_SCHEMA.to_spark_schema()

        try:
            bc_session = spark.sparkContext.broadcast(session_bytes)
            bc_config = spark.sparkContext.broadcast(worker_config)
            use_broadcast = True
        except Exception:
            use_broadcast = False

        def _send_partition(
            batches: Iterator[pa.RecordBatch],
        ) -> Iterator[pa.RecordBatch]:
            if use_broadcast:
                sess = pickle.loads(bc_session.value)
                part_config = bc_config.value
            else:
                sess = pickle.loads(session_bytes)
                part_config = worker_config
            for batch in batches:
                reqs = list(HTTPRequest.from_arrow(batch))
                if not reqs:
                    continue

                def _row_batches() -> Iterator[pa.RecordBatch]:
                    for resp in sess.send_many(iter(reqs), part_config):
                        yield resp.to_arrow_batch(parse=False)

                yield from rechunk_arrow_batches(
                    _row_batches(),
                    byte_size=_SPARK_RESPONSE_BATCH_BYTE_LIMIT,
                )

        result_df = request_df.mapInArrow(
            _send_partition, schema=response_spark_schema,
        )

        return result_df

    @property
    def misses(self) -> list:
        self._ensure_split()
        return self._misses

    @misses.setter
    def misses(self, value: list) -> None:
        self._misses = value

    @property
    def local_tabular(self) -> "Tabular | None":
        self._ensure_split()
        if self._local_tabular is ...:
            self._local_tabular = self._read_cache_hits(
                self._send_config.local_cache if self._send_config else None,
                self._local_hashes,
            )
        return self._local_tabular

    @property
    def remote_tabular(self) -> "Tabular | None":
        self._ensure_split()
        if self._remote_tabular is ...:
            self._remote_tabular = self._read_cache_hits(
                self._send_config.remote_cache if self._send_config else None,
                self._remote_hashes,
            )
        return self._remote_tabular

    @property
    def new_tabular(self) -> "Tabular | None":
        if self._new_tabular is ...:
            if self._new_responses:
                self._new_tabular = responses_to_tabular(self._new_responses)
            else:
                self._new_tabular = None
        return self._new_tabular

    @new_tabular.setter
    def new_tabular(self, value) -> None:
        self._new_responses = None
        self._new_tabular = _to_tabular(value)

    @property
    def new_responses(self) -> "list[HTTPResponse] | None":
        return self._new_responses

    @property
    def failed(self) -> "Tabular | None":
        return self._failed

    @failed.setter
    def failed(self, value) -> None:
        if isinstance(value, list):
            self._failed = responses_to_tabular(value) if value else None
        else:
            self._failed = _to_tabular(value)

    @property
    def failed_count(self) -> int:
        return self._failed.count() if self._failed is not None else 0

    @property
    def ignored_count(self) -> int:
        return self._ignored_count

    @property
    def send_config(self) -> "SendConfig":
        return self._send_config

    def __repr__(self) -> str:
        return (
            f"HTTPResponseBatch(requests={len(self._requests)}, "
            f"split={self._split_done})"
        )

    # ------------------------------------------------------------------
    # Holders
    # ------------------------------------------------------------------

    def _holders(self) -> list[Tabular]:
        return [h for h in (self.local_tabular, self.remote_tabular, self.new_tabular) if h is not None]

    @property
    def is_spark(self) -> bool:
        try:
            from yggdrasil.spark.tabular import SparkDataset
            return any(isinstance(h, SparkDataset) for h in self._holders())
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Tabular implementation
    # ------------------------------------------------------------------

    def _collect_schema(self, options=None):
        return RESPONSE_SCHEMA

    def _read_arrow_batches(self, options=None):
        for holder in self._holders():
            yield from holder.read_arrow_batches(options=options)

    def _write_arrow_batches(self, batches, options=None):
        raise NotImplementedError("HTTPResponseBatch is read-only")

    def _delete(self, predicate=None, *, wait=True, missing_ok=False,
                delete_staging=True, **kwargs):
        raise NotImplementedError("HTTPResponseBatch is read-only")

    def _read_spark_frame(self, options):
        spark = PyEnv.spark_session(options.spark_session, create=True)
        result = None

        for holder in self._holders():
            df = holder.read_spark_frame(options=options)

            result = df if result is None else result.unionByName(
                df, allowMissingColumns=True,
            )

        if result is None:
            return spark.createDataFrame(
                [], schema=RESPONSE_SCHEMA.to_spark_schema(),
            )
        return result

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def _count(self, options) -> int:
        return sum(holder.count(options) for holder in self._holders())

    @property
    def counts(self, options = None) -> dict[str, int]:
        return {
            "local": self.local_tabular.count(options) if self.local_tabular else 0,
            "remote": self.remote_tabular.count(options) if self.remote_tabular else 0,
            "new": self.new_tabular.count(options) if self.new_tabular else 0,
        }

    def __len__(self) -> int:
        return sum(self.counts.values())

    def __bool__(self) -> bool:
        return bool(self._holders())

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[HTTPResponse]:
        return self.responses()

    def responses(self) -> Iterator[HTTPResponse]:
        for tab in (self.local_tabular, self.remote_tabular):
            if tab is not None:
                yield from HTTPResponse.from_records(tab.read_records())
        if self._new_responses is not None:
            yield from self._new_responses
        elif self._new_tabular is not None and self._new_tabular is not ...:
            yield from HTTPResponse.from_records(self._new_tabular.read_records())

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def extend(self, other: "HTTPResponseBatch") -> "HTTPResponseBatch":
        self._local_tabular = _union(self.local_tabular, other.local_tabular)
        self._remote_tabular = _union(self.remote_tabular, other.remote_tabular)
        if self._new_responses is not None and other._new_responses is not None:
            self._new_responses = self._new_responses + other._new_responses
            self._new_tabular = ...
        else:
            self._new_tabular = _union(self.new_tabular, other.new_tabular)
            self._new_responses = None
        self._misses.extend(other._misses)
        self._local_hashes |= other._local_hashes
        self._remote_hashes |= other._remote_hashes
        self._failed = _union(self._failed, other._failed)
        return self


def _union(a: Optional[Tabular], b: Optional[Tabular]) -> Optional[Tabular]:
    if b is None:
        return a
    if a is None:
        return b
    return a.union(b)
