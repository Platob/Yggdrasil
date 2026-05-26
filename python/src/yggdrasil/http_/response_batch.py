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
from yggdrasil.io.request import PreparedRequest, REQUEST_SCHEMA
from yggdrasil.io.response import RESPONSE_SCHEMA, Response
from yggdrasil.io.send_config import SendConfig, CacheConfig, MATCH_KEY
from yggdrasil.io.tabular import ArrowTabular
from yggdrasil.arrow.tabular import ArrowTabular as Dataset
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


def _synthetic_not_found(request: "PreparedRequest") -> Response:
    return Response(
        request=request,
        status_code=404,
        headers={"Content-Type": "application/json"},
        tags={"synthetic": "cache_only_miss"},
        buffer=b'{"error": "not found in cache"}',
        received_at=dt.datetime.now(dt.timezone.utc),
    )


def responses_to_tabular(responses: list[Response]) -> ArrowTabular:
    return ArrowTabular(
        [Response.values_to_arrow_batch(responses)],
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
        return Dataset(data)
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
        "_split_done", "_misses",
    )

    def __init__(
        self,
        send_config: "SendConfig",
        requests: "list[PreparedRequest]",
        new_responses: "list[Response] | None" = None,
        new_responses_tabular: "Tabular | pa.Table | SparkDataFrame | None" = None,
        *,
        misses: "list[PreparedRequest] | None" = None,
        failed: "list[Response] | None" = None,
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

        if new_responses is not None:
            self._new_tabular: Optional[Tabular] = responses_to_tabular(new_responses)
        else:
            self._new_tabular = _to_tabular(new_responses_tabular)

        self._misses: list = misses if misses is not None else list(requests)
        self._failed: Optional[Tabular] = (
            responses_to_tabular(failed) if failed else None
        )

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
        kept: list[Response] = []
        for resp in Response.from_arrow_tabular(tab.read_arrow_batches()):
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
        """Resolve cache hits, fetch misses, write back."""
        cfg = self._send_config

        misses = list(self.misses)
        local_tab = self.local_tabular
        remote_tab = self.remote_tabular

        served = set()
        for tab in (local_tab, remote_tab):
            if tab is None:
                continue
            for resp in Response.from_arrow_tabular(tab.read_arrow_batches()):
                req = resp.request
                if req is not None:
                    served.add(req.match_value(MATCH_KEY))

        rejected = (self._local_hashes | self._remote_hashes) - served
        if rejected:
            misses = misses + [
                r for r in self._requests
                if r.match_value(MATCH_KEY) in rejected and r not in misses
            ]
            self._misses = misses

        if self._remote_hashes and cfg.local_cache is not None:
            if remote_tab is not None:
                try:
                    remote_table = remote_tab.read_arrow_table()
                    if remote_table.num_rows > 0:
                        cfg.local_cache.write_responses_tabular(
                            remote_table, mode=Mode.OVERWRITE, session=self._session,
                        )
                except Exception:
                    LOGGER.debug("Remote→local backfill failed", exc_info=True)

        if not misses or cfg.cache_only:
            if misses:
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
        misses: "list[PreparedRequest]",
        *,
        ordered: bool = False,
        max_in_flight: int | None = None,
    ) -> None:
        """Fetch misses via the session's thread pool."""
        cfg = self._send_config
        ok_list: list[Response] = []
        all_list: list[Response] = []
        err_list: list[Response] = []
        for response in self._session._fetch_misses(
            misses, ordered=ordered, max_in_flight=max_in_flight,
        ):
            all_list.append(response)
            if response.ok:
                ok_list.append(response)
            elif cfg.raise_error:
                err_list.append(response)
        if err_list:
            self.failed = err_list

        LOGGER.info(
            "Fetched %d/%d miss(es) (ok=%d, failed=%d)",
            len(ok_list) + len(err_list), len(misses),
            len(ok_list), len(err_list),
        )

        if all_list:
            self.new_tabular = pa.Table.from_batches(
                [Response.values_to_arrow_batch(all_list)]
            )
        if ok_list:
            write_data = pa.Table.from_batches(
                [Response.values_to_arrow_batch(ok_list)]
            )
            cfg.write_responses_tabular(write_data, session=self._session)

    def _fetch_spark(
        self,
        misses: "list[PreparedRequest]",
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
            if len(err_table) > 0:
                self.failed = err_table
            write_data = ok_df

        self.new_tabular = result_df
        cfg.write_responses_tabular(write_data, session=self._session)

    def _spark_scatter(
        self,
        misses: "list[PreparedRequest]",
        spark: "SparkSession",
    ) -> "Dataset":
        """Scatter misses to Spark workers via mapInArrow."""
        session = self._session
        cfg = self._send_config

        if not misses:
            schema = RESPONSE_SCHEMA.to_spark_schema()
            return spark.createDataFrame([], schema=schema)

        for req in misses:
            session.prepare_request_before_send(req)

        request_table = pa.Table.from_batches(
            [PreparedRequest.values_to_arrow_batch(misses)]
        )

        try:
            default_par = max(spark.sparkContext.defaultParallelism, 1)
        except Exception:
            default_par = 8
        n_parts = max(1, min(len(misses), default_par * 8))
        LOGGER.info(
            "Scattering %d miss(es) across %d Spark partition(s)",
            len(misses), n_parts,
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
                reqs = list(PreparedRequest.from_arrow(batch))
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
        return self._new_tabular

    @new_tabular.setter
    def new_tabular(self, value) -> None:
        self._new_tabular = _to_tabular(value)

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
        return any(isinstance(h, Dataset) for h in self._holders())

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

    def __iter__(self) -> Iterator[Response]:
        return self.responses()

    def responses(self) -> Iterator[Response]:
        for holder in self._holders():
            if holder is None:
                continue
            yield from Response.from_records(holder.read_records())

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def extend(self, other: "HTTPResponseBatch") -> "HTTPResponseBatch":
        self._local_tabular = _union(self.local_tabular, other.local_tabular)
        self._remote_tabular = _union(self.remote_tabular, other.remote_tabular)
        self._new_tabular = _union(self._new_tabular, other._new_tabular)
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
